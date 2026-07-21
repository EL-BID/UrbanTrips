import logging
import multiprocessing
import os
import warnings
from functools import partial
from itertools import repeat
from math import sqrt

import h3
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from osmnx import distance
from shapely import LineString, Polygon

from urbantrips.carto.carto import (
    create_coarse_h3_from_line,
    create_route_section_ids,
    floor_rounding,
)
from urbantrips.geo import geo
from urbantrips.storage.context import StorageContext
from urbantrips.utils.utils import (
    duracion,
    leer_configs_generales,
)
from urbantrips.utils.paths import get_paths

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in line_locate_point_normalized",
    category=RuntimeWarning,
    module=r"shapely\.linear",
)

logger = logging.getLogger(__name__)


def process_routes_into_h3_parallel(routes_gdf, route_id_column, res=10):
    """
    Process routes into H3 cells in parallel using multiprocessing.

    Parameters
    ----------
    routes_gdf : geopandas.GeoDataFrame
        GeoDataFrame with route geometries and route_id_column
    route_id_column : str
        Name of the column containing route IDs
    res : int, optional
        H3 resolution, by default 10

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with all H3 cells for all routes
    """
    n_cores = max(int(multiprocessing.cpu_count() - 1), 1)
    n = len(routes_gdf)
    chunksize = max(1, int(sqrt(n)))

    logger.info("Procesando %d rutas en paralelo con %d cores", n, n_cores)

    # Convert rows to list of tuples for parallel processing
    rows_data = [(idx, row) for idx, row in routes_gdf.iterrows()]

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(
            partial(
                turn_route_geom_into_h3_cells_wrapper,
                route_id_column=route_id_column,
                res=res,
            ),
            rows_data,
            chunksize=chunksize,
        )

    # Concatenate all results
    routes_h3 = pd.concat(results, ignore_index=True)

    return routes_h3


def turn_route_geom_into_h3_cells_wrapper(row_data, route_id_column, res):
    """
    Wrapper for turn_route_geom_into_h3_cells to work with
    multiprocessing.Pool.

    Parameters
    ----------
    row_data : tuple
        Tuple of (index, row) from DataFrame.iterrows()
    route_id_column : str
        Name of the column containing route IDs
    res : int
        H3 resolution

    Returns
    -------
    pandas.DataFrame
        DataFrame with H3 cells for the route
    """
    idx, row = row_data
    try:
        result = turn_route_geom_into_h3_cells(
            row=row, route_id_column=route_id_column, res=res
        )
        return result
    except Exception as e:
        logger.error(
            "Error procesando ruta %s: %s", row.get(route_id_column, idx), str(e)
        )
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[route_id_column, "direction", "section_id", "h3", "wkt"]
        )


def process_parent_h3_parallel(
    routes_h3_df, routes_geoms_gdf, route_id_column, parent_res
):
    """
    Process parent H3 cells in parallel for multiple routes.

    Parameters
    ----------
    routes_h3_df : pandas.DataFrame
        DataFrame with H3 cells at child resolution
    routes_geoms_gdf : geopandas.GeoDataFrame
        GeoDataFrame with original route geometries
    route_id_column : str
        Name of the column containing route IDs
    parent_res : int
        Target parent H3 resolution

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with all parent H3 cells for all routes
    """
    n_cores = max(int(multiprocessing.cpu_count() - 1), 1)

    # Group routes_h3_df by route_id and direction
    grouped = routes_h3_df.groupby([route_id_column, "direction"])
    n = len(grouped)
    chunksize = max(1, int(sqrt(n)))

    logger.info(
        "Procesando %d rutas a resolución padre %d en paralelo con %d cores",
        n,
        parent_res,
        n_cores,
    )

    # Prepare data for parallel processing
    tasks = []
    for (route_id, direction), route_h3 in grouped:
        # Get the corresponding geometry
        mask = (routes_geoms_gdf[route_id_column] == route_id) & (
            routes_geoms_gdf["direction"] == direction
        )
        route_geom = routes_geoms_gdf[mask]

        if len(route_geom) > 0:
            tasks.append((route_h3, route_geom, route_id_column, parent_res))

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(
            turn_child_h3_into_parent_h3_wrapper, tasks, chunksize=chunksize
        )

    # Filter out empty results and concatenate
    valid_results = [r for r in results if len(r) > 0]
    if valid_results:
        parent_routes_h3 = pd.concat(valid_results, ignore_index=True)
        return parent_routes_h3
    else:
        return pd.DataFrame()


def turn_child_h3_into_parent_h3_wrapper(task_data):
    """
    Wrapper for turn_child_h3_into_parent_h3 to work with
    multiprocessing.Pool.

    Parameters
    ----------
    task_data : tuple
        (route_h3, route_geom, route_id_column, parent_res)

    Returns
    -------
    pandas.DataFrame
        DataFrame with parent H3 cells for the route
    """
    route_h3, route_geom, route_id_column, parent_res = task_data

    try:
        result = turn_child_h3_into_parent_h3(
            route_h3=route_h3, parent_res=parent_res, route_geom=route_geom
        )
        return result
    except Exception as e:
        route_id = (
            route_h3[route_id_column].iloc[0]
            if len(route_h3) > 0 and route_id_column in route_h3.columns
            else "unknown"
        )
        logger.error("Error procesando parent H3 para ruta %s: %s", route_id, str(e))
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                route_id_column,
                "direction",
                "section_id",
                "parent_h3",
                "resolution",
                "wkt",
            ]
        )


@duracion
def process_routes_geoms(ctx: StorageContext):
    """
    Checks for route geoms in config file, process line and route geoms,
    upload to db, and checks if stops table needs to be created from routes
    """

    # Deletes old data
    delete_old_route_geoms_data(ctx)

    configs = leer_configs_generales(autogenerado=False)
    h3_legs_res = configs["resolucion_h3"]

    # Default resolution for routes H3 processing
    routes_h3_res = 10
    # Check if we need parent H3 at config resolution
    need_parent_h3 = h3_legs_res < routes_h3_res

    if route_geoms_not_present(configs):
        logger.info(
            "No hay recorridos en el archivo de config — No se procesaran recorridos"
        )
        return None

    geojson_name = configs["recorridos_geojson"]
    geojson_path = str(get_paths().input_dir / geojson_name)
    geojson_data = gpd.read_file(geojson_path)

    branches_present = configs["lineas_contienen_ramales"]

    # If the geojson has direction-split rows, keep one row per ramal/line+direction (first occurrence)

    if "direction" not in geojson_data.columns:
        # raise a warning and create a direction column with 0
        logger.warning(
            "El archivo geojson no contiene una columna 'direction'. Se creará una columna 'direction' con valor 0 para todas las filas."
        )
        geojson_data["direction"] = 0

    id_col = "id_ramal" if branches_present else "id_linea"
    dedup_cols = [id_col, "direction"]

    if geojson_data.duplicated(subset=dedup_cols).any():
        geojson_data = geojson_data.drop_duplicates(subset=dedup_cols, keep="first")

    # Check columns
    check_route_geoms_columns(geojson_data, branches_present)
    geojson_data = check_directions_on_geoms(geojson_data, branches_present)

    # if data has lines and branches, split them
    if branches_present:
        branches_routes = geojson_data.reindex(
            columns=["id_ramal", "direction", "geometry"]
        )
        branches_routes["wkt"] = branches_routes.geometry.to_wkt()
        branches_routes = branches_routes.reindex(
            columns=["id_ramal", "direction", "wkt"]
        )
        # add branches routes to db
        ctx.insumos.save_raw(branches_routes, "official_branches_geoms")

        # compute h3 routes for branches
        logger.info("Calculando recorridos en H3 con resolución %s", routes_h3_res)

        # CREATE H3 CELLS ROUTE FOR EVERY LINE/BRANCH DIRECTION
        # Process routes in parallel at routes_h3_res (typically 10)
        branches_routes_h3 = process_routes_into_h3_parallel(
            routes_gdf=geojson_data, route_id_column="id_ramal", res=routes_h3_res
        )

        branches_routes_h3 = branches_routes_h3.reindex(
            columns=["id_ramal", "direction", "section_id", "h3", "wkt"]
        )

        ctx.insumos.save_raw(branches_routes_h3, "official_branches_geoms_h3")

        # Create parent H3 if config resolution is lower
        if need_parent_h3:
            logger.info("Creando H3 padre en resolución %s para ramales", h3_legs_res)
            branches_parent_h3 = process_parent_h3_parallel(
                routes_h3_df=branches_routes_h3,
                routes_geoms_gdf=geojson_data,
                route_id_column="id_ramal",
                parent_res=h3_legs_res,
            )

            if len(branches_parent_h3) > 0:
                # Rename parent_h3 to h3 for consistency
                branches_parent_h3 = branches_parent_h3.rename(
                    columns={"parent_h3": "h3"}
                )
                branches_parent_h3 = branches_parent_h3.reindex(
                    columns=[
                        "id_ramal",
                        "direction",
                        "section_id",
                        "h3",
                        "resolution",
                        "wkt",
                    ]
                )
                ctx.insumos.save_raw(
                    branches_parent_h3, "official_branches_geoms_h3_parent"
                )

        # produce a line from branches with lowess
        lines_routes = create_line_geom_from_branches(geojson_data)

    else:
        lines_routes = geojson_data.reindex(columns=["id_linea", "geometry"])

    lines_routes["wkt"] = lines_routes.geometry.to_wkt()

    lines_routes = lines_routes.reindex(columns=["id_linea", "direction", "wkt"])
    logger.info("Subiendo tabla de recorridos")

    ctx.insumos.save_raw(lines_routes, "official_lines_geoms")

    # Compute H3 routes for lines
    logger.info(
        "Calculando recorridos de lineas en H3 con resolución %s", routes_h3_res
    )
    lines_routes_gdf = gpd.GeoDataFrame(
        lines_routes, geometry=gpd.GeoSeries.from_wkt(lines_routes.wkt), crs="EPSG:4326"
    )

    lines_routes_h3 = process_routes_into_h3_parallel(
        routes_gdf=lines_routes_gdf, route_id_column="id_linea", res=routes_h3_res
    )

    lines_routes_h3 = lines_routes_h3.reindex(
        columns=["id_linea", "direction", "section_id", "h3", "wkt"]
    )

    ctx.insumos.save_raw(lines_routes_h3, "official_lines_geoms_h3")

    # Create parent H3 if config resolution is lower
    if need_parent_h3:
        logger.info("Creando H3 padre en resolución %s para lineas", h3_legs_res)
        lines_parent_h3 = process_parent_h3_parallel(
            routes_h3_df=lines_routes_h3,
            routes_geoms_gdf=lines_routes_gdf,
            route_id_column="id_linea",
            parent_res=h3_legs_res,
        )

        if len(lines_parent_h3) > 0:
            # Rename parent_h3 to h3 for consistency
            lines_parent_h3 = lines_parent_h3.rename(columns={"parent_h3": "h3"})
            lines_parent_h3 = lines_parent_h3.reindex(
                columns=[
                    "id_linea",
                    "direction",
                    "section_id",
                    "h3",
                    "resolution",
                    "wkt",
                ]
            )
            ctx.insumos.save_raw(lines_parent_h3, "official_lines_geoms_h3_parent")


def check_directions_on_geoms(geojson_data, branches_present):
    """
    This function check that for every id_linea (and id_ramal if
    branches_present) there are two rows with direction 0 and 1
    if there are not, it will create the missing direction by
    inverting the geometry of the other direction
    """
    geojson_data = geojson_data.copy()
    id_col = "id_ramal" if branches_present else "id_linea"

    # Get all unique route IDs
    unique_ids = geojson_data[id_col].unique()

    missing_rows = []

    for route_id in unique_ids:
        route_data = geojson_data[geojson_data[id_col] == route_id]
        existing_directions = set(route_data["direction"].unique())

        # Check if both directions exist
        if 0 not in existing_directions and 1 in existing_directions:
            # Create direction 0 by inverting direction 1
            dir1_row = route_data[route_data["direction"] == 1].iloc[0].copy()
            dir1_row["direction"] = 0
            dir1_row["geometry"] = LineString(list(dir1_row["geometry"].coords)[::-1])
            missing_rows.append(dir1_row)
        elif 1 not in existing_directions and 0 in existing_directions:
            # Create direction 1 by inverting direction 0
            dir0_row = route_data[route_data["direction"] == 0].iloc[0].copy()
            dir0_row["direction"] = 1
            dir0_row["geometry"] = LineString(list(dir0_row["geometry"].coords)[::-1])
            missing_rows.append(dir0_row)

    # Append missing rows if any
    if missing_rows:
        missing_df = gpd.GeoDataFrame(
            missing_rows, geometry="geometry", crs=geojson_data.crs
        )
        geojson_data = pd.concat([geojson_data, missing_df], ignore_index=True)

    return geojson_data


@duracion
def infer_routes_geoms(ctx: StorageContext):
    """
    Esta funcion crea a partir de las etapas un recorrido simplificado
    de las lineas y lo guarda en la db
    """

    q = """
    select e.id_linea,e.longitud,e.latitud
    from etapas e
    """
    etapas = ctx.data.query(q)

    recorridos_lowess = etapas.groupby("id_linea").apply(geo.lowess_linea).reset_index()

    # Elminar geometrias invalidas
    validas = recorridos_lowess.geometry.map(lambda g: g.is_valid)
    recorridos_lowess = recorridos_lowess.loc[validas, :]

    recorridos_lowess_direction0 = recorridos_lowess.copy()
    recorridos_lowess_direction0["direction"] = 0
    recorridos_lowess_direction1 = recorridos_lowess.copy()
    recorridos_lowess_direction1["direction"] = 1
    # invert the geometry for direction 1
    recorridos_lowess_direction1["geometry"] = (
        recorridos_lowess_direction1.geometry.map(
            lambda g: LineString(list(g.coords)[::-1])
        )
    )
    recorridos_lowess = pd.concat(
        [recorridos_lowess_direction0, recorridos_lowess_direction1], ignore_index=True
    )

    recorridos_lowess["wkt"] = recorridos_lowess.geometry.to_wkt()

    recorridos_lowess = recorridos_lowess.reindex(
        columns=["id_linea", "direction", "wkt"]
    )

    ctx.insumos.save_raw(recorridos_lowess, "inferred_lines_geoms")


@duracion
def build_routes_from_official_inferred(ctx: StorageContext):

    for table in ("lines_geoms", "branches_geoms"):
        try:
            ctx.insumos.execute(f"DELETE FROM {table}")
        except Exception:
            pass

    ctx.insumos.execute("""
        INSERT INTO lines_geoms
            select i.id_linea, i.direction, coalesce(o.wkt, i.wkt) as wkt
            from inferred_lines_geoms i
            left join official_lines_geoms o
            on i.id_linea = o.id_linea
            and i.direction = o.direction  
        """)

    ctx.insumos.execute("""
        INSERT INTO branches_geoms
        select * from official_branches_geoms
        """)


def create_line_geom_from_branches(geojson_data):
    """
    Takes a geoDataFrame with lines and branches, and creates a single
    linestring for each line using lowess regression over interpolated
    points on all branches

    Parameters
    ----------
    geojson_data : geopandas.geoDataFrame
        geoDataFrame containing the LineStrings for each branch with
        an id_linea atrribute identifying to which line it belongs

    Returns
    -------
    geopandas.geoDataFrame
        DataFrame containing a single LineString for each id_linea
    """
    epsg_m = geo.get_epsg_m()
    geojson_data = geojson_data.to_crs(epsg=epsg_m)

    lines_routes = geojson_data.groupby(
        ["id_linea", "direction"], as_index=False
    ).apply(get_line_lowess_from_branch_routes)
    lines_routes.columns = ["id_linea", "direction", "geometry"]
    lines_routes = gpd.GeoDataFrame(lines_routes, geometry="geometry", crs=epsg_m)

    lines_routes = lines_routes.to_crs(epsg=4326)

    return lines_routes


def get_line_lowess_from_branch_routes(gdf):
    if len(gdf) > 1:
        import statsmodels.api as sm

        line_routes = gdf.geometry
        # create points every 100 meters over the route
        points = list(map(geo.get_points_over_route, line_routes, repeat(100)))
        points = list(np.concatenate(points).flat)
        x = list(map(lambda point: point.x, points))
        y = list(map(lambda point: point.y, points))

        # run lowess regression
        lowess = sm.nonparametric.lowess
        lowess_points = lowess(y, x, frac=0.40, delta=5)

        # build linestring
        lowess_line = LineString(lowess_points)
    else:
        lowess_line = gdf.geometry.iloc[0]

    return lowess_line


def check_route_geoms_columns(geojson_data, branches_present):
    # Check all columns are present
    cols = ["id_linea", "geometry"]

    assert (
        not geojson_data.id_linea.isna().any()
    ), "id_linea vacios en geojson recorridos"

    if branches_present:
        cols.append("id_ramal")
        assert (
            not geojson_data.id_ramal.isna().any()
        ), "id_ramal vacios en geojson recorridos"

    cols = pd.Series(cols)
    columns_ok = cols.isin(geojson_data.columns)

    if not columns_ok.all():
        cols_not_ok = ",".join(cols[~columns_ok].values)

        raise ValueError(f"Faltan columnas en el dataset: {cols_not_ok}")

    assert (
        not geojson_data.direction.isna().any()
    ), "direction vacios en geojson recorridos"

    # check all values for direction are 0 or 1
    assert geojson_data.direction.isin(
        [0, 1]
    ).all(), "direction debe ser 0 o 1 en geojson recorridos"

    # Check geometry type
    geo.check_all_geoms_linestring(geojson_data)


def delete_old_route_geoms_data(ctx: StorageContext):
    for table in (
        "lines_geoms",
        "branches_geoms",
        "official_lines_geoms",
        "official_branches_geoms",
    ):
        try:
            ctx.insumos.execute(f"DELETE FROM {table}")
        except Exception:
            pass


def route_geoms_not_present(configs):
    # check if config has the parameter
    param_present = "recorridos_geojson" in configs
    if param_present:
        # check if full
        param_full = configs["recorridos_geojson"] is not None

        if param_full:
            return False
        else:
            return True
    else:
        return True


@duracion
def process_routes_metadata(ctx: StorageContext):
    """
    This function reads from config file the location of the csv table
    with routes metadata, check if lines and branches are present
    and uploads metadata to the db
    """

    configs = leer_configs_generales(autogenerado=False)

    try:
        tabla_lineas = configs["nombre_archivo_informacion_lineas"]
        branches_present = configs["lineas_contienen_ramales"]
    except KeyError:
        tabla_lineas = None
        branches_present = False
        logger.warning("No hay tabla con informacion configs")

    # Check modes matches config standarized modes
    try:
        modos_homologados = configs["modos"]
        zipped = zip(modos_homologados.values(), modos_homologados.keys())
        modos_homologados = {k: v for k, v in zipped}

    except KeyError:
        pass

    # Line metadata is mandatory
    logger.info("Leyendo tabla con informacion de lineas")
    from urbantrips.utils.io import open_csv, resolve_zip

    ruta = resolve_zip(str(get_paths().input_dir / tabla_lineas))
    with open_csv(ruta) as f:
        info = pd.read_csv(f)

    # Check all columns are present
    if branches_present:
        cols = ["id_linea", "nombre_linea", "id_ramal", "nombre_ramal", "modo"]
    else:
        cols = ["id_linea", "nombre_linea", "modo"]

    assert (
        pd.Series(cols).isin(info.columns).all()
    ), f"La tabla {ruta} debe tener los campos: {cols}"

    # check no missing data in line id
    assert not info.id_linea.isna().any(), "id_linea no debe ser NULL"
    # fill nombre_linea from id_linea when absent
    info["nombre_linea"] = info["nombre_linea"].fillna(info["id_linea"].astype(str))
    # fill nombre_ramal from id_ramal when absent
    if branches_present:
        info["nombre_ramal"] = info["nombre_ramal"].fillna(info["id_ramal"].astype(str))

    if "id_linea_agg" not in info.columns:
        info["id_linea_agg"] = info["id_linea"]
        info["nombre_linea_agg"] = info["nombre_linea"]

    line_cols = [
        "id_linea",
        "nombre_linea",
        "id_linea_agg",
        "nombre_linea_agg",
        "modo",
        "empresa",
        "descripcion",
    ]

    assert pd.Series(info.modo.unique()).isin(modos_homologados.keys()).all()

    info["modo"] = info["modo"].replace(modos_homologados)

    # fuerza la columna a object para que acepte strings
    info["nombre_linea_agg"] = info["nombre_linea_agg"].astype("object")
    # fill missing line agg
    info.loc[info.id_linea_agg.isna(), "nombre_linea_agg"] = info.loc[
        info.id_linea_agg.isna(), "nombre_linea"
    ]
    info.loc[info.id_linea_agg.isna(), "id_linea_agg"] = info.loc[
        info.id_linea_agg.isna(), "id_linea"
    ]

    # keep only line data
    info_lineas = info.reindex(columns=line_cols)
    info_lineas = info_lineas.drop_duplicates(subset="id_linea")

    ctx.insumos.save_metadata_lineas(info_lineas)

    if branches_present:
        ramales_cols = [
            "id_ramal",
            "id_linea",
            "nombre_ramal",
            "modo",
            "empresa",
            "descripcion",
        ]

        info_ramales = info.reindex(columns=ramales_cols)

        # Checks for missing and duplicated
        s = "Existen nulos en el campo id_ramal"
        assert not info_ramales.id_ramal.isna().any(), s

        assert (
            not info_ramales.id_ramal.duplicated().any()
        ), "Existen duplicados en id_ramal"

        ctx.insumos.save_metadata_ramales(info_ramales)


def create_line_g(line_id, ctx: StorageContext):
    """
    Takes linea id, read from the stops data
    and produces a line graph composing from branch's graphs
    """
    stops = ctx.insumos.get_stops()
    line_stops = stops[stops.id_linea == line_id]

    branches_id = line_stops.id_ramal.unique()

    G_line = nx.compose_all(
        [
            create_branch_g_from_stops_df(line_stops, branch_id)
            for branch_id in branches_id
        ]
    )

    return G_line


def create_branch_g_from_stops_df(line_stops, id_ramal):
    """
    Takes a line stops with node_id and coordinates (node_x, node_y)
    and a branch_id, selects branch's stops and produces a graph
    """

    branch_stops = line_stops.loc[line_stops.id_ramal == id_ramal, :]

    # remove duplicated stops with same node_id
    branch_stops = branch_stops.drop_duplicates(subset="node_id")

    G = create_branch_graph(branch_stops)
    return G


def create_branch_graph(branch_stops):
    """
    Takes a line's branch stops with a node_id
    and coordinates (node_x, node_y) and produces
    a branch graph
    """
    metadata = {
        "crs": "epsg:4326",
        "id_linea": branch_stops["id_linea"].unique().item(),
        "id_ramal": branch_stops["id_ramal"].unique().item(),
    }
    G = nx.MultiGraph(**metadata)

    branch_stops = branch_stops.sort_values("branch_stop_order").reindex(
        columns=["node_id", "node_x", "node_y"]
    )
    nodes = [
        (int(row["node_id"]), {"x": row["node_x"], "y": row["node_y"]})
        for _, row in branch_stops.iterrows()
    ]
    G.add_nodes_from(nodes)

    edges_from = branch_stops["node_id"].iloc[:-1].map(int)
    edges_to = branch_stops["node_id"].shift(-1).iloc[:-1].map(int)
    edges = [(i, j, 0) for i, j in zip(edges_from, edges_to)]
    G.add_edges_from(edges)

    # add distance in meters
    G = distance.add_edge_lengths(G)

    return G


def read_branch_routes(branch_ids, ctx: StorageContext):
    """
    This function take a list of branch ids and returns a geodataframe
    with route geoms
    """
    route_geoms = ctx.insumos.get_raw("branches_geoms")
    route_geoms = route_geoms.loc[route_geoms.direction == 0, :]
    if branch_ids is not None and branch_ids is not False:
        ids = [branch_ids] if isinstance(branch_ids, (int, str)) else list(branch_ids)
        route_geoms = route_geoms[route_geoms.id_ramal.isin(ids)]
    route_geoms["geometry"] = gpd.GeoSeries.from_wkt(route_geoms.wkt)
    route_geoms = gpd.GeoDataFrame(
        route_geoms.drop("wkt", axis=1), geometry="geometry", crs="EPSG:4326"
    )
    return route_geoms


def read_routes(route_ids, route_type, ctx: StorageContext):
    """
    This function take a list of branches or lines ids and returns a geodataframe
    with route geoms

    Parameters
    ----------
    route_ids : list
        list of branches or lines ids
    route_type : str
        branches or lines
    ctx : StorageContext

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with route geoms
    """
    table = f"{route_type}_geoms"
    id_col = "id_ramal" if route_type == "branches" else "id_linea"
    route_geoms = ctx.insumos.get_raw(table)
    route_geoms = route_geoms.loc[route_geoms.direction == 0, :].drop(
        columns="direction"
    )

    if route_ids is not None and route_ids is not False:
        ids = [route_ids] if isinstance(route_ids, (int, str)) else list(route_ids)
        route_geoms = route_geoms[route_geoms[id_col].isin(ids)]

    route_geoms = route_geoms.rename(columns={id_col: "route_id"})
    route_geoms["geometry"] = gpd.GeoSeries.from_wkt(route_geoms.wkt)
    route_geoms = gpd.GeoDataFrame(
        route_geoms.drop("wkt", axis=1), geometry="geometry", crs="EPSG:4326"
    )
    return route_geoms


def get_route_geoms_with_sections_data(
    line_ids, section_meters, n_sections, ctx: StorageContext
):

    route_geoms = ctx.insumos.get_routes()
    # only use direction 0
    route_geoms = route_geoms.loc[route_geoms.direction == 0,]

    if line_ids and line_ids is not False:
        ids = [line_ids] if isinstance(line_ids, int) else list(line_ids)
        route_geoms = route_geoms[route_geoms.id_linea.isin(ids)].copy()

    # Set which parameter to use to split route geoms into sections
    epsg_m = geo.get_epsg_m()

    # project geoms and get for each geom both n_sections and meter
    route_geoms = route_geoms.to_crs(epsg=epsg_m)

    if section_meters:
        # warning if meters params give to many sections
        # get how many sections given the meters
        n_sections = (route_geoms.geometry.length / section_meters).astype(int)

    else:
        section_meters = (route_geoms.geometry.length / n_sections).astype(int)

    if isinstance(n_sections, int):
        n_sections_check = pd.Series([n_sections])
    else:
        n_sections_check = n_sections

    if any(n_sections_check > 1000):
        warnings.warn(
            "Algunos recorridos tienen mas de 1000 segmentos"
            "Puede arrojar resultados imprecisos "
        )

    route_geoms = route_geoms.to_crs(epsg=4326)

    # set the section length in meters
    route_geoms["section_meters"] = section_meters

    # set the number of sections
    route_geoms["n_sections"] = n_sections

    return route_geoms


def check_exists_route_section_points_table(route_geoms, ctx: StorageContext):
    """
    This function checks if the route section points table exists
    for those lines and n_sections in the route geoms gdf
    """
    route_sections = ctx.insumos.get_raw("routes_section_id_coords")

    if not route_sections.empty:
        route_sections = (
            route_sections[["id_linea", "n_sections"]].drop_duplicates().copy()
        )
        route_sections["section_exists"] = 1
    else:
        route_sections = pd.DataFrame(
            columns=["id_linea", "n_sections", "section_exists"]
        )

    new_route_geoms = route_geoms.merge(
        route_sections, on=["id_linea", "n_sections"], how="left"
    )
    new_route_geoms = new_route_geoms.loc[
        new_route_geoms.section_exists.isna(), ["id_linea", "n_sections", "geometry"]
    ]

    return new_route_geoms


def upload_route_section_points_table(
    route_geoms, ctx: StorageContext, delete_old_data=False
):
    """
    Uploads a table with route section points from a route geom row
    and returns a table with line_id, number of sections and the
    xy point for that section
    """
    # delete old records
    if delete_old_data:
        delete_old_routes_section_id_coords_data_q(route_geoms, ctx)

    logger.info("Creando tabla de secciones de recorrido")
    route_section_points = pd.concat(
        [create_route_section_points(row) for _, row in route_geoms.iterrows()]
    )

    ctx.insumos.append_raw(route_section_points, "routes_section_id_coords")
    logger.debug("Fin creacion de tabla de secciones de recorrido")


def delete_old_routes_section_id_coords_data_q(route_geoms, ctx: StorageContext):
    """
    Deletes old data in table routes_section_id_coords
    """
    delete_df = route_geoms.reindex(columns=["id_linea", "n_sections"])
    for _, row in delete_df.iterrows():
        q_delete = f"""
            DELETE FROM routes_section_id_coords
            WHERE id_linea = {row.id_linea}
            AND n_sections = {row.n_sections}
            """
        ctx.insumos.execute(q_delete)
    logger.debug("Fin borrado datos previos")


def create_route_section_points(row):
    """
    Creates a table with route section points from a route geom row
    and returns a table with line_id, number of sections and the
    xy point for that section
    """

    n_sections = row.n_sections
    route_geom = row.geometry
    line_id = row.id_linea
    sections_lrs = create_route_section_ids(n_sections)
    sections_id = list(range(1, len(sections_lrs))) + [-1]
    points = route_geom.interpolate(sections_lrs, normalized=True)
    route_section_points = pd.DataFrame(
        {
            "id_linea": [line_id] * len(sections_id),
            "n_sections": [n_sections] * len(sections_id),
            "section_id": sections_id,
            "section_lrs": sections_lrs,
            "x": points.map(lambda p: p.x),
            "y": points.map(lambda p: p.y),
        }
    )
    return route_section_points


def get_route_section_id(point, route_geom):
    """
    Computes the route section id as a 3 digit float projecing
    a point on to the route geom in a normalized way
    """
    return floor_rounding(route_geom.project(point, normalized=True))


def build_leg_route_sections_df(row):
    """
    Computes for a leg a table with all sections id traversed by
    that leg based on the origin and destionation's section id
    """

    sentido = row["sentido"]
    dia = row["dia"]
    f_exp = row["factor_expansion_linea"]

    # always build it in increasing order
    if sentido == "ida":
        o_id = row["o_proj"]
        d_id = row["d_proj"]
    else:
        o_id = row["d_proj"]
        d_id = row["o_proj"]

    leg_route_sections = list(range(o_id, d_id + 1))
    leg_route_sections_df = pd.DataFrame(
        {
            "dia": [dia] * len(leg_route_sections),
            "sentido": [sentido] * len(leg_route_sections),
            "section_id": leg_route_sections,
            "factor_expansion_linea": [f_exp] * len(leg_route_sections),
        }
    )
    return leg_route_sections_df


def build_gps_route_sections_df(row):
    """
    Computes for a gps a table with all sections id traversed by
    that gps based on the gps point section id and the next
    """

    sentido = row["sentido"]
    dia = row["dia"]
    ramal = row["id_ramal"]
    interno = row["interno"]

    # always build it in increasing order
    if sentido == "ida":
        o_id = row["section_id"]
        d_id = row["section_id_next"]
    else:
        o_id = row["section_id_next"]
        d_id = row["section_id"]

    gps_route_sections = list(range(o_id, d_id + 1))
    gps_route_sections_df = pd.DataFrame(
        {
            "id_ramal": [ramal] * len(gps_route_sections),
            "interno": [interno] * len(gps_route_sections),
            "dia": [dia] * len(gps_route_sections),
            "sentido": [sentido] * len(gps_route_sections),
            "section_id": gps_route_sections,
        }
    )
    return gps_route_sections_df


def turn_route_geom_into_h3_cells(
    row,
    route_id_column,
    res=10,
):
    """
    Convierte la geometría de una ruta en una secuencia de celdas H3,
    interpolando puntos a lo largo de la ruta y asignándoles celdas H3.
    """
    route_geom = row.geometry
    direction = row.direction
    route_id = row[route_id_column]

    epsg_m = geo.get_epsg_m()
    route_geom_m = (
        gpd.GeoSeries(route_geom, crs=4326).to_crs(epsg=epsg_m).geometry.iloc[0]
    )

    interpolating_distance = h3.average_hexagon_edge_length(10, unit="m") * 0.5

    # Interpolate points along the route geometry at regular intervals
    points = interpolate_points(
        route_geom_m=route_geom_m, interpolating_distance=interpolating_distance
    )

    # create a GeoDataFrame from the interpolated points indexed in h3 res 10
    points["h3_id"] = points.geometry.apply(lambda p: h3.latlng_to_cell(p.y, p.x, res))
    points["block"] = (points["h3_id"] != points["h3_id"].shift()).cumsum()

    geom_h3 = points.drop_duplicates(subset=["h3_id", "block"]).reset_index(drop=True)
    geom_h3 = gpd.GeoDataFrame(
        geom_h3, geometry=geom_h3["h3_id"].map(h3_to_polygon), crs="EPSG:4326"
    )
    geom_h3 = geom_h3.sort_values("lrs").reset_index(drop=True)
    geom_h3["section_id"] = range(len(geom_h3))

    """
    cell_shift = points.loc[
        points["h3_id"] != points["h3_id"].shift(), "h3_id"
    ].value_counts()
    cells_with_shift = cell_shift[cell_shift > 1].index.tolist()

    if len(cells_with_shift) > 0:
        print(
            f"⚠️  Warning: Detected {len(cells_with_shift)} cells with multiple visits:"
        )
        for cell in cells_with_shift[:10]:  # Show up to 10 problematic cells
            print(f"  Cell {cell} visited {cell_shift[cell]} times")
        if len(cells_with_shift) > 10:
            print(f"  ... and {len(cells_with_shift) - 10} more")
    """

    # First, check how many gaps exist
    gaps = []
    for i in range(len(geom_h3) - 1):
        current_cell = geom_h3.iloc[i]["h3_id"]
        next_cell = geom_h3.iloc[i + 1]["h3_id"]
        if not h3.are_neighbor_cells(current_cell, next_cell):
            distance = h3.grid_distance(current_cell, next_cell)
            gaps.append({"from_idx": i, "to_idx": i + 1, "distance": distance})

    # print(f"Found {len(gaps)} gaps in the route:")
    if len(gaps) > 0:
        print(f"Found {len(gaps)} gaps ")

    only_one_cell_gaps = [g["distance"] <= 2 for g in gaps]
    if not all(only_one_cell_gaps):
        print(
            f"⚠️  Warning: {sum(not d for d in only_one_cell_gaps)} gaps have distance greater than 2, which may indicate significant route discontinuities."
        )

    if len(gaps) > 0:
        geom_h3_filled = fill_h3_gaps(
            geom_h3=geom_h3, line_geom=route_geom, h3_column="h3_id", verbose=False
        )
    else:
        geom_h3_filled = geom_h3.copy()

    # Validate that the filled route is fully connected
    non_adjacent_count = 0
    for i in range(len(geom_h3_filled) - 1):
        current_cell = geom_h3_filled.iloc[i]["h3_id"]
        next_cell = geom_h3_filled.iloc[i + 1]["h3_id"]
        if not h3.are_neighbor_cells(current_cell, next_cell):
            non_adjacent_count += 1
            print(f"❌ Cells at positions {i} and {i+1} are still NOT adjacent")

    if non_adjacent_count != 0:
        print(f"\n⚠️  Warning: {non_adjacent_count} gaps remain")

    geom_h3_filled[route_id_column] = route_id
    geom_h3_filled["direction"] = direction
    geom_h3_filled = geom_h3_filled.rename(columns={"h3_id": "h3"})
    geom_h3_filled["wkt"] = geom_h3_filled.geometry.to_wkt()

    geom_h3_filled = geom_h3_filled.reindex(
        columns=[route_id_column, "direction", "section_id", "h3", "wkt"]
    )

    # check for routes that are circular and end in cells that overlap with the start
    # Check if any of the last few cells are in the first few cells and remove them
    n_cells_to_check = min(
        5, len(geom_h3_filled) // 2
    )  # Don't check more than half the route
    if n_cells_to_check > 0:
        first_cells = set(geom_h3_filled.iloc[:n_cells_to_check]["h3"])

        # Count consecutive cells from the end that are in the first cells
        cells_to_remove = 0
        for i in range(1, n_cells_to_check + 1):
            if geom_h3_filled.iloc[-i]["h3"] in first_cells:
                cells_to_remove += 1
            else:
                break  # Stop at the first non-overlapping cell from the end

        if cells_to_remove > 0:
            print(
                f"⚠️  Warning: Detected circular route with {cells_to_remove} overlapping cell(s) at the end. Removing them."
            )
            # Remove overlapping cells from the end
            geom_h3_filled = geom_h3_filled.iloc[:-cells_to_remove].copy()

            # Re-sequence section_id to be sequential
            geom_h3_filled["section_id"] = range(len(geom_h3_filled))

    return geom_h3_filled


def fill_h3_gaps(geom_h3, line_geom, h3_column="h3_id", verbose=True):
    """
    Fill gaps in an H3 route by adding shortest paths between non-adjacent consecutive cells.

    When cell at position i is not adjacent to cell at position i+1, this function:
    1. Finds the shortest path between them using h3.grid_path_cells()
    2. Inserts the intermediate cells
    3. Updates section_id values to maintain sequence

    Parameters:
    -----------
    geom_h3 : GeoDataFrame
        GeoDataFrame with H3 cell identifiers and section_id
    h3_column : str
        Name of the column containing H3 cell IDs (default: 'h3_10')
    verbose : bool
        Print information about gaps filled

    Returns:
    --------
    GeoDataFrame
        GeoDataFrame with gaps filled
    """

    # Start with a copy
    df = geom_h3.copy().reset_index(drop=True)

    # We'll build a new dataframe with filled gaps
    new_rows = []
    section_counter = 0
    total_gaps_filled = 0

    for i in range(len(df)):
        current_row = df.iloc[i]
        current_cell = current_row[h3_column]

        # Add current row with updated section_id
        current_row_dict = current_row.to_dict()
        current_row_dict["section_id"] = section_counter
        new_rows.append(current_row_dict)
        section_counter += 1

        # Check if there's a next cell
        if i < len(df) - 1:
            next_cell = df.iloc[i + 1][h3_column]

            # Check if current and next are adjacent
            if not h3.are_neighbor_cells(current_cell, next_cell):
                # Find shortest path between them
                distance = h3.grid_distance(current_cell, next_cell)
                if distance == 2:
                    print(
                        "Distance of 2 detected between cells at positions {} and {}. Attempting to fill gap with common neighbor.".format(
                            i, i + 1
                        )
                    )
                    # find adjacent cells to current and next_cell
                    neighbors_current = set(h3.grid_ring(current_cell, 1))
                    neighbors_next = set(h3.grid_ring(next_cell, 1))
                    common_neighbors = neighbors_current.intersection(neighbors_next)

                    if common_neighbors:
                        # If there's a common neighbor, we can fill the gap with that cell
                        inter_cell = common_neighbors.pop()  # Get one common neighbor
                        cell_polygon = h3_to_polygon(inter_cell)
                        # check if the intermediate cell intersects the line geometry
                        print()
                        if not cell_polygon.intersects(line_geom):
                            print("inter_cell", inter_cell)
                            print(
                                "Retry next common neighbor for cells at positions {} and {} as the first one does not intersect the line geometry.".format(
                                    i, i + 1
                                )
                            )
                            inter_cell = (
                                common_neighbors.pop()
                            )  # Get one common neighbor
                            cell_polygon = h3_to_polygon(inter_cell)
                            if not cell_polygon.intersects(line_geom):
                                print("inter_cell", inter_cell)

                                print(
                                    "Warning: No common neighbor intersects the line geometry for cells at positions {} and {}.".format(
                                        i, i + 1
                                    )
                                )

                        new_row = {
                            h3_column: inter_cell,
                            "section_id": section_counter,
                            "geometry": cell_polygon,
                            "is_filled_gap": True,  # Mark as filled gap
                        }
                        # Copy other relevant columns if they exist
                        for col in ["lrs"]:
                            if col in current_row:
                                new_row[col] = None  # or interpolate if needed

                        new_rows.append(new_row)
                        section_counter += 1
                        total_gaps_filled += 1
                    else:
                        if verbose:
                            print(
                                f"Warning: No common neighbor found between cells at {i} and {i+1}. Distance: {distance}"
                            )
                else:

                    # grid_path_cells returns the path including start and end
                    path = h3.grid_path_cells(current_cell, next_cell)

                    # Skip first (current) and last (next) cells as they're already in the sequence
                    intermediate_cells = path[1:-1]

                    total_gaps_filled += 1

                    # Add intermediate cells
                    for inter_cell in intermediate_cells:
                        # Create a new row for each intermediate cell
                        new_row = {
                            h3_column: inter_cell,
                            "section_id": section_counter,
                            "geometry": h3_to_polygon(inter_cell),
                            "is_filled_gap": True,  # Mark as filled gap
                        }
                        # Copy other relevant columns if they exist
                        for col in ["lrs"]:
                            if col in current_row:
                                new_row[col] = None  # or interpolate if needed

                        new_rows.append(new_row)
                        section_counter += 1

    # Create new GeoDataFrame
    result_df = gpd.GeoDataFrame(new_rows, crs=geom_h3.crs)

    if verbose:
        print(f"\nSummary:")
        print(f"  Original cells: {len(df)}")
        print(f"  Gaps filled: {total_gaps_filled}")
        print(f"  Cells added: {len(result_df) - len(df)}")
        print(f"  Total cells: {len(result_df)}")

    return result_df


def interpolate_points(route_geom_m, interpolating_distance=5):
    """Crea puntos cada X metros a lo largo de una línea (LRS)"""
    epsg_m = geo.get_epsg_m()

    distancias = np.arange(0, route_geom_m.length, interpolating_distance)
    if distancias[-1] < route_geom_m.length:
        distancias = np.append(distancias, route_geom_m.length)

    puntos = [route_geom_m.interpolate(d) for d in distancias]

    gdf_pts = gpd.GeoDataFrame({"lrs": distancias}, geometry=puntos, crs=epsg_m)
    gdf_pts = gdf_pts.sort_values("lrs").reset_index(drop=True)

    return gdf_pts.to_crs(epsg=4326)


def h3_to_polygon(hex_id):
    """Convierte un ID de H3 en una geometría de Polygon para GeoPandas"""
    boundary = h3.cell_to_boundary(hex_id)
    return Polygon([(lng, lat) for lat, lng in boundary])


def turn_child_h3_into_parent_h3(route_h3, parent_res, route_geom):
    # parent_res = 9
    parent_routes_h3_gdf = route_h3.copy()
    parent_routes_h3_gdf = gpd.GeoDataFrame(
        parent_routes_h3_gdf.drop("wkt", axis=1),
        geometry=gpd.GeoSeries.from_wkt(parent_routes_h3_gdf.wkt),
        crs="EPSG:4326",
    )

    if "id_ramal" in route_h3.columns:
        route_id_column = "id_ramal"
    else:
        route_id_column = "id_linea"

    parent_routes_h3_gdf["parent_h3"] = parent_routes_h3_gdf["h3"].map(
        lambda x: h3.cell_to_parent(x, parent_res)
    )
    parent_routes_h3_gdf["block"] = (
        parent_routes_h3_gdf["parent_h3"] != parent_routes_h3_gdf["parent_h3"].shift()
    ).cumsum()

    # Para el par block y parent_h3, se podria borrar la celd res 10 cuyo
    #  parent_h3 no toca el pedazo de ruta que le corresponde a la celda child res 10
    child_cell_union_route_line = (
        gpd.overlay(
            route_geom,
            parent_routes_h3_gdf,
            how="union",
            keep_geom_type=True,
        )
        .dropna(subset=["section_id"])
        .reindex(columns=["h3", "block", "parent_h3", "geometry"])
    )
    child_cell_union_route_line["parent_geometry"] = child_cell_union_route_line[
        "parent_h3"
    ].map(h3_to_polygon)

    idx_to_remove = [
        (row["h3"], row.block)
        for i, row in child_cell_union_route_line.iterrows()
        if not row.geometry.intersects(row.parent_geometry)
    ]
    parent_routes_h3_gdf = parent_routes_h3_gdf[
        ~parent_routes_h3_gdf.set_index(["h3", "block"]).index.isin(idx_to_remove)
    ].reset_index(drop=True)
    parent_routes_h3_gdf["block"] = (
        parent_routes_h3_gdf["parent_h3"] != parent_routes_h3_gdf["parent_h3"].shift()
    ).cumsum()

    # Get a single geometry for each parent_h3 and block combination
    parent_routes_h3_gdf = parent_routes_h3_gdf.groupby(
        ["parent_h3", "block"], as_index=False
    ).first()
    parent_routes_h3_gdf["geometry"] = parent_routes_h3_gdf["parent_h3"].map(
        h3_to_polygon
    )
    parent_routes_h3_gdf = parent_routes_h3_gdf.sort_values("section_id").reset_index(
        drop=True
    )
    parent_routes_h3_gdf["section_id"] = range(len(parent_routes_h3_gdf))
    parent_routes_h3_gdf["wkt"] = parent_routes_h3_gdf.geometry.to_wkt()
    parent_routes_h3_gdf["resolution"] = parent_res
    parent_routes_h3_gdf = parent_routes_h3_gdf.reindex(
        columns=[
            route_id_column,
            "direction",
            "section_id",
            "parent_h3",
            "resolution",
            "wkt",
        ]
    )
    return parent_routes_h3_gdf


def process_all_ramales_into_parent_h3(ramales, parent_res):
    conn_insumos = iniciar_conexion_db(
        tipo="insumos",
        alias_db=leer_configs_generales(autogenerado=False).get("alias_db", ""),
    )
    # Aplicar turn_child_h3_into_parent_h3 a todos los ramales
    parent_ramales = []

    # Obtener los id_ramal únicos
    unique_ramales = ramales["id_ramal"].unique()

    print(f"Procesando {len(unique_ramales)} ramales...")

    for id_ramal in unique_ramales:
        print(f"Procesando ramal: {id_ramal}")

        # Obtener la geometría H3 del ramal desde ramales
        ramal_h3 = ramales[ramales["id_ramal"] == id_ramal].copy()

        # Obtener la geometría original del ramal desde la base de datos
        query = f"SELECT * FROM official_branches_geoms where id_ramal = '{id_ramal}' and direction = 0"
        df = pd.read_sql(query, conn_insumos)
        df["geometry"] = gpd.GeoSeries.from_wkt(df.wkt)
        ramal_geom = gpd.GeoDataFrame(
            df.drop(columns=["wkt"]), geometry="geometry", crs="EPSG:4326"
        )

        # Aplicar la función
        parent_ramal_h3 = turn_child_h3_into_parent_h3(
            route_h3=ramal_h3, parent_res=parent_res, route_geom=ramal_geom
        )

        parent_ramales.append(parent_ramal_h3)

    # Concatenar todos los resultados
    parent_ramales_gdf = pd.concat(parent_ramales, ignore_index=True)
    parent_ramales_gdf["geometry"] = gpd.GeoSeries.from_wkt(parent_ramales_gdf.wkt)
    parent_ramales_gdf = gpd.GeoDataFrame(
        parent_ramales_gdf, geometry="geometry", crs="EPSG:4326"
    )

    print(
        f"\nProcesamiento completo. Total de celdas H3 padre: {len(parent_ramales_gdf)}"
    )
    return parent_ramales_gdf


def create_edges_between_h3_centroids(parent_ramales_gdf):
    # Crear directed edges y linestrings entre centroides de H3 para cada ramal
    edges_data = []

    # Procesar cada ramal
    for id_ramal in parent_ramales_gdf["id_ramal"].unique():
        print(f"Procesando ramal: {id_ramal}")

        # Filtrar y ordenar por section_id
        ramal_data = (
            parent_ramales_gdf[parent_ramales_gdf["id_ramal"] == id_ramal]
            .sort_values("section_id")
            .reset_index(drop=True)
        )

        # Crear edges entre celdas consecutivas
        for i in range(len(ramal_data) - 1):
            current_row = ramal_data.iloc[i]
            next_row = ramal_data.iloc[i + 1]

            current_h3 = current_row["parent_h3"]
            next_h3 = next_row["parent_h3"]

            # Obtener centroides de las celdas H3
            current_centroid_lat, current_centroid_lng = h3.cell_to_latlng(current_h3)
            next_centroid_lat, next_centroid_lng = h3.cell_to_latlng(next_h3)

            # Crear linestring entre centroides
            linestring = LineString(
                [
                    (current_centroid_lng, current_centroid_lat),
                    (next_centroid_lng, next_centroid_lat),
                ]
            )

            # Guardar información del edge
            edges_data.append(
                {
                    "id_ramal": id_ramal,
                    "direction": current_row["direction"],
                    "section_id_from": current_row["section_id"],
                    "section_id_to": next_row["section_id"],
                    "h3_from": current_h3,
                    "h3_to": next_h3,
                    "geometry": linestring,
                }
            )

    # Crear GeoDataFrame con los edges
    edges_gdf = gpd.GeoDataFrame(edges_data, geometry="geometry", crs="EPSG:4326")

    print(f"\nTotal de edges creados: {len(edges_gdf)}")
    return edges_gdf


def turn_edges_into_directed_graph(od_to_graph):
    # Crear un grafo de NetworkX compatible con OSMnx
    G = nx.MultiDiGraph()

    # Extraer todos los nodos únicos (celdas H3)
    unique_h3_cells = set(od_to_graph["h3_1"].unique()) | set(
        od_to_graph["h3_2"].unique()
    )

    print(f"Total de nodos únicos: {len(unique_h3_cells)}")

    # Agregar nodos con sus coordenadas (x=lon, y=lat)
    for h3_cell in unique_h3_cells:
        lat, lng = h3.cell_to_latlng(h3_cell)
        G.add_node(h3_cell, x=lng, y=lat)

    # Agregar edges del grafo
    for idx, row in od_to_graph.iterrows():
        h3_from = row["h3_1"]
        h3_to = row["h3_2"]

        # Calcular longitud del edge (distancia entre centroides)
        geom = row["geometry"]

        # Calcular distancia en metros usando coordenadas
        from_lat, from_lng = h3.cell_to_latlng(h3_from)
        to_lat, to_lng = h3.cell_to_latlng(h3_to)

        # Usar fórmula simple para distancia
        from math import radians, cos, sqrt

        R = 6371000  # Radio de la Tierra en metros

        dlat = radians(to_lat - from_lat)
        dlng = radians(to_lng - from_lng)
        a = dlat**2 + (cos(radians((from_lat + to_lat) / 2)) * dlng) ** 2
        distance_m = R * sqrt(a)

        # Agregar edge bidireccional (ida y vuelta)
        G.add_edge(h3_from, h3_to, length=distance_m, geometry=geom)
        G.add_edge(h3_to, h3_from, length=distance_m, geometry=geom)

    # Configurar atributos del grafo para OSMnx
    G.graph["crs"] = "epsg:4326"
    G.graph["simplified"] = True

    print(f"\nGrafo creado:")
    print(f"  Nodos: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    return G


def turn_edges_into_undirected_graph(edges_gdf):
    # Crear grafo no dirigido desde edges_gdf para evitar duplicación de edges
    G_undirected = nx.Graph()

    # Extraer todos los nodos únicos (celdas H3)
    unique_h3_cells = set(edges_gdf["h3_from"].unique()) | set(
        edges_gdf["h3_to"].unique()
    )

    print(f"Total de nodos únicos: {len(unique_h3_cells)}")

    # Agregar nodos con sus coordenadas (x=lon, y=lat)
    for h3_cell in unique_h3_cells:
        lat, lng = h3.cell_to_latlng(h3_cell)
        G_undirected.add_node(h3_cell, x=lng, y=lat)

    # Agregar edges del grafo (no dirigidos, se eliminan duplicados automáticamente)
    for idx, row in edges_gdf.iterrows():
        h3_from = row["h3_from"]
        h3_to = row["h3_to"]

        # Solo agregar si el edge no existe ya
        if not G_undirected.has_edge(h3_from, h3_to):
            geom = row["geometry"]

            # Calcular distancia en metros
            from_lat, from_lng = h3.cell_to_latlng(h3_from)
            to_lat, to_lng = h3.cell_to_latlng(h3_to)

            from math import radians, cos, sqrt

            R = 6371000  # Radio de la Tierra en metros

            dlat = radians(to_lat - from_lat)
            dlng = radians(to_lng - from_lng)
            a = dlat**2 + (cos(radians((from_lat + to_lat) / 2)) * dlng) ** 2
            distance_m = R * sqrt(a)

            # Agregar edge no dirigido
            G_undirected.add_edge(h3_from, h3_to, length=distance_m, geometry=geom)

    # Configurar atributos del grafo para OSMnx
    G_undirected.graph["crs"] = "epsg:4326"
    G_undirected.graph["simplified"] = True

    print(f"\nGrafo no dirigido creado:")
    print(f"  Nodos: {G_undirected.number_of_nodes()}")
    print(f"  Edges: {G_undirected.number_of_edges()}")

    return G_undirected
