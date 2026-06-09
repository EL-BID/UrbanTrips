import logging
import os
import warnings
from itertools import repeat

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from osmnx import distance
from shapely import LineString

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

    if route_geoms_not_present(configs):
        logger.info("No hay recorridos en el archivo de config — No se procesaran recorridos")
        return None

    geojson_name = configs["recorridos_geojson"]
    geojson_path = str(get_paths().input_dir / geojson_name)
    geojson_data = gpd.read_file(geojson_path)

    branches_present = configs["lineas_contienen_ramales"]

    # If the geojson has direction-split rows, keep one row per ramal/line (first occurrence)
    id_col = "id_ramal" if branches_present else "id_linea"
    if geojson_data[id_col].duplicated().any():
        geojson_data = geojson_data.drop_duplicates(subset=id_col, keep="first")

    # Check columns
    check_route_geoms_columns(geojson_data, branches_present)

    # if data has lines and branches, split them
    if branches_present:
        branches_routes = geojson_data.reindex(columns=["id_ramal", "geometry"])

        logger.info("Calculando recorridos en H3 con resolución %s", h3_legs_res)
        branches_routes_h3 = [
            create_coarse_h3_from_line(
                branch_geom.geometry, h3_legs_res, branch_geom.id_ramal
            )
            for _, branch_geom in branches_routes.iterrows()
        ]
        branches_routes_h3 = pd.concat(branches_routes_h3, ignore_index=True)
        branches_routes_h3["wkt"] = branches_routes_h3.geometry.to_wkt()
        branches_routes_h3 = branches_routes_h3.reindex(
            columns=["route_id", "section_id", "h3", "wkt"]
        ).rename(columns={"route_id": "id_ramal"})

        ctx.insumos.save_raw(branches_routes_h3, "official_branches_geoms_h3")

        branches_routes["wkt"] = branches_routes.geometry.to_wkt()
        branches_routes = branches_routes.reindex(columns=["id_ramal", "wkt"])

        ctx.insumos.save_raw(branches_routes, "official_branches_geoms")

        # produce a line from branches with lowess
        lines_routes = create_line_geom_from_branches(geojson_data)

    else:
        lines_routes = geojson_data.reindex(columns=["id_linea", "geometry"])

    assert (
        not lines_routes.id_linea.duplicated().any()
    ), "id_linea duplicados en geojson de recorridos - Verificar si no contiene ramales (y modificar parámetro en configuraciones_generales.yaml)"

    lines_routes["wkt"] = lines_routes.geometry.to_wkt()

    lines_routes = lines_routes.reindex(columns=["id_linea", "wkt"])
    logger.info("Subiendo tabla de recorridos")

    ctx.insumos.save_raw(lines_routes, "official_lines_geoms")


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

    recorridos_lowess["wkt"] = recorridos_lowess.geometry.to_wkt()

    # Elminar geometrias invalidas
    validas = recorridos_lowess.geometry.map(lambda g: g.is_valid)

    recorridos_lowess = recorridos_lowess.loc[validas, :]
    recorridos_lowess = recorridos_lowess.reindex(columns=["id_linea", "wkt"])

    ctx.insumos.save_raw(recorridos_lowess, "inferred_lines_geoms")


@duracion
def build_routes_from_official_inferred(ctx: StorageContext):

    for table in ("lines_geoms", "branches_geoms"):
        try:
            ctx.insumos.execute(f"DELETE FROM {table}")
        except Exception:
            pass

    ctx.insumos.execute(
        """
        INSERT INTO lines_geoms
            select i.id_linea, coalesce(o.wkt, i.wkt) as wkt
            from inferred_lines_geoms i
            left join official_lines_geoms o
            on i.id_linea = o.id_linea
        """
    )

    ctx.insumos.execute(
        """
        INSERT INTO branches_geoms
        select * from official_branches_geoms
        """
    )


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

    lines_routes = geojson_data.groupby("id_linea", as_index=False).apply(
        get_line_lowess_from_branch_routes
    )
    lines_routes.columns = ["id_linea", "geometry"]
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

    assert not geojson_data.id_linea.isna().any(), (
        "id_linea vacios en geojson recorridos"
    )

    if branches_present:
        cols.append("id_ramal")
        assert not geojson_data.id_ramal.isna().any(), (
            "id_ramal vacios en geojson recorridos"
        )
        assert not geojson_data.id_ramal.duplicated().any(), (
            "id_ramal duplicados en geojson recorridos"
        )

    cols = pd.Series(cols)
    columns_ok = cols.isin(geojson_data.columns)

    if not columns_ok.all():
        cols_not_ok = ",".join(cols[~columns_ok].values)

        raise ValueError(f"Faltan columnas en el dataset: {cols_not_ok}")

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

    assert pd.Series(cols).isin(info.columns).all(), (
        f"La tabla {ruta} debe tener los campos: {cols}"
    )

    # check no missing data in line id
    assert not info.id_linea.isna().any(), "id_linea no debe ser NULL"
    # fill nombre_linea from id_linea when absent
    info["nombre_linea"] = info["nombre_linea"].fillna(info["id_linea"].astype(str))

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

        assert not info_ramales.id_ramal.duplicated().any(), (
            "Existen duplicados en id_ramal"
        )

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
