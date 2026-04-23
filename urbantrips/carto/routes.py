import os
import pandas as pd
import geopandas as gpd
import networkx as nx
from osmnx import distance
import statsmodels.api as sm
import h3
from shapely import LineString, Polygon
from itertools import repeat
import numpy as np
import warnings
from urbantrips.carto.carto import (
    create_coarse_h3_from_line,
    floor_rounding,
    create_route_section_ids,
)
from urbantrips.geo import geo
from urbantrips.utils.utils import (
    leer_configs_generales,
    duracion,
    iniciar_conexion_db,
    leer_alias,
    create_branch_ids_sql_filter,
    create_line_ids_sql_filter,
)


import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in line_locate_point_normalized",
    category=RuntimeWarning,
    module=r"shapely\.linear",
)


@duracion
def process_routes_geoms():
    """
    Checks for route geoms in config file, process line and route geoms,
    upload to db, and checks if stops table needs to be created from routes
    """

    # Deletes old data
    delete_old_route_geoms_data()

    # Leer alias de insumos del config de usuario
    configs = leer_configs_generales(autogenerado=False)
    h3_legs_res = configs["resolucion_h3"]
    alias_db = configs.get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_db)

    if route_geoms_not_present(configs):
        print(
            "No hay recorridos en el archivo de config\n" "No se procesaran recorridos"
        )
        return None

    geojson_name = configs["recorridos_geojson"]
    geojson_path = os.path.join("data", "data_ciudad", geojson_name)
    geojson_data = gpd.read_file(geojson_path)
    branches_present = configs["lineas_contienen_ramales"]

    # Check columns
    check_route_geoms_columns(geojson_data, branches_present)

    # if data has lines and branches, split them
    if branches_present:
        branches_routes = geojson_data.reindex(
            columns=["id_ramal", "direction", "geometry"]
        )

        print("Calculando recorridos en h3 con resolucion ", h3_legs_res)
        routes_h3 = []
        for i, route in branches_routes.iterrows():
            print(f"Procesando ruta: {route.id_ramal}")
            geom_h3 = turn_route_geom_into_h3_cells(route, route_id_column="id_ramal")
            routes_h3.append(geom_h3)
        branches_routes_h3 = pd.concat(routes_h3, ignore_index=True)

        branches_routes_h3.to_sql(
            "official_branches_geoms_h3",
            conn_insumos,
            if_exists="replace",
            index=False,
        )

        branches_routes["wkt"] = branches_routes.geometry.to_wkt()
        branches_routes = branches_routes.reindex(
            columns=["id_ramal", "direction", "wkt"]
        )

        branches_routes.to_sql(
            "official_branches_geoms",
            conn_insumos,
            if_exists="replace",
            index=False,
        )

        # produce a line from branches with lowess
        lines_routes = create_line_geom_from_branches(geojson_data)

    else:
        lines_routes = geojson_data.reindex(
            columns=["id_linea", "direction", "geometry"]
        )

    assert not lines_routes.duplicated(
        subset=["id_linea", "direction"]
    ).any(), "id_linea duplicados en geojson de recorridos"

    routes_h3 = []
    for i, route in lines_routes.iterrows():
        print(f"Procesando ruta: {route.id_linea}")
        geom_h3 = turn_route_geom_into_h3_cells(route, route_id_column="id_linea")
        routes_h3.append(geom_h3)
    lines_routes_h3 = pd.concat(routes_h3, ignore_index=True)

    lines_routes["wkt"] = lines_routes.geometry.to_wkt()

    lines_routes = lines_routes.reindex(columns=["id_linea", "direction", "wkt"])
    print("Subiendo tabla de recorridos")

    # Upload geoms
    lines_routes.to_sql(
        "official_lines_geoms",
        conn_insumos,
        if_exists="replace",
        index=False,
    )

    lines_routes_h3.to_sql(
        "official_lines_geoms_h3",
        conn_insumos,
        if_exists="replace",
        index=False,
    )
    conn_insumos.close()


@duracion
def infer_routes_geoms(plotear_lineas=False):
    """
    Esta funcion crea a partir de las etapas un recorrido simplificado
    de las lineas y lo guarda en la db
    """

    conn_data = iniciar_conexion_db(tipo="data")
    # Leer alias de insumos del config de usuario
    configs = leer_configs_generales(autogenerado=False)
    alias_db = configs.get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_db)

    # traer la coordenadas de las etapas con suficientes datos
    q = """
    select e.id_linea,e.longitud,e.latitud
    from etapas e
    """
    etapas = pd.read_sql(q, conn_data)

    recorridos_lowess = etapas.groupby("id_linea").apply(geo.lowess_linea).reset_index()

    recorridos_lowess["wkt"] = recorridos_lowess.geometry.to_wkt()

    # Elminar geometrias invalidas
    validas = recorridos_lowess.geometry.map(lambda g: g.is_valid)

    recorridos_lowess = recorridos_lowess.loc[validas, :]
    recorridos_lowess["direction"] = 0
    recorridos_lowess = recorridos_lowess.reindex(
        columns=["id_linea", "direction", "wkt"]
    )

    recorridos_lowess.to_sql(
        "inferred_lines_geoms",
        conn_insumos,
        if_exists="replace",
        index=False,
    )

    conn_insumos.close()
    conn_data.close()


@duracion
def build_routes_from_official_inferred():

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    # Delete old data
    conn_insumos.execute("DELETE FROM lines_geoms;")
    conn_insumos.execute("DELETE FROM branches_geoms;")
    conn_insumos.commit()

    # Crear una tabla de recorridos unica
    conn_insumos.execute(
        """
        INSERT INTO lines_geoms
            select i.id_linea, i.direction, coalesce(o.wkt,i.wkt) as wkt
            from inferred_lines_geoms i
            left join official_lines_geoms o
            on i.id_linea = o.id_linea
            and i.direction = o.direction
        ;
        """
    )
    conn_insumos.commit()

    # There is no inferred branches
    conn_insumos.execute(
        """
        INSERT INTO branches_geoms
        select * from official_branches_geoms
        ;
        """
    )
    conn_insumos.commit()

    conn_insumos.close()


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
    cols = ["id_linea", "direction", "geometry"]

    assert (
        not geojson_data.id_linea.isna().any()
    ), "id_linea vacios en geojson recorridos"
    assert (
        not geojson_data.direction.isna().any()
    ), "direction vacios en geojson recorridos"
    # check all values for direction are 0 or 1
    assert geojson_data.direction.isin(
        [0, 1]
    ).all(), "direction debe ser 0 o 1 en geojson recorridos"

    if branches_present:
        cols.append("id_ramal")
        assert (
            not geojson_data.id_ramal.isna().any()
        ), "id_ramal vacios en geojson recorridos"
        assert not geojson_data.duplicated(
            subset=["id_ramal", "direction"]
        ).any(), "id_ramal duplicados en geojson recorridos"

    cols = pd.Series(cols)
    columns_ok = cols.isin(geojson_data.columns)

    if not columns_ok.all():
        cols_not_ok = ",".join(cols[~columns_ok].values)

        raise ValueError(f"Faltan columnas en el dataset: {cols_not_ok}")

    # Check geometry type
    geo.check_all_geoms_linestring(geojson_data)


def delete_old_route_geoms_data():
    # Leer alias de insumos del config de usuario
    configs = leer_configs_generales(autogenerado=False)
    alias_db = configs.get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_db)

    conn_insumos.execute("DELETE FROM lines_geoms;")
    conn_insumos.execute("DELETE FROM branches_geoms;")
    conn_insumos.execute("DELETE FROM official_lines_geoms;")
    conn_insumos.execute("DELETE FROM official_branches_geoms;")
    conn_insumos.commit()
    conn_insumos.close()


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
def process_routes_metadata():
    """
    This function reads from config file the locatino of the csv table
    with routes metadata, check if lines and branches are present
    and uploads metadata to the db
    """
    # Leer alias de insumos del config de usuario
    configs = leer_configs_generales(autogenerado=False)
    alias_db = configs.get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_db)

    # Deletes old data
    conn_insumos.execute("DELETE FROM metadata_lineas;")
    conn_insumos.execute("DELETE FROM metadata_ramales;")
    conn_insumos.commit()

    try:
        tabla_lineas = configs["nombre_archivo_informacion_lineas"]
        branches_present = configs["lineas_contienen_ramales"]
    except KeyError:
        tabla_lineas = None
        branches_present = False
        print("No hay tabla con informacion configs")

    # Check modes matches config standarized modes
    try:
        modos_homologados = configs["modos"]
        zipped = zip(modos_homologados.values(), modos_homologados.keys())
        modos_homologados = {k: v for k, v in zipped}

    except KeyError:
        pass

    # Line metadata is mandatory

    print("Leyendo tabla con informacion de lineas")
    ruta = os.path.join("data", "data_ciudad", tabla_lineas)
    info = pd.read_csv(ruta)

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

    # upload to db
    info_lineas.to_sql(
        "metadata_lineas", conn_insumos, if_exists="replace", index=False
    )

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

        info_ramales.to_sql(
            "metadata_ramales", conn_insumos, if_exists="replace", index=False
        )

    conn_insumos.close()


def create_line_g(line_id):
    """
    Takes linea id, read from the stops data
    and produces a line graph composing from branch's graphs

    Parameters
    ----------
    branch_stops : pandas.DataFrame
        branch's stops with order and node_id

    Returns
    -------
    networkx.MultiGraph
        Graph with the branch route id by node_id and ordered
        by stops order
    """
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    query = f"select * from stops where id_linea = {line_id}"
    line_stops = pd.read_sql(query, conn_insumos)

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

    Parameters
    ----------
    line_stops : pandas.DataFrame
        lines's stops with order and node_id

    Returns
    -------
    networkx.MultiGraph
        Graph with the branch route id by node_id and ordered
        by stops branch order
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

    Parameters
    ----------
    branch_stops : pandas.DataFrame
        branch's stops with order and node_id

    Returns
    -------
    networkx.MultiGraph
        Graph with the branch route id by node_id and ordered
        by stops branch order
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


def read_branch_routes(branch_ids):
    """
    This function take a list of branch ids and returns a geodataframe
    with route geoms
    """
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)
    line_ids_where = create_branch_ids_sql_filter(branch_ids)
    q_route_geoms = "select * from branches_geoms" + line_ids_where
    route_geoms = pd.read_sql(q_route_geoms, conn_insumos)
    route_geoms["geometry"] = gpd.GeoSeries.from_wkt(route_geoms.wkt)
    route_geoms = gpd.GeoDataFrame(
        route_geoms.drop("wkt", axis=1), geometry="geometry", crs="EPSG:4326"
    )
    return route_geoms


def read_routes(route_ids, route_type):
    """
    This function take a list of branches or lines ids and returns a geodataframe
    with route geoms

    Parameters
    ----------
    route_ids : list
        list of branches or lines ids
    route_type : str
        branches or lines

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with route geoms
    """
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)
    if route_type == "branches":
        ids_where = create_branch_ids_sql_filter(route_ids)
    else:
        ids_where = create_line_ids_sql_filter(route_ids)

    q_route_geoms = f"select * from {route_type}_geoms" + ids_where

    route_geoms = pd.read_sql(q_route_geoms, conn_insumos)
    route_geoms.columns = ["route_id", "wkt"]
    route_geoms["geometry"] = gpd.GeoSeries.from_wkt(route_geoms.wkt)
    route_geoms = gpd.GeoDataFrame(
        route_geoms.drop("wkt", axis=1), geometry="geometry", crs="EPSG:4326"
    )
    return route_geoms


def get_route_geoms_with_sections_data(line_ids_where, section_meters, n_sections):

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    q_route_geoms = "select * from lines_geoms"
    q_route_geoms = q_route_geoms + line_ids_where
    route_geoms = pd.read_sql(q_route_geoms, conn_insumos)
    route_geoms["geometry"] = gpd.GeoSeries.from_wkt(route_geoms.wkt)
    route_geoms = gpd.GeoDataFrame(route_geoms, geometry="geometry", crs="EPSG:4326")

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


def check_exists_route_section_points_table(route_geoms):
    """
    This function checks if the route section points table exists
    for those lines and n_sections in the route geoms gdf
    """

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)
    q = """
    select distinct id_linea,n_sections, 1 as section_exists from routes_section_id_coords
    """
    route_sections = pd.read_sql(q, conn_insumos)
    conn_insumos.close()

    new_route_geoms = route_geoms.merge(
        route_sections, on=["id_linea", "n_sections"], how="left"
    )
    new_route_geoms = new_route_geoms.loc[
        new_route_geoms.section_exists.isna(), ["id_linea", "n_sections", "geometry"]
    ]

    return new_route_geoms


def upload_route_section_points_table(route_geoms, delete_old_data=False):
    """
    Uploads a table with route section points from a route geom row
    and returns a table with line_id, number of sections and the
    xy point for that section

    Parameters
    ----------
    row : GeoPandas GeoDataFrame
        routes geom GeoDataFrame with geometry, n_sections and line id

    """
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    # delete old records
    if delete_old_data:
        delete_old_routes_section_id_coords_data_q(route_geoms)

    print("Creando tabla de secciones de recorrido")
    route_section_points = pd.concat(
        [create_route_section_points(row) for _, row in route_geoms.iterrows()]
    )

    route_section_points.to_sql(
        "routes_section_id_coords",
        conn_insumos,
        if_exists="append",
        index=False,
    )
    print("Fin creacion de tabla de secciones de recorrido")
    conn_insumos.close()


def delete_old_routes_section_id_coords_data_q(route_geoms):
    """
    Deletes old data in table routes_section_id_coords
    """
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    # create a df with n sections for each line
    delete_df = route_geoms.reindex(columns=["id_linea", "n_sections"])
    for _, row in delete_df.iterrows():
        # Delete old data for those parameters

        q_delete = f"""
            delete from routes_section_id_coords
            where id_linea = {row.id_linea} 
            and n_sections = {row.n_sections}
            """

        cur = conn_insumos.cursor()
        cur.execute(q_delete)
        conn_insumos.commit()

    conn_insumos.close()
    print("Fin borrado datos previos")


def create_route_section_points(row):
    """
    Creates a table with route section points from a route geom row
    and returns a table with line_id, number of sections and the
    xy point for that section

    Parameters
    ----------
    row : GeoPandas GeoSeries
        Row from route geom GeoDataFrame
        with geometry, n_sections and line id

    Returns
    ----------
    pandas.DataFrame
        dataFrame with line id, number of sections and the
        latlong for each section id
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

    Parameters
    ----------
    point : shapely Point
        a Point for the leg's origin or destination
    route_geom : shapely Linestring
        a Linestring representing the leg's route geom
    """
    return floor_rounding(route_geom.project(point, normalized=True))


def build_leg_route_sections_df(row):
    """
    Computes for a leg a table with all sections id traversed by
    that leg based on the origin and destionation's section id

    Parameters
    ----------
    row : dict
        row in a legs df with origin, destination and direction

    Returns
    ----------
    leg_route_sections_df: pandas.DataFrame
        a dataframe with all section ids traversed by the leg's
        trajectory

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

    Parameters
    ----------
    row : dict
        row in a legs df with origin, destination and direction

    Returns
    ----------
    gps_route_sections_df: pandas.DataFrame
        a dataframe with all section ids traversed by the leg's
        trajectory

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
    # route_geom, route_id_column, route_type, direction,
    row,
    route_id_column,
    res=10,
):
    """
    Convierte la geometría de una ruta en una secuencia de celdas H3,
    interpolando puntos a lo largo de la ruta y asignándoles celdas H3.
    """
    print(row)
    print(route_id_column)
    print(row[route_id_column])
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

    print(f"Found {len(gaps)} gaps in the route:")
    if len(gaps) > 0:
        print(f"Found {len(gaps)} gaps ")
    else:
        print("  No gaps found - route is fully connected!")
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

    if non_adjacent_count == 0:
        print("✅ SUCCESS! All consecutive cells in the filled route are adjacent.")
        print(f"   Route length: {len(geom_h3_filled)} cells")
    else:
        print(f"\n⚠️  Warning: {non_adjacent_count} gaps remain")

    geom_h3_filled[route_id_column] = route_id
    geom_h3_filled["direction"] = direction
    geom_h3_filled = geom_h3_filled.rename(columns={"h3_id": "h3"})
    geom_h3_filled["wkt"] = geom_h3_filled.geometry.to_wkt()

    geom_h3_filled = geom_h3_filled.reindex(
        columns=[route_id_column, "direction", "section_id", "h3", "wkt"]
    )
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

    parent_routes_h3_gdf = parent_routes_h3_gdf.reindex(
        columns=[route_id_column, "direction", "parent_h3", "section_id", "wkt"]
    )
    return parent_routes_h3_gdf
