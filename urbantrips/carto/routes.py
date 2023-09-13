from urbantrips.viz.viz import plotear_recorrido_lowess
import os
import pandas as pd
import geopandas as gpd
import networkx as nx
from osmnx import distance
import statsmodels.api as sm
from shapely import LineString
from itertools import repeat
import numpy as np

from urbantrips.geo import geo
from urbantrips.carto import carto
from urbantrips.utils.utils import (leer_configs_generales,
                                    duracion,
                                    iniciar_conexion_db,
                                    leer_alias
                                    )


@duracion
def process_routes_geoms():
    """
    Checks for route geoms in config file, process line and route geoms,
    upload to db, and checks if stops table needs to be created from routes
    """

    # Deletes old data
    delete_old_route_geoms_data()

    configs = leer_configs_generales()

    if route_geoms_not_present(configs):
        print("No hay recorridos en el archivo de config\n"
              "No se procesaran recorridos")
        return None

    geojson_name = configs["recorridos_geojson"]
    geojson_path = os.path.join("data", "data_ciudad", geojson_name)
    geojson_data = gpd.read_file(geojson_path)

    branches_present = configs["lineas_contienen_ramales"]

    # Checl columns
    check_route_geoms_columns(geojson_data, branches_present)

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    # if data has lines and branches, split them
    if branches_present:
        branches_routes = geojson_data\
            .reindex(columns=['id_ramal', 'geometry'])

        branches_routes['wkt'] = branches_routes.geometry.to_wkt()
        branches_routes = branches_routes\
            .reindex(columns=['id_ramal', 'wkt'])

        branches_routes.to_sql(
            "official_branches_geoms", conn_insumos, if_exists="replace",
            index=False,)

        # produce a line from branches with lowess
        lines_routes = create_line_geom_from_branches(geojson_data)

    else:
        lines_routes = geojson_data\
            .reindex(columns=['id_linea', 'geometry'])

    assert not lines_routes.id_linea.duplicated().any(
    ), "id_linea duplicados en geojson de recorridos"

    lines_routes['wkt'] = lines_routes.geometry.to_wkt()

    lines_routes = lines_routes.reindex(columns=['id_linea', 'wkt'])
    print('Subiendo tabla de recorridos')

    # Upload geoms
    lines_routes.to_sql(
        "official_lines_geoms", conn_insumos, if_exists="replace",
        index=False,)

    conn_insumos.close()


@duracion
def infer_routes_geoms(plotear_lineas):
    """
    Esta funcion crea a partir de las etapas un recorrido simplificado
    de las lineas y lo guarda en la db
    """

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    # traer la coordenadas de las etapas con suficientes datos
    q = """
    select e.id_linea,e.longitud,e.latitud
    from etapas e
    """
    etapas = pd.read_sql(q, conn_data)

    recorridos_lowess = etapas.groupby(
        'id_linea').apply(geo.lowess_linea).reset_index()

    if plotear_lineas:
        print('Imprimiento bosquejos de lineas')
        alias = leer_alias()
        [plotear_recorrido_lowess(id_linea, etapas, recorridos_lowess, alias)
         for id_linea in recorridos_lowess.id_linea]

    print("Subiendo recorridos a la db...")
    recorridos_lowess['wkt'] = recorridos_lowess.geometry.to_wkt()

    # Elminar geometrias invalidas
    validas = recorridos_lowess.geometry.map(lambda g: g.is_valid)

    recorridos_lowess = recorridos_lowess.loc[validas, :]
    recorridos_lowess = recorridos_lowess.reindex(columns=['id_linea', 'wkt'])

    recorridos_lowess.to_sql("inferred_lines_geoms",
                             conn_insumos, if_exists="replace", index=False,)

    conn_insumos.close()
    conn_data.close()


@duracion
def build_routes_from_official_inferred():

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    # Delete old data
    conn_insumos.execute("DELETE FROM lines_geoms;")
    conn_insumos.execute("DELETE FROM branches_geoms;")
    conn_insumos.commit()

    # Crear una tabla de recorridos unica
    conn_insumos.execute(
        """
        INSERT INTO lines_geoms
            select i.id_linea,coalesce(o.wkt,i.wkt) as wkt
            from inferred_lines_geoms i
            left join official_lines_geoms o
            on i.id_linea = o.id_linea
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

    lines_routes = geojson_data\
        .groupby('id_linea', as_index=False)\
        .apply(get_line_lowess_from_branch_routes)
    lines_routes.columns = ['id_linea', 'geometry']
    lines_routes = gpd.GeoDataFrame(
        lines_routes, geometry='geometry', crs=epsg_m)

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
    cols = ['id_linea', 'geometry']

    assert not geojson_data.id_linea.isna().any(),\
        "id_linea vacios en geojson recorridos"
    # assert geojson_data.dtypes['id_linea'] == int,\
    #     "id_linea deben ser int en geojson recorridos"

    if branches_present:
        cols.append('id_ramal')
        assert not geojson_data.id_ramal.isna().any(),\
            "id_ramal vacios en geojson recorridos"
        assert not geojson_data.id_ramal.duplicated().any(),\
            "id_ramal duplicados en geojson recorridos"
        # assert geojson_data.dtypes['id_ramal'] == int,\
        #     "id_ramal deben ser int en geojson recorridos"

    cols = pd.Series(cols)
    columns_ok = cols.isin(geojson_data.columns)

    if not columns_ok.all():
        cols_not_ok = ','.join(cols[~columns_ok].values)

        raise ValueError(
            f'Faltan columnas en el dataset: {cols_not_ok}')

    # Check geometry type
    geo.check_all_geoms_linestring(geojson_data)


def delete_old_route_geoms_data():
    conn_insumos = iniciar_conexion_db(tipo='insumos')

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

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    # Deletes old data
    conn_insumos.execute("DELETE FROM metadata_lineas;")
    conn_insumos.execute("DELETE FROM metadata_ramales;")
    conn_insumos.commit()

    configs = leer_configs_generales()

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
        zipped = zip(modos_homologados.values(),
                     modos_homologados.keys())
        modos_homologados = {k: v for k, v in zipped}

    except KeyError:
        pass

    # Line metadata is mandatory

    print('Leyendo tabla con informacion de lineas')
    ruta = os.path.join("data", "data_ciudad", tabla_lineas)
    info = pd.read_csv(ruta)

    # Check all columns are present
    if branches_present:
        cols = ['id_linea', 'nombre_linea',
                            'id_ramal', 'nombre_ramal', 'modo']
    else:
        cols = ['id_linea', 'nombre_linea', 'modo']

    assert pd.Series(cols).isin(info.columns).all(
    ), f"La tabla {ruta} debe tener los campos: {cols}"

    # check no missing data in line id
    assert not info.id_linea.isna().any(), "id_linea no debe ser NULL"

    if 'id_linea_agg' not in info.columns:
        info['id_linea_agg'] = info['id_linea']
        info['nombre_linea_agg'] = info['nombre_linea']

    line_cols = ["id_linea",
                 "nombre_linea",
                 "id_linea_agg",
                 "nombre_linea_agg",
                 "modo",
                 "empresa",
                 "descripcion"]

    assert pd.Series(info.modo.unique()).isin(
        modos_homologados.keys()).all()

    info['modo'] = info['modo'].replace(modos_homologados)

    # fill missing line agg
    info.loc[info.id_linea_agg.isna(
    ), 'nombre_linea_agg'] = info.loc[info.id_linea_agg.isna(), 'nombre_linea']
    info.loc[info.id_linea_agg.isna(
    ), 'id_linea_agg'] = info.loc[info.id_linea_agg.isna(), 'id_linea']

    # keep only line data
    info_lineas = info.reindex(columns=line_cols)
    info_lineas = info_lineas.drop_duplicates(subset='id_linea')

    # upload to db
    info_lineas.to_sql(
        "metadata_lineas", conn_insumos, if_exists="replace",
        index=False)

    if branches_present:
        ramales_cols = ['id_ramal', 'id_linea',
                        'nombre_ramal', 'modo', 'empresa', 'descripcion']

        info_ramales = info.reindex(columns=ramales_cols)

        # Checks for missing and duplicated
        s = "Existen nulos en el campo id_ramal"
        assert not info_ramales.id_ramal.isna().any(), s

        assert not info_ramales.id_ramal.duplicated(
        ).any(), "Existen duplicados en id_ramal"

        info_ramales.to_sql(
            "metadata_ramales", conn_insumos, if_exists="replace",
            index=False)

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
    conn = iniciar_conexion_db(tipo='insumos')
    query = f"select * from stops where id_linea = {line_id}"
    line_stops = pd.read_sql(query, conn)

    branches_id = line_stops.id_ramal.unique()

    G_line = nx.compose_all([create_branch_g_from_stops_df(
        line_stops, branch_id) for branch_id in branches_id])

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
    branch_stops = branch_stops.drop_duplicates(subset='node_id')

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
        "id_linea": branch_stops['id_linea'].unique().item(),
        "id_ramal": branch_stops['id_ramal'].unique().item()
    }
    G = nx.MultiGraph(**metadata)

    branch_stops = branch_stops.sort_values(
        'branch_stop_order').reindex(columns=['node_id', 'node_x', 'node_y'])
    nodes = [(int(row['node_id']), {'x': row['node_x'], 'y':row['node_y']})
             for _, row in branch_stops.iterrows()]
    G.add_nodes_from(nodes)

    edges_from = branch_stops['node_id'].iloc[:-1].map(int)
    edges_to = branch_stops['node_id'].shift(-1).iloc[:-1].map(int)
    edges = [(i, j, 0) for i, j in zip(edges_from, edges_to)]
    G.add_edges_from(edges)

    # add distance in meters
    G = distance.add_edge_lengths(G)

    return G
