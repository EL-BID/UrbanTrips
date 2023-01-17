from datetime import datetime
import networkx as nx
import osmnx as ox
import pandas as pd
from pandas.io.sql import DatabaseError
import numpy as np
import itertools
import os
import geopandas as gpd
from shapely.geometry import LineString
from shapely import wkt
import statsmodels.api as sm
import h3
from networkx import NetworkXNoPath
import multiprocessing
from itertools import repeat
from math import ceil
from multiprocessing import Pool, Manager
from functools import partial
from math import sqrt
from urbantrips.geo.geo import (
    get_stop_hex_ring, h3togeo, add_geometry,
    create_voronoi, normalizo_lat_lon, h3dist
)
from urbantrips.viz.viz import (
    plotear_recorrido_lowess,
    plot_voronoi_zones)
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    leer_alias)


@duracion
def update_stations_catchment_area(ring_size):
    """
    Esta funcion toma la matriz de validacion de paradas
    y la actualiza en base a datos de fechas que no esten
    ya en la matriz
    """

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    # Leer las paradas en base a las etapas
    q = """
    select id_linea,h3_o as parada from etapas
    group by id_linea,h3_o having count(*) >1 and parada <> 0
    """
    paradas_etapas = pd.read_sql(q, conn_data)

    # Leer las paradas ya existentes en la matriz
    q = """
    select distinct id_linea, parada, 1 as m from matriz_validacion
    """
    paradas_en_matriz = pd.read_sql(q, conn_insumos)

    # Detectar que paradas son nuevas para cada linea
    paradas_nuevas = paradas_etapas\
        .merge(paradas_en_matriz,
               on=['id_linea', 'parada'],
               how='left')

    paradas_nuevas = paradas_nuevas.loc[paradas_nuevas.m.isna(), [
        'id_linea', 'parada']]

    if len(paradas_nuevas) > 0:
        areas_influencia_nuevas = pd.concat(
            (map(get_stop_hex_ring, np.unique(paradas_nuevas['parada']),
             itertools.repeat(ring_size))))
        matriz_nueva = paradas_nuevas.merge(
            areas_influencia_nuevas, how='left', on='parada')

        # Subir a la db
        print("Subiendo matriz a db")
        matriz_nueva.to_sql("matriz_validacion", conn_insumos,
                            if_exists="append", index=False)
        print("Fin actualizacion matriz de validacion")
    else:
        print("La matriz de validacion ya tiene los datos más actuales" +
              " en base a la informacion existente en la tabla de etapas")
    return None


@duracion
def upload_routes_geoms():
    """
    Esta funcion lee la ubicacion del archivo de recorridos de
    la ciudad y crea una tabla con los mismos con la geometria en WKT
    """

    conn_insumos = iniciar_conexion_db(tipo='insumos')
    configs = leer_configs_generales()
    try:
        nombre_recorridos = configs["recorridos_geojson"]
        if nombre_recorridos is not None:
            print('Leyendo tabla de recorridos')
            q = f"""
            select distinct id_linea
            from recorridos_reales
            """
            lineas_existentes = pd.read_sql(q, conn_insumos)
            lineas_existentes = lineas_existentes.id_linea

            ruta = os.path.join("data", "data_ciudad", nombre_recorridos)
            recorridos = gpd.read_file(ruta)
            print('Elminando lineas que ya estan en tabla de recorridos')
            recorridos = recorridos.loc[~recorridos.id_linea.isin(
                lineas_existentes), ]
            if len(recorridos) == 0:
                print('Todas las lineas en el geojson se encuentran en la db')
                return None

            # exigir que sean todos LineString
            tipo_no_linestring = pd.Series(
                [not isinstance(g, type(LineString()))
                 for g in recorridos['geometry']])

            if tipo_no_linestring.any():
                lin_str = recorridos.loc[tipo_no_linestring, 'id_linea'].map(
                    str)

                print("La geometria de las lineas debe ser LineString en 2D")
                print(
                    "Hay lineas que no son de este tipo. Editarlas y " +
                    "volver a cargar el dataset")
                print(
                    ','.join(lin_str))
                recorridos = recorridos.loc[~tipo_no_linestring, :]

                if len(recorridos) == 0:
                    print('Se eliminaron las geometrias no LineString')
                    print('y las que ya estaban en la db')
                    print('Y el dataset quedo vacio')
                    return None

            recorridos['wkt'] = recorridos.geometry.to_wkt()

            recorridos = recorridos.reindex(columns=['id_linea', 'wkt'])
            print('Subiendo tabla de recorridos')
            recorridos.to_sql(
                "recorridos_reales", conn_insumos, if_exists="append",
                index=False,)
    except KeyError:
        print("No hay nombre de archivo de recorridos en configs")
    conn_insumos.close()


def infer_routes_geoms(plotear_lineas):
    """
    Esta funcion crea a partir de las etapas un recorrido simplificado
    de las lineas y lo guarda en la db
    """
    print('Creo líneas de transporte')

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    # traer la coordenadas de las etapas con suficientes datos
    q = """
    select e.id_linea,e.longitud,e.latitud
    from etapas e
    """
    etapas = pd.read_sql(q, conn_data)

    recorridos_lowess = etapas.groupby(
        'id_linea').apply(lowess_linea).reset_index()

    if plotear_lineas:
        print('Imprimiento bosquejos de lineas')
        alias = leer_alias()
        [plotear_recorrido_lowess(id_linea, etapas, recorridos_lowess, alias)
         for id_linea in recorridos_lowess.id_linea]

    print("Subiendo recorridos a la db...")
    recorridos_lowess['wkt'] = recorridos_lowess.geometry.to_wkt()
    recorridos_lowess = recorridos_lowess.reindex(columns=['id_linea', 'wkt'])
    # Elminar geometrias invalidas
    geoms = recorridos_lowess.wkt.apply(wkt.loads)
    validas = geoms.map(lambda g: g.is_valid)
    recorridos_lowess = recorridos_lowess.loc[validas, :]

    recorridos_lowess.to_sql("recorridos_estimados",
                             conn_insumos, if_exists="replace", index=False,)

    # Crear una tabla de recorridos unica
    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS recorridos AS
            select e.id_linea,coalesce(r.wkt,e.wkt) as wkt
            from recorridos_estimados e
            left join recorridos_reales r
            on e.id_linea = r.id_linea
        ;
        """
    )

    conn_insumos.close()
    conn_data.close()


def lowess_linea(df):
    """
    Esta funcion toma un df de etapas para una linea
    y produce el geom de la linea simplificada en un
    gdf con el id linea
    """
    id_linea = df.id_linea.unique()[0]
    print("Obteniendo lowess linea:", id_linea)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
        df['longitud'], df['latitud']), crs=4326).to_crs(9265)
    y = gdf.geometry.y
    x = gdf.geometry.x
    lowess = sm.nonparametric.lowess
    lowess_points = lowess(x, y, frac=0.4, delta=500)
    lowess_points_df = pd.DataFrame(lowess_points.tolist(), columns=['y', 'x'])
    lowess_points_df = lowess_points_df.drop_duplicates()

    if len(lowess_points_df) > 1:
        geom = LineString([(x, y) for x, y in zip(
            lowess_points_df.x, lowess_points_df.y)])
        out = gpd.GeoDataFrame({'geometry': geom}, geometry='geometry',
                               crs='EPSG:9265', index=[0]).to_crs(4326)
        return out

    else:
        print("Imposible de generar una linea lowess para id_linea = ",
              id_linea)


@duracion
def create_zones_table():
    """
    This function takes orign geo data from etapas and geoms from zones
    in the config file and produces a table with the corresponding zone
    for each h3 with data in etapas
    """

    print("Creo zonificación para matrices OD")

    conn_insumos = iniciar_conexion_db(tipo='insumos')
    conn_data = iniciar_conexion_db(tipo='data')

    # leer origenes de la tabla etapas
    etapas = pd.read_sql_query(
        """
        SELECT id, h3_o as h3, latitud, longitud from etapas
        """,
        conn_data,
    )

    # Lee la tabla zonas o la crea
    try:
        zonas_ant = pd.read_sql_query(
            """
            SELECT * from zonas
            """,
            conn_insumos,
        )
    except DatabaseError as e:
        print("No existe la tabla zonas en la base")
        zonas_ant = pd.DataFrame([])

    # A partir de los origenes de etapas, crea una nueva tabla zonas
    # con el promedio de la latitud y longitud
    zonas = (
        etapas.groupby(
            "h3",
            as_index=False,
        ).agg({'id': 'count',
               'latitud': 'mean',
               'longitud': 'mean'}).rename(columns={'id': 'fex'})
    )
    # TODO: redo how geoms are created here
    zonas = pd.concat([zonas, zonas_ant], ignore_index=True)
    agg_dict = {
        'fex': 'mean',
        'latitud': 'mean',
        'longitud': 'mean',
    }
    zonas = zonas.groupby("h3",
                          as_index=False,
                          ).agg(agg_dict)

    # Crea la latitud y la longitud en base al h3
    zonas["origin"] = zonas["h3"].apply(h3togeo)
    zonas["lon"] = (
        zonas["origin"].str.split(",").apply(
            lambda x: x[1]).str.strip().astype(float)
    )
    zonas["lat"] = (
        zonas["origin"].str.split(",").apply(
            lambda x: x[0]).str.strip().astype(float)
    )

    zonas = gpd.GeoDataFrame(
        zonas,
        geometry=gpd.points_from_xy(zonas["lon"], zonas["lat"]),
        crs=4326,
    )
    zonas = zonas.drop(["origin", "lon", "lat"], axis=1)

    # Suma a la tabla las zonificaciones del config
    configs = leer_configs_generales()
    if configs['zonificaciones']:

        for n in range(0, 5):
            try:
                file_geo = configs["zonificaciones"][f"geo{n+1}"]
                var_geo = configs["zonificaciones"][f"var{n+1}"]
                zn_path = os.path.join("data", "data_ciudad", file_geo)
                zn = gpd.read_file(zn_path)
                zn = zn.drop_duplicates()
                zn = zn[[var_geo, "geometry"]]
                zonas = gpd.sjoin(zonas, zn, how="left")
                zonas = zonas.drop(["index_right"], axis=1)

            except KeyError:
                pass

    zonas = zonas.drop(["geometry"], axis=1)
    zonas.to_sql("zonas", conn_insumos, if_exists="replace", index=False)
    conn_insumos.close()
    print("Graba zonas en sql lite")


@duracion
def create_voronoi_zones(res=8, max_zonas=15, show_map=False):
    """
    This function creates transport zones based on the points in the dataset
    """
    print('Crea zonas de transporte')

    alias = leer_alias()
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    # Leer informacion en tabla zonas
    zonas = pd.read_sql_query(
        """
        SELECT *
        FROM zonas
        """,
        conn_insumos,
    )

    # Si existe la columna de zona voronoi la elimina
    if 'Zona_voi' in zonas.columns:
        zonas.drop(['Zona_voi'],
                   axis=1,
                   inplace=True)

    # agrega datos a un hexagono mas grande
    zonas['h3_r'] = zonas['h3'].apply(h3.h3_to_parent,
                                      res=res)

    # Computa para ese hexagono el promedio ponderado de latlong
    hexs = zonas.groupby('h3_r',
                         as_index=False).fex.sum()
    hexs = hexs.merge(zonas
                      .groupby('h3_r')
                      .apply(
                          lambda x: np.average(x['longitud'], weights=x['fex']))
                      .reset_index().rename(columns={0: 'longitud'}),
                      how='left')
    hexs = hexs.merge(zonas
                      .groupby('h3_r')
                      .apply(
                          lambda x: np.average(x['latitud'], weights=x['fex'])
                      ).reset_index().rename(columns={0: 'latitud'}),
                      how='left')

    hexs = gpd.GeoDataFrame(
        hexs,
        geometry=gpd.points_from_xy(hexs['longitud'], hexs['latitud']),
        crs=4326,
    )

    cant_zonas = len(hexs)+10
    k_ring = 1

    if cant_zonas <= max_zonas:
        hexs2 = hexs.copy()

    while cant_zonas > max_zonas:
        # Construye un set de hexagonos aun mas grandes
        hexs2 = hexs.copy()
        hexs2['h3_r2'] = hexs2.h3_r.apply(h3.h3_to_parent, res=res-1)
        hexs2['geometry'] = hexs2.h3_r2.apply(add_geometry)
        hexs2 = hexs2.sort_values(['h3_r2', 'fex'], ascending=[True, False])
        hexs2['orden'] = hexs2.groupby(['h3_r2']).cumcount()
        hexs2 = hexs2[hexs2.orden == 0]

        hexs2 = hexs2.sort_values('fex', ascending=False)
        hexs['cambiado'] = 0
        for i in hexs2.h3_r.tolist():
            vecinos = h3.k_ring(i, k_ring)
            hexs.loc[(hexs.h3_r.isin(vecinos)) & (
                hexs.cambiado == 0), 'h3_r'] = i
            hexs.loc[(hexs.h3_r.isin(vecinos)) & (
                hexs.cambiado == 0), 'cambiado'] = 1

        hexs_tmp = hexs.groupby('h3_r', as_index=False).fex.sum()
        hexs_tmp = hexs_tmp.merge(
            hexs
            .groupby('h3_r')
            .apply(
                lambda x: np.average(x['longitud'], weights=x['fex']))
            .reset_index().rename(columns={0: 'longitud'}),
            how='left')
        hexs_tmp = hexs_tmp.merge(
            hexs
            .groupby('h3_r')
            .apply(lambda x: np.average(x['latitud'], weights=x['fex']))
            .reset_index().rename(columns={0: 'latitud'}),
            how='left')
        hexs_tmp = gpd.GeoDataFrame(
            hexs_tmp,
            geometry=gpd.points_from_xy(hexs_tmp['longitud'],
                                        hexs_tmp['latitud']),
            crs=4326)

        hexs = hexs_tmp.copy()

        if cant_zonas == len(hexs):
            k_ring += 1
        else:
            cant_zonas = len(hexs)

    voi = create_voronoi(hexs)
    voi = gpd.sjoin(voi,
                    hexs[['fex', 'geometry']],
                    how='left')
    voi = voi.sort_values('fex',
                          ascending=False)
    voi = voi.drop(['Zona',
                    'index_right'],
                   axis=1)
    voi = voi.reset_index(drop=True).reset_index().rename(
        columns={'index': 'Zona_voi'})
    voi['Zona_voi'] = voi['Zona_voi']+1
    voi['Zona_voi'] = voi['Zona_voi'].astype(str)

    zonas = zonas.drop(['h3_r'], axis=1)
    zonas['geometry'] = zonas['h3'].apply(add_geometry)

    zonas = gpd.GeoDataFrame(
        zonas,
        geometry='geometry',
        crs=4326)
    zonas['geometry'] = zonas['geometry'].representative_point()

    zonas = gpd.sjoin(zonas,
                      voi[['Zona_voi', 'geometry']],
                      how='left')

    zonas = zonas.drop(['index_right', 'geometry'], axis=1)
    zonas.to_sql("zonas", conn_insumos, if_exists="replace", index=False)
    conn_insumos.close()
    print("Graba zonas en sql lite")

    # Plotea geoms de voronoi
    plot_voronoi_zones(voi, hexs, hexs2, show_map, alias)


@duracion
def create_distances_table(use_parallel=False):
    """
    Esta tabla toma los h3 de la tablas de etapas y viajes
    y calcula diferentes distancias para cada par que no tenga
    """
    configs = leer_configs_generales()
    resolucion_h3 = configs["resolucion_h3"]
    distancia_entre_hex = h3.edge_length(resolution=resolucion_h3, unit="km")
    distancia_entre_hex = distancia_entre_hex * 2

    conn_insumos = iniciar_conexion_db(tipo='insumos')
    conn_data = iniciar_conexion_db(tipo='data')

    q = """
    select distinct h3_o,h3_d
    from viajes
    union
    select distinct h3_o,h3_d
    from (
            SELECT h3_o,h3_d
            from etapas
            INNER JOIN destinos
            ON etapas.id = destinos.id
    )
    """
    pares_h3_data = pd.read_sql_query(q, conn_data).dropna()

    q = """
    select h3_o,h3_d, 1 as d from distancias
    """
    pares_h3_distancias = pd.read_sql_query(q, conn_insumos).dropna()

    # Unir pares od h desde data y desde distancias y quedarse con
    # los que estan en data pero no en distancias
    pares_h3 = pares_h3_data\
        .merge(pares_h3_distancias, how='left')
    pares_h3 = pares_h3.loc[pares_h3.d.isna(), ['h3_o', 'h3_d']]

    print(f"Hay {len(pares_h3)} nuevos pares od para sumar a tabla distancias")
    print(f"de los {len(pares_h3_data)} originales en la data.")

    if len(pares_h3) > 0:
        pares_h3_norm = normalizo_lat_lon(pares_h3)

        # usa la función osmnx para distancias en caso de error con Pandana
        print('')
        print('No se pudo usar la librería pandana. ')
        print('Se va a utilizar osmnx para el cálculo de distancias')
        print('')
        agg2 = pares_h3_norm.groupby(
            ['h3_o_norm', 'h3_d_norm'],
            as_index=False).size().drop(['size'], axis=1)

        agg2 = calculo_distancias_osm(
            agg2,
            h3_o="h3_o_norm",
            h3_d="h3_d_norm",
            distancia_entre_hex=distancia_entre_hex,
            processing="osmnx",
            modes=["drive"],
            use_parallel=use_parallel
        )

        distancias_new = pares_h3_norm.merge(agg2, how='left', on=[
            "h3_o_norm", 'h3_d_norm'])

        cols = ['h3_o', 'h3_d', 'h3_o_norm', 'h3_d_norm',
                'distance_osm_drive', 'distance_osm_walk', 'distance_h3']

        distancias_new = distancias_new.reindex(columns=cols)
        distancias_new.to_sql("distancias", conn_insumos,
                              if_exists="append", index=False)


def calculo_distancias_osm(
    df,
    origin="",
    destination="",
    lat_o_tmp="",
    lon_o_tmp="",
    lat_d_tmp="",
    lon_d_tmp="",
    h3_o="",
    h3_d="",
    processing="osmnx",
    modes=["drive", "walk"],
    distancia_entre_hex=1,
    use_parallel=False
):

    cols = df.columns.tolist()

    if len(lat_o_tmp) == 0:
        lat_o_tmp = "lat_o_tmp"
    if len(lon_o_tmp) == 0:
        lon_o_tmp = "lon_o_tmp"
    if len(lat_d_tmp) == 0:
        lat_d_tmp = "lat_d_tmp"
    if len(lon_d_tmp) == 0:
        lon_d_tmp = "lon_d_tmp"

    if (lon_o_tmp not in df.columns) | (lat_o_tmp not in df.columns):
        if (origin not in df.columns) & (len(h3_o) > 0):
            origin = "origin"
            df[origin] = df[h3_o].apply(h3togeo)
        df["lon_o_tmp"] = (
            df[origin].str.split(",").apply(
                lambda x: x[1]).str.strip().astype(float)
        )
        df["lat_o_tmp"] = (
            df[origin].str.split(",").apply(
                lambda x: x[0]).str.strip().astype(float)
        )

    if (lon_d_tmp not in df.columns) | (lat_d_tmp not in df.columns):
        if (destination not in df.columns) & (len(h3_d) > 0):
            destination = "destination"
            df[destination] = df[h3_d].apply(h3togeo)
        df["lon_d_tmp"] = (
            df[destination]
            .str.split(",")
            .apply(lambda x: x[1])
            .str.strip()
            .astype(float)
        )
        df["lat_d_tmp"] = (
            df[destination]
            .str.split(",")
            .apply(lambda x: x[0])
            .str.strip()
            .astype(float)
        )

    ymin, xmin, ymax, xmax = (
        min(df["lat_o_tmp"].min(), df["lat_d_tmp"].min()),
        min(df["lon_o_tmp"].min(), df["lon_d_tmp"].min()),
        max(df["lat_o_tmp"].max(), df["lat_d_tmp"].max()),
        max(df["lon_o_tmp"].max(), df["lon_d_tmp"].max()),
    )
    xmin -= 0.2
    ymin -= 0.2
    xmax += 0.2
    ymax += 0.2

    var_distances = []

    for mode in modes:
        print("")
        print(f"Coords OSM {mode} - Download map")
        print("ymin, xmin, ymax, xmax", ymin, xmin, ymax, xmax)
        print('Comienzo descarga de red', datetime.now())

        print("")
        G = ox.graph_from_bbox(ymax, ymin, xmax, xmin, network_type=mode)
        print('Fin descarga de red', datetime.now())

        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        nodes_from = ox.distance.nearest_nodes(
            G, df[lon_o_tmp].values, df[lat_o_tmp].values, return_dist=True
        )

        nodes_to = ox.distance.nearest_nodes(
            G, df[lon_d_tmp].values, df[lat_d_tmp].values, return_dist=True
        )
        nodes_from = nodes_from[0]
        nodes_to = nodes_to[0]

        if use_parallel:
            results = run_network_distance_parallel(
                mode, G, nodes_from, nodes_to)
            df[f"distance_osm_{mode}"] = results

        else:
            df = run_network_distance_not_parallel(
                df, mode, G, nodes_from, nodes_to)

        var_distances += [f"distance_osm_{mode}"]
        df[f"distance_osm_{mode}"] = (
            df[f"distance_osm_{mode}"] / 1000).round(2)

        print("")

    condition = ('distance_osm_drive' in df.columns) & (
        'distance_osm_walk' in df.columns)

    if condition:
        mask = (df.distance_osm_drive * 1.3) < df.distance_osm_walk
        df.loc[mask, "distance_osm_walk"] = df.loc[mask, "distance_osm_drive"]

    if 'distance_osm_drive' in df.columns:
        df.loc[df.distance_osm_drive > 2000, "distance_osm_drive"] = np.nan
    if 'distance_osm_walk' in df.columns:
        df.loc[df.distance_osm_walk > 2000, "distance_osm_walk"] = np.nan

    df = df[cols + var_distances]

    if (len(h3_o) > 0) & (len(h3_d) > 0):
        df["distance_h3"] = df[[h3_o, h3_d]].apply(
            h3dist,
            axis=1,
            distancia_entre_hex=distancia_entre_hex,
            h3_o=h3_o,
            h3_d=h3_d
        )

    return df


def run_network_distance_not_parallel(df, mode, G, nodes_from, nodes_to):
    """
    This function will run the networkd distance using
    pandas apply method
    """
    df["node_from"] = nodes_from
    df["node_to"] = nodes_to

    df = df.reset_index().rename(columns={"index": "idmatrix"})
    df[f"distance_osm_{mode}"] = df.apply(
        lambda x: distancias_osmnx(
            x["idmatrix"],
            x["node_from"],
            x["node_to"],
            G=G,
            lenx=len(df),
        ),
        axis=1,
    )
    return df


def distancias_osmnx(idmatrix, node_from, node_to, G, lenx):
    """
    Función de apoyo de measure_distances_osm
    """

    if idmatrix % 2000 == 0:
        date_str = datetime.now().strftime("%H:%M:%S")
        print(f"{date_str} processing {int(idmatrix)} / {lenx}")

    try:
        ret = nx.shortest_path_length(G, node_from, node_to, weight="length")
    except:
        ret = np.nan
    return ret


def run_network_distance_parallel(mode, G, nodes_from, nodes_to):
    """
    This function runs the network distance in parallel
    """
    n_cores = max(int(multiprocessing.cpu_count() - 1), 1)
    n = len(nodes_from)
    chunksize = int(sqrt(n) * 10)

    print(f'Comenzando a correr distancias para {n} pares OD',
          datetime.now().strftime("%H:%M:%S"))
    print("Este proceso puede demorar algunas horas dependiendo del tamaño de la ciudad" +
          " y si se corre por primera vez por lo que en la base de insumos no estan estos pares")

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(partial(get_network_distance_osmnx, G=G), zip(
            nodes_from, nodes_to), chunksize=chunksize)

    print('Distancias calculadas:', datetime.now().strftime("%H:%M:%S"))

    return results


def get_network_distance_osmnx(par, G, *args, **kwargs):
    node_from, node_to = par
    try:
        out = nx.shortest_path_length(G, node_from, node_to, weight="length")
    except NetworkXNoPath:
        out = np.nan
    return out
