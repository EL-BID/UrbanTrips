from datetime import datetime
import networkx as nx
import multiprocessing
from functools import partial
from math import sqrt
import osmnx as ox
import pandas as pd
from pandas.io.sql import DatabaseError
import numpy as np
import itertools
import os
import geopandas as gpd
import h3
from shapely.geometry import Point, MultiPolygon, Polygon
from networkx import NetworkXNoPath
from pandana.loaders import osm as osm_pandana
from urbantrips.geo.geo import (
    get_stop_hex_ring,
    h3togeo, h3_from_row,
    add_geometry,
    create_voronoi,
    normalizo_lat_lon,
    h3dist,
    bring_latlon,
)
from urbantrips.viz import viz
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    leer_alias, levanto_tabla_sql, guardar_tabla_sql
)

import subprocess


def get_library_version(library_name):
    result = subprocess.run(
        ["pip", "show", library_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if line.startswith("Version:"):
                return line.split(":")[1].strip()
    return None


@duracion
def update_stations_catchment_area(ring_size):
    """
    Esta funcion toma la matriz de validacion de paradas
    y la actualiza en base a datos de fechas que no esten
    ya en la matriz
    """

    conn_data = iniciar_conexion_db(tipo="data")
    conn_insumos = iniciar_conexion_db(tipo="insumos")

    # Leer las paradas en base a las etapas
    q = """
    select id_linea,h3_o as parada from etapas
    """
    paradas_etapas = pd.read_sql(q, conn_data)

    metadata_lineas = pd.read_sql_query(
        """
        SELECT *
        FROM metadata_lineas
        """,
        conn_insumos,
    )

    paradas_etapas = paradas_etapas.merge(
        metadata_lineas[["id_linea", "id_linea_agg"]], how="left", on="id_linea"
    ).drop(["id_linea"], axis=1)

    paradas_etapas = paradas_etapas.groupby(
        ["id_linea_agg", "parada"], as_index=False
    ).size()
    paradas_etapas = paradas_etapas[(paradas_etapas["size"] > 1)].drop(["size"], axis=1)

    # Leer las paradas ya existentes en la matriz
    q = """
    select distinct id_linea_agg, parada, 1 as m from matriz_validacion
    """
    paradas_en_matriz = pd.read_sql(q, conn_insumos)

    # Detectar que paradas son nuevas para cada linea
    paradas_nuevas = paradas_etapas.merge(
        paradas_en_matriz, on=["id_linea_agg", "parada"], how="left"
    )

    paradas_nuevas = paradas_nuevas.loc[
        paradas_nuevas.m.isna(), ["id_linea_agg", "parada"]
    ]

    if len(paradas_nuevas) > 0:
        areas_influencia_nuevas = pd.concat(
            (
                map(
                    get_stop_hex_ring,
                    np.unique(paradas_nuevas["parada"]),
                    itertools.repeat(ring_size),
                )
            )
        )
        matriz_nueva = paradas_nuevas.merge(
            areas_influencia_nuevas, how="left", on="parada"
        )

        # Subir a la db
        print("Subiendo matriz a db")
        matriz_nueva.to_sql(
            "matriz_validacion", conn_insumos, if_exists="append", index=False
        )
        print("Fin actualizacion matriz de validacion")
    else:
        print(
            "La matriz de validacion ya tiene los datos más actuales"
            + " en base a la informacion existente en la tabla de etapas"
        )
    return None

def h3_to_polygon(h3_index):
    # Obtener las coordenadas del hexágono
    boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
    return Polygon(boundary)

def guardo_zonificaciones():
    configs = leer_configs_generales()
    alias = leer_alias()
    matriz_zonas = []
    vars_zona = []
    
    if configs["zonificaciones"]:
        print('Actualizo zonificaciones en db')
        zonificaciones = pd.DataFrame([])
        for n in range(0, 5):
            try:
                file_zona = configs["zonificaciones"][f"geo{n+1}"]
                var_zona = configs["zonificaciones"][f"var{n+1}"]
    
                try:
                    matriz_order = configs["zonificaciones"][f"orden{n+1}"]
                except KeyError:
                    matriz_order = ""
    
                if matriz_order is None:
                    matriz_order = ""
    
                if file_zona:
                    db_path = os.path.join("data", "data_ciudad", file_zona)
                    if os.path.exists(db_path):
                        zonif = gpd.read_file(db_path)
                        zonif = zonif[[var_zona, 'geometry']]
                        zonif.columns = ['id', 'geometry']
                        zonif['zona'] = var_zona
                        zonif = zonif[['zona', 'id', 'geometry']]
                        
                        if len(matriz_order) > 0:
                            order = pd.DataFrame(matriz_order, columns=['id']).reset_index().rename(columns={'index':'orden'})
                            zonif = zonif.merge(order, how='left')    
                        else:
                            zonif['orden'] = 0

                        zonif['id'] = zonif['id'].astype(str)    
                        zonif.loc[zonif['id'].str[-2:] == '.0', 'id'] = zonif.loc[zonif['id'].str[-2:] == '.0', 'id'].str[:-2]    

                        zonificaciones = pd.concat([zonificaciones, zonif], ignore_index=True)    
    
            except KeyError:
                pass
    
        db_path = os.path.join("resultados", f'{alias}Zona_voi.geojson')
        if os.path.exists(db_path):
            zonif = gpd.read_file(db_path)
            zonif = zonif[['Zona_voi', 'geometry']]
            zonif.columns = ['id', 'geometry']
            zonif['zona'] = 'Zona_voi'
            zonif['orden'] = 0
            zonif = zonif[['zona', 'orden', 'id', 'geometry']]        
            zonificaciones = pd.concat([zonificaciones, zonif], ignore_index=True)
        
        if len(zonificaciones) > 0:
            conn_dash = iniciar_conexion_db(tipo='dash')
            conn_insumos = iniciar_conexion_db(tipo='insumos')
            zonificaciones = zonificaciones.dissolve(['zona', 'id', 'orden'], as_index=False)
            zonificaciones['wkt'] = zonificaciones.geometry.to_wkt()
            zonificaciones = zonificaciones.drop(['geometry'], axis=1)
            zonificaciones = zonificaciones.sort_values(['zona', 'orden', 'id']).reset_index(drop=True)
    
            zonificaciones.to_sql("zonificaciones",
                                 conn_insumos, if_exists="replace", index=False,)
            zonificaciones.to_sql("zonificaciones",
                         conn_dash, if_exists="replace", index=False,)
            
            zonificaciones = levanto_tabla_sql('zonificaciones', 'insumos')
            zonificaciones = zonificaciones[zonificaciones.zona!='Zona_voi']
            zonificaciones['all'] = 1
            zonificaciones = zonificaciones[['all', 'geometry']].dissolve(by='all')
            
            zonas = create_zones_table()
            
            zonas = zonas.drop(['fex'], axis=1)
            
            # Añadir geometrías al DataFrame
            zonas['geometry'] = zonas['h3'].apply(h3_to_polygon)
            
            # Convertir a GeoDataFrame
            zonas = gpd.GeoDataFrame(zonas, geometry='geometry', crs='EPSG:4326')
            
            # Mostrar el GeoDataFrame
            
            zonas = zonas[zonas.geometry.within(zonificaciones.geometry.unary_union)]
            zonas = zonas.drop(['geometry'], axis=1)
            guardar_tabla_sql(zonas, 'equivalencia_zonas', 'dash')
            
            # Agrego res_6 y res_8 en zonificaciones
            zonificaciones = levanto_tabla_sql('zonificaciones', 'insumos')
            zonificaciones = zonificaciones[~(zonificaciones.zona.isin(['res_6', 'res_8']))]

            res_6 = generate_h3_hexagons_within_polygon(zonificaciones, 6)    
            res_8 = upscale_h3_resolution(res_6, 8)
            res_6['zona'] = 'res_6'
            res_6['orden'] = 0
            res_6 = res_6.rename(columns={'h3_index':'id'})
            res_6 = res_6[['zona', 'id', 'orden','geometry']]
            res_8['zona'] = 'res_8'
            res_8['orden'] = 0
            res_8 = res_8.rename(columns={'h3_index':'id'})
            res_8 = res_8[['zona', 'id', 'orden','geometry']]
            zonificaciones = pd.concat([zonificaciones, res_6, res_8], ignore_index=True)

            zonificaciones['wkt'] = zonificaciones.geometry.to_wkt()
            zonificaciones = zonificaciones.drop(['geometry'], axis=1)
            zonificaciones = zonificaciones.sort_values(['zona', 'orden', 'id']).reset_index(drop=True)

            zonificaciones.to_sql("zonificaciones",
                     conn_insumos, if_exists="replace", index=False,)
            zonificaciones.to_sql("zonificaciones",
                         conn_dash, if_exists="replace", index=False,)
            
            conn_insumos.close()
            conn_dash.close()
            
    if configs['poligonos']:
        
        poly_file = configs['poligonos']

        db_path = os.path.join("data", "data_ciudad", poly_file)

        if os.path.exists(db_path):
            poly = gpd.read_file(db_path)
            conn_dash = iniciar_conexion_db(tipo='dash')
            conn_insumos = iniciar_conexion_db(tipo='insumos')
            poly['wkt'] = poly.geometry.to_wkt()
            poly = poly.drop(['geometry'], axis=1)
        
            poly.to_sql("poligonos",
                                 conn_insumos, if_exists="replace", index=False,)
            poly.to_sql("poligonos",
                         conn_dash, if_exists="replace", index=False,)
        
        
            conn_insumos.close()
            conn_dash.close()
        

def create_zones_table():
    """
    This function takes orign geo data from etapas and geoms from zones
    in the config file and produces a table with the corresponding zone
    for each h3 with data in etapas
    """

    conn_insumos = iniciar_conexion_db(tipo="insumos")
    conn_data = iniciar_conexion_db(tipo="data")

    # leer origenes de la tabla etapas
    etapas = pd.read_sql_query(
        """
        SELECT id, h3_o as h3, latitud, longitud, factor_expansion_linea
        from etapas
        where od_validado == 1
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
    except DatabaseError:
        print("No existe la tabla zonas en la base")
        zonas_ant = pd.DataFrame([])

    # A partir de los origenes de etapas, crea una nueva tabla zonas
    # con el promedio de la latitud y longitud
    zonas = (
        etapas.groupby(
            "h3",
            as_index=False,
        )
        .agg({"factor_expansion_linea": "sum", "latitud": "mean", "longitud": "mean"})
        .rename(columns={"factor_expansion_linea": "fex"})
    )
    # TODO: redo how geoms are created here
    zonas = pd.concat([zonas, zonas_ant], ignore_index=True)
    agg_dict = {
        "fex": "mean",
        "latitud": "mean",
        "longitud": "mean",
    }
    zonas = zonas.groupby(
        "h3",
        as_index=False,
    ).agg(agg_dict)

    # Crea la latitud y la longitud en base al h3
    zonas["origin"] = zonas["h3"].apply(h3togeo)
    zonas["lon"] = zonas["origin"].apply(bring_latlon, latlon="lon")
    zonas["lat"] = zonas["origin"].apply(bring_latlon, latlon="lat")

    zonas = gpd.GeoDataFrame(
        zonas,
        geometry=gpd.points_from_xy(zonas["lon"], zonas["lat"]),
        crs=4326,
    )
    zonas = zonas.drop(["origin", "lon", "lat"], axis=1)

    # Suma a la tabla las zonificaciones del config
    configs = leer_configs_generales()
    if configs["zonificaciones"]:

        for n in range(0, 5):
            try:
                file_geo = configs["zonificaciones"][f"geo{n+1}"]
                var_geo = configs["zonificaciones"][f"var{n+1}"]
                zn_path = os.path.join("data", "data_ciudad", file_geo)
                zn = gpd.read_file(zn_path)
                zn = zn.drop_duplicates()
                zn = zn[[var_geo, "geometry"]]
                if zn[var_geo].dtype != 'O':
                    zn.loc[zn[var_geo].notna(), var_geo] = zn.loc[zn[var_geo].notna(), var_geo].astype(int).astype(str)                   

                zonas = gpd.sjoin(zonas, zn, how="left")
                zonas = zonas.drop(["index_right"], axis=1)

            except (KeyError, TypeError):
                pass

    zonas["res_8"] = zonas.apply(h3_from_row, axis=1, args=(8, "latitud", "longitud"))
    zonas["res_6"] = zonas.apply(h3_from_row, axis=1, args=(6, "latitud", "longitud"))

    zonas = zonas.drop(["geometry"], axis=1)
    zonas.to_sql("zonas", conn_insumos, if_exists="replace", index=False)
    conn_insumos.close()
    print("Graba zonas en sql lite")
    return zonas

@duracion
def create_voronoi_zones(res=8, max_zonas=15, show_map=False):
    """
    This function creates transport zones based on the points in the dataset
    """

    alias = leer_alias()
    conn_insumos = iniciar_conexion_db(tipo="insumos")

    # Leer informacion en tabla zonas
    try:
        zonas = pd.read_sql_query(
            """
            SELECT *
            FROM zonas
            """,
            conn_insumos,
        )
    except DatabaseError:
            print("No existe la tabla zonas en la base")
            zonas = pd.DataFrame([])        

    if len(zonas) > 0:
        # Si existe la columna de zona voronoi la elimina
        if "Zona_voi" in zonas.columns:
            zonas.drop(["Zona_voi"], axis=1, inplace=True)
    
        # agrega datos a un hexagono mas grande
        zonas["h3_r"] = zonas["h3"].apply(h3.h3_to_parent, res=res)
    
        # Computa para ese hexagono el promedio ponderado de latlong
        zonas_for_hexs = zonas.loc[zonas.fex != 0, :]
    
        hexs = zonas_for_hexs.groupby("h3_r", as_index=False).fex.sum()
    
        hexs = hexs.merge(
            zonas_for_hexs.groupby("h3_r")
            .apply(lambda x: np.average(x["longitud"], weights=x["fex"]))
            .reset_index()
            .rename(columns={0: "longitud"}),
            how="left",
        )
    
        hexs = hexs.merge(
            zonas_for_hexs.groupby("h3_r")
            .apply(lambda x: np.average(x["latitud"], weights=x["fex"]))
            .reset_index()
            .rename(columns={0: "latitud"}),
            how="left",
        )
    
        hexs = gpd.GeoDataFrame(
            hexs,
            geometry=gpd.points_from_xy(hexs["longitud"], hexs["latitud"]),
            crs=4326,
        )
    
        cant_zonas = len(hexs) + 10
        k_ring = 1
    
        if cant_zonas <= max_zonas:
            hexs2 = hexs.copy()
    
        while cant_zonas > max_zonas:
            # Construye un set de hexagonos aun mas grandes
            hexs2 = hexs.copy()
            hexs2["h3_r2"] = hexs2.h3_r.apply(h3.h3_to_parent, res=res - 1)
            hexs2["geometry"] = hexs2.h3_r2.apply(add_geometry)
            hexs2 = hexs2.sort_values(["h3_r2", "fex"], ascending=[True, False])
            hexs2["orden"] = hexs2.groupby(["h3_r2"]).cumcount()
            hexs2 = hexs2[hexs2.orden == 0]
    
            hexs2 = hexs2.sort_values("fex", ascending=False)
            hexs["cambiado"] = 0
            for i in hexs2.h3_r.tolist():
                vecinos = h3.k_ring(i, k_ring)
                hexs.loc[(hexs.h3_r.isin(vecinos)) & (hexs.cambiado == 0), "h3_r"] = i
                hexs.loc[(hexs.h3_r.isin(vecinos)) & (hexs.cambiado == 0), "cambiado"] = 1
    
            hexs_tmp = hexs.groupby("h3_r", as_index=False).fex.sum()
            hexs_tmp = hexs_tmp.merge(
                hexs[hexs.fex != 0]
                .groupby("h3_r")
                .apply(lambda x: np.average(x["longitud"], weights=x["fex"]))
                .reset_index()
                .rename(columns={0: "longitud"}),
                how="left",
            )
            hexs_tmp = hexs_tmp.merge(
                hexs[hexs.fex != 0]
                .groupby("h3_r")
                .apply(lambda x: np.average(x["latitud"], weights=x["fex"]))
                .reset_index()
                .rename(columns={0: "latitud"}),
                how="left",
            )
            hexs_tmp = gpd.GeoDataFrame(
                hexs_tmp,
                geometry=gpd.points_from_xy(hexs_tmp["longitud"], hexs_tmp["latitud"]),
                crs=4326,
            )
    
            hexs = hexs_tmp.copy()
    
            if cant_zonas == len(hexs):
                k_ring += 1
            else:
                cant_zonas = len(hexs)
    
        voi = create_voronoi(hexs)
        voi = gpd.sjoin(voi, hexs[["fex", "geometry"]], how="left")
        voi = voi.sort_values("fex", ascending=False)
        voi = voi.drop(["Zona", "index_right"], axis=1)
        voi = voi.reset_index(drop=True).reset_index().rename(columns={"index": "Zona_voi"})
        voi["Zona_voi"] = voi["Zona_voi"] + 1
        voi["Zona_voi"] = voi["Zona_voi"].astype(str)
    
        file = os.path.join("data", "data_ciudad", "zona_voi.geojson")
        voi[["Zona_voi", "geometry"]].to_file(file)
    
        zonas = zonas.drop(["h3_r"], axis=1)
        zonas["geometry"] = zonas["h3"].apply(add_geometry)
    
        zonas = gpd.GeoDataFrame(zonas, geometry="geometry", crs=4326)
        zonas["geometry"] = zonas["geometry"].representative_point()
    
        zonas = gpd.sjoin(zonas, voi[["Zona_voi", "geometry"]], how="left")
    
        zonas = zonas.drop(["index_right", "geometry"], axis=1)
        zonas.to_sql("zonas", conn_insumos, if_exists="replace", index=False)
        conn_insumos.close()
        print("Graba zonas en sql lite")
    
        # Plotea geoms de voronoi
        viz.plot_voronoi_zones(voi, hexs, hexs2, show_map, alias)


@duracion
def create_distances_table(use_parallel=False):
    """
    Esta tabla toma los h3 de la tablas de etapas y viajes
    y calcula diferentes distancias para cada par que no tenga
    """

    conn_insumos = iniciar_conexion_db(tipo="insumos")
    conn_data = iniciar_conexion_db(tipo="data")

    print("Verifica viajes sin distancias calculadas")

    q = """
    select distinct h3_o,h3_d
    from viajes
    WHERE h3_d != ''
    union
    select distinct h3_o,h3_d
    from (
            SELECT h3_o,h3_d
            from etapas
            WHERE h3_d != ''
    )
    """

    pares_h3_data = pd.read_sql_query(q, conn_data)

    q = """
    select h3_o,h3_d, 1 as d from distancias
    """
    pares_h3_distancias = pd.read_sql_query(q, conn_insumos)

    # Unir pares od h desde data y desde distancias y quedarse con
    # los que estan en data pero no en distancias
    pares_h3 = pares_h3_data.merge(pares_h3_distancias, how="left")
    pares_h3 = pares_h3.loc[
        (pares_h3.d.isna()) & (pares_h3.h3_o != pares_h3.h3_d), ["h3_o", "h3_d"]
    ]

    if len(pares_h3) > 0:
        pares_h3_norm = normalizo_lat_lon(pares_h3)

        # usa la función osmnx para distancias en caso de error con Pandana
        print(
            "Este proceso puede demorar algunas horas dependiendo del tamaño "
            + " de la ciudad y si se corre por primera vez por lo que en la base"
            + " de insumos no estan estos pares"
        )

        agg2_total = (
            pares_h3_norm.groupby(["h3_o_norm", "h3_d_norm"], as_index=False)
            .size()
            .drop(["size"], axis=1)
        )

        print(f"Hay {len(agg2_total)} nuevos pares od para sumar a tabla distancias")
        print(f"de los {len(pares_h3_data)} originales en la data.")
        print("")
        print("Procesa distancias con Pandana")

        agg2 = compute_distances_osm(
            agg2_total,
            h3_o="h3_o_norm",
            h3_d="h3_d_norm",
            processing="pandana",
            modes=["drive"],
            use_parallel=False,
        )

        if len(agg2) > 0:

            dist1 = agg2.copy()
            dist1["h3_o"] = dist1["h3_o_norm"]
            dist1["h3_d"] = dist1["h3_d_norm"]
            dist2 = agg2.copy()
            dist2["h3_d"] = dist2["h3_o_norm"]
            dist2["h3_o"] = dist2["h3_d_norm"]
            distancias_new = pd.concat([dist1, dist2], ignore_index=True)
            distancias_new = distancias_new.groupby(
                ["h3_o", "h3_d", "h3_o_norm", "h3_d_norm"], as_index=False
            )[["distance_osm_drive", "distance_h3"]].first()

            distancias_new.to_sql(
                "distancias", conn_insumos, if_exists="append", index=False
            )

        else:
            print("Procesa distancias con OSMNX")
            # Determine the size of each chunk (500 rows in this case)
            chunk_size = 25000

            # Get the total number of rows in the DataFrame
            total_rows = len(agg2_total)

            # Loop through the DataFrame in chunks of 500 rows
            for start in range(0, total_rows, chunk_size):
                end = start + chunk_size
                # Select the chunk of 500 rows from the DataFrame
                agg2 = agg2_total.iloc[start:end].copy()
                # Call the process_chunk function with the selected chunk
                print(
                    f"Bajando distancias entre {start} a {end} de {len(agg2_total)} - {str(datetime.now())[:19]}"
                )

                agg2 = compute_distances_osm(
                    agg2,
                    h3_o="h3_o_norm",
                    h3_d="h3_d_norm",
                    processing="osmnx",
                    modes=["drive"],
                    use_parallel=use_parallel,
                )

                dist1 = agg2.copy()
                dist1["h3_o"] = dist1["h3_o_norm"]
                dist1["h3_d"] = dist1["h3_d_norm"]
                dist2 = agg2.copy()
                dist2["h3_d"] = dist2["h3_o_norm"]
                dist2["h3_o"] = dist2["h3_d_norm"]
                distancias_new = pd.concat([dist1, dist2], ignore_index=True)
                distancias_new = distancias_new.groupby(
                    ["h3_o", "h3_d", "h3_o_norm", "h3_d_norm"], as_index=False
                )[["distance_osm_drive", "distance_h3"]].first()

                distancias_new.to_sql(
                    "distancias", conn_insumos, if_exists="append", index=False
                )

                conn_insumos.close()
                conn_insumos = iniciar_conexion_db(tipo="insumos")

        conn_insumos.close()
        conn_data.close()


def compute_distances_osmx(df, mode, use_parallel):
    """
    Takes a dataframe with pairs of h3 with origins and destinations
    and computes distances between those pairs using OSMNX.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame representing a chunk with OD pairs
        with h3 indexes
    modes: list
        list of modes to compute distances for. Must be a valid
        network_type parameter for either osmnx graph_from_bbox
        or pandana pdna_network_from_bbox
    use_parallel: bool
        use parallel processing when computin omsnx distances

    Returns
    -------
    pandas.DataFrame
        DataFrame containing od pairs with distances
    """
    print("Computando distancias entre pares OD con OSMNX")
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

    G = ox.graph_from_bbox(ymax, ymin, xmax, xmin, network_type=mode)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    nodes_from = ox.distance.nearest_nodes(
        G, df["lon_o_tmp"].values, df["lat_o_tmp"].values, return_dist=True
    )

    nodes_to = ox.distance.nearest_nodes(
        G, df["lon_d_tmp"].values, df["lat_d_tmp"].values, return_dist=True
    )
    nodes_from = nodes_from[0]
    nodes_to = nodes_to[0]

    if use_parallel:
        results = run_network_distance_parallel(mode, G, nodes_from, nodes_to)
        df[f"distance_osm_{mode}"] = results

    else:
        df = run_network_distance_not_parallel(df, mode, G, nodes_from, nodes_to)
    return df


def compute_distances_pandana(df, mode):
    """
    Takes a dataframe with pairs of h3 with origins and destinations
    and computes distances between those pairs using pandana.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame representing a chunk with OD pairs
        with h3 indexes
    modes: list
        list of modes to compute distances for. Must be a valid
        network_type parameter for either osmnx graph_from_bbox
        or pandana pdna_network_from_bbox

    Returns
    -------
    pandas.DataFrame
        DataFrame containing od pairs with distances
    """

    print("Computando distancias entre pares OD con Pandana")
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

    network = osm_pandana.pdna_network_from_bbox(
        ymin, xmin, ymax, xmax, network_type=mode
    )

    df["node_from"] = network.get_node_ids(df["lon_o_tmp"], df["lat_o_tmp"]).values
    df["node_to"] = network.get_node_ids(df["lon_d_tmp"], df["lat_d_tmp"]).values
    df[f"distance_osm_{mode}"] = network.shortest_path_lengths(
        df["node_to"].values, df["node_from"].values
    )
    return df


def compute_distances_osm(
    df,
    h3_o="",
    h3_d="",
    processing="pandana",
    modes=["drive", "walk"],
    use_parallel=False,
):
    """
    Takes a dataframe with pairs of h3 with origins and destinations
    and computes distances between those pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame representing a chunk with OD pairs
        with h3 indexes
    h3_o: str (h3Index)
        origin h3 index
    h3_d: str (h3Index)
        destination h3 index
    processing: str
        processing method, either use 'osmnx' or 'pandana'
    modes: list
        list of modes to compute distances for. Must be a valid
        network_type parameter for either osmnx graph_from_bbox
        or pandana pdna_network_from_bbox
    use_parallel: bool
        use parallel processing when computin omsnx distances

    Returns
    -------
    pandas.DataFrame
        DataFrame containing od pairs with distances
    """

    cols = df.columns.tolist()

    df["origin"] = df[h3_o].apply(h3togeo)
    df["lon_o_tmp"] = df["origin"].apply(bring_latlon, latlon="lon")
    df["lat_o_tmp"] = df["origin"].apply(bring_latlon, latlon="lat")

    df["destination"] = df[h3_d].apply(h3togeo)
    df["lon_d_tmp"] = df["destination"].apply(bring_latlon, latlon="lon")
    df["lat_d_tmp"] = df["destination"].apply(bring_latlon, latlon="lat")

    var_distances = []

    for mode in modes:

        if processing == "pandana":
            max_retries = 10
            retries = 0
            while retries < max_retries:
                try:
                    # computing distances with pandana
                    df = compute_distances_pandana(df=df, mode=mode)
                    break  
                except Exception as e:  # Captura excepciones específicas si es necesario
                    retries += 1
                    print(f"Intento {retries} con Pandana falló: {e}")
                    if retries == max_retries:
                        print("No se pudo computar distancias con Pandana después de varios intentos. Recurriendo a OSMNX.")

                        library_name = "Pandana"
                        version = get_library_version(library_name)
                        if version:
                            print(f"{library_name} version {version} is installed.")
                        else:
                            print(f"{library_name} is not installed.")

                        library_name = "OSMnet"
                        version = get_library_version(library_name)
                        if version:
                            print(f"{library_name} version {version} is installed.")
                        else:
                            print(f"{library_name} is not installed.")
                        return pd.DataFrame([])
                        df = compute_distances_osmx(df=df, mode=mode, use_parallel=use_parallel)

    var_distances += [f"distance_osm_{mode}"]
    df[f"distance_osm_{mode}"] = (df[f"distance_osm_{mode}"] / 1000).round(2)

    condition = ("distance_osm_drive" in df.columns) & (
        "distance_osm_walk" in df.columns
    )

    if condition:
        mask = (df.distance_osm_drive * 1.3) < df.distance_osm_walk
        df.loc[mask, "distance_osm_walk"] = df.loc[mask, "distance_osm_drive"]

    if "distance_osm_drive" in df.columns:
        df.loc[df.distance_osm_drive > 2000, "distance_osm_drive"] = np.nan
    if "distance_osm_walk" in df.columns:
        df.loc[df.distance_osm_walk > 2000, "distance_osm_walk"] = np.nan

    df = df[cols + var_distances].copy()

    # get distance between h3 cells
    configs = leer_configs_generales()
    h3_res = configs["resolucion_h3"]
    distance_between_hex = h3.edge_length(resolution=h3_res, unit="km")
    distance_between_hex = distance_between_hex * 2

    if (len(h3_o) > 0) & (len(h3_d) > 0):
        df["distance_h3"] = df[[h3_o, h3_d]].apply(
            h3dist,
            axis=1,
            distancia_entre_hex=distance_between_hex,
            h3_o=h3_o,
            h3_d=h3_d,
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
        print(f"{date_str} processing {int(idmatrix)} / ")

    try:
        ret = nx.shortest_path_length(G, node_from, node_to, weight="length")
    except NetworkXNoPath:
        ret = np.nan
    return ret


def run_network_distance_parallel(mode, G, nodes_from, nodes_to):
    """
    This function runs the network distance in parallel
    """
    n_cores = max(int(multiprocessing.cpu_count() - 1), 1)
    n = len(nodes_from)
    chunksize = int(sqrt(n) * 10)

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(
            partial(get_network_distance_osmnx, G=G),
            zip(nodes_from, nodes_to),
            chunksize=chunksize,
        )

    return results


def get_network_distance_osmnx(par, G, *args, **kwargs):
    node_from, node_to = par
    try:
        out = nx.shortest_path_length(G, node_from, node_to, weight="length")
    except NetworkXNoPath:
        out = np.nan
    return out

def generate_h3_hexagons_within_polygon(geo_df, resolution):
    """
    Genera un GeoDataFrame con hexágonos H3 dentro del polígono dado.

    Parameters:
        geo_df (GeoDataFrame): GeoDataFrame que contiene el polígono de entrada.
        resolution (int): Resolución H3 deseada (0-15).

    Returns:
        GeoDataFrame: Nuevo GeoDataFrame con hexágonos H3 dentro del polígono.
    """
    # Asegurarse de que el GeoDataFrame usa el CRS correcto (WGS84 - EPSG:4326)
    geo_df = geo_df.to_crs("EPSG:4326")

    # Convertir la geometría del GeoDataFrame en un único polígono o multipolígono
    polygon = geo_df.geometry.iloc[0]

    # Asegurar que sea un Polygon o MultiPolygon
    if isinstance(polygon, MultiPolygon):
        # Combinar en un único polígono si hay varios
        polygon = max(polygon.geoms, key=lambda p: p.area)  # Seleccionar el más grande
    elif not isinstance(polygon, Polygon):
        raise ValueError("La geometría proporcionada debe ser un Polygon o MultiPolygon.")

    # Obtener los hexágonos que cubren el polígono
    hexagons = list(h3.polyfill_geojson(polygon.__geo_interface__, resolution))

    # Crear un GeoDataFrame con los hexágonos dentro del polígono
    hexagon_geometries = []
    for h in hexagons:
        hex_polygon = Polygon(h3.h3_to_geo_boundary(h, geo_json=True))
        if polygon.contains(hex_polygon.centroid):
            hexagon_geometries.append(hex_polygon)

    # Crear un GeoDataFrame con los hexágonos seleccionados
    hexagon_gdf = gpd.GeoDataFrame({"h3_index": [h for h in hexagons if Polygon(h3.h3_to_geo_boundary(h, geo_json=True)).centroid.within(polygon)]},
                                   geometry=hexagon_geometries, crs="EPSG:4326")

    return hexagon_gdf

def upscale_h3_resolution(hexagon_gdf, target_resolution):
    """
    Aumenta la resolución de hexágonos H3 en un GeoDataFrame.

    Parameters:
        hexagon_gdf (GeoDataFrame): GeoDataFrame con hexágonos H3 en una resolución inicial.
        target_resolution (int): Resolución H3 objetivo.

    Returns:
        GeoDataFrame: GeoDataFrame con los hexágonos en la resolución objetivo.
    """
    # Validar que la resolución objetivo sea mayor que la resolución actual
    current_resolution = h3.h3_get_resolution(hexagon_gdf["h3_index"].iloc[0])
    if target_resolution <= current_resolution:
        raise ValueError("La resolución objetivo debe ser mayor que la resolución actual.")

    # Generar los hijos para cada hexágono
    hexagon_children = []
    for h3_index in hexagon_gdf["h3_index"]:
        children = h3.h3_to_children(h3_index, target_resolution)
        for child in children:
            hex_polygon = Polygon(h3.h3_to_geo_boundary(child, geo_json=True))
            hexagon_children.append({"h3_index": child, "geometry": hex_polygon})

    # Crear el nuevo GeoDataFrame
    upscale_gdf = gpd.GeoDataFrame(hexagon_children, crs=hexagon_gdf.crs)

    return upscale_gdf