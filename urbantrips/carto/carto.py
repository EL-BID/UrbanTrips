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
from shapely.geometry import Point, LineString, MultiPolygon, Polygon, shape
from networkx import NetworkXNoPath
from pandana.loaders import osm as osm_pandana
from urbantrips.geo.geo import (
    get_stop_hex_ring,
    h3togeo,
    h3_from_row,
    add_geometry,
    normalizo_lat_lon,
    h3dist,
    bring_latlon,
    h3_to_polygon,
)
import warnings

warnings.filterwarnings(
    "ignore",
    message="Unsigned integer: shortest path distance is trying to be calculated",
    category=UserWarning,
    module="pandana.network",
)
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    leer_alias,
    levanto_tabla_sql,
    guardar_tabla_sql,
)

import subprocess
from math import floor
from shapely import wkt


def create_route_section_ids(n_sections):
    step = 1 / n_sections
    sections = np.arange(0, 1 + step, step)
    section_ids = pd.Series(map(floor_rounding, sections))
    # n sections like 6 returns steps with max setion > 1
    section_ids = section_ids[section_ids <= 1]

    return section_ids


def floor_rounding(num):
    """
    Rounds a number to the floor at 3 digits to use as route section id
    """
    return floor(num * 1000) / 1000


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
    print("RING SIZE ", ring_size)

    conn_data = iniciar_conexion_db(tipo="data")

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

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

    # filtrar paradas solo los que estan en recorridos h3
    q = """  
    select distinct mr.id_linea as id_linea_agg, obgh.h3 as parada
    from official_branches_geoms_h3 obgh 
    inner join metadata_ramales mr 
    ON obgh.id_ramal = mr.id_ramal
    """
    h3_recorridos = pd.read_sql(q, conn_insumos)
    print(len(h3_recorridos.id_linea_agg.unique()))

    if len(h3_recorridos) > 0:
        h3_recorridos["parada_en_recorridos"] = True
        paradas_etapas["id_linea_recorridos"] = paradas_etapas["id_linea_agg"].isin(
            h3_recorridos["id_linea_agg"].unique()
        )
        print("PARADAS EN ETAPAS 1")
        print(len(paradas_etapas.id_linea_agg.unique()))

        print("LINEAS EN RECORRIDOS", paradas_etapas["id_linea_recorridos"].sum())

        paradas_etapas = paradas_etapas.merge(
            h3_recorridos, on=["id_linea_agg", "parada"], how="left"
        )
        print("PARADAS EN ETAPAS 2")
        print(len(paradas_etapas.id_linea_agg.unique()))

        paradas_etapas["parada_en_recorridos"] = (
            paradas_etapas["parada_en_recorridos"].fillna(False).astype(bool)
        )

        paradas_etapas["borrar"] = (paradas_etapas.id_linea_recorridos) & (
            ~paradas_etapas.parada_en_recorridos
        )

        print(
            paradas_etapas.drop(columns=["id_linea_agg", "parada"]).sample(
                30, random_state=1
            )
        )

        paradas_pre = len(paradas_etapas)
        print("Paradas antes de eliminar por recorridos", paradas_pre)
        paradas_etapas = paradas_etapas.loc[
            ~paradas_etapas.borrar, ["id_linea_agg", "parada"]
        ]
        print(f"Eliminadas {paradas_pre - len(paradas_etapas)} paradas sin recorridos")

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


def guardo_zonificaciones():
    """
    Processes and updates zoning information in the database based on configuration
    files and geospatial data.
    This function performs the following tasks:
    - Reads general configuration settings to determine zoning files and variables.
    - Loads and processes multiple zoning GeoJSON files, extracting relevant columns
      and standardizing their format.
    - Optionally merges ordering information into the zoning data.
    - Cleans and standardizes zone identifiers.
    - Dissolves certain zones to create a unified polygon, then generates H3 hexagon
      grids at resolutions 6 and 7 within this polygon, adding them as new zoning
      layers.
    - Creates a zones equivalence table, associates geometries, and filters zones
      within the unified zoning geometry.
    - Saves the processed zoning and equivalence tables to the "dash" and "insumos"
      databases.
    - Optionally processes and saves additional polygon data if specified in the
      configuration.
    Returns:
        None
    """

    configs = leer_configs_generales(autogenerado=False)
    alias = configs.get("alias_db", "")

    # Lee las 5 posibles configuraciones de zonificaciones
    if configs["zonificaciones"]:
        print("Crear zonificaciones en db")
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

                # Si existe un archivo para esa zona, lo lee
                if file_zona:
                    db_path = os.path.join("data", "data_ciudad", file_zona)
                    if os.path.exists(db_path):
                        zonif = gpd.read_file(db_path)
                        zonif = zonif[[var_zona, "geometry"]]
                        zonif.columns = ["id", "geometry"]
                        zonif["zona"] = var_zona
                        zonif = zonif[["zona", "id", "geometry"]]

                        if len(matriz_order) > 0:
                            order = (
                                pd.DataFrame(matriz_order, columns=["id"])
                                .reset_index()
                                .rename(columns={"index": "orden"})
                            )
                            zonif = zonif.merge(order, how="left")
                        else:
                            zonif["orden"] = 0

                        zonif["id"] = zonif["id"].astype(str)
                        zonif.loc[zonif["id"].str[-2:] == ".0", "id"] = zonif.loc[
                            zonif["id"].str[-2:] == ".0", "id"
                        ].str[:-2]

                        zonificaciones = pd.concat(
                            [zonificaciones, zonif], ignore_index=True
                        )

            except KeyError:
                pass

        if len(zonificaciones) > 0:
            zonificaciones["orden"] = zonificaciones["orden"].fillna(0)
            zonificaciones["zona"] = zonificaciones["zona"].fillna("")
            zonificaciones["id"] = zonificaciones["id"].fillna("")
            zonificaciones = zonificaciones.dissolve(
                ["zona", "id", "orden"], as_index=False
            )

            crs_val = configs["epsg_m"]
            crs_actual = zonificaciones.crs

            zonificaciones_disolved = zonificaciones[
                ~(zonificaciones.zona.isin(["res_6", "res_7", "res_8"]))
            ].copy()
            zonificaciones_disolved["all"] = 1
            zonificaciones_disolved = (
                zonificaciones_disolved[["all", "geometry"]]
                .dissolve(by="all")
                .to_crs(crs_val)
                .buffer(2000)
                .to_crs(crs_actual)
            )

            # Agrego res_6 y res_8 en zonificaciones
            res_6 = generate_h3_hexagons_within_polygon(
                zonificaciones_disolved, 6, crs_val
            )
            res_6["zona"] = "res_6"
            res_6["orden"] = 0
            res_6 = res_6.rename(columns={"h3_index": "id"})
            res_6 = res_6[["zona", "id", "orden", "geometry"]]
            zonificaciones = pd.concat([zonificaciones, res_6], ignore_index=True)

            res_7 = generate_h3_hexagons_within_polygon(
                zonificaciones_disolved, 7, crs_val
            )
            res_7["zona"] = "res_7"
            res_7["orden"] = 0
            res_7 = res_7.rename(columns={"h3_index": "id"})
            res_7 = res_7[["zona", "id", "orden", "geometry"]]
            zonificaciones = pd.concat([zonificaciones, res_7], ignore_index=True)

            # Crear una tabla de equivalencias para cada h3 de urbantrips a las zonificaciones
            full_area_res_urbantrips = generate_h3_hexagons_within_polygon(
                zonificaciones_disolved, configs.get("resolucion_h3"), crs_val
            )
            full_area_res_urbantrips.geometry = (
                full_area_res_urbantrips.geometry.representative_point()
            )

            equivalencias_zonas = gpd.sjoin(
                full_area_res_urbantrips,
                zonificaciones,
                how="inner",
                predicate="intersects",
            )
            equivalencias_zonas["latitud"] = equivalencias_zonas.geometry.y
            equivalencias_zonas["longitud"] = equivalencias_zonas.geometry.x
            equivalencias_zonas = (
                equivalencias_zonas.reindex(
                    columns=["h3_index", "latitud", "longitud", "zona", "id"]
                )
                .pivot_table(
                    index=["h3_index", "latitud", "longitud"],
                    columns="zona",
                    values="id",
                    aggfunc="first",
                )
                .reset_index()
                .rename(columns={"h3_index": "h3"})
            )

            # Guardo zonificaciones

            guardar_tabla_sql(
                zonificaciones, "zonificaciones", "insumos", modo="replace"
            )
            guardar_tabla_sql(
                equivalencias_zonas, "equivalencias_zonas", "insumos", modo="replace"
            )

    if configs["poligonos"]:

        poly_file = configs["poligonos"]

        db_path = os.path.join("data", "data_ciudad", poly_file)
        poligonos_db = levanto_tabla_sql("poligonos", "insumos")
        print(poligonos_db.head(2))
        if os.path.exists(db_path):
            poly = gpd.read_file(db_path)

            if len(poligonos_db) > 0:
                poligonos_db = poligonos_db.loc[
                    poligonos_db["id"] == "estimacion de demanda dibujada",
                ]
                # poligonos_db["geometry"] = poligonos_db["wkt"].apply(wkt.loads)
                # poligonos_db = poligonos_db.reindex(columns=["id", "tipo", "geometry"])
                poly = pd.concat([poly, poligonos_db], ignore_index=True)

            guardar_tabla_sql(poly, "poligonos", "insumos", modo="replace")


@duracion
def create_distances_table(use_parallel=False):
    """
    Esta tabla toma los h3 de la tablas de etapas
    y calcula diferentes distancias para cada par que no tenga
    """

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)
    conn_data = iniciar_conexion_db(tipo="data")

    print("Verifica viajes sin distancias calculadas")

    q = """
    select distinct h3_o,h3_d
    from etapas
    WHERE h3_d != ''
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
                except (
                    Exception
                ) as e:  # Captura excepciones específicas si es necesario
                    retries += 1
                    print(f"Intento {retries} con Pandana falló: {e}")
                    if retries == max_retries:
                        print(
                            "No se pudo computar distancias con Pandana después de varios intentos. Recurriendo a OSMNX."
                        )

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
                        df = compute_distances_osmx(
                            df=df, mode=mode, use_parallel=use_parallel
                        )

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
    resolution = h3.get_resolution(df[h3_o].iloc[0])
    distance_between_hex = h3.average_hexagon_edge_length(resolution, unit="km")
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


# Convert geometry to H3 indices
def get_h3_indices_in_geometry(geometry, resolution):
    poly = h3.geo_to_h3shape(geometry)
    h3_indices = list(h3.h3shape_to_cells(poly, res=resolution))

    return h3_indices


def generate_h3_hexagons_within_polygon(geo_df, resolution, crs_val):
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
        raise ValueError(
            "La geometría proporcionada debe ser un Polygon o MultiPolygon."
        )

    # Obtener los hexágonos que cubren el polígono
    hexagons = pd.Series(get_h3_indices_in_geometry(polygon, resolution))
    hexagons_geoms = gpd.GeoSeries([h3_to_polygon(h) for h in hexagons], crs=4326)

    # Filtrar los hexágonos que están dentro del polígono
    mask = hexagons_geoms.representative_point().within(polygon).values
    hexagons = hexagons[mask]
    hexagons_geoms = hexagons_geoms[mask]

    # Crear un GeoDataFrame con los hexágonos seleccionados
    hexagon_gdf = gpd.GeoDataFrame(
        {"h3_index": hexagons},
        geometry=hexagons_geoms,
        crs="EPSG:4326",
    )

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
    current_resolution = h3.get_resolution(hexagon_gdf["h3_index"].iloc[0])
    print(
        f"Resolución actual: {current_resolution}, Resolución objetivo: {target_resolution}"
    )
    if target_resolution <= current_resolution:
        raise ValueError(
            "La resolución objetivo debe ser mayor que la resolución actual."
        )

    # Generar los hijos para cada hexágono
    hexagon_children = []
    for h3_index in hexagon_gdf["h3_index"]:
        children = h3.cell_to_children(h3_index, target_resolution)

        for child in children:
            hex_polygon = h3_to_polygon(child)
            hexagon_children.append({"h3_index": child, "geometry": hex_polygon})

    # Crear el nuevo GeoDataFrame
    upscale_gdf = gpd.GeoDataFrame(hexagon_children, crs=hexagon_gdf.crs)

    return upscale_gdf


def from_linestring_to_h3(linestring, h3_res=8):
    """
    This function takes a shapely linestring and
    returns all h3 hecgrid cells that intersect that linestring
    """
    linestring_buffer = linestring.buffer(0.002)
    linestring_h3 = get_h3_indices_in_geometry(linestring_buffer, 10)
    linestring_h3 = {h3.cell_to_parent(h, h3_res) for h in linestring_h3}
    return pd.Series(list(linestring_h3)).drop_duplicates()


def create_coarse_h3_from_line(
    linestring: LineString, h3_res: int, route_id: int
) -> dict:

    # Reference to coarser H3 for those lines
    linestring_h3 = from_linestring_to_h3(linestring, h3_res=h3_res)

    # Creeate geodataframes with hex geoms and index and LRS
    gdf = gpd.GeoDataFrame(
        {"h3": linestring_h3}, geometry=linestring_h3.map(add_geometry), crs=4326
    )
    gdf["route_id"] = route_id

    # Create LRS for each hex index
    gdf["h3_lrs"] = [
        floor_rounding(linestring.project(Point(p[::-1]), True))
        for p in gdf.h3.map(h3.cell_to_latlng)
    ]

    # Create section ids for each line
    df_section_ids_LRS = create_route_section_ids(len(gdf))

    # Create cut points for each section based on H3 LRS
    df_section_ids_LRS_cut = df_section_ids_LRS.copy().drop_duplicates()
    df_section_ids_LRS_cut.loc[0] = -0.001

    # Use cut points to come up with a unique integer id
    df_section_ids = list(range(1, len(df_section_ids_LRS_cut)))

    gdf["section_id"] = pd.cut(
        gdf.h3_lrs, bins=df_section_ids_LRS_cut, labels=df_section_ids, right=True
    )

    # ESTO REEMPLAZA PARA ATRAS
    gdf = gdf.sort_values("h3_lrs")
    gdf["section_id"] = range(len(gdf))

    return gdf
