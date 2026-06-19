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
    modos_con_ramal,
    id_ramal_efectivo,
    RAMAL_SENTINEL,
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
    Actualiza la matriz de validacion de paradas (matriz_validacion) combinando
    dos fuentes de paradas:
      - transacciones (etapas.h3_o), descartando pares con una sola ocurrencia
      - puntos GPS (gps.h3), descartando hexagonos con muy pocos puntos (outliers
        de densidad: glitches de GPS)
    Aplica un filtro por buffer alrededor del corredor real (footprint de puntos
    GPS validos; si no hay GPS, el recorrido oficial como fallback) descartando las
    paradas que caen muy por fuera, y construye el area de influencia (anillo H3 de
    tamano ring_size) para las paradas nuevas. Es append-only (incremental).

    El uso de ramal se decide por modo (modo_valida_ramal). Para los modos que
    validan por ramal construye por (id_linea_agg, id_ramal); para los que no,
    colapsa los ramales por (id_linea_agg) y deja id_ramal en NULL. En memoria
    los modos sin ramal usan un centinela (RAMAL_SENTINEL) para que el merge por
    [id_linea_agg, id_ramal] funcione uniforme; se persiste NULL.

    Parametros leidos de configuraciones_generales.yaml:
      - resolucion_h3
      - tolerancia_validacion_recorrido (metros, default 600): buffer del filtro
      - frac_mediana_gps (default 0.15): umbral de outliers GPS como fraccion de la
        mediana de puntos por hexagono
      - lineas_contienen_ramales, modo_valida_ramal, nombre_archivo_gps
    """
    configs = leer_configs_generales(autogenerado=False)
    alias_insumos = configs.get("alias_db", "")
    modos_ramal = modos_con_ramal(configs)
    resolucion_h3 = configs["resolucion_h3"]
    frac_mediana_gps = configs.get("frac_mediana_gps", 0.25)
    tol_filtro = configs.get("tolerancia_validacion_recorrido", 600)

    # Buffer del filtro de recorrido: mas chico que el area de influencia.
    lado_m = h3.average_hexagon_edge_length(res=resolucion_h3, unit="m")
    ring_filtro = max(1, round(tol_filtro / (lado_m * 2)))

    conn_data = iniciar_conexion_db(tipo="data")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    # Key fija; id_ramal lleva el valor efectivo (real para modos con ramal,
    # RAMAL_SENTINEL para los que no). Se convierte a NULL al persistir.
    key = ["id_linea_agg", "id_ramal"]

    # --- Migracion: agregar columna id_ramal a matriz_validacion si falta ---
    cols = [r[1] for r in conn_insumos.execute(
        "PRAGMA table_info(matriz_validacion)").fetchall()]
    if "id_ramal" not in cols:
        conn_insumos.execute(
            "ALTER TABLE matriz_validacion ADD COLUMN id_ramal int")
        conn_insumos.commit()

    metadata_lineas = pd.read_sql_query(
        "SELECT id_linea, id_linea_agg, modo FROM metadata_lineas", conn_insumos)

    # --- Fuente A: paradas desde transacciones (etapas) ---
    paradas_etapas = pd.read_sql(
        "select id_linea, id_ramal, h3_o as parada, count(*) as n "
        "from etapas group by id_linea, id_ramal, h3_o", conn_data)
    paradas_etapas = paradas_etapas.merge(
        metadata_lineas, how="left", on="id_linea")
    paradas_etapas["id_ramal"] = id_ramal_efectivo(
        paradas_etapas["modo"], paradas_etapas["id_ramal"], modos_ramal)
    paradas_etapas = paradas_etapas.drop(columns=["id_linea", "modo"])
    # Re-sumar el conteo por la clave efectiva (al colapsar ramales de un modo sin
    # ramal hay que sumar sus n). Filtro >1: descartar paradas de una sola obs.
    paradas_etapas = paradas_etapas.groupby(key + ["parada"], as_index=False)["n"].sum()
    paradas_etapas = paradas_etapas[paradas_etapas["n"] > 1].drop(columns=["n"])

    # --- Fuente B: paradas desde GPS, con filtro de outliers por densidad ---
    # Se detecta GPS por presencia de datos en la tabla (mas robusto que el flag de
    # config nombre_archivo_gps, que puede quedar en None aunque la tabla este poblada).
    # gps no tiene columna modo: se trae del merge con metadata_lineas.
    gps = levanto_tabla_sql(
        "gps", "data",
        query="select id_linea, id_ramal, h3 as parada, count(*) as n_pts "
              "from gps where h3 is not null group by id_linea, id_ramal, h3",
    )
    usa_gps = len(gps) > 0
    if usa_gps:
        gps = gps.merge(metadata_lineas, how="left", on="id_linea")
        gps["id_ramal"] = id_ramal_efectivo(
            gps["modo"], gps["id_ramal"], modos_ramal)
        gps = gps.drop(columns=["id_linea", "modo"])
        # Sumar n_pts por la clave efectiva + parada: colapsa los ramales de un modo
        # sin ramal para que el conteo del hexagono sea el total antes del filtro.
        gps = gps.groupby(key + ["parada"], as_index=False)["n_pts"].sum()
        mediana = gps.groupby(key)["n_pts"].transform("median")
        umbral = np.maximum(2, np.round(mediana * frac_mediana_gps))
        gps_validos = gps[gps["n_pts"] >= umbral].copy()
        gps_validos = gps_validos[key + ["parada"]]
        print('frac_mediana_gps', frac_mediana_gps, 'outliers', len(gps)-len(gps_validos))
    else:
        gps_validos = pd.DataFrame(columns=key + ["parada"])

    # --- Combinar fuentes ---
    paradas = pd.concat(
        [paradas_etapas[key + ["parada"]], gps_validos], ignore_index=True
    ).drop_duplicates(subset=key + ["parada"])

    # Descartar paradas con h3 nulo/vacio: cuando lat/lon es (0, 0) no se asigna
    # h3 y queda un string vacio. `is not null` en SQL no lo filtra, y luego
    # rompe h3.grid_disk('', ...). No corresponden a una parada real.
    paradas_pre_na = len(paradas)
    paradas = paradas[paradas["parada"].notna() & (paradas["parada"] != "")]
    if len(paradas) < paradas_pre_na:
        print(
            f"Descartadas {paradas_pre_na - len(paradas)} paradas con h3 vacio/nulo "
            "(lat/lon = 0)")

    # --- Footprint del corredor real por grupo: GPS valido; fallback recorrido ---
    footprint_por_grupo = {}
    if usa_gps and len(gps_validos) > 0:
        for gkey, sub in gps_validos.groupby(key):
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            footprint_por_grupo[gkey] = set(sub["parada"])

    # Recorridos oficiales: se traen id_linea (real) y modo desde metadata_ramales
    # y se resuelve id_linea_agg via metadata_lineas, para que la clave coincida
    # con las candidatas de etapas/gps (importante en modos agregados como metro).
    q_rec = """
    select distinct obgh.id_ramal as id_ramal, mr.id_linea as id_linea,
           mr.modo as modo, obgh.h3 as parada
    from official_branches_geoms_h3 obgh
    join metadata_ramales mr on obgh.id_ramal = mr.id_ramal
    """
    h3_recorridos = pd.read_sql(q_rec, conn_insumos)
    if len(h3_recorridos) > 0:
        h3_recorridos = h3_recorridos.merge(
            metadata_lineas[["id_linea", "id_linea_agg"]], how="left", on="id_linea")
        h3_recorridos["id_ramal"] = id_ramal_efectivo(
            h3_recorridos["modo"], h3_recorridos["id_ramal"], modos_ramal)
        h3_recorridos = h3_recorridos[key + ["parada"]].drop_duplicates()
        for gkey, sub in h3_recorridos.groupby(key):
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            footprint_por_grupo.setdefault(gkey, set(sub["parada"]))

    # --- Filtro por buffer: conservar candidatas dentro del footprint buffereado.
    # Grupos sin footprint (ni GPS ni recorrido) no se filtran (se confia en trx). ---
    if len(footprint_por_grupo) > 0:
        permitidas_rows = []
        for gkey, hexes in footprint_por_grupo.items():
            buffered = set()
            for hx in hexes:
                buffered.update(h3.grid_disk(hx, ring_filtro))
            for hx in buffered:
                permitidas_rows.append((*gkey, hx))
        permitidas = pd.DataFrame(
            permitidas_rows, columns=key + ["parada"]).drop_duplicates()

        grupos_con_footprint = set(footprint_por_grupo.keys())
        clave_tuplas = list(map(tuple, paradas[key].to_numpy()))
        mask_con = pd.Series(
            [t in grupos_con_footprint for t in clave_tuplas], index=paradas.index)
        paradas_con = paradas[mask_con].merge(
            permitidas, on=key + ["parada"], how="inner")
        paradas_sin = paradas[~mask_con]
        paradas_pre = len(paradas)
        paradas = pd.concat([paradas_con, paradas_sin], ignore_index=True)
        print(
            f"Filtro por recorrido/buffer (ring_filtro={ring_filtro}): "
            f"{paradas_pre} -> {len(paradas)} paradas")

    # --- Reconstruccion total: se recalcula la matriz entera en cada corrida ---
    # etapas y gps son acumulativas y se leen completas, asi que el estado actual ya
    # refleja todo el historico. Reconstruir (en vez de append-only) re-evalua los
    # outliers y saca paradas que ya no califican, y simplifica la funcion.
    if len(paradas) > 0:
        areas_influencia = pd.concat(
            map(
                get_stop_hex_ring,
                np.unique(paradas["parada"]),
                itertools.repeat(ring_size),
            )
        )
        matriz = paradas.merge(areas_influencia, how="left", on="parada")
        # Persistir NULL (no el centinela) para los modos sin ramal.
        matriz.loc[matriz["id_ramal"] == RAMAL_SENTINEL, "id_ramal"] = None
        matriz = matriz.reindex(
            columns=["id_linea_agg", "id_ramal", "parada", "area_influencia"])
        # Truncar + reescribir (preserva el schema/tipos definido en utils.py).
        conn_insumos.execute("DELETE FROM matriz_validacion")
        conn_insumos.commit()
        matriz.to_sql(
            "matriz_validacion", conn_insumos, if_exists="append", index=False)
        print(
            f"matriz_validacion reconstruida: {len(matriz)} filas, "
            f"{paradas['parada'].nunique()} paradas")
    else:
        print("Sin paradas candidatas: matriz_validacion no se modifica")

    conn_data.close()
    conn_insumos.close()
    return None

@duracion
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
