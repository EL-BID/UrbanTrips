import logging
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
from shapely.geometry.base import BaseGeometry
from networkx import NetworkXNoPath
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
from urbantrips.utils.paths import get_paths
from urbantrips.carto.equivalencias import (
    construir_equivalencias_zonas,
    upsert_equivalencias_zonas,
)
import warnings

try:
    from pandana.loaders import osm as osm_pandana  # noqa: F401

    warnings.filterwarnings(
        "ignore",
        message="Unsigned integer: shortest path distance is trying to be calculated",
        category=UserWarning,
        module="pandana.network",
    )
except ImportError:
    pass
from urbantrips.utils.utils import (
    duracion,
    leer_configs_generales,
    leer_alias,
    modos_con_ramal,
    id_ramal_efectivo,
    RAMAL_SENTINEL,
)
from urbantrips.storage.context import StorageContext

import subprocess
from math import floor
from shapely import wkt

logger = logging.getLogger(__name__)


def _normalize_zone_ids(ids):
    """Cast zone IDs to strings without float artifacts like '123.0'."""
    numeric_ids = pd.to_numeric(ids, errors="coerce")
    integral_numeric = numeric_ids.notna() & np.isclose(
        numeric_ids, np.floor(numeric_ids)
    )
    normalized = ids.astype(str)
    normalized.loc[integral_numeric] = (
        numeric_ids.loc[integral_numeric].astype("Int64").astype(str)
    )
    return normalized


def _as_geodataframe_wkt(df, crs_default="EPSG:4326"):
    """Return a GeoDataFrame parsing WKT strings in 'geometry' when needed."""
    out = df.copy()
    out["geometry"] = out["geometry"].apply(
        lambda g: wkt.loads(g) if isinstance(g, str) else g
    )
    crs = getattr(df, "crs", None) or crs_default
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def _with_wkt_geometry(df):
    """Return a plain DataFrame with geometry values serialized to WKT."""
    out = pd.DataFrame(df.copy())
    if "geometry" not in out.columns:
        return out

    geometry_dtype = getattr(df.dtypes.get("geometry"), "name", None)
    if geometry_dtype == "geometry":
        out["geometry"] = df["geometry"].to_wkt()
    else:
        out["geometry"] = out["geometry"].map(
            lambda geom: geom.wkt if isinstance(geom, BaseGeometry) else geom
        )
    return out


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
def update_stations_catchment_area(ring_size, ctx: StorageContext):
    """
    Actualiza la matriz de validacion de paradas (matriz_validacion) combinando
    dos fuentes de paradas:
      - transacciones (etapas.h3_o), descartando pares con una sola ocurrencia
      - puntos GPS (gps.h3), descartando hexagonos con muy pocos puntos (outliers
        de densidad: glitches de GPS)
    El filtro de outliers se aplica SOLO a los puntos GPS (por densidad): las
    paradas de transacciones (etapas.h3_o con n>1) son verdad de campo y nunca se
    descartan. El GPS solo AGREGA cobertura donde es denso; no puede eliminar una
    parada de transacciones. Asi, lineas con GPS escaso o poco representativo
    (p.ej. FFCC Roca: 582 puntos GPS) no pierden estaciones reales que la gente si
    usa. Construye el area de influencia (anillo H3 de tamano ring_size) para todas
    las paradas resultantes.

    El uso de ramal se decide por modo (modo_valida_ramal). Para los modos que
    validan por ramal construye por (id_linea_agg, id_ramal); para los que no,
    colapsa los ramales por (id_linea_agg) y deja id_ramal en NULL. En memoria
    los modos sin ramal usan un centinela (RAMAL_SENTINEL) para que el merge por
    [id_linea_agg, id_ramal] funcione uniforme; se persiste NULL.

    Es reconstruccion total: etapas y gps son acumulativas y se leen completas,
    asi que la matriz se reescribe entera en cada corrida (re-evalua outliers).

    Parametros leidos de configuraciones_generales.yaml:
      - frac_mediana_gps (default 0.25): umbral de outliers GPS como fraccion de la
        mediana de puntos por hexagono
      - lineas_contienen_ramales, modo_valida_ramal
    """
    configs = leer_configs_generales(autogenerado=False)
    modos_ramal = modos_con_ramal(configs)
    frac_mediana_gps = configs.get("frac_mediana_gps", 0.25)

    # Key fija; id_ramal lleva el valor efectivo (real para modos con ramal,
    # RAMAL_SENTINEL para los que no). Se convierte a NULL al persistir.
    key = ["id_linea_agg", "id_ramal"]

    def _es_h3_valido(cell):
        return isinstance(cell, str) and h3.is_valid_cell(cell)

    # --- Migracion: si la tabla persistida quedo con el schema viejo
    # (id_linea / sin id_ramal), se descarta y recrea con la nueva semantica
    # (id_linea_agg, id_ramal). Asi save_matrix_validation (DELETE+INSERT) escribe
    # contra las columnas correctas y los consumidores encuentran id_linea_agg. ---
    from urbantrips.storage.schema.insumos import MATRIZ_VALIDACION

    cols_actuales = set(
        ctx.insumos.query("SELECT * FROM matriz_validacion LIMIT 0").columns
    )
    schema_viejo = bool(cols_actuales) and (
        "id_linea_agg" not in cols_actuales or "id_ramal" not in cols_actuales
    )
    if schema_viejo:
        logger.info(
            "matriz_validacion con schema viejo (%s); se reconstruye con "
            "id_linea_agg, id_ramal",
            sorted(cols_actuales),
        )
        ctx.insumos.execute("DROP TABLE IF EXISTS matriz_validacion")
        ctx.insumos.execute(MATRIZ_VALIDACION)

    metadata_lineas = ctx.insumos.get_metadata_lineas()[
        ["id_linea", "id_linea_agg", "modo"]
    ]

    # --- Fuente A: paradas desde transacciones (etapas) ---
    paradas_etapas = ctx.data.query(
        "select id_linea, id_ramal, h3_o as parada, count(*) as n "
        "from etapas group by id_linea, id_ramal, h3_o"
    )
    paradas_etapas = paradas_etapas[paradas_etapas["parada"].map(_es_h3_valido)].copy()
    paradas_etapas = paradas_etapas.merge(metadata_lineas, how="left", on="id_linea")
    paradas_etapas["id_ramal"] = id_ramal_efectivo(
        paradas_etapas["modo"], paradas_etapas["id_ramal"], modos_ramal
    )
    paradas_etapas = paradas_etapas.drop(columns=["id_linea", "modo"])
    # Re-sumar el conteo por la clave efectiva (al colapsar los ramales de un modo
    # sin ramal hay que sumar sus n). Filtro >1: descartar paradas de una sola obs.
    paradas_etapas = paradas_etapas.groupby(key + ["parada"], as_index=False)["n"].sum()
    paradas_etapas = paradas_etapas[paradas_etapas["n"] > 1].drop(columns=["n"])

    # --- Fuente B: paradas desde GPS, con filtro de outliers por densidad ---
    # Se detecta GPS por presencia de datos en la tabla (mas robusto que el flag
    # de config, que puede quedar en None aunque la tabla este poblada).
    # gps no tiene columna modo: se trae del merge con metadata_lineas.
    gps = ctx.data.query(
        "select id_linea, id_ramal, h3 as parada, count(*) as n_pts "
        "from gps where h3 is not null group by id_linea, id_ramal, h3"
    )
    usa_gps = len(gps) > 0
    if usa_gps:
        gps = gps[gps["parada"].map(_es_h3_valido)].copy()
        gps = gps.merge(metadata_lineas, how="left", on="id_linea")
        gps["id_ramal"] = id_ramal_efectivo(gps["modo"], gps["id_ramal"], modos_ramal)
        gps = gps.drop(columns=["id_linea", "modo"])
        # Sumar n_pts por la clave efectiva + parada: colapsa los ramales de un modo
        # sin ramal para que el conteo del hexagono sea el total antes del filtro.
        gps = gps.groupby(key + ["parada"], as_index=False)["n_pts"].sum()
        mediana = gps.groupby(key)["n_pts"].transform("median")
        umbral = np.maximum(2, np.round(mediana * frac_mediana_gps))
        gps_validos = gps[gps["n_pts"] >= umbral].copy()
        gps_validos = gps_validos[key + ["parada"]]
        logger.info(
            "frac_mediana_gps=%s, outliers GPS descartados=%d",
            frac_mediana_gps,
            len(gps) - len(gps_validos),
        )
    else:
        gps_validos = pd.DataFrame(columns=key + ["parada"])

    # --- Combinar fuentes ---
    paradas = pd.concat(
        [paradas_etapas[key + ["parada"]], gps_validos], ignore_index=True
    ).drop_duplicates(subset=key + ["parada"])

    # Descartar paradas con h3 nulo/vacio: cuando lat/lon es (0, 0) no se asigna
    # h3 y queda un string vacio que rompe h3.grid_disk('', ...).
    paradas_pre_na = len(paradas)
    paradas = paradas[paradas["parada"].notna() & (paradas["parada"] != "")]
    if len(paradas) < paradas_pre_na:
        logger.info(
            "Descartadas %d paradas con h3 vacio/nulo (lat/lon = 0)",
            paradas_pre_na - len(paradas),
        )

    # --- Sin filtro por corredor sobre transacciones ---
    # Las paradas de transacciones (n>1) son verdad de campo: la gente pico la
    # tarjeta ahi, asi que nunca se descartan por un corredor GPS/recorrido. El GPS
    # ya viene limpio de outliers por densidad (Fuente B) y solo SUMA cobertura;
    # como el GPS define su propio corredor, filtrar el GPS por ese corredor seria
    # un no-op. El unico efecto del viejo filtro por footprint era borrar paradas de
    # transacciones en lineas con GPS escaso o no representativo (p.ej. FFCC Roca:
    # 93 -> 35 paradas), perdiendo estaciones reales; por eso se elimina.

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
            columns=["id_linea_agg", "id_ramal", "parada", "area_influencia"]
        )
        # DELETE + INSERT (preserva el schema/tipos del DDL de insumos).
        ctx.insumos.save_matrix_validation(matriz)
        logger.info(
            "matriz_validacion reconstruida: %d filas, %d paradas",
            len(matriz),
            paradas["parada"].nunique(),
        )
    else:
        logger.info("Sin paradas candidatas: matriz_validacion no se modifica")

    return None


def _load_zonificaciones_from_config(configs):
    """Load and normalise each zone layer declared in config, return a GeoDataFrame."""
    zona_cfg = configs["zonificaciones"]
    frames = []
    for n in range(1, 6):
        file_zona = zona_cfg.get(f"geo{n}")
        var_zona = zona_cfg.get(f"var{n}")
        if not file_zona or not var_zona:
            continue

        db_path = str(get_paths().input_dir / file_zona)
        if not os.path.exists(db_path):
            continue

        zonif = gpd.read_file(db_path)[[var_zona, "geometry"]].copy()
        zonif.columns = ["id", "geometry"]
        zonif["zona"] = var_zona

        zonif["id"] = _normalize_zone_ids(zonif["id"])

        matriz_order = zona_cfg.get(f"orden{n}") or ""
        if matriz_order:
            order = (
                pd.DataFrame(matriz_order, columns=["id"])
                .reset_index()
                .rename(columns={"index": "orden"})
            )
            zonif = zonif.merge(order, how="left")
        else:
            zonif["orden"] = 0

        frames.append(zonif[["zona", "id", "orden", "geometry"]])

    if not frames:
        return gpd.GeoDataFrame()

    return pd.concat(frames, ignore_index=True)


@duracion
def guardo_zonificaciones(ctx: StorageContext, resoluciones_equivalencias=None):
    """Persist zoning layers and polygons, and rebuild the long-format
    equivalencias_zonas table for them.

    Parameters
    ----------
    ctx : StorageContext
    resoluciones_equivalencias : iterable of int, optional
        Extra H3 resolutions for equivalencias_zonas, on top of
        configs['resolucion_h3'] (e.g. the resolution used by chains_norm).
    """
    configs = leer_configs_generales(autogenerado=False)

    from urbantrips.preparo_dashboard.chains import RES_CHAINS_NORM

    resoluciones = {configs.get("resolucion_h3"), RES_CHAINS_NORM}
    if resoluciones_equivalencias is not None:
        resoluciones |= set(resoluciones_equivalencias)
    resoluciones = sorted(r for r in resoluciones if r)

    zonificaciones_para_equivalencias = None
    poligonos_para_equivalencias = None

    if configs["zonificaciones"]:
        logger.info("Crear zonificaciones en db")
        zonificaciones = _load_zonificaciones_from_config(configs)

        if len(zonificaciones) > 0:
            logger.info("guardo_zonificaciones: disolviendo polígonos de zonificación")
            zonificaciones["orden"] = zonificaciones["orden"].fillna(0)
            zonificaciones["zona"] = zonificaciones["zona"].fillna("")
            zonificaciones["id"] = zonificaciones["id"].fillna("")
            zonificaciones = zonificaciones.dissolve(
                ["zona", "id", "orden"], as_index=False
            )

            crs_val = configs["epsg_m"]
            crs_actual = zonificaciones.crs

            zonificaciones_disolved = zonificaciones[
                ~zonificaciones.zona.isin(["res_6", "res_7", "res_8"])
            ].copy()
            zonificaciones_disolved["all"] = 1
            zonificaciones_disolved = (
                zonificaciones_disolved[["all", "geometry"]]
                .dissolve(by="all")
                .to_crs(crs_val)
                .buffer(2000)
                .to_crs(crs_actual)
            )

            logger.info("guardo_zonificaciones: generando hexágonos H3 res 6 y 7")
            h3_layers = []
            for res in (6, 7):
                layer = generate_h3_hexagons_within_polygon(
                    zonificaciones_disolved, res, crs_val
                )
                layer["zona"] = f"res_{res}"
                layer["orden"] = 0
                layer = layer.rename(columns={"h3_index": "id"})[
                    ["zona", "id", "orden", "geometry"]
                ]
                h3_layers.append(layer)

            zonificaciones = pd.concat([zonificaciones, *h3_layers], ignore_index=True)

            logger.info("guardo_zonificaciones: guardando zonificaciones")
            zonificaciones_to_save = _with_wkt_geometry(zonificaciones)

            ctx.insumos.save_raw(zonificaciones_to_save, "zonificaciones")
            zonificaciones_para_equivalencias = zonificaciones

    if configs["poligonos"]:

        poly_file = configs["poligonos"]

        db_path = str(get_paths().input_dir / poly_file)
        logger.info("guardo_zonificaciones: cargando polígonos desde %s", poly_file)
        poligonos_db = ctx.insumos.get_raw("poligonos")

        if os.path.exists(db_path):
            poly = gpd.read_file(db_path)

            if len(poligonos_db) > 0:
                poligonos_db = poligonos_db.loc[
                    poligonos_db["id"] == "estimacion de demanda dibujada",
                ]
                poly = pd.concat(
                    [poly, poligonos_db.dropna(axis=1, how="all")],
                    ignore_index=True,
                )

            logger.info(
                "guardo_zonificaciones: guardando polígonos (%d filas)", len(poly)
            )
            poly_to_save = _with_wkt_geometry(poly)
            ctx.insumos.save_raw(poly_to_save, "poligonos")
            poligonos_para_equivalencias = _as_geodataframe_wkt(poly)

    if (
        zonificaciones_para_equivalencias is not None
        or poligonos_para_equivalencias is not None
    ):
        logger.info(
            "guardo_zonificaciones: generando equivalencias_zonas long (res %s)",
            resoluciones,
        )
        equivalencias = construir_equivalencias_zonas(
            gdf_zonas=zonificaciones_para_equivalencias,
            gdf_poligonos=poligonos_para_equivalencias,
            resoluciones=resoluciones,
        )
        upsert_equivalencias_zonas(equivalencias, ctx=ctx)


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
    logger.debug(
        "Resolución actual: %s, Resolución objetivo: %s",
        current_resolution,
        target_resolution,
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
