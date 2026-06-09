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
    Esta funcion toma la matriz de validacion de paradas
    y la actualiza en base a datos de fechas que no esten
    ya en la matriz
    """
    # Leer las paradas en base a las etapas
    q = """
    select id_linea, h3_o as parada from etapas
    """
    paradas_etapas = ctx.data.query(q)
    metadata_lineas = ctx.insumos.get_metadata_lineas()
    paradas_etapas = paradas_etapas[
        paradas_etapas["parada"].map(lambda cell: isinstance(cell, str) and h3.is_valid_cell(cell))
    ].copy()

    paradas_etapas = paradas_etapas.merge(
        metadata_lineas[["id_linea", "id_linea_agg"]], how="left", on="id_linea"
    ).drop(["id_linea"], axis=1)

    paradas_etapas = paradas_etapas.groupby(
        ["id_linea_agg", "parada"], as_index=False
    ).size()

    paradas_etapas = paradas_etapas[(paradas_etapas["size"] > 1)].drop(["size"], axis=1)

    # filtrar paradas solo los que estan en recorridos h3
    obgh = ctx.insumos.get_raw("official_branches_geoms_h3")
    mr = ctx.insumos.get_metadata_ramales()

    if not obgh.empty and not mr.empty:
        obgh = obgh.copy()
        obgh["id_ramal"] = obgh["id_ramal"].astype("int64")
        h3_recorridos = (
            obgh[["id_ramal", "h3"]]
            .merge(mr[["id_ramal", "id_linea"]], on="id_ramal")
            [["id_linea", "h3"]]
            .rename(columns={"id_linea": "id_linea_agg", "h3": "parada"})
            .drop_duplicates()
        )
    else:
        h3_recorridos = pd.DataFrame(columns=["id_linea_agg", "parada"])

    if len(h3_recorridos) > 0:
        h3_recorridos["parada_en_recorridos"] = True
        paradas_etapas["id_linea_recorridos"] = paradas_etapas["id_linea_agg"].isin(
            h3_recorridos["id_linea_agg"].unique()
        )

        paradas_etapas = paradas_etapas.merge(
            h3_recorridos, on=["id_linea_agg", "parada"], how="left"
        )

        paradas_etapas["parada_en_recorridos"] = (
            paradas_etapas["parada_en_recorridos"]
            .fillna(False)
            .infer_objects(copy=False)
            .astype(bool)
        )

        paradas_etapas["borrar"] = (paradas_etapas.id_linea_recorridos) & (
            ~paradas_etapas.parada_en_recorridos
        )

        paradas_pre = len(paradas_etapas)
        logger.debug("Paradas antes de eliminar por recorridos: %d", paradas_pre)
        paradas_etapas = paradas_etapas.loc[
            ~paradas_etapas.borrar, ["id_linea_agg", "parada"]
        ]
        logger.debug("Eliminadas %d paradas sin recorridos", paradas_pre - len(paradas_etapas))

    # Leer las paradas ya existentes en la matriz
    paradas_en_matriz = ctx.insumos.get_matrix_validation()
    if not paradas_en_matriz.empty:
        paradas_en_matriz = (
            paradas_en_matriz[["id_linea_agg", "parada"]]
            .drop_duplicates()
            .copy()
        )
        paradas_en_matriz["m"] = 1
    else:
        paradas_en_matriz = pd.DataFrame(columns=["id_linea_agg", "parada", "m"])

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

        ctx.insumos.append_raw(matriz_nueva, "matriz_validacion")
        logger.info("Fin actualizacion matriz de validacion")
    else:
        logger.info(
            "La matriz de validacion ya tiene los datos más actuales"
            " en base a la informacion existente en la tabla de etapas"
        )

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
def guardo_zonificaciones(ctx: StorageContext):
    configs = leer_configs_generales(autogenerado=False)

    if configs["zonificaciones"]:
        logger.info("Crear zonificaciones en db")
        zonificaciones = _load_zonificaciones_from_config(configs)

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

            zonificaciones = pd.concat(
                [zonificaciones, *h3_layers], ignore_index=True
            )

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

            zonificaciones_to_save = _with_wkt_geometry(zonificaciones)

            ctx.insumos.save_raw(zonificaciones_to_save, "zonificaciones")
            ctx.insumos.save_raw(equivalencias_zonas, "equivalencias_zonas")

    if configs["poligonos"]:

        poly_file = configs["poligonos"]

        db_path = str(get_paths().input_dir / poly_file)
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

            poly_to_save = _with_wkt_geometry(poly)
            ctx.insumos.save_raw(poly_to_save, "poligonos")

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
    logger.debug("Resolución actual: %s, Resolución objetivo: %s", current_resolution, target_resolution)
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
