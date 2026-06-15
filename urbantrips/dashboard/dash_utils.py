import logging
from shapely.geometry import LineString
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import yaml
import sqlite3
import duckdb
from shapely import wkt
from matplotlib import colors as mcolors
from folium import Figure
from shapely.geometry import LineString, Point, Polygon, shape, mapping
import h3
from datetime import datetime
from pathlib import Path
import shutil
import json
import mapclassify
try:
    import pydeck as pdk
    _PYDECK_AVAILABLE = True
except ImportError:
    _PYDECK_AVAILABLE = False

logger = logging.getLogger(__name__)

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.utils.dataframe import calculate_weighted_means  # noqa: F401 — used by callers via this module
from urbantrips.utils.paths import get_paths
from urbantrips.dashboard.dash_storage import (
    _load_yaml_simple,
    _find_first_valid_yaml,
    leer_configs_generales,
    resolve_db_aliases,
    normalize_vars,
    _fetch_sql_dataframe,
    get_project_root,
)

# def leer_configs_generales(autogenerado=True):
#     """
#     Lee el archivo de configuración YAML, probando primero con UTF-8
#     y luego con latin-1 si es necesario. Devuelve un dict o {} si falla.
#     """
#     archivo = (
#         "configuraciones_generales_autogenerado.yaml"
#         if autogenerado
#         else "configuraciones_generales.yaml"
#     )
#     path = os.path.join("configs", archivo)

#     try:
#         with open(path, "r", encoding="utf-8") as file:
#             return yaml.safe_load(file)
#     except UnicodeDecodeError:
#         try:
#             with open(path, "r", encoding="latin-1") as file:
#                 return yaml.safe_load(file)
#         except yaml.YAMLError as error:
#             print(f"❌ Error YAML en archivo con latin-1: {error}")
#         except Exception as e:
#             print(f"❌ Error general con latin-1: {e}")
#     except yaml.YAMLError as error:
#         print(f"❌ Error YAML en archivo con UTF-8: {error}")
#     except Exception as e:
#         print(f"❌ Error general leyendo archivo: {e}")

#     return {}



def leer_alias(tipo="dash"):
    configs = leer_configs_generales(autogenerado=False)
    corridas = configs.get("corridas", [])

    # Multi-corrida day-selector: data and dash use the selected corrida name.
    # Only applies when each corrida has its own DB (no shared alias_db key).
    if tipo in ("data", "dash") and len(corridas) > 1 and "dia_seleccionado" in st.session_state and "alias_db" not in configs:
        posicion = corridas.index(st.session_state.dia_seleccionado)
        return corridas[posicion] + "_"

    aliases = resolve_db_aliases(configs)
    if tipo not in aliases:
        raise ValueError("tipo invalido: %s" % tipo)
    alias = aliases[tipo]
    return alias + "_" if alias else ""


def get_db_path(tipo="data", alias_db=""):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve el path a una base de datos con esa informacion
    """
    if tipo not in ("data", "insumos", "dash", "general"):
        raise ValueError("tipo invalido: %s" % tipo)
    if len(alias_db) == 0:
        alias_db = leer_alias(tipo)
    if not alias_db.endswith("_"):
        alias_db += "_"

    db_dir = get_paths().db_dir
    candidates = [
        db_dir / f"{alias_db}{tipo}.duckdb",
        db_dir / f"{alias_db}{tipo}.sqlite",
    ]
    db_path = next((p for p in candidates if p.exists()), None)
    if db_path is None:
        raise FileNotFoundError(
            f"No se encontró {alias_db}{tipo} en {db_dir}"
        )

    return db_path


def iniciar_conexion_db(tipo="data", alias_db=""):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve una conexion a la db (DuckDB o SQLite segun el archivo disponible)
    """
    if len(alias_db) == 0:
        alias_db = leer_alias(tipo)
    if not alias_db.endswith("_"):
        alias_db += "_"
    db_path = get_db_path(tipo, alias_db)

    if str(db_path).endswith(".duckdb"):
        return duckdb.connect(str(db_path), read_only=False)
    return sqlite3.connect(db_path, timeout=10)


# Calculate weighted mean, handling division by zero or empty inputs


def weighted_mean(series, weights):
    try:
        result = (series * weights).sum() / weights.sum()
    except ZeroDivisionError:
        result = np.nan
    return result



# def _load_table_sql(tabla_sql, tabla_tipo="dash", query="", alias_db="", params=None):
#     if alias_db and not alias_db.endswith("_"):
#         alias_db += "_"

#     if len(query) == 0:
#         tabla_sql = validate_table_name(tabla_sql)
#         query = f"SELECT * FROM {tabla_sql}"

#     conn = iniciar_conexion_db(tipo=tabla_tipo, alias_db=alias_db)

#     try:
#         tabla = _fetch_sql_dataframe(conn, query, params=params)
#     except (sqlite3.OperationalError, duckdb.Error, pd.io.sql.DatabaseError) as e:
#         error_message = str(e).lower()
#         if "no such table" in error_message or "does not exist" in error_message:
#             logger.warning("La tabla '%s' no existe.", tabla_sql)
#             tabla = pd.DataFrame([])
#         else:
#             raise
#     finally:
#         conn.close()

#     if "wkt" in tabla.columns and not tabla.empty:
#         tabla["geometry"] = tabla.wkt.apply(wkt.loads)        
#         tabla = tabla.drop(["wkt"], axis=1)
#     if "geometry" in tabla.columns:
#         tabla = gpd.GeoDataFrame(
#             tabla,
#             geometry="geometry",
#             crs="EPSG:4326"
#         )

#     tabla = normalize_vars(tabla)

#     return tabla
from shapely import wkt
from shapely.geometry.base import BaseGeometry

def _load_table_sql(tabla_sql, tabla_tipo="dash", query="", alias_db="", params=None):
    if alias_db and not alias_db.endswith("_"):
        alias_db += "_"

    if len(query) == 0:
        tabla_sql = validate_table_name(tabla_sql)
        query = f"SELECT * FROM {tabla_sql}"

    conn = iniciar_conexion_db(tipo=tabla_tipo, alias_db=alias_db)

    try:
        tabla = _fetch_sql_dataframe(conn, query, params=params)
    except (sqlite3.OperationalError, duckdb.Error, pd.io.sql.DatabaseError) as e:
        error_message = str(e).lower()
        if "no such table" in error_message or "does not exist" in error_message:
            logger.warning("La tabla '%s' no existe.", tabla_sql)
            tabla = pd.DataFrame([])
        else:
            raise
    finally:
        conn.close()

    if "wkt" in tabla.columns and not tabla.empty:
        tabla["geometry"] = tabla["wkt"].apply(wkt.loads)
        tabla = tabla.drop(columns=["wkt"])

    elif "geometry" in tabla.columns and not tabla.empty:
        sample_geom = tabla["geometry"].dropna().iloc[0] if tabla["geometry"].notna().any() else None

        if isinstance(sample_geom, str):
            tabla["geometry"] = tabla["geometry"].apply(
                lambda x: wkt.loads(x) if pd.notna(x) else None
            )

        elif sample_geom is not None and not isinstance(sample_geom, BaseGeometry):
            raise TypeError(
                f"La columna geometry existe pero no contiene geometrías válidas. "
                f"Tipo detectado: {type(sample_geom)}"
            )

    if "geometry" in tabla.columns and not tabla.empty:
        tabla = gpd.GeoDataFrame(
            tabla,
            geometry="geometry",
            crs="EPSG:4326"
        )

    tabla = normalize_vars(tabla)

    return tabla

@st.cache_data
def levanto_tabla_sql(tabla_sql, tabla_tipo="dash", query="", alias_db=""):
    return _load_table_sql(tabla_sql, tabla_tipo=tabla_tipo, query=query, alias_db=alias_db)


def levanto_tabla_sql_local(tabla_sql, tabla_tipo="dash", query="", alias_db=""):
    return _load_table_sql(tabla_sql, tabla_tipo=tabla_tipo, query=query, alias_db=alias_db)


def build_where_clauses(filters: dict, table_alias: str = "c") -> str:
    """
    Build optional WHERE clauses from a filters dict.
    Keys are column names in chains_norm.
    None values are skipped (user selected 'Todos'/'Todas').
    Returns a string starting with ' AND ' or empty string.
    """
    clauses = []
    for col, val in filters.items():
        if val is not None:
            if isinstance(val, int):
                clauses.append(f"{table_alias}.{col} = {val}")
            else:
                clauses.append(f"{table_alias}.{col} = '{val}'")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


# ---------------------------------------------------------------------------
# chains_norm loaders — on-the-fly aggregation replacing the precomputed
# agg_etapas / agg_matrices / poly_etapas / poly_matrices tables.
# ---------------------------------------------------------------------------

_CHAINS_COLS = (
    "c.dia, c.h3_inicio, c.h3_fin, "
    "c.h3_inicio_norm, c.h3_transfer1_norm, c.h3_transfer2_norm, c.h3_fin_norm, "
    "c.transferencia, c.modo_agregado, c.rango_hora, c.distancia_agregada, "
    "c.genero_agregado, c.tarifa_agregada, "
    "c.distance_od, c.travel_time_min, c.travel_speed, c.seq_lineas, "
    "c.factor_expansion_linea"
)

# (output metric, source column in chains_norm) for fex-weighted means
_CHAINS_METRICAS = [
    ("distance_od", "distance_od"),
    ("travel_time_min", "travel_time_min"),
    ("kmh_od", "travel_speed"),
]


def _agg_chains_ponderado(df, dims):
    """Group chains rows by dims summing fex and adding the fex-weighted
    means of distance_od / travel_time_min / kmh_od. NaN and 0 values are
    excluded from each mean (same convention as zero_to_nan downstream)."""
    df = df.copy()
    sumas = {"factor_expansion_linea": ("factor_expansion_linea", "sum")}
    for met, src in _CHAINS_METRICAS:
        valido = df[src].notna() & (df[src] != 0)
        df[f"_{met}_num"] = (df[src] * df["factor_expansion_linea"]).where(valido, 0)
        df[f"_{met}_den"] = df["factor_expansion_linea"].where(valido, 0)
        sumas[f"_{met}_num"] = (f"_{met}_num", "sum")
        sumas[f"_{met}_den"] = (f"_{met}_den", "sum")

    agg = df.groupby(dims, as_index=False).agg(**sumas)
    for met, _src in _CHAINS_METRICAS:
        agg[met] = (
            (agg[f"_{met}_num"] / agg[f"_{met}_den"].replace(0, np.nan))
            .fillna(0)
            .round(2)
        )
        agg = agg.drop(columns=[f"_{met}_num", f"_{met}_den"])
    return agg


@st.cache_data
def traer_dias_chains():
    """Available days in chains_norm, sorted ascending."""
    df = levanto_tabla_sql(
        "chains_norm", "dash", "SELECT DISTINCT dia FROM chains_norm ORDER BY dia;"
    )
    return df["dia"].tolist() if len(df) > 0 else []


@st.cache_data
def traer_opciones_chains(col):
    """Distinct non-empty values of a chains_norm filter column."""
    df = levanto_tabla_sql(
        "chains_norm", "dash",
        f"SELECT DISTINCT {col} FROM chains_norm ORDER BY {col};",
    )
    if len(df) == 0:
        return []
    return [v for v in df[col].dropna().tolist() if str(v) != ""]


@st.cache_data
def traer_mapa_zona(zona, solo_zonificacion=True):
    """h3 -> zone id mapping for one layer of the long equivalencias_zonas."""
    tipo_filtro = " AND tipo = 'zonificacion'" if solo_zonificacion else ""
    eq = levanto_tabla_sql(
        "equivalencias_zonas", "insumos",
        query=(
            "SELECT h3, id FROM equivalencias_zonas "
            f"WHERE zona = '{zona}'{tipo_filtro}"
        ),
    )
    if len(eq) == 0:
        return {}
    return dict(zip(eq["h3"], eq["id"]))


@st.cache_data
def traer_h3_poligono(id_polygon):
    """H3 cell set of one analysis polygon (tipo 'poligono' or 'cuenca')."""
    eq = levanto_tabla_sql(
        "equivalencias_zonas", "insumos",
        query=f"SELECT h3 FROM equivalencias_zonas WHERE zona = '{id_polygon}'",
    )
    return set(eq["h3"]) if len(eq) > 0 else set()


@st.cache_data
def traer_poligonos_largos():
    """Analysis polygons (id, tipo) present in long equivalencias_zonas."""
    return levanto_tabla_sql(
        "equivalencias_zonas", "insumos",
        query=(
            "SELECT DISTINCT zona AS id, tipo FROM equivalencias_zonas "
            "WHERE tipo IN ('poligono', 'cuenca') ORDER BY zona"
        ),
    )


def levanto_chains_norm(dia_seleccionado=None, where_extra=""):
    """Read chains_norm rows for one day (or all days) with optional extra
    WHERE clauses produced by build_where_clauses (alias 'c')."""
    where = " WHERE 1=1"
    if dia_seleccionado is not None and dia_seleccionado != "Todos":
        where += f" AND c.dia = '{dia_seleccionado}'"
    query = f"SELECT {_CHAINS_COLS} FROM chains_norm c{where}{where_extra}"
    return levanto_tabla_sql("chains_norm", "dash", query=query)


def coordenadas_zonas(zonificaciones, zona_seleccionada):
    """Per-zone representative point and display order for one zoning layer.

    Returns a DataFrame (id, lat, lon, orden_id) where orden_id is the
    '###_id' label the OD matrices sort and display by. Zone ids that are
    valid H3 cells but missing from zonificaciones (e.g. res_X layers) get
    the cell centroid as coordinates.
    """
    cols = ["id", "lat", "lon", "orden_id"]
    if len(zonificaciones) == 0 or "geometry" not in zonificaciones.columns:
        return pd.DataFrame(columns=cols)

    zonif = zonificaciones[zonificaciones.zona == zona_seleccionada].copy()
    if len(zonif) == 0:
        return pd.DataFrame(columns=cols)

    puntos = zonif.geometry.representative_point()
    zonif["lat"] = puntos.y
    zonif["lon"] = puntos.x

    if "orden" in zonif.columns and zonif["orden"].notna().all():
        orden = zonif["orden"].astype(int)
    else:
        orden = zonif["id"].rank(method="dense").astype(int)
    zonif["orden_id"] = (
        orden.astype(str).str.zfill(3) + "_" + zonif["id"].astype(str)
    )

    return zonif[cols].drop_duplicates(subset=["id"])


def _coords_h3_faltantes(ids, coords):
    """Complete the coords frame with H3 centroids for zone ids that are
    valid H3 cells (res_X layers built straight from cells)."""
    conocidos = set(coords["id"])
    faltantes = [
        i for i in pd.unique(pd.Series(list(ids)).dropna())
        if i not in conocidos and isinstance(i, str) and h3.is_valid_cell(i)
    ]
    if not faltantes:
        return coords
    latlon = [h3.cell_to_latlng(c) for c in faltantes]
    extra = pd.DataFrame(
        {
            "id": faltantes,
            "lat": [p[0] for p in latlon],
            "lon": [p[1] for p in latlon],
        }
    ).sort_values("id")
    extra["orden_id"] = (
        (pd.RangeIndex(len(extra)) + len(coords) + 1)
        .astype(str).str.zfill(3) + "_" + extra["id"].astype(str)
    )
    return pd.concat([coords, extra], ignore_index=True)


def _aplicar_sufijo_cuenca(serie_zona, serie_h3, h3_cuenca):
    """Append ' (cuenca)' to zone names whose H3 cell is inside the basin."""
    out = serie_zona.copy()
    mask = serie_h3.isin(h3_cuenca) & out.notna()
    out.loc[mask] = out.loc[mask].astype(str) + " (cuenca)"
    return out


# ---------------------------------------------------------------------------
# SQL path: chains_norm x equivalencias_zonas joined and aggregated inside
# the dash DB (single connection), so only aggregated rows reach Streamlit.
# Requires the dash copy of equivalencias_zonas (asegurar_equivalencias_dash).
# ---------------------------------------------------------------------------

def _sql_txt(valor):
    """Escape a value for inclusion in a SQL string literal."""
    return str(valor).replace("'", "''")


def _sub_h3_equivalencias(zona, valor=None):
    """Subquery with the H3 cells of one layer (and optionally one zone)."""
    sub = f"SELECT h3 FROM equivalencias_zonas WHERE zona = '{_sql_txt(zona)}'"
    if valor is not None:
        sub += f" AND id = '{_sql_txt(valor)}'"
    return sub


@st.cache_data
def asegurar_equivalencias_dash():
    """Ensure the dash DB holds copies of insumos tables used by the dashboard.

    The pipeline copies them via sincronizar_equivalencias_dash; if missing
    (older run), they are created here from insumos on the fly.
    """
    chk = levanto_tabla_sql(
        "equivalencias_zonas", "dash",
        query="SELECT h3 FROM equivalencias_zonas LIMIT 1",
    )
    if len(chk) == 0:
        eq = levanto_tabla_sql("equivalencias_zonas", "insumos")
        if len(eq) > 0 and "zona" in eq.columns:
            guardar_tabla_sql(
                pd.DataFrame(eq.drop(columns="geometry", errors="ignore")),
                "equivalencias_zonas", "dash", modo="replace",
            )

    for tabla in ("zonificaciones", "poligonos"):
        chk2 = levanto_tabla_sql(tabla, "dash",
                                 query=f"SELECT 1 FROM {tabla} LIMIT 1")
        if len(chk2) > 0:
            continue
        src = levanto_tabla_sql_local(tabla, "insumos")
        if len(src) == 0:
            continue
        raw = pd.DataFrame(src.copy())
        if hasattr(src, "geometry") and "geometry" in raw.columns:
            raw["geometry"] = src.geometry.to_wkt()
        guardar_tabla_sql(raw, tabla, "dash", modo="replace")

    return True


def condicion_zona_sql(zona_filtro, valor_filtro,
                       tipo_filtro="OD y Transferencias", direccional=False):
    """SQL condition: the chain touches one zone of any zoning layer.

    With 'Solo OD' only origin/destination are checked; otherwise transfers
    count too. direccional=True uses the raw chain instead of the
    normalized one.
    """
    if valor_filtro in (None, "", "Todos"):
        return ""
    sub = _sub_h3_equivalencias(zona_filtro, valor_filtro)
    sufijo = "" if direccional else "_norm"
    cols = [f"h3_inicio{sufijo}", f"h3_fin{sufijo}"]
    if tipo_filtro == "OD y Transferencias":
        cols += [f"h3_transfer1{sufijo}", f"h3_transfer2{sufijo}"]
    partes = " OR ".join(f"c.{col} IN ({sub})" for col in cols)
    return f" AND ({partes})"


def condicion_poligono_sql(id_polygon, filtro_od):
    """SQL condition for analysis-polygon membership.

    filtro_od: "Origen y Destino" (both ends inside — strict cuenca mode),
               "OD y Transferencias" (any chain point inside),
               "Origen o Destino" (either end inside — default).
    """
    if id_polygon in (None, "", "NONE"):
        return ""
    sub = _sub_h3_equivalencias(id_polygon)
    if filtro_od == "Origen y Destino":
        return (
            f" AND c.h3_inicio_norm IN ({sub})"
            f" AND c.h3_fin_norm IN ({sub})"
        )
    if filtro_od == "OD y Transferencias":
        return (
            f" AND (c.h3_inicio_norm IN ({sub})"
            f" OR c.h3_fin_norm IN ({sub})"
            f" OR c.h3_transfer1_norm IN ({sub})"
            f" OR c.h3_transfer2_norm IN ({sub}))"
        )
    return (
        f" AND (c.h3_inicio_norm IN ({sub})"
        f" OR c.h3_fin_norm IN ({sub}))"
    )


def condicion_linea_sql(nombre_linea):
    """SQL condition: seq_lineas contains the line as an exact segment."""
    if nombre_linea in (None, "", "Todas"):
        return ""
    linea = _sql_txt(nombre_linea)
    return (
        f" AND (c.seq_lineas = '{linea}'"
        f" OR c.seq_lineas LIKE '{linea} -- %'"
        f" OR c.seq_lineas LIKE '% -- {linea}'"
        f" OR c.seq_lineas LIKE '% -- {linea} -- %')"
    )


def _sql_media_ponderada(col):
    """fex-weighted mean of a chains_norm column, excluding NULL and 0."""
    valido = f"c.{col} IS NOT NULL AND c.{col} != 0"
    return (
        f"ROUND(COALESCE("
        f"SUM(CASE WHEN {valido} THEN c.{col} * c.factor_expansion_linea ELSE 0 END)"
        f" / NULLIF(SUM(CASE WHEN {valido} THEN c.factor_expansion_linea ELSE 0 END), 0)"
        f", 0), 2)"
    )


def _where_chains(dia_seleccionado=None, where_extra="", condiciones=""):
    where = " WHERE 1=1"
    if dia_seleccionado is not None and dia_seleccionado != "Todos":
        where += f" AND c.dia = '{_sql_txt(dia_seleccionado)}'"
    return where + where_extra + condiciones


_SQL_METRICAS = (
    "SUM(c.factor_expansion_linea) AS factor_expansion_linea, "
    + _sql_media_ponderada("distance_od") + " AS distance_od, "
    + _sql_media_ponderada("travel_time_min") + " AS travel_time_min, "
    + _sql_media_ponderada("travel_speed") + " AS kmh_od"
)

_SQL_DIMS = (
    "c.transferencia, c.modo_agregado, c.rango_hora, c.distancia_agregada, "
    "c.genero_agregado, c.tarifa_agregada"
)


def traer_etapas_matrices_sql(
    zona_seleccionada,
    zonificaciones,
    dia_seleccionado=None,
    where_extra="",
    condiciones="",
    id_polygon="NONE",
    tipo_poligono=None,
):
    """Aggregate chains_norm joined with equivalencias_zonas fully in SQL.

    Returns (etapas_all, matrices_all) already decorated for
    create_data_folium: etapas_all is bidirectional (normalized chain with
    transfers), matrices_all is directional OD. Metrics are fex-weighted
    means computed in the same query.

    When ``id_polygon`` is set, endpoints whose H3 cell falls inside the
    polygon are tagged with a suffix — ' (cuenca)' if ``tipo_poligono`` is
    'cuenca', ' (poligono)' otherwise — so the OD matrix distinguishes
    '<zona>' from '<zona> (poligono|cuenca)' and decorar draws those
    endpoints from the polygon-restricted centroid.
    """
    vacios = (pd.DataFrame([]), pd.DataFrame([]))
    if not asegurar_equivalencias_dash():
        return vacios

    z = _sql_txt(zona_seleccionada)
    where = _where_chains(dia_seleccionado, where_extra, condiciones)

    if id_polygon not in (None, "", "NONE"):
        sub_poly = _sub_h3_equivalencias(id_polygon)
        sufijo = " (cuenca)" if str(tipo_poligono).lower() == "cuenca" else " (poligono)"

        def _con_sufijo(h3_col, id_expr):
            return (
                f"CASE WHEN c.{h3_col} IN ({sub_poly}) "
                f"THEN {id_expr} || '{sufijo}' ELSE {id_expr} END"
            )
    else:
        def _con_sufijo(h3_col, id_expr):
            return id_expr

    expr_inicio_norm = _con_sufijo("h3_inicio_norm", "eq_o.id")
    expr_fin_norm = _con_sufijo("h3_fin_norm", "eq_d.id")
    query_etapas = f"""
        SELECT {expr_inicio_norm} AS inicio_norm,
               COALESCE(eq_t1.id, '') AS transfer1_norm,
               COALESCE(eq_t2.id, '') AS transfer2_norm,
               {expr_fin_norm} AS fin_norm,
               {_SQL_DIMS},
               {_SQL_METRICAS}
        FROM chains_norm c
        JOIN equivalencias_zonas eq_o
            ON c.h3_inicio_norm = eq_o.h3 AND eq_o.zona = '{z}'
        JOIN equivalencias_zonas eq_d
            ON c.h3_fin_norm = eq_d.h3 AND eq_d.zona = '{z}'
        LEFT JOIN equivalencias_zonas eq_t1
            ON c.h3_transfer1_norm = eq_t1.h3 AND eq_t1.zona = '{z}'
        LEFT JOIN equivalencias_zonas eq_t2
            ON c.h3_transfer2_norm = eq_t2.h3 AND eq_t2.zona = '{z}'
        {where}
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    """
    etapas_all = levanto_tabla_sql("chains_norm", "dash", query=query_etapas)

    expr_inicio = _con_sufijo("h3_inicio", "eq_o.id")
    expr_fin = _con_sufijo("h3_fin", "eq_d.id")
    query_matrices = f"""
        SELECT {expr_inicio} AS inicio,
               {expr_fin} AS fin,
               {_SQL_DIMS},
               {_SQL_METRICAS}
        FROM chains_norm c
        JOIN equivalencias_zonas eq_o
            ON c.h3_inicio = eq_o.h3 AND eq_o.zona = '{z}'
        JOIN equivalencias_zonas eq_d
            ON c.h3_fin = eq_d.h3 AND eq_d.zona = '{z}'
        {where}
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
    """
    matrices_all = levanto_tabla_sql("chains_norm", "dash", query=query_matrices)

    return decorar_etapas_matrices(
        etapas_all, matrices_all, zona_seleccionada, zonificaciones, id_polygon
    )


def _explotar_etapas(df, solo_linea=None):
    """One row per leg from chains_norm rows.

    Leg k's origin is chain position k (h3_inicio, h3_transfer1,
    h3_transfer2); its destination is the next position, or h3_fin for the
    last leg. The line of leg k is position k of seq_lineas. Legs whose
    origin equals their destination are dropped (urbantrips_viejo
    convention in etapas_agregadas).
    """
    if len(df) == 0 or "seq_lineas" not in df.columns:
        return pd.DataFrame([])

    seq_vacia = df["seq_lineas"].fillna("") == ""
    seqs = df["seq_lineas"].fillna("").str.split(" -- ")
    n_etapas = seqs.str.len().where(~seq_vacia, 0)

    origenes = ["h3_inicio", "h3_transfer1", "h3_transfer2"]
    siguientes = ["h3_transfer1", "h3_transfer2", None]
    cols_h3 = ["h3_inicio", "h3_transfer1", "h3_transfer2", "h3_fin"]
    attr_cols = [c for c in df.columns if c not in cols_h3 + ["seq_lineas"]]

    partes = []
    for k in range(3):
        o = df[origenes[k]].fillna("")
        if siguientes[k] is None:
            d = df["h3_fin"].fillna("")
        else:
            d = df[siguientes[k]].fillna("")
            d = d.where(d != "", df["h3_fin"].fillna(""))
        linea_k = seqs.str[k]
        mask = (o != "") & (n_etapas > k) & linea_k.notna() & (o != d)
        if solo_linea is not None:
            mask &= linea_k == solo_linea
        if mask.any():
            parte = df.loc[mask, attr_cols].copy()
            parte["h3_o"] = o[mask]
            parte["h3_d"] = d[mask]
            parte["nombre_linea"] = linea_k[mask]
            partes.append(parte)

    if not partes:
        return pd.DataFrame([])
    return pd.concat(partes, ignore_index=True)


def traer_etapas_matrices_linea(
    zona_seleccionada,
    zonificaciones,
    nombre_linea,
    dia_seleccionado=None,
    where_extra="",
    condiciones="",
):
    """Leg-level OD frames for one line (urbantrips_viejo semantics).

    The old line selector mapped the OD pairs of that line's LEGS
    (etapas_agregadas WHERE nombre_linea = x), not whole trips. Here trips
    containing the line are fetched, their legs exploded and only the legs
    of the selected line kept.
    """
    vacios = (pd.DataFrame([]), pd.DataFrame([]))
    if not asegurar_equivalencias_dash():
        return vacios

    where = _where_chains(
        dia_seleccionado, where_extra, condiciones + condicion_linea_sql(nombre_linea)
    )
    query = (
        "SELECT c.h3_inicio, c.h3_transfer1, c.h3_transfer2, c.h3_fin, "
        "c.seq_lineas, c.transferencia, c.modo_agregado, c.rango_hora, "
        "c.distancia_agregada, c.genero_agregado, c.tarifa_agregada, "
        "c.distance_od, c.travel_time_min, c.travel_speed, "
        f"c.factor_expansion_linea FROM chains_norm c{where}"
    )
    df = levanto_tabla_sql("chains_norm", "dash", query=query)
    legs = _explotar_etapas(df, solo_linea=nombre_linea)
    if len(legs) == 0:
        return vacios

    zmap = traer_mapa_zona(zona_seleccionada)
    legs["zona_o"] = legs["h3_o"].map(zmap)
    legs["zona_d"] = legs["h3_d"].map(zmap)
    legs = legs[legs["zona_o"].notna() & legs["zona_d"].notna()].copy()
    if len(legs) == 0:
        return vacios

    dims_filtros = [
        "transferencia", "modo_agregado", "rango_hora",
        "distancia_agregada", "genero_agregado", "tarifa_agregada",
    ]

    # bidirectional pairs for the desire-lines frame
    invertir = legs["zona_o"] > legs["zona_d"]
    legs["inicio_norm"] = legs["zona_o"].where(~invertir, legs["zona_d"])
    legs["fin_norm"] = legs["zona_d"].where(~invertir, legs["zona_o"])
    legs["transfer1_norm"] = ""
    legs["transfer2_norm"] = ""
    etapas_all = _agg_chains_ponderado(
        legs,
        dims_filtros + ["inicio_norm", "transfer1_norm", "transfer2_norm", "fin_norm"],
    )

    # directional pairs for the OD matrix
    legs["inicio"] = legs["zona_o"]
    legs["fin"] = legs["zona_d"]
    matrices_all = _agg_chains_ponderado(legs, dims_filtros + ["inicio", "fin"])

    return decorar_etapas_matrices(
        etapas_all, matrices_all, zona_seleccionada, zonificaciones
    )


@st.cache_data
def traer_h3_zona_valor(zona_filtro, valor_filtro):
    """H3 cell set of one zone of one zoning layer."""
    eq = levanto_tabla_sql(
        "equivalencias_zonas", "insumos",
        query=(
            "SELECT h3 FROM equivalencias_zonas "
            f"WHERE zona = '{_sql_txt(zona_filtro)}' AND id = '{_sql_txt(valor_filtro)}'"
        ),
    )
    return set(eq["h3"]) if len(eq) > 0 else set()


def viajes_con_origen_en_zona(dia_seleccionado, where_extra, zona_filtro, valor_filtro):
    """Trips whose (directional) origin falls in the zone, by modo_agregado.

    Mirrors urbantrips_viejo: viajes_agregados WHERE {zonif}_o = zona.
    """
    sub = _sub_h3_equivalencias(zona_filtro, valor_filtro)
    where = _where_chains(
        dia_seleccionado, where_extra, f" AND c.h3_inicio IN ({sub})"
    )
    query = (
        "SELECT c.modo_agregado, "
        "SUM(c.factor_expansion_linea) AS factor_expansion_linea "
        f"FROM chains_norm c{where} GROUP BY 1"
    )
    return levanto_tabla_sql("chains_norm", "dash", query=query)


def etapas_por_linea_en_zona(dia_seleccionado, where_extra, zona_filtro, valor_filtro):
    """Legs whose (directional) origin falls in the zone, by line.

    Mirrors urbantrips_viejo: etapas_agregadas WHERE {zonif}_o = zona
    grouped by nombre_linea (leg-level counts).
    """
    sub = _sub_h3_equivalencias(zona_filtro, valor_filtro)
    cond = (
        f" AND (c.h3_inicio IN ({sub})"
        f" OR c.h3_transfer1 IN ({sub})"
        f" OR c.h3_transfer2 IN ({sub}))"
    )
    where = _where_chains(dia_seleccionado, where_extra, cond)
    query = (
        "SELECT c.h3_inicio, c.h3_transfer1, c.h3_transfer2, c.h3_fin, "
        "c.seq_lineas, c.factor_expansion_linea "
        f"FROM chains_norm c{where}"
    )
    df = levanto_tabla_sql("chains_norm", "dash", query=query)
    legs = _explotar_etapas(df)
    if len(legs) == 0:
        return pd.DataFrame(columns=["nombre_linea", "factor_expansion_linea"])

    h3_zona = traer_h3_zona_valor(zona_filtro, valor_filtro)
    legs = legs[legs["h3_o"].isin(h3_zona)]
    return (
        legs.groupby("nombre_linea", as_index=False)
        .factor_expansion_linea.sum()
        .sort_values("factor_expansion_linea", ascending=False)
    )


def viajes_entre_zonas_sql(
    dia_seleccionado, where_extra, zonif1, valor1, zonif2, valor2
):
    """Trips with one (directional) end in each filtered zone, labeled
    Zona_1 / Zona_2 by their origin."""
    sub1 = _sub_h3_equivalencias(zonif1, valor1)
    sub2 = _sub_h3_equivalencias(zonif2, valor2)
    cond = (
        f" AND ((c.h3_inicio IN ({sub1}) AND c.h3_fin IN ({sub2}))"
        f" OR (c.h3_inicio IN ({sub2}) AND c.h3_fin IN ({sub1})))"
    )
    where = _where_chains(dia_seleccionado, where_extra, cond)
    query = (
        "SELECT c.h3_inicio, c.h3_fin, c.modo_agregado, c.seq_lineas, "
        "c.transferencia, c.factor_expansion_linea "
        f"FROM chains_norm c{where}"
    )
    df = levanto_tabla_sql("chains_norm", "dash", query=query)
    if len(df) == 0:
        return df

    en_zona1 = df["h3_inicio"].isin(traer_h3_zona_valor(zonif1, valor1))
    df["Zona_1"] = np.where(en_zona1, "Zona 1", "Zona 2")
    df["Zona_2"] = np.where(en_zona1, "Zona 2", "Zona 1")
    return df


def etapas_entre_zonas_sql(
    dia_seleccionado, where_extra, zonif1, valor1, zonif2, valor2
):
    """Legs of the trips that travel between the two filtered zones.

    Trips are selected by their full OD (same condition as
    viajes_entre_zonas_sql) and then exploded into legs, so a trip A->B
    with one transfer counts 1 viaje and 2 etapas regardless of where each
    leg starts or ends. This answers "how many legs are needed to travel
    between the zones" (deliberate departure from urbantrips_viejo, which
    only counted legs whose own OD crossed directly and therefore showed
    empty for zone pairs that require transfers).
    """
    sub1 = _sub_h3_equivalencias(zonif1, valor1)
    sub2 = _sub_h3_equivalencias(zonif2, valor2)
    cond = (
        f" AND ((c.h3_inicio IN ({sub1}) AND c.h3_fin IN ({sub2}))"
        f" OR (c.h3_inicio IN ({sub2}) AND c.h3_fin IN ({sub1})))"
    )
    where = _where_chains(dia_seleccionado, where_extra, cond)
    query = (
        "SELECT c.h3_inicio, c.h3_transfer1, c.h3_transfer2, c.h3_fin, "
        "c.seq_lineas, c.factor_expansion_linea "
        f"FROM chains_norm c{where}"
    )
    df = levanto_tabla_sql("chains_norm", "dash", query=query)
    if len(df) == 0:
        return df

    # direction labeled at trip level so legs inherit it through the explode
    en_zona1 = df["h3_inicio"].isin(traer_h3_zona_valor(zonif1, valor1))
    df["Zona_1"] = np.where(en_zona1, "Zona 1", "Zona 2")
    df["Zona_2"] = np.where(en_zona1, "Zona 2", "Zona 1")
    return _explotar_etapas(df)


# Suffixes that mark a zone endpoint as falling inside the analysis polygon.
_SUFIJOS_POLY = (" (cuenca)", " (poligono)")


def _base_zona(serie):
    """Strip the polygon/basin suffix from zone names."""
    out = serie.astype(str)
    for suf in _SUFIJOS_POLY:
        out = out.str.replace(suf, "", regex=False)
    return out


def _sufijo_zona(serie):
    """Return the polygon/basin suffix present on each zone name ('' if none)."""
    s = serie.astype(str)
    return np.select(
        [s.str.endswith(_SUFIJOS_POLY[0]), s.str.endswith(_SUFIJOS_POLY[1])],
        list(_SUFIJOS_POLY),
        default="",
    )


@st.cache_data
def _coords_zona_poligono(id_polygon, zona_seleccionada):
    """Per-zone coordinates restricted to the analysis polygon.

    Double join over equivalencias_zonas (dash): the H3 cells that belong
    both to the polygon (p.zona = id_polygon) and to each zone of the
    selected layer (z.zona = zona_seleccionada). The coordinate of each zone
    is the mean centroid of those shared cells (h3.cell_to_latlng), so a
    desire line / matrix endpoint flagged '<zona> (poligono|cuenca)' is drawn
    from the polygon-restricted centroid instead of the full-zone one.
    Returns base zone ids; decorar builds the suffixed keys.
    """
    cols = ["id", "lat", "lon"]
    if id_polygon in (None, "", "NONE"):
        return pd.DataFrame(columns=cols)
    query = (
        "SELECT z.id AS id, z.h3 AS h3 "
        "FROM equivalencias_zonas p "
        "JOIN equivalencias_zonas z ON p.h3 = z.h3 "
        f"WHERE p.zona = '{_sql_txt(id_polygon)}' "
        f"AND z.zona = '{_sql_txt(zona_seleccionada)}'"
    )
    df = levanto_tabla_sql("equivalencias_zonas", "dash", query=query)
    if len(df) == 0:
        return pd.DataFrame(columns=cols)

    latlng = df["h3"].apply(
        lambda c: h3.cell_to_latlng(c) if h3.is_valid_cell(c) else (np.nan, np.nan)
    )
    df["lat"] = [p[0] for p in latlng]
    df["lon"] = [p[1] for p in latlng]
    df = df.dropna(subset=["lat", "lon"])
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    return df.groupby("id", as_index=False)[["lat", "lon"]].mean()


def decorar_etapas_matrices(
    etapas_all, matrices_all, zona_seleccionada, zonificaciones, id_polygon="NONE"
):
    """Attach zone coordinates, matrix labels (Origen/Destino) and constant
    columns to the aggregated frames. Shared by the SQL and pandas paths.

    Zone names carrying a ' (poligono)' / ' (cuenca)' suffix are drawn from
    the polygon-restricted centroid (_coords_zona_poligono); plain names use
    the full-zone representative point.
    """
    series_ids = []
    if len(etapas_all) > 0:
        series_ids += [
            etapas_all["inicio_norm"], etapas_all["fin_norm"],
            etapas_all["transfer1_norm"], etapas_all["transfer2_norm"],
        ]
    if len(matrices_all) > 0:
        series_ids += [matrices_all["inicio"], matrices_all["fin"]]
    if not series_ids:
        return etapas_all, matrices_all

    coords = coordenadas_zonas(zonificaciones, zona_seleccionada)
    ids_usados = _base_zona(pd.concat(series_ids))
    coords = _coords_h3_faltantes(ids_usados[ids_usados != ""], coords)
    lat_map = dict(zip(coords["id"], coords["lat"]))
    lon_map = dict(zip(coords["id"], coords["lon"]))
    orden_map = dict(zip(coords["id"], coords["orden_id"]))

    coords_poly = _coords_zona_poligono(id_polygon, zona_seleccionada)
    lat_map_poly = dict(zip(coords_poly["id"], coords_poly["lat"]))
    lon_map_poly = dict(zip(coords_poly["id"], coords_poly["lon"]))

    def _lat_lon(serie_full):
        """(lat, lon) Series: suffixed names use the polygon centroid (with
        fallback to the full-zone one); plain names use the full-zone one."""
        base = _base_zona(serie_full)
        con_sufijo = pd.Series(
            np.asarray(_sufijo_zona(serie_full)) != "", index=serie_full.index
        )
        lat = base.map(lat_map)
        lon = base.map(lon_map)
        lat_p = base.map(lat_map_poly).fillna(lat)
        lon_p = base.map(lon_map_poly).fillna(lon)
        lat = lat.where(~con_sufijo, lat_p).fillna(0)
        lon = lon.where(~con_sufijo, lon_p).fillna(0)
        return lat, lon

    if len(etapas_all) > 0:
        for n, col in enumerate(
            ["inicio_norm", "transfer1_norm", "transfer2_norm", "fin_norm"], start=1
        ):
            lat, lon = _lat_lon(etapas_all[col])
            etapas_all[f"lat{n}_norm"] = lat.values
            etapas_all[f"lon{n}_norm"] = lon.values
        etapas_all["zona"] = zona_seleccionada
        etapas_all["id_polygon"] = id_polygon

    if len(matrices_all) > 0:
        base_inicio = _base_zona(matrices_all["inicio"])
        base_fin = _base_zona(matrices_all["fin"])
        sufijo_inicio = _sufijo_zona(matrices_all["inicio"])
        sufijo_fin = _sufijo_zona(matrices_all["fin"])
        matrices_all["Origen"] = (
            base_inicio.map(orden_map).fillna(base_inicio) + sufijo_inicio
        )
        matrices_all["Destino"] = (
            base_fin.map(orden_map).fillna(base_fin) + sufijo_fin
        )
        lat1, lon1 = _lat_lon(matrices_all["inicio"])
        lat4, lon4 = _lat_lon(matrices_all["fin"])
        matrices_all["lat1"] = lat1.values
        matrices_all["lon1"] = lon1.values
        matrices_all["lat4"] = lat4.values
        matrices_all["lon4"] = lon4.values
        matrices_all["zona"] = zona_seleccionada
        matrices_all["id_polygon"] = id_polygon

    return etapas_all, matrices_all


def armar_etapas_matrices_chains(
    chains,
    zona_seleccionada,
    zonificaciones,
    id_polygon="NONE",
    h3_cuenca=None,
):
    """Aggregate chains_norm rows into the agg_etapas / agg_matrices shapes
    create_data_folium expects.

    - etapas_all: bidirectional (uses *_norm chains) with transfer stops.
    - matrices_all: directional OD (h3_inicio / h3_fin, no *_norm).

    chains_norm carries no distance_od / travel_time_min, so those columns
    are filled with 0 (same convention as the old line-level loader);
    kmh_od is the fex-weighted mean of travel_speed.

    When h3_cuenca is given, zone names whose cell falls inside the basin
    get the ' (cuenca)' suffix (post-processing for tipo 'cuenca').
    """
    cols_etapas_vacias = [
        "id_polygon", "zona", "inicio_norm", "transfer1_norm", "transfer2_norm",
        "fin_norm", "transferencia", "modo_agregado", "rango_hora",
        "distancia_agregada", "genero_agregado", "tarifa_agregada",
        "lat1_norm", "lon1_norm", "lat2_norm", "lon2_norm",
        "lat3_norm", "lon3_norm", "lat4_norm", "lon4_norm",
        "distance_od", "travel_time_min", "kmh_od", "factor_expansion_linea",
    ]
    cols_matrices_vacias = [
        "id_polygon", "zona", "inicio", "fin", "Origen", "Destino",
        "transferencia", "modo_agregado", "rango_hora", "distancia_agregada",
        "genero_agregado", "tarifa_agregada", "lat1", "lon1", "lat4", "lon4",
        "distance_od", "travel_time_min", "kmh_od", "factor_expansion_linea",
    ]
    if len(chains) == 0:
        return (
            pd.DataFrame(columns=cols_etapas_vacias),
            pd.DataFrame(columns=cols_matrices_vacias),
        )

    zmap = traer_mapa_zona(zona_seleccionada)
    if not zmap:
        return (
            pd.DataFrame(columns=cols_etapas_vacias),
            pd.DataFrame(columns=cols_matrices_vacias),
        )

    dims_filtros = [
        "transferencia", "modo_agregado", "rango_hora",
        "distancia_agregada", "genero_agregado", "tarifa_agregada",
    ]

    # ── etapas (bidirectional, with transfers) ──────────────────────────
    e = chains.copy()
    e["inicio_norm"] = e["h3_inicio_norm"].map(zmap)
    e["transfer1_norm"] = e["h3_transfer1_norm"].map(zmap)
    e["transfer2_norm"] = e["h3_transfer2_norm"].map(zmap)
    e["fin_norm"] = e["h3_fin_norm"].map(zmap)

    if h3_cuenca:
        e["inicio_norm"] = _aplicar_sufijo_cuenca(
            e["inicio_norm"], e["h3_inicio_norm"], h3_cuenca
        )
        e["fin_norm"] = _aplicar_sufijo_cuenca(
            e["fin_norm"], e["h3_fin_norm"], h3_cuenca
        )

    e = e[e["inicio_norm"].notna() & e["fin_norm"].notna()].copy()
    e[["transfer1_norm", "transfer2_norm"]] = (
        e[["transfer1_norm", "transfer2_norm"]].fillna("")
    )

    dims_e = dims_filtros + [
        "inicio_norm", "transfer1_norm", "transfer2_norm", "fin_norm"
    ]
    etapas_all = _agg_chains_ponderado(e, dims_e)

    # ── matrices (directional OD) ───────────────────────────────────────
    m = chains.copy()
    m["inicio"] = m["h3_inicio"].map(zmap)
    m["fin"] = m["h3_fin"].map(zmap)

    if h3_cuenca:
        m["inicio"] = _aplicar_sufijo_cuenca(m["inicio"], m["h3_inicio"], h3_cuenca)
        m["fin"] = _aplicar_sufijo_cuenca(m["fin"], m["h3_fin"], h3_cuenca)

    m = m[m["inicio"].notna() & m["fin"].notna()].copy()

    dims_m = dims_filtros + ["inicio", "fin"]
    matrices_all = _agg_chains_ponderado(m, dims_m)

    return decorar_etapas_matrices(
        etapas_all, matrices_all, zona_seleccionada, zonificaciones, id_polygon
    )


def filtrar_chains_por_zona(chains, zona_filtro, valor_filtro, tipo_filtro="OD y Transferencias"):
    """Keep chains whose normalized chain touches one zone of any layer.

    zona_filtro is the zoning layer of the filter, valor_filtro the zone id.
    With 'Solo OD' only origin/destination are checked; otherwise transfers
    count too.
    """
    if len(chains) == 0 or valor_filtro is None:
        return chains
    zmap = traer_mapa_zona(zona_filtro)
    if not zmap:
        return chains.iloc[0:0]

    mask = (
        (chains["h3_inicio_norm"].map(zmap) == valor_filtro)
        | (chains["h3_fin_norm"].map(zmap) == valor_filtro)
    )
    if tipo_filtro == "OD y Transferencias":
        mask = mask | (
            (chains["h3_transfer1_norm"].map(zmap) == valor_filtro)
            | (chains["h3_transfer2_norm"].map(zmap) == valor_filtro)
        )
    return chains[mask]


@st.cache_data
def traer_lineas_chains():
    """Line names available for the line filter (from metadata_lineas)."""
    df = levanto_tabla_sql(
        "metadata_lineas", "insumos",
        query=(
            "SELECT DISTINCT nombre_linea FROM metadata_lineas "
            "WHERE nombre_linea IS NOT NULL ORDER BY nombre_linea"
        ),
    )
    if len(df) == 0:
        return []
    return [v for v in df["nombre_linea"].dropna().tolist() if str(v) != ""]


def filtrar_chains_por_linea(chains, nombre_linea):
    """Keep trips whose seq_lineas includes the given line name."""
    if (
        len(chains) == 0
        or nombre_linea in (None, "", "Todas")
        or "seq_lineas" not in chains.columns
    ):
        return chains
    import re

    patron = r"(?:^| -- )" + re.escape(str(nombre_linea)) + r"(?: -- |$)"
    return chains[
        chains["seq_lineas"].fillna("").str.contains(patron, regex=True)
    ]


def filtrar_chains_por_poligono(chains, id_polygon, tipo_poligono, od_en_poligono=False):
    """Filter chains by analysis polygon membership.

    tipo 'poligono' (or basin with the 'origin OR destination' checkbox on):
    keep trips touching the polygon at either end. tipo 'cuenca' strict mode:
    both ends must fall inside the basin.
    """
    if len(chains) == 0 or id_polygon in (None, "", "NONE"):
        return chains
    h3_poly = traer_h3_poligono(id_polygon)
    if not h3_poly:
        return chains.iloc[0:0]

    en_o = chains["h3_inicio_norm"].isin(h3_poly)
    en_d = chains["h3_fin_norm"].isin(h3_poly)
    if tipo_poligono == "cuenca" and not od_en_poligono:
        return chains[en_o & en_d]
    return chains[en_o | en_d]


@st.cache_data
def get_logo():
    file_logo = str(get_paths().base / "docs" / "urbantrips_logo.jpg")
    if not os.path.isfile(file_logo):
        # URL of the image file on Github
        url = "https://raw.githubusercontent.com/EL-BID/UrbanTrips/main/docs/urbantrips_logo.jpg"

        # Send a request to get the content of the image file
        response = requests.get(url)

        # Save the content to a local file
        with open(file_logo, "wb") as f:
            f.write(response.content)
    image = Image.open(file_logo)
    return image


@st.cache_data
def create_linestring_od(
    df, lat_o="lat_o", lon_o="lon_o", lat_d="lat_d", lon_d="lon_d"
):

    # Create LineString objects from the coordinates
    geometry = [
        LineString([(row[lon_o], row[lat_o]), (row[lon_d], row[lat_d])])
        for _, row in df.iterrows()
    ]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=4326)

    return gdf


def calculate_weighted_means_ods(
    df,
    aggregate_cols,
    weighted_mean_cols,
    weight_col,
    agg_transferencias=False,
    agg_modo=False,
    agg_hora=False,
    agg_distancia=False,
    agg_genero_agregado=False,
    agg_tarifa_agregada=False,
    zero_to_nan=[],
):

    if agg_transferencias:
        df["transferencia"] = 99
    if agg_modo:
        df["modo_agregado"] = 99
    if agg_hora:
        df["rango_hora"] = 99
    if agg_distancia:
        df["distancia_agregada"] = 99
    if agg_genero_agregado:
        df["genero_agregado"] = 99
    if agg_tarifa_agregada:
        df["tarifa_agregada"] = 99

    df = calculate_weighted_means(
        df, aggregate_cols, weighted_mean_cols, weight_col, zero_to_nan
    )
    return df


def agg_matriz(
    df,
    aggregate_cols=[
        "id_polygon",
        "zona",
        "Origen",
        "Destino",
        "transferencia",
        "modo_agregado",
        "rango_hora",
        "distancia_agregada",
        "genero_agregado",
        "tarifa_agregada",
    ],
    weight_col=["distance_od", "travel_time_min", "kmh_od"],
    weight_var="factor_expansion_linea",
    zero_to_nan=["distance_od", "travel_time_min", "kmh_od"],
    agg_transferencias=False,
    agg_modo=False,
    agg_hora=False,
    agg_distancia=False,
    agg_genero_agregado=False,
    agg_tarifa_agregada=False,
):

    if len(df) > 0:
        if agg_transferencias:
            df["transferencia"] = 99
        if agg_modo:
            df["modo_agregado"] = 99
        if agg_hora:
            df["rango_hora"] = 99
        if agg_distancia:
            df["distancia_agregada"] = 99
        if agg_genero_agregado:
            df["genero_agregado"] = 99
        if agg_tarifa_agregada:
            df["tarifa_agregada"] = 99

        df1 = df.groupby(aggregate_cols, as_index=False)[weight_var].sum()

        df2 = calculate_weighted_means(
            df,
            aggregate_cols=aggregate_cols,
            weighted_mean_cols=weight_col,
            weight_col=weight_var,
            zero_to_nan=zero_to_nan,
        )

        if len(df2) > 0:
            df = df1.merge(df2, how="left")
        else:
            df = df1.copy()
            for i in weight_col:
                df[i] = 0

    return df


def creo_bubble_od(
    df,
    aggregate_cols,
    weighted_mean_cols,
    weight_col,
    agg_transferencias=False,
    agg_modo=False,
    agg_hora=False,
    agg_distancia=False,
    agg_genero_agregado=False,
    agg_tarifa_agregada=False,
    od="",
    lat="lat1",
    lon="lon1",
):

    if "id_polygon" not in df.columns:
        df["id_polygon"] = "NONE"

    orig = pd.DataFrame([])
    if len(df) > 0:
        if agg_transferencias:
            df["transferencia"] = 99
        if agg_modo:
            df["modo_agregado"] = 99
        if agg_hora:
            df["rango_hora"] = 99
        if agg_distancia:
            df["distancia_agregada"] = 99
        if agg_genero_agregado:
            df["genero_agregado"] = 99
        if agg_tarifa_agregada:
            df["tarifa_agregada"] = 99


        orig = calculate_weighted_means_ods(
            df,
            aggregate_cols,
            [lat, lon],
            "factor_expansion_linea",
            agg_transferencias=agg_transferencias,
            agg_modo=agg_modo,
            agg_hora=agg_hora,
            agg_distancia=agg_distancia,
            agg_genero_agregado=agg_genero_agregado,
            agg_tarifa_agregada=agg_tarifa_agregada,
        )

        orig["tot"] = orig.groupby(
            [
                "id_polygon",
                "zona",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "distancia_agregada",
                "genero_agregado",
                "tarifa_agregada",
            ]
        ).factor_expansion_linea.transform("sum")
        geometry = [Point(xy) for xy in zip(orig[lon], orig[lat])]
        orig = gpd.GeoDataFrame(orig, geometry=geometry, crs=4326)
        orig["viajes_porc"] = (orig.factor_expansion_linea / orig.tot * 100).round(1)
        orig = orig.rename(columns={od: "od", lat: "lat", lon: "lon"})

    return orig


def df_to_linestrings(df, lat_cols, lon_cols):
    """
    Converts DataFrame rows into LineStrings based on specified lat/lon columns,
    ignoring pairs where either lat or lon is zero.

    Parameters:
    - df: pandas DataFrame containing the data.
    - lat_cols: List of column names for latitudes.
    - lon_cols: List of column names for longitudes.

    Returns:
    - GeoDataFrame with an added 'geometry' column containing LineStrings.
    """

    def create_linestring(row):
        # Filter out coordinate pairs where lat or lon is 0
        points = [
            (row[lon_cols[i]], row[lat_cols[i]])
            for i in range(len(lat_cols))
            if row[lat_cols[i]] != 0 and row[lon_cols[i]] != 0
        ]
        # Create a LineString if there are at least two points
        return LineString(points) if len(points) >= 2 else None

    # Create 'geometry' column with LineStrings
    df["geometry"] = df.apply(create_linestring, axis=1)

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=4326)

    return gdf


def create_data_folium(
    etapas,
    viajes_matrices,
    agg_transferencias=False,
    agg_modo=False,
    agg_hora=False,
    agg_distancia=False,
    agg_genero_agregado=False,
    agg_tarifa_agregada=False,
    agg_cols_etapas=[],
    agg_cols_viajes=[],
    etapas_seleccionada=True,
    viajes_seleccionado=True,
    origenes_seleccionado=True,
    destinos_seleccionado=True,
    transferencias_seleccionado=False,
    mostrar_lineas_principales=True,
):

    if transferencias_seleccionado:

        t1 = etapas.loc[
            etapas.transfer1_norm != "",
            [
                "zona",
                "transfer1_norm",
                "lat2_norm",
                "lon2_norm",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "distancia_agregada",
                "genero_agregado",
                "tarifa_agregada",
                "factor_expansion_linea",
            ],
        ].rename(
            columns={
                "transfer1_norm": "transfer",
                "lat2_norm": "lat_norm",
                "lon2_norm": "lon_norm",
            }
        )
        t2 = etapas.loc[
            etapas.transfer2_norm != "",
            [
                "zona",
                "transfer2_norm",
                "lat3_norm",
                "lon3_norm",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "distancia_agregada",
                "genero_agregado",
                "tarifa_agregada",
                "factor_expansion_linea",
            ],
        ].rename(
            columns={
                "transfer2_norm": "transfer",
                "lat3_norm": "lat_norm",
                "lon3_norm": "lon_norm",
            }
        )
        transferencias = pd.concat([t1, t2], ignore_index=True)
        transferencias["id_polygon"] = "NONE"

        trans_cols_o = [
            "id_polygon",
            "zona",
            "transfer",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            "genero_agregado",
            "tarifa_agregada",
        ]

        transferencias = creo_bubble_od(
            transferencias,
            aggregate_cols=trans_cols_o,
            weighted_mean_cols=["lat_norm", "lon_norm"],
            weight_col="factor_expansion_linea",
            agg_transferencias=agg_transferencias,
            agg_modo=agg_modo,
            agg_hora=agg_hora,
            agg_distancia=agg_distancia,
            agg_genero_agregado=agg_genero_agregado,
            agg_tarifa_agregada=agg_tarifa_agregada,
            od="transfer",
            lat="lat_norm",
            lon="lon_norm",
        )
        if len(transferencias) > 0:
            transferencias["factor_expansion_linea"] = transferencias[
                "factor_expansion_linea"
            ].round(0)
    else:
        transferencias = pd.DataFrame([])

    if etapas_seleccionada | transferencias_seleccionado:

        etapas = calculate_weighted_means_ods(
            etapas,
            agg_cols_etapas,
            [
                "distance_od",
                "lat1_norm",
                "lon1_norm",
                "lat2_norm",
                "lon2_norm",
                "lat3_norm",
                "lon3_norm",
                "lat4_norm",
                "lon4_norm",
            ],
            "factor_expansion_linea",
            agg_transferencias=agg_transferencias,
            agg_modo=agg_modo,
            agg_hora=agg_hora,
            agg_distancia=agg_distancia,
            agg_genero_agregado=agg_genero_agregado,
            agg_tarifa_agregada=agg_tarifa_agregada,
            zero_to_nan=[
                "lat1_norm",
                "lon1_norm",
                "lat2_norm",
                "lon2_norm",
                "lat3_norm",
                "lon3_norm",
                "lat4_norm",
                "lon4_norm",
            ],
        )

        etapas[
            [
                "lat1_norm",
                "lon1_norm",
                "lat2_norm",
                "lon2_norm",
                "lat3_norm",
                "lon3_norm",
                "lat4_norm",
                "lon4_norm",
            ]
        ] = etapas[
            [
                "lat1_norm",
                "lon1_norm",
                "lat2_norm",
                "lon2_norm",
                "lat3_norm",
                "lon3_norm",
                "lat4_norm",
                "lon4_norm",
            ]
        ].fillna(
            0
        )

        etapas = (
            etapas[(etapas.inicio_norm != etapas.fin_norm)]
            .sort_values("factor_expansion_linea", ascending=False)
            .reset_index(drop=True)
            .copy()
        )
        if (len(etapas) >= 2000) & (mostrar_lineas_principales):
            logger.debug("Se muestran las etapas con más viajes")
            etapas = etapas.head(2000)

        etapas["factor_expansion_linea"] = etapas["factor_expansion_linea"].round(0)

        etapas = df_to_linestrings(
            etapas,
            lat_cols=["lat1_norm", "lat2_norm", "lat3_norm", "lat4_norm"],
            lon_cols=["lon1_norm", "lon2_norm", "lon3_norm", "lon4_norm"],
        )

    if viajes_seleccionado:
        viajes = calculate_weighted_means_ods(
            etapas,
            agg_cols_viajes,
            ["distance_od", "lat1_norm", "lon1_norm", "lat4_norm", "lon4_norm"],
            "factor_expansion_linea",
            agg_transferencias=agg_transferencias,
            agg_modo=agg_modo,
            agg_hora=agg_hora,
            agg_distancia=agg_distancia,
            agg_genero_agregado=agg_genero_agregado,
            agg_tarifa_agregada=agg_tarifa_agregada,
            zero_to_nan=["lat1_norm", "lon1_norm", "lat4_norm", "lon4_norm"],
        )

        viajes[["lat1_norm", "lon1_norm", "lat4_norm", "lon4_norm"]] = viajes[
            ["lat1_norm", "lon1_norm", "lat4_norm", "lon4_norm"]
        ].fillna(0)

        if "id_polygon" not in viajes_matrices.columns:
            viajes_matrices["id_polygon"] = "NONE"

        viajes = (
            viajes[(viajes.inicio_norm != viajes.fin_norm)]
            .sort_values("factor_expansion_linea", ascending=False)
            .reset_index(drop=True)
            .copy()
        )
        if (len(etapas) >= 1500) & (mostrar_lineas_principales):
            logger.debug("Se muestran las lineas con más viajes")
            viajes = viajes.head(1500)

        # viajes = viajes[viajes.inicio_norm != viajes.fin_norm].copy()
        viajes["factor_expansion_linea"] = viajes["factor_expansion_linea"].round(0)

        viajes = df_to_linestrings(
            viajes,
            lat_cols=["lat1_norm", "lat4_norm"],
            lon_cols=["lon1_norm", "lon4_norm"],
        )

    else:
        viajes = pd.DataFrame([])

    matriz = agg_matriz(
        viajes_matrices,
        aggregate_cols=[
            "id_polygon",
            "zona",
            "Origen",
            "Destino",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            "genero_agregado",
            "tarifa_agregada",
        ],
        weight_col=["distance_od", "travel_time_min", "kmh_od"],
        zero_to_nan=["distance_od", "travel_time_min", "kmh_od"],
        weight_var="factor_expansion_linea",
        agg_transferencias=agg_transferencias,
        agg_modo=agg_modo,
        agg_hora=agg_hora,
        agg_distancia=agg_distancia,
        agg_genero_agregado=agg_genero_agregado,
        agg_tarifa_agregada=agg_tarifa_agregada,
    )

    matriz["factor_expansion_linea"] = matriz["factor_expansion_linea"].round(0)
    matriz = matriz.sort_values("factor_expansion_linea", ascending=False).reset_index(
        drop=True
    )
    matriz["porcentaje"] = (
        matriz["factor_expansion_linea"] / matriz["factor_expansion_linea"].sum() * 100
    ).round(2)
    matriz["resumen"] = 0
    matriz.loc[0:20, "resumen"] = 1
    lst_resumen = (
        matriz[matriz.resumen == 1].Origen.unique().tolist()
        + matriz[matriz.resumen == 1].Destino.unique().tolist()
    )
    matriz.loc[
        (matriz.Origen.isin(lst_resumen)) & (matriz.Destino.isin(lst_resumen)),
        "resumen",
    ] = 1

    if ("poly_inicio" in viajes_matrices.columns) | (
        "poly_fin" in viajes_matrices.columns
    ):
        bubble_cols_o = [
            "id_polygon",
            "zona",
            "inicio",
            "poly_inicio",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            "tarifa_agregada",
            "genero_agregado",
        ]
        bubble_cols_d = [
            "id_polygon",
            "zona",
            "fin",
            "poly_fin",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            "tarifa_agregada",
            "genero_agregado",
        ]
    else:
        bubble_cols_o = [
            "id_polygon",
            "zona",
            "inicio",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            'tarifa_agregada',
            "genero_agregado",            
        ]
        bubble_cols_d = [
            "id_polygon",
            "zona",
            "fin",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            "tarifa_agregada",
            "genero_agregado",
        ]

    if origenes_seleccionado:
        origen = creo_bubble_od(
            viajes_matrices,
            aggregate_cols=bubble_cols_o,
            weighted_mean_cols=["lat1", "lon1"],
            weight_col="factor_expansion_linea",
            agg_transferencias=agg_transferencias,
            agg_modo=agg_modo,
            agg_hora=agg_hora,
            agg_distancia=agg_distancia,
            agg_genero_agregado=agg_genero_agregado,
            agg_tarifa_agregada=agg_tarifa_agregada,
            od="inicio",
            lat="lat1",
            lon="lon1",
        )
        origen["factor_expansion_linea"] = origen["factor_expansion_linea"].round()
    else:
        origen = pd.DataFrame([])

    if destinos_seleccionado:
        destino = creo_bubble_od(
            viajes_matrices,
            aggregate_cols=bubble_cols_d,
            weighted_mean_cols=["lat4", "lon4"],
            weight_col="factor_expansion_linea",
            agg_transferencias=agg_transferencias,
            agg_modo=agg_modo,
            agg_hora=agg_hora,
            agg_distancia=agg_distancia,
            agg_genero_agregado=agg_genero_agregado,
            agg_tarifa_agregada=agg_tarifa_agregada,
            od="fin",
            lat="lat4",
            lon="lon4",
        )
        destino["factor_expansion_linea"] = destino["factor_expansion_linea"].round(0)
    else:
        destino = pd.DataFrame([])

    if not etapas_seleccionada:
        etapas = pd.DataFrame([])

    return etapas, viajes, matriz, origen, destino, transferencias


@st.cache_data
def traigo_indicadores(tipo="all"):
    if tipo == "all":
        indicadores_all = levanto_tabla_sql("agg_indicadores")
    else:
        indicadores_all = levanto_tabla_sql("poly_indicadores")

    general = indicadores_all[indicadores_all.Tipo == "General"]
    modal = indicadores_all[indicadores_all.Tipo == "Modal"]
    distancias = indicadores_all[indicadores_all.Tipo == "Distancias"]
    return general, modal, distancias


def get_epsg_m():
    """
    Gets the epsg id for a coordinate reference system in meters from config
    """
    configs = leer_configs_generales()
    epsg_m = configs["epsg_m"]

    return epsg_m


def create_squared_polygon(min_x, min_y, max_x, max_y, epsg):

    width = max(max_x - min_x, max_y - min_y)
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    square_bbox_min_x = center_x - width / 2
    square_bbox_min_y = center_y - width / 2
    square_bbox_max_x = center_x + width / 2
    square_bbox_max_y = center_y + width / 2

    square_bbox_coords = [
        (square_bbox_min_x, square_bbox_min_y),
        (square_bbox_max_x, square_bbox_min_y),
        (square_bbox_max_x, square_bbox_max_y),
        (square_bbox_min_x, square_bbox_max_y),
    ]

    p = Polygon(square_bbox_coords)
    s = gpd.GeoSeries([p], crs=f"EPSG:{epsg}")
    return s


def extract_hex_colors_from_cmap(cmap, n=5):
    # Choose a colormap
    cmap = plt.get_cmap(cmap)

    # Extract colors from the colormap
    colors = cmap(np.linspace(0, 1, n))

    # Convert the colors to hex format
    hex_colors = [mcolors.rgb2hex(color) for color in colors]

    return hex_colors


@st.cache_data
def bring_latlon():
    """Map center: mean representative point of the zoning layers."""
    try:
        zonif = levanto_tabla_sql("zonificaciones", "dash")
        puntos = zonif.geometry.representative_point()
        latlon = [puntos.y.mean(), puntos.x.mean()]
    except Exception:
        latlon = [-32.891401, -68.843242]
    return latlon


@st.cache_data
def traigo_lista_zonas(tipo="etapas"):
    """Zone names per zoning layer, from the zonificaciones table.

    Replaces the old lookup over agg_etapas / poly_etapas (removed): zones
    now come straight from the zoning layers used to build
    equivalencias_zonas.
    """
    zonas_values = levanto_tabla_sql(
        "zonificaciones",
        "insumos",
        "SELECT DISTINCT zona, id FROM zonificaciones;",
    )
    if len(zonas_values) == 0:
        return pd.DataFrame(columns=["zona", "Nombre"])

    zonas_values = (
        zonas_values[
            (zonas_values.id.notna()) & (zonas_values.id.astype(str) != "")
        ]
        .sort_values(["zona", "id"])
        .rename(columns={"id": "Nombre"})
    )

    return zonas_values[["zona", "Nombre"]]


def normalizar_zonas(df, inicio_col, lat1_col, lon1_col, fin_col, lat2_col, lon2_col):
    """
    Normaliza las zonas para que los pares inicio/fin siempre estén ordenados de forma consistente,
    dejando sin cambios los registros donde inicio_col o fin_col estén vacíos (="").
    """
    # Máscara para identificar registros válidos (sin valores vacíos)
    mask_valid = (df[inicio_col] != "") & (df[fin_col] != "")

    # Máscara para el orden correcto (solo en registros válidos)
    mask_order = mask_valid & (df[inicio_col] < df[fin_col])

    # Asignar valores normalizados columna por columna
    df[f"{inicio_col}_norm"] = np.where(
        mask_valid, np.where(mask_order, df[inicio_col], df[fin_col]), df[inicio_col]
    )
    df[f"{lat1_col}_norm"] = np.where(
        mask_valid, np.where(mask_order, df[lat1_col], df[lat2_col]), df[lat1_col]
    )
    df[f"{lon1_col}_norm"] = np.where(
        mask_valid, np.where(mask_order, df[lon1_col], df[lon2_col]), df[lon1_col]
    )
    df[f"{fin_col}_norm"] = np.where(
        mask_valid, np.where(mask_order, df[fin_col], df[inicio_col]), df[fin_col]
    )
    df[f"{lat2_col}_norm"] = np.where(
        mask_valid, np.where(mask_order, df[lat2_col], df[lat1_col]), df[lat2_col]
    )
    df[f"{lon2_col}_norm"] = np.where(
        mask_valid, np.where(mask_order, df[lon2_col], df[lon1_col]), df[lon2_col]
    )

    return df


def traigo_tablas_con_filtros(
    dia,
    var_zonif,
    var_filtro1,
    det_filtro1,
    var_filtro2,
    det_filtro2,
    tipo_filtro,
    zonas,
    zonificaciones,
):

    lst1 = zonas[zonas[var_filtro1] == det_filtro1][var_zonif].unique().tolist()
    lst2 = zonas[zonas[var_filtro2] == det_filtro2][var_zonif].unique().tolist()

    zonas = zonas.groupby([var_zonif], as_index=False)[["latitud", "longitud"]].mean()

    conn = iniciar_conexion_db(tipo="dash")

    # Crear marcadores de posición para SQL
    placeholders1 = ", ".join(["?"] * len(lst1))  # Para lista origen
    placeholders2 = ", ".join(["?"] * len(lst2))  # Para lista destino

    # Parámetros de la consulta

    # Consulta SQL
    if tipo_filtro == "OD y Transferencias":

        if det_filtro1 != det_filtro2:
            if (det_filtro1 != "Todos") & (det_filtro2 != "Todos"):
                query = f"""
                SELECT * FROM agg_etapas 
                WHERE zona = ?
                AND dia = ? 
                AND (
                    (inicio_norm IN ({placeholders1}) OR transfer1_norm IN ({placeholders1}) OR transfer2_norm IN ({placeholders1}) OR fin_norm IN ({placeholders1}))
                    AND 
                    (inicio_norm IN ({placeholders2}) OR transfer1_norm IN ({placeholders2}) OR transfer2_norm IN ({placeholders2}) OR fin_norm IN ({placeholders2}))
                );
                """
                params = [var_zonif, dia] + lst1 * 4 + lst2 * 4
            elif (det_filtro1 != "Todos") & (det_filtro2 == "Todos"):
                query = f"""
                SELECT * FROM agg_etapas 
                WHERE zona = ?
                AND dia = ? 
                AND (
                    (inicio_norm IN ({placeholders1}) OR transfer1_norm IN ({placeholders1}) OR transfer2_norm IN ({placeholders1}) OR fin_norm IN ({placeholders1}))
                    ) 
                ;
                """
                params = [var_zonif, dia] + lst1 * 4
            elif (det_filtro1 == "Todos") & (det_filtro2 != "Todos"):
                query = f"""
                SELECT * FROM agg_etapas 
                WHERE zona = ?
                AND dia = ? 
                AND (                    
                    (inicio_norm IN ({placeholders2}) OR transfer1_norm IN ({placeholders2}) OR transfer2_norm IN ({placeholders2}) OR fin_norm IN ({placeholders2}))
                    )
                ;
                """
                params = [var_zonif, dia] + lst2 * 4
        else:
            query = f"""
            SELECT * FROM agg_etapas 
            WHERE zona = ?
            AND dia = ? 
            AND (
                    (CASE WHEN inicio_norm IN ({placeholders1}) THEN 1 ELSE 0 END) +
                    (CASE WHEN transfer1_norm IN ({placeholders1}) THEN 1 ELSE 0 END) +
                    (CASE WHEN transfer2_norm IN ({placeholders1}) THEN 1 ELSE 0 END) +
                    (CASE WHEN fin_norm IN ({placeholders1}) THEN 1 ELSE 0 END)
                ) >= 2;
            """
            params = [var_zonif, dia] + lst1 * 4

    else:
        if det_filtro1 != det_filtro2:
            if (det_filtro1 != "Todos") & (det_filtro2 != "Todos"):
                query = f"""
                SELECT * FROM agg_etapas 
                WHERE zona = ?
                AND dia = ? 
                AND (
                    (inicio_norm IN ({placeholders1}) OR fin_norm IN ({placeholders1}))
                    AND 
                    (inicio_norm IN ({placeholders2}) OR fin_norm IN ({placeholders2}))
                );
                """
                params = [var_zonif, dia] + lst1 * 2 + lst2 * 2
            elif (det_filtro1 != "Todos") & (det_filtro2 == "Todos"):
                query = f"""
                SELECT * FROM agg_etapas 
                WHERE zona = ?
                AND dia = ? 
                AND (
                    (inicio_norm IN ({placeholders1}) OR fin_norm IN ({placeholders1}))                
                );
                """
                params = [var_zonif, dia] + lst1 * 2

            elif (det_filtro1 == "Todos") & (det_filtro2 != "Todos"):
                query = f"""
                SELECT * FROM agg_etapas 
                WHERE zona = ?
                AND dia = ? 
                AND (
                    (inicio_norm IN ({placeholders2}) OR fin_norm IN ({placeholders2}))
                );
                """
                params = [var_zonif, dia] + lst2 * 2

        else:
            query = f"""
            SELECT * FROM agg_etapas 
            WHERE zona = ?
            AND dia = ? 
            AND (
                    (CASE WHEN inicio_norm IN ({placeholders1}) THEN 1 ELSE 0 END) +
                    (CASE WHEN fin_norm IN ({placeholders1}) THEN 1 ELSE 0 END)
                ) >= 2;
            """
            params = [var_zonif, dia] + lst1 * 2

    # Ejecutar consulta

    agg_etapas = _fetch_sql_dataframe(conn, query, params=params)

    if len(agg_etapas) > 0:
        zonas_renamed = zonas[[var_zonif, "latitud", "longitud"]]
        for i, z in enumerate(["inicio", "transfer1", "transfer2", "fin"], start=1):

            zonas_temp = zonas_renamed.rename(
                columns={
                    var_zonif: f"{z}_norm",
                    "latitud": f"lat{i}",
                    "longitud": f"lon{i}",
                }
            )
            zonas_temp[z] = zonas_temp[f"{z}_norm"]
            agg_etapas = agg_etapas.merge(zonas_temp, how="left")
            agg_etapas[f"{z}"] = agg_etapas[f"{z}"].fillna("")

        # Filtros innecesarios en un solo paso
        agg_etapas = agg_etapas[
            ~(
                ((agg_etapas.inicio == "") & (agg_etapas.inicio_norm != ""))
                | ((agg_etapas.fin == "") & (agg_etapas.fin_norm != ""))
                | ((agg_etapas.transfer1 == "") & (agg_etapas.transfer1_norm != ""))
                | ((agg_etapas.transfer2 == "") & (agg_etapas.transfer2_norm != ""))
            )
        ]

        aggregate_cols = [
            "dia",
            "inicio",
            "transfer1",
            "transfer2",
            "fin",
            "zona",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "genero_agregado",
            "tarifa_agregada",
            "coincidencias",
            "distancia_agregada",
        ]
        weighted_mean_cols = [
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "lat1",
            "lon1",
            "lat2",
            "lon2",
            "lat3",
            "lon3",
            "lat4",
            "lon4",
        ]
        zero_to_nan = [
            "lat1",
            "lon1",
            "lat2",
            "lon2",
            "lat3",
            "lon3",
            "lat4",
            "lon4",
            "distance_od",
            "travel_time_min",
            "kmh_od",
        ]

        agg_etapas = calculate_weighted_means(
            agg_etapas,
            aggregate_cols=aggregate_cols,
            weighted_mean_cols=weighted_mean_cols,
            weight_col="factor_expansion_linea",
            zero_to_nan=zero_to_nan,
            var_fex_summed=False,
        )

        agg_etapas = normalizar_zonas(
            agg_etapas, "inicio", "lat1", "lon1", "fin", "lat4", "lon4"
        )
        agg_etapas = normalizar_zonas(
            agg_etapas, "transfer1", "lat2", "lon2", "transfer2", "lat3", "lon3"
        )

        agg_etapas["zona"] = var_zonif

    # Crear una lista de valores para la cláusula IN de forma segura
    placeholders1 = ", ".join(["?"] * len(lst1))
    placeholders2 = ", ".join(["?"] * len(lst2))

    if det_filtro1 != det_filtro2:
        if (det_filtro1 != "Todos") & (det_filtro2 != "Todos"):
            query = f"""
            SELECT * FROM agg_matrices 
            WHERE zona = ?
            AND dia = ? 
                AND (
                (inicio IN ({placeholders1}) OR fin IN ({placeholders1}))
                AND 
                (inicio IN ({placeholders2}) OR fin IN ({placeholders2}))
            );
            """
            params = [var_zonif, dia] + lst1 * 2 + lst2 * 2
        elif (det_filtro1 != "Todos") & (det_filtro2 == "Todos"):
            query = f"""
            SELECT * FROM agg_matrices 
            WHERE zona = ?
            AND dia = ? 
                AND (
                (inicio IN ({placeholders1}) OR fin IN ({placeholders1}))
                )
            ;
            """
            params = [var_zonif, dia] + lst1 * 2

        elif (det_filtro1 == "Todos") & (det_filtro2 != "Todos"):

            query = f"""
            SELECT * FROM agg_matrices 
            WHERE zona = ?
            AND dia = ? 
                AND 
                (inicio IN ({placeholders2}) OR fin IN ({placeholders2}))
                )
            ;
            """
            params = [var_zonif, dia] + lst2 * 2

    else:
        query = f"""
        SELECT * FROM agg_matrices 
        WHERE zona = ?
        AND dia = ? 
        AND (
                (CASE WHEN inicio IN ({placeholders1}) THEN 1 ELSE 0 END) +
                (CASE WHEN fin IN ({placeholders1}) THEN 1 ELSE 0 END)
            ) >= 2;
        """
        params = [var_zonif, dia] + lst1 * 2

    agg_matrices = _fetch_sql_dataframe(conn, query, params=params)

    if len(agg_matrices) > 0:
        zonas_renamed = zonas[[var_zonif, "latitud", "longitud"]]
        for i, z in enumerate(["inicio", "fin"], start=1):
            zonas_temp = zonas_renamed.rename(
                columns={
                    "latitud": f"lat{i}_new",
                    "longitud": f"lon{i}_new",
                    var_zonif: f"{z}_new",
                }
            )

            zonas_temp[z] = zonas_temp[f"{z}_new"]
            agg_matrices = agg_matrices.merge(zonas_temp, how="left")
            agg_matrices[z] = agg_matrices[z].fillna("")

        agg_matrices = agg_matrices.drop(
            ["inicio", "fin", "lat1", "lon1", "lat4", "lon4"], axis=1
        )
        agg_matrices = agg_matrices.rename(
            columns={
                "inicio_new": "inicio",
                "fin_new": "fin",
                "lat1_new": "lat1",
                "lon1_new": "lon1",
                "lat2_new": "lat4",
                "lon2_new": "lon4",
            }
        )

        agg_matrices = agg_matrices.merge(
            zonificaciones[["id", "orden"]].rename(
                columns={"id": "inicio", "orden": "orden_inicio"}
            )
        )
        agg_matrices = agg_matrices.merge(
            zonificaciones[["id", "orden"]].rename(
                columns={"id": "fin", "orden": "orden_fin"}
            )
        )

        agg_matrices["orden_inicio"] = (
            pd.to_numeric(agg_matrices["orden_inicio"], errors="coerce")
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
            .astype(int)
        )

        agg_matrices["orden_fin"] = (
            pd.to_numeric(agg_matrices["orden_fin"], errors="coerce")
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
            .astype(int)
        )

        # Construcción de columnas Origen y Destino
        agg_matrices["Origen"] = (
            agg_matrices["orden_inicio"].astype(str).str.zfill(3)
            + "_"
            + agg_matrices["inicio"]
        )

        agg_matrices["Destino"] = (
            agg_matrices["orden_fin"].astype(str).str.zfill(3)
            + "_"
            + agg_matrices["fin"]
        )
        agg_matrices = agg_matrices.drop(["orden_inicio", "orden_fin"], axis=1)

    conn.close()

    return agg_etapas, agg_matrices


@st.cache_data
def traer_dias_disponibles():
    try:
        corridas = levanto_tabla_sql(
            "corridas", "general", query="select corrida from corridas"
        ).corrida.values.tolist()
        if corridas:
            return corridas
    except Exception:
        pass
    configs = leer_configs_generales(autogenerado=False)
    return configs.get("corridas", [])


def configurar_selector_dia():

    # dias_disponibles = traer_dias_disponibles()

    # if len(dias_disponibles) > 1:

    #     # Inicialización una única vez
    #     if "dia_seleccionado" not in st.session_state:
    #         st.session_state.dia_seleccionado = dias_disponibles[0]
    #         st.session_state.dia_anterior = dias_disponibles[0]

    #     # Sidebar con lógica aislada, sin pisar valores
    #     with st.sidebar:
    #         seleccion = st.selectbox(
    #             "Seleccioná un día",
    #             dias_disponibles,
    #             index=dias_disponibles.index(st.session_state.dia_seleccionado),
    #             key="__selector_dia",  # distinto del nombre en session_state
    #         )

    #     # Si la selección cambió, actualizar estado y reiniciar app
    #     if seleccion != st.session_state.dia_anterior:
    #         st.session_state.dia_seleccionado = seleccion
    #         st.session_state.dia_anterior = seleccion
    #         st.cache_data.clear()
    #         st.rerun()
    # else:
    #     seleccion = dias_disponibles[0]

    # base_path = Path() / 'configs'
    # autogen_dir = base_path / "autogenerados"
    # archivo_autogen = autogen_dir /  f"configuraciones_generales_autogenerado_{seleccion}.yaml"
    
    # # Verificar que existan el directorio y el archivo
    # if autogen_dir.exists() and archivo_autogen.exists():
    #     destino = base_path / "configuraciones_generales_autogenerado.yaml"
    #     shutil.copy(archivo_autogen, destino)
    #     logger.info("Archivo %s copiado", archivo_autogen)
    # else:
    #     logger.warning("No existe el directorio 'autogenerados' o el archivo especificado.")
    seleccion = ''
    return seleccion

def tabla_existe(conn, table_name):
    try:
        conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        return True
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return False
        else:
            raise

def guardar_tabla_sql(
    df, table_name, tabla_tipo="dash", filtros=None, alias_db="", modo="append"
):
    """
    Guarda un DataFrame en una base de datos SQLite.

    Parámetros:
    df (pd.DataFrame): DataFrame que se desea guardar.
    table_name (str): Nombre de la tabla en la base de datos.
    tabla_tipo (str): Tipo de conexión a la base de datos.
    alias_db (str): Alias para identificar el archivo de base de datos.
    filtros (dict, optional): Filtros para eliminar registros si modo='append'. Las claves son los nombres
                               de los campos y los valores pueden ser un valor unico o una lista de valores.
    modo (str): 'append' para agregar registros o 'replace' para reemplazar la tabla completa.
    """
    # Conectar a la base de datos
    if alias_db and not alias_db.endswith("_"):
        alias_db += "_"

    conn = iniciar_conexion_db(tipo=tabla_tipo, alias_db=alias_db)
    cursor = conn.cursor()

    if modo == "replace":
        # Reemplaza completamente la tabla
        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=40,
        )

    else:
        table_exists = tabla_existe(conn, table_name)

        # Si la tabla existe y se han proporcionado filtros, elimina los registros que coincidan
        if table_exists and filtros:
            condiciones = []
            valores = []

            for campo, valor in filtros.items():
                if isinstance(valor, list):
                    condiciones.append(f"{campo} IN ({','.join(['?'] * len(valor))})")
                    valores.extend(valor)
                else:
                    condiciones.append(f"{campo} = ?")
                    valores.append(valor)

            where_clause = " AND ".join(condiciones)
            cursor.execute(f"DELETE FROM {table_name} WHERE {where_clause}", valores)
            conn.commit()

        # Agrega los datos al final de la tabla
        df.to_sql(
            table_name,
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=40,
        )

    # Cierra conexión
    cursor.close()
    conn.close()


# Convert geometry to H3 indices
def get_h3_indices_in_geometry(geometry, resolution):
    poly = h3.geo_to_h3shape(geometry)
    h3_indices = list(h3.h3shape_to_cells(poly, res=resolution))

    return h3_indices


def h3_to_polygon(h3_index):
    # Obtener las coordenadas del hexágono
    geom = shape(h3.cells_to_h3shape([h3_index]).__geo_interface__)
    return geom


def _hex_to_rgba(hex_color: str, alpha: int = 200) -> list:
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


def obtener_clases_fisherjenks(
    df: pd.DataFrame, var_fex: str, max_clases: int = 5, min_clases: int = 1
):
    unique_values = df[var_fex].nunique()
    k = min(max_clases, max(min_clases, unique_values))
    while k >= min_clases:
        try:
            bins = [df[var_fex].min() - 1] + mapclassify.FisherJenks(
                df[var_fex], k=k
            ).bins.tolist()
            return bins
        except ValueError:
            k -= 1
    return [df[var_fex].min() - 1, df[var_fex].max()]


def simplificar_geometrias(df: pd.DataFrame, tolerance: float = 0.001):
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].simplify(tolerance, preserve_topology=True)
    return df


def crear_mapa_lineas_deseo(
    df_viajes: pd.DataFrame,
    df_etapas: pd.DataFrame,
    zonif,
    origenes: pd.DataFrame,
    destinos: pd.DataFrame,
    transferencias: pd.DataFrame,
    var_fex: str,
    cmap_viajes: str = "viridis_r",
    cmap_etapas: str = "magma_r",
    cmap_puntos: str = "YlOrRd",
    map_title: str = "",
    savefile: str = "",
    k_jenks: int = 5,
    latlon: list = None,
    tipo_visualizacion: str = "Líneas",
    poly=None,
    show_poly: bool = False,
):
    """Crea mapa interactivo pydeck (WebGL) para viajes, etapas, orígenes, destinos
    y transferencias.  poly/show_poly dibujan el polígono de análisis como
    GeoJsonLayer gris debajo de las líneas."""
    if not _PYDECK_AVAILABLE:
        return None

    if len(df_viajes) > 0:
        df_viajes = df_viajes[df_viajes["geometry"].notna()]
    if len(df_etapas) > 0:
        df_etapas = df_etapas[df_etapas["geometry"].notna()]

    df_etapas, df_viajes, origenes, destinos, transferencias = [
        simplificar_geometrias(df)
        for df in [df_etapas, df_viajes, origenes, destinos, transferencias]
    ]

    layers = []

    def agregar_capa_lineas(df, nombre, var_fex, cmap, weight_base=0.5):
        if len(df) == 0:
            return
        bins = obtener_clases_fisherjenks(df, var_fex)
        n_bins = len(bins) - 1
        colors_hex = extract_hex_colors_from_cmap(cmap="viridis_r", n=k_jenks)
        df = df.copy()
        df["_bin"] = (
            pd.cut(df[var_fex], bins=bins, labels=False, include_lowest=True)
            .fillna(0).astype(int).clip(0, n_bins - 1)
        )
        color_map = {i: _hex_to_rgba(colors_hex[i], 255) for i in range(n_bins)}
        opacity_map = {
            i: round(0.08 + (i / max(n_bins - 1, 1)) ** 2 * 0.92, 3)
            for i in range(n_bins)
        }
        df["color"] = df["_bin"].map(color_map)
        df["label"] = nombre
        DEPTH_OFF = {"depthTest": False}

        if tipo_visualizacion == "Arcos":
            df["_segments"] = df.geometry.apply(
                lambda g: [
                    (g.coords[i], g.coords[i + 1])
                    for i in range(len(g.coords) - 1)
                ]
            )
            df = df.explode("_segments", ignore_index=True)
            df = df[df["_segments"].notna()]
            df["src_lon"] = df["_segments"].apply(lambda s: round(s[0][0], 6))
            df["src_lat"] = df["_segments"].apply(lambda s: round(s[0][1], 6))
            df["tgt_lon"] = df["_segments"].apply(lambda s: round(s[1][0], 6))
            df["tgt_lat"] = df["_segments"].apply(lambda s: round(s[1][1], 6))
            src_color_map = {i: _hex_to_rgba(colors_hex[i], 60) for i in range(n_bins)}
            df["color_src"] = df["_bin"].map(src_color_map)
            width_map = {i: max(3, int(weight_base + i * 3.5)) for i in range(n_bins)}
            df["width"] = df["_bin"].map(width_map)
            cols = ["src_lon", "src_lat", "tgt_lon", "tgt_lat",
                    "color_src", "color", "width", "_bin", "label", var_fex]
            pdk_df = df[cols].copy()
            for bin_idx in range(n_bins):
                subset = pdk_df[pdk_df["_bin"] == bin_idx]
                if len(subset) == 0:
                    continue
                layers.append(pdk.Layer(
                    "ArcLayer", subset,
                    get_source_position=["src_lon", "src_lat"],
                    get_target_position=["tgt_lon", "tgt_lat"],
                    get_source_color="color_src",
                    get_target_color="color",
                    get_width="width",
                    opacity=opacity_map[bin_idx],
                    parameters=DEPTH_OFF,
                    pickable=True, auto_highlight=True,
                ))
        else:
            width_map = {i: max(80, int((weight_base + i * 3) * 120)) for i in range(n_bins)}
            df["width"] = df["_bin"].map(width_map)
            df["path"] = df.geometry.apply(
                lambda g: [[round(p[0], 6), round(p[1], 6)] for p in g.coords]
            )
            cols = ["path", "color", "width", "_bin", "label", var_fex]
            pdk_df = df[cols].copy()
            for bin_idx in range(n_bins):
                subset = pdk_df[pdk_df["_bin"] == bin_idx]
                if len(subset) == 0:
                    continue
                layers.append(pdk.Layer(
                    "PathLayer", subset,
                    get_path="path", get_color="color", get_width="width",
                    width_min_pixels=1,
                    opacity=opacity_map[bin_idx],
                    parameters=DEPTH_OFF,
                    pickable=True, auto_highlight=True,
                ))

    def agregar_capa_puntos(df, nombre, var_fex, cmap):
        if len(df) == 0:
            return
        colors_hex = extract_hex_colors_from_cmap(cmap=cmap, n=10)
        df = df.copy()
        max_val = df[var_fex].max()
        min_val = df[var_fex].min()
        if max_val > min_val:
            df["_cidx"] = (
                (df[var_fex] - min_val) / (max_val - min_val) * 9
            ).astype(int).clip(0, 9)
        else:
            df["_cidx"] = 0
        df["color"] = df["_cidx"].apply(lambda i: _hex_to_rgba(colors_hex[i]))
        df["lon"] = df.geometry.x
        df["lat"] = df.geometry.y
        df["radius"] = 300 + (df[var_fex] / max_val) * 1500
        df["label"] = nombre
        pdk_df = df[["lon", "lat", "color", "radius", "label", var_fex]].copy()
        layers.append(pdk.Layer(
            "ScatterplotLayer", pdk_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=5,
            pickable=True, auto_highlight=True,
        ))

    agregar_capa_lineas(df_etapas, "Etapas", var_fex, cmap_etapas, weight_base=0.5)
    agregar_capa_lineas(df_viajes, "Viajes", var_fex, cmap_viajes, weight_base=0.5)
    agregar_capa_puntos(origenes, "Orígenes", var_fex, cmap_puntos)
    agregar_capa_puntos(destinos, "Destinos", var_fex, cmap_puntos)
    agregar_capa_puntos(transferencias, "Transferencias", var_fex, cmap_puntos)

    if isinstance(zonif, pd.DataFrame) and len(zonif) > 0:
        zonif_json = json.loads(zonif.to_json())
        for feature in zonif_json.get("features", []):
            props = feature.get("properties", {})
            props["label"] = props.get("id", "")
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=zonif_json, id="zonas",
            stroked=True, filled=True,
            get_fill_color=[0, 0, 255, 0],
            get_line_color=[0, 0, 128, 180],
            line_width_min_pixels=1,
            pickable=True, auto_highlight=True,
        ))

    if show_poly and poly is not None and hasattr(poly, "geometry") and len(poly) > 0:
        poly_json = json.loads(poly.to_json())
        for feature in poly_json.get("features", []):
            props = feature.get("properties", {})
            props["label"] = props.get("id", "")
        layers = [pdk.Layer(
            "GeoJsonLayer", data=poly_json, id="poligono",
            stroked=True, filled=True,
            get_fill_color=[128, 128, 128, 60],
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=2,
            pickable=True, auto_highlight=True,
        )] + layers

    if not layers:
        return None

    if latlon is None:
        latlon = [-32.891401, -68.843242]

    view_state = pdk.ViewState(latitude=latlon[0], longitude=latlon[1], zoom=9)
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip={
            "html": "<b>{label}</b><br/>{" + var_fex + "}",
            "style": {
                "backgroundColor": "rgba(0,0,0,0.75)",
                "color": "white",
                "padding": "6px 10px",
                "borderRadius": "4px",
            },
        },
    )


def calcular_bins(df_viajes, var_fex, k_max, cut_col="cuts"):
    """
    Aplica Fisher–Jenks para generar cortes y asigna la columna de categorías:
      - Si hay un solo valor unico, asigna ese valor como etiqueta única.
      - Si hay >1 valor, intenta k=k_max…2; si falla, usa [mínimo, máximo].
      - Limpia duplicados consecutivos en los bins.
      - Añade en la copia del DataFrame una columna `cut_col` con los intervalos.
    """
    
    valores = df_viajes[var_fex]
    if valores.isnull().any():
        raise ValueError(f"La columna {var_fex} contiene valores nulos")
    valores = valores.astype(float)

    # Caso unico
    if valores.nunique() == 1:
        unico = int(valores.iloc[0])
        df = df_viajes.copy()
        df[cut_col] = str(unico)
        labels = [str(unico)]
        return df, labels

    v_min, v_max = valores.min(), valores.max()
    raw_bins = None

    # Generar bins
    for k in range(k_max, 1, -1):
        try:
            clasif = mapclassify.FisherJenks(valores, k=k)
            raw_bins = [v_min] + clasif.bins.tolist()
            break
        except ValueError:
            continue
    if raw_bins is None:
        raw_bins = [v_min, v_max]

    # Limpiar duplicados consecutivos
    bins = []
    for b in raw_bins:
        if not bins or b != bins[-1]:
            bins.append(b)

    # Asignar categorías
    df = df_viajes.copy()
    if len(bins) > 1:
        labels = [f"{int(bins[i])} a {int(bins[i+1])}" for i in range(len(bins) - 1)]
        df[cut_col] = pd.cut(valores, bins=bins, labels=labels, include_lowest=True)
    else:
        etiqueta = str(int(bins[0]))
        df[cut_col] = etiqueta

    return df, labels


def formatear_columnas_numericas(df, columnas, forzar_entero=False):
    df_formateado = df.copy()
    for col in columnas:
        if forzar_entero:
            # Mostrar todo como entero (sin decimales)
            df_formateado[col] = df[col].apply(
                lambda x: f"{int(x):,}".replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )
        else:
            if pd.api.types.is_integer_dtype(df[col]):
                # Enteros sin decimales
                df_formateado[col] = df[col].apply(
                    lambda x: f"{x:,}".replace(",", "X")
                    .replace(".", ",")
                    .replace("X", ".")
                )
            elif pd.api.types.is_float_dtype(df[col]):
                # Floats con 2 decimales
                df_formateado[col] = df[col].apply(
                    lambda x: f"{x:,.2f}".replace(",", "X")
                    .replace(".", ",")
                    .replace("X", ".")
                )
    return df_formateado
