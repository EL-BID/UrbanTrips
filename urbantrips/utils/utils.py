import datetime
import logging
import math
import os
import sqlite3
import time
from functools import wraps
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import weightedstats as ws
import yaml
from pandas.io.sql import DatabaseError
from shapely import wkt
from shapely.geometry import base as shapely_geom

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.utils.decorators import duracion  # noqa: F401 — re-exported for compat
from urbantrips.utils.dataframe import normalize_vars, calculate_weighted_means  # noqa: F401 — re-exported for compat

logger = logging.getLogger(__name__)



def leer_alias(tipo="data"):
    configs = leer_configs_generales(autogenerado=False)
    alias_insumos = configs.get("alias_db_insumos") or configs.get("alias_db") or ""
    alias_data    = configs.get("alias_db") or alias_insumos
    aliases = {
        "data":    configs.get("alias_db_data")      or alias_data,
        "insumos": alias_insumos,
        "dash":    configs.get("alias_db_dashboard") or alias_data,
        "general": alias_data,
    }
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

    candidates = [
        Path("data") / "db" / f"{alias_db}{tipo}.duckdb",
        Path("/data/db") / f"{alias_db}{tipo}.duckdb",
        Path("data") / "db" / f"{alias_db}{tipo}.sqlite",
        Path("/data/db") / f"{alias_db}{tipo}.sqlite",
    ]
    db_path = next((p for p in candidates if p.exists()), None)
    if db_path is None:
        db_path = Path("data") / "db" / f"{alias_db}{tipo}.duckdb"

    return db_path


def iniciar_conexion_db(tipo="data", alias_db=""):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve una conexion a la db (DuckDB o SQLite segun el archivo disponible)
    """
    import duckdb as _duckdb

    if len(alias_db) == 0:
        alias_db = leer_alias(tipo)
    if not alias_db.endswith("_"):
        alias_db += "_"
    db_path = get_db_path(tipo, alias_db)
    if str(db_path).endswith(".duckdb"):
        return _duckdb.connect(str(db_path), read_only=False)
    return sqlite3.connect(db_path, timeout=10)


def leer_configs_generales(autogenerado=True):
    """
    Lee el archivo de configuración YAML, probando primero con UTF-8
    y luego con latin-1 si es necesario. Devuelve un dict o {} si falla.

    Respeta la variable de entorno URBANTRIPS_CONFIG si está definida
    (establecida por --config en run_all_urbantrips.py).
    """
    env_path = os.environ.get("URBANTRIPS_CONFIG")
    if env_path:
        path = env_path
    else:
        archivo = (
            "configuraciones_generales_autogenerado.yaml"
            if autogenerado
            else "configuraciones_generales.yaml"
        )
        path = os.path.join("configs", archivo)

    try:
        # Primer intento: UTF-8
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except UnicodeDecodeError:
        # Segundo intento: latin-1
        try:
            with open(path, "r", encoding="latin-1") as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as error:
            logger.error("Error de sintaxis YAML con latin-1: %s", error)
        except Exception as e:
            logger.error("Error general con latin-1: %s", e)
    except yaml.YAMLError as error:
        logger.error("Error de sintaxis YAML con UTF-8: %s", error)
    except Exception as e:
        logger.error("Error general leyendo archivo: %s", e)

    return {}


_TUNING_DEFAULTS: dict = {
    "dbscan": {
        "grid_steps": 5,
        "early_stop_silhouette": 0.7,
    },
}


def leer_configs_tuning() -> dict:
    """
    Load optional performance-tuning parameters from configs/tuning.yaml.
    Returns hardcoded defaults for any key not present in the file.
    The file is optional — if absent, all defaults apply.
    """
    import copy

    def _deep_merge(base: dict, overrides: dict) -> dict:
        result = copy.deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    path = os.path.join("configs", "tuning.yaml")
    if not os.path.exists(path):
        return copy.deepcopy(_TUNING_DEFAULTS)

    try:
        with open(path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        return _deep_merge(_TUNING_DEFAULTS, overrides)
    except Exception as e:
        logger.warning("Could not load configs/tuning.yaml: %s — using defaults", e)
        return copy.deepcopy(_TUNING_DEFAULTS)


def agrego_indicador(
    df_indicador,
    detalle,
    tabla,
    nivel=0,
    var="indicador",
    var_fex="factor_expansion_linea",
    aggfunc="sum",
    *,
    ctx=None,
):
    """
    Agrego indicadores de tablas utilizadas.
    Pass ctx=<StorageContext> to use the port-based adapter; otherwise falls
    back to the legacy SQLite connection (backward compat during migration).
    """
    from urbantrips.storage.context import StorageContext

    df = df_indicador.copy()

    if ctx is not None:
        indicadores = ctx.data.get_indicators()
    else:
        conn_data = iniciar_conexion_db(tipo="data")
        try:
            indicadores = pd.read_sql_query("SELECT * FROM indicadores", conn_data)
        except DatabaseError:
            indicadores = pd.DataFrame([])

    if var not in df.columns:
        if not var_fex:
            df[var] = 1
        else:
            df[var] = df[var_fex]

    if var != "indicador":
        df = df.rename(columns={var: "indicador"})

    df = df[(df.indicador.notna())].copy()

    if len(df) == 0:
        logger.warning('Para el indicador "%s" no hay datos para agregar', var)
    else:
        if (not var_fex) | (aggfunc == "sum"):
            resultado = (
                df.groupby("dia", as_index=False).agg({"indicador": aggfunc}).round(2)
            )

        elif aggfunc == "mean":
            resultado = df.groupby("dia").apply(
                lambda x: np.average(x["indicador"], weights=x[var_fex])
            )
            resultado = resultado.reset_index()
            resultado.columns = ["dia", "indicador"]
            resultado = resultado.round(2)

        elif aggfunc == "median":
            resultado = df.groupby("dia").apply(
                lambda x: ws.weighted_median(
                    x["indicador"].tolist(), weights=x[var_fex].tolist()
                )
            )
            resultado = resultado.reset_index()
            resultado.columns = ["dia", "indicador"]
            resultado = resultado.round(2)

        resultado["detalle"] = detalle
        resultado = resultado[["dia", "detalle", "indicador"]]
        resultado["tabla"] = tabla
        resultado["nivel"] = nivel

        if len(indicadores) > 0:
            indicadores = indicadores[
                ~(
                    (indicadores.dia.isin(resultado.dia.unique()))
                    & (indicadores.detalle == detalle)
                    & (indicadores.tabla == tabla)
                )
            ]

        indicadores = pd.concat([indicadores, resultado], ignore_index=True)
        if nivel > 0:
            for i in indicadores[
                (indicadores.tabla == tabla) & (indicadores.nivel == nivel)
            ].dia.unique():
                for x in indicadores.loc[
                    (indicadores.tabla == tabla)
                    & (indicadores.nivel == nivel)
                    & (indicadores.dia == i),
                    "detalle",
                ]:
                    valores = round(
                        indicadores.loc[
                            (indicadores.tabla == tabla)
                            & (indicadores.nivel == nivel)
                            & (indicadores.dia == i)
                            & (indicadores.detalle == x),
                            "indicador",
                        ].values[0]
                        / indicadores.loc[
                            (indicadores.tabla == tabla)
                            & (indicadores.nivel == nivel - 1)
                            & (indicadores.dia == i),
                            "indicador",
                        ].values[0]
                        * 100,
                        1,
                    )
                    indicadores.loc[
                        (indicadores.tabla == tabla)
                        & (indicadores.nivel == nivel)
                        & (indicadores.dia == i)
                        & (indicadores.detalle == x),
                        "porcentaje",
                    ] = valores

        indicadores.fillna(0, inplace=True)

        if ctx is not None:
            ctx.data.save_indicators(indicadores)
        else:
            SAFE_CHUNKSIZE = (
                math.floor((999 / len(indicadores.columns)) * 0.9)
                if len(indicadores.columns) > 0
                else 1
            )
            indicadores.to_sql(
                "indicadores",
                conn_data,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=SAFE_CHUNKSIZE,
            )
            conn_data.close()


@duracion
def eliminar_tarjetas_trx_unica(trx):
    """
    Esta funcion toma el DF de trx y elimina las trx de una tarjeta con
    una unica trx en el dia
    """

    tarjetas_dia_multiples = (
        trx.reindex(columns=["id_tarjeta", "dia"])
        .groupby(["dia", "id_tarjeta"], as_index=False)
        .size()
        .query("size > 1")
    )

    pre = len(trx)
    trx = trx.merge(tarjetas_dia_multiples, on=["dia", "id_tarjeta"], how="inner").drop(
        "size", axis=1
    )
    post = len(trx)
    logger.info("%d casos eliminados por trx únicas en el dia", pre - post)
    return trx




from urbantrips.utils.sql import (  # noqa: F401 — re-exported for compat
    is_date_string,
    check_date_type,
    create_line_ids_sql_filter,
    create_branch_ids_sql_filter,
)



def delete_data_from_table_run_days(table_name):

    conn_data = iniciar_conexion_db(tipo="data")

    dias_ultima_corrida = pd.read_sql_query(
        """
                                    SELECT *
                                    FROM dias_ultima_corrida
                                    """,
        conn_data,
    )
    # delete data from same day if exists
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM {table_name} WHERE dia IN ({values})"
    conn_data.execute(query)
    conn_data.commit()
    conn_data.close()


def tabla_existe(conn, table_name):
    try:
        conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        return True
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return False
        else:
            raise


def _aplicar_pragmas_wal(conn):
    """Aplica pragmas de performance WAL a una conexión SQLite abierta."""
    import sqlite3
    if not isinstance(conn, sqlite3.Connection):
        return
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB
    conn.execute("PRAGMA temp_store=MEMORY")


def _drop_indices(conn, table_name):
    """Devuelve lista de índices existentes y los elimina."""
    cur = conn.execute(
        f"SELECT name, sql FROM sqlite_master "
        f"WHERE type='index' AND tbl_name='{table_name}' AND sql IS NOT NULL"
    )
    indices = cur.fetchall()
    for name, _ in indices:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.commit()
    return indices


def _recreate_indices(conn, indices):
    """Recrea los índices a partir del SQL original."""
    for _, sql in indices:
        if sql:
            conn.execute(sql)
    conn.commit()


def _executemany_df(
    df: pd.DataFrame, table_name: str, conn, if_exists: str = "append", indices=None
):
    """
    Escribe un DataFrame en SQLite usando executemany.
    Reemplaza to_sql(..., method='multi') manteniendo la misma semántica
    de if_exists='replace' / 'append'.
    Recibe índices opcionales desde el caller para evitar doble drop.
    """
    if df.empty:
        return

    if if_exists == "replace":
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    def _sqlite_type(series):
        if pd.api.types.is_float_dtype(series):
            return "REAL"
        elif pd.api.types.is_integer_dtype(series):
            return "INTEGER"
        else:
            return "TEXT"

    cols_def = ", ".join(f'"{c}" {_sqlite_type(df[c])}' for c in df.columns)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_def})')

    # Solo dropear índices si no vienen del caller
    if indices is None:
        indices = _drop_indices(conn, table_name)

    cols_str = ", ".join(f'"{c}"' for c in df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    sql = f'INSERT INTO "{table_name}" ({cols_str}) VALUES ({placeholders})'

    CHUNK = 10_000
    # total_chunks = math.ceil(len(df) / CHUNK)
    # t0 = time.time()
    # t_chunk = time.time()

    # print(f"\n  Escribiendo {len(df):,} registros en '{table_name}' ({total_chunks} chunks de {CHUNK:,})")

    for i in range(0, len(df), CHUNK):
        chunk = df.iloc[i : i + CHUNK]
        conn.executemany(sql, chunk.itertuples(index=False, name=None))
        # n_chunk = i // CHUNK + 1
        # elapsed_chunk = time.time() - t_chunk
        # elapsed_total = time.time() - t0
        # pct = n_chunk / total_chunks * 100
        # restante = (elapsed_total / n_chunk) * (total_chunks - n_chunk)
        # print(
        #     f"  chunk {n_chunk:>{len(str(total_chunks))}}/{total_chunks} "
        #     f"({pct:5.1f}%) — chunk: {elapsed_chunk:.1f}s — "
        #     f"total: {elapsed_total/60:.1f}min — restante: {restante/60:.1f}min",
        #     end="\r",
        # )
        # t_chunk = time.time()

    conn.commit()
    # print(f"\nEscritura finalizada en {(time.time()-t0)/60:.1f} min")

    if indices:
        # print(f"  Recreando {len(indices)} índices...", end=" ")
        # t_idx = time.time()
        _recreate_indices(conn, indices)
        # print(f"hecho en {(time.time()-t_idx)/60:.1f} min")

    # print(f"Escritura en {table_name} - Total finalizado en {(time.time()-t0)/60:.1f} min")


def levanto_tabla_sql(
    tabla_sql,
    tabla_tipo="dash",
    query="",
    alias_db="",
    index_cols=None,
):
    """
    Lee una tabla SQLite y la devuelve como DataFrame.
    Si la tabla tiene columna 'wkt', devuelve GeoDataFrame (CRS 4326).
    Si la tabla no existe, devuelve DataFrame vacío.

    Parameters
    ----------
    tabla_sql : str
        Nombre de la tabla.
    tabla_tipo : str
        DB a conectar: "dash" (default), "data", "insumos", "general".
    query : str
        Query personalizada. Si se omite, ejecuta SELECT * FROM tabla_sql.
    alias_db : str
        Prefijo del archivo SQLite. Si se omite, se lee desde configuración.
    index_cols : list of str
        Columnas sobre las que crear índices (CREATE INDEX IF NOT EXISTS).
        Mejora performance en JOINs y filtros. Default: None.

    Examples
    --------
    # Lectura simple
    etapas = levanto_tabla_sql("etapas", tabla_tipo="data")

    # Query personalizada con índice
    etapas = levanto_tabla_sql(
        "etapas",
        tabla_tipo="data",
        query="SELECT e.* FROM etapas e JOIN dias_ultima_corrida d ON e.dia = d.dia",
        index_cols=["dia"],
    )
    """

    if alias_db and not alias_db.endswith("_"):
        alias_db += "_"

    conn = iniciar_conexion_db(tipo=tabla_tipo, alias_db=alias_db)
    _aplicar_pragmas_wal(conn)

    if index_cols:
        tabla_sql = validate_table_name(tabla_sql)
        for col in index_cols:
            col = validate_table_name(col)
            idx_name = f"idx_{tabla_sql}_{col}"
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tabla_sql}({col})")
        conn.commit()

    try:
        if len(query) == 0:
            tabla_sql = validate_table_name(tabla_sql)
            query = f"SELECT * FROM {tabla_sql}"
        tabla = pd.read_sql_query(query, conn)
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        if "no such table" in str(e):
            logger.debug("La tabla '%s' no existe.", tabla_sql)
            tabla = pd.DataFrame([])
        else:
            raise

    conn.close()

    if "wkt" in tabla.columns and not tabla.empty:
        tabla["geometry"] = tabla.wkt.apply(wkt.loads)
        tabla = gpd.GeoDataFrame(tabla, crs=4326)
        tabla = tabla.drop(["wkt"], axis=1)

    tabla = normalize_vars(tabla)

    return tabla


def guardar_tabla_sql(
    df, table_name, tabla_tipo="dash", filtros=None, alias_db="", modo="append"
):
    """
    Guarda un DataFrame en SQLite usando executemany + WAL (alta performance).
    Convierte geometrías a WKT y tipos no compatibles a string automáticamente.
    Dropea índices antes de escribir y los recrea al final para máxima velocidad.

    Parameters
    ----------
    df : DataFrame o GeoDataFrame
        Datos a guardar. Si contiene columna 'geometry', se convierte a WKT
        y se guarda como 'wkt'. Tipos no compatibles con SQLite se convierten
        a string automáticamente.
    table_name : str
        Nombre de la tabla destino en la base de datos.
    tabla_tipo : str
        DB a conectar: "dash" (default), "data", "insumos", "general".
    filtros : dict
        Elimina registros coincidentes antes del append. Soporta escalares
        (igualdad) y listas (IN). Solo aplica con modo="append". Default: None.
    alias_db : str
        Prefijo del archivo SQLite. Si se omite, se lee desde configuración.
    modo : str
        "append" (default): agrega filas. "replace": recrea la tabla completa.

    Examples
    --------
    # Append simple
    guardar_tabla_sql(etapas, "etapas", tabla_tipo="data")

    # Reemplazar tabla completa
    guardar_tabla_sql(kpis, "kpis_resumen", tabla_tipo="dash", modo="replace")

    # Append eliminando registros previos del mismo día
    guardar_tabla_sql(
        etapas, "etapas", tabla_tipo="data",
        filtros={"dia": "2024-03-15"},
    )

    # Filtro con múltiples valores
    guardar_tabla_sql(
        etapas, "etapas", tabla_tipo="data",
        filtros={"dia": ["2024-03-15", "2024-03-16"], "id_linea": 41},
    )
    """
    t0 = time.time()
    table_name = validate_table_name(table_name)
    if filtros:
        filtros = {validate_table_name(campo): valor for campo, valor in filtros.items()}

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Se esperaba un DataFrame, pero se recibió un {type(df).__name__}"
        )

    if alias_db and not alias_db.endswith("_"):
        alias_db += "_"

    df = df.copy()

    if "geometry" in df.columns:
        df["wkt"] = df["geometry"].apply(
            lambda g: g.wkt if isinstance(g, shapely_geom.BaseGeometry) else None
        )
        df.drop(columns=["geometry"], inplace=True)

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        if len(sample) == 0:
            continue
        first = sample.iloc[0]
        if not isinstance(first, (str, int, float, bool)):
            df[col] = df[col].astype(str).where(df[col].notna(), None)

    conn = iniciar_conexion_db(tipo=tabla_tipo, alias_db=alias_db)
    _aplicar_pragmas_wal(conn)
    cursor = conn.cursor()

    try:
        if modo == "replace":
            _executemany_df(df, table_name, conn, if_exists="replace")
            logger.debug(
                "Tabla '%s' reemplazada en %.1f min", table_name, (time.time() - t0) / 60
            )
        else:
            table_exists = tabla_existe(conn, table_name)

            # 1. Dropear índices primero
            indices = _drop_indices(conn, table_name) if table_exists else []

            # 2. DELETE con filtros (ahora sin índices que mantener, más rápido)
            if table_exists and filtros:
                condiciones = []
                valores = []
                for campo, valor in filtros.items():
                    if isinstance(valor, list):
                        condiciones.append(
                            f"{campo} IN ({','.join(['?'] * len(valor))})"
                        )
                        valores.extend(valor)
                    else:
                        condiciones.append(f"{campo} = ?")
                        valores.append(valor)
                where_clause = " AND ".join(condiciones)
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE {where_clause}", valores
                )
                conn.commit()

            # 3. Escribir pasando índices ya dropeados para no repetir el drop
            _executemany_df(df, table_name, conn, if_exists="append", indices=indices)
            # print(f"Datos agregados exitosamente en '{table_name}'. Total finalizado en {(time.time()-t0)/60:.1f} min")

    finally:
        cursor.close()
        conn.close()
