import duckdb
import numpy as np
import pandas as pd


def normalize_vars(tabla):
    if "day_type" in tabla.columns:
        tabla.loc[tabla.day_type == "weekday", "day_type"] = "Día hábil"
        tabla.loc[tabla.day_type == "weekend", "day_type"] = "Fin de semana"
    if "nombre_linea" in tabla.columns:
        tabla["nombre_linea"] = tabla["nombre_linea"].str.replace(" -", "")
    if "Modo" in tabla.columns:
        tabla["Modo"] = tabla["Modo"].str.capitalize()
    if "modo" in tabla.columns:
        tabla["modo"] = tabla["modo"].str.capitalize()
    return tabla


def calculate_weighted_means(
    df_,
    aggregate_cols,
    weighted_mean_cols,
    weight_col,
    zero_to_nan=None,
    var_fex_summed=True,
    query_fn=None,
    source=None,
    cte_prefix="",
):
    """Weighted means + summed/averaged weight, grouped by aggregate_cols.

    Two execution modes (the SQL is byte-identical between them):

    - In-RAM (default): registers the local DataFrame `df_` and runs the query
      via duckdb.sql. This is the original behaviour; callers that pass only the
      first four args are unaffected.
    - Pushed-down: pass `query_fn` (e.g. ctx.data.query) and `source` (a table or
      CTE alias such as "etapas_proc", optionally fed by `cte_prefix="WITH ..."`).
      The aggregation then runs inside the data DB without materialising the rows
      in pandas. `df_` is ignored for data and may be None.
    """
    if zero_to_nan is None:
        zero_to_nan = []

    pushed_down = query_fn is not None
    if source is None:
        source = "df_"

    if not pushed_down:
        if not set(aggregate_cols + weighted_mean_cols + [weight_col]).issubset(df_.columns):
            raise ValueError("One or more columns specified do not exist in the DataFrame.")

    def q(name):
        return f'"{name}"'

    wm_sql = []
    for col in weighted_mean_cols:
        src = f"NULLIF({q(col)}, 0)" if col in zero_to_nan else q(col)
        wm_sql.append(
            f"SUM(CASE WHEN {src} IS NOT NULL THEN CAST({src} AS DOUBLE) * {q(weight_col)} END) / "
            f"NULLIF(SUM(CASE WHEN {src} IS NOT NULL THEN {q(weight_col)} END), 0) AS {q(col)}"
        )

    fex_agg = "SUM" if var_fex_summed else "AVG"
    keys = ", ".join(q(c) for c in aggregate_cols)
    cols_sql = ",\n       ".join(wm_sql)

    query = f"""
        SELECT {keys},
               {cols_sql},
               {fex_agg}({q(weight_col)}) AS {q(weight_col)}
        FROM {source}
        GROUP BY {keys}
    """
    if pushed_down:
        return query_fn(cte_prefix + query)

    # The implicit duckdb.sql() connection starts with stock defaults in every
    # process, spawned workers included: pin memory_limit/temp_directory first.
    from urbantrips.storage.adapters.duckdb.data import ensure_global_duckdb
    ensure_global_duckdb()
    return duckdb.sql(query).df()


# ---------------------------------------------------------------------------
# Lectura de a un dia de las tablas grandes (transacciones, gps)
# ---------------------------------------------------------------------------
# `transacciones` y `gps` son las dos tablas mas grandes de una corrida (208M y
# 104M filas en el mes de AMBA del cliente). DuckDB entrega las columnas TEXT
# como un objeto `str` por fila --sin compartirlos, aunque haya 31 valores
# distintos-- asi que un `SELECT` de 6 columnas sobre transacciones cuesta 181
# B/fila medidos: ~38 GB para el mes, mas que la RAM de la maquina. Todos los
# agregados que el dashboard hace sobre esas tablas llevan `dia` en la clave, y
# por eso se pueden calcular leyendo un dia por vez y combinando los parciales.
# Estas tres funciones son la maquinaria compartida por `resumen_x_linea` y
# `levanto_data`.


def dias_para_leer_por_dia(query_fn, tabla, dias=None, dia_col="dia"):
    """Dias a recorrer para leer `tabla` de a un dia por vez.

    Con `dias` (el scope de la corrida) se usan esos; sin scope se toman los
    dias presentes en la tabla, que equivale a leerla entera.
    """
    if dias:
        return [str(d) for d in sorted(dias)]
    df = query_fn(
        f"SELECT DISTINCT {dia_col} AS dia FROM {tabla} "
        f"WHERE {dia_col} IS NOT NULL ORDER BY 1"
    )
    if len(df) == 0:
        return []
    return df["dia"].astype(str).tolist()


def leer_dia(query_fn, tabla, columnas, dia, dia_col="dia", extra_where=""):
    """Un dia de `tabla`, proyectado a `columnas`."""
    dia_sql = str(dia).replace("'", "''")
    where = f"WHERE {dia_col} = '{dia_sql}'"
    if extra_where:
        where += f" AND {extra_where}"
    return query_fn(f"SELECT {', '.join(columnas)} FROM {tabla} {where}")


def combinar_suma(parciales, cols, valor, observed=True):
    """Cierra un `groupby(cols)[valor].sum()` a partir de los parciales por dia.

    Volver a sumar es lo que reune un grupo que haya quedado partido entre dos
    chunks (pasa cuando la clave de agregacion no es la columna por la que se
    chunkea, p. ej. el `dia` derivado de `fecha` en gps). Cuando no hay grupos
    partidos --el caso normal, con `dia` en `cols`-- cada grupo queda con un
    solo sumando y el resultado es identico al global bit a bit.
    """
    if not parciales:
        return pd.DataFrame(columns=list(cols) + [valor])
    df = pd.concat(parciales, ignore_index=True)
    return df.groupby(list(cols), as_index=False, observed=observed)[valor].sum()


def combinar_conteo_distintos(parciales, cols, out_col, observed=True):
    """Cierra un `groupby(cols + [clave]).size().groupby(cols).size()` a partir
    de los `groupby(cols + [clave]).size()` parciales de cada dia.

    Es exactamente el conteo global: el groupby parcial ya descarto las filas
    con clave nula igual que lo haria el global, la suma vuelve a unir los
    grupos partidos entre chunks, y el segundo groupby cuenta grupos.
    """
    if not parciales:
        return pd.DataFrame(columns=list(cols) + [out_col])
    df = pd.concat(parciales, ignore_index=True)
    claves = [c for c in df.columns if c != "size"]
    df = df.groupby(claves, as_index=False, observed=observed)["size"].sum()
    return (
        df.groupby(list(cols), as_index=False, observed=observed)
        .size()
        .rename(columns={"size": out_col})
    )
