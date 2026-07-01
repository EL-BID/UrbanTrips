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
    return duckdb.sql(query).df()
