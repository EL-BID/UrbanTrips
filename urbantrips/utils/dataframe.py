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
):
    if zero_to_nan is None:
        zero_to_nan = []
    df = df_.copy()
    for i in zero_to_nan:
        df.loc[df[i] == 0, i] = np.nan

    if not set(aggregate_cols + weighted_mean_cols + [weight_col]).issubset(df.columns):
        raise ValueError("One or more columns specified do not exist in the DataFrame.")
    result = pd.DataFrame([])
    for col in weighted_mean_cols:
        df.loc[df[col].notna(), f"{col}_weighted"] = (
            df.loc[df[col].notna(), col] * df.loc[df[col].notna(), weight_col]
        )
        grouped = (
            df.loc[df[col].notna()]
            .groupby(aggregate_cols, as_index=False)[[f"{col}_weighted", weight_col]]
            .sum()
        )
        grouped[col] = grouped[f"{col}_weighted"] / grouped[weight_col]
        grouped = grouped.drop([f"{col}_weighted", weight_col], axis=1)
        if len(result) == 0:
            result = grouped.copy()
        else:
            result = result.merge(grouped, how="left", on=aggregate_cols)

    if var_fex_summed:
        fex = df.groupby(aggregate_cols, as_index=False)[weight_col].sum()
    else:
        fex = df.groupby(aggregate_cols, as_index=False)[weight_col].mean()
    result = result.merge(fex, how="left", on=aggregate_cols)
    return result
