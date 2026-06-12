# urbantrips/datamodel/ingestion.py
from __future__ import annotations

import pandas as pd

from urbantrips.storage.context import StorageContext


_KEY_COLS = ["id_tarjeta", "fecha_ts", "id_linea", "latitud", "longitud"]


def _standardize_chunk(
    df: pd.DataFrame,
    nombres_variables: dict,
    formato_fecha: str,
    tipo_trx_invalidas: dict | None,
    lineas_contienen_ramales: bool,
) -> pd.DataFrame:
    """
    Structural transforms on one CSV chunk. Returns the subset of columns
    expected by transacciones_raw. Row-wise only — no cross-row aggregation.
    """
    from urbantrips.datamodel.transactions import (
        filtrar_transacciones_invalidas,
        renombrar_columnas_tablas,
    )

    if tipo_trx_invalidas:
        df = filtrar_transacciones_invalidas(df, tipo_trx_invalidas)

    df = renombrar_columnas_tablas(df, nombres_variables, postfijo="_trx")
    if "orden_trx" not in df.columns and "orden" in df.columns:
        df = df.rename(columns={"orden": "orden_trx"})

    # parse date → unix timestamp (integer seconds)
    df["fecha_parsed"] = pd.to_datetime(df["fecha"], format=formato_fecha, errors="coerce")
    df["fecha_ts"] = (
        df["fecha_parsed"].astype("int64") // 10**9
    ).where(df["fecha_parsed"].notna()).astype("Int64")
    df["dia"] = df["fecha_parsed"].dt.strftime("%Y-%m-%d")
    df["hora"] = df["fecha_parsed"].dt.hour
    df["tiempo"] = df["fecha_parsed"].dt.strftime("%H:%M:%S")

    # drop rows with nulls in key fields
    df = df.dropna(subset=_KEY_COLS)

    # when lines do not contain branches, mirror id_linea
    if not lineas_contienen_ramales:
        df["id_ramal"] = df["id_linea"]

    # normalise id_tarjeta to str
    if df["id_tarjeta"].dtype == "float":
        df["id_tarjeta"] = pd.to_numeric(df["id_tarjeta"], downcast="integer")
    df["id_tarjeta"] = df["id_tarjeta"].astype(str)

    # fallback defaults
    if "genero" not in df.columns:
        df["genero"] = "-"
    if "tarifa" not in df.columns:
        df["tarifa"] = "-"
    if "modo" not in df.columns or df["modo"].isna().all():
        df["modo"] = "autobus"
    else:
        try:
            from urbantrips.utils.utils import leer_configs_generales

            modos = leer_configs_generales(autogenerado=False).get("modos", {})
        except Exception:
            modos = {}
        modos_homologados = {
            raw: estandar
            for estandar, raw in modos.items()
            if raw is not None and raw != ""
        }
        if modos_homologados:
            modos_validos = set(modos_homologados) | set(modos_homologados.values())
            df.loc[~df["modo"].isin(modos_validos), "modo"] = "otros"
            df["modo"] = df["modo"].replace(modos_homologados)
    df["genero"] = df["genero"].fillna("-")
    df["tarifa"] = df["tarifa"].fillna("-")

    factor_col = "factor_expansion" if "factor_expansion" in df.columns else None
    df["factor_expansion_raw"] = df[factor_col] if factor_col else 1.0

    if "id_original" not in df.columns:
        df["id_original"] = df.index.astype(str)

    raw_cols = [
        "id_original", "id_tarjeta", "dia", "tiempo", "hora", "modo",
        "id_linea", "id_ramal", "interno", "orden_trx", "genero", "tarifa",
        "latitud", "longitud", "fecha_ts", "factor_expansion_raw",
    ]
    return df.reindex(columns=raw_cols)


def ingest_day_csv(
    ctx: StorageContext,
    csv_path: str,
    nombres_variables: dict,
    formato_fecha: str,
    tipo_trx_invalidas: dict | None,
    lineas_contienen_ramales: bool,
    chunk_size: int = 100_000,
) -> None:
    """
    Stream one day's CSV into transacciones_raw in fixed-size chunks.
    No full-day load into memory. No cross-row aggregation.
    """
    from urbantrips.utils.io import open_csv, resolve_zip
    csv_path = resolve_zip(csv_path)
    with open_csv(csv_path) as f:
        for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
            standardized = _standardize_chunk(
                chunk,
                nombres_variables=nombres_variables,
                formato_fecha=formato_fecha,
                tipo_trx_invalidas=tipo_trx_invalidas,
                lineas_contienen_ramales=lineas_contienen_ramales,
            )
            if len(standardized) > 0:
                ctx.data.save_raw_chunk(standardized)
