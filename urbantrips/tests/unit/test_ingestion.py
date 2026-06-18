# urbantrips/tests/unit/test_ingestion.py
import pandas as pd
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from urbantrips.datamodel.ingestion import (
    _standardize_chunk,
    ingest_day_csv,
)


NOMBRES_VARIABLES = {
    "id_trx": "id",
    "id_tarjeta": "id_tarjeta",
    "fecha": "fecha",
    "id_linea": "id_linea",
    "id_ramal": "id_ramal",
    "interno": "interno",
    "orden": "orden_trx",
    "latitud": "latitud",
    "longitud": "longitud",
    "modo": "modo",
    "tarifa": "tarifa",
    "fex": "factor_expansion",
}


def make_chunk(n=3, include_null_tarjeta=False):
    rows = {
        "id_trx": [f"TRX{i}" for i in range(n)],
        "id_tarjeta": ["card_A", None, "card_B"][:n] if include_null_tarjeta else [f"card_{i}" for i in range(n)],
        "fecha": [f"2022-01-01 08:{i:02d}:00" for i in range(n)],
        "id_linea": [1] * n,
        "id_ramal": [1] * n,
        "interno": [10] * n,
        "orden": list(range(1, n + 1)),
        "latitud": [-34.6] * n,
        "longitud": [-58.4] * n,
        "modo": ["autobus"] * n,
        "tarifa": ["-"] * n,
        "fex": [1.0] * n,
    }
    return pd.DataFrame(rows)


def test_standardize_chunk_produces_transacciones_raw_columns():
    df = make_chunk(3)
    result = _standardize_chunk(
        df,
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=True,
    )
    expected_cols = {
        "id_original", "id_tarjeta", "dia", "tiempo", "hora", "modo",
        "id_linea", "id_ramal", "interno", "orden_trx", "genero", "tarifa",
        "latitud", "longitud", "fecha_ts", "factor_expansion_raw",
    }
    assert expected_cols == set(result.columns)


def test_standardize_chunk_parses_fecha_to_unix_timestamp():
    df = make_chunk(1)
    result = _standardize_chunk(
        df,
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=True,
    )
    assert pd.api.types.is_integer_dtype(result["fecha_ts"])
    assert result["fecha_ts"].iloc[0] > 0


def test_standardize_chunk_drops_rows_with_null_key_fields():
    df = make_chunk(3, include_null_tarjeta=True)
    result = _standardize_chunk(
        df,
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=True,
    )
    # row with null id_tarjeta is dropped
    assert len(result) == 2
    assert result["id_tarjeta"].notna().all()


def test_standardize_chunk_sets_id_ramal_from_id_linea_when_no_branches():
    df = make_chunk(1)
    df["id_ramal"] = 99  # will be overwritten
    result = _standardize_chunk(
        df,
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=False,
    )
    assert result["id_ramal"].iloc[0] == result["id_linea"].iloc[0]


def test_standardize_chunk_keeps_rows_missing_latlong_when_geolocalizar_trx():
    df = make_chunk(2)
    df["latitud"] = None
    df["longitud"] = None
    result = _standardize_chunk(
        df,
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=True,
        geolocalizar_trx=True,
    )
    assert len(result) == 2
    assert result["latitud"].isna().all()
    assert result["longitud"].isna().all()


def test_standardize_chunk_drops_rows_missing_latlong_when_not_geolocalizar_trx():
    df = make_chunk(2)
    df["latitud"] = None
    df["longitud"] = None
    result = _standardize_chunk(
        df,
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=True,
        geolocalizar_trx=False,
    )
    assert len(result) == 0


def test_ingest_day_csv_calls_save_raw_chunk_once_per_chunk(tmp_path):
    csv_path = tmp_path / "trx.csv"
    rows = "\n".join(
        f"TRX{i},card_{i},2022-01-01 08:{i % 60:02d}:00,1,1,10,{i+1},-34.6,-58.4,autobus,-,1.0"
        for i in range(250)
    )
    header = "id_trx,id_tarjeta,fecha,id_linea,id_ramal,interno,orden,latitud,longitud,modo,tarifa,fex"
    csv_path.write_text(header + "\n" + rows)

    ctx = MagicMock()
    ingest_day_csv(
        ctx=ctx,
        csv_path=str(csv_path),
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=True,
        chunk_size=100,
    )
    # 250 rows / 100 = 3 chunks
    assert ctx.data.save_raw_chunk.call_count == 3


def test_ingest_day_csv_total_rows_stored(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    csv_path = tmp_path / "trx.csv"
    rows = "\n".join(
        f"TRX{i},card_{i},2022-01-01 08:{i % 60:02d}:00,1,1,10,{i+1},-34.6,-58.4,autobus,-,1.0"
        for i in range(90)
    )
    header = "id_trx,id_tarjeta,fecha,id_linea,id_ramal,interno,orden,latitud,longitud,modo,tarifa,fex"
    csv_path.write_text(header + "\n" + rows)

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    ctx = MagicMock()
    ctx.data = adapter

    ingest_day_csv(
        ctx=ctx,
        csv_path=str(csv_path),
        nombres_variables=NOMBRES_VARIABLES,
        formato_fecha="%Y-%m-%d %H:%M:%S",
        tipo_trx_invalidas=None,
        lineas_contienen_ramales=True,
        chunk_size=50,
    )
    count = adapter.query("SELECT COUNT(*) as n FROM transacciones_raw")["n"].iloc[0]
    assert count == 90
