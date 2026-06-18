import pandas as pd

from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

RAW_COLS = [
    "id_original", "id_tarjeta", "dia", "tiempo", "hora", "modo",
    "id_linea", "id_ramal", "interno", "orden_trx", "genero", "tarifa",
    "latitud", "longitud", "fecha_ts", "factor_expansion_raw",
]


def _raw_row(**overrides):
    row = {
        "id_original": "1", "id_tarjeta": "card_1", "dia": "2022-01-01",
        "tiempo": "08:00:00", "hora": 8, "modo": "autobus",
        "id_linea": 1, "id_ramal": 1, "interno": 10, "orden_trx": 1,
        "genero": "-", "tarifa": "-", "latitud": None, "longitud": None,
        "fecha_ts": 1641024000, "factor_expansion_raw": 1.0,
    }
    row.update(overrides)
    return row


def test_geolocate_raw_transactions_fills_from_nearest_preceding_gps_ping(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")

    # trx at 08:05 (1641024300)
    trx = pd.DataFrame([_raw_row(fecha_ts=1641024300)]).reindex(columns=RAW_COLS)
    adapter.save_raw_chunk(trx)

    gps = pd.DataFrame([
        {  # 08:00 ping, the nearest one before the trx -> must be picked
            "id": 1, "id_original": "g1", "dia": "2022-01-01", "id_linea": 1,
            "id_ramal": 1, "interno": 10, "fecha": 1641024000,
            "latitud": -34.6, "longitud": -58.4,
        },
        {  # 07:00 ping, also before, but further away -> must NOT be picked
            "id": 2, "id_original": "g2", "dia": "2022-01-01", "id_linea": 1,
            "id_ramal": 1, "interno": 10, "fecha": 1641020400,
            "latitud": -34.0, "longitud": -58.0,
        },
    ])
    adapter.save_gps(gps)

    adapter.geolocate_raw_transactions_from_gps(lineas_contienen_ramales=True)

    result = adapter.query("SELECT latitud, longitud FROM transacciones_raw")
    assert result["latitud"].iloc[0] == -34.6
    assert result["longitud"].iloc[0] == -58.4


def test_geolocate_raw_transactions_leaves_null_when_no_preceding_gps_ping(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")

    # trx before any gps ping for this vehicle
    trx = pd.DataFrame([_raw_row(fecha_ts=1640990000)]).reindex(columns=RAW_COLS)
    adapter.save_raw_chunk(trx)

    gps = pd.DataFrame([{
        "id": 1, "id_original": "g1", "dia": "2022-01-01", "id_linea": 1,
        "id_ramal": 1, "interno": 10, "fecha": 1641024000,
        "latitud": -34.6, "longitud": -58.4,
    }])
    adapter.save_gps(gps)

    adapter.geolocate_raw_transactions_from_gps(lineas_contienen_ramales=True)

    result = adapter.query("SELECT latitud, longitud FROM transacciones_raw")
    assert result["latitud"].isna().iloc[0]


def test_geolocate_raw_transactions_leaves_already_geolocated_rows_untouched(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")

    # one row already has lat/lon, another needs geolocation
    already_located = _raw_row(
        id_original="1", id_tarjeta="card_1", fecha_ts=1641024300,
        latitud=-12.34, longitud=-56.78,
    )
    needs_location = _raw_row(
        id_original="2", id_tarjeta="card_2", fecha_ts=1641024300,
    )
    trx = pd.DataFrame([already_located, needs_location]).reindex(columns=RAW_COLS)
    adapter.save_raw_chunk(trx)

    gps = pd.DataFrame([{
        "id": 1, "id_original": "g1", "dia": "2022-01-01", "id_linea": 1,
        "id_ramal": 1, "interno": 10, "fecha": 1641024000,
        "latitud": -34.6, "longitud": -58.4,
    }])
    adapter.save_gps(gps)

    adapter.geolocate_raw_transactions_from_gps(lineas_contienen_ramales=True)

    result = adapter.query(
        "SELECT * FROM transacciones_raw ORDER BY id_original"
    ).reindex(columns=RAW_COLS)

    before = trx.sort_values("id_original").reset_index(drop=True)
    already_row_after = result[result["id_original"] == "1"].reset_index(drop=True)
    already_row_before = before[before["id_original"] == "1"].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        already_row_after, already_row_before, check_dtype=False
    )

    needs_row_after = result[result["id_original"] == "2"].iloc[0]
    assert needs_row_after["latitud"] == -34.6
    assert needs_row_after["longitud"] == -58.4
