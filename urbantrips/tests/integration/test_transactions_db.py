import pandas as pd
from urbantrips.utils.utils import guardar_tabla_sql, levanto_tabla_sql


def test_transacciones_table_schema_accepts_expected_columns(patched_db):
    """Verify the transacciones table accepts all columns in the current schema."""
    ts = int(pd.Timestamp("2022-08-11 09:30:00").timestamp())
    df = pd.DataFrame([{
        "id": 1, "fecha": ts, "id_original": "orig1", "id_tarjeta": "CARD1",
        "dia": "2022-08-11", "tiempo": "09:30:00", "hora": 9, "modo": "bus",
        "id_linea": 1, "id_ramal": 1, "interno": 1, "orden_trx": 1,
        "genero": None, "tarifa": None,
        "latitud": -34.6158037, "longitud": -58.5033381, "factor_expansion": 1.0,
    }])
    guardar_tabla_sql(df, "transacciones", tabla_tipo="data", modo="append")
    result = levanto_tabla_sql("transacciones", tabla_tipo="data")
    assert len(result) == 1
    assert result.iloc[0]["id_tarjeta"] == "CARD1"


def test_transacciones_multi_day_filter_by_dia(patched_db):
    """Rows from different days can be stored and filtered by dia."""
    ts1 = int(pd.Timestamp("2022-08-11 09:00:00").timestamp())
    ts2 = int(pd.Timestamp("2022-08-12 09:00:00").timestamp())
    df = pd.DataFrame([
        {
            "id": 1, "fecha": ts1, "id_tarjeta": "A", "dia": "2022-08-11",
            "tiempo": "09:00:00", "hora": 9, "modo": "bus", "id_linea": 1,
            "id_ramal": 1, "interno": 1, "orden_trx": 1, "genero": None,
            "tarifa": None, "latitud": -34.6, "longitud": -58.5, "factor_expansion": 1.0,
        },
        {
            "id": 2, "fecha": ts2, "id_tarjeta": "B", "dia": "2022-08-12",
            "tiempo": "09:00:00", "hora": 9, "modo": "bus", "id_linea": 1,
            "id_ramal": 1, "interno": 1, "orden_trx": 1, "genero": None,
            "tarifa": None, "latitud": -34.7, "longitud": -58.6, "factor_expansion": 1.0,
        },
    ])
    guardar_tabla_sql(df, "transacciones", tabla_tipo="data", modo="append")

    result = levanto_tabla_sql(
        "transacciones", tabla_tipo="data",
        query="SELECT * FROM transacciones WHERE dia = '2022-08-11'",
    )
    assert len(result) == 1
    assert result.iloc[0]["id_tarjeta"] == "A"
