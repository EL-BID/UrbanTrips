import pandas as pd
import pytest
from urbantrips.utils.utils import guardar_tabla_sql, levanto_tabla_sql
from urbantrips.datamodel.legs import asignar_id_viaje_etapa_fecha_completa


def _seed_transacciones(db_conn, rows):
    """Insert rows into transacciones and dias_ultima_corrida."""
    dias = list({r["dia"] for r in rows})
    db_conn._conn.executemany(
        "INSERT OR IGNORE INTO dias_ultima_corrida VALUES (?)",
        [(d,) for d in dias],
    )
    db_conn._conn.executemany(
        """INSERT INTO transacciones
           (id, fecha, id_tarjeta, dia, tiempo, hora, modo, id_linea,
            id_ramal, interno, orden_trx, latitud, longitud, factor_expansion)
           VALUES (:id, :fecha, :id_tarjeta, :dia, :tiempo, :hora, :modo,
                   :id_linea, :id_ramal, :interno, :orden_trx,
                   :latitud, :longitud, :factor_expansion)""",
        rows,
    )
    db_conn._conn.commit()


def test_transacciones_join_dias_corrida_query(db_conn):
    """The SQL query joining transacciones and dias_ultima_corrida returns the right rows."""
    ts = int(pd.Timestamp("2022-08-11 09:30:00").timestamp())
    _seed_transacciones(db_conn, [{
        "id": 1, "fecha": ts, "id_tarjeta": "CARD1", "dia": "2022-08-11",
        "tiempo": "09:30:00", "hora": 9, "modo": "bus", "id_linea": 1,
        "id_ramal": 1, "interno": 1, "orden_trx": 1,
        "latitud": -34.6158037, "longitud": -58.5033381, "factor_expansion": 1.0,
    }])

    result = pd.read_sql_query(
        "SELECT t.* FROM transacciones t JOIN dias_ultima_corrida d ON t.dia = d.dia",
        db_conn._conn,
    )
    assert len(result) == 1
    assert result.iloc[0]["id_tarjeta"] == "CARD1"
    assert result.iloc[0]["dia"] == "2022-08-11"


def test_transacciones_only_returns_rows_for_run_days(db_conn):
    """Rows for days not in dias_ultima_corrida must be excluded by the JOIN."""
    ts1 = int(pd.Timestamp("2022-08-11 09:00:00").timestamp())
    ts2 = int(pd.Timestamp("2022-08-12 09:00:00").timestamp())
    # Only seed 2022-08-11 in dias_ultima_corrida
    db_conn._conn.execute("INSERT INTO dias_ultima_corrida VALUES ('2022-08-11')")
    db_conn._conn.executemany(
        """INSERT INTO transacciones
           (id, fecha, id_tarjeta, dia, tiempo, hora, modo, id_linea,
            id_ramal, interno, orden_trx, latitud, longitud, factor_expansion)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (1, ts1, "A", "2022-08-11", "09:00:00", 9, "bus", 1, 1, 1, 1, -34.6, -58.5, 1.0),
            (2, ts2, "B", "2022-08-12", "09:00:00", 9, "bus", 1, 1, 1, 1, -34.7, -58.6, 1.0),
        ],
    )
    db_conn._conn.commit()

    result = pd.read_sql_query(
        "SELECT t.* FROM transacciones t JOIN dias_ultima_corrida d ON t.dia = d.dia",
        db_conn._conn,
    )
    assert len(result) == 1
    assert result.iloc[0]["id_tarjeta"] == "A"


def test_etapas_save_and_load_roundtrip(patched_db):
    """Legs processed into etapas shape can be saved and loaded correctly."""
    etapas = pd.DataFrame({
        "id": [1, 2, 3],
        "id_tarjeta": ["CARD1", "CARD1", "CARD1"],
        "dia": ["2022-08-11", "2022-08-11", "2022-08-11"],
        "id_viaje": [1, 1, 2],
        "id_etapa": [1, 2, 1],
        "modo": ["bus", "bus", "bus"],
        "id_linea": [10, 20, 10],
        "latitud": [-34.6, -34.7, -34.8],
        "longitud": [-58.5, -58.6, -58.7],
        "h3_o": ["88c2e312b9fffff"] * 3,
        "factor_expansion": [1.0, 1.0, 1.0],
    })
    guardar_tabla_sql(etapas, "etapas_test", tabla_tipo="data", modo="append")
    result = levanto_tabla_sql("etapas_test", tabla_tipo="data")

    assert len(result) == 3
    assert sorted(result["id_viaje"].tolist()) == [1, 1, 2]
    assert sorted(result["id_etapa"].tolist()) == [1, 1, 2]


def test_multiday_db_trip_count_equivalence(patched_db, df_trx_multiday):
    """Per-day DB saves and combined DB save must produce the same trip counts.
    This is the integration-level contract the new multi-day processing must satisfy."""
    # Process each day separately and save
    for day in sorted(df_trx_multiday.dia.unique()):
        day_slice = df_trx_multiday[df_trx_multiday.dia == day].copy()
        processed = asignar_id_viaje_etapa_fecha_completa(day_slice, ventana_viajes=120)
        to_save = processed[["id", "id_tarjeta", "dia", "id_viaje", "id_etapa"]].copy()
        guardar_tabla_sql(to_save, "etapas_per_day", tabla_tipo="data", modo="append")

    # Process all days at once and save
    processed_all = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    to_save_all = processed_all[["id", "id_tarjeta", "dia", "id_viaje", "id_etapa"]].copy()
    guardar_tabla_sql(to_save_all, "etapas_combined", tabla_tipo="data", modo="append")

    per_day = levanto_tabla_sql("etapas_per_day", tabla_tipo="data")
    combined = levanto_tabla_sql("etapas_combined", tabla_tipo="data")

    per_day_counts = per_day.groupby(["dia", "id_tarjeta"])["id_viaje"].max().sort_index()
    combined_counts = combined.groupby(["dia", "id_tarjeta"])["id_viaje"].max().sort_index()

    pd.testing.assert_series_equal(per_day_counts, combined_counts)
