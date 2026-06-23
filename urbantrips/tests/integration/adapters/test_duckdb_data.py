# urbantrips/tests/integration/adapters/test_duckdb_data.py
import pytest
import pandas as pd


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3],
        "fecha": [20240101, 20240101, 20240102],
        "id_original": ["a", "b", "c"],
        "id_tarjeta": ["T001", "T002", "T001"],
        "dia": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "tiempo": ["08:00", "09:00", "08:30"],
        "hora": [8, 9, 8],
        "modo": ["bus", "bus", "metro"],
        "id_linea": [1, 2, 3],
        "id_ramal": [10, 20, 30],
        "interno": [100, 200, 300],
        "orden_trx": [1, 1, 1],
        "genero": [None, None, None],
        "tarifa": [None, None, None],
        "latitud": [-34.6, -34.7, -34.8],
        "longitud": [-58.4, -58.5, -58.6],
        "factor_expansion": [1.0, 1.0, 1.0],
    })


def _sample_legs() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2],
        "id_tarjeta": ["T001", "T002"],
        "dia": ["2024-01-01", "2024-01-01"],
        "id_viaje": [1, 2],
        "id_etapa": [1, 1],
        "tiempo": ["08:00", "09:00"],
        "hora": [8, 9],
        "modo": ["bus", "bus"],
        "id_linea": [1, 2],
        "id_ramal": [10, 20],
        "interno": [100, 200],
        "genero": [None, None],
        "tarifa": [None, None],
        "latitud": [-34.6, -34.7],
        "longitud": [-58.4, -58.5],
        "h3_o": ["882a100d2bfffff", "882a100d3bfffff"],
        "h3_d": ["882a100d3bfffff", "882a100d4bfffff"],
        "od_validado": [1, 1],
        "etapa_validada": [1, 1],
        "factor_expansion_original": [1.0, 1.0],
        "factor_expansion_linea": [1.0, 1.0],
        "factor_expansion_tarjeta": [1.0, 1.0],
        "factor_expansion_etapa": [1.0, 1.0],
        "distancia": [500.0, 750.0],
        "travel_time_min": [15.0, 20.0],
    })


def test_run_days_roundtrip(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    days = pd.DataFrame({"dia": ["2024-01-01", "2024-01-02"]})
    adapter.save_run_days(days)
    result = adapter.get_run_days()
    assert set(result["dia"]) == {"2024-01-01", "2024-01-02"}


def test_transactions_roundtrip(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    trx = _sample_transactions()
    adapter.save_transactions(trx)
    result = adapter.get_transactions()
    assert len(result) == 3
    assert set(result["id"]) == {1, 2, 3}


def test_legs_roundtrip(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    legs = _sample_legs()
    adapter.save_legs(legs)
    result = adapter.get_legs()
    assert len(result) == 2


def test_save_legs_chunked_upsert(tmp_path, monkeypatch):
    from urbantrips.storage.adapters.duckdb import data as duckdb_data

    monkeypatch.setattr(duckdb_data, "_DUCKDB_INSERT_CHUNK_ROWS", 1)
    adapter = duckdb_data.DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter.save_legs(_sample_legs())

    replacement = _sample_legs()
    replacement.loc[replacement["id"] == 1, "h3_d"] = "882a100d5bfffff"
    adapter.save_legs(replacement)

    result = adapter.get_legs()
    assert len(result) == 2
    assert result.loc[result["id"] == 1, "h3_d"].iloc[0] == "882a100d5bfffff"


def test_update_leg_destinations_with_index_bracket(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter.save_legs(_sample_legs())

    adapter.begin_leg_destination_updates()
    try:
        updates = pd.DataFrame({
            "id": [1],
            "dia": ["2024-01-01"],
            "h3_d": ["882a100d9bfffff"],
            "od_validado": [0],
            "etapa_validada": [0],
        })
        adapter.update_leg_destinations(updates)
    finally:
        adapter.end_leg_destination_updates()

    result = adapter.get_legs().set_index("id")
    assert result.loc[1, "h3_d"] == "882a100d9bfffff"
    assert result.loc[1, "od_validado"] == 0
    assert result.loc[1, "etapa_validada"] == 0
    # leg 2 untouched
    assert result.loc[2, "od_validado"] == 1

    # the index must exist again after the bracket
    idx = adapter._conn.execute(
        "SELECT index_name FROM duckdb_indexes() "
        "WHERE table_name = 'etapas' AND index_name = 'idx_etapas_dia_od_validado'"
    ).fetchall()
    assert idx, "idx_etapas_dia_od_validado was not recreated"


def test_replace_legs_for_days_restores_threads_setting(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter._conn.execute("PRAGMA threads=4")
    adapter.replace_legs_for_days(_sample_legs(), ["2024-01-01"])
    threads = adapter._conn.execute(
        "SELECT current_setting('threads')"
    ).fetchone()[0]
    assert int(threads) == 4


def test_replace_legs_for_days_replaces_only_requested_days(tmp_path, monkeypatch):
    from urbantrips.storage.adapters.duckdb import data as duckdb_data

    monkeypatch.setattr(duckdb_data, "_DUCKDB_INSERT_CHUNK_ROWS", 1)
    adapter = duckdb_data.DuckDBDataAdapter(tmp_path / "data.duckdb")
    existing = _sample_legs()
    other_day = _sample_legs()
    other_day["id"] = [3, 4]
    other_day["dia"] = "2024-01-02"
    adapter.save_legs(pd.concat([existing, other_day], ignore_index=True))

    replacement = _sample_legs().iloc[[0]].copy()
    replacement["h3_d"] = "882a100d5bfffff"
    adapter.replace_legs_for_days(replacement, ["2024-01-01"])

    result = adapter.get_legs()
    assert len(result) == 3
    assert set(result["dia"]) == {"2024-01-01", "2024-01-02"}
    assert result.loc[result["dia"] == "2024-01-01", "h3_d"].tolist() == [
        "882a100d5bfffff"
    ]
    assert len(result.loc[result["dia"] == "2024-01-02"]) == 2


def test_has_rows(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    assert not adapter.has_rows("etapas")
    assert not adapter.has_rows("missing_table")

    adapter.save_legs(_sample_legs())

    assert adapter.has_rows("etapas")
    assert adapter.has_rows("etapas", "dia = '2024-01-01'")
    assert not adapter.has_rows("etapas", "dia = '2024-01-02'")


def test_delete_run_days(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    trx = _sample_transactions()
    adapter.save_transactions(trx)
    adapter.delete_run_days(["2024-01-01"])
    result = adapter.get_transactions()
    assert all(r != "2024-01-01" for r in result["dia"])


def test_satisfies_data_port(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.ports import DataPort
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    assert isinstance(adapter, DataPort)


def test_get_user_batches_returns_correct_count(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.ports import BatchSpec
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter.save_transactions(_sample_transactions())
    batches = adapter.get_user_batches(n_batches=2)
    assert len(batches) == 2
    assert batches[0] == BatchSpec(batch_id=0, total_batches=2)
    assert batches[1] == BatchSpec(batch_id=1, total_batches=2)


def test_batch_reads_are_disjoint_and_complete(tmp_path):
    """All transaction rows appear in exactly one batch."""
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    trx = _sample_transactions()
    adapter.save_transactions(trx)

    batches = adapter.get_user_batches(n_batches=2)
    all_ids = []
    for b in batches:
        result = adapter.get_transactions(batch=b)
        all_ids.extend(result["id_tarjeta"].tolist())

    assert sorted(all_ids) == sorted(trx["id_tarjeta"].tolist())


def test_batch_none_returns_all_rows(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    trx = _sample_transactions()
    adapter.save_transactions(trx)
    result = adapter.get_transactions(batch=None)
    assert len(result) == len(trx)


def test_service_kpi_schema_allows_big_branch_ids(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    df = pd.DataFrame(
        {
            "id_linea": [1006],
            "dia": ["2026-05-01"],
            "id_ramal": [1006000422001],
            "interno": ["42"],
            "service_id": [1],
            "hora_inicio": [8.0],
            "hora_fin": [9.0],
            "tot_km": [12.0],
            "tot_pax": [40.0],
            "dmt_mean_od": [1.5],
            "dmt_median_od": [1.3],
            "ipk_route": [3.3],
            "fo_mean_od": [0.1],
            "fo_median_od": [0.09],
        }
    )

    adapter.append_raw(df, "kpi_by_day_line_service")

    result = adapter.query("SELECT id_ramal FROM kpi_by_day_line_service")
    assert result.loc[0, "id_ramal"] == 1006000422001
