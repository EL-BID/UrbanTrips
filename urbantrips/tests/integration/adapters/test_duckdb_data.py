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


def test_get_transactions_for_chunk_filters_by_run_days(tmp_path):
    """Regresión: get_transactions_for_chunk debe acotar la lectura a run_days.

    transacciones es ACUMULATIVA; sin este filtro cada worker de Fase 2 carga
    TODOS los días acumulados en RAM (la memoria crece con lo acumulado, no con
    los días de la corrida) — causó un OOM incremental a escala AMBA.
    """
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    trx = _sample_transactions()   # ids 1,2 -> 2024-01-01 ; id 3 -> 2024-01-02
    trx["batch_id"] = 0
    adapter.save_transactions(trx)

    # sin run_days: todo el batch (los 2 días acumulados)
    todo = adapter.get_transactions_for_chunk([0], 1)
    assert set(todo["dia"]) == {"2024-01-01", "2024-01-02"}
    assert len(todo) == 3

    # con run_days: SOLO el día de la corrida (no arrastra lo acumulado)
    solo = adapter.get_transactions_for_chunk([0], 1, run_days=["2024-01-02"])
    assert set(solo["dia"]) == {"2024-01-02"}
    assert len(solo) == 1


def test_get_transactions_filters_by_run_days(tmp_path):
    """Regresión #7: el path serial de get_transactions debe acotar a run_days.

    Espeja get_transactions_for_chunk (path paralelo): transacciones es ACUMULATIVA,
    así que sin el filtro build_legs_from_transactions cargaría todos los días
    acumulados y recién los descartaría en pandas (tiempo/RAM O(acumulado)).
    """
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    trx = _sample_transactions()   # ids 1,2 -> 2024-01-01 ; id 3 -> 2024-01-02
    adapter.save_transactions(trx)

    # sin run_days: todos los días acumulados
    todo = adapter.get_transactions()
    assert set(todo["dia"]) == {"2024-01-01", "2024-01-02"}
    assert len(todo) == 3

    # con run_days: SOLO el día de la corrida
    solo = adapter.get_transactions(run_days=["2024-01-02"])
    assert set(solo["dia"]) == {"2024-01-02"}
    assert len(solo) == 1


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


def test_save_legs_batch_preserves_previously_saved_days(tmp_path):
    """Regresión: save_legs(batch) NO debe borrar días de corridas previas.

    batch_id particiona VIAJEROS y abarca TODOS los días, así que un
    `DELETE WHERE batch_id = ?` pelado borra las filas de ese batch de los
    días ya procesados — pérdida de datos incremental (las etapas de corridas
    viejas desaparecen mientras viajes las conserva). El delete debe acotarse
    a los días presentes en df.
    """
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.ports import BatchSpec

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    batch = BatchSpec(batch_id=0, total_batches=1)

    # corrida previa: día A escrito bajo el batch 0
    adapter.save_legs(_sample_legs(), batch)  # dia = 2024-01-01

    # corrida incremental: día B bajo el MISMO batch 0
    day_b = _sample_legs()
    day_b["dia"] = "2024-01-02"
    day_b["id"] = [3, 4]
    adapter.save_legs(day_b, batch)

    result = adapter.get_legs()
    assert set(result["dia"]) == {"2024-01-01", "2024-01-02"}, (
        "save_legs(batch) borró el día previo: el DELETE por batch_id no está "
        "acotado por día (pérdida de datos incremental)"
    )
    assert len(result) == 4


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

    # Auditoría 2026-07-18: los índices ART son puro costo (no aceleran ninguna query,
    # se mantienen en cada escritura). end_leg_destination_updates ya NO los recrea; el
    # UPDATE de destinos corre correcto igual (verificado arriba) apoyándose en zonemap.
    idx = adapter._conn.execute(
        "SELECT index_name FROM duckdb_indexes() "
        "WHERE table_name = 'etapas' AND index_name = 'idx_etapas_dia_od_validado'"
    ).fetchall()
    assert not idx, "idx_etapas_dia_od_validado ya no debe recrearse (política sin índices)"


def test_update_leg_destinations_from_parquet_es_day_scoped(tmp_path):
    """Regresión #8: el write-back de destinos reescribe SOLO los días del parquet
    (run-days). Los días congelados (sin fila staged) quedan intactos; los run-days
    toman los destinos del parquet (COALESCE) — bit-idéntico al rebuild viejo, pero
    O(run-days) en vez de O(acumulado).
    """
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    # día A (congelado) + día B (run-day)
    day_a = _sample_legs()  # dia 2024-01-01, ids 1,2, od_validado 1
    adapter.save_legs(day_a)
    day_b = _sample_legs()
    day_b["dia"] = "2024-01-02"
    day_b["id"] = [3, 4]
    adapter.save_legs(day_b)
    a_h3d = day_a.set_index("id").loc[1, "h3_d"]

    # parquet con destinos de SOLO el día B (run-day)
    upd = pd.DataFrame({
        "id": [3, 4],
        "dia": ["2024-01-02", "2024-01-02"],
        "h3_d": ["882a100d99fffff", "882a100daafffff"],
        "od_validado": [0, 0],
        "etapa_validada": [0, 0],
    })
    pq = tmp_path / "dest.parquet"
    upd.to_parquet(pq, index=False)

    adapter.update_leg_destinations_from_parquet(str(pq))

    res = adapter.get_legs().set_index("id")
    assert len(res) == 4  # row count preservado
    # día B (run-day) tomó los destinos del parquet
    assert res.loc[3, "h3_d"] == "882a100d99fffff"
    assert res.loc[3, "od_validado"] == 0
    assert res.loc[3, "etapa_validada"] == 0
    # día A (congelado, no estaba en el parquet) INTACTO
    assert res.loc[1, "dia"] == "2024-01-01"
    assert res.loc[1, "od_validado"] == 1
    assert res.loc[1, "h3_d"] == a_h3d


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
            "tot_km_route": [12.0],
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
