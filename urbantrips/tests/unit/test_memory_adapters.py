# urbantrips/tests/unit/test_memory_adapters.py
import tempfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString


def test_in_memory_data_adapter_legs_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter
    adapter = InMemoryDataAdapter()
    legs = pd.DataFrame({"id": [1, 2], "id_tarjeta": ["T1", "T2"], "dia": ["2024-01-01", "2024-01-01"]})
    adapter.save_legs(legs)
    result = adapter.get_legs()
    assert len(result) == 2


def test_in_memory_data_adapter_replace_legs_for_days():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter

    adapter = InMemoryDataAdapter(
        etapas=pd.DataFrame(
            {
                "id": [1, 2],
                "dia": ["2024-01-01", "2024-01-02"],
                "h3_d": ["old", "keep"],
            }
        )
    )
    adapter.replace_legs_for_days(
        pd.DataFrame({"id": [3], "dia": ["2024-01-01"], "h3_d": ["new"]}),
        ["2024-01-01"],
    )
    result = adapter.get_legs()
    assert result["h3_d"].tolist() == ["keep", "new"]


def test_in_memory_data_adapter_has_rows():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter

    adapter = InMemoryDataAdapter(
        etapas=pd.DataFrame({"id": [1, 2], "dia": ["2024-01-01", "2024-01-02"]}),
        gps=pd.DataFrame(columns=["id"]),
    )

    assert adapter.has_rows("etapas")
    assert adapter.has_rows("etapas", "dia = '2024-01-02'")
    assert not adapter.has_rows("etapas", "dia = '2024-01-03'")
    assert not adapter.has_rows("gps")
    assert not adapter.has_rows("missing_table")


def test_in_memory_data_adapter_user_batches():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter
    from urbantrips.storage.ports import BatchSpec
    adapter = InMemoryDataAdapter()
    batches = adapter.get_user_batches(3)
    assert len(batches) == 3
    assert batches[0] == BatchSpec(0, 3)


def test_in_memory_insumo_adapter_distances_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryInsumoAdapter
    adapter = InMemoryInsumoAdapter()
    dist = pd.DataFrame({"h3_o": ["abc"], "h3_d": ["def"], "distance_h3": [500.0]})
    adapter.save_distances(dist)
    result = adapter.get_distances()
    assert len(result) == 1


def test_in_memory_dash_adapter_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDashAdapter
    adapter = InMemoryDashAdapter()
    df = pd.DataFrame({"Titulo": ["x"], "Valor": ["1"]})
    adapter.save_indicator(df, "indicadores")
    result = adapter.get_indicator("indicadores")
    assert len(result) == 1
    assert "indicadores" in adapter.list_indicators()


def test_in_memory_general_adapter_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryGeneralAdapter
    adapter = InMemoryGeneralAdapter()
    adapter.register_run("r1", "transactions_completed")
    result = adapter.get_completed_runs()
    assert len(result) == 1


def test_in_memory_adapters_satisfy_ports():
    from urbantrips.storage.adapters.memory.adapters import (
        InMemoryDataAdapter, InMemoryInsumoAdapter,
        InMemoryDashAdapter, InMemoryGeneralAdapter,
    )
    from urbantrips.storage.ports import DataPort, InsumoPort, DashPort, GeneralPort
    assert isinstance(InMemoryDataAdapter(), DataPort)
    assert isinstance(InMemoryInsumoAdapter(), InsumoPort)
    assert isinstance(InMemoryDashAdapter(), DashPort)
    assert isinstance(InMemoryGeneralAdapter(), GeneralPort)


def test_indicators_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter
    adapter = InMemoryDataAdapter()
    df = pd.DataFrame({
        "dia": ["2024-01-01"], "detalle": ["total"],
        "indicador": [42.0], "tabla": ["etapas_expandidas"], "nivel": [0],
    })
    adapter.save_indicators(df)
    result = adapter.get_indicators()
    assert len(result) == 1
    assert result["indicador"].iloc[0] == 42.0


def test_vehicle_expansion_factors_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter
    adapter = InMemoryDataAdapter()
    df = pd.DataFrame({"id_linea": [1, 2], "factor": [1.5, 2.0]})
    adapter.save_vehicle_expansion_factors(df)
    result = adapter.get_vehicle_expansion_factors()
    assert len(result) == 2


def test_services_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter
    adapter = InMemoryDataAdapter()
    df = pd.DataFrame({"id_linea": [1], "dia": ["2024-01-01"], "id_service": [10]})
    adapter.save_services(df)
    result = adapter.get_services()
    assert len(result) == 1


def test_line_transactions_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter
    adapter = InMemoryDataAdapter()
    df = pd.DataFrame({"id_linea": [1, 2], "dia": ["2024-01-01", "2024-01-01"], "n_trx": [100, 200]})
    adapter.save_line_transactions(df)
    result = adapter.get_line_transactions()
    assert len(result) == 2


# ── DuckDB adapter tests ──────────────────────────────────────────────────────

@pytest.fixture
def tmp_adapter(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    return DuckDBDataAdapter(tmp_path / "test_data.duckdb")


def test_save_raw_chunk_appends_rows(tmp_adapter):
    chunk = pd.DataFrame({
        "id_original": ["A", "B"],
        "id_tarjeta": ["001", "002"],
        "dia": ["2022-01-01", "2022-01-01"],
        "tiempo": ["08:00", "09:00"],
        "hora": [8, 9],
        "modo": ["autobus", "autobus"],
        "id_linea": [1, 1],
        "id_ramal": [1, 1],
        "interno": [10, 10],
        "orden_trx": [1, 2],
        "genero": ["-", "-"],
        "tarifa": ["-", "-"],
        "latitud": [-34.6, -34.7],
        "longitud": [-58.4, -58.5],
        "fecha_ts": [1641024000, 1641027600],
        "factor_expansion_raw": [1.0, 1.0],
    })
    tmp_adapter.save_raw_chunk(chunk)
    tmp_adapter.save_raw_chunk(chunk)
    result = tmp_adapter.query("SELECT COUNT(*) as n FROM transacciones_raw")
    assert result["n"].iloc[0] == 4


def test_clear_raw_empties_staging_table(tmp_adapter):
    chunk = pd.DataFrame({
        "id_original": ["A"], "id_tarjeta": ["001"], "dia": ["2022-01-01"],
        "tiempo": ["08:00"], "hora": [8], "modo": ["autobus"],
        "id_linea": [1], "id_ramal": [1], "interno": [10], "orden_trx": [1],
        "genero": ["-"], "tarifa": ["-"], "latitud": [-34.6], "longitud": [-58.4],
        "fecha_ts": [1641024000], "factor_expansion_raw": [1.0],
    })
    tmp_adapter.save_raw_chunk(chunk)
    tmp_adapter.clear_raw()
    result = tmp_adapter.query("SELECT COUNT(*) as n FROM transacciones_raw")
    assert result["n"].iloc[0] == 0


def test_standardize_raw_populates_transacciones_with_batch_id(tmp_adapter):
    chunk = pd.DataFrame({
        "id_original": ["A", "B"],
        "id_tarjeta": ["card_0", "card_1"],
        "dia": ["2022-01-01", "2022-01-01"],
        "tiempo": ["08:00", "09:00"],
        "hora": [8, 9],
        "modo": ["autobus", "autobus"],
        "id_linea": [1, 2],
        "id_ramal": [1, 2],
        "interno": [10, 20],
        "orden_trx": [1, 1],
        "genero": ["-", "-"],
        "tarifa": ["-", "-"],
        "latitud": [-34.6, -34.7],
        "longitud": [-58.4, -58.5],
        "fecha_ts": [1641024000, 1641027600],
        "factor_expansion_raw": [1.0, 1.0],
    })
    tmp_adapter.save_raw_chunk(chunk)
    tmp_adapter.standardize_raw_to_transacciones(n_batches=2, id_offset=0)
    result = tmp_adapter.query("SELECT id_tarjeta, batch_id FROM transacciones ORDER BY id_tarjeta")
    assert len(result) == 2
    assert result["batch_id"].notna().all()
    assert set(result["batch_id"]).issubset({0, 1})


def test_get_transactions_for_batch_returns_only_matching_batch(tmp_adapter):
    from urbantrips.storage.ports import BatchSpec
    # card_1_a hashes to batch 1, card_1_b hashes to batch 0 under DuckDB hash() % 2
    # (verified: hash('card_1_a') % 2 == 1, hash('card_1_b') % 2 == 0)
    chunk = pd.DataFrame({
        "id_original": ["A", "B", "C"],
        "id_tarjeta": ["card_1_a", "card_1_b", "card_1_a"],
        "dia": ["2022-01-01", "2022-01-01", "2022-01-02"],
        "tiempo": ["08:00", "09:00", "08:00"],
        "hora": [8, 9, 8],
        "modo": ["autobus", "autobus", "autobus"],
        "id_linea": [1, 2, 1],
        "id_ramal": [1, 2, 1],
        "interno": [10, 20, 10],
        "orden_trx": [1, 1, 1],
        "genero": ["-", "-", "-"],
        "tarifa": ["-", "-", "-"],
        "latitud": [-34.6, -34.7, -34.6],
        "longitud": [-58.4, -58.5, -58.4],
        "fecha_ts": [1641024000, 1641027600, 1641110400],
        "factor_expansion_raw": [1.0, 1.0, 1.0],
    })
    tmp_adapter.save_raw_chunk(chunk)
    tmp_adapter.standardize_raw_to_transacciones(n_batches=2, id_offset=0)

    all_trx = tmp_adapter.query("SELECT id_tarjeta, batch_id FROM transacciones")
    card_a_batch = all_trx.loc[all_trx.id_tarjeta == "card_1_a", "batch_id"].iloc[0]

    result = tmp_adapter.get_transactions_for_batch(BatchSpec(batch_id=int(card_a_batch), total_batches=2))
    assert (result["id_tarjeta"] == "card_1_a").all()
    assert len(result) == 2  # card_1_a appears on both days
