# urbantrips/tests/integration/adapters/test_duckdb_general.py
import pytest
import pandas as pd


def test_register_and_get_completed_runs(tmp_path):
    # register_run es el shim legacy: marca la corrida como COMPLETA en el log
    # nuevo (los 4 step_ts seteados). run_exists la ve.
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    adapter = DuckDBGeneralAdapter(tmp_path / "general.duckdb")
    adapter.register_run("corrida_01", "transactions_completed")
    result = adapter.get_completed_runs()
    assert len(result) == 1
    assert result.iloc[0]["corrida"] == "corrida_01"
    assert pd.notna(result.iloc[0]["ingest_ts"])
    assert pd.notna(result.iloc[0]["dashboard_ts"])
    assert adapter.run_exists("corrida_01")


def test_get_completed_runs_empty_on_new_db(tmp_path):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    adapter = DuckDBGeneralAdapter(tmp_path / "general.duckdb")
    result = adapter.get_completed_runs()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_satisfies_general_port(tmp_path):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    from urbantrips.storage.ports import GeneralPort
    adapter = DuckDBGeneralAdapter(tmp_path / "general.duckdb")
    assert isinstance(adapter, GeneralPort)
