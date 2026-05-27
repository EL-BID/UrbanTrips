# urbantrips/tests/integration/adapters/test_duckdb_general.py
import pytest
import pandas as pd


def test_register_and_get_completed_runs(tmp_path):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    adapter = DuckDBGeneralAdapter(tmp_path / "general.duckdb")
    adapter.register_run("corrida_01", "transactions_completed")
    result = adapter.get_completed_runs()
    assert len(result) == 1
    assert result.iloc[0]["corrida"] == "corrida_01"
    assert result.iloc[0]["process"] == "transactions_completed"


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
