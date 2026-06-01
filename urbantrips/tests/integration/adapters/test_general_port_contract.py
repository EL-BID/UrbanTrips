import pandas as pd
import pytest

from urbantrips.storage.ports import GeneralPort


@pytest.fixture(params=["memory", "duckdb"])
def general_adapter(request, tmp_path) -> GeneralPort:
    if request.param == "memory":
        from urbantrips.storage.adapters.memory.adapters import InMemoryGeneralAdapter

        return InMemoryGeneralAdapter()

    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter

    return DuckDBGeneralAdapter(tmp_path / "contract_general.duckdb")


def test_general_port_contract_run_lifecycle(general_adapter):
    assert isinstance(general_adapter, GeneralPort)
    assert general_adapter.get_completed_runs().empty
    assert not general_adapter.run_exists("corrida_01")

    general_adapter.register_run("corrida_01", "transactions_completed")

    runs = general_adapter.get_completed_runs()
    assert len(runs) == 1
    assert runs.iloc[0]["corrida"] == "corrida_01"
    assert runs.iloc[0]["process"] == "transactions_completed"
    assert general_adapter.run_exists("corrida_01")

    general_adapter.clear_runs()
    assert general_adapter.get_completed_runs().empty
    assert not general_adapter.run_exists("corrida_01")


def test_general_port_contract_raw_append_roundtrip(general_adapter):
    first = pd.DataFrame({"id": [1], "value": ["a"]})
    second = pd.DataFrame({"id": [2], "value": ["b"]})

    general_adapter.append_raw(first, "contract_raw")
    general_adapter.append_raw(second, "contract_raw")

    result = general_adapter.get_raw("contract_raw").sort_values("id").reset_index(drop=True)
    expected = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_general_port_contract_missing_raw_returns_empty_frame(general_adapter):
    result = general_adapter.get_raw("missing_raw")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_general_port_contract_query_reads_existing_table(general_adapter):
    raw = pd.DataFrame({"id": [1], "value": ["a"]})
    general_adapter.append_raw(raw, "contract_raw")

    result = general_adapter.query("SELECT * FROM contract_raw")

    pd.testing.assert_frame_equal(result[raw.columns], raw)


def test_general_port_contract_rejects_invalid_raw_table_names(general_adapter):
    raw = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError, match="Invalid table name"):
        general_adapter.append_raw(raw, "contract_raw; DROP TABLE corridas")
