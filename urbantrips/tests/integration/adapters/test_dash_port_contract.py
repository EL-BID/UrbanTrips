import pandas as pd
import pytest

from urbantrips.storage.ports import DashPort


def _sample_indicator() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "desc_dia": ["2024-01-01"],
            "tipo_dia": ["laboral"],
            "Titulo": ["Total viajes"],
            "orden": [1],
            "Indicador": ["viajes_totales"],
            "Valor": ["12345"],
        }
    )


@pytest.fixture(params=["memory", "duckdb"])
def dash_adapter(request, tmp_path) -> DashPort:
    if request.param == "memory":
        from urbantrips.storage.adapters.memory.adapters import InMemoryDashAdapter

        return InMemoryDashAdapter()

    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter

    return DuckDBDashAdapter(tmp_path / "contract_dash.duckdb")


def test_dash_port_contract_indicator_roundtrip(dash_adapter):
    assert isinstance(dash_adapter, DashPort)

    indicator = _sample_indicator()
    dash_adapter.save_indicator(indicator, "indicadores")

    result = dash_adapter.get_indicator("indicadores")
    assert len(result) == 1
    assert result.iloc[0]["Titulo"] == "Total viajes"
    assert "indicadores" in dash_adapter.list_indicators()


def test_dash_port_contract_missing_indicator_returns_empty_frame(dash_adapter):
    result = dash_adapter.get_indicator("viajes_hora")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_dash_port_contract_append_raw_is_readable_as_indicator(dash_adapter):
    first = pd.DataFrame({"desc_dia": ["2024-01-01"], "tipo_dia": ["laboral"], "Hora": [8], "Viajes": [10], "Modo": ["bus"]})
    second = pd.DataFrame({"desc_dia": ["2024-01-01"], "tipo_dia": ["laboral"], "Hora": [9], "Viajes": [20], "Modo": ["bus"]})

    dash_adapter.append_raw(first, "viajes_hora")
    dash_adapter.append_raw(second, "viajes_hora")

    result = dash_adapter.get_indicator("viajes_hora").sort_values("Hora").reset_index(drop=True)
    assert result["Viajes"].tolist() == [10, 20]


def test_dash_port_contract_query_reads_existing_table(dash_adapter):
    indicator = _sample_indicator()
    dash_adapter.save_indicator(indicator, "indicadores")

    result = dash_adapter.query("SELECT * FROM indicadores")

    assert result["Titulo"].tolist() == ["Total viajes"]


def test_dash_port_contract_rejects_invalid_raw_table_names(dash_adapter):
    raw = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError, match="Invalid table name"):
        dash_adapter.append_raw(raw, "viajes_hora; DROP TABLE indicadores")
