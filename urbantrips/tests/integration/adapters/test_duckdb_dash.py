# urbantrips/tests/integration/adapters/test_duckdb_dash.py
import pytest
import pandas as pd


def _sample_indicator() -> pd.DataFrame:
    return pd.DataFrame({
        "desc_dia": ["2024-01-01"],
        "tipo_dia": ["laboral"],
        "Titulo": ["Total viajes"],
        "orden": [1],
        "Indicador": ["viajes_totales"],
        "Valor": ["12345"],
    })


def test_save_and_get_indicator(tmp_path):
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    adapter = DuckDBDashAdapter(tmp_path / "dash.duckdb")
    df = _sample_indicator()
    adapter.save_indicator(df, "indicadores")
    result = adapter.get_indicator("indicadores")
    assert len(result) == 1
    assert result.iloc[0]["Titulo"] == "Total viajes"


def test_list_indicators_empty(tmp_path):
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    adapter = DuckDBDashAdapter(tmp_path / "dash.duckdb")
    result = adapter.list_indicators()
    assert isinstance(result, list)


def test_list_indicators_after_save(tmp_path):
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    adapter = DuckDBDashAdapter(tmp_path / "dash.duckdb")
    adapter.save_indicator(_sample_indicator(), "indicadores")
    result = adapter.list_indicators()
    assert "indicadores" in result


def test_get_indicator_empty_returns_empty_df(tmp_path):
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    adapter = DuckDBDashAdapter(tmp_path / "dash.duckdb")
    result = adapter.get_indicator("viajes_hora")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_satisfies_dash_port(tmp_path):
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    from urbantrips.storage.ports import DashPort
    adapter = DuckDBDashAdapter(tmp_path / "dash.duckdb")
    assert isinstance(adapter, DashPort)
