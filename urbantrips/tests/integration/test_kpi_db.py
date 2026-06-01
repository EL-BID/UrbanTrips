import pandas as pd
from urbantrips.utils.utils import agrego_indicador, levanto_tabla_sql


def test_agrego_indicador_writes_to_indicadores(patched_db):
    """agrego_indicador aggregates by dia and writes to the indicadores table."""
    df = pd.DataFrame({
        "dia": ["2022-08-11", "2022-08-11", "2022-08-12"],
        "factor_expansion_linea": [1.0, 2.0, 3.0],
    })
    agrego_indicador(
        df_indicador=df,
        detalle="test_detalle",
        tabla="etapas",
        nivel=0,
        var="factor_expansion_linea",
        var_fex="factor_expansion_linea",
        aggfunc="sum",
    )
    result = levanto_tabla_sql("indicadores", tabla_tipo="data")
    assert len(result) >= 2
    assert "detalle" in result.columns
    assert "dia" in result.columns
    assert "2022-08-11" in set(result["dia"])
    assert "2022-08-12" in set(result["dia"])


def test_agrego_indicador_sum_values(patched_db):
    """Sum aggregation produces correct totals per day."""
    df = pd.DataFrame({
        "dia": ["2022-08-11", "2022-08-11"],
        "factor_expansion_linea": [10.0, 20.0],
    })
    agrego_indicador(
        df_indicador=df,
        detalle="sum_test",
        tabla="etapas",
        nivel=0,
        var="factor_expansion_linea",
        var_fex="factor_expansion_linea",
        aggfunc="sum",
    )
    result = levanto_tabla_sql("indicadores", tabla_tipo="data")
    row = result.loc[result["detalle"] == "sum_test"]
    assert len(row) == 1
    assert abs(row.iloc[0]["indicador"] - 30.0) < 0.01
