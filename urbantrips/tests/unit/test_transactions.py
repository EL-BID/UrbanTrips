import pandas as pd
from urbantrips.datamodel.transactions import filtrar_transacciones_invalidas


def test_filtrar_transacciones_invalidas_removes_flagged_rows():
    df = pd.DataFrame({
        "id_tarjeta": ["A", "B", "C"],
        "tipo": ["valid", "invalid", "valid"],
        "valor": [1, 2, 3],
    })
    result = filtrar_transacciones_invalidas(df, {"tipo": ["invalid"]})
    assert len(result) == 2
    assert "B" not in result["id_tarjeta"].values


def test_filtrar_transacciones_invalidas_multiple_columns():
    df = pd.DataFrame({
        "id_tarjeta": ["A", "B", "C", "D"],
        "tipo": ["valid", "bad_type", "valid", "valid"],
        "modo": ["bus", "bus", "tren", "bus"],
    })
    result = filtrar_transacciones_invalidas(df, {"tipo": ["bad_type"], "modo": ["tren"]})
    assert len(result) == 2
    assert set(result["id_tarjeta"]) == {"A", "D"}


def test_filtrar_transacciones_invalidas_empty_filter_keeps_all():
    df = pd.DataFrame({"id_tarjeta": ["A", "B"], "tipo": ["x", "y"]})
    result = filtrar_transacciones_invalidas(df, {})
    assert len(result) == 2
