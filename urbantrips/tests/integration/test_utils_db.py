import pandas as pd
import pytest
from urbantrips.utils.utils import (
    guardar_tabla_sql,
    levanto_tabla_sql,
    delete_data_from_table_run_days,
    tabla_existe,
)


def test_guardar_levanto_roundtrip(patched_db):
    df = pd.DataFrame({"dia": ["2022-08-11", "2022-08-12"], "valor": [10, 20]})
    guardar_tabla_sql(df, "test_roundtrip", tabla_tipo="data", modo="append")
    result = levanto_tabla_sql("test_roundtrip", tabla_tipo="data")
    assert len(result) == 2
    assert sorted(result["dia"].tolist()) == ["2022-08-11", "2022-08-12"]


def test_levanto_tabla_sql_empty_returns_empty_dataframe(patched_db):
    result = levanto_tabla_sql("dias_ultima_corrida", tabla_tipo="data")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_levanto_tabla_sql_missing_table_returns_empty_dataframe(patched_db):
    result = levanto_tabla_sql("tabla_inexistente", tabla_tipo="data")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_levanto_tabla_sql_rejects_invalid_default_table_name(patched_db):
    with pytest.raises(ValueError, match="Invalid table name"):
        levanto_tabla_sql("test_roundtrip; DROP TABLE transacciones", tabla_tipo="data")


def test_guardar_tabla_sql_with_filtros_replaces_matching_rows(patched_db):
    df1 = pd.DataFrame({"dia": ["2022-08-11", "2022-08-12"], "valor": [1, 2]})
    guardar_tabla_sql(df1, "test_filtros", tabla_tipo="data", modo="append")

    df2 = pd.DataFrame({"dia": ["2022-08-11"], "valor": [99]})
    guardar_tabla_sql(
        df2, "test_filtros", tabla_tipo="data", modo="append",
        filtros={"dia": ["2022-08-11"]},
    )

    result = levanto_tabla_sql("test_filtros", tabla_tipo="data")
    assert len(result) == 2
    assert result.loc[result.dia == "2022-08-11", "valor"].iloc[0] == 99
    assert result.loc[result.dia == "2022-08-12", "valor"].iloc[0] == 2


def test_guardar_tabla_sql_rejects_invalid_table_name(patched_db):
    df = pd.DataFrame({"dia": ["2022-08-11"], "valor": [1]})

    with pytest.raises(ValueError, match="Invalid table name"):
        guardar_tabla_sql(
            df,
            "test_filtros; DROP TABLE transacciones",
            tabla_tipo="data",
            modo="append",
        )


def test_guardar_tabla_sql_rejects_invalid_filter_field(patched_db):
    df = pd.DataFrame({"dia": ["2022-08-11"], "valor": [1]})
    guardar_tabla_sql(df, "test_safe_filters", tabla_tipo="data", modo="append")

    with pytest.raises(ValueError, match="Invalid table name"):
        guardar_tabla_sql(
            df,
            "test_safe_filters",
            tabla_tipo="data",
            modo="append",
            filtros={"dia; DROP TABLE transacciones": ["2022-08-11"]},
        )


def test_levanto_tabla_sql_with_custom_query(patched_db):
    df = pd.DataFrame({"dia": ["2022-08-11", "2022-08-12"], "valor": [10, 20]})
    guardar_tabla_sql(df, "test_query", tabla_tipo="data", modo="append")

    result = levanto_tabla_sql(
        "test_query",
        tabla_tipo="data",
        query="SELECT * FROM test_query WHERE dia = '2022-08-11'",
    )
    assert len(result) == 1
    assert result.iloc[0]["dia"] == "2022-08-11"


def test_delete_data_from_table_run_days_removes_matching(patched_db):
    patched_db._conn.execute("INSERT INTO dias_ultima_corrida VALUES ('2022-08-11')")
    patched_db._conn.commit()

    df = pd.DataFrame({"dia": ["2022-08-11", "2022-08-12"], "valor": [1, 2]})
    guardar_tabla_sql(df, "test_delete", tabla_tipo="data", modo="append")

    delete_data_from_table_run_days("test_delete")

    result = levanto_tabla_sql("test_delete", tabla_tipo="data")
    assert len(result) == 1
    assert result.iloc[0]["dia"] == "2022-08-12"


def test_delete_data_from_table_run_days_leaves_other_days(patched_db):
    patched_db._conn.execute("INSERT INTO dias_ultima_corrida VALUES ('2022-08-11')")
    patched_db._conn.commit()

    df = pd.DataFrame({
        "dia": ["2022-08-10", "2022-08-11", "2022-08-12"],
        "valor": [0, 1, 2],
    })
    guardar_tabla_sql(df, "test_delete2", tabla_tipo="data", modo="append")
    delete_data_from_table_run_days("test_delete2")

    result = levanto_tabla_sql("test_delete2", tabla_tipo="data")
    assert set(result["dia"]) == {"2022-08-10", "2022-08-12"}


def test_tabla_existe_returns_true_for_existing(db_conn):
    assert tabla_existe(db_conn, "transacciones") is True


def test_tabla_existe_returns_false_for_missing(db_conn):
    assert tabla_existe(db_conn, "tabla_inexistente") is False
