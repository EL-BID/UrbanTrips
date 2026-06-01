import duckdb
import geopandas as gpd
import pandas as pd
import pytest


def test_dash_utils_load_table_sql_reads_duckdb_and_normalizes(monkeypatch, tmp_path):
    from urbantrips.dashboard import dash_utils

    db_path = tmp_path / "dash.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE indicadores AS
            SELECT
                'weekday' AS dia,
                'bus' AS modo,
                'LINEA -' AS nombre_linea
            """
        )

    monkeypatch.setattr(
        dash_utils,
        "iniciar_conexion_db",
        lambda tipo="dash", alias_db="": duckdb.connect(str(db_path)),
    )

    result = dash_utils._load_table_sql("indicadores")

    assert result["dia"].tolist() == ["Hábil"]
    assert result["modo"].tolist() == ["Bus"]
    assert result["nombre_linea"].tolist() == ["LINEA"]


def test_dash_utils_load_table_sql_converts_wkt(monkeypatch, tmp_path):
    from urbantrips.dashboard import dash_utils

    db_path = tmp_path / "dash.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE geoms AS SELECT 'POINT (0 1)' AS wkt")

    monkeypatch.setattr(
        dash_utils,
        "iniciar_conexion_db",
        lambda tipo="dash", alias_db="": duckdb.connect(str(db_path)),
    )

    result = dash_utils._load_table_sql("geoms")

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.geometry.iloc[0].x == 0
    assert result.geometry.iloc[0].y == 1


def test_dash_utils_load_table_sql_rejects_invalid_default_table_name():
    from urbantrips.dashboard import dash_utils

    with pytest.raises(ValueError, match="Invalid table name"):
        dash_utils._load_table_sql("indicadores; DROP TABLE indicadores")


def test_dash_utils_load_table_sql_missing_table_returns_empty(monkeypatch, tmp_path):
    from urbantrips.dashboard import dash_utils

    db_path = tmp_path / "dash.duckdb"
    duckdb.connect(str(db_path)).close()
    monkeypatch.setattr(
        dash_utils,
        "iniciar_conexion_db",
        lambda tipo="dash", alias_db="": duckdb.connect(str(db_path)),
    )

    result = dash_utils._load_table_sql("missing_table")

    assert isinstance(result, pd.DataFrame)
    assert result.empty

