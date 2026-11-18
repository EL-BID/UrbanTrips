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



def _zonificaciones_fake():
    return pd.DataFrame(
        {
            "zona": ["res_7", "res_7", "res_7", "depto"],
            "id": ["a", "b", "c", "1"],
        }
    )


def test_traigo_lista_zonas_etapas_solo_zonas_con_viajes(monkeypatch):
    from urbantrips.dashboard import dash_utils

    monkeypatch.setattr(
        dash_utils, "levanto_tabla_sql", lambda *a, **k: _zonificaciones_fake()
    )
    # ids numericos en una capa y texto en la otra: se comparan como texto
    monkeypatch.setattr(
        dash_utils,
        "_zonas_con_viajes",
        lambda: pd.DataFrame({"zona": ["res_7", "depto"], "id": ["b", 1]}),
    )

    result = dash_utils.traigo_lista_zonas.__wrapped__("etapas")

    assert list(zip(result.zona, result.Nombre)) == [("depto", "1"), ("res_7", "b")]


def test_traigo_lista_zonas_sin_datos_de_viajes_lista_todo(monkeypatch):
    from urbantrips.dashboard import dash_utils

    monkeypatch.setattr(
        dash_utils, "levanto_tabla_sql", lambda *a, **k: _zonificaciones_fake()
    )
    monkeypatch.setattr(
        dash_utils, "_zonas_con_viajes", lambda: pd.DataFrame(columns=["zona", "id"])
    )

    assert len(dash_utils.traigo_lista_zonas.__wrapped__("etapas")) == 4


def test_traigo_lista_zonas_poligonos_no_filtra(monkeypatch):
    from urbantrips.dashboard import dash_utils

    monkeypatch.setattr(
        dash_utils, "levanto_tabla_sql", lambda *a, **k: _zonificaciones_fake()
    )

    def _no_llamar():
        raise AssertionError("no deberia consultar viajes")

    monkeypatch.setattr(dash_utils, "_zonas_con_viajes", _no_llamar)

    assert len(dash_utils.traigo_lista_zonas.__wrapped__("poligonos")) == 4


def test_zonas_con_viajes_cuenta_origen_transferencias_y_destino(
    monkeypatch, tmp_path
):
    from urbantrips.dashboard import dash_utils

    db_path = tmp_path / "dash.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE equivalencias_zonas AS
            SELECT * FROM (VALUES
                ('h_o', 'res_7', 'a'), ('h_t1', 'res_7', 'b'),
                ('h_t2', 'res_7', 'c'), ('h_d', 'res_7', 'd'),
                ('h_x', 'res_7', 'sin_viajes')
            ) t(h3, zona, id)
            """
        )
        conn.execute(
            """
            CREATE TABLE chains_norm AS
            SELECT 'h_o' AS h3_inicio_norm, 'h_t1' AS h3_transfer1_norm,
                   'h_t2' AS h3_transfer2_norm, 'h_d' AS h3_fin_norm
            UNION ALL
            SELECT 'h_o', NULL, NULL, 'h_d'
            """
        )

    monkeypatch.setattr(
        dash_utils,
        "iniciar_conexion_db",
        lambda tipo="dash", alias_db="": duckdb.connect(str(db_path)),
    )
    monkeypatch.setattr(dash_utils, "asegurar_equivalencias_dash", lambda: True)

    result = dash_utils._zonas_con_viajes()

    assert sorted(result["id"]) == ["a", "b", "c", "d"]
