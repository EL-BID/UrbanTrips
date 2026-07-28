# urbantrips/tests/integration/adapters/test_config_snapshot.py
"""Copia del yaml dentro de la base general.

Deja la base auto-descriptiva: con el alias alcanza para saber con qué config se
generaron los datos, aunque el archivo de `configs/` se edite o se borre después.
Lo consume el selector de corridas del dashboard.
"""
import duckdb
import pytest


_YAML = 'alias_db_insumos: "mi_corrida"\nresolucion_h3: 9\nepsg_m: 5347\n'


@pytest.fixture(params=["duckdb", "memoria"])
def adapter(request, tmp_path):
    """Los dos adapters tienen que comportarse igual (contrato del puerto)."""
    if request.param == "duckdb":
        from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
        a = DuckDBGeneralAdapter(tmp_path / "x_general.duckdb")
        yield a
        a.close()
    else:
        from urbantrips.storage.adapters.memory.adapters import InMemoryGeneralAdapter
        yield InMemoryGeneralAdapter()


def test_vacio_al_principio(adapter):
    df = adapter.get_config_snapshot()
    assert df.empty
    assert list(df.columns) == ["alias", "corrida", "archivo", "contenido", "date"]


def test_roundtrip_exacto(adapter):
    adapter.save_config_snapshot("mi_corrida", "lunes", "mi.yaml", _YAML)

    df = adapter.get_config_snapshot()
    assert len(df) == 1
    fila = df.iloc[0]
    assert fila["contenido"] == _YAML       # byte a byte, sin re-serializar
    assert fila["archivo"] == "mi.yaml"
    assert fila["alias"] == "mi_corrida"


def test_varias_corridas_conviven(adapter):
    adapter.save_config_snapshot("mi_corrida", "lunes", "mi.yaml", _YAML)
    adapter.save_config_snapshot("mi_corrida", "martes", "mi.yaml", _YAML)

    assert len(adapter.get_config_snapshot()) == 2


def test_recorrer_actualiza_sin_duplicar(adapter):
    """Upsert por (alias, corrida): re-correr no acumula filas."""
    adapter.save_config_snapshot("mi_corrida", "lunes", "mi.yaml", _YAML)
    adapter.save_config_snapshot("mi_corrida", "lunes", "mi.yaml", _YAML + "extra: 1\n")

    df = adapter.get_config_snapshot()
    assert len(df) == 1
    assert "extra: 1" in df.iloc[0]["contenido"]


def test_alias_nulo_no_rompe(adapter):
    """Las filas migradas de bases legacy tienen alias NULL."""
    adapter.save_config_snapshot(None, "sin_alias", "mi.yaml", _YAML)
    assert len(adapter.get_config_snapshot()) == 1


def test_base_sin_la_tabla_devuelve_vacio(tmp_path):
    """Bases anteriores a esta feature, abiertas en solo lectura por el dashboard.

    No se puede crear la tabla (es read-only), así que tiene que degradar a vacío
    en vez de tirar CatalogException.
    """
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter

    db = tmp_path / "vieja_general.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE corridas (corrida TEXT, process TEXT, date TEXT)")
    con.close()

    a = DuckDBGeneralAdapter(db, read_only=True)
    try:
        df = a.get_config_snapshot()
        assert df.empty
        assert list(df.columns) == ["alias", "corrida", "archivo", "contenido", "date"]
    finally:
        a.close()


def test_satisface_el_puerto(tmp_path):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    from urbantrips.storage.adapters.memory.adapters import InMemoryGeneralAdapter
    from urbantrips.storage.ports import GeneralPort

    a = DuckDBGeneralAdapter(tmp_path / "p_general.duckdb")
    try:
        assert isinstance(a, GeneralPort)
    finally:
        a.close()
    assert isinstance(InMemoryGeneralAdapter(), GeneralPort)
