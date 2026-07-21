"""Adapters DuckDB en modo solo lectura.

Cubre lo que hace posible tener varios dashboards leyendo la misma base:
no aplicar DDL al abrir, tolerar tablas ausentes, y el ciclo
lectura -> ventana de escritura -> lectura.
"""

import duckdb
import pytest

from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter


@pytest.fixture
def db_vacia(tmp_path):
    """Base DuckDB existente pero sin ninguna tabla del esquema."""
    path = tmp_path / "test.duckdb"
    duckdb.connect(str(path)).close()
    return path


@pytest.mark.parametrize(
    "adapter_cls", [DuckDBDashAdapter, DuckDBGeneralAdapter, DuckDBInsumoAdapter]
)
def test_read_only_no_aplica_esquema(db_vacia, adapter_cls):
    adapter = adapter_cls(db_vacia, read_only=True)
    try:
        tablas = adapter._conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    finally:
        adapter.close()
    assert tablas == [], f"{adapter_cls.__name__} creó tablas en modo solo lectura"


def test_read_only_tolera_tablas_ausentes(db_vacia):
    adapter = DuckDBGeneralAdapter(db_vacia, read_only=True)
    try:
        assert adapter.get_completed_runs().empty
    finally:
        adapter.close()


def test_dos_lectores_simultaneos(db_vacia):
    """Varios adapters de solo lectura pueden convivir sobre el mismo archivo."""
    a = DuckDBDashAdapter(db_vacia, read_only=True)
    b = DuckDBDashAdapter(db_vacia, read_only=True)
    try:
        assert a.get_indicator("no_existe").empty
        assert b.get_indicator("no_existe").empty
    finally:
        a.close()
        b.close()


def test_ciclo_lectura_escritura_lectura(db_vacia):
    """Cerrar los lectores permite escalar a escritura, y el dato se ve después.

    DuckDB rechaza mezclar conexiones read-only y read-write al mismo archivo en
    un proceso, así que la ventana de escritura tiene que cerrar los lectores.
    """
    import pandas as pd

    lector = DuckDBDashAdapter(db_vacia, read_only=True)
    assert lector.get_indicator("indicador_test").empty

    # Con el lector abierto no se puede escalar a escritura.
    with pytest.raises(duckdb.Error):
        DuckDBDashAdapter(db_vacia, read_only=False)

    lector.close()

    escritor = DuckDBDashAdapter(db_vacia, read_only=False)
    escritor.save_indicator(pd.DataFrame({"a": [1, 2]}), "indicador_test")
    escritor.close()

    lector = DuckDBDashAdapter(db_vacia, read_only=True)
    try:
        assert len(lector.get_indicator("indicador_test")) == 2
    finally:
        lector.close()
