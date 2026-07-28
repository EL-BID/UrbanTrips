"""dashboard_ctx + write_access: el ciclo lectura -> escritura -> lectura.

Es el nudo del diseño: DuckDB rechaza conexiones read-only y read-write al mismo
archivo dentro de un proceso, así que la ventana de escritura tiene que cerrar
los StorageContext de lectura que estén abiertos.
"""

import duckdb
import pandas as pd
import pytest

import urbantrips.dashboard as dashboard
from urbantrips.storage import access


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    """Cuatro bases DuckDB vacías, resueltas por get_db_path."""
    for tipo in ("data", "insumos", "dash", "general"):
        duckdb.connect(str(tmp_path / f"{tipo}.duckdb")).close()

    monkeypatch.setattr(
        "urbantrips.utils.utils.get_db_path",
        lambda tipo="data", alias_db="": tmp_path / f"{tipo}.duckdb",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_mode():
    access.set_read_only_mode(None)
    yield
    access.set_read_only_mode(None)
    dashboard._release_open_ctxs()


def test_dashboard_ctx_abre_read_only_dentro_de_streamlit(dbs, monkeypatch):
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)

    with dashboard.dashboard_ctx() as ctx:
        assert ctx.data._read_only
        assert ctx.insumos._read_only
        assert ctx.dash._read_only
        assert ctx.general._read_only


def test_dashboard_ctx_cierra_al_salir(dbs, monkeypatch):
    """Cerrar libera el lock del archivo: es lo que desbloquea al pipeline."""
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)

    with dashboard.dashboard_ctx():
        pass

    assert dashboard._open_ctxs == []
    # Ya no hay lectores: se puede abrir en escritura.
    conn = duckdb.connect(str(dbs / "dash.duckdb"))
    conn.close()


def test_write_access_cierra_los_ctx_de_lectura_abiertos(dbs, monkeypatch):
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)

    lector = dashboard.get_dashboard_ctx()
    dashboard._open_ctxs.append(lector)
    assert lector.dash._read_only

    with access.write_access("test"):
        # write_access cerró al lector, así que ahora sí se puede escribir.
        with dashboard.dashboard_ctx() as ctx:
            assert not ctx.dash._read_only
            ctx.dash.save_indicator(pd.DataFrame({"a": [1, 2, 3]}), "indicador_test")

    # De vuelta en solo lectura, y el dato escrito se ve.
    assert access.read_only_mode() is True
    with dashboard.dashboard_ctx() as ctx:
        assert len(ctx.dash.get_indicator("indicador_test")) == 3
