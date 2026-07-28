"""Modo de acceso a las bases: solo lectura por defecto en el dashboard."""

import pytest

from urbantrips.storage import access


@pytest.fixture(autouse=True)
def _reset_mode():
    access.set_read_only_mode(None)
    yield
    access.set_read_only_mode(None)


def test_default_is_write_mode_outside_streamlit(monkeypatch):
    monkeypatch.delenv("URBANTRIPS_DB_READ_ONLY", raising=False)
    assert access.read_only_mode() is False


def test_env_var_forces_read_only(monkeypatch):
    monkeypatch.setenv("URBANTRIPS_DB_READ_ONLY", "1")
    assert access.read_only_mode() is True


def test_streamlit_script_run_implies_read_only(monkeypatch):
    monkeypatch.delenv("URBANTRIPS_DB_READ_ONLY", raising=False)
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)
    assert access.read_only_mode() is True


def test_write_access_opens_a_write_window(monkeypatch):
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)
    assert access.read_only_mode() is True
    with access.write_access("test"):
        assert access.read_only_mode() is False
    assert access.read_only_mode() is True


def test_write_access_restores_mode_on_error(monkeypatch):
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)
    with pytest.raises(ValueError):
        with access.write_access("test"):
            raise ValueError("boom")
    assert access.read_only_mode() is True


def test_write_access_releases_read_connections(monkeypatch):
    """Antes de escalar a escritura hay que cerrar las conexiones de lectura.

    DuckDB no admite conexiones read-only y read-write al mismo archivo dentro
    de un mismo proceso.
    """
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)
    llamadas = []
    monkeypatch.setattr(access, "_release_hooks", [lambda: llamadas.append(1)])

    with access.write_access("test"):
        assert llamadas == [1]  # se libera al entrar
    assert llamadas == [1, 1]  # y al salir


def test_resolve_read_only_prefers_explicit_value(monkeypatch):
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)
    assert access.resolve_read_only(None) is True
    assert access.resolve_read_only(False) is False
    assert access.resolve_read_only(True) is True


def test_require_write_access_raises_in_read_only(monkeypatch):
    monkeypatch.setattr(access, "_in_streamlit_script_run", lambda: True)
    with pytest.raises(access.ReadOnlyModeError):
        access.require_write_access("guardar la tabla 'x'")

    with access.write_access("test"):
        access.require_write_access("guardar la tabla 'x'")  # no levanta


def test_retry_on_busy_translates_lock_error():
    intentos = []

    def _falla():
        intentos.append(1)
        raise OSError('Cannot open file "x.duckdb": held by PID 1728')

    with pytest.raises(access.DatabaseBusyError):
        access.retry_on_busy(_falla, "abrir x.duckdb", intentos=2, espera=0)
    assert len(intentos) == 2


def test_retry_on_busy_reraises_other_errors():
    def _falla():
        raise ValueError("otra cosa")

    with pytest.raises(ValueError):
        access.retry_on_busy(_falla, "abrir x.duckdb", intentos=3, espera=0)
