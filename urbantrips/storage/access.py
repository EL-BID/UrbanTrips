"""Modo de acceso a las bases DuckDB (solo lectura vs escritura).

DuckDB tiene un único escritor por archivo y, ademas, dentro de un mismo proceso
no se pueden mezclar conexiones read-only y read-write al mismo archivo
(``ConnectionException: Can't open a connection to same database file with a
different configuration``). Por eso el modo de acceso es una decision *del
proceso*, no de cada llamada.

Reglas:

* El pipeline (``run_all_urbantrips.py``) corre siempre en modo escritura.
* El dashboard (Streamlit) corre en modo solo lectura, para que varios
  dashboards puedan leer la misma base a la vez y para no bloquear al pipeline.
* Las pocas escrituras del dashboard se hacen dentro de :func:`write_access`,
  que cierra las conexiones de solo lectura del proceso, escribe, y vuelve a
  modo lectura.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# None = autodetectar. True/False = forzado (lo usa write_access y los tests).
_override: bool | None = None

# Serializa las ventanas de escritura dentro del proceso: dos script-runs de
# Streamlit en paralelo no pueden estar escribiendo a la vez.
_write_lock = threading.RLock()

# Callables sin argumentos que cierran conexiones/adapters abiertos en modo
# lectura. write_access() las ejecuta antes de escalar a escritura.
_release_hooks: list = []

_TRUTHY = ("1", "true", "True", "yes", "on")


class DatabaseBusyError(RuntimeError):
    """La base esta tomada por otro proceso (otro dashboard o una corrida)."""


class ReadOnlyModeError(RuntimeError):
    """Se intento escribir estando en modo solo lectura."""


def _in_streamlit_script_run() -> bool:
    """True si estamos ejecutando dentro de un script-run de Streamlit."""
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # pragma: no cover - version de streamlit sin esa API
        return False


def read_only_mode() -> bool:
    """True si este proceso debe abrir las bases en solo lectura."""
    if _override is not None:
        return _override
    if os.environ.get("URBANTRIPS_DB_READ_ONLY") in _TRUTHY:
        return True
    return _in_streamlit_script_run()


def resolve_read_only(read_only: bool | None) -> bool:
    """Resuelve el parametro ``read_only`` de las funciones de conexion.

    ``None`` delega en :func:`read_only_mode`; un booleano explicito manda.
    """
    if read_only is None:
        return read_only_mode()
    return bool(read_only)


def set_read_only_mode(value: bool | None) -> None:
    """Fuerza el modo (o vuelve a autodeteccion con ``None``). Para tests."""
    global _override
    _override = value


def register_release_hook(hook) -> None:
    """Registra un callable que cierra conexiones abiertas en modo lectura."""
    if hook not in _release_hooks:
        _release_hooks.append(hook)


def _release_read_connections() -> None:
    for hook in list(_release_hooks):
        try:
            hook()
        except Exception:  # pragma: no cover - cerrar nunca debe romper
            logger.warning("Fallo al liberar conexiones de lectura", exc_info=True)


def require_write_access(what: str = "escribir") -> None:
    """Falla con un mensaje claro si el proceso esta en modo solo lectura."""
    if read_only_mode():
        raise ReadOnlyModeError(
            f"No se puede {what}: la base esta abierta en modo solo lectura. "
            "Las escrituras del dashboard tienen que hacerse dentro de "
            "`with write_access(...)`."
        )


def is_busy_error(exc: BaseException) -> bool:
    """True si la excepcion es 'el archivo lo tiene tomado otro proceso'."""
    msg = str(exc).lower()
    return "cannot open file" in msg or "could not set lock" in msg


def retry_on_busy(fn, descripcion: str, intentos: int = 3, espera: float = 2.0):
    """Ejecuta ``fn()`` reintentando si la base la tiene tomada otro proceso.

    Traduce el ``IOException`` de DuckDB a :class:`DatabaseBusyError` con un
    mensaje entendible en vez de dejar salir el traceback crudo.
    """
    ultimo = None
    for intento in range(1, intentos + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_busy_error(exc):
                raise
            ultimo = exc
            if intento < intentos:
                logger.info(
                    "Base ocupada (%s), reintento %d/%d", descripcion, intento, intentos
                )
                time.sleep(espera)
    raise DatabaseBusyError(
        f"La base esta ocupada por otro proceso (otro dashboard o una corrida "
        f"en curso) y no se pudo {descripcion} despues de {intentos} intentos."
    ) from ultimo


@contextmanager
def write_access(motivo: str = ""):
    """Ventana de escritura: cierra las conexiones de lectura y abre en RW.

    Al salir vuelve al modo anterior y limpia el cache de datos de Streamlit
    para que las lecturas siguientes vean lo recien escrito.
    """
    global _override
    with _write_lock:
        anterior = _override
        _release_read_connections()
        _override = False
        logger.info("Ventana de escritura abierta%s", f": {motivo}" if motivo else "")
        try:
            yield
        finally:
            _release_read_connections()
            _override = anterior
            _clear_streamlit_data_cache()
            logger.info("Ventana de escritura cerrada%s", f": {motivo}" if motivo else "")


def _clear_streamlit_data_cache() -> None:
    if "streamlit" not in sys.modules:
        return
    try:
        import streamlit as st

        st.cache_data.clear()
    except Exception:  # pragma: no cover - fuera de un runtime de streamlit
        logger.debug("No se pudo limpiar st.cache_data", exc_info=True)
