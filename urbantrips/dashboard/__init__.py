from contextlib import contextmanager


# StorageContexts vivos construidos por dashboard_ctx(). write_access() los
# cierra antes de escalar a escritura: DuckDB no permite mezclar conexiones
# read-only y read-write al mismo archivo dentro de un mismo proceso.
_open_ctxs: list = []


def get_dashboard_ctx(read_only=None):
    """Build a StorageContext for dashboard use.

    Resolves the DB files like the rest of the dashboard
    (``utils.get_db_path`` / ``leer_alias``), so pages using a StorageContext
    and pages using ``utils.levanto_tabla_sql`` open the same databases.

    ``read_only=None`` delega en ``storage.access.read_only_mode()``: dentro de
    Streamlit abre en solo lectura (varios dashboards a la vez, sin bloquear al
    pipeline); dentro de ``write_access()`` abre en escritura.

    Preferir ``dashboard_ctx()`` (context manager) para que las conexiones se
    cierren al terminar y no queden tomando el lock del archivo.
    """
    from urbantrips.storage.access import retry_on_busy, resolve_read_only
    from urbantrips.storage.context import StorageContext
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    from urbantrips.utils.utils import get_db_path

    ro = resolve_read_only(read_only)

    def _build():
        # Se abren de a uno para poder cerrar los ya abiertos si alguno falla
        # (base tomada por otro proceso) y no dejar el lock tomado a medias.
        abiertos = []
        try:
            for tipo, cls in (
                ("data", DuckDBDataAdapter),
                ("insumos", DuckDBInsumoAdapter),
                ("dash", DuckDBDashAdapter),
                ("general", DuckDBGeneralAdapter),
            ):
                abiertos.append(cls(get_db_path(tipo), read_only=ro))
        except Exception:
            for adapter in abiertos:
                try:
                    adapter.close()
                except Exception:
                    pass
            raise
        return StorageContext(*abiertos)

    return retry_on_busy(_build, "abrir las bases del dashboard")


def close_ctx(ctx) -> None:
    """Cierra los cuatro adapters de un StorageContext (idempotente)."""
    if ctx is None:
        return
    for nombre in ("data", "insumos", "dash", "general"):
        adapter = getattr(ctx, nombre, None)
        close = getattr(adapter, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception:
            pass
    if ctx in _open_ctxs:
        _open_ctxs.remove(ctx)


@contextmanager
def dashboard_ctx(read_only=None):
    """StorageContext de vida corta: se cierra al salir del bloque.

    Cerrar libera el lock del archivo DuckDB, que es lo que permite que otro
    dashboard lea la misma base y que el pipeline pueda arrancar.
    """
    ctx = get_dashboard_ctx(read_only=read_only)
    _open_ctxs.append(ctx)
    try:
        yield ctx
    finally:
        close_ctx(ctx)


def _release_open_ctxs() -> None:
    """Cierra todos los contexts vivos (hook de storage.access.write_access)."""
    for ctx in list(_open_ctxs):
        close_ctx(ctx)


def _register_release_hook() -> None:
    from urbantrips.storage.access import register_release_hook

    register_release_hook(_release_open_ctxs)


_register_release_hook()
