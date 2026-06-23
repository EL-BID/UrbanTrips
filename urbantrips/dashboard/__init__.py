from pathlib import Path


def get_dashboard_ctx():
    """Build a StorageContext for dashboard use.

    Resolves the DB files the same way the rest of the dashboard does
    (``utils.get_db_path`` / ``leer_alias``), so the pages that build a
    StorageContext (e.g. 4_Herramientas, 6_Comparación) read and write the
    SAME databases as the pages that read via ``utils.levanto_tabla_sql``
    (e.g. 8_Estimar_demanda).

    Previously this built the context from ``config.db_path``, which resolves
    the data/dash alias from the run corrida (``corridas[0]``) and pointed at a
    different (often empty) DB than the rest of the dashboard — that mismatch
    is why supply/section queries failed with CatalogException.
    """
    from urbantrips.storage.context import StorageContext
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    from urbantrips.utils.utils import get_db_path

    return StorageContext(
        data=DuckDBDataAdapter(get_db_path("data")),
        insumos=DuckDBInsumoAdapter(get_db_path("insumos")),
        dash=DuckDBDashAdapter(get_db_path("dash")),
        general=DuckDBGeneralAdapter(get_db_path("general")),
    )
