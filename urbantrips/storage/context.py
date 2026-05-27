# urbantrips/storage/context.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from urbantrips.storage.ports import DataPort, InsumoPort, DashPort, GeneralPort


@dataclass
class StorageContext:
    """Holds one adapter per port. Built once at startup, injected through the pipeline."""
    data: DataPort
    insumos: InsumoPort
    dash: DashPort
    general: GeneralPort


def build_storage_context(config, base_dir: Path | None = None) -> StorageContext:
    """Instantiate the right adapters based on config.storage_backend.

    Parameters
    ----------
    config : Config
        Typed config object (from urbantrips.config.config).
    base_dir : Path | None
        Project root for resolving DB file paths. Defaults to current directory.
    """
    base = base_dir or Path(".")
    backend = config.storage_backend

    if backend == "duckdb":
        from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
        from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
        from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
        from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter

        return StorageContext(
            data=DuckDBDataAdapter(config.db_path("data", base_dir=base)),
            insumos=DuckDBInsumoAdapter(config.db_path("insumos", base_dir=base)),
            dash=DuckDBDashAdapter(config.db_path("dash", base_dir=base)),
            general=DuckDBGeneralAdapter(config.db_path("general", base_dir=base)),
        )

    if backend == "sqlite":
        raise NotImplementedError(
            "SQLite adapter not yet implemented. See Plan 2 (sqlite-adapters)."
        )

    if backend == "postgresql":
        raise NotImplementedError("PostgreSQL adapter not yet implemented.")

    raise ValueError(f"Unknown storage_backend: {backend!r}")
