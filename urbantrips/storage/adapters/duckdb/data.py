# urbantrips/storage/adapters/duckdb/data.py
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.ports import BatchSpec
from urbantrips.storage.schema import data as schema

logger = logging.getLogger(__name__)


def _resolve_memory_limit(configured: str | None) -> str:
    """Return a DuckDB memory_limit string.

    Priority:
    1. Value passed explicitly (e.g. from tests).
    2. ``duckdb.memory_limit`` in configs/tuning.yaml.
    3. Auto-computed as 25% of total system RAM (floor 1 GB).
    """
    if configured:
        return configured

    try:
        from urbantrips.utils.utils import leer_configs_tuning
        tuning_val = leer_configs_tuning().get("duckdb", {}).get("memory_limit")
        if tuning_val:
            logger.info("[DuckDB] memory_limit=%s (from tuning.yaml)", tuning_val)
            return tuning_val
    except Exception:
        pass

    try:
        import psutil
        total_gb = psutil.virtual_memory().total / 1e9
        limit_gb = max(round(total_gb * 0.25), 1)
        limit = f"{limit_gb}GB"
    except Exception:
        limit = "4GB"
    logger.info("[DuckDB] memory_limit=%s (auto: 25%% of RAM; override in configs/tuning.yaml)", limit)
    return limit


def configure_global_duckdb() -> None:
    """Pin settings on duckdb's module-level default connection.

    The pipeline uses ``duckdb.sql(...)`` as a SQL engine over in-memory
    pandas frames (e.g. calculate_weighted_means). That implicit connection
    never goes through the adapters, so without this it runs with stock
    defaults — memory_limit at 80% of RAM and threads = all cores — making
    peak memory scale with whatever machine the run lands on.
    """
    limit = _resolve_memory_limit(None)
    duckdb.sql(f"SET memory_limit='{limit}'")

    # In-memory connections cannot spill without a temp_directory; with one,
    # queries that exceed memory_limit offload instead of failing.
    tmp_dir = Path(tempfile.gettempdir()) / "urbantrips_duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    duckdb.sql(f"SET temp_directory='{tmp_dir}'")

    try:
        from urbantrips.utils.utils import leer_configs_tuning
        threads = leer_configs_tuning().get("duckdb", {}).get("threads")
    except Exception:
        threads = None
    if threads:
        duckdb.sql(f"SET threads={int(threads)}")

    logger.info(
        "[DuckDB] global connection pinned: memory_limit=%s, temp_directory=%s%s",
        limit, tmp_dir, f", threads={threads}" if threads else "",
    )

# Tables with a 'dia' column, purged on delete_run_days
_TABLES_WITH_DIA = [
    "transacciones", "etapas", "viajes", "usuarios", "gps",
    "legs_to_gps_origin", "legs_to_gps_destination",
    "legs_to_station_origin", "legs_to_station_destination",
    "travel_times_gps", "travel_times_stations",
    "travel_times_legs", "travel_times_trips",
    "transacciones_linea", "tarjetas_duplicadas",
    "dias_ultima_corrida",
]

_TRANSACCIONES_COLUMNS = [
    "id", "batch_id", "fecha", "id_original", "id_tarjeta", "dia", "tiempo",
    "hora", "modo", "id_linea", "id_ramal", "interno", "orden_trx", "genero",
    "tarifa", "latitud", "longitud", "factor_expansion",
]

_ETAPAS_COLUMNS = [
    "id", "batch_id", "id_tarjeta", "dia", "id_viaje", "id_etapa", "tiempo",
    "hora", "modo", "id_linea", "id_ramal", "interno", "genero", "tarifa",
    "latitud", "longitud", "h3_o", "h3_d", "od_validado", "etapa_validada",
    "factor_expansion_original", "factor_expansion_linea",
    "factor_expansion_tarjeta", "factor_expansion_etapa", "distancia",
    "travel_time_min",
]

_VIAJES_COLUMNS = [
    "id_tarjeta", "id_viaje", "dia", "tiempo", "hora", "cant_etapas", "modo",
    "autobus", "tren", "metro", "tranvia", "brt", "cable", "lancha", "otros",
    "h3_o", "h3_d", "genero", "tarifa", "od_validado",
    "factor_expansion_linea", "factor_expansion_tarjeta", "distancia",
    "travel_time_min",
]

_DUCKDB_INSERT_CHUNK_ROWS = 250_000

_TRANSACCIONES_DEFAULTS: dict = {
    "id":               0,
    "batch_id":         0,
    "fecha":            0,
    "id_original":      "",
    "id_tarjeta":       "",
    "dia":              "",
    "tiempo":           "",
    "hora":             pd.NA,
    "modo":             "",
    "id_linea":         pd.NA,
    "id_ramal":         pd.NA,
    "interno":          pd.NA,
    "orden_trx":        pd.NA,
    "genero":           "",
    "tarifa":           "",
    "latitud":          np.nan,
    "longitud":         np.nan,
    "factor_expansion": np.nan,
}

# Type-correct defaults for each etapas column, matched to the DuckDB schema.
# Using None/object dtype causes parquet to write a null-typed column, which
# triggers a DuckDB statistics assertion ("SetMin or SetMax") on INSERT.
_ETAPAS_DEFAULTS: dict = {
    "id":                        0,
    "batch_id":                  0,
    "id_tarjeta":                "",
    "dia":                       "",
    "id_viaje":                  pd.NA,    # INT nullable
    "id_etapa":                  pd.NA,
    "tiempo":                    "",
    "hora":                      pd.NA,
    "modo":                      "",
    "id_linea":                  pd.NA,    # BIGINT nullable
    "id_ramal":                  pd.NA,
    "interno":                   pd.NA,
    "genero":                    "",
    "tarifa":                    "",
    "latitud":                   np.nan,   # FLOAT
    "longitud":                  np.nan,
    "h3_o":                      "",
    "h3_d":                      "",
    "od_validado":               0,
    "etapa_validada":            0,
    "factor_expansion_original": np.nan,
    "factor_expansion_linea":    np.nan,
    "factor_expansion_tarjeta":  np.nan,
    "factor_expansion_etapa":    np.nan,
    "distancia":                 np.nan,
    "travel_time_min":           np.nan,
}


class DuckDBDataAdapter:
    """Implements DataPort using DuckDB."""

    def __init__(
        self,
        db_path: Path,
        read_only: bool = False,
        memory_limit: str | None = None,
    ) -> None:
        self._path = Path(db_path)
        self._read_only = read_only
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path), read_only=self._read_only)
        self._conn.execute(f"SET memory_limit='{_resolve_memory_limit(memory_limit)}'")
        if not read_only:
            self._apply_schema()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def _apply_schema(self) -> None:
        self._migrate_schema()
        for ddl in schema.ALL_TABLES:
            self._conn.execute(ddl)
        for ddl in schema.ALL_INDEXES:
            self._conn.execute(ddl)

    def _migrate_schema(self) -> None:
        # hora_inicio/hora_fin were incorrectly typed FLOAT; drop so CREATE TABLE rebuilds them as TEXT
        row = self._conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'kpi_by_day_line_service' AND column_name = 'hora_inicio'"
        ).fetchone()
        if row and row[0].upper() != "TEXT":
            self._conn.execute("DROP TABLE IF EXISTS kpi_by_day_line_service")

    # ── batch helpers ─────────────────────────────────────────────────────────

    def get_user_batches(self, n_batches: int) -> list[BatchSpec]:
        """Return n_batches BatchSpec objects covering all users."""
        return [BatchSpec(batch_id=i, total_batches=n_batches) for i in range(n_batches)]

    def _batch_where(self, batch: BatchSpec | None, col: str = "id_tarjeta") -> str:
        if batch is None:
            return ""
        return f"WHERE hash({col}) % {batch.total_batches} = {batch.batch_id}"

    def _prepare_legs_df(
        self, df: pd.DataFrame, batch: BatchSpec | None = None
    ) -> pd.DataFrame:
        df = df.copy()
        if batch is not None:
            df["batch_id"] = batch.batch_id
        for col in _ETAPAS_COLUMNS:
            if col not in df.columns:
                df[col] = _ETAPAS_DEFAULTS.get(col, np.nan)
        return df[_ETAPAS_COLUMNS]

    # ── run days ──────────────────────────────────────────────────────────────

    def get_run_days(self) -> pd.DataFrame:
        return self._conn.execute("SELECT * FROM dias_ultima_corrida").fetchdf()

    def save_run_days(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM dias_ultima_corrida")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO dias_ultima_corrida SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    # ── transactions ──────────────────────────────────────────────────────────

    def get_transactions(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        where = self._batch_where(batch, "id_tarjeta")
        return self._conn.execute(f"SELECT * FROM transacciones {where}").fetchdf()

    def get_transactions_for_chunk(self, batch_ids: list[int], total_batches: int) -> pd.DataFrame:
        """Load rows for the given batch IDs in one scan, with _batch_id column for splitting."""
        ids = ", ".join(str(b) for b in batch_ids)
        return self._conn.execute(
            f"SELECT *, hash(id_tarjeta) % {total_batches} AS _batch_id"
            f" FROM transacciones WHERE hash(id_tarjeta) % {total_batches} IN ({ids})"
        ).fetchdf()

    def save_transactions(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        df = df.copy()
        for col in _TRANSACCIONES_COLUMNS:
            if col not in df.columns:
                df[col] = _TRANSACCIONES_DEFAULTS.get(col, np.nan)
        df = df[_TRANSACCIONES_COLUMNS]
        cols = ", ".join(_TRANSACCIONES_COLUMNS)
        self._conn.register("_df", df)
        try:
            self._conn.execute(f"INSERT INTO transacciones ({cols}) SELECT {cols} FROM _df")
        finally:
            self._conn.unregister("_df")

    # ── raw staging ───────────────────────────────────────────────────────────

    def save_raw_chunk(self, df: pd.DataFrame) -> None:
        """Append one CSV chunk (already structurally standardized) to transacciones_raw."""
        self._conn.register("_chunk", df)
        try:
            self._conn.execute("INSERT INTO transacciones_raw SELECT * FROM _chunk")
        finally:
            self._conn.unregister("_chunk")

    def clear_raw(self) -> None:
        """Truncate the staging table after standardization is complete."""
        self._conn.execute("DELETE FROM transacciones_raw")

    def standardize_raw_to_transacciones(self, n_batches: int, id_offset: int) -> None:
        """
        Move rows from transacciones_raw into transacciones, computing:
        - batch_id = hash(id_tarjeta) % n_batches  (DuckDB native hash, always unsigned)
        - id = sequential integer starting at id_offset
        - factor_expansion from factor_expansion_raw (or 1 if null)
        Filters out cards where any transaction has a NULL in a critical column,
        so only fully-valid cards are promoted.
        """
        self._conn.execute(f"""
                INSERT INTO transacciones
                SELECT
                    ROW_NUMBER() OVER () + {id_offset} - 1  AS id,
                    hash(id_tarjeta) % {n_batches}          AS batch_id,
                    fecha_ts                                 AS fecha,
                    id_original,
                    id_tarjeta,
                    dia,
                    tiempo,
                    hora,
                    modo,
                    id_linea,
                    id_ramal,
                    interno,
                    orden_trx,
                    genero,
                    tarifa,
                    latitud,
                    longitud,
                    COALESCE(factor_expansion_raw, 1.0)      AS factor_expansion
                FROM transacciones_raw r
                WHERE id_tarjeta IN (
                    SELECT id_tarjeta
                    FROM transacciones_raw
                    GROUP BY id_tarjeta
                    HAVING COUNT(*) = COUNT(CASE
                        WHEN id_tarjeta IS NOT NULL
                         AND fecha_ts   IS NOT NULL
                         AND id_linea   IS NOT NULL
                         AND latitud    IS NOT NULL
                         AND longitud   IS NOT NULL
                        THEN 1 END)
                )
            """)

    # ── batch-indexed reads ───────────────────────────────────────────────────

    def get_transactions_for_batch(self, batch: "BatchSpec") -> pd.DataFrame:
        """Read all transactions for one traveler batch across all ingested days."""
        return self._conn.execute(
            f"SELECT * FROM transacciones WHERE hash(id_tarjeta) % {batch.total_batches} = {batch.batch_id}",
        ).fetchdf()

    def get_legs_for_batch(self, batch: "BatchSpec") -> pd.DataFrame:
        """Read all legs for one traveler batch."""
        return self._conn.execute(
            "SELECT * FROM etapas WHERE batch_id = ?",
            [batch.batch_id],
        ).fetchdf()

    # ── legs (etapas) ─────────────────────────────────────────────────────────

    def get_legs(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        if batch is not None:
            return self._conn.execute(
                "SELECT * FROM etapas WHERE batch_id = ?",
                [batch.batch_id],
            ).fetchdf()
        return self._conn.execute("SELECT * FROM etapas").fetchdf()

    def update_leg_trip_ids(self, df: pd.DataFrame) -> None:
        """Update only id_viaje and id_etapa for existing legs, matched by id.

        Much faster than save_legs for rearrange operations that only modify
        trip/stage numbering: avoids full DELETE + parquet staging + INSERT.
        """
        if df.empty:
            return
        updates = df[["id", "id_viaje", "id_etapa"]].copy()
        self._conn.register("_trip_id_updates", updates)
        try:
            self._conn.execute("""
                UPDATE etapas
                SET id_viaje = u.id_viaje,
                    id_etapa = u.id_etapa
                FROM _trip_id_updates u
                WHERE etapas.id = u.id
            """)
        finally:
            self._conn.unregister("_trip_id_updates")

    def begin_leg_destination_updates(self) -> None:
        """Drop the index covering od_validado before bulk destination updates.

        DuckDB executes an UPDATE that touches an indexed column as a per-row
        DELETE+INSERT, maintaining every ART index of the table (including the
        PRIMARY KEY) row by row — ~34 min per ~1M-leg day in production. With
        the index dropped, none of the updated columns (h3_d, od_validado,
        etapa_validada) is indexed, so the UPDATE runs as an in-place,
        vectorized column rewrite (seconds). Call end_leg_destination_updates()
        to recreate the index once all days are written.
        """
        self._conn.execute("DROP INDEX IF EXISTS idx_etapas_dia_od_validado")

    def end_leg_destination_updates(self) -> None:
        """Recreate the index dropped by begin_leg_destination_updates()."""
        self._conn.execute(schema.IDX_ETAPAS_DIA_OD_VALIDADO)

    def update_leg_destinations(self, df: pd.DataFrame) -> None:
        """Update only h3_d, od_validado, etapa_validada for existing legs, matched by id.

        Called after destination inference to avoid rewriting all 26 columns
        for every leg — only the 3 columns that destination inference changes
        are touched.

        od_validado is covered by idx_etapas_dia_od_validado: bracket calls to
        this method with begin/end_leg_destination_updates(), otherwise DuckDB
        degrades the UPDATE to a per-row DELETE+INSERT (see begin docstring).
        """
        if df.empty:
            return
        cols = ["id", "h3_d", "od_validado", "etapa_validada"]
        if "dia" in df.columns:
            cols = ["id", "dia", "h3_d", "od_validado", "etapa_validada"]
        updates = df[cols].copy()
        self._conn.register("_dest_updates", updates)
        try:
            dia_filter = "AND etapas.dia = u.dia" if "dia" in df.columns else ""
            self._conn.execute(f"""
                UPDATE etapas
                SET h3_d          = u.h3_d,
                    od_validado   = u.od_validado,
                    etapa_validada = u.etapa_validada
                FROM _dest_updates u
                WHERE etapas.id = u.id {dia_filter}
            """)
        finally:
            self._conn.unregister("_dest_updates")

    def save_legs(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        """Persist legs to DuckDB via parquet staging to avoid Arrow-registration
        memory hazards.  Uses the same strategy as replace_legs_for_days."""
        df = self._prepare_legs_df(df, batch)
        if df.empty:
            return
        cols = ", ".join(_ETAPAS_COLUMNS)

        with tempfile.TemporaryDirectory(prefix="urbantrips_legs_") as tmpdir:
            tmp_path = Path(tmpdir)
            for idx, start in enumerate(range(0, len(df), _DUCKDB_INSERT_CHUNK_ROWS)):
                chunk = df.iloc[start : start + _DUCKDB_INSERT_CHUNK_ROWS]
                chunk.to_parquet(tmp_path / f"part-{idx:05d}.parquet", index=False)

            parquet_glob = str(tmp_path / "*.parquet").replace("'", "''")

            self._conn.execute("BEGIN TRANSACTION")
            try:
                # Delete by batch_id (indexed) instead of joining on id —
                # avoids an O(n²) scan as the etapas table grows across batches.
                if batch is not None:
                    self._conn.execute(
                        "DELETE FROM etapas WHERE batch_id = ?", [batch.batch_id]
                    )
                else:
                    self._conn.execute(
                        f"DELETE FROM etapas WHERE id IN "
                        f"(SELECT id FROM read_parquet('{parquet_glob}'))"
                    )
                self._conn.execute(
                    f"INSERT INTO etapas ({cols}) "
                    f"SELECT {cols} FROM read_parquet('{parquet_glob}')"
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def replace_legs_for_days(self, df: pd.DataFrame, days: list[str]) -> None:
        if not days:
            return

        df = self._prepare_legs_df(df)
        cols = ", ".join(_ETAPAS_COLUMNS)

        with tempfile.TemporaryDirectory(prefix="urbantrips_etapas_") as tmpdir:
            tmp_path = Path(tmpdir)
            for idx, start in enumerate(range(0, len(df), _DUCKDB_INSERT_CHUNK_ROWS)):
                chunk = df.iloc[start : start + _DUCKDB_INSERT_CHUNK_ROWS]
                chunk.to_parquet(tmp_path / f"part-{idx:05d}.parquet", index=False)

            parquet_glob = str(tmp_path / "*.parquet").replace("'", "''")
            placeholders = ", ".join("?" for _ in days)

            # threads=1 only for this insert; the setting persists on the
            # connection, so restore the previous value or every later query
            # (e.g. destination write-backs) runs single-threaded.
            prev_threads = self._conn.execute(
                "SELECT current_setting('threads')"
            ).fetchone()[0]
            self._conn.execute("PRAGMA threads=1")
            self._conn.execute("BEGIN TRANSACTION")
            try:
                self._conn.execute(
                    f"DELETE FROM etapas WHERE dia IN ({placeholders})",
                    days,
                )
                self._conn.execute(
                    f"""
                    INSERT INTO etapas ({cols})
                    SELECT {cols}
                    FROM read_parquet('{parquet_glob}')
                    """
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            finally:
                self._conn.execute(f"PRAGMA threads={int(prev_threads)}")

    # ── trips (viajes) ────────────────────────────────────────────────────────

    def get_trips(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        where = self._batch_where(batch, "id_tarjeta")
        return self._conn.execute(f"SELECT * FROM viajes {where}").fetchdf()

    def save_trips(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        df = df.copy()
        for col in _VIAJES_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[_VIAJES_COLUMNS]
        cols = ", ".join(_VIAJES_COLUMNS)
        self._conn.register("_df", df)
        try:
            self._conn.execute(f"INSERT INTO viajes ({cols}) SELECT {cols} FROM _df")
        finally:
            self._conn.unregister("_df")

    # ── users (usuarios) ──────────────────────────────────────────────────────

    def get_users(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        where = self._batch_where(batch, "id_tarjeta")
        return self._conn.execute(f"SELECT * FROM usuarios {where}").fetchdf()

    def save_users(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        cols = ", ".join(df.columns)
        self._conn.register("_df", df)
        try:
            self._conn.execute(f"INSERT INTO usuarios ({cols}) SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    # ── gps ───────────────────────────────────────────────────────────────────

    def get_gps(self) -> pd.DataFrame:
        return self._conn.execute("SELECT * FROM gps").fetchdf()

    def save_gps(self, df: pd.DataFrame) -> None:
        cols = ", ".join(df.columns)
        self._conn.register("_df", df)
        try:
            self._conn.execute(f"INSERT INTO gps ({cols}) SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def delete_run_days(self, days: list[str]) -> None:
        for table in _TABLES_WITH_DIA:
            try:
                for day in days:
                    self._conn.execute(f"DELETE FROM {table} WHERE dia = ?", [day])
            except duckdb.CatalogException:
                pass

    def execute(self, sql: str) -> None:
        self._conn.execute(sql)

    def has_rows(self, table_name: str, where: str | None = None) -> bool:
        table_name = validate_table_name(table_name)
        where_sql = f" WHERE {where}" if where else ""
        try:
            result = self._conn.execute(
                f"SELECT 1 FROM {table_name}{where_sql} LIMIT 1"
            ).fetchone()
        except duckdb.CatalogException:
            return False
        return result is not None

    def get_indicators(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM indicadores").fetchdf()
        except Exception:
            return pd.DataFrame()

    def save_indicators(self, df: pd.DataFrame) -> None:
        self._conn.register("_ind_df", df)
        try:
            self._conn.execute("CREATE OR REPLACE TABLE indicadores AS SELECT * FROM _ind_df")
        finally:
            self._conn.unregister("_ind_df")

    def get_vehicle_expansion_factors(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM vehicle_expansion_factors").fetchdf()
        except Exception:
            return pd.DataFrame()

    def save_vehicle_expansion_factors(self, df: pd.DataFrame) -> None:
        self._conn.register("_vef_df", df)
        try:
            self._conn.execute("INSERT INTO vehicle_expansion_factors SELECT * FROM _vef_df")
        finally:
            self._conn.unregister("_vef_df")

    def get_services(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM services").fetchdf()
        except Exception:
            return pd.DataFrame()

    def save_services(self, df: pd.DataFrame) -> None:
        self._conn.register("_svc_df", df)
        try:
            self._conn.execute("INSERT INTO services SELECT * FROM _svc_df")
        finally:
            self._conn.unregister("_svc_df")

    def get_line_transactions(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM transacciones_linea").fetchdf()
        except Exception:
            return pd.DataFrame()

    def save_line_transactions(self, df: pd.DataFrame) -> None:
        self._conn.register("_lt_df", df)
        try:
            self._conn.execute("INSERT INTO transacciones_linea SELECT * FROM _lt_df")
        finally:
            self._conn.unregister("_lt_df")

    def get_max_id(self, table: str) -> int:
        table = validate_table_name(table)
        try:
            result = self._conn.execute(f"SELECT COALESCE(MAX(id), -1) FROM {table}").fetchone()
            return int(result[0]) + 1
        except Exception:
            return 0

    def query(self, sql: str) -> pd.DataFrame:
        return self._conn.execute(sql).fetchdf()

    def save_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._conn.register("_raw_df", df)
        try:
            self._conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _raw_df")
        finally:
            self._conn.unregister("_raw_df")

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._conn.register("_raw_df", df)
        try:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} AS "
                f"SELECT * FROM _raw_df WHERE FALSE"
            )
            cols = ", ".join(f'"{c}"' for c in df.columns)
            self._conn.execute(
                f"INSERT INTO {table_name} ({cols}) SELECT * FROM _raw_df"
            )
        finally:
            self._conn.unregister("_raw_df")

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        try:
            return self._conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except Exception:
            return pd.DataFrame()
