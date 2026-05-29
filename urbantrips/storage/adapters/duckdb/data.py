# urbantrips/storage/adapters/duckdb/data.py
from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.ports import BatchSpec
from urbantrips.storage.schema import data as schema

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


class DuckDBDataAdapter:
    """Implements DataPort using DuckDB."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._read_only = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._apply_schema()

    def _conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._path), read_only=getattr(self, "_read_only", False))

    def _apply_schema(self) -> None:
        with self._conn() as conn:
            for ddl in schema.ALL_TABLES:
                conn.execute(ddl)
            for ddl in schema.ALL_INDEXES:
                conn.execute(ddl)

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
                df[col] = None
        return df[_ETAPAS_COLUMNS]

    # ── run days ──────────────────────────────────────────────────────────────

    def get_run_days(self) -> pd.DataFrame:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM dias_ultima_corrida").fetchdf()

    def save_run_days(self, df: pd.DataFrame) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM dias_ultima_corrida")
            conn.register("_df", df)
            try:
                conn.execute("INSERT INTO dias_ultima_corrida SELECT * FROM _df")
            finally:
                conn.unregister("_df")

    # ── transactions ──────────────────────────────────────────────────────────

    def get_transactions(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        where = self._batch_where(batch, "id_tarjeta")
        with self._conn() as conn:
            return conn.execute(f"SELECT * FROM transacciones {where}").fetchdf()

    def save_transactions(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        df = df.copy()
        for col in _TRANSACCIONES_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[_TRANSACCIONES_COLUMNS]
        cols = ", ".join(_TRANSACCIONES_COLUMNS)
        with self._conn() as conn:
            conn.register("_df", df)
            try:
                conn.execute(f"INSERT INTO transacciones ({cols}) SELECT {cols} FROM _df")
            finally:
                conn.unregister("_df")

    # ── raw staging ───────────────────────────────────────────────────────────

    def save_raw_chunk(self, df: pd.DataFrame) -> None:
        """Append one CSV chunk (already structurally standardized) to transacciones_raw."""
        with self._conn() as conn:
            conn.register("_chunk", df)
            try:
                conn.execute("INSERT INTO transacciones_raw SELECT * FROM _chunk")
            finally:
                conn.unregister("_chunk")

    def clear_raw(self) -> None:
        """Truncate the staging table after standardization is complete."""
        with self._conn() as conn:
            conn.execute("DELETE FROM transacciones_raw")

    def standardize_raw_to_transacciones(self, n_batches: int, id_offset: int) -> None:
        """
        Move rows from transacciones_raw into transacciones, computing:
        - batch_id = hash(id_tarjeta) % n_batches  (DuckDB native hash, always unsigned)
        - id = sequential integer starting at id_offset
        - factor_expansion from factor_expansion_raw (or 1 if null)
        Filters out cards where any transaction has a NULL in a critical column,
        so only fully-valid cards are promoted.
        """
        with self._conn() as conn:
            conn.execute(f"""
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
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM transacciones WHERE batch_id = ?",
                [batch.batch_id],
            ).fetchdf()

    def get_legs_for_batch(self, batch: "BatchSpec") -> pd.DataFrame:
        """Read all legs for one traveler batch."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM etapas WHERE batch_id = ?",
                [batch.batch_id],
            ).fetchdf()

    # ── legs (etapas) ─────────────────────────────────────────────────────────

    def get_legs(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        with self._conn() as conn:
            if batch is not None:
                return conn.execute(
                    "SELECT * FROM etapas WHERE batch_id = ?",
                    [batch.batch_id],
                ).fetchdf()
            return conn.execute("SELECT * FROM etapas").fetchdf()

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

            with self._conn() as conn:
                conn.execute("PRAGMA threads=1")
                conn.execute("BEGIN TRANSACTION")
                try:
                    conn.execute(
                        f"DELETE FROM etapas WHERE id IN "
                        f"(SELECT id FROM read_parquet('{parquet_glob}'))"
                    )
                    conn.execute(
                        f"INSERT INTO etapas ({cols}) "
                        f"SELECT {cols} FROM read_parquet('{parquet_glob}')"
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
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

            with self._conn() as conn:
                conn.execute("PRAGMA threads=1")
                conn.execute("BEGIN TRANSACTION")
                try:
                    conn.execute(
                        f"DELETE FROM etapas WHERE dia IN ({placeholders})",
                        days,
                    )
                    conn.execute(
                        f"""
                        INSERT INTO etapas ({cols})
                        SELECT {cols}
                        FROM read_parquet('{parquet_glob}')
                        """
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise

    # ── trips (viajes) ────────────────────────────────────────────────────────

    def get_trips(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        where = self._batch_where(batch, "id_tarjeta")
        with self._conn() as conn:
            return conn.execute(f"SELECT * FROM viajes {where}").fetchdf()

    def save_trips(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        df = df.copy()
        for col in _VIAJES_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[_VIAJES_COLUMNS]
        cols = ", ".join(_VIAJES_COLUMNS)
        with self._conn() as conn:
            conn.register("_df", df)
            try:
                conn.execute(f"INSERT INTO viajes ({cols}) SELECT {cols} FROM _df")
            finally:
                conn.unregister("_df")

    # ── users (usuarios) ──────────────────────────────────────────────────────

    def get_users(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        where = self._batch_where(batch, "id_tarjeta")
        with self._conn() as conn:
            return conn.execute(f"SELECT * FROM usuarios {where}").fetchdf()

    def save_users(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        cols = ", ".join(df.columns)
        with self._conn() as conn:
            conn.register("_df", df)
            try:
                conn.execute(f"INSERT INTO usuarios ({cols}) SELECT * FROM _df")
            finally:
                conn.unregister("_df")

    # ── gps ───────────────────────────────────────────────────────────────────

    def get_gps(self) -> pd.DataFrame:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM gps").fetchdf()

    def save_gps(self, df: pd.DataFrame) -> None:
        cols = ", ".join(df.columns)
        with self._conn() as conn:
            conn.register("_df", df)
            try:
                conn.execute(f"INSERT INTO gps ({cols}) SELECT * FROM _df")
            finally:
                conn.unregister("_df")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def delete_run_days(self, days: list[str]) -> None:
        with self._conn() as conn:
            for table in _TABLES_WITH_DIA:
                try:
                    for day in days:
                        conn.execute(f"DELETE FROM {table} WHERE dia = ?", [day])
                except duckdb.CatalogException:
                    pass

    def execute(self, sql: str) -> None:
        with self._conn() as conn:
            conn.execute(sql)

    def has_rows(self, table_name: str, where: str | None = None) -> bool:
        table_name = validate_table_name(table_name)
        where_sql = f" WHERE {where}" if where else ""
        with self._conn() as conn:
            try:
                result = conn.execute(
                    f"SELECT 1 FROM {table_name}{where_sql} LIMIT 1"
                ).fetchone()
            except duckdb.CatalogException:
                return False
            return result is not None

    def get_indicators(self) -> pd.DataFrame:
        with self._conn() as conn:
            try:
                return conn.execute("SELECT * FROM indicadores").fetchdf()
            except Exception:
                return pd.DataFrame()

    def save_indicators(self, df: pd.DataFrame) -> None:
        with self._conn() as conn:
            conn.register("_ind_df", df)
            try:
                conn.execute("CREATE OR REPLACE TABLE indicadores AS SELECT * FROM _ind_df")
            finally:
                conn.unregister("_ind_df")

    def get_vehicle_expansion_factors(self) -> pd.DataFrame:
        with self._conn() as conn:
            try:
                return conn.execute("SELECT * FROM vehicle_expansion_factors").fetchdf()
            except Exception:
                return pd.DataFrame()

    def save_vehicle_expansion_factors(self, df: pd.DataFrame) -> None:
        with self._conn() as conn:
            conn.register("_vef_df", df)
            try:
                conn.execute("INSERT INTO vehicle_expansion_factors SELECT * FROM _vef_df")
            finally:
                conn.unregister("_vef_df")

    def get_services(self) -> pd.DataFrame:
        with self._conn() as conn:
            try:
                return conn.execute("SELECT * FROM services").fetchdf()
            except Exception:
                return pd.DataFrame()

    def save_services(self, df: pd.DataFrame) -> None:
        with self._conn() as conn:
            conn.register("_svc_df", df)
            try:
                conn.execute("INSERT INTO services SELECT * FROM _svc_df")
            finally:
                conn.unregister("_svc_df")

    def get_line_transactions(self) -> pd.DataFrame:
        with self._conn() as conn:
            try:
                return conn.execute("SELECT * FROM transacciones_linea").fetchdf()
            except Exception:
                return pd.DataFrame()

    def save_line_transactions(self, df: pd.DataFrame) -> None:
        with self._conn() as conn:
            conn.register("_lt_df", df)
            try:
                conn.execute("INSERT INTO transacciones_linea SELECT * FROM _lt_df")
            finally:
                conn.unregister("_lt_df")

    def get_max_id(self, table: str) -> int:
        table = validate_table_name(table)
        with self._conn() as conn:
            try:
                result = conn.execute(f"SELECT COALESCE(MAX(id), -1) FROM {table}").fetchone()
                return int(result[0]) + 1
            except Exception:
                return 0

    def query(self, sql: str) -> pd.DataFrame:
        with self._conn() as conn:
            return conn.execute(sql).fetchdf()

    def save_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        with self._conn() as conn:
            conn.register("_raw_df", df)
            try:
                conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _raw_df")
            finally:
                conn.unregister("_raw_df")

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        with self._conn() as conn:
            conn.register("_raw_df", df)
            try:
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table_name} AS "
                    f"SELECT * FROM _raw_df WHERE FALSE"
                )
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM _raw_df")
            finally:
                conn.unregister("_raw_df")

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        with self._conn() as conn:
            try:
                return conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            except Exception:
                return pd.DataFrame()
