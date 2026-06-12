# urbantrips/storage/adapters/duckdb/dash.py
from __future__ import annotations

import tempfile
from pathlib import Path
import re

import duckdb
import pandas as pd

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.schema import dash as schema

# Same chunking as the data adapter's legs staging: writing big DataFrames
# through Arrow registration converts mixed-dtype object columns row by row
# (very slow at millions of rows); parquet staging avoids it.
_DUCKDB_INSERT_CHUNK_ROWS = 250_000


class DuckDBDashAdapter:
    """Implements DashPort using DuckDB."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path))
        self._apply_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def _apply_schema(self) -> None:
        for ddl in schema.ALL_TABLES:
            self._conn.execute(ddl)

    def save_indicator(self, df: pd.DataFrame, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(
                f"Invalid indicator table name: {name!r}"
            )
        self._conn.register("_df", df)
        try:
            self._conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def get_indicator(self, name: str) -> pd.DataFrame:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return pd.DataFrame()
        try:
            return self._conn.execute(f"SELECT * FROM {name}").fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame()

    def list_indicators(self) -> list[str]:
        """Return names of dash tables that have at least one row."""
        result = []
        for table in schema.VALID_TABLE_NAMES:
            try:
                count = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count > 0:
                    result.append(table)
            except duckdb.CatalogException:
                pass
        return sorted(result)

    def save_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._conn.register("_raw_df", df)
        try:
            self._conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _raw_df")
        finally:
            self._conn.unregister("_raw_df")

    def execute(self, sql: str) -> None:
        self._conn.execute(sql)

    def query(self, sql: str) -> pd.DataFrame:
        try:
            return self._conn.execute(sql).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame()

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._conn.register("_raw_df", df)
        try:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} AS "
                f"SELECT * FROM _raw_df WHERE FALSE"
            )
            self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM _raw_df")
        finally:
            self._conn.unregister("_raw_df")

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        try:
            return self._conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except Exception:
            return pd.DataFrame()

    def upsert_chains_norm(self, df: pd.DataFrame, dias: list) -> None:
        """Replace chains_norm rows for the given days; keep the rest.

        Staged through temporary parquet files (same strategy as the data
        adapter's legs writes): chains_norm has object columns mixing NaN and
        strings (h3_transfer*, seq_lineas), which make Arrow registration fall
        back to row-by-row conversion — minutes at millions of rows. Parquet
        round-trips them as nullable strings and the INSERT is a fast scan.

        If the table has a stale schema (e.g. from a previous run) it is
        dropped so _apply_schema recreates it with the correct DDL.
        """
        if len(df) == 0:
            return

        existing_cols = self._conn.execute(
            "SELECT COUNT(*) FROM pragma_table_info('chains_norm')"
        ).fetchone()[0]
        if existing_cols and existing_cols != len(df.columns):
            self._conn.execute("DROP TABLE chains_norm")
            self._apply_schema()

        # ART index maintenance makes bulk DELETE+INSERT crawl (minutes per
        # day at ~5M rows). Drop the table's indexes before loading;
        # crear_indices_unificados recreates them at the end of the run.
        idxs = self._conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name = 'chains_norm'"
        ).fetchall()
        for (idx_name,) in idxs:
            self._conn.execute(f'DROP INDEX IF EXISTS "{idx_name}"')

        placeholders = ", ".join("?" for _ in dias)

        with tempfile.TemporaryDirectory(prefix="urbantrips_chains_") as tmpdir:
            tmp_path = Path(tmpdir)
            for idx, start in enumerate(range(0, len(df), _DUCKDB_INSERT_CHUNK_ROWS)):
                chunk = df.iloc[start : start + _DUCKDB_INSERT_CHUNK_ROWS]
                chunk.to_parquet(tmp_path / f"part-{idx:05d}.parquet", index=False)

            parquet_glob = str(tmp_path / "*.parquet").replace("'", "''")

            self._conn.execute("BEGIN TRANSACTION")
            try:
                self._conn.execute(
                    f"DELETE FROM chains_norm WHERE dia IN ({placeholders})",
                    list(dias),
                )
                self._conn.execute(
                    "INSERT INTO chains_norm BY NAME "
                    f"SELECT * FROM read_parquet('{parquet_glob}')"
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
