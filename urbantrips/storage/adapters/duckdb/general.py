# urbantrips/storage/adapters/duckdb/general.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.schema import general as schema


class DuckDBGeneralAdapter:
    """Implements GeneralPort using DuckDB."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._apply_schema()

    def _conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._path))

    def _apply_schema(self) -> None:
        with self._conn() as conn:
            for ddl in schema.ALL_TABLES:
                conn.execute(ddl)

    def get_completed_runs(self) -> pd.DataFrame:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM corridas").fetchdf()

    def register_run(self, alias: str, process: str) -> None:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO corridas (corrida, process, date) VALUES (?, ?, ?)",
                [alias, process, date],
            )

    def execute(self, sql: str) -> None:
        with self._conn() as conn:
            conn.execute(sql)

    def query(self, sql: str) -> pd.DataFrame:
        with self._conn() as conn:
            try:
                return conn.execute(sql).fetchdf()
            except duckdb.CatalogException:
                return pd.DataFrame()

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        with self._conn() as conn:
            conn.register("_raw_df", df)
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} AS "
                f"SELECT * FROM _raw_df WHERE FALSE"
            )
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM _raw_df")

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        with self._conn() as conn:
            try:
                return conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            except Exception:
                return pd.DataFrame()

    def run_exists(self, alias: str) -> bool:
        runs = self.get_completed_runs()
        if runs.empty or "corrida" not in runs.columns:
            return False
        return alias in runs["corrida"].values

    def clear_runs(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM corridas")
