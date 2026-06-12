# urbantrips/storage/adapters/duckdb/dash.py
from __future__ import annotations

from pathlib import Path
import re

import duckdb
import pandas as pd

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.schema import dash as schema


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
