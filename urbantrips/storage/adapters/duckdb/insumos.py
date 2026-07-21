# urbantrips/storage/adapters/duckdb/insumos.py
from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.schema import insumos as schema


class DuckDBInsumoAdapter:
    """Implements InsumoPort using DuckDB."""

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

    # ── geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _geo_to_df(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        df = gdf.copy()
        df["wkt"] = gdf.geometry.to_wkt()
        return df.drop(columns="geometry")

    @staticmethod
    def _df_to_geo(df: pd.DataFrame) -> gpd.GeoDataFrame:
        df = df.copy()
        df["geometry"] = df["wkt"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        return gdf.drop(columns="wkt")

    # ── read methods ──────────────────────────────────────────────────────────

    def get_routes(self) -> gpd.GeoDataFrame:
        try:
            df = self._conn.execute("SELECT * FROM lines_geoms").fetchdf()
        except Exception:
            return gpd.GeoDataFrame()
        if df.empty:
            return gpd.GeoDataFrame()
        return self._df_to_geo(df)

    def get_stops(self) -> pd.DataFrame:
        return self._conn.execute("SELECT * FROM stops").fetchdf()

    def get_distances(self, h3_ids: list[str] | None = None) -> pd.DataFrame:
        if h3_ids:
            placeholders = ", ".join("?" for _ in h3_ids)
            query = (
                f"SELECT * FROM distancias "
                f"WHERE h3_o IN ({placeholders}) OR h3_d IN ({placeholders})"
            )
            params = h3_ids + h3_ids
        else:
            query = "SELECT * FROM distancias"
            params = None
        return self._conn.execute(query, params).fetchdf()

    def get_zones(self) -> gpd.GeoDataFrame:
        # Zone tables are heterogeneous across configs; returns empty until
        # zone storage is standardised in Plan 2.
        return gpd.GeoDataFrame()

    def get_metadata_lineas(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM metadata_lineas").fetchdf()
        except Exception:
            return pd.DataFrame()

    def get_metadata_ramales(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM metadata_ramales").fetchdf()
        except Exception:
            return pd.DataFrame()

    def get_matrix_validation(self) -> pd.DataFrame:
        return self._conn.execute("SELECT * FROM matriz_validacion").fetchdf()

    def get_travel_times_stations(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM travel_times_stations").fetchdf()
        except Exception:
            return pd.DataFrame()

    # ── write methods ─────────────────────────────────────────────────────────

    def save_routes(self, df: gpd.GeoDataFrame) -> None:
        flat = self._geo_to_df(df)
        self._conn.execute("DELETE FROM lines_geoms")
        self._conn.register("_df", flat)
        try:
            self._conn.execute("INSERT INTO lines_geoms SELECT id_linea, wkt FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_stops(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM stops")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO stops SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_distances(self, df: pd.DataFrame) -> None:
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO distancias SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_zones(self, df: gpd.GeoDataFrame) -> None:
        # No-op until zone storage is standardised in Plan 2.
        pass

    def save_matrix_validation(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM matriz_validacion")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO matriz_validacion SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_travel_times_stations(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM travel_times_stations")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO travel_times_stations SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_metadata_lineas(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM metadata_lineas")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO metadata_lineas SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_metadata_ramales(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM metadata_ramales")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO metadata_ramales SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def has_routes(self) -> bool:
        return not self.get_routes().empty

    def execute(self, sql: str) -> None:
        self._conn.execute(sql)

    def query(self, sql: str) -> pd.DataFrame:
        try:
            return self._conn.execute(sql).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame()

    def save_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._conn.register("_raw_df", df)
        try:
            self._conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _raw_df"
            )
        finally:
            self._conn.unregister("_raw_df")

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        try:
            return self._conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except Exception:
            return pd.DataFrame()

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._conn.register("_raw_df", df)
        try:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} "
                f"AS SELECT * FROM _raw_df WHERE FALSE"
            )
            self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM _raw_df")
        finally:
            self._conn.unregister("_raw_df")
