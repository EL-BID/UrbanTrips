# urbantrips/storage/adapters/duckdb/insumos.py
from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.schema import insumos as schema
from urbantrips.storage.adapters.duckdb.data import apply_temp_directory


class DuckDBInsumoAdapter:
    """Implements InsumoPort using DuckDB."""

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._path = Path(db_path)
        self._read_only = read_only
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path), read_only=self._read_only)
        apply_temp_directory(self._conn)
        if not read_only:
            self._apply_schema()

    def close(self) -> None:
        # getattr: si duckdb.connect falla en __init__ (base tomada por otro
        # proceso) el atributo no existe y __del__ no tiene que romper.
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def _apply_schema(self) -> None:
        self._migrate_schema()
        for ddl in schema.ALL_TABLES:
            self._conn.execute(ddl)

    def _migrate_schema(self) -> None:
        # `distancias` se eliminó del esquema el 2026-07-27: no tenía productor
        # (el cache real de distancias OD es el archivo aparte `od_distances`,
        # ver carto/compute_distances.py) y quedaba vacía en toda base. Se dropea
        # acá porque CREATE TABLE IF NOT EXISTS no borra tablas ya creadas.
        self._conn.execute("DROP TABLE IF EXISTS distancias")

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

    def get_zones(self) -> gpd.GeoDataFrame:
        """Capas de zonificación (tabla `zonificaciones`) como GeoDataFrame 4326.

        Es la fuente del bbox del área de estudio: `bbox_area_estudio` (filtro
        geográfico de trx/gps) y `resolve_network_cache` (bbox de la red OSM) la
        usan y tienen que ver lo mismo, si no el filtro acepta puntos que la red
        no cubre.

        No usa `_df_to_geo`: `guardo_zonificaciones` escribe esta tabla con
        `_with_wkt_geometry` (carto.py), que serializa el WKT en la columna
        `geometry` en vez de en `wkt`. Devuelve vacío si la tabla todavía no
        existe (antes de la primera `guardo_zonificaciones`) o si el proyecto no
        declaró zonificaciones: ahí el bbox cae al config.

        NOTA: es la tabla de polígonos, NO `equivalencias_zonas` (h3 → zona),
        que se lee con `get_raw("equivalencias_zonas")`.
        """
        try:
            df = self._conn.execute("SELECT * FROM zonificaciones").fetchdf()
        except Exception:
            return gpd.GeoDataFrame()
        if df.empty or "geometry" not in df.columns:
            return gpd.GeoDataFrame()
        df = df[df["geometry"].notna()].copy()
        if df.empty:
            return gpd.GeoDataFrame()
        df["geometry"] = df["geometry"].apply(wkt.loads)
        return gpd.GeoDataFrame(df, geometry="geometry", crs=4326)

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

    def get_matriz_paradas(self) -> pd.DataFrame:
        """Conteos crudos acumulados por (id_linea, id_ramal, parada) y su flag valido.

        Es la evidencia que respalda cada parada candidata: nunca se borra una fila,
        solo cambia su `valido`. matriz_validacion se deriva de las que tienen valido=1.
        """
        try:
            return self._conn.execute("SELECT * FROM matriz_paradas").fetchdf()
        except Exception:
            return pd.DataFrame()

    def get_matriz_paradas_dias(self) -> list[str]:
        """Días ya sumados a matriz_paradas (evita el doble conteo al reprocesar)."""
        try:
            rows = self._conn.execute("SELECT dia FROM matriz_paradas_dias").fetchall()
        except Exception:
            return []
        return [r[0] for r in rows]

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
            self._conn.execute(
                "INSERT INTO lines_geoms SELECT id_linea, direction, wkt FROM _df"
            )
        finally:
            self._conn.unregister("_df")

    def save_stops(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM stops")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO stops SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_zones(self, df: gpd.GeoDataFrame) -> None:
        # Sin cuerpo a proposito y sin llamadores: la escritura real de la tabla
        # es `save_raw(df, "zonificaciones")` desde `guardo_zonificaciones`
        # (carto.py), que serializa la geometria con `_with_wkt_geometry`.
        # Implementar acá una segunda vía de escritura daría dos formatos para la
        # misma tabla. Ver `get_zones`.
        pass

    def save_matrix_validation(self, df: pd.DataFrame) -> None:
        self._conn.execute("DELETE FROM matriz_validacion")
        self._conn.register("_df", df)
        try:
            self._conn.execute("INSERT INTO matriz_validacion SELECT * FROM _df")
        finally:
            self._conn.unregister("_df")

    def save_matriz_paradas(self, df: pd.DataFrame, dias: list[str]) -> None:
        """Reemplaza los conteos acumulados y la lista de días incorporados.

        La tabla es chica (~450k filas), así que DELETE+INSERT evita tener que
        resolver NULLs de id_ramal en un upsert por clave.
        """
        self._conn.execute("DELETE FROM matriz_paradas")
        self._conn.register("_df", df)
        try:
            self._conn.execute(
                "INSERT INTO matriz_paradas "
                "SELECT id_linea, id_ramal, parada, n_trx, n_gps, valido FROM _df"
            )
        finally:
            self._conn.unregister("_df")

        self._conn.execute("DELETE FROM matriz_paradas_dias")
        if dias:
            dias_df = pd.DataFrame({"dia": list(dias)})
            self._conn.register("_dias", dias_df)
            try:
                self._conn.execute(
                    "INSERT INTO matriz_paradas_dias SELECT dia FROM _dias"
                )
            finally:
                self._conn.unregister("_dias")

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
