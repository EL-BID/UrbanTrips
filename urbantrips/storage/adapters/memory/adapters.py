# urbantrips/storage/adapters/memory/adapters.py
from __future__ import annotations

from datetime import datetime
import re

import geopandas as gpd
import pandas as pd

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.ports import BatchSpec


def _query_store(store: dict[str, object], sql: str) -> pd.DataFrame:
    match = re.fullmatch(
        r"\s*SELECT\s+\*\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)\s*;?\s*",
        sql,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise NotImplementedError(
            "In-memory adapters only support SELECT * FROM <table> queries"
        )
    table = store.get(match.group(1), pd.DataFrame())
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame()


class InMemoryDataAdapter:
    """In-process DataPort implementation for testing."""

    def __init__(self, **tables: pd.DataFrame) -> None:
        self._store: dict[str, pd.DataFrame] = {k: v.copy() for k, v in tables.items()}

    def _get(self, table: str) -> pd.DataFrame:
        return self._store.get(table, pd.DataFrame()).copy()

    def _append(self, table: str, df: pd.DataFrame) -> None:
        existing = self._store.get(table, pd.DataFrame())
        self._store[table] = pd.concat([existing, df], ignore_index=True)

    def _filter_batch(self, df: pd.DataFrame, batch: BatchSpec | None) -> pd.DataFrame:
        if batch is None or df.empty or "id_tarjeta" not in df.columns:
            return df
        mask = df["id_tarjeta"].apply(
            lambda x: hash(str(x)) % batch.total_batches == batch.batch_id
        )
        return df[mask].copy()

    def get_user_batches(self, n_batches: int) -> list[BatchSpec]:
        return [BatchSpec(batch_id=i, total_batches=n_batches) for i in range(n_batches)]

    def get_run_days(self) -> pd.DataFrame:
        return self._get("dias_ultima_corrida")

    def save_run_days(self, df: pd.DataFrame) -> None:
        self._store["dias_ultima_corrida"] = df.copy()

    def get_transactions(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        return self._filter_batch(self._get("transacciones"), batch)

    def get_transactions_for_chunk(self, batch_ids: list[int], total_batches: int) -> pd.DataFrame:
        df = self._get("transacciones")
        if df.empty:
            return df.assign(_batch_id=pd.Series(dtype="int64"))
        df = df.copy()
        df["_batch_id"] = df["id_tarjeta"].apply(hash) % total_batches
        return df[df["_batch_id"].isin(batch_ids)].reset_index(drop=True)

    def save_transactions(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        self._append("transacciones", df)

    def get_legs(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        return self._filter_batch(self._get("etapas"), batch)

    def save_legs(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        self._append("etapas", df)

    def update_leg_trip_ids(self, df: pd.DataFrame) -> None:
        existing = self._store.get("etapas", pd.DataFrame())
        if existing.empty or df.empty:
            return
        updates = df.set_index("id")[["id_viaje", "id_etapa"]]
        existing = existing.set_index("id")
        existing.update(updates)
        self._store["etapas"] = existing.reset_index()

    def replace_legs_for_days(self, df: pd.DataFrame, days: list[str]) -> None:
        existing = self._store.get("etapas", pd.DataFrame())
        if existing.empty or "dia" not in existing.columns:
            self._store["etapas"] = df.copy()
            return
        kept = existing[~existing["dia"].isin(days)].copy()
        self._store["etapas"] = pd.concat([kept, df], ignore_index=True)

    def get_trips(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        return self._filter_batch(self._get("viajes"), batch)

    def save_trips(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        self._append("viajes", df)

    def get_users(self, batch: BatchSpec | None = None) -> pd.DataFrame:
        return self._filter_batch(self._get("usuarios"), batch)

    def save_users(self, df: pd.DataFrame, batch: BatchSpec | None = None) -> None:
        self._append("usuarios", df)

    def get_gps(self) -> pd.DataFrame:
        return self._get("gps")

    def save_gps(self, df: pd.DataFrame) -> None:
        self._append("gps", df)

    def delete_run_days(self, days: list[str]) -> None:
        for table in list(self._store.keys()):
            df = self._store[table]
            if "dia" in df.columns:
                self._store[table] = df[~df["dia"].isin(days)].copy()

    def execute(self, sql: str) -> None:
        pass  # no-op for in-memory adapter

    def has_rows(self, table_name: str, where: str | None = None) -> bool:
        table_name = validate_table_name(table_name)
        df = self._store.get(table_name)
        if df is None or df.empty:
            return False
        if where is None:
            return True

        match = re.fullmatch(r"\s*(\w+)\s*=\s*('?)([^']+)\2\s*", where)
        if match is None:
            raise NotImplementedError(
                "InMemoryDataAdapter.has_rows only supports simple equality predicates"
            )
        column, _, value = match.groups()
        if column not in df.columns:
            return False
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            try:
                value = float(value)
            except ValueError:
                return False
        return bool((series == value).any())

    def get_indicators(self) -> pd.DataFrame:
        return self._get("indicadores")

    def save_indicators(self, df: pd.DataFrame) -> None:
        self._store["indicadores"] = df.copy()

    def get_vehicle_expansion_factors(self) -> pd.DataFrame:
        return self._get("vehicle_expansion_factors")

    def save_vehicle_expansion_factors(self, df: pd.DataFrame) -> None:
        self._append("vehicle_expansion_factors", df)

    def get_services(self) -> pd.DataFrame:
        return self._get("services")

    def save_services(self, df: pd.DataFrame) -> None:
        self._append("services", df)

    def get_line_transactions(self) -> pd.DataFrame:
        return self._get("transacciones_linea")

    def save_line_transactions(self, df: pd.DataFrame) -> None:
        self._append("transacciones_linea", df)

    def get_max_id(self, table: str) -> int:
        table = validate_table_name(table)
        df = self._get(table)
        if df.empty or "id" not in df.columns:
            return 0
        return int(df["id"].max()) + 1

    def query(self, sql: str) -> pd.DataFrame:
        return _query_store(self._store, sql)

    def save_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._store[table_name] = df.copy()

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        existing = self._store.get(table_name, pd.DataFrame())
        self._store[table_name] = pd.concat([existing, df], ignore_index=True)

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        return self._store.get(table_name, pd.DataFrame()).copy()

    def save_raw_chunk(self, df: pd.DataFrame) -> None:
        self._append("transacciones_raw", df)

    def clear_raw(self) -> None:
        self._store["transacciones_raw"] = pd.DataFrame()

    def standardize_raw_to_transacciones(self, n_batches: int, id_offset: int) -> None:
        raise NotImplementedError("InMemoryDataAdapter does not support standardize_raw_to_transacciones")

    def get_transactions_for_batch(self, batch: BatchSpec) -> pd.DataFrame:
        df = self._get("transacciones")
        if df.empty or "batch_id" not in df.columns:
            return df
        return df[df["batch_id"] == batch.batch_id].copy()

    def get_legs_for_batch(self, batch: BatchSpec) -> pd.DataFrame:
        df = self._get("etapas")
        if df.empty or "batch_id" not in df.columns:
            return df
        return df[df["batch_id"] == batch.batch_id].copy()


class InMemoryInsumoAdapter:
    """In-process InsumoPort implementation for testing."""

    def __init__(self, **tables) -> None:
        self._store: dict[str, object] = {k: v for k, v in tables.items()}

    def get_routes(self) -> gpd.GeoDataFrame:
        return self._store.get("routes", gpd.GeoDataFrame())  # type: ignore[return-value]

    def get_stops(self) -> pd.DataFrame:
        return self._store.get("stops", pd.DataFrame())  # type: ignore[return-value]

    def get_distances(self, h3_ids: list[str] | None = None) -> pd.DataFrame:
        df = self._store.get("distancias", pd.DataFrame())
        if h3_ids and not df.empty and "h3_o" in df.columns:  # type: ignore[union-attr]
            return df[df["h3_o"].isin(h3_ids) | df["h3_d"].isin(h3_ids)].copy()  # type: ignore[return-value]
        return df.copy()  # type: ignore[return-value]

    def get_zones(self) -> gpd.GeoDataFrame:
        return self._store.get("zones", gpd.GeoDataFrame())  # type: ignore[return-value]

    def get_metadata_lineas(self) -> pd.DataFrame:
        return self._store.get("metadata_lineas", pd.DataFrame())  # type: ignore[return-value]

    def get_metadata_ramales(self) -> pd.DataFrame:
        return self._store.get("metadata_ramales", pd.DataFrame())  # type: ignore[return-value]

    def get_matrix_validation(self) -> pd.DataFrame:
        return self._store.get("matriz_validacion", pd.DataFrame())  # type: ignore[return-value]

    def get_travel_times_stations(self) -> pd.DataFrame:
        return self._store.get("travel_times_stations", pd.DataFrame())  # type: ignore[return-value]

    def save_routes(self, df: gpd.GeoDataFrame) -> None:
        self._store["routes"] = df.copy()

    def save_stops(self, df: pd.DataFrame) -> None:
        self._store["stops"] = df.copy()

    def save_distances(self, df: pd.DataFrame) -> None:
        existing = self._store.get("distancias", pd.DataFrame())
        self._store["distancias"] = pd.concat([existing, df], ignore_index=True)  # type: ignore[arg-type]

    def save_zones(self, df: gpd.GeoDataFrame) -> None:
        self._store["zones"] = df.copy()

    def save_matrix_validation(self, df: pd.DataFrame) -> None:
        self._store["matriz_validacion"] = df.copy()

    def save_travel_times_stations(self, df: pd.DataFrame) -> None:
        self._store["travel_times_stations"] = df.copy()

    def save_metadata_lineas(self, df: pd.DataFrame) -> None:
        self._store["metadata_lineas"] = df.copy()

    def save_metadata_ramales(self, df: pd.DataFrame) -> None:
        self._store["metadata_ramales"] = df.copy()

    def execute(self, sql: str) -> None:
        pass  # no-op for in-memory adapter

    def query(self, sql: str) -> pd.DataFrame:
        return _query_store(self._store, sql)

    def save_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._store[table_name] = df.copy()

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        return self._store.get(table_name, pd.DataFrame()).copy()

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        existing = self._store.get(table_name, pd.DataFrame())
        self._store[table_name] = pd.concat([existing, df], ignore_index=True)

    def has_routes(self) -> bool:
        return not self.get_routes().empty


class InMemoryDashAdapter:
    """In-process DashPort implementation for testing."""

    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}

    def save_indicator(self, df: pd.DataFrame, name: str) -> None:
        self._store[name] = df.copy()

    def get_indicator(self, name: str) -> pd.DataFrame:
        return self._store.get(name, pd.DataFrame()).copy()

    def list_indicators(self) -> list[str]:
        return [k for k, v in self._store.items() if not v.empty]

    def execute(self, sql: str) -> None:
        pass  # no-op for in-memory adapter

    def query(self, sql: str) -> pd.DataFrame:
        return _query_store(self._store, sql)

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        existing = self._store.get(table_name, pd.DataFrame())
        self._store[table_name] = pd.concat([existing, df], ignore_index=True)


class InMemoryGeneralAdapter:
    """In-process GeneralPort implementation for testing."""

    def __init__(self) -> None:
        self._runs: list[dict] = []
        self._store: dict[str, pd.DataFrame] = {}

    def get_completed_runs(self) -> pd.DataFrame:
        return pd.DataFrame(self._runs)

    def register_run(self, alias: str, process: str) -> None:
        self._runs.append({
            "corrida": alias,
            "process": process,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def execute(self, sql: str) -> None:
        pass  # no-op for in-memory adapter

    def query(self, sql: str) -> pd.DataFrame:
        return _query_store(self._store, sql)

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        existing = self._store.get(table_name, pd.DataFrame())
        self._store[table_name] = pd.concat([existing, df], ignore_index=True)

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        return self._store.get(table_name, pd.DataFrame()).copy()

    def run_exists(self, alias: str) -> bool:
        runs = self.get_completed_runs()
        if runs.empty or "corrida" not in runs.columns:
            return False
        return alias in runs["corrida"].values

    def clear_runs(self) -> None:
        self._runs.clear()
