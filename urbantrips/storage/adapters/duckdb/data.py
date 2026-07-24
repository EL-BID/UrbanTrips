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
        # hora_inicio/hora_fin were incorrectly typed FLOAT in old DBs; drop so CREATE TABLE rebuilds
        # them as TEXT. DuckDB reporta las columnas TEXT como 'VARCHAR' en information_schema, así que
        # comparar sólo contra "TEXT" hacía que la condición fuera SIEMPRE verdadera y la tabla se
        # borrara en CADA apertura de conexión de escritura (por eso kpi_by_day_line_service quedaba
        # vacía: --step dashboard destruía lo que --step outputs había escrito). Con "VARCHAR" incluido,
        # el drop sólo se dispara en DBs legacy con el tipo viejo (caso raro; se regenera re-corriendo).
        row = self._conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'kpi_by_day_line_service' AND column_name = 'hora_inicio'"
        ).fetchone()
        if row and row[0].upper() not in ("TEXT", "VARCHAR"):
            self._conn.execute("DROP TABLE IF EXISTS kpi_by_day_line_service")

        # id_gps_o/id_gps_d agregadas a travel_times_legs para QA de la imputacion
        # de anclas GPS. CREATE TABLE IF NOT EXISTS no las agrega en DBs ya
        # existentes, asi que se migran aca si la tabla existe sin esas columnas.
        table_exists = self._conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'travel_times_legs'"
        ).fetchone()
        if table_exists:
            existing_cols = {
                r[0]
                for r in self._conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'travel_times_legs'"
                ).fetchall()
            }
            for col in ("id_gps_o", "id_gps_d"):
                if col not in existing_cols:
                    self._conn.execute(
                        f"ALTER TABLE travel_times_legs ADD COLUMN {col} INT"
                    )

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

    def get_transactions(
        self, batch: BatchSpec | None = None, run_days: list[str] | None = None
    ) -> pd.DataFrame:
        # `transacciones` es ACUMULATIVA (nunca se limpia por corrida). Sin el filtro
        # por día, el path serial de Fase 2 cargaría TODOS los días acumulados y recién
        # build_legs_dataframe los descarta en pandas → RAM/tiempo O(acumulado). Con
        # run_days la lectura queda acotada a la corrida (mismo resultado). Espeja el
        # filtro que ya aplica get_transactions_for_chunk en el path paralelo.
        conds = []
        if batch is not None:
            conds.append(f"hash(id_tarjeta) % {batch.total_batches} = {batch.batch_id}")
        if run_days:
            dias = ", ".join(f"'{d}'" for d in run_days)
            conds.append(f"dia IN ({dias})")
        where = f"WHERE {' AND '.join(conds)}" if conds else ""
        return self._conn.execute(f"SELECT * FROM transacciones {where}").fetchdf()

    def get_transactions_for_chunk(self, batch_ids: list[int], total_batches: int, run_days: list[str] | None = None) -> pd.DataFrame:
        """Load rows for the given batch IDs in one scan, with _batch_id column for splitting.

        Reads the batch_id stamped at standardize time (= hash(id_tarjeta) % n_batches,
        see standardize_raw_to_transacciones) instead of recomputing the hash over the
        whole 253M-row table on every chunk. Because transacciones is written ordered by
        batch_id, a chunk's contiguous batch_ids let DuckDB prune row groups and read
        only its slice — turning ~one full scan per chunk into ~one full scan total
        across Phase 2. total_batches is kept for signature compatibility, unused now.

        `run_days` acota la lectura a los días de la corrida. `transacciones` es
        ACUMULATIVA (nunca se limpia por corrida), así que sin este filtro cada worker
        cargaría TODOS los días acumulados y recién build_legs_dataframe los descarta en
        pandas — la RAM de Fase 2 crecería con lo acumulado y no con los días de la
        corrida (causó un OOM incremental a escala AMBA). Con el filtro la memoria queda
        acotada a la corrida; el resultado es idéntico (el worker filtra los mismos días).
        """
        ids = ", ".join(str(b) for b in batch_ids)
        where = f"batch_id IN ({ids})"
        if run_days:
            dias = ", ".join(f"'{d}'" for d in run_days)
            where += f" AND dia IN ({dias})"
        return self._conn.execute(
            f"SELECT *, batch_id AS _batch_id FROM transacciones WHERE {where}"
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

    def geolocate_raw_transactions_from_gps(self, lineas_contienen_ramales: bool) -> None:
        """Fill missing latitud/longitud in transacciones_raw from the
        nearest preceding gps ping for the same vehicle (dia + id_linea
        [+ id_ramal] + interno). Mirrors the legacy `geolocalizar_trx`
        semantics. Rows with no matching prior ping are left NULL.

        Implemented as a full vectorized table rebuild (CREATE OR REPLACE
        TABLE ... AS SELECT) rather than a correlated UPDATE ... FROM:
        DuckDB is column-oriented and an UPDATE keyed on a per-row rowid
        join degrades to row-by-row execution. The nearest-match lookup
        (one row per trx needing geolocation) is computed once in a CTE
        via ROW_NUMBER() and joined back by rowid; rows that already have
        non-null latitud/longitud pass through COALESCE unchanged.
        """
        ramal_join = (
            "AND (t.id_ramal = g.id_ramal OR (t.id_ramal IS NULL AND g.id_ramal IS NULL))"
            if lineas_contienen_ramales else ""
        )
        select_cols = ", ".join(
            f"COALESCE(t.{c}, m.{c}) AS {c}" if c in ("latitud", "longitud") else f"t.{c}"
            for c in schema.TRANSACCIONES_RAW_COLUMNS
        )
        self._conn.execute(f"""
            CREATE OR REPLACE TABLE transacciones_raw AS
            WITH nearest_gps AS (
                SELECT
                    t.rowid AS rid,
                    g.latitud,
                    g.longitud,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.rowid ORDER BY g.fecha DESC
                    ) AS rn
                FROM transacciones_raw t
                JOIN gps g
                    ON t.dia = g.dia
                    AND t.id_linea = g.id_linea
                    AND t.interno = g.interno
                    {ramal_join}
                    AND g.fecha <= t.fecha_ts
                WHERE t.latitud IS NULL OR t.longitud IS NULL
            )
            SELECT {select_cols}
            FROM transacciones_raw t
            LEFT JOIN (
                SELECT rid, latitud, longitud FROM nearest_gps WHERE rn = 1
            ) m ON t.rowid = m.rid
        """)

    def standardize_raw_to_transacciones(self, n_batches: int, id_offset: int) -> None:
        """
        Move rows from transacciones_raw into transacciones, computing:
        - batch_id = hash(id_tarjeta) % n_batches  (DuckDB native hash, always unsigned)
        - id = sequential integer starting at id_offset
        - factor_expansion from factor_expansion_raw (or 1 if null)
        Filters out cards where any transaction has a NULL in a critical column,
        so only fully-valid cards are promoted.

        Rows are written ORDER BY batch_id so each row group holds a contiguous
        batch_id range: Phase 2 reads one chunk of contiguous batch_ids at a time
        (get_transactions_for_chunk), and DuckDB's row-group min/max zonemaps then
        prune to just that slice instead of re-scanning the whole table per chunk.
        (On a fresh run transacciones starts empty, so this one INSERT orders the
        whole table; on incremental re-ingest each append is internally ordered.)
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
                ORDER BY batch_id
            """)

    # ── batch-indexed reads ───────────────────────────────────────────────────

    def get_transactions_for_batch(self, batch: "BatchSpec") -> pd.DataFrame:
        """Read all transactions for one traveler batch across all ingested days.

        Filters on the stamped batch_id column (= hash(id_tarjeta) % n_batches) instead
        of recomputing the hash; with transacciones ordered by batch_id this prunes to
        the batch's row groups.
        """
        return self._conn.execute(
            "SELECT * FROM transacciones WHERE batch_id = ?",
            [batch.batch_id],
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

    def update_leg_trip_ids(self, df: pd.DataFrame, dia: str | None = None) -> None:
        """Update only id_viaje and id_etapa for existing legs, matched by id.

        Much faster than save_legs for rearrange operations that only modify
        trip/stage numbering: avoids full DELETE + parquet staging + INSERT.

        `dia` es un predicado semánticamente REDUNDANTE (etapas.id es único
        global y `df` trae solo filas de ese día) pero clave para performance:
        sin él el planner no tiene ningún filtro sobre etapas y cada UPDATE
        barre la tabla entera (169M filas, una vez por partición procesada).
        Con etapas clusterizada por dia, el zonemap poda a los row-groups del
        día. No cambia qué filas se actualizan.
        """
        if df.empty:
            return
        updates = df[["id", "id_viaje", "id_etapa"]].copy()
        dia_filter = f" AND etapas.dia = '{dia}'" if dia is not None else ""
        self._conn.register("_trip_id_updates", updates)
        try:
            self._conn.execute(f"""
                UPDATE etapas
                SET id_viaje = u.id_viaje,
                    id_etapa = u.id_etapa
                FROM _trip_id_updates u
                WHERE etapas.id = u.id{dia_filter}
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
        """No-op: idx_etapas_dia_od_validado ya no se recrea (auditoría 2026-07-18:
        los índices ART son puro costo — ver schema.ALL_INDEXES). begin_* lo dropea
        por si la DB es legacy y lo trae."""
        return

    def begin_bulk_leg_writes(self, drop_batch_index: bool = False) -> None:
        """Drop the id + secondary etapas indexes before a bulk DELETE+INSERT of legs.

        save_legs (Phase 2) and create_trips_from_legs_and_fex (Phase 4) rewrite
        large slices of etapas. With the ART indexes active, DuckDB maintains them
        row by row during the INSERT — the dominant cost at scale, and it GROWS with
        the table (the unique-key index on id is the worst offender: at month scale,
        253M keys, the per-batch stall climbs 7→13→18 min...). etapas has no PRIMARY
        KEY (DuckDB can't drop one mid-load); id is covered by a plain idx_etapas_id
        that is dropped here and rebuilt once in bulk by end_bulk_leg_writes, so the
        batch INSERTs are plain appends.

        idx_etapas_batch is only needed by save_legs (deletes by batch_id), i.e.
        Phase 2. Phase 4 deletes by dia and nothing there queries batch_id, so
        create_trips passes drop_batch_index=True to stop maintaining it row by
        row during the 28 per-day INSERTs; end_bulk_leg_writes rebuilds it.
        """
        self._conn.execute("DROP INDEX IF EXISTS idx_etapas_id")
        self._conn.execute("DROP INDEX IF EXISTS idx_etapas_dia_od_validado")
        self._conn.execute("DROP INDEX IF EXISTS idx_etapas_dia_line_ramal_interno")
        self._batch_index_dropped = False
        if drop_batch_index:
            self._conn.execute("DROP INDEX IF EXISTS idx_etapas_batch")
            self._batch_index_dropped = True

    def end_bulk_leg_writes(self) -> None:
        """No-op: los índices de etapas ya no se recrean (auditoría 2026-07-18: puro
        costo de escritura, cero ganancia de lectura — ver schema.ALL_INDEXES). Los
        begin_* siguen dropeando por si la DB es legacy y los trae, así el bulk INSERT
        queda como append puro sin mantenimiento de ART."""
        self._batch_index_dropped = False

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

    @staticmethod
    def _rebuild_memory_limit():
        """Temporary DuckDB memory_limit for the etapas rebuild, adapted to whatever
        machine this runs on. DuckDB spills to disk past the limit (it never OOM-
        crashes on it), so the ONLY risk is setting it ABOVE physically-free RAM.
        We therefore take a fraction of the RAM AVAILABLE right now, keep an OS
        reserve, and never drop below the pinned 25%-of-RAM default: a big idle
        machine gets a faster rebuild, a small or busy one just stays at the safe
        default (worst case = more spill, never a crash). Returns None (→ caller
        leaves the current limit untouched) if psutil is unavailable.
        """
        try:
            import psutil

            vm = psutil.virtual_memory()
            # Decimal GB throughout, to match the pinned "25% of RAM" default exactly.
            reserve = min(6e9, vm.total * 0.25)   # 6 GB, or 25% on tiny machines
            floor = vm.total * 0.25               # the pinned default; never go below it
            budget = min(vm.available * 0.6, vm.total - reserve)
            gb = max(int(budget / 1e9), int(floor / 1e9), 2)
            return f"{gb}GB"
        except Exception:
            return None

    def update_leg_destinations_from_parquet(self, parquet_glob: str) -> None:
        """Write the destination columns back for the RUN's days (day-scoped rewrite).

        infer_destinations stages every RUN-DAY leg's (id, dia, h3_d, od_validado,
        etapa_validada) to parquet. Los días de corridas previas están CONGELADOS
        (no se re-imputan) → no tienen fila staged. Antes esto reconstruía TODA
        `etapas` (O(acumulado)) para actualizar 3 columnas de destino de solo los
        run-days: la reescritura del slice congelado era puro costo que crecía con lo
        acumulado. Ahora se reescribe SOLO el slice de run-days (derivados del parquet):
        se materializa el slice actualizado ORDER BY dia y se hace DELETE+INSERT de
        esos días. Resultado BIT-IDÉNTICO (COALESCE deja igual las columnas sin fila
        staged; los congelados ni se tocan) y O(días de la corrida).

        Se sigue evitando `UPDATE ... FROM` (DuckDB lo degrada a DELETE+INSERT fila por
        fila que mantiene los índices ART). Acá el bulk DELETE+INSERT del slice es un
        append: `etapas` no tiene índices secundarios (política 2026-07-18, ver
        begin/end_bulk_leg_writes) → sin mantenimiento ART. El ORDER BY dia mantiene el
        clustering por día (cada corrida appendea sus días como bloque contiguo, y las
        corridas procesan días ascendentes) → el zonemap sigue podando `WHERE dia=X`
        downstream. Atómico: DELETE+INSERT en una transacción; ante error, ROLLBACK
        deja `etapas` intacta. El guard de row-count aborta antes de tocar nada.
        """
        glob_sql = parquet_glob.replace("'", "''")
        # run-days = días presentes en el parquet staged (infer stagea solo run-days)
        dias = self._conn.execute(
            f"SELECT DISTINCT dia FROM read_parquet('{glob_sql}')"
        ).fetchdf()["dia"].tolist()
        if not dias:
            return
        dias_str = ", ".join(f"'{d}'" for d in dias)

        n_before = self._conn.execute(
            f"SELECT count(*) FROM etapas WHERE dia IN ({dias_str})"
        ).fetchone()[0]
        if n_before == 0:
            return

        dest = {"h3_d", "od_validado", "etapa_validada"}
        cols = ", ".join(_ETAPAS_COLUMNS)
        select_cols = ", ".join(
            f"COALESCE(u.{c}, e.{c}) AS {c}" if c in dest else f"e.{c}"
            for c in _ETAPAS_COLUMNS
        )

        prev_mem = self._conn.execute(
            "SELECT current_setting('memory_limit')"
        ).fetchone()[0]
        bump = self._rebuild_memory_limit()
        if bump:
            self._conn.execute(f"PRAGMA memory_limit='{bump}'")
        try:
            # slice de run-days con los destinos mergeados, ORDER BY dia (clustering)
            self._conn.execute("DROP TABLE IF EXISTS _ut_dest_new")
            self._conn.execute(
                f"CREATE TEMP TABLE _ut_dest_new AS "
                f"SELECT {select_cols} FROM etapas e "
                f"LEFT JOIN read_parquet('{glob_sql}') u "
                f"ON e.id = u.id AND e.dia = u.dia "
                f"WHERE e.dia IN ({dias_str}) "
                f"ORDER BY e.dia"
            )
            n_after = self._conn.execute(
                "SELECT count(*) FROM _ut_dest_new"
            ).fetchone()[0]
            if n_after != n_before:
                self._conn.execute("DROP TABLE IF EXISTS _ut_dest_new")
                raise RuntimeError(
                    f"etapas day-scoped rebuild row-count mismatch "
                    f"({n_after} != {n_before}); aborted, etapas left intact"
                )
            # swap del slice, atómico: borrar run-days y re-appendear los actualizados
            self._conn.execute("BEGIN TRANSACTION")
            try:
                self._conn.execute(f"DELETE FROM etapas WHERE dia IN ({dias_str})")
                self._conn.execute(
                    f"INSERT INTO etapas ({cols}) SELECT {cols} FROM _ut_dest_new"
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("DROP TABLE IF EXISTS _ut_dest_new")
        finally:
            if bump:
                self._conn.execute(f"PRAGMA memory_limit='{prev_mem}'")

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
                # SCOPED BY DAY: batch_id partitions travelers and spans ALL days,
                # so a bare `WHERE batch_id = ?` would wipe this batch's rows for
                # previously-processed days too (incremental data loss — etapas of
                # old runs vanish while viajes keep them). df carries only the
                # current run's legs (filtered to run days in build_legs_dataframe),
                # so restrict the delete to the days actually present in it.
                if batch is not None:
                    dias_batch = df["dia"].unique().tolist()
                    ph = ", ".join("?" for _ in dias_batch)
                    self._conn.execute(
                        f"DELETE FROM etapas WHERE batch_id = ? AND dia IN ({ph})",
                        [batch.batch_id, *dias_batch],
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
