import logging
import math
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from urbantrips.storage.context import StorageContext
from urbantrips.storage.ports import BatchSpec
from urbantrips.utils.utils import leer_configs_generales

logger = logging.getLogger(__name__)


def _build_ctx() -> StorageContext:
    from urbantrips.storage.adapters.duckdb.data import (
        DuckDBDataAdapter,
        configure_global_duckdb,
    )
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    from urbantrips.utils import utils
    from urbantrips.utils.paths import get_paths

    # The default duckdb.sql() connection is used as a compute engine over
    # pandas frames all over the pipeline; pin its limits so peak memory is
    # machine-independent.
    configure_global_duckdb()

    configs = utils.leer_configs_generales(autogenerado=False)
    # alias_db       → prefix for run-specific DBs (data, dash, general)
    # alias_db_insumos → prefix for the shared insumos DB
    # If alias_db is absent, fall back to alias_db_insumos so single-alias
    # configs keep working.
    alias_insumos = configs.get("alias_db_insumos", configs.get("alias_db", ""))
    alias_data    = configs.get("alias_db",          alias_insumos)
    db_dir = get_paths().db_dir
    db_dir.mkdir(parents=True, exist_ok=True)
    return StorageContext(
        data=DuckDBDataAdapter(db_dir / f"{alias_data}_data.duckdb"),
        insumos=DuckDBInsumoAdapter(db_dir / f"{alias_insumos}_insumos.duckdb"),
        dash=DuckDBDashAdapter(db_dir / f"{alias_data}_dash.duckdb"),
        general=DuckDBGeneralAdapter(db_dir / f"{alias_data}_general.duckdb"),
    )


def inicializo_ambiente(ctx: StorageContext):
    from urbantrips.carto.carto import guardo_zonificaciones
    from urbantrips.carto.routes import process_routes_geoms, process_routes_metadata
    from urbantrips.carto.stops import create_stops_table
    from urbantrips.utils import utils
    from urbantrips.utils.check_configs import check_config
    from urbantrips.utils.fs import create_directories

    corridas_nuevas = []

    configs_usuario = utils.leer_configs_generales(autogenerado=False)
    corridas = configs_usuario.get("corridas", None)

    if corridas is None or len(corridas) == 0:
        raise ValueError("No se han definido corridas en el archivo de configuracion.")

    if not ctx.insumos.has_routes():
        logger.info("Inicializo ambiente por primera vez")
        check_config(corridas[0])
        create_directories()
        process_routes_metadata(ctx)
        process_routes_geoms(ctx)
        create_stops_table(ctx)
        guardo_zonificaciones(ctx)

    for alias_db in corridas:
        if not ctx.general.run_exists(alias_db):
            corridas_nuevas.append(alias_db)

    return corridas_nuevas


def procesar_transacciones(ctx: StorageContext, corrida: str):
    from urbantrips.carto import carto, routes
    from urbantrips.datamodel import legs, services, trips
    from urbantrips.datamodel import transactions as trx
    from urbantrips.datamodel.misc import persist_indicators
    from urbantrips.destinations import destinations as dest
    from urbantrips.geo import geo
    from urbantrips.kpi.kpi import compute_kpi
    from urbantrips.utils import utils
    from urbantrips.utils.check_configs import check_config

    check_config(corrida)

    configs = utils.leer_configs_generales(autogenerado=False)
    geolocalizar_trx_config = configs["geolocalizar_trx"]
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]
    nombre_archivo_gps = configs.get("nombre_archivo_gps")
    if nombre_archivo_gps is None and configs.get("usa_archivo_gps", False):
        nombre_archivo_gps = f"{corrida}_gps.csv"
    nombres_variables_gps = configs["nombres_variables_gps"]
    tiempos_viaje_estaciones = configs["tiempos_viaje_estaciones"]
    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    resolucion_h3 = configs["resolucion_h3"]
    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }

    ring_size = geo.get_h3_buffer_ring_size(resolucion_h3, tolerancia_parada_destino)

    trx.create_transactions(
        ctx,
        geolocalizar_trx_config,
        nombre_archivo_trx,
        nombres_variables_trx,
        formato_fecha,
        col_hora,
        tipo_trx_invalidas,
        nombre_archivo_gps,
        nombres_variables_gps,
    )

    legs.create_legs_from_transactions(ctx, trx_order_params)
    carto.update_stations_catchment_area(ring_size, ctx)
    dest.infer_destinations(ctx)

    if nombre_archivo_gps is not None:
        services.process_services(ctx, line_ids=None)
        legs.assign_gps_origin(ctx)

    trips.rearrange_trip_id_same_od(ctx)
    legs.assign_time_distances(ctx)

    if tiempos_viaje_estaciones is not None:
        legs.assign_stations_od(ctx)

    trips.create_trips_from_legs_and_fex(ctx)
    routes.infer_routes_geoms(ctx)
    routes.build_routes_from_official_inferred(ctx)

    trx.write_transactions_to_db(ctx, corrida)
    compute_kpi(ctx)
    persist_indicators(ctx)


def borrar_corridas(ctx: StorageContext | None = None, alias_db="all"):
    from urbantrips.utils import utils
    from urbantrips.utils.paths import get_paths

    configs_usuario = utils.leer_configs_generales(autogenerado=False)

    if not alias_db:
        return

    alias_insumos = configs_usuario.get("alias_db_insumos", configs_usuario.get("alias_db", ""))
    alias_data    = configs_usuario.get("alias_db", alias_insumos)
    base = get_paths().db_dir

    if alias_db == "all":
        if ctx is not None:
            ctx.general.clear_runs()
        for suffix, alias in [("data", alias_data), ("dash", alias_data),
                               ("insumos", alias_insumos), ("general", alias_data)]:
            for ext in [".sqlite", ".duckdb"]:
                p = base / f"{alias}_{suffix}{ext}"
                if p.exists():
                    p.unlink()
                    logger.info("Se borró %s", p)
    else:
        for suffix in ["data", "dash"]:
            for ext in [".sqlite", ".duckdb"]:
                p = base / f"{alias_db}_{suffix}{ext}"
                if p.exists():
                    p.unlink()
                    logger.info("Se borró %s", p)


def _configured_n_batches() -> int | None:
    """Return n_batches from config when explicitly set to a positive int.

    A missing key, a null/empty value, or 0 all mean "not configured" and make
    the caller fall back to auto-resolution.
    """
    configs = leer_configs_generales(autogenerado=False)
    raw = configs.get("n_batches")
    if raw is None or raw == "":
        return None
    n = int(raw)
    return n if n > 0 else None


def _auto_n_batches(ctx: StorageContext, safety_factor: float = 0.4) -> int:
    """Compute n_batches from RAM and CPU so workers stay saturated."""
    import psutil

    vm = psutil.virtual_memory()
    logger.info(
        "[n_batches] RAM — total: %.1f GB, available: %.1f GB (%.0f%% free)",
        vm.total / 1e9, vm.available / 1e9, vm.available / vm.total * 100,
    )

    total_rows = ctx.data.query(
        "SELECT COUNT(*) AS n FROM transacciones_raw"
    ).iloc[0, 0]

    if total_rows == 0:
        logger.info("[n_batches] No raw transactions found — using 1 batch")
        return 1

    sample = ctx.data.query("SELECT * FROM transacciones_raw LIMIT 10000")
    bytes_per_row = sample.memory_usage(deep=True).sum() / max(len(sample), 1)
    total_mb = total_rows * bytes_per_row / 1e6
    cpu_workers = max(multiprocessing.cpu_count() - 1, 1)

    # Each chunk loads `cpu_workers` batches at once; size that chunk to fit in RAM.
    target_chunk = vm.available * safety_factor
    target_per_batch = target_chunk / cpu_workers
    raw = max(math.ceil(total_rows * bytes_per_row / target_per_batch), cpu_workers)
    # Round up to the nearest multiple of cpu_workers so every chunk is full
    ram_batches = math.ceil(raw / cpu_workers) * cpu_workers

    logger.info(
        "[n_batches] Auto-tuned: %d rows × %.0f B/row = %.0f MB total"
        " | target %.1f GB/chunk (%d workers × %.1f GB/batch, %.0f%% of available)"
        " → %d batches",
        total_rows, bytes_per_row, total_mb,
        target_chunk / 1e9, cpu_workers, target_per_batch / 1e9,
        safety_factor * 100, ram_batches,
    )
    return ram_batches


def _resolve_n_batches(ctx: StorageContext) -> int:
    """Single source of truth for the batch count, shared by ingest and legs.

    Resolution order:
    1. Explicit positive int in config → use it verbatim.
    2. Otherwise (key missing, null/empty, or 0) auto-resolve:
       a. If transacciones already carries stamped batch_ids, reuse that exact
          partition count (MAX(batch_id) + 1). This keeps the legs phase — and
          incremental re-ingests — consistent with whatever ingest stamped, even
          across separate process invocations, so no batch is left unread.
       b. On a fresh database (nothing stamped yet) auto-tune from the raw rows
          about to be standardized and the available RAM.
    """
    n = _configured_n_batches()
    if n is not None:
        logger.info("[n_batches] Using configured value: %d", n)
        return n

    import duckdb as _duckdb

    try:
        stamped = ctx.data.query(
            "SELECT MAX(batch_id) AS m FROM transacciones"
        ).iloc[0, 0]
    except _duckdb.CatalogException:
        stamped = None
        n = int(stamped) + 1
        logger.info(
            "[n_batches] No n_batches in config — inheriting stamped partition"
            " count from transacciones: %d batches", n,
        )
        return n

    logger.info(
        "[n_batches] No n_batches in config and nothing stamped yet —"
        " auto-tuning from raw data and available RAM"
    )
    return _auto_n_batches(ctx)


def _get_parallel_workers(n_batches: int) -> int:
    configs = leer_configs_generales(autogenerado=False)
    configured = configs.get("parallel_workers")
    if configured is not None:
        return max(min(int(configured), n_batches), 1)
    cpu_workers = max(multiprocessing.cpu_count() - 1, 1)
    return max(min(cpu_workers, n_batches), 1)


def _ingest_all_days(ctx: StorageContext, corridas: list[str]) -> None:
    """Phase 1: stream every corrida's CSV into transacciones_raw."""
    import os
    from urbantrips.datamodel import transactions as trx
    from urbantrips.datamodel.ingestion import ingest_day_csv
    from urbantrips.utils.check_configs import check_config

    gps_corridas = []

    ctx.data.clear_raw()

    try:
        for corrida in corridas:
            check_config(corrida)
            # Read the base config directly — derive filenames from corrida name
            # rather than relying on the autogenerated file.
            configs = leer_configs_generales(autogenerado=False)
            nombres_variables = configs["nombres_variables_trx"]
            formato_fecha = configs["formato_fecha"]
            tipo_trx_invalidas = configs.get("tipo_trx_invalidas")
            lineas_contienen_ramales = configs.get("lineas_contienen_ramales", True)
            # Prefer an explicit value from the config; fall back to the {corrida}_trx.csv convention
            from urbantrips.utils.paths import get_paths
            nombre_archivo_trx = configs.get("nombre_archivo_trx") or f"{corrida}_trx.csv"
            csv_path = str(get_paths().input_dir / nombre_archivo_trx)
            logger.info("[Phase 1] Ingesting %s from %s", corrida, csv_path)
            ingest_day_csv(
                ctx=ctx,
                csv_path=csv_path,
                nombres_variables=nombres_variables,
                formato_fecha=formato_fecha,
                tipo_trx_invalidas=tipo_trx_invalidas,
                lineas_contienen_ramales=lineas_contienen_ramales,
            )
            usa_gps = configs.get("usa_archivo_gps", False)
            if usa_gps:
                gps_corridas.append(corrida)

        n_batches = _resolve_n_batches(ctx)
        id_offset = ctx.data.get_max_id("transacciones")
        logger.info("[Phase 1] Standardizing raw → transacciones (n_batches=%d)", n_batches)
        ctx.data.standardize_raw_to_transacciones(n_batches=n_batches, id_offset=id_offset)
    finally:
        ctx.data.clear_raw()

    all_dias = ctx.data.query(
        "SELECT DISTINCT dia FROM transacciones ORDER BY dia"
    ).rename(columns={"dia": "dia"})
    ctx.data.save_run_days(all_dias)

    for corrida in gps_corridas:
        check_config(corrida)
        configs = leer_configs_generales(autogenerado=False)
        nombre_archivo_gps = configs.get("nombre_archivo_gps") or f"{corrida}_gps.csv"
        logger.info("[Phase 1] Ingesting GPS for %s", corrida)
        trx.process_and_upload_gps_table(
            ctx=ctx,
            nombre_archivo_gps=nombre_archivo_gps,
            nombres_variables_gps=configs["nombres_variables_gps"],
            formato_fecha=configs["formato_fecha"],
        )


def _create_legs_for_batch(ctx: StorageContext, batch, trx_order_params: dict) -> None:
    """Create raw legs for one traveler batch."""
    from urbantrips.datamodel import legs

    legs_df, duplicate_cards = legs.build_legs_from_transactions(
        ctx,
        trx_order_params,
        batch=batch,
    )
    ctx.data.save_legs(legs_df, batch)
    _save_duplicate_cards(ctx, duplicate_cards)


def _save_duplicate_cards(ctx: StorageContext, duplicate_cards: pd.DataFrame) -> None:
    if len(duplicate_cards) > 0:
        ctx.data.append_raw(duplicate_cards, "tarjetas_duplicadas")


def _save_batch_results(
    ctx: StorageContext,
    legs_df: "pd.DataFrame",
    batch: BatchSpec,
    duplicate_cards: "pd.DataFrame",
) -> None:
    ctx.data.save_legs(legs_df, batch)
    if len(duplicate_cards):
        ctx.data.append_raw(duplicate_cards, "tarjetas_duplicadas")


def _can_parallelize_batches(ctx: StorageContext) -> bool:
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    return isinstance(ctx.data, DuckDBDataAdapter)


def _build_legs_for_batch_worker(
    batch: BatchSpec,
    trx: "pd.DataFrame",
    dias_ultima_corrida: "pd.DataFrame",
    trx_order_params: dict,
    h3_res: int,
):
    from urbantrips.datamodel.legs import build_legs_dataframe

    legs_df, duplicate_cards = build_legs_dataframe(trx, dias_ultima_corrida, trx_order_params, h3_res=h3_res)
    return batch, legs_df, duplicate_cards


def _create_legs_for_batches(
    ctx: StorageContext,
    batches: list[BatchSpec],
    trx_order_params: dict,
    parallel_workers: int,
) -> None:
    if parallel_workers <= 1 or len(batches) <= 1 or not _can_parallelize_batches(ctx):
        for batch in batches:
            logger.debug("  batch %d/%d", batch.batch_id + 1, batch.total_batches)
            _create_legs_for_batch(ctx, batch, trx_order_params)
        return

    # Process in chunks of parallel_workers: one DuckDB scan per chunk keeps
    # peak RAM at chunk_size × batch_size instead of the full dataset.
    # Saves run in a background thread so the main thread can immediately read
    # the next worker result (unblocking the IPC pipe). The save thread is
    # flushed before each chunk's DuckDB read to prevent concurrent connection
    # access between the save thread and the main thread.
    from concurrent.futures import ThreadPoolExecutor

    dias_ultima_corrida = ctx.data.get_run_days()
    h3_res = leer_configs_generales(autogenerado=False)["resolucion_h3"]
    n = len(batches)
    t_start = time.monotonic()
    done = 0

    logger.info("  using %d worker processes", parallel_workers)
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor, \
         ThreadPoolExecutor(max_workers=1) as save_pool:
        save_futures: list = []
        for chunk_start in range(0, n, parallel_workers):
            # Flush previous chunk's saves before touching DuckDB again
            for sf in save_futures:
                sf.result()
            save_futures.clear()

            chunk = batches[chunk_start: chunk_start + parallel_workers]

            # One scan loads this chunk's rows; DuckDB computes _batch_id for splitting
            chunk_trx = ctx.data.get_transactions_for_chunk(
                [b.batch_id for b in chunk], n
            )
            splits = {
                b.batch_id: chunk_trx[chunk_trx["_batch_id"] == b.batch_id]
                    .drop(columns=["_batch_id"]).reset_index(drop=True)
                for b in chunk
            }
            del chunk_trx

            futures = {
                executor.submit(
                    _build_legs_for_batch_worker,
                    b, splits[b.batch_id], dias_ultima_corrida, trx_order_params, h3_res,
                ): b
                for b in chunk
            }
            del splits

            for future in as_completed(futures):
                batch_res, legs_df, duplicate_cards = future.result()
                save_futures.append(
                    save_pool.submit(_save_batch_results, ctx, legs_df, batch_res, duplicate_cards)
                )

                done += 1
                elapsed = time.monotonic() - t_start
                eta = (elapsed / done * (n - done)) if done < n else 0.0
                logger.info(
                    "  [Phase 2] %d/%d batches done (%.0f%%) — %.0fs elapsed, ~%.0fs remaining",
                    done, n, done / n * 100, elapsed, eta,
                )

        for sf in save_futures:
            sf.result()


def _clear_current_run_legs(ctx: StorageContext) -> None:
    """Clear etapas rows for the current run days before rebuilding.

    Deleting by day (not by batch_id) is robust to n_batches changing between
    reruns: the old rows were stored under whatever batch_id mapping was active
    at the time, so a batch_id-based delete would miss them.
    """
    import duckdb as _duckdb
    run_days = ctx.data.get_run_days()
    if run_days.empty:
        return
    dias_str = ", ".join(f"'{d}'" for d in run_days["dia"].tolist())
    try:
        ctx.data.execute(f"DELETE FROM etapas WHERE dia IN ({dias_str})")
        logger.info("Cleared existing etapas for current run days before rebuilding")
    except (_duckdb.CatalogException, Exception):
        pass  # Table doesn't exist yet


def _clear_current_run_travel_times(ctx: StorageContext) -> None:
    """Clear derived travel-time tables for the current run days before rebuilding."""
    import duckdb as _duckdb
    run_days = ctx.data.get_run_days()
    if run_days.empty:
        return
    dias_str = ", ".join(f"'{d}'" for d in run_days["dia"].tolist())
    for table in ["travel_times_legs", "travel_times_trips"]:
        try:
            ctx.data.execute(f"DELETE FROM {table} WHERE dia IN ({dias_str})")
        except (_duckdb.CatalogException, Exception):
            pass  # Table doesn't exist yet; will be created by assign_time_distances


def _enrich_all_legs(ctx: StorageContext, configs: dict, batches=None) -> None:
    """Run shared leg enrichment once after all batch legs have been created."""
    from urbantrips.carto import carto
    from urbantrips.datamodel import legs, services, trips
    from urbantrips.destinations import destinations as dest
    from urbantrips.geo import geo

    resolucion_h3 = configs["resolucion_h3"]
    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    ring_size = geo.get_h3_buffer_ring_size(resolucion_h3, tolerancia_parada_destino)

    carto.update_stations_catchment_area(ring_size, ctx)
    dest.infer_destinations(ctx)

    usa_archivo_gps = configs.get("usa_archivo_gps", False) or configs.get("nombre_archivo_gps") is not None
    if usa_archivo_gps:
        services.process_services(ctx, line_ids=None)
        legs.assign_gps_origin(ctx)

    tiempos_viaje_estaciones = configs.get("tiempos_viaje_estaciones")

    if batches is None:
        trips.rearrange_trip_id_same_od(ctx)
    else:
        n = len(batches)
        logger.info("Iniciando rearrange_trip_id_same_od (%d batches)", n)
        ts_total = time.perf_counter()
        for batch in batches:
            logger.info("  rearrange_trip_id_same_od batch %d/%d ...", batch.batch_id + 1, n)
            trips.rearrange_trip_id_same_od(ctx, batch=batch, _silent=True)
        logger.info(
            "Finalizado rearrange_trip_id_same_od (%d batches, %.2fs)",
            n, time.perf_counter() - ts_total,
        )

    # Compute distances and travel times for all legs; writes travel_times_legs/trips
    legs.assign_time_distances(ctx)

    if tiempos_viaje_estaciones is not None:
        legs.assign_stations_od(ctx)


def _build_final_outputs(ctx: StorageContext) -> None:
    """Build aggregate trip/user outputs after leg enrichment is complete."""
    from urbantrips.datamodel import trips

    trips.create_trips_from_legs_and_fex(ctx)
    trips.add_distance_and_travel_time(ctx)


def run_ingest(ctx: StorageContext) -> None:
    """Phase 1: ingest all pending corridas."""
    from urbantrips.utils import utils

    corridas = inicializo_ambiente(ctx)
    logger.info("[Phase 1] Ingesting %d day(s)", len(corridas))
    _ingest_all_days(ctx, corridas)


def run_legs(ctx: StorageContext) -> None:
    """Phases 2+3: create legs and enrich them."""
    from urbantrips.utils import utils

    configs = utils.leer_configs_generales(autogenerado=False)
    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }
    _clear_current_run_legs(ctx)
    n_batches = _resolve_n_batches(ctx)
    batches = ctx.data.get_user_batches(n_batches)
    parallel_workers = _get_parallel_workers(n_batches) if _can_parallelize_batches(ctx) else 1
    logger.info("[Phase 2] Creating legs for %d traveler batches", n_batches)
    _create_legs_for_batches(ctx, batches, trx_order_params, parallel_workers)
    logger.info("[Phase 3] Enriching legs")
    _enrich_all_legs(ctx, configs, batches=batches)


def run_outputs(ctx: StorageContext) -> None:
    """Phase 4 + routes + KPIs + indicators."""
    from urbantrips.carto import routes
    from urbantrips.datamodel.misc import persist_indicators
    from urbantrips.kpi.kpi import compute_kpi

    logger.info("[Phase 4] Building trips and users")
    _build_final_outputs(ctx)
    routes.infer_routes_geoms(ctx)
    routes.build_routes_from_official_inferred(ctx)
    compute_kpi(ctx)
    persist_indicators(ctx)


def run_dashboard(ctx: StorageContext) -> None:
    """Dashboard preparation."""
    from urbantrips.preparo_dashboard.preparo_dashboard import preparo_indicadores_dash

    preparo_indicadores_dash(ctx)


_STEP_ORDER = ["ingest", "legs", "outputs", "dashboard"]


def check_prerequisites(step: str, ctx: StorageContext) -> None:
    """Raise RuntimeError if the data required before `step` is absent."""
    if step == "ingest":
        return
    if step == "legs":
        if not ctx.data.has_rows("transacciones"):
            raise RuntimeError(
                "Step 'legs' requires ingest to have been run first "
                "(transacciones table is empty). Run with --through legs first."
            )
    elif step == "outputs":
        if not ctx.data.has_rows("etapas", where="h3_o IS NOT NULL"):
            raise RuntimeError(
                "Step 'outputs' requires legs to have been run first "
                "(etapas.h3_o is empty). Run with --through outputs first."
            )
    elif step == "dashboard":
        if not ctx.data.has_rows("viajes"):
            raise RuntimeError(
                "Step 'dashboard' requires outputs to have been run first "
                "(viajes table is empty). Run with --through outputs first."
            )


def run_all(ctx: StorageContext | None = None, borrar_corrida="", crear_dashboard=True):
    inicio = time.time()
    logger.info("borrar_corrida = '%s'", borrar_corrida)
    logger.info("crear_dashboard = %s", crear_dashboard)

    if ctx is None and borrar_corrida:
        borrar_corridas(alias_db=borrar_corrida)
        borrar_corrida = ""

    if ctx is None:
        ctx = _build_ctx()

    borrar_corridas(ctx, borrar_corrida)
    if borrar_corrida:
        ctx = _build_ctx()

    run_ingest(ctx)
    run_legs(ctx)
    run_outputs(ctx)
    if crear_dashboard:
        run_dashboard(ctx)

    fin = time.time()
    logger.info("tiempo total de la corrida: %.2f min", (fin - inicio) / 60)
