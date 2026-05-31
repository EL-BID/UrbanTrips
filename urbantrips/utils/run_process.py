import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from urbantrips.storage.context import StorageContext
from urbantrips.storage.ports import BatchSpec
from urbantrips.utils.utils import leer_configs_generales

logger = logging.getLogger(__name__)


def _build_ctx() -> StorageContext:
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    from urbantrips.utils import utils

    configs = utils.leer_configs_generales(autogenerado=False)
    alias = configs.get("alias_db", "")
    base = Path("data") / "db"
    base.mkdir(parents=True, exist_ok=True)
    return StorageContext(
        data=DuckDBDataAdapter(base / f"{alias}_data.duckdb"),
        insumos=DuckDBInsumoAdapter(base / f"{alias}_insumos.duckdb"),
        dash=DuckDBDashAdapter(base / f"{alias}_dash.duckdb"),
        general=DuckDBGeneralAdapter(base / f"{alias}_general.duckdb"),
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

    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]
    nombre_archivo_gps = configs["nombre_archivo_gps"]
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

    configs_usuario = utils.leer_configs_generales(autogenerado=False)
    corridas = configs_usuario.get("corridas", None)
    
    alias_insumos = configs_usuario['alias_db_insumos']
    if ((Path() / "data" / "db" / f'{alias_insumos}_general.sqlite').exists()):

        if corridas is None or len(corridas) == 0:
            raise ValueError("No se han definido corridas en el archivo de configuracion.")
        
        
        if (len(alias_db) == 0):
            # Verificar corridas que no terminaron de correr
            corridas_anteriores = levanto_tabla_sql(
                "corridas",
                "general",
                query="SELECT DISTINCT corrida FROM corridas WHERE process = 'transactions_completed'",
            )
            if len(corridas_anteriores) > 0:
                corridas_anteriores = corridas_anteriores.corrida.unique().tolist()
                print('Corridas anteriores:', corridas_anteriores)
                for i in corridas:
                    if (Path() / "data" / "db" / f'{i}_data.sqlite').exists() & (i not in corridas_anteriores):
                        path_data = Path() / "data" / "db" / f"{i}_data.sqlite"
                        path_dash = Path() / "data" / "db" / f"{i}_dash.sqlite"
                        if path_data.exists():
                            path_data.unlink()
                            print(f"Se borró {path_data}")
                        if path_dash.exists():
                            path_dash.unlink()
                            print(f"Se borró {path_dash}")
        else:
            path_ = configs_usuario["alias_db_insumos"]
            path_insumos = Path() / "data" / "db" / f"{path_}_insumos.sqlite"
            path_general = Path() / "data" / "db" / f"{path_}_general.sqlite"

    if not alias_db:
        return

    alias = configs_usuario.get("alias_db", "")

    if alias_db == "all":
        if ctx is not None:
            ctx.general.clear_runs()
        for suffix in ["data", "dash", "insumos", "general"]:
            for ext in [".sqlite", ".duckdb"]:
                p = Path("data") / "db" / f"{alias}_{suffix}{ext}"
                if p.exists():
                    p.unlink()
                    logger.info("Se borró %s", p)
    else:
        for suffix in ["data", "dash"]:
            for ext in [".sqlite", ".duckdb"]:
                p = Path("data") / "db" / f"{alias_db}_{suffix}{ext}"
                if p.exists():
                    p.unlink()
                    logger.info("Se borró %s", p)


def _get_n_batches() -> int:
    configs = leer_configs_generales()
    return int(configs.get("n_batches", 30))


def _auto_n_batches(ctx: StorageContext, safety_factor: float = 0.4) -> int:
    """Compute n_batches so each batch fits within safety_factor of available RAM."""
    import psutil

    vm = psutil.virtual_memory()
    logger.info(
        "[n_batches] RAM — total: %.1f GB, available: %.1f GB (%.0f%% free)",
        vm.total / 1e9, vm.available / 1e9, vm.available / vm.total * 100,
    )

    total_rows = ctx.data.query(
        "SELECT COUNT(*) AS n FROM transacciones"
    ).iloc[0, 0]

    if total_rows == 0:
        logger.info("[n_batches] No transactions found — using 1 batch")
        return 1

    sample = ctx.data.query("SELECT * FROM transacciones LIMIT 10000")
    bytes_per_row = sample.memory_usage(deep=True).sum() / max(len(sample), 1)
    total_mb = total_rows * bytes_per_row / 1e6
    target = vm.available * safety_factor
    n = max(math.ceil(total_rows * bytes_per_row / target), 1)

    logger.info(
        "[n_batches] Auto-tuned: %d rows × %.0f B/row = %.0f MB total"
        " | target %.1f GB/batch (%.0f%% of available) → %d batches",
        total_rows, bytes_per_row, total_mb,
        target / 1e9, safety_factor * 100, n,
    )
    return n


def _resolve_n_batches(ctx: StorageContext) -> int:
    """Return configured n_batches if set, otherwise auto-tune from available RAM."""
    import psutil

    configs = leer_configs_generales()
    if "n_batches" in configs:
        n = int(configs["n_batches"])
        vm = psutil.virtual_memory()
        logger.info(
            "[n_batches] Using configured value: %d"
            " (available RAM: %.1f GB — auto-tune skipped)",
            n, vm.available / 1e9,
        )
        return n
    logger.info("[n_batches] No n_batches in config — auto-tuning from available RAM")
    return _auto_n_batches(ctx)


def _get_parallel_workers(n_batches: int) -> int:
    configs = leer_configs_generales()
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
            configs = leer_configs_generales()
            nombres_variables = configs["nombres_variables_trx"]
            formato_fecha = configs["formato_fecha"]
            tipo_trx_invalidas = configs.get("tipo_trx_invalidas")
            lineas_contienen_ramales = configs.get("lineas_contienen_ramales", True)
            nombre_archivo_trx = configs["nombre_archivo_trx"]
            csv_path = os.path.join("data", "data_ciudad", nombre_archivo_trx)
            logger.info("[Phase 1] Ingesting %s from %s", corrida, csv_path)
            ingest_day_csv(
                ctx=ctx,
                csv_path=csv_path,
                nombres_variables=nombres_variables,
                formato_fecha=formato_fecha,
                tipo_trx_invalidas=tipo_trx_invalidas,
                lineas_contienen_ramales=lineas_contienen_ramales,
            )
            if configs.get("nombre_archivo_gps") is not None:
                gps_corridas.append(corrida)

        n_batches = _get_n_batches()
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
        configs = leer_configs_generales()
        logger.info("[Phase 1] Ingesting GPS for %s", corrida)
        trx.process_and_upload_gps_table(
            ctx=ctx,
            nombre_archivo_gps=configs["nombre_archivo_gps"],
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


def _can_parallelize_batches(ctx: StorageContext) -> bool:
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    return isinstance(ctx.data, DuckDBDataAdapter)


def _build_legs_for_batch_worker(batch: BatchSpec, trx_order_params: dict):
    from urbantrips.datamodel import legs
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.utils import utils

    configs = utils.leer_configs_generales(autogenerado=False)
    alias = configs.get("alias_db", "")
    data_adapter = DuckDBDataAdapter.__new__(DuckDBDataAdapter)
    data_adapter._path = Path("data") / "db" / f"{alias}_data.duckdb"
    data_adapter._read_only = True
    ctx = StorageContext(data=data_adapter, insumos=None, dash=None, general=None)
    legs_df, duplicate_cards = legs.build_legs_from_transactions(
        ctx,
        trx_order_params,
        batch=batch,
    )
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

    logger.info("  using %d worker processes", parallel_workers)
    results = []
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(_build_legs_for_batch_worker, batch, trx_order_params): batch
            for batch in batches
        }
        for future in as_completed(futures):
            results.append(future.result())

    for batch, legs_df, duplicate_cards in sorted(
        results,
        key=lambda result: result[0].batch_id,
    ):
        logger.debug("  saving batch %d/%d", batch.batch_id + 1, batch.total_batches)
        ctx.data.save_legs(legs_df, batch)
        _save_duplicate_cards(ctx, duplicate_cards)


def _clear_current_run_travel_times(ctx: StorageContext) -> None:
    """Clear derived travel-time tables for the current run days before rebuilding."""
    run_days = ctx.data.get_run_days()
    if run_days.empty:
        return
    dias_str = ", ".join(f"'{d}'" for d in run_days["dia"].tolist())
    for table in ["travel_times_legs", "travel_times_trips"]:
        ctx.data.execute(f"DELETE FROM {table} WHERE dia IN ({dias_str})")


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

    nombre_archivo_gps = configs.get("nombre_archivo_gps")
    if nombre_archivo_gps is not None:
        services.process_services(ctx, line_ids=None)
        legs.assign_gps_origin(ctx)
        legs.assign_gps_destination(ctx)

    tiempos_viaje_estaciones = configs.get("tiempos_viaje_estaciones")
    if tiempos_viaje_estaciones is not None:
        legs.assign_stations_od(ctx)

    _clear_current_run_travel_times(ctx)
    trips.compute_trips_travel_time(ctx)
    legs.add_distance_and_travel_time(ctx)
    if batches is None:
        trips.rearrange_trip_id_same_od(ctx)
    else:
        logger.info("[Phase 3] Rearranging trip ids for %d traveler batches", len(batches))
        for batch in batches:
            logger.debug("  batch %d/%d", batch.batch_id + 1, batch.total_batches)
            trips.rearrange_trip_id_same_od(ctx, batch=batch)


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

    configs = utils.leer_configs_generales()
    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }
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
        if not ctx.data.has_rows("etapas"):
            raise RuntimeError(
                "Step 'legs' requires ingest to have been run first "
                "(etapas table is empty). Run with --through legs first."
            )
    elif step == "outputs":
        if not ctx.data.has_rows("etapas", where="h3 IS NOT NULL"):
            raise RuntimeError(
                "Step 'outputs' requires legs to have been run first "
                "(etapas.h3 is empty). Run with --through outputs first."
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
