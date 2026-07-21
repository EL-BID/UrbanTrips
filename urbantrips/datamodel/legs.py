import gc
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import pandas as pd
import geopandas as gpd
import numpy as np
import h3
import sqlite3
from datetime import datetime
import math
from urbantrips.geo.geo import (
    referenciar_h3,
    convert_h3_to_resolution,
    classify_leg_into_station,
    get_epsg_m,
)
from urbantrips.utils.utils import (
    duracion,
    leer_configs_generales,
    agrego_indicador,
    VELOCIDAD_MAXIMA_KMH,
    modos_con_ramal,
    id_ramal_efectivo,
    RAMAL_SENTINEL,
)
from urbantrips.storage.context import StorageContext
from urbantrips.storage.ports import BatchSpec

# from urbantrips.kpi.kpi import add_distances_to_legs
from urbantrips.carto.compute_distances import compute_od_distances
import warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
    module=r".*urbantrips.*services",
)


@duracion
def create_legs_from_transactions(
    ctx: StorageContext, trx_order_params, batch: BatchSpec | None = None
):
    """
    Esta function toma las transacciones de la db
    las estructura en etapas con sus id y id viaje
    y crea la tabla etapas en la db
    """
    legs, tarjetas_duplicadas = build_legs_from_transactions(
        ctx, trx_order_params, batch
    )
    ctx.data.save_legs(legs, batch)
    if len(tarjetas_duplicadas) > 0:
        ctx.data.save_raw(tarjetas_duplicadas, "tarjetas_duplicadas")


def build_legs_from_transactions(
    ctx: StorageContext,
    trx_order_params,
    batch: BatchSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build legs and duplicated-card records without writing them to storage.
    """
    dias_ultima_corrida = ctx.data.get_run_days()
    trx = ctx.data.get_transactions(batch)
    return build_legs_dataframe(trx, dias_ultima_corrida, trx_order_params)


def build_legs_dataframe(
    trx: pd.DataFrame,
    dias_ultima_corrida: pd.DataFrame,
    trx_order_params,
    h3_res: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the legs dataframe from transactions and return duplicate-card metadata.
    """
    legs = trx[trx.dia.isin(dias_ultima_corrida.dia)]

    print("Eliminando transacciones con longitud o latitud igual a cero")
    n_trx_ = len(legs)
    print(f"Transacciones antes de eliminar: {len(legs)}")
    legs = legs.loc[(legs.longitud != 0) & (legs.latitud != 0), :]
    print(f"Transacciones eliminadas: {n_trx_ - len(legs)}")

    # parse dates using local timezone
    legs = legs.copy()
    legs["fecha"] = pd.to_datetime(legs.fecha, unit="s", errors="coerce")

    # asignar id h3
    if h3_res is None:
        h3_res = leer_configs_generales(autogenerado=False)["resolucion_h3"]
    legs = referenciar_h3(df=legs, res=h3_res, nombre_h3="h3_o")

    # crear columna delta
    if trx_order_params["criterio"] == "orden_trx":
        legs["delta"] = None
    elif trx_order_params["criterio"] == "fecha_completa":
        legs = crear_delta_trx(legs)
    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    # asignar nuevo id tarjeta trx simultaneas
    legs, tarjetas_duplicadas = _change_card_id_for_concurrent_trx(
        legs, trx_order_params
    )

    # crear columna delta nuevamente para los nuevos ids tarjeta
    if trx_order_params["criterio"] == "orden_trx":
        legs["delta"] = None
    elif trx_order_params["criterio"] == "fecha_completa":
        legs = crear_delta_trx(legs)
    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    # asignar ids de viajes y etapas
    legs = asignar_id_viaje_etapa(legs, trx_order_params)

    legs = legs.reindex(
        columns=[
            "id",
            "id_tarjeta",
            "dia",
            "id_viaje",
            "id_etapa",
            "tiempo",
            "hora",
            "modo",
            "id_linea",
            "id_ramal",
            "interno",
            "genero",
            "tarifa",
            "latitud",
            "longitud",
            "h3_o",
            "factor_expansion",
        ]
    )

    legs = legs.rename(columns={"factor_expansion": "factor_expansion_original"})
    return legs, tarjetas_duplicadas


def crear_delta_trx(trx):
    """
    Takes a transactions df with a date in datetime format with hour minutes
    and seconds and computes a time delta in seconds with the previous
    transaction
    Parameters
    ----------
    trx : pandas DataFrame
        transactions data with complete datetime

    Returns
    ----------

    X: pandas DataFrame
        transactions data with time delta in seconds

    """

    trx = trx.sort_values(["dia", "id_tarjeta", "fecha"])

    # Calcular la cantidad de minutos con respecto a la trx anterior
    trx["hora_shift"] = (
        trx.reindex(columns=["dia", "id_tarjeta", "fecha"])
        .groupby(["dia", "id_tarjeta"])
        .shift(1)
    )
    trx["delta"] = trx.fecha - trx.hora_shift
    trx["delta"] = trx["delta"].fillna(pd.Timedelta(seconds=0))
    trx["delta"] = trx.delta.dt.total_seconds()
    trx["delta"] = trx["delta"].map(int)

    return trx


def change_card_id_for_concurrent_trx(
    trx, trx_order_params, dias_ultima_corrida, ctx: StorageContext
):
    """
    Changes card id for those cards with concurrent transactions as defined by
    the parameters in  trx_order_params.
    Adds a _0 to the card id for the first concurrent transaction, _1 for the
    next and so on. It creates a duplicated cards table in the db.

    Parameters
    ----------
    trx : pandas DataFrame
        transactions data

    trx_order_params : dict
        parameters that define order of transactions and concurrent criteria

    dias_ultima_corrida: pd.Series
        last processsed days for urbantrips

    Returns
    ----------

    X: pandas DataFrame
        legs with new card ids

    """
    trx_c, tarjetas_duplicadas = _change_card_id_for_concurrent_trx(
        trx, trx_order_params
    )
    if len(tarjetas_duplicadas) > 0:

        # # borro si ya existen etapas de una corrida anterior
        # values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
        # query = f"DELETE FROM tarjetas_duplicadas WHERE dia IN ({values})"
        # conn.execute(query)
        # conn.commit()

        # tarjetas_duplicadas.to_sql(
        #     "tarjetas_duplicadas", conn, if_exists="append", index=False
        # )
        ctx.data.save_raw(tarjetas_duplicadas, "tarjetas_duplicadas")

    return trx_c


def _change_card_id_for_concurrent_trx(trx, trx_order_params):
    trx_c = trx.copy()
    trx_c, tarjetas_duplicadas = pago_doble_tarjeta(trx_c, trx_order_params)
    logger.debug("Subiendo %d tarjetas duplicadas a la db", len(tarjetas_duplicadas))
    return trx_c, tarjetas_duplicadas


def pago_doble_tarjeta(trx, trx_order_params):
    """
    Takes a transaction dataframe with a time delta and
    a time window for duplicates in minutes,
    detects duplicated transactions and assigns a new card id

    Parameters
    ----------
    trx : pandas DataFrame
        transactions data

    trx_order_params : dict
        parameters that define order of transactions and concurrent criteria

    Returns
    ----------

    trx: pandas DataFrame
        transactions with new card ids

    tarjetas_duplicadas: pandas DataFrame
        dataframe with old and new card ids

    """

    ventana_duplicado = trx_order_params["ventana_duplicado"]

    cols = trx.columns

    if trx_order_params["criterio"] == "fecha_completa":
        diff_segundos = ventana_duplicado * 60

        trx["fecha_aux"] = trx["fecha"].astype(str).str[-8:]

        parts = trx["fecha_aux"].str.split(":", expand=True).astype(int)
        trx["fecha_aux"] = parts[0].mul(3600) + parts[1].mul(60) + parts[2]

    elif trx_order_params["criterio"] == "orden_trx":
        trx.loc[:, ["fecha_aux"]] = trx["hora"]
        diff_segundos = 1

    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    trx = trx.sort_values(
        ["dia", "id_tarjeta", "id_linea", "fecha_aux", "orden_trx"]
    ).reset_index(drop=True)

    trx["datetime_proximo"] = trx["fecha_aux"].shift(-1)

    trx["diff_datetime"] = (trx.fecha_aux - trx.datetime_proximo).abs()

    trx["diff_datetime2"] = trx.groupby(
        ["dia", "id_tarjeta", "id_linea"]
    ).diff_datetime.shift(+1)

    g = ["dia", "id_tarjeta", "id_linea"]
    trx["_is_start"] = (
        trx.diff_datetime2.isna() | (trx.diff_datetime2 > diff_segundos)
    ).astype(int)
    trx["_run_id"] = trx.groupby(g)["_is_start"].cumsum()
    trx["nro"] = trx.groupby(g + ["_run_id"]).cumcount()
    trx = trx.drop(columns=["_is_start", "_run_id"])

    trx["id_tarjeta_nuevo"] = (
        trx["id_tarjeta"] + "_" + trx["nro"].astype(int).astype(str)
    )

    tarjetas_duplicadas = (
        trx.loc[trx["nro"] > 0]
        .reindex(columns=["dia", "id_tarjeta", "id_tarjeta_nuevo"])
        .rename(columns={"id_tarjeta": "id_tarjeta_original"})
        .drop_duplicates()
    )

    trx["id_tarjeta"] = trx["id_tarjeta_nuevo"]
    trx = trx.reindex(columns=cols)

    return trx, tarjetas_duplicadas


def cambiar_id_tarjeta_trx_simul_fecha(trx, ventana_duplicado):
    """
    Takes a transaction dataframe with a time delta and
    a time window for duplciates in minutes,
    detects duplicated transactions and assigns a new card id

    Parameters
    ----------
    trx : pandas DataFrame
        transactions data

    ventana_duplicado : int
        minutes to consider two transactions as duplicated

    Returns
    ----------

    X: pandas DataFrame
        legs with new card ids

    """

    # convertir ventana en segundos
    ventana_duplicado = ventana_duplicado * 60
    # seleccinar atributos para considerar duplicados
    subset_dup = ["dia", "id_tarjeta", "id_linea"]

    # detectar duplicados por criterio de delta y atributos
    duplicados_ventana = (trx.delta > 0) & (trx.delta <= ventana_duplicado)
    duplicados_atributos = trx.duplicated(subset=subset_dup)
    duplicados = duplicados_ventana & duplicados_atributos
    trx["duplicados_ventana"] = duplicados_ventana

    subset_dup = subset_dup + ["duplicados_ventana"]
    # crear para duplicado por delta dentro de dia tarjeta linea
    # un nuevo id_tarjeta con un incremental para cada duplicado
    nro_duplicado = trx[duplicados].groupby(subset_dup).cumcount() + 1
    nro_duplicado = nro_duplicado.map(str)

    logger.debug("Hay %d casos duplicados", duplicados.sum())

    if duplicados.sum() > 0:
        # crear una tabla de registro de cambio de id tarjeta
        tarjetas_duplicadas = trx.loc[nro_duplicado.index, ["dia", "id_tarjeta"]]
        tarjetas_duplicadas = tarjetas_duplicadas.rename(
            columns={"id_tarjeta": "id_tarjeta_original"}
        )
        tarjetas_duplicadas["id_tarjeta_nuevo"] = (
            tarjetas_duplicadas.id_tarjeta_original + "_" + nro_duplicado
        )

        # crear un nuevo vector con los incrementales y concatenarlos
        nuevo_id_tarjeta = pd.Series(["0"] * len(trx))
        nuevo_id_tarjeta.loc[nro_duplicado.index] = nro_duplicado
        trx.id_tarjeta = trx.id_tarjeta.map(str) + "_" + nuevo_id_tarjeta
    else:
        tarjetas_duplicadas = pd.DataFrame()
    trx = trx.drop("duplicados_ventana", axis=1)

    logger.debug("Fin creacion de nuevos id tarjetas para duplicados con delta")
    return trx, tarjetas_duplicadas


def cambiar_id_tarjeta_trx_simul_orden_trx(trx):
    """
    Esta funcion toma un DF de trx y asigna un nuevo id_tarjeta a los casos
    duplicados en base al dia,id_tarjeta, hora y orden_trx para un mismo modo
    y ubicacion
    """
    subset_dup = [
        "dia",
        "id_tarjeta",
        "hora",
        "orden_trx",
        "id_linea",
        "h3_o",
    ]

    # detectar duplicados por criterio de atributos
    duplicados = trx.duplicated(subset=subset_dup)

    if not duplicados.any():
        tarjetas_duplicadas = pd.DataFrame()
        return trx, tarjetas_duplicadas

    # crear para duplicado por dia tarjeta linea
    # un nuevo id_tarjeta con un incremental para cada duplicado
    nro_duplicado = trx[duplicados].groupby(subset_dup).cumcount() + 1
    nro_duplicado = nro_duplicado.map(str)

    # crear una tabla de registro de cambio de id tarjeta
    tarjetas_duplicadas = trx.loc[nro_duplicado.index, ["dia", "id_tarjeta"]]
    tarjetas_duplicadas = tarjetas_duplicadas.rename(
        columns={"id_tarjeta": "id_tarjeta_original"}
    )
    tarjetas_duplicadas["id_tarjeta_nuevo"] = (
        tarjetas_duplicadas.id_tarjeta_original + nro_duplicado
    )

    logger.debug("Hay %d casos duplicados", duplicados.sum())
    # crear un nuevo vector con los incrementales y concatenarlos

    if duplicados.sum() > 0:
        nuevo_id_tarjeta = pd.Series(["0"] * len(trx))
        nuevo_id_tarjeta.loc[nro_duplicado.index] = nro_duplicado
        trx.id_tarjeta = trx.id_tarjeta + "_" + nuevo_id_tarjeta

    logger.debug("Fin creacion de nuevos id tarjetas para duplicados con orden trx")
    return trx, tarjetas_duplicadas


def asignar_id_viaje_etapa(trx, trx_order_params):
    """
    Takes a transaction dataframe with a time delta and
    a ordering parameters dict and assigns trips and leg id
    based on the transactions ordering parameters

    Parameters
    ----------
    trx : pandas DataFrame
        transactions data

    trx_order_params : dict
        dict with parameters for ordering criteria, trips window in minutes
        and duplicated time window in minutes

    Returns
    ----------

    X: pandas DataFrame
        legs with new trips and legs ids

    """

    if trx_order_params["criterio"] == "orden_trx":
        logger.debug("Utilizando orden_trx")
        trx = asignar_id_viaje_etapa_orden_trx(trx)

    elif trx_order_params["criterio"] == "fecha_completa":
        logger.debug("Utilizando fecha_completa")
        ventana_viajes = trx_order_params["ventana_viajes"]
        trx = asignar_id_viaje_etapa_fecha_completa(trx, ventana_viajes)

    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    return trx


def asignar_id_viaje_etapa_fecha_completa(trx, ventana_viajes):
    """
    Takes a transaction dataframe with a time delta in seconds and
    a time window in minutes

    The window is anchored to the FIRST transaction of the trip: every tap
    within `ventana_viajes` minutes of the trip's first tap belongs to the
    same trip, regardless of the gap between consecutive taps. The first tap
    beyond the window starts a new trip (and a new window). This mirrors
    integrated-fare schemes such as AMBA's, where the discount window runs
    from the first boarding. Smaller cities can configure a shorter window
    in configuraciones_generales.yaml.

    Parameters
    ----------
    trx : pandas DataFrame
        transactions data

    ventana_viajes : int
        time window in minutes, measured from the trip's first transaction,
        to consider transactions as part of the same trip

    Returns
    ----------

    X: pandas DataFrame
        legs with new trips and legs ids

    """

    # turn into seconds
    ventana_viajes = ventana_viajes * 60

    trx = trx.sort_values(["id_tarjeta", "fecha"])

    # Calcular los id_viajes
    trx["id_viaje"] = trx.groupby(["id_tarjeta"])["delta"].transform(
        lambda s: _trip_ids_from_deltas(s.to_numpy(dtype=np.float64), ventana_viajes)
    )
    lista = ["id_tarjeta", "id_viaje"]
    trx["id_etapa"] = trx.groupby(lista).cumcount() + 1
    return trx


def asignar_id_viaje_etapa_orden_trx(trx):
    """
    Esta funcion toma un DF de trx y asigna id_viaje y id_etapa
    en base al dia, hora y orden_trx
    """
    variables_secuencia = [
        "id_tarjeta",
        "tiempo",
        "hora",
        "orden_trx",
        "modo",
        "id_linea",
    ]
    trx = trx.sort_values(variables_secuencia)
    trx["secuencia"] = trx.groupby(["id_tarjeta"]).cumcount() + 1

    trx["nro_viaje_temp"] = trx.secuencia - trx["orden_trx"]

    temp = trx.groupby(["id_tarjeta", "nro_viaje_temp"]).size().reset_index()
    temp["id_viaje"] = temp.groupby(["id_tarjeta"]).cumcount() + 1
    temp = temp.reindex(columns=["id_tarjeta", "nro_viaje_temp", "id_viaje"])

    trx = trx.merge(temp, on=["id_tarjeta", "nro_viaje_temp"], how="left")
    trx = trx.drop(["secuencia", "nro_viaje_temp"], axis=1)

    sort = ["id_tarjeta", "id_viaje", "hora", "orden_trx"]
    trx = trx.sort_values(sort)
    g = ["id_tarjeta", "id_viaje"]
    trx["id_etapa"] = trx.groupby(g).cumcount() + 1

    return trx


def _trip_ids_from_deltas(deltas: np.ndarray, ventana_sec: float) -> np.ndarray:
    """
    Vectorized replacement for the former Python for-loop.
    Uses numpy searchsorted to jump O(n_trips) times instead of iterating O(n_rows).
    """
    n = len(deltas)
    if n == 0:
        return np.array([], dtype=np.int32)
    cumsum = np.cumsum(deltas)
    ids = np.ones(n, dtype=np.int32)
    viaje, base, j = 1, 0.0, 0
    while j < n:
        k = int(np.searchsorted(cumsum[j:], base + ventana_sec, side="right")) + j
        if k >= n:
            break
        viaje += 1
        ids[k:] = viaje
        base = float(cumsum[k])
        j = k + 1
    return ids


def crear_viaje_id_acumulada(df, ventana_viajes=120):
    return _trip_ids_from_deltas(
        np.asarray(df.delta, dtype=np.float64), float(ventana_viajes)
    ).tolist()


def _gps_origin_dia(legs, gps):
    """Empareja (pandas puro) cada etapa del día con el punto GPS más cercano en
    el tiempo (mismo dia/linea/ramal/interno, tolerancia 7 min).

    Función PURA respecto de la DB: puede correr en un worker process o en el main.
    """
    cols = ["dia", "id_linea", "id_ramal", "interno", "fecha", "id"]
    legs["fecha"] = pd.to_datetime(legs["dia"] + " " + legs["tiempo"])
    gps["fecha"] = pd.to_datetime(gps["fecha"], unit="s")

    legs_to_join = legs.reindex(columns=cols).sort_values("fecha")
    gps_to_join = gps.reindex(columns=cols).sort_values("fecha")

    legs_to_gps_o = pd.merge_asof(
        legs_to_join,
        gps_to_join,
        on="fecha",
        by=["dia", "id_linea", "id_ramal", "interno"],
        direction="nearest",
        tolerance=pd.Timedelta("7 minutes"),
        suffixes=("_legs", "_gps"),
    )
    return legs_to_gps_o.reindex(columns=["dia", "id_legs", "id_gps"]).dropna()


@duracion
def assign_gps_origin(ctx: StorageContext):
    """
    This function read legs data and if there is gps table
    assigns a gps to the leg origin
    """
    configs = leer_configs_generales(autogenerado=False)
    usa_gps = configs.get("usa_archivo_gps", False)

    if not usa_gps:
        return

    # Se procesa UN DÍA POR VEZ para acotar el pico de RAM (antes se levantaban
    # etapas y gps del mes entero de una sola vez). El merge_asof usa dia dentro
    # de `by`, así que el emparejamiento nunca cruza días: el resultado por día es
    # idéntico al del mes entero. La tabla existe por schema; se limpia una vez
    # (reemplaza la semántica de save_raw de la versión mes-entero) y luego se
    # appendea por día.
    dias = sorted(ctx.data.get_run_days()["dia"].tolist())
    dias_str = ", ".join(f"'{d}'" for d in dias)
    if dias_str:
        try:
            ctx.data.execute(
                f"DELETE FROM legs_to_gps_origin WHERE dia IN ({dias_str})"
            )
        except Exception as e:
            logger.debug("[assign_gps_origin] DELETE omitido: %s", e)

    def _fetch_origin_inputs(dia):
        """Lecturas del día (main): etapas y gps mínimos para el merge_asof."""
        legs = ctx.data.query(
            f"""
            SELECT e.dia, e.id_linea, e.id_ramal, e.interno, e.tiempo, e.id
            FROM etapas e
            WHERE e.dia = '{dia}'
            """
        )
        gps = ctx.data.query(
            f"""
            SELECT g.dia, g.id_linea, g.id_ramal, g.interno, g.fecha, g.id
            FROM gps g
            WHERE g.dia = '{dia}'
            """
        )
        return legs, gps

    n_workers = _parallel_day_workers(len(dias))

    if n_workers <= 1:
        # ── Camino serial (comportamiento previo, mismos resultados) ──
        for i, dia in enumerate(dias, 1):
            logger.info("[assign_gps_origin] día %d/%d (%s)", i, len(dias), dia)
            legs, gps = _fetch_origin_inputs(dia)
            if gps.empty or legs.empty:
                del legs, gps
                continue
            legs_to_gps_o = _gps_origin_dia(legs, gps)
            ctx.data.append_raw(legs_to_gps_o, "legs_to_gps_origin")
            del legs, gps, legs_to_gps_o
            gc.collect()
        return

    # ── Camino paralelo: merge_asof (pandas puro) en workers; lecturas y
    # escrituras a DuckDB en el main. Chunks con barrera para acotar RAM.
    logger.info("[assign_gps_origin] paralelizando: %d días en vuelo", n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, len(dias), n_workers):
            chunk = dias[chunk_start: chunk_start + n_workers]
            futures = {}
            for offset, dia in enumerate(chunk, 1):
                logger.info(
                    "[assign_gps_origin] día %d/%d (%s) — leyendo insumos",
                    chunk_start + offset, len(dias), dia,
                )
                legs, gps = _fetch_origin_inputs(dia)
                if gps.empty or legs.empty:
                    del legs, gps
                    continue
                futures[executor.submit(_gps_origin_dia, legs, gps)] = dia
                del legs, gps
            for future in as_completed(futures):
                ctx.data.append_raw(future.result(), "legs_to_gps_origin")
            gc.collect()


def _fetch_time_distance_inputs_dia(ctx, dia, next_dia):
    """Lee (en el PROCESO MAIN) los insumos de _gps_destino_y_tiempos_dia para un día.

    Separado del cómputo para poder despachar el cómputo a workers: DuckDB no es
    fork/spawn-safe, así que toda lectura/escritura a la DB queda en el main y los
    workers reciben DataFrames (mismo patrón que la Fase 2 de create_legs).
    El ORDER BY del gps es imprescindible: los cumsum de distancia acumulada por
    servicio dependen del orden por fecha dentro de (dia, linea, ramal, interno).
    """
    gps_dias = [dia] + ([next_dia] if next_dia else [])
    gps_str = ", ".join(f"'{d}'" for d in gps_dias)
    gps = ctx.data.query(
        f"""
        SELECT g.* FROM gps g
        WHERE g.dia IN ({gps_str})
        ORDER BY dia, id_linea, id_ramal, interno, fecha
        """
    )
    legs_to_gps_o = ctx.data.query(
        f"""
        SELECT lo.id_legs, lo.id_gps AS id_gps_o
        FROM legs_to_gps_origin lo
        WHERE lo.dia = '{dia}'
        """
    )
    return gps, legs_to_gps_o


@duracion
def _gps_destino_y_tiempos_dia(
    dia, next_dia, legs_all, gps, legs_to_gps_o,
    metadata_lineas, matriz, modos_ramal, legs_h3_res,
):
    """Imputación de GPS de destino + tiempos/distancias de viaje para UN día.

    Reproduce, para un solo día, exactamente lo que la versión previa hacía sobre
    todos los días juntos: geocodifica el gps del día (+ madrugada de next_dia para
    legs que cruzan medianoche), imputa el GPS de destino (_process_dia) y
    calcula distance_route / distance_route_gps / travel_time_min / kmh_*.

    Función PURA respecto de la DB: recibe gps y legs_to_gps_o ya leídos
    (_fetch_time_distance_inputs_dia) para poder ejecutarse en un worker process.

    Devuelve (travel_times, travel_times_trips, legs_to_gps_d) ya armados para el día.
    legs_to_gps_d es None si no se imputó ningún destino GPS ese día.
    """
    logger.info("[_gps_destino_y_tiempos_dia] día %s", dia)
    # gps no tiene modo: se trae de metadata_lineas para el id_ramal efectivo.
    gps = gps.merge(metadata_lineas, how="left", on="id_linea")
    gps["id_ramal"] = id_ramal_efectivo(gps["modo"], gps["id_ramal"], modos_ramal)

    legs = legs_all[(legs_all.id_linea.isin(gps.id_linea.unique()))].copy()
    legs = legs.merge(
        metadata_lineas[["id_linea", "id_linea_agg"]], how="left", on="id_linea"
    )
    legs["id_ramal"] = id_ramal_efectivo(legs["modo"], legs["id_ramal"], modos_ramal)
    legs["fecha"] = pd.to_datetime(legs["dia"] + " " + legs["tiempo"])

    gps_h3_res = h3.get_resolution(gps["h3"].sample().item())
    gps = referenciar_h3(
        gps, res=legs_h3_res, nombre_h3="h3_legs_res", lat="latitud", lon="longitud"
    )
    gps["fecha_gps"] = gps.fecha.map(lambda ts: pd.Timestamp(ts, unit="s"))
    gps["hora"] = gps.fecha_gps.dt.hour
    legs["h3_d_gps_res"] = legs["h3_d"].apply(
        lambda x: convert_h3_to_resolution(x, gps_h3_res)
    )

    # GPS del día + madrugada del día siguiente (offset 24-27) para legs cross-medianoche.
    gps_dia = gps[gps["dia"] == dia].copy()
    if next_dia is not None:
        gps_next = gps[(gps["dia"] == next_dia) & (gps["hora"] <= 3)].copy()
        if len(gps_next) > 0:
            gps_next["hora"] = gps_next["hora"] + 24
            gps_dia = pd.concat([gps_dia, gps_next], ignore_index=True)

    etapas_result_list = _process_dia(
        dia, legs[legs["dia"] == dia].copy(), gps_dia, matriz
    )

    if len(etapas_result_list) == 0:
        # sin destinos GPS imputados ese día: las etapas conservan distance_od (route NULL)
        travel_times = legs_all.reindex(
            columns=["dia", "id", "id_tarjeta", "id_viaje", "id_etapa", "distance_od"]
        ).copy()
        travel_times_trips = travel_times.groupby(
            ["dia", "id_tarjeta", "id_viaje"], as_index=False
        )[["distance_od"]].sum(min_count=1)
        return travel_times, travel_times_trips, None

    etapas_result = pd.concat(etapas_result_list, ignore_index=True)
    legs_to_gps_d = etapas_result.reindex(columns=["dia", "id_legs", "id_gps"])

    # ── distance_route y distance_route_gps: ambas desde GPS ──
    # legs_to_gps_o llega como parámetro (leído en el main por _fetch_time_distance_inputs_dia)
    legs_to_gps_d_dist = etapas_result.reindex(
        columns=["id_legs", "id_gps"]
    ).rename(columns={"id_gps": "id_gps_d"})
    gps_anchors = legs_to_gps_o.merge(legs_to_gps_d_dist, on="id_legs")

    gps_ranked = gps.reindex(
        columns=[
            "id",
            "dia",
            "id_linea",
            "id_ramal",
            "interno",
            "distance_km",
            "distance_servicio_mts",
        ]
    ).copy()
    del gps
    gc.collect()

    gps_ranked["distance_km"] = pd.to_numeric(
        gps_ranked["distance_km"], errors="coerce"
    )
    gps_ranked["distance_servicio_mts"] = pd.to_numeric(
        gps_ranked["distance_servicio_mts"], errors="coerce"
    )

    # Acumulada por servicio (dia × linea × ramal × interno); el gps ya viene por fecha.
    gps_ranked["acum_km"] = gps_ranked.groupby(
        ["dia", "id_linea", "id_ramal", "interno"]
    )["distance_km"].cumsum()
    dist_mts = gps_ranked["distance_servicio_mts"]
    group_keys = [
        gps_ranked["dia"],
        gps_ranked["id_linea"],
        gps_ranked["id_ramal"],
        gps_ranked["interno"],
    ]
    gps_ranked["acum_mts"] = dist_mts.fillna(0).groupby(group_keys).cumsum()
    gps_ranked["acum_mts_nan_count"] = (
        dist_mts.isna().astype(int).groupby(group_keys).cumsum()
    )

    acum_km_map = gps_ranked.set_index("id")["acum_km"]
    acum_mts_map = gps_ranked.set_index("id")["acum_mts"]
    acum_nan_map = gps_ranked.set_index("id")["acum_mts_nan_count"]

    gps_anchors["acum_km_o"] = gps_anchors["id_gps_o"].map(acum_km_map)
    gps_anchors["acum_km_d"] = gps_anchors["id_gps_d"].map(acum_km_map)
    gps_anchors["acum_mts_o"] = gps_anchors["id_gps_o"].map(acum_mts_map)
    gps_anchors["acum_mts_d"] = gps_anchors["id_gps_d"].map(acum_mts_map)
    gps_anchors["nan_o"] = gps_anchors["id_gps_o"].map(acum_nan_map)
    gps_anchors["nan_d"] = gps_anchors["id_gps_d"].map(acum_nan_map)

    del gps_ranked, acum_km_map, acum_mts_map, acum_nan_map
    gc.collect()

    gps_distances = gps_anchors.reindex(columns=["id_legs"]).copy()
    gps_distances["distance_route"] = (
        gps_anchors["acum_km_d"].values - gps_anchors["acum_km_o"].values
    )
    nan_entre_anclas = gps_anchors["nan_d"].values - gps_anchors["nan_o"].values
    diff_mts = gps_anchors["acum_mts_d"].values - gps_anchors["acum_mts_o"].values
    gps_distances["distance_route_gps"] = np.where(
        nan_entre_anclas > 0,
        np.nan,
        diff_mts / 1000,
    )
    gps_distances = gps_distances.rename(columns={"id_legs": "id"})
    del gps_anchors
    gc.collect()

    # Distancia recorrida negativa (anclas fuera de orden / cross-medianoche) → NaN.
    for _col in ("distance_route", "distance_route_gps"):
        gps_distances.loc[gps_distances[_col] < 0, _col] = np.nan

    travel_times = legs.reindex(columns=["dia", "id", "fecha", "distance_od"]).merge(
        etapas_result.reindex(columns=["id_legs", "fecha_gps"]),
        how="left",
        left_on=["id"],
        right_on=["id_legs"],
    )
    del legs, etapas_result
    gc.collect()

    travel_times["travel_time_min"] = round(
        (travel_times["fecha_gps"] - travel_times["fecha"]).dt.total_seconds() / 60,
        1,
    )
    travel_times = travel_times.loc[travel_times.travel_time_min > 0, :]
    travel_times["kmh_od"] = (
        travel_times["distance_od"] / (travel_times["travel_time_min"] / 60)
    ).round(1)
    travel_times.loc[
        (travel_times.kmh_od == np.inf) | (travel_times.kmh_od >= VELOCIDAD_MAXIMA_KMH),
        "kmh_od",
    ] = np.nan

    travel_times = travel_times.merge(gps_distances, on="id", how="left")

    tot_gps = len(travel_times)
    tot_gps_asig = travel_times.travel_time_min.notna().sum()
    logger.info("GPS imputado (%s): %.1f%%", dia, tot_gps_asig / max(tot_gps, 1) * 100)

    travel_times["kmh_route"] = (
        travel_times["distance_route"] / (travel_times["travel_time_min"] / 60)
    ).round(1)
    travel_times.loc[
        (travel_times.kmh_route == np.inf)
        | (travel_times.kmh_route >= VELOCIDAD_MAXIMA_KMH),
        "kmh_route",
    ] = np.nan
    travel_times["kmh_route_gps"] = (
        travel_times["distance_route_gps"] / (travel_times["travel_time_min"] / 60)
    ).round(1)
    travel_times.loc[
        (travel_times.kmh_route_gps == np.inf)
        | (travel_times.kmh_route_gps >= VELOCIDAD_MAXIMA_KMH),
        "kmh_route_gps",
    ] = np.nan

    travel_times = travel_times.reindex(
        columns=[
            "dia",
            "id",
            "travel_time_min",
            "distance_od",
            "distance_route",
            "distance_route_gps",
            "kmh_od",
            "kmh_route",
            "kmh_route_gps",
        ]
    )

    # Merge explicito por (dia, id): distance_od de legs_all es la autoritativa
    # (incluye las etapas sin GPS, con route NULL).
    travel_times = legs_all[
        ["dia", "id", "id_tarjeta", "id_viaje", "id_etapa", "distance_od"]
    ].merge(
        travel_times.drop(columns=["distance_od"]),
        on=["dia", "id"],
        how="left",
    )
    travel_times = travel_times.merge(
        legs_to_gps_o.rename(columns={"id_legs": "id"}),
        on="id",
        how="left",
    )
    travel_times = travel_times.merge(
        legs_to_gps_d.reindex(columns=["id_legs", "id_gps"]).rename(
            columns={"id_legs": "id", "id_gps": "id_gps_d"}
        ),
        on="id",
        how="left",
    )

    travel_times_trips = travel_times.groupby(
        ["dia", "id_tarjeta", "id_viaje"], as_index=False
    )[["travel_time_min", "distance_od", "distance_route", "distance_route_gps"]].sum(
        min_count=1
    )
    travel_times_trips["kmh_od"] = (
        travel_times_trips["distance_od"] / (travel_times_trips["travel_time_min"] / 60)
    ).round(1)
    travel_times_trips["kmh_route"] = (
        travel_times_trips["distance_route"]
        / (travel_times_trips["travel_time_min"] / 60)
    ).round(1)
    travel_times_trips["kmh_route_gps"] = (
        travel_times_trips["distance_route_gps"]
        / (travel_times_trips["travel_time_min"] / 60)
    ).round(1)
    for col in ["kmh_od", "kmh_route", "kmh_route_gps"]:
        travel_times_trips.loc[
            (travel_times_trips[col] == np.inf)
            | (travel_times_trips[col] >= VELOCIDAD_MAXIMA_KMH),
            col,
        ] = np.nan

    return travel_times, travel_times_trips, legs_to_gps_d


def _duckdb_memory_limit_gb() -> float:
    """GB que DuckDB tiene reservados (su memory_limit resuelto), como proxy del pico
    del proceso main durante la fase paralela.

    Espeja la prioridad de `_resolve_memory_limit` (override en tuning.yaml → 25% de
    la RAM) SIN llamarla, para no re-emitir su log INFO en cada corrida. Se usa como
    reserva porque en el punto donde se decide el paralelismo `compute_od_distances`
    todavía no corrió: DuckDB puede crecer hasta su límite y el working set en pandas
    del main vive por encima. Se autoescala con la máquina.
    """
    try:
        from urbantrips.utils.utils import leer_configs_tuning
        val = (leer_configs_tuning().get("duckdb", {}) or {}).get("memory_limit")
    except Exception:
        val = None
    if val:
        s = str(val).strip().upper()
        try:
            num = float("".join(c for c in s if c.isdigit() or c == "."))
            return num / 1024.0 if ("MB" in s or "MIB" in s) else num
        except Exception:
            pass
    try:
        import psutil
        return max(psutil.virtual_memory().total / 2**30 * 0.25, 1.0)
    except Exception:
        return 8.0


def _parallel_day_workers(n_days: int) -> int:
    """Cantidad de días del enrichment a procesar en paralelo.

    Override manual: clave `parallel_day_workers` en configs/tuning.yaml (1 fuerza
    serial, útil para comparar). Sin override, autotune por RAM:

      workers = (RAM_libre_ahora − reserva_main) / (~12 GB por día en vuelo)

    `RAM_libre_ahora` (psutil.available) ya refleja OS, apps/IDE y lo que el main
    lleva cargado — por eso es mejor que la RAM total con un factor fijo. `reserva_main`
    = el memory_limit de DuckDB (_duckdb_memory_limit_gb): aproxima el pico que el main
    todavía va a alcanzar (buffers de DuckDB hasta su límite + pandas de compute_od).
    Tope de 3 (los saves son seriales en el main, más workers no rinde).

    Historia: el colchón fijo de 8 GB subestimaba el main real (~21 GB con memory_limit
    17 GB) y sobre-committeaba; con 2 workers la máquina quedaba en ~5 GB libres
    thrasheando (ver corrida 2026-07-17).
    """
    try:
        import yaml
        from urbantrips.utils.paths import get_paths

        tuning = get_paths().base / "configs" / "tuning.yaml"
        if tuning.exists():
            override = (yaml.safe_load(tuning.read_text()) or {}).get(
                "parallel_day_workers"
            )
            if override:
                return max(1, min(int(override), n_days))
    except Exception as e:
        logger.debug("[parallel_day_workers] override ilegible: %s", e)
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 2**30
    except Exception:
        return 1
    reserve_gb = _duckdb_memory_limit_gb()
    per_day_gb = 12.0
    budget_gb = avail_gb - reserve_gb
    workers = int(max(0.0, budget_gb) // per_day_gb)
    n = max(1, min(3, workers, n_days))
    logger.info(
        "[parallel_day_workers] autotune: %d worker(s) — RAM libre %.1f GB "
        "− reserva main %.1f GB = %.1f GB presupuesto / %.0f GB por día (tope 3)",
        n, avail_gb, reserve_gb, max(0.0, budget_gb), per_day_gb,
    )
    return n


def _fetch_legs_all_dia(ctx, dia):
    """Lee (main) las etapas validadas del día y les computa distance_od.

    compute_od_distances queda en el MAIN a propósito: usa un cache DuckDB propio
    (od_distances) de un solo escritor; llamarlo desde varios workers a la vez
    rompería el file-lock del cache.
    """
    legs_all = ctx.data.query(
        f"""
        SELECT e.*
        FROM etapas e
        WHERE e.etapa_validada = 1 AND e.dia = '{dia}'
        ORDER BY e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa, e.id_linea, e.id_ramal, e.interno
        """
    )
    if len(legs_all) == 0:
        return None
    return compute_od_distances(
        od_df=legs_all,
        origin_col="h3_o",
        dest_col="h3_d",
        distance_col="distance_od",
        unit="km",
        symmetric=False,
        precompute_dist=50_000,
        max_tile_deg=99,
        verbose=True,
        ctx=ctx,
    )


def _travel_times_sin_gps(legs_all):
    """Camino sin tabla gps: solo distance_od por etapa y suma por viaje."""
    travel_times = legs_all.reindex(
        columns=["dia", "id", "id_tarjeta", "id_viaje", "id_etapa", "distance_od"]
    ).copy()
    travel_times_trips = (
        travel_times.groupby(["dia", "id_tarjeta", "id_viaje"], as_index=False)
        [["distance_od"]].sum(min_count=1)
    )
    return travel_times, travel_times_trips


def _save_travel_times_dia(ctx, dia, travel_times, travel_times_trips, legs_to_gps_d):
    """Normaliza esquemas y appendea (main) las salidas de un día."""
    travel_times = travel_times.reindex(
        columns=["dia", "id", "id_tarjeta", "id_viaje", "id_etapa", "travel_time_min",
                 "distance_od", "distance_route", "distance_route_gps",
                 "kmh_od", "kmh_route", "kmh_route_gps",
                 "id_gps_o", "id_gps_d"]
    )
    travel_times_trips = travel_times_trips.reindex(
        columns=["dia", "id_tarjeta", "id_viaje", "travel_time_min",
                 "distance_od", "distance_route", "distance_route_gps",
                 "kmh_od", "kmh_route", "kmh_route_gps"]
    )
    travel_times["distance_route_gps"] = travel_times["distance_route_gps"].round(2)

    tot = len(travel_times)
    pct = travel_times["travel_time_min"].notna().sum() / max(tot, 1) * 100
    logger.info(
        "[assign_time_distances] %s guardado: %d etapas, travel_time imputado %.1f%%",
        dia, tot, pct,
    )

    if legs_to_gps_d is not None:
        ctx.data.append_raw(legs_to_gps_d, "legs_to_gps_destination")
    ctx.data.append_raw(travel_times, "travel_times_legs")
    ctx.data.append_raw(travel_times_trips, "travel_times_trips")


def assign_time_distances(ctx: StorageContext):
    """
    Lee las etapas DIA POR DIA y, si hay tabla gps, imputa el gps de destino y
    calcula distance_od / distance_route / distance_route_gps / travel_time_min
    por dia, guardando y liberando antes del siguiente. Acota la RAM (escala a un
    mes) preservando exactamente la logica intra-dia (ver _gps_destino_y_tiempos_dia).

    Con RAM suficiente, el cómputo pesado de cada día corre en workers
    (_parallel_day_workers); las lecturas/escrituras a la DB quedan en el main.
    """
    configs = leer_configs_generales(autogenerado=False)
    usa_gps = configs.get("usa_archivo_gps", False)

    dias_ultima_corrida = ctx.data.get_run_days()
    dias = sorted(dias_ultima_corrida["dia"].tolist())
    dia_to_next = {dias[i]: dias[i + 1] for i in range(len(dias) - 1)}

    # Inputs compartidos (no dependen del dia): se cargan una sola vez.
    metadata_lineas = matriz = modos_ramal = legs_h3_res = None
    if usa_gps:
        legs_h3_res = configs["resolucion_h3"]
        modos_ramal = modos_con_ramal(configs)
        metadata_lineas = ctx.insumos.get_metadata_lineas()[
            ["id_linea", "id_linea_agg", "modo"]
        ]
        mv = ctx.insumos.get_matrix_validation()
        matriz = mv[
            ["id_linea_agg", "id_ramal", "parada", "area_influencia"]
        ].drop_duplicates()
        matriz["id_ramal"] = matriz["id_ramal"].fillna(RAMAL_SENTINEL).astype("int64")
        matriz["ring"] = matriz.apply(
            lambda row: h3.grid_distance(row.parada, row.area_influencia), axis=1
        )
        lado_m = h3.average_hexagon_edge_length(res=legs_h3_res, unit="m")
        ring_max = max(
            1, round(configs.get("tolerancia_destino_gps", 1000) / (lado_m * 2))
        )
        matriz = matriz[matriz.ring <= ring_max]

    # Limpiar las salidas de los dias de ESTA corrida una sola vez (un solo scan;
    # travel_times_* / legs_to_gps_destination no tienen indices). Luego se appendea
    # por dia. Solo se borran los dias de dias_ultima_corrida (semantica incremental).
    dias_str = ", ".join(f"'{d}'" for d in dias)
    for table in ("travel_times_legs", "travel_times_trips", "legs_to_gps_destination"):
        try:
            ctx.data.execute(f"DELETE FROM {table} WHERE dia IN ({dias_str})")
        except Exception as e:
            logger.debug("[delete omitido] %s: %s", table, e)

    n_workers = _parallel_day_workers(len(dias)) if usa_gps else 1

    if n_workers <= 1:
        # ── Camino serial (comportamiento previo, mismos resultados) ──
        for i, dia in enumerate(dias, 1):
            logger.info("[assign_time_distances] día %d/%d (%s)", i, len(dias), dia)
            legs_all = _fetch_legs_all_dia(ctx, dia)
            if legs_all is None:
                continue

            if usa_gps:
                gps, legs_to_gps_o = _fetch_time_distance_inputs_dia(
                    ctx, dia, dia_to_next.get(dia)
                )
                travel_times, travel_times_trips, legs_to_gps_d = (
                    _gps_destino_y_tiempos_dia(
                        dia, dia_to_next.get(dia), legs_all, gps, legs_to_gps_o,
                        metadata_lineas, matriz, modos_ramal, legs_h3_res,
                    )
                )
                del gps, legs_to_gps_o
            else:
                travel_times, travel_times_trips = _travel_times_sin_gps(legs_all)
                legs_to_gps_d = None

            _save_travel_times_dia(
                ctx, dia, travel_times, travel_times_trips, legs_to_gps_d
            )
            del legs_all, travel_times, travel_times_trips
            gc.collect()
        return

    # ── Camino paralelo: el cómputo pesado de cada día (_gps_destino_y_tiempos_dia,
    # pandas puro) corre en worker processes; el main hace TODAS las lecturas y
    # escrituras a DuckDB y compute_od_distances (cache de un solo escritor).
    # Chunks de n_workers días con barrera por chunk para acotar la RAM en vuelo
    # (mismo patrón que la Fase 2 de create_legs). El resultado por día es idéntico
    # al serial; solo cambia el orden físico de inserción entre días.
    logger.info("[assign_time_distances] paralelizando: %d días en vuelo", n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, len(dias), n_workers):
            chunk = dias[chunk_start: chunk_start + n_workers]
            futures = {}
            for offset, dia in enumerate(chunk, 1):
                logger.info(
                    "[assign_time_distances] día %d/%d (%s) — leyendo insumos",
                    chunk_start + offset, len(dias), dia,
                )
                legs_all = _fetch_legs_all_dia(ctx, dia)
                if legs_all is None:
                    continue
                gps, legs_to_gps_o = _fetch_time_distance_inputs_dia(
                    ctx, dia, dia_to_next.get(dia)
                )
                futures[executor.submit(
                    _gps_destino_y_tiempos_dia,
                    dia, dia_to_next.get(dia), legs_all, gps, legs_to_gps_o,
                    metadata_lineas, matriz, modos_ramal, legs_h3_res,
                )] = dia
                del legs_all, gps, legs_to_gps_o

            for future in as_completed(futures):
                dia = futures[future]
                travel_times, travel_times_trips, legs_to_gps_d = future.result()
                _save_travel_times_dia(
                    ctx, dia, travel_times, travel_times_trips, legs_to_gps_d
                )
                del travel_times, travel_times_trips, legs_to_gps_d
            gc.collect()


def _process_dia(dia, legs_dia, gps_dia, matriz):
    """Process all hours for one day; called in a subprocess (own memory space)."""
    results = []
    for hora in sorted(legs_dia["hora"].unique()):
        result = _process_dia_hora(dia, hora, legs_dia, gps_dia, matriz)
        if result is not None:
            results.append(result)
    return results


def _process_dia_hora(dia, hora, legs, gps, matriz):
    """Process one (dia, hora) slice for GPS destination imputation."""
    etapas_tx = legs.loc[
        (legs["hora"] == hora) & (legs["dia"] == dia),
        [
            "dia",
            "id",
            "id_linea",
            "id_linea_agg",
            "id_ramal",
            "interno",
            "h3_o",
            "h3_d",
            "h3_d_gps_res",
            "distance_od",
            "fecha",
        ],
    ].copy()

    if len(etapas_tx) == 0:
        return None

    # Merge legs<->matriz acotado por id_linea_agg (red agregada) + id_ramal
    # efectivo (ambos lados lo traen ya resuelto). Derivar la clave de las columnas
    # de `matriz` evita pasar modos_ramal a traves del limite de multiprocessing.
    clave = [c for c in ("id_linea_agg", "id_ramal") if c in matriz.columns]
    etapas_tx = etapas_tx.merge(
        matriz, how="left", left_on=clave + ["h3_d"], right_on=clave + ["parada"]
    )

    hora_filtro = [hora + i for i in range(0, 4)]
    gps_tx = (
        gps.loc[gps["hora"].isin(hora_filtro)]
        .reindex(
            columns=[
                "id",
                "id_linea",
                "id_ramal",
                "interno",
                "h3_legs_res",
                "h3",
                "fecha_gps",
            ]
        )
        .rename(columns={"h3_legs_res": "area_influencia"})
    )

    # Pre-filter GPS to only timestamps after the earliest boarding per vehicle.
    # id_ramal is excluded because it can be None/NaN, causing dtype mismatches
    # in the merge; ramal filtering is handled by the main join below.
    min_fecha_vehicle = (
        etapas_tx.groupby(["id_linea", "interno"])["fecha"]
        .min()
        .reset_index()
        .rename(columns={"fecha": "min_fecha_leg"})
    )
    gps_tx = gps_tx.merge(min_fecha_vehicle, on=["id_linea", "interno"], how="inner")
    gps_tx = gps_tx.loc[gps_tx["fecha_gps"] > gps_tx["min_fecha_leg"]].drop(
        columns=["min_fecha_leg"]
    )

    if len(gps_tx) == 0:
        return None

    etapas_tx = etapas_tx.merge(
        gps_tx,
        how="inner",
        on=["id_linea", "id_ramal", "interno", "area_influencia"],
        suffixes=("_legs", "_gps"),
    )

    if len(etapas_tx) == 0:
        return None

    etapas_tx["fecha_dif"] = (
        etapas_tx["fecha_gps"] - etapas_tx["fecha"]
    ).dt.total_seconds() / 60
    etapas_tx = etapas_tx.loc[etapas_tx.fecha_dif > 0]

    if len(etapas_tx) == 0:
        return None

    etapas_tx["min_fecha_d"] = etapas_tx.groupby(["id_legs"]).fecha_gps.transform("min")
    etapas_tx["min_fecha_d"] = round(
        (etapas_tx.fecha_gps - etapas_tx["min_fecha_d"]).dt.total_seconds() / 60,
        1,
    )
    etapas_tx = etapas_tx.loc[etapas_tx.min_fecha_d < 20]

    if len(etapas_tx) == 0:
        return None

    h3_pairs = etapas_tx[["h3_d_gps_res", "h3"]].drop_duplicates().copy()
    h3_pairs["distancia_h3"] = [
        h3.grid_distance(r["h3_d_gps_res"], r["h3"])
        for r in h3_pairs.to_dict("records")
    ]
    etapas_tx = etapas_tx.merge(h3_pairs, on=["h3_d_gps_res", "h3"], how="left")

    etapas_tx = etapas_tx.sort_values(
        ["id_legs", "ring", "distancia_h3", "min_fecha_d"]
    )
    return etapas_tx.groupby("id_legs", as_index=False).first()


def distancia_h3_gps_leg(row):
    return h3.grid_distance(row["h3_d_gps_res"], row["h3"])


@duracion
def assign_stations_od(ctx: StorageContext):
    """
    This function reads legs, classifies OD into stations,
    reads travel times in gps and computes a single travel time
    for each leg
    """

    configs = leer_configs_generales(autogenerado=False)
    tiempos_viaje_estaciones = configs.get("tiempos_viaje_estaciones")

    if tiempos_viaje_estaciones is not None:

        # Si no hay datos de tiempos entre estaciones (p.ej. no existe el archivo
        # travel_time_stations.csv), no hay nada que clasificar: salir antes del
        # LEFT JOIN caro sobre etapas (63M filas, ~6 min) que igual daría 0 etapas
        # asignadas a estaciones.
        try:
            travel_times_stations = ctx.insumos.get_travel_times_stations()
        except Exception:
            travel_times_stations = None
        if travel_times_stations is None or len(travel_times_stations) == 0:
            logger.info(
                "No hay datos de tiempos de viaje entre estaciones "
                "(travel_time_stations vacío); se omite assign_stations_od."
            )
            return

        # Insumos estáticos (no dependen del día): se construyen UNA sola vez fuera
        # del loop. Antes esto estaba después de leer etapas del mes entero.
        epsg_m = get_epsg_m()

        stations_o = (
            travel_times_stations.reindex(
                columns=["id_o", "id_linea_o", "id_ramal_o", "lat_o", "lon_o"]
            )
            .drop_duplicates()
            .rename(
                columns={
                    "id_o": "id",
                    "lat_o": "lat",
                    "lon_o": "lon",
                    "id_linea_o": "id_linea",
                    "id_ramal_o": "id_ramal",
                }
            )
        )

        stations_d = (
            travel_times_stations.reindex(
                columns=["id_d", "id_linea_d", "id_ramal_d", "lat_d", "lon_d"]
            )
            .drop_duplicates()
            .rename(
                columns={
                    "id_d": "id",
                    "lat_d": "lat",
                    "lon_d": "lon",
                    "id_linea_d": "id_linea",
                    "id_ramal_d": "id_ramal",
                }
            )
        )

        stations = (
            pd.concat([stations_o, stations_d]).drop_duplicates().reset_index(drop=True)
        )

        geom = gpd.GeoSeries.from_xy(x=stations.lon, y=stations.lat, crs=4326)
        stations = gpd.GeoDataFrame(stations, geometry=geom, crs=4326).reindex(
            columns=["id", "id_linea", "geometry"]
        )

        stations = stations.to_crs(epsg=epsg_m)

        # OPTIMIZACIÓN (bit-idéntica): classify_leg_into_station filtra las estaciones por
        # id_linea (geo.py:490), así que una etapa de una línea SIN estaciones clasifica a 0
        # (stations queda vacío para ese grupo). Filtrar la query de etapas a las líneas que
        # sí tienen estaciones evita la conversión h3→latlng + join geométrico sobre el resto
        # (en AMBA el CSV cubre pocas líneas → ~99% de las 177M etapas se salteaban a 0 igual).
        # El resultado (legs_to_station_*, travel_times_stations de salida) es idéntico.
        station_lines = [int(x) for x in stations["id_linea"].dropna().unique().tolist()]
        if not station_lines:
            logger.info(
                "[assign_stations_od] las estaciones no tienen id_linea válido → nada que "
                "clasificar; se omite."
            )
            return
        station_lines_str = ", ".join(str(x) for x in station_lines)
        logger.info(
            "[assign_stations_od] %d línea(s) con estaciones → se clasifican solo esas etapas",
            len(station_lines),
        )

        # Se procesa UN DÍA POR VEZ para acotar RAM: antes se levantaba etapas del mes
        # entero (~63M filas) al DataFrame `legs`. Cada etapa se clasifica por sus
        # h3_o/h3_d contra `stations` (insumo estático) y los tiempos salen de lookups
        # O→D estáticos → sin dependencia inter-día, resultado idéntico por día.
        # Los días se derivan de todas las etapas presentes (la query original NO
        # filtraba por dias_ultima_corrida), no de get_run_days.
        dias = sorted(ctx.data.query("SELECT DISTINCT dia FROM etapas")["dia"].tolist())
        dias_str = ", ".join(f"'{d}'" for d in dias)
        for _tabla in (
            "legs_to_station_origin",
            "legs_to_station_destination",
            "travel_times_stations",
        ):
            if dias_str:
                try:
                    ctx.data.execute(f"DELETE FROM {_tabla} WHERE dia IN ({dias_str})")
                except Exception as e:
                    logger.debug(
                        "[assign_stations_od] DELETE omitido en %s: %s", _tabla, e
                    )

        for i, dia in enumerate(dias, 1):
            logger.info("[assign_stations_od] día %d/%d (%s)", i, len(dias), dia)
            # read legs without travel time in gps and distances (un día)
            legs = ctx.data.query(
                f"""
                SELECT e.dia, e.id, e.id_linea, e.id_ramal, e.h3_o, e.h3_d
                FROM etapas e
                LEFT JOIN travel_times_gps tt
                ON e.dia = tt.dia AND e.id = tt.id
                WHERE tt.id IS NULL
                AND e.etapa_validada = 1
                AND e.dia = '{dia}'
                AND e.id_linea IN ({station_lines_str})
                """
            )

            if len(legs) == 0:
                del legs
                continue

            # classify legs' origin and destination with station id
            legs_with_origin_station = (
                legs.groupby(["id_linea"])
                .apply(
                    classify_leg_into_station,
                    stations=stations,
                    leg_h3_field="h3_o",
                    join_branch_id=False,
                )
                .reset_index(drop=True)
                .rename(columns={"id_station": "id_station_o"})
            )

            logger.info(
                "Día %s — etapas clasificadas en estaciones de origen: %.1f%%",
                dia,
                len(legs_with_origin_station) / len(legs) * 100,
            )

            legs_with_destination_station = (
                legs.groupby(["id_linea"])
                .apply(
                    classify_leg_into_station,
                    stations=stations,
                    leg_h3_field="h3_d",
                    join_branch_id=False,
                )
                .reset_index(drop=True)
                .rename(columns={"id_station": "id_station_d"})
            )

            logger.info(
                "Día %s — etapas clasificadas en estaciones de destino: %.1f%%",
                dia,
                len(legs_with_destination_station) / len(legs) * 100,
            )

            # upload od station into db
            stations_o = legs_with_origin_station.rename(
                columns={"id_station_o": "id_station"}
            ).reindex(columns=["dia", "id_legs", "id_station"])

            stations_d = legs_with_destination_station.rename(
                columns={"id_station_d": "id_station"}
            ).reindex(columns=["dia", "id_legs", "id_station"])

            ctx.data.append_raw(stations_o, "legs_to_station_origin")
            ctx.data.append_raw(stations_d, "legs_to_station_destination")

            del stations_o
            del stations_d

            # add stations to legs data
            travel_times = (
                legs.reindex(columns=["dia", "id", "id_linea", "distance_od"])
                .merge(
                    legs_with_origin_station,
                    left_on=["id"],
                    right_on=["id_legs"],
                    how="left",
                )
                .merge(
                    legs_with_destination_station,
                    left_on=["id"],
                    right_on=["id_legs"],
                    how="left",
                )
                .drop(["id_legs_x", "id_legs_y", "dia_x", "dia_y"], axis=1)
                .dropna(subset=["id_station_o", "id_station_d"])
            )

            if len(travel_times) == 0:
                logger.info("Día %s — no hay etapas con estaciones OD asignadas.", dia)
            else:
                logger.info(
                    "Día %s — etapas clasificadas en la misma estación OD: %.1f%%",
                    dia,
                    len(travel_times[travel_times.id_station_o == travel_times.id_station_d])
                    / len(travel_times) * 100,
                )

            travel_times = travel_times.loc[
                travel_times.id_station_o != travel_times.id_station_d, :
            ]

            # compute travel time
            travel_times = travel_times.merge(
                travel_times_stations.reindex(columns=["id_o", "id_d", "travel_time_min"]),
                left_on=["id_station_o", "id_station_d"],
                right_on=["id_o", "id_d"],
                how="left",
            )

            if len(travel_times) > 0:
                logger.info(
                    "Día %s — sin tiempos de viaje: %.1f%%",
                    dia,
                    travel_times.travel_time_min.isna().sum() / len(travel_times) * 100,
                )
            travel_times = travel_times.dropna(subset=["travel_time_min"])
            travel_times.loc[:, "kmh_od"] = (
                travel_times.loc[:, "distance_od"]
                / (travel_times.loc[:, "travel_time_min"] / 60)
            ).round(1)

            travel_times.loc[
                (travel_times.kmh_od == np.inf) | (travel_times.kmh_od >= VELOCIDAD_MAXIMA_KMH),
                "kmh_od",
            ] = np.nan

            # upload to db
            travel_times = travel_times.reindex(
                columns=["dia", "id", "travel_time_min", "kmh_od"]
            )

            travel_times = travel_times.reindex(
                columns=["dia", "id", "travel_time_min", "travel_speed"]
            )

            ctx.data.append_raw(travel_times, "travel_times_stations")

            del legs, legs_with_origin_station, legs_with_destination_station, travel_times
            gc.collect()

# NOTA: la antigua `add_distance_and_travel_time` (etapas) fue ELIMINADA
# (2026-07-17). Estaba MUERTA desde el commit ebe0789 (2026-05-31), reemplazada
# por `assign_time_distances`, que calcula distance_od/route/route_gps y
# travel_time_min y los guarda en travel_times_legs/travel_times_trips (de donde
# los leen kpi, chains y el dashboard). Nadie consumía etapas.distancia.

    
