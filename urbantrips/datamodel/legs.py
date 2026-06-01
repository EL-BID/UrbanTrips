import logging
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
def create_legs_from_transactions(ctx: StorageContext, trx_order_params, batch: BatchSpec | None = None):
    """
    Esta function toma las transacciones de la db
    las estructura en etapas con sus id y id viaje
    y crea la tabla etapas en la db
    """
    legs, tarjetas_duplicadas = build_legs_from_transactions(ctx, trx_order_params, batch)
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the legs dataframe from transactions and return duplicate-card metadata.
    """
    legs = trx[trx.dia.isin(dias_ultima_corrida.dia)]

    # parse dates using local timezone
    legs = legs.copy()
    legs["fecha"] = pd.to_datetime(legs.fecha, unit="s", errors="coerce")

    # asignar id h3
    configs = leer_configs_generales(autogenerado=False)
    res = configs["resolucion_h3"]
    legs = referenciar_h3(df=legs, res=res, nombre_h3="h3_o")

    # crear columna delta
    if trx_order_params["criterio"] == "orden_trx":
        legs["delta"] = None
    elif trx_order_params["criterio"] == "fecha_completa":
        legs = crear_delta_trx(legs)
    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    # asignar nuevo id tarjeta trx simultaneas
    legs, tarjetas_duplicadas = _change_card_id_for_concurrent_trx(legs, trx_order_params)

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


def change_card_id_for_concurrent_trx(trx, trx_order_params, dias_ultima_corrida, ctx: StorageContext):
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
    trx_c, tarjetas_duplicadas = _change_card_id_for_concurrent_trx(trx, trx_order_params)
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
    trx["_is_start"] = (trx.diff_datetime2.isna() | (trx.diff_datetime2 > diff_segundos)).astype(int)
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

    trx = trx.drop("id_tarjeta", axis=1).rename(
        columns={"id_tarjeta_nuevo": "id_tarjeta"}
    )

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

    Parameters
    ----------
    trx : pandas DataFrame
        transactions data

    ventana_viajes : int
        time window in minutes to consider transactions as part of the
        same trip

    Returns
    ----------

    X: pandas DataFrame
        legs with new trips and legs ids

    """

    # turn into seconds
    ventana_viajes = ventana_viajes * 60

    trx = trx.sort_values(["id_tarjeta", "fecha"])

    # Calcular los id_viajes
    trx["id_viaje"] = (
        trx.groupby(["id_tarjeta"])["delta"]
        .transform(lambda s: _trip_ids_from_deltas(s.to_numpy(dtype=np.float64), ventana_viajes))
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


@duracion
def assign_gps_origin(ctx: StorageContext):
    """
    This function read legs data and if there is gps table
    assigns a gps to the leg origin
    """
    configs = leer_configs_generales(autogenerado=False)
    usa_gps = configs.get("usa_archivo_gps", False)
    nombre_archivo_gps = configs.get("nombre_archivo_gps") if usa_gps else None

    if nombre_archivo_gps is not None:
        legs = ctx.data.query(
            """
            SELECT e.dia, e.id_linea, e.id_ramal, e.interno, e.tiempo, e.id
            FROM etapas e
            JOIN dias_ultima_corrida d
            ON e.dia = d.dia
            """
        )
        legs["fecha"] = pd.to_datetime(legs["dia"] + " " + legs["tiempo"])

        gps = ctx.data.query(
            """
            SELECT g.dia, g.id_linea, g.id_ramal, g.interno, g.fecha, g.id
            FROM gps g
            JOIN dias_ultima_corrida d
            ON g.dia = d.dia
            """
        )
        if gps.empty or legs.empty:
            ctx.data.save_raw(
                pd.DataFrame(columns=["dia", "id_legs", "id_gps"]),
                "legs_to_gps_origin",
            )
            return

        gps["fecha"] = pd.to_datetime(gps["fecha"], unit="s")

        cols = ["dia", "id_linea", "id_ramal", "interno", "fecha", "id"]
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

        legs_to_gps_o = legs_to_gps_o.reindex(
            columns=["dia", "id_legs", "id_gps"]
        ).dropna()

        ctx.data.save_raw(legs_to_gps_o, "legs_to_gps_origin")


@duracion
def assign_time_distances(ctx: StorageContext):
    """
    This function read legs data and if there is gps table
    assigns a gps to the leg destination
    """

    configs = leer_configs_generales(autogenerado=False)
    usa_gps = configs.get("usa_archivo_gps", False)
    nombre_archivo_gps = configs.get("nombre_archivo_gps") if usa_gps else None

    query = """
    SELECT e.*
    FROM etapas e
    JOIN dias_ultima_corrida d
    ON e.dia = d.dia
    WHERE e.od_validado = 1
    ORDER BY e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa, e.id_linea, e.id_ramal, e.interno
    """        
    legs_all = ctx.data.query(query)
            
    legs_all = compute_od_distances(
        od_df             = legs_all,
        origin_col        = "h3_o",
        dest_col          = "h3_d",
        distance_col      = 'distance_od',
        unit              = 'km',
        db_path           = "data/matriz_distancia/matriz_distancia.duckdb",
        network_cache_dir = "data/matriz_distancia",
        symmetric         = False,
        precompute_dist   = 50_000,   
        max_tile_deg      = 99,      
        verbose           = True
    )

    if nombre_archivo_gps is not None:

        legs_h3_res = configs["resolucion_h3"]

        # read stops zone of influence
        mv = ctx.insumos.get_matrix_validation()
        matriz = mv[["parada", "area_influencia"]].drop_duplicates()
        matriz["ring"] = matriz.apply(
            lambda row: h3.grid_distance(row.parada, row.area_influencia), axis=1
        )
        matriz = matriz[matriz.ring < 3]

        gps = ctx.data.query(
            """
            SELECT g.* FROM gps g
            JOIN dias_ultima_corrida d
            ON g.dia = d.dia
            ORDER BY dia, id_linea, id_ramal, interno, fecha
            """
        )

        legs = legs_all[(legs_all.id_linea.isin( gps.id_linea.unique() )) ].copy()

        legs["fecha"] = pd.to_datetime(legs["dia"] + " " + legs["tiempo"])

        # get h3 res for gps
        gps_h3_res = h3.get_resolution(gps["h3"].sample().item())

        # geocode gps with same h3 res than legs
        gps = referenciar_h3(
            gps, res=legs_h3_res, nombre_h3="h3_legs_res", lat="latitud", lon="longitud"
        )
        gps["fecha_gps"] = gps.fecha.map(lambda ts: pd.Timestamp(ts, unit="s"))
        gps["hora"] = gps.fecha_gps.dt.hour

        # Geocode legs destination in the same h3 resolution than gps
        legs["h3_d_gps_res"] = legs["h3_d"].apply(
            lambda x: convert_h3_to_resolution(x, gps_h3_res)
        )

        # Lista para acumular resultados parciales
        etapas_result_list = []

        # IteraciÃ³n por cada hora y cada dia
        legs_days = legs.dia.unique()
        legs_hours = legs.hora.unique()

        legs_days.sort()
        legs_hours.sort()

        logger.info("Imputando GPS de destino")

        etapas_asignadas_total = 0
        for dia in legs_days:
            n_asignadas_dia = 0
            for hora in legs_hours:
                # t_iter_start = time.time()

                # Filtrar las etapas por la hora especÃ­fica y eliminar valores nulos en 'h3_d'
                etapas_tx = legs.loc[
                    (legs["hora"] == hora) & (legs["dia"] == dia),
                    [
                        "dia",
                        "id",
                        "id_linea",
                        "id_ramal",
                        "interno",
                        "h3_o",
                        "h3_d",
                        "h3_d_gps_res",
                        "distance_od",
                        "fecha",
                    ],
                ].copy()

                n_etapas_raw = len(etapas_tx)
                n_etapas_unicas = etapas_tx["id"].nunique()

                # Agregar anillos a las etapas
                etapas_tx = etapas_tx.merge(
                    matriz, how="left", left_on="h3_d", right_on="parada"
                )
                n_post_matriz = len(etapas_tx)
                factor_matriz = n_post_matriz / n_etapas_raw if n_etapas_raw > 0 else 0

                # Determinar horas consecutivas para el filtrado de datos GPS
                hora_filtro = [hora + i for i in range(0, 4)]
                gps_tx = gps.loc[gps["hora"].isin(hora_filtro), :].copy()

                # Renombrar y seleccionar columnas relevantes en los datos GPS
                gps_tx = gps_tx.reindex(
                    columns=[
                        "id",
                        "id_linea",
                        "id_ramal",
                        "interno",
                        "h3_legs_res",
                        "h3",
                        "fecha_gps",
                    ]
                ).rename(columns={"h3_legs_res": "area_influencia"})

                n_gps = len(gps_tx)

                # Join gps to legs destination rings dataframe by the same resolution (legs resolution)
                etapas_tx = etapas_tx.merge(
                    gps_tx,
                    how="inner",
                    on=["id_linea", "id_ramal", "interno", "area_influencia"],
                    suffixes=("_legs", "_gps"),
                )
                n_expanded = len(etapas_tx)

                # Calcular la diferencia de tiempo entre cada punto de gps y cada etapa
                etapas_tx["fecha_dif"] = (
                    etapas_tx["fecha_gps"] - etapas_tx["fecha"]
                ).dt.total_seconds() / 60

                # Filtrar por diferencia de fecha positiva y ordenar por id, anillo y fecha_dif
                n_pre_fecha = len(etapas_tx)
                etapas_tx = etapas_tx.loc[etapas_tx.fecha_dif > 0, :]
                n_post_fecha = len(etapas_tx)

                n_asignadas_iter = 0
                if len(etapas_tx) > 0:

                    # Calcular la distancia entre h3 del destino de la etapa y h3 del gps
                    gps_dict = etapas_tx.reindex(
                        columns=["h3_d_gps_res", "h3"]
                    ).to_dict("records")
                    etapas_tx.loc[:, ["distancia_h3"]] = list(
                        map(distancia_h3_gps_leg, gps_dict)
                    )

                    # Calcular el tiempo mÃ­nimo de destino por id
                    etapas_tx["min_fecha_d"] = etapas_tx.groupby(
                        ["id_legs"]
                    ).fecha_gps.transform("min")
                    etapas_tx["min_fecha_d"] = round(
                        (
                            etapas_tx.fecha_gps - etapas_tx["min_fecha_d"]
                        ).dt.total_seconds()
                        / 60,
                        1,
                    )

                    # Filtrar por tiempo mÃ­nimo de destino menor a 20 minutos y ordenar por distancia_h3
                    etapas_tx = etapas_tx.loc[etapas_tx.min_fecha_d < 20, :]
                    etapas_tx = etapas_tx.sort_values(
                        ["id_legs", "ring", "distancia_h3", "min_fecha_d"]
                    )

                    # Obtener la primera ocurrencia por id - elijo el gps que se encuentra mÃ¡s cerca del destino
                    etapas_tx = etapas_tx.groupby("id_legs", as_index=False).first()
                    n_asignadas_iter = len(etapas_tx)
                    n_asignadas_dia += n_asignadas_iter

                    # Agregar resultado a la lista
                    etapas_result_list.append(etapas_tx)

                # t_iter = time.time() - t_iter_start
                # iter_times.append(t_iter)
                # if n_etapas_unicas > 0:
                #     print(
                #         f"  dia={dia} hora={hora:02d} | "
                #         f"etapas={n_etapas_unicas:>7,} "
                #         f"post_matriz={n_post_matriz:>9,} (x{factor_matriz:.1f}) "
                #         f"gps={n_gps:>8,} "
                #         f"expanded={n_expanded:>9,} "
                #         f"post_fecha={n_post_fecha:>9,} "
                #         f"asignadas={n_asignadas_iter:>7,} ({round(n_asignadas_iter/n_etapas_unicas*100) if n_etapas_unicas else 0}%) "
                #         f"| {t_iter:.1f}s"
                #     )

            etapas_asignadas_total += n_asignadas_dia
            # t_dia = time.time() - t_dia_start
            legs_dia = legs[legs["dia"] == dia]["id"].nunique()
            # print(f"  â†’ DÃ­a {dia}: {n_asignadas_dia:,} asignadas de {legs_dia:,} etapas ({round(n_asignadas_dia/legs_dia*100) if legs_dia else 0}%) | {t_dia:.1f}s")

        # t_total = time.time() - t_total_start
        # legs_total_unicas = legs["id"].nunique()
        # print(f"\nLoop destino completado en {t_total:.1f}s ({t_total/60:.1f} min)")
        # print(f"  Total asignadas: {etapas_asignadas_total:,} de {legs_total_unicas:,} etapas ({round(etapas_asignadas_total/legs_total_unicas*100) if legs_total_unicas else 0}%)")
        # if iter_times:
        #     print(f"  Tiempo promedio por iteraciÃ³n: {sum(iter_times)/len(iter_times):.2f}s | MÃ¡s lenta: {max(iter_times):.2f}s")

        # Concatenar todos los resultados acumulados
        if len(etapas_result_list) == 0:
            ctx.data.save_raw(
                pd.DataFrame(columns=["dia", "id_legs", "id_gps"]),
                "legs_to_gps_destination",
            )
            ctx.data.save_raw(
                pd.DataFrame(columns=["dia", "id", "travel_time_min", "travel_speed"]),
                "travel_times_gps",
            )
            logger.info("No se encontraron destinos GPS para imputar")
            return

        etapas_result = pd.concat(etapas_result_list, ignore_index=True)

        legs_to_gps_d = etapas_result.reindex(columns=["dia", "id_legs", "id_gps"])
        ctx.data.save_raw(legs_to_gps_d, "legs_to_gps_destination")

        logger.info("Computando tiempos de viaje en GPS")
        
        # â”€â”€ distance_route_gps y distance_route: ambas desde GPS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        legs_to_gps_o = ctx.data.query(
            """
            SELECT lo.id_legs, lo.id_gps AS id_gps_o
            FROM legs_to_gps_origin lo
            JOIN dias_ultima_corrida d ON lo.dia = d.dia
            """
        )


        legs_to_gps_d_dist = etapas_result.reindex(
            columns=["id_legs", "id_gps"]
        ).rename(columns={"id_gps": "id_gps_d"})

        gps_anchors = legs_to_gps_o.merge(legs_to_gps_d_dist, on="id_legs")

        # ──────────────────────────────────────────────────────────────────
        # distance_route y distance_route_gps: distancia recorrida entre anclas GPS
        # ──────────────────────────────────────────────────────────────────
        #
        # Para cada etapa se calcula la distancia recorrida por el vehículo entre
        # el ping GPS de origen y el ping GPS de destino. Hay dos fuentes:
        #
        #   distance_route     → suma de distance_km          (calculado por urbantrips)
        #   distance_route_gps → suma de distance_servicio_mts (odómetro del operador)
        #
        # distance_servicio_mts puede no estar reportado:
        #   - Mendoza, AMBA cuando el operador no provee odómetro: todos None
        #   - Algunas líneas sí lo reportan y otras no
        #   - Una línea puede reportarlo de forma parcial (algunos pings sí, otros no)
        #
        # Estrategia: cumsum sobre el valor con None→0 para no contaminar la
        # acumulada, más cumsum del conteo de NaN. Una etapa queda con
        # distance_route_gps = NaN solo si entre sus anclas hubo algún ping
        # sin odómetro reportado. Si todos los tramos intermedios están
        # reportados, el valor se calcula correctamente.
        #
        # distance_route se computa siempre (distance_km es siempre numérico).
        #
        # Supuesto general: origen y destino pertenecen al mismo servicio
        # continuo del interno. Si acum_d < acum_o, indica error de asignación
        # upstream (anclas de servicios distintos) → distance_route negativa
        # → detectar en QA.

        gps_ranked = gps.reindex(
            columns=["id", "dia", "id_linea", "id_ramal", "interno",
                     "distance_km", "distance_servicio_mts"]
        ).copy()

        # Asegurar tipo numérico — la tabla gps puede traer distance_servicio_mts
        # como object (None literal) cuando el operador no reporta odómetro.
        # cumsum no soporta dtype object, hay que coercer a float antes.
        gps_ranked["distance_km"] = pd.to_numeric(
            gps_ranked["distance_km"], errors="coerce"
        )
        gps_ranked["distance_servicio_mts"] = pd.to_numeric(
            gps_ranked["distance_servicio_mts"], errors="coerce"
        )

        # Acumulada de distance_km por servicio (dia × linea × ramal × interno).
        # El GPS ya viene ordenado por fecha, por lo que cumsum respeta el orden temporal.
        gps_ranked["acum_km"] = gps_ranked.groupby(
            ["dia", "id_linea", "id_ramal", "interno"]
        )["distance_km"].cumsum()

        # Acumulada de distance_servicio_mts: dos cumsums separadas para tolerar
        # NaN sin contaminar tramos limpios.
        # - acum_mts: cumsum sobre el valor con NaN→0 (acumulada utilizable)
        # - acum_mts_nan_count: cumsum del indicador de NaN (permite invalidar
        #   solo las etapas cuyas anclas caen en tramos con algún None)
        dist_mts = gps_ranked["distance_servicio_mts"]
        group_keys = [
            gps_ranked["dia"], gps_ranked["id_linea"],
            gps_ranked["id_ramal"], gps_ranked["interno"],
        ]
        gps_ranked["acum_mts"] = (
            dist_mts.fillna(0).groupby(group_keys).cumsum()
        )
        gps_ranked["acum_mts_nan_count"] = (
            dist_mts.isna().astype(int).groupby(group_keys).cumsum()
        )

        # Lookup de acumuladas en cada ancla mediante el id de ping GPS
        acum_km_map      = gps_ranked.set_index("id")["acum_km"]
        acum_mts_map     = gps_ranked.set_index("id")["acum_mts"]
        acum_nan_map     = gps_ranked.set_index("id")["acum_mts_nan_count"]

        gps_anchors["acum_km_o"]  = gps_anchors["id_gps_o"].map(acum_km_map)
        gps_anchors["acum_km_d"]  = gps_anchors["id_gps_d"].map(acum_km_map)
        gps_anchors["acum_mts_o"] = gps_anchors["id_gps_o"].map(acum_mts_map)
        gps_anchors["acum_mts_d"] = gps_anchors["id_gps_d"].map(acum_mts_map)
        gps_anchors["nan_o"]      = gps_anchors["id_gps_o"].map(acum_nan_map)
        gps_anchors["nan_d"]      = gps_anchors["id_gps_d"].map(acum_nan_map)

        # Resta de acumuladas → distancia recorrida entre anclas
        gps_distances = gps_anchors.reindex(columns=["id_legs"]).copy()
        gps_distances["distance_route"] = (
            gps_anchors["acum_km_d"].values - gps_anchors["acum_km_o"].values
        )

        # distance_route_gps: si hubo algún NaN en distance_servicio_mts entre
        # las anclas, el resultado es NaN. Si todos los pings intermedios
        # tienen el valor reportado, se calcula normalmente.
        nan_entre_anclas = (
            gps_anchors["nan_d"].values - gps_anchors["nan_o"].values
        )
        diff_mts = (
            gps_anchors["acum_mts_d"].values - gps_anchors["acum_mts_o"].values
        )
        gps_distances["distance_route_gps"] = np.where(
            nan_entre_anclas > 0,
            np.nan,
            diff_mts / 1000,
        )
        gps_distances = gps_distances.rename(columns={"id_legs": "id"})

        travel_times = legs.reindex(
            columns=["dia", "id", "fecha", "distance_od"]
        ).merge(
            etapas_result.reindex(columns=["id_legs", "fecha_gps"]),
            how="left",
            left_on=["id"],
            right_on=["id_legs"],
        )

        travel_times["travel_time_min"] = round(
            (travel_times["fecha_gps"] - travel_times["fecha"]).dt.total_seconds() / 60,
            1,
        )

        travel_times = travel_times.loc[travel_times.travel_time_min > 0, :]
        travel_times["kmh_od"] = (
            travel_times["distance_od"] / (travel_times["travel_time_min"] / 60)
        ).round(1)

        travel_times.loc[
            (travel_times.kmh_od == np.inf) | (travel_times.kmh_od >= 70),
            "kmh_od",
        ] = np.nan

        travel_times = travel_times.merge(gps_distances, on="id", how="left")

        tot_gps = len(travel_times)
        tot_gps_asig = travel_times.travel_time_min.notna().sum()
        logger.info("GPS imputado: %.1f%%", tot_gps_asig / tot_gps * 100)

        travel_times["kmh_route"] = (
            travel_times["distance_route"] / (travel_times["travel_time_min"] / 60)
        ).round(1)

        travel_times.loc[
            (travel_times.kmh_route == np.inf) | (travel_times.kmh_route >= 70),
            "kmh_route",
        ] = np.nan

        travel_times["kmh_route_gps"] = (
            travel_times["distance_route_gps"] / (travel_times["travel_time_min"] / 60)
        ).round(1)

        travel_times.loc[
            (travel_times.kmh_route_gps == np.inf) | (travel_times.kmh_route_gps >= 70),
            "kmh_route_gps",
        ] = np.nan

        travel_times = travel_times.reindex(
            columns=['dia', 
                     'id', 
                     'travel_time_min', 
                     'distance_od', 
                     'distance_route', 
                     'distance_route_gps', 
                     'kmh_od', 
                     'kmh_route', 
                     'kmh_route_gps'] )


        travel_times = legs_all[['dia', 'id', 'id_tarjeta', 'id_viaje', 'id_etapa', 'distance_od']].merge(travel_times, how='left')
        
        travel_times_trips = (
                travel_times
                .groupby(["dia", "id_tarjeta", "id_viaje"], as_index=False)
                [["travel_time_min", "distance_od", "distance_route", "distance_route_gps"]]
                .sum(min_count=1)
            )
        
        travel_times_trips["kmh_od"] = (
            travel_times_trips["distance_od"] / (travel_times_trips["travel_time_min"] / 60)
        ).round(1)

        travel_times_trips["kmh_route"] = (
            travel_times_trips["distance_route"] / (travel_times_trips["travel_time_min"] / 60)
        ).round(1)

        travel_times_trips["kmh_route_gps"] = (
            travel_times_trips["distance_route_gps"] / (travel_times_trips["travel_time_min"] / 60)
        ).round(1)

        for col in ["kmh_od", "kmh_route", "kmh_route_gps"]:
            travel_times_trips.loc[
                (travel_times_trips[col] == np.inf) | (travel_times_trips[col] >= 70), col
            ] = np.nan

    else:
        travel_times = legs_all[['dia', 'id', 'id_tarjeta', 'id_viaje', 'distance_od']].copy()
        dias_ultima_corrida = ctx.data.get_run_days()

        travel_times_trips = (
                travel_times
                .groupby(["dia", "id_tarjeta", "id_viaje"], as_index=False)
                [["distance_od"]]
                .sum(min_count=1)
            )

    travel_times = travel_times.reindex(
        columns=["dia", "id", "id_tarjeta", "id_viaje", "id_etapa", "travel_time_min", 
                 "distance_od", "distance_route", "distance_route_gps",
                 "kmh_od", "kmh_route", "kmh_route_gps"]
    )

    travel_times_trips = travel_times_trips.reindex(
        columns=["dia", "id_tarjeta", "id_viaje", "travel_time_min",
                 "distance_od", "distance_route", "distance_route_gps",
                 "kmh_od", "kmh_route", "kmh_route_gps"]
    )
        
    travel_times['distance_route_gps'] = travel_times['distance_route_gps'].round(2)

    dias_ultima_corrida = ctx.data.get_run_days()
    dias = dias_ultima_corrida["dia"].tolist()
    dias_str = ", ".join(f"'{d}'" for d in dias)

    for table, df in [("travel_times_legs", travel_times), ("travel_times_trips", travel_times_trips)]:
        ctx.data.execute(f"DELETE FROM {table} WHERE dia IN ({dias_str})")
        ctx.data.append_raw(df, table)
        

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

        # read legs without travel time in gps and distances
        legs = ctx.data.query(
            """
            SELECT e.dia, e.id, e.id_linea, e.id_ramal, e.h3_o, e.h3_d
            FROM etapas e
            LEFT JOIN travel_times_gps tt
            ON e.dia = tt.dia AND e.id = tt.id
            WHERE tt.id IS NULL
            AND e.od_validado = 1
            """
        )
        
        if len(legs) == 0:
            logger.info("No hay etapas sin tiempo de viaje asignado. assign_stations_od no tiene nada que procesar.")
            return
        
        # read stations data
        epsg_m = get_epsg_m()

        travel_times_stations = ctx.insumos.get_travel_times_stations()

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
            "Etapas clasificadas en estaciones de origen: %.1f%%",
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
            "Etapas clasificadas en estaciones de destino: %.1f%%",
            len(legs_with_destination_station) / len(legs) * 100,
        )

        # upload od station into db
        stations_o = legs_with_origin_station.rename(
            columns={"id_station_o": "id_station"}
        ).reindex(columns=["dia", "id_legs", "id_station"])

        stations_d = legs_with_destination_station.rename(
            columns={"id_station_d": "id_station"}
        ).reindex(columns=["dia", "id_legs", "id_station"])

        ctx.data.save_raw(stations_o, "legs_to_station_origin")
        ctx.data.save_raw(stations_d, "legs_to_station_destination")
        
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
            logger.info("No hay etapas con estaciones OD asignadas.")
        else:
            logger.info(
                "Etapas clasificadas en la misma estación OD: %.1f%%",
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

        logger.info(
            "Sin tiempos de viaje: %.1f%%",
            travel_times.travel_time_min.isna().sum() / len(travel_times) * 100,
        )
        travel_times = travel_times.dropna(subset=["travel_time_min"])
        travel_times.loc[:, "kmh_od"] = (
            travel_times.loc[:, "distance_od"]
            / (travel_times.loc[:, "travel_time_min"] / 60)
        ).round(1)

        travel_times.loc[
            (travel_times.kmh_od == np.inf) | (travel_times.kmh_od >= 70),
            "kmh_od",
        ] = np.nan

        # upload to db
        travel_times = travel_times.reindex(
            columns=["dia", "id", "travel_time_min", "kmh_od"]
        )
        
        travel_times = travel_times.reindex(
            columns=["dia", "id", "travel_time_min", "travel_speed"]
        )
        
        ctx.data.save_raw(travel_times, "travel_times_stations")

def add_distance_and_travel_time(ctx: StorageContext):
    """
    This function reads legs data and adds distances and travel times
    from the distances table.
    It also computes the travel speed.
    """

    logger.info("Agregando distancias y tiempos de viaje a las etapas")

    # Leer etapas válidas de la última corrida
    legs = ctx.data.query(
        """
        SELECT e.id, e.h3_d, e.h3_o
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        WHERE e.od_validado = 1
        """
    )
    
    # Calcular distancias
    legs = compute_od_distances(
        od_df=legs,
        origin_col="h3_o",
        dest_col="h3_d",
        distance_col="distance",
        unit="km",
        db_path="data/matriz_distancia/matriz_distancia.duckdb",
        network_cache_dir="data/matriz_distancia",
        symmetric=False,
        precompute_dist=50_000,
        max_tile_deg=99,
        verbose=True
    )

    # Guardar tabla temporal con distancias
    ctx.data.save_raw(legs, "temp_distancias")

    logger.debug("Actualizando distancias a etapas")
    ctx.data.execute(
        """
        UPDATE etapas
        SET distancia = temp_distancias.distance
        FROM temp_distancias
        WHERE etapas.id = temp_distancias.id
        """
    )

    logger.debug("Actualizando tiempos de viaje a etapas")
    ctx.data.execute(
        """
        UPDATE etapas
        SET travel_time_min = travel_times_legs.travel_time_min
        FROM travel_times_legs
        WHERE etapas.id = travel_times_legs.id
        """
    )

    ctx.data.execute("DROP TABLE IF EXISTS temp_distancias")

    
