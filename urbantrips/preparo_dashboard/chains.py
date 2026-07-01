"""Trip-chain pipeline producing the chains_norm table.

chains_norm holds one row per trip with the H3 cells of the chain
(origin, up to two transfers, destination) plus their normalized
(direction-independent) counterparts and the aggregated filter columns
the dashboards group by on the fly:

    dia, id_tarjeta, id_viaje,
    h3_inicio, h3_transfer1, h3_transfer2, h3_fin,
    h3_inicio_norm, h3_transfer1_norm, h3_transfer2_norm, h3_fin_norm,
    modo_agregado, rango_hora, genero_agregado, tarifa_agregada,
    transferencia, distancia_agregada, distance_od, travel_time_min,
    travel_speed, seq_lineas, factor_expansion_linea, tipo_dia, mes

distance_od / travel_time_min are the trip-level values (travel_times_trips),
so dashboards can compute fex-weighted means. seq_lineas is the ' -- '
joined sequence of line names of the trip's legs, used by the line filter
and the transfers-by-line tables.

The table replaces the precomputed etapas_agregadas / viajes_agregados /
poly_etapas / poly_matrices tables: dashboards aggregate chains_norm joined
with the long-format equivalencias_zonas instead.
"""

import logging
import re

import h3
import numpy as np
import pandas as pd

from urbantrips.storage.context import StorageContext
from urbantrips.utils.utils import VELOCIDAD_MAXIMA_KMH
from urbantrips.preparo_dashboard.aggregation import (
    calcular_modo_agregado,
    clasificar_distancia_agregada,
    clasificar_genero_agregado,
    clasificar_mes,
    clasificar_rango_hora,
    clasificar_tarifa_agregada_social,
    clasificar_tipo_dia,
)

logger = logging.getLogger(__name__)

RES_CHAINS_NORM = 10


def compute_od_coordinates_h3(etapas: pd.DataFrame, h3_resolution: int = 8) -> pd.DataFrame:
    """
    Computes destination coordinates (lat_d, lon_d) for each leg by shifting
    origin coordinates within each (dia, id_tarjeta) group, closing the daily
    chain by assigning the first leg's origin as the last leg's destination.
    Also recomputes h3_o and h3_d at the requested resolution.

    Parameters
    ----------
    etapas : DataFrame
        Must contain columns: dia, id_tarjeta, id_viaje, id_etapa,
        latitud, longitud.
    h3_resolution : int, default 8
        H3 resolution for h3_o and h3_d.

    Returns
    -------
    DataFrame with lat_d, lon_d, h3_o, h3_d updated in place.
    """
    etapas = etapas.sort_values(["dia", "id_tarjeta", "id_viaje", "id_etapa"])

    # destination = next leg's origin within same (dia, id_tarjeta)
    etapas["lat_d"] = etapas.groupby(["dia", "id_tarjeta"])["latitud"].shift(-1)
    etapas["lon_d"] = etapas.groupby(["dia", "id_tarjeta"])["longitud"].shift(-1)

    # close daily chain: last leg's destination = first leg's origin
    primero = etapas.groupby(["dia", "id_tarjeta"])[["latitud", "longitud"]].first()
    ultimo_mask = (
        etapas["lat_d"].isna()
        & (etapas["id_etapa"] == etapas.groupby(
            ["dia", "id_tarjeta", "id_viaje"])["id_etapa"].transform("max"))
    )
    filled = etapas.loc[ultimo_mask, ["dia", "id_tarjeta"]].join(
        primero, on=["dia", "id_tarjeta"]
    )
    etapas.loc[ultimo_mask, "lat_d"] = filled["latitud"].values
    etapas.loc[ultimo_mask, "lon_d"] = filled["longitud"].values

    # recompute h3_o and h3_d at requested resolution
    valid_o = etapas["latitud"].notna() & etapas["longitud"].notna()
    etapas.loc[valid_o, "h3_o"] = np.vectorize(
        lambda lat, lon: h3.latlng_to_cell(lat, lon, h3_resolution)
    )(etapas.loc[valid_o, "latitud"], etapas.loc[valid_o, "longitud"])

    valid_d = etapas["lat_d"].notna() & etapas["lon_d"].notna()
    etapas.loc[valid_d, "h3_d"] = np.vectorize(
        lambda lat, lon: h3.latlng_to_cell(lat, lon, h3_resolution)
    )(etapas.loc[valid_d, "lat_d"], etapas.loc[valid_d, "lon_d"])

    return etapas


def build_filters_tables(
    etapas: pd.DataFrame,
    viajes: pd.DataFrame,
    duplicar_filtros_viaje_en_legs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds two precomputed filter tables for legs and trips.

    Parameters
    ----------
    etapas : DataFrame
        Legs table already loaded with distance_od resolved (previously
        joined from travel_times_legs). Required columns: id, dia,
        id_tarjeta, id_viaje, id_etapa, h3_o, h3_d, hora, modo, tarifa,
        genero, distance_od, travel_time_min, factor_expansion_linea.
    viajes : DataFrame
        Trips table already loaded. Required columns: dia, id_tarjeta,
        id_viaje, tiempo, hora, cant_etapas, distance_od, travel_time_min,
        factor_expansion_linea.
    duplicar_filtros_viaje_en_legs : bool, default True
        If True, copies 'modo_agregado' and 'transferencia' to filters_legs
        to avoid extra joins when filtering legs by trip attributes.

    Returns
    -------
    filters_legs : DataFrame
        Key: id (legs PK). Includes dia for partitioning.
    filters_trips : DataFrame
        Keys: (dia, id_tarjeta, id_viaje).
    """

    keys_viaje = ["dia", "id_tarjeta", "id_viaje"]

    # =========================================================
    # 1. filters_legs
    # =========================================================
    legs = pd.DataFrame({
        "id": etapas["id"].values,
        "dia": etapas["dia"].values,
        "id_tarjeta": etapas["id_tarjeta"].values,
        "id_viaje": etapas["id_viaje"].values,
        "id_etapa": etapas["id_etapa"].values,
        "h3_o": etapas["h3_o"].values,
        "h3_d": etapas["h3_d"].values,
    })
    if "nombre_linea" in etapas.columns:
        legs["nombre_linea"] = etapas["nombre_linea"].values

    # --- rango_hora ------------------------------------------------------
    legs["rango_hora"] = clasificar_rango_hora(etapas["hora"]).values

    # --- distancia_agregada ----------------------------------------------
    legs["distancia_agregada"] = clasificar_distancia_agregada(
        etapas["distance_od"], nivel="etapa"
    ).values

    # --- tipo_dia y mes ---------------------------------------------------
    legs["tipo_dia"] = clasificar_tipo_dia(etapas["dia"]).values
    legs["mes"] = clasificar_mes(etapas["dia"]).values

    # --- tarifa y genero --------------------------------------------------
    legs["tarifa_agregada"] = clasificar_tarifa_agregada_social(etapas["tarifa"]).values
    legs["genero_agregado"] = clasificar_genero_agregado(etapas["genero"]).values

    # --- factor de expansión ----------------------------------------------
    legs["factor_expansion_linea"] = (
        pd.to_numeric(etapas["factor_expansion_linea"], errors="coerce")
        .astype("float32").values
    )

    # --- modo_agregado y transferencia (trip level, computed from legs) ---
    agg = calcular_modo_agregado(etapas, keys_viaje)

    if duplicar_filtros_viaje_en_legs:
        legs_keys = etapas[keys_viaje].reset_index(drop=True)
        legs_keys = legs_keys.merge(agg, on=keys_viaje, how="left")
        legs["modo_agregado"] = legs_keys["modo_agregado"].values
        legs["transferencia"] = legs_keys["transferencia"].fillna(0).astype("int8").values

    # =========================================================
    # 2. filters_trips
    # =========================================================
    trips = viajes[keys_viaje].copy()

    # --- travel_speed (trip level, capped) ---------------------------------
    tt_min_v = pd.to_numeric(viajes["travel_time_min"], errors="coerce").fillna(0)
    dist_v = pd.to_numeric(viajes["distance_od"], errors="coerce")
    speed_v = np.where(
        tt_min_v > 0,
        (dist_v / (tt_min_v / 60)).round(1),
        np.nan,
    )
    speed_v = np.where(speed_v >= VELOCIDAD_MAXIMA_KMH, np.nan, speed_v)
    trips["travel_speed"] = speed_v.astype("float32")

    # --- rango_hora ------------------------------------------------------
    trips["rango_hora"] = clasificar_rango_hora(viajes["hora"]).values

    # --- distancia_agregada ----------------------------------------------
    trips["distancia_agregada"] = clasificar_distancia_agregada(
        viajes["distance_od"], nivel="viaje"
    ).values

    # --- tipo_dia y mes ---------------------------------------------------
    trips["tipo_dia"] = clasificar_tipo_dia(viajes["dia"]).values
    trips["mes"] = clasificar_mes(viajes["dia"]).values

    # --- distance_od y travel_time_min (trip level, for weighted means) ----
    trips["distance_od"] = dist_v.astype("float32").values
    trips["travel_time_min"] = tt_min_v.astype("float32").values

    # --- transferencia (directly from cant_etapas) -------------------------
    trips["transferencia"] = (viajes["cant_etapas"] > 1).astype("int8").values

    # --- modo_agregado (joined from agg) -----------------------------------
    trips = trips.merge(agg[keys_viaje + ["modo_agregado"]], on=keys_viaje, how="left")

    # --- diff_time (gap to the user's next trip on the same day) -----------
    fecha_v = pd.to_datetime(viajes["dia"].astype(str) + " " + viajes["tiempo"].astype(str))
    fecha_next = fecha_v.groupby([viajes["dia"], viajes["id_tarjeta"]]).shift(-1)
    trips["diff_time"] = (
        ((fecha_next - fecha_v).dt.total_seconds() / 60).round().astype("float32")
    )

    # =========================================================
    # Quick validations
    # =========================================================
    assert len(legs) == len(etapas), "filters_legs perdió filas respecto de etapas"
    assert len(trips) == len(viajes), "filters_trips perdió filas respecto de viajes"
    assert legs["id"].is_unique, "filters_legs.id no es único"

    # use trip-level distancia_agregada / distance_od / travel_time_min /
    # travel_speed on legs, dropping the leg-level distancia_agregada.
    # travel_speed must come from the trips frame (capped, trip-level):
    # construir_cadenas_viajes reads the chain's attributes from the first
    # leg, so a leg-level speed here would leak the first leg's uncapped
    # speed into chains_norm.
    legs = legs.drop(["distancia_agregada"], axis=1).merge(
        trips[["dia", "id_tarjeta", "id_viaje", "distancia_agregada",
               "distance_od", "travel_time_min", "travel_speed"]],
        on=["dia", "id_tarjeta", "id_viaje"],
        how="left",
    )

    return legs, trips


def construir_cadenas_viajes(
    etapas,
    col_origen="h3_o",
    col_destino="h3_d",
    max_etapas=4,
    verbose=True,
):
    """Collapse legs into one row per trip with the chain of H3 cells.

    Trips with more than ``max_etapas`` legs are dropped (and logged).
    Output columns: keys + h3_inicio, h3_transfer1..h3_transfer{max-1},
    h3_fin + the trip-level filter attributes present in ``etapas``.
    """
    keys = ["dia", "id_tarjeta", "id_viaje"]
    df = etapas.copy()

    df["cant_etapas"] = df.groupby(keys)["id_etapa"].transform("nunique")
    mask_excluir = df["cant_etapas"] > max_etapas

    if mask_excluir.any():
        viajes_total = df[keys].drop_duplicates().shape[0]
        viajes_excluidos = df.loc[mask_excluir, keys].drop_duplicates().shape[0]

        if verbose:
            logger.info(
                "Se borran viajes con más de %d etapas: %s viajes (%.2f%%).",
                max_etapas,
                f"{viajes_excluidos:,}",
                viajes_excluidos / viajes_total * 100,
            )

        df = df.loc[~mask_excluir].copy()

    df["etapa_orden"] = (
        df.groupby(keys)["id_etapa"]
        .rank(method="dense")
        .astype(int)
    )

    cols_inicio = [
        "dia",
        "id_tarjeta",
        "id_viaje",
        "modo_agregado",
        "rango_hora",
        "genero_agregado",
        "tarifa_agregada",
        "transferencia",
        "distancia_agregada",
        "distance_od",
        "travel_time_min",
        "travel_speed",
        "factor_expansion_linea",
        "tipo_dia",
        "mes",
    ]

    cols_inicio = [c for c in cols_inicio if c in df.columns]

    inicio = (
        df.loc[df["etapa_orden"] == 1, cols_inicio + [col_origen]]
        .rename(columns={col_origen: "h3_inicio"})
    )

    transfers = (
        df.loc[df["etapa_orden"].between(2, max_etapas), keys + ["etapa_orden", col_origen]]
        .assign(tipo=lambda x: "h3_transfer" + (x["etapa_orden"] - 1).astype(str))
        .pivot_table(
            index=keys,
            columns="tipo",
            values=col_origen,
            aggfunc="first",
        )
        .reset_index()
    )

    fin = (
        df.loc[df["etapa_orden"] == df["cant_etapas"], keys + [col_destino]]
        .rename(columns={col_destino: "h3_fin"})
    )

    cadenas = (
        inicio
        .merge(transfers, on=keys, how="left")
        .merge(fin, on=keys, how="left")
    )

    for i in range(1, max_etapas):
        col = f"h3_transfer{i}"
        if col not in cadenas.columns:
            cadenas[col] = ""

    columnas_h3 = (
        ["h3_inicio"]
        + [f"h3_transfer{i}" for i in range(1, max_etapas)]
        + ["h3_fin"]
    )
    # only the chain columns get "" for missing — numeric metrics keep NaN
    cadenas[columnas_h3] = cadenas[columnas_h3].fillna("")

    # line-name sequence of the trip's legs (' -- ' joined, in leg order)
    tiene_lineas = "nombre_linea" in df.columns
    if tiene_lineas:
        con_linea = df[df["nombre_linea"].notna() & (df["nombre_linea"] != "")]
        seq = (
            con_linea.sort_values(keys + ["etapa_orden"])
            .groupby(keys)["nombre_linea"]
            .agg(" -- ".join)
            .reset_index(name="seq_lineas")
        )
        cadenas = cadenas.merge(seq, on=keys, how="left")
        cadenas["seq_lineas"] = cadenas["seq_lineas"].fillna("")

    columnas_atributos = [c for c in cols_inicio if c not in keys]
    if tiene_lineas:
        columnas_atributos = columnas_atributos + ["seq_lineas"]

    return cadenas[keys + columnas_h3 + columnas_atributos]


def normalizar_cadenas(df):
    """Add direction-independent *_norm columns to a chains table.

    A chain is reversed (origin <-> destination, transfers mirrored) when
    h3_inicio > h3_fin, so that A->B and B->A trips share the same
    normalized chain and can be aggregated bidirectionally.
    """
    df = df.copy()
    transfer_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"h3_transfer\d+", c)],
        key=lambda x: int(x.replace("h3_transfer", ""))
    )
    cols_base = ["h3_inicio"] + transfer_cols + ["h3_fin"]

    # mask: rows that must be reversed (inicio > fin)
    invertir = df["h3_inicio"] > df["h3_fin"]

    for col in cols_base:
        df[col + "_norm"] = df[col]

    # swap inicio <-> fin, vectorized
    df.loc[invertir, "h3_inicio_norm"] = df.loc[invertir, "h3_fin"]
    df.loc[invertir, "h3_fin_norm"] = df.loc[invertir, "h3_inicio"]

    # reorder transfers by count (vectorized per case)
    if "h3_transfer1" in df.columns and "h3_transfer2" in df.columns:
        t1 = df["h3_transfer1"].fillna("")
        t2 = df["h3_transfer2"].fillna("")
        n2 = invertir & (t1 != "") & (t2 != "")
        df.loc[n2, "h3_transfer1_norm"] = t2[n2]
        df.loc[n2, "h3_transfer2_norm"] = t1[n2]

    if "h3_transfer1" in df.columns and "h3_transfer3" in df.columns:
        t1 = df["h3_transfer1"].fillna("")
        t3 = df["h3_transfer3"].fillna("")
        n3 = invertir & (t1 != "") & (t3 != "")
        df.loc[n3, "h3_transfer1_norm"] = t3[n3]
        df.loc[n3, "h3_transfer3_norm"] = t1[n3]

    cols_norm = [c + "_norm" for c in cols_base]
    cols_extra = [
        c for c in ["distance_od", "travel_time_min", "seq_lineas"]
        if c in df.columns
    ]
    return df[["dia", "id_tarjeta", "id_viaje"] + cols_base + cols_norm +
              ["modo_agregado", "rango_hora", "genero_agregado", "tarifa_agregada",
               "transferencia", "distancia_agregada", "travel_speed",
               "factor_expansion_linea", "tipo_dia", "mes"] + cols_extra]


def guardar_chains_norm(chains_norm, ctx: StorageContext):
    """Persist chains_norm partitioned by 'dia' via the DuckDB dash adapter.

    Existing rows for the days in ``chains_norm`` are deleted before inserting,
    so re-running a day replaces it instead of duplicating rows.
    """
    if chains_norm is None or len(chains_norm) == 0:
        logger.info("guardar_chains_norm: nada que guardar.")
        return

    dias = chains_norm["dia"].unique().tolist()
    logger.info(
        "guardar_chains_norm: guardando %s filas (%d días).",
        f"{len(chains_norm):,}", len(dias),
    )
    ctx.dash.upsert_chains_norm(chains_norm, dias)


def procesar_pipeline_por_dia(
    res: int = RES_CHAINS_NORM,
    max_etapas: int = 3,
    verbose: bool = True,
    guardar: bool = False,
    ctx: StorageContext | None = None,
) -> pd.DataFrame:
    """
    Runs the filters + chains pipeline day by day and returns chains_norm
    (concatenation of every day).

    Parameters
    ----------
    res : int, default RES_CHAINS_NORM (10)
        H3 resolution for h3_o / h3_d recomputation.
    max_etapas : int, default 3
        Trips with more legs than this are dropped.
    verbose : bool, default True
        Log per-day progress.
    guardar : bool, default False
        If True, each day is persisted via ctx.dash as soon as it is processed
        (keeps memory flat for long runs).
    ctx : StorageContext, optional
        When None a temporary context is built and closed after use.
    """
    _own_ctx = ctx is None
    if _own_ctx:
        from urbantrips.utils.run_process import _build_ctx
        ctx = _build_ctx()

    try:
        dias = ctx.data.query(
            "SELECT DISTINCT dia FROM dias_ultima_corrida ORDER BY dia"
        )["dia"].tolist()

        if len(dias) == 0:
            logger.warning("procesar_pipeline_por_dia: no hay días en dias_ultima_corrida.")
            return pd.DataFrame([])

        logger.info("Procesando %d días: %s → %s", len(dias), dias[0], dias[-1])

        lineas = ctx.insumos.query(
            "SELECT id_linea, nombre_linea FROM metadata_lineas"
        )

        chains_list = []

        for dia in dias:
            if verbose:
                logger.info("Procesando día %s ...", dia)

            etapas_dia = ctx.data.query(f"""
                SELECT e.id, e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa,
                       e.latitud, e.longitud,
                       e.hora, e.modo, e.id_linea, e.tarifa, e.genero,
                       e.factor_expansion_linea,
                       tt.distance_od, tt.travel_time_min
                FROM etapas e
                JOIN dias_ultima_corrida d ON e.dia = d.dia
                LEFT JOIN travel_times_legs tt ON e.id = tt.id
                WHERE e.od_validado = 1 AND e.dia = '{dia}'
            """)

            if etapas_dia.empty:
                if verbose:
                    logger.info("Día %s sin datos, skip.", dia)
                continue

            if len(lineas) > 0:
                etapas_dia = etapas_dia.merge(lineas, on="id_linea", how="left")

            viajes_dia = ctx.data.query(f"""
                SELECT v.dia, v.id_tarjeta, v.id_viaje,
                       v.hora, v.tiempo, v.cant_etapas,
                       tt.distance_od, tt.travel_time_min
                FROM viajes v
                JOIN dias_ultima_corrida d ON v.dia = d.dia
                LEFT JOIN travel_times_trips tt
                    ON v.dia = tt.dia AND v.id_tarjeta = tt.id_tarjeta
                    AND v.id_viaje = tt.id_viaje
                WHERE v.od_validado = 1 AND v.dia = '{dia}'
            """)

            etapas_dia = compute_od_coordinates_h3(etapas_dia, res)
            legs, _ = build_filters_tables(etapas_dia, viajes_dia)
            chains_dia = normalizar_cadenas(
                construir_cadenas_viajes(legs, max_etapas=max_etapas, verbose=verbose)
            )

            if guardar:
                guardar_chains_norm(chains_dia, ctx)
            else:
                # Solo se acumula en memoria cuando el caller espera el concat
                # de retorno. Con guardar=True cada día ya quedó persistido, así
                # que acumular las 26.5M filas y concatenarlas al final es trabajo
                # y RAM desperdiciados (el caller descarta el retorno).
                chains_list.append(chains_dia)

            if verbose:
                logger.info("Día %s: %s etapas procesadas.", dia, f"{len(etapas_dia):,}")

        if guardar:
            return pd.DataFrame([])

        if not chains_list:
            return pd.DataFrame([])

        chains_norm = pd.concat(chains_list, ignore_index=True)
        logger.info("Total cadenas: %s", f"{len(chains_norm):,}")
        return chains_norm

    finally:
        if _own_ctx:
            for port in (ctx.data, ctx.insumos, ctx.dash, ctx.general):
                if hasattr(port, "close"):
                    port.close()
