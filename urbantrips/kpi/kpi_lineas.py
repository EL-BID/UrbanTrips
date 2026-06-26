import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from urbantrips.utils import utils
from urbantrips.utils.utils import duracion
from urbantrips.storage.context import StorageContext
from datetime import datetime

logger = logging.getLogger(__name__)


def _sql_in_values(values):
    escaped = [str(value).replace("'", "''") for value in values]
    return ", ".join(f"'{value}'" for value in escaped)


def _delete_kpis_lineas_if_exists(ctx: StorageContext, dias):
    if len(dias) == 0:
        return

    dias_str = _sql_in_values(dias)
    try:
        ctx.general.execute(f"DELETE FROM kpis_lineas WHERE dia IN ({dias_str})")
    except Exception as exc:
        if "does not exist" not in str(exc):
            raise


def cal_velocidad_comercial(servicios):
    # Conversión de columnas a datetime
    servicios["min_datetime"] = pd.to_datetime(servicios["min_datetime"])
    servicios["max_datetime"] = pd.to_datetime(servicios["max_datetime"])

    # Cálculo de duración del servicio en minutos
    servicios["diff_minutes"] = (
        servicios["max_datetime"] - servicios["min_datetime"]
    ).dt.total_seconds() / 60

    # Cálculo de velocidad comercial
    servicios["velocidad_comercial"] = servicios["distance_route"] / (
        servicios["diff_minutes"] / 60
    )

    # Extraer hora de finalización del servicio
    servicios["hour"] = servicios["max_datetime"].dt.hour

    # Velocidad comercial por línea y ramal en hora pico AM
    filtro_pico_am = (servicios["diff_minutes"] < 180) & (
        servicios["hour"].between(6, 10)
    )
    vel_comercial_linea_ramal_pico = (
        servicios[filtro_pico_am]
        .groupby(["dia", "id_linea", "id_ramal"], as_index=False)["velocidad_comercial"]
        .mean()
        .round(1)
    )

    # Distancia media recorrida por vehículo en ramal
    km_recorridos_ramal = (
        servicios.groupby(["dia", "id_linea", "id_ramal", "interno"], as_index=False)[
            "distance_route"
        ]
        .sum()
        .groupby(["dia", "id_linea", "id_ramal"], as_index=False)["distance_route"]
        .mean()
        .rename(columns={"distance_route": "distancia_media_veh"})
        .round(1)
    )

    vel_comercial_linea_ramal_pico = vel_comercial_linea_ramal_pico.merge(
        km_recorridos_ramal, how="left"
    )

    # Velocidad comercial total por línea (todo el día)
    vel_comercial_linea_all = (
        servicios.groupby(["dia", "id_linea"], as_index=False)["velocidad_comercial"]
        .mean()
        .round(1)
    )

    # Velocidad comercial AM
    vel_comercial_linea_am = (
        servicios[filtro_pico_am]
        .groupby(["dia", "id_linea"], as_index=False)["velocidad_comercial"]
        .mean()
        .round(1)
        .rename(columns={"velocidad_comercial": "velocidad_comercial_am"})
    )

    # Velocidad comercial PM (15 a 19 hs)
    filtro_pico_pm = (servicios["diff_minutes"] < 180) & (
        servicios["hour"].between(15, 19)
    )
    vel_comercial_linea_pm = (
        servicios[filtro_pico_pm]
        .groupby(["dia", "id_linea"], as_index=False)["velocidad_comercial"]
        .mean()
        .round(1)
        .rename(columns={"velocidad_comercial": "velocidad_comercial_pm"})
    )

    # Consolidar velocidades comerciales
    vel_comercial_linea = vel_comercial_linea_all.merge(
        vel_comercial_linea_am, how="left"
    ).merge(vel_comercial_linea_pm, how="left")

    # Distancia media recorrida por vehículo (total)
    km_recorridos_linea = (
        servicios.groupby(["dia", "id_linea", "interno"], as_index=False)["distance_route"]
        .sum()
        .groupby(["dia", "id_linea"], as_index=False)["distance_route"]
        .mean()
        .rename(columns={"distance_route": "distancia_media_veh"})
        .round(1)
    )

    vel_comercial_linea = vel_comercial_linea.merge(km_recorridos_linea, how="left")

    return vel_comercial_linea


def levanto_data(ctx: StorageContext, etapas=[], viajes=[]):

    # Only the columns used below — gps and transacciones are the two largest
    # tables in the run; loading them whole multiplies peak RSS.
    gps = ctx.data.query("SELECT fecha, id_linea, id_ramal, interno FROM gps")

    trx = ctx.data.query("SELECT dia, id_linea, id_ramal, interno FROM transacciones")

    lineas = ctx.insumos.get_metadata_lineas()[
        ["id_linea", "nombre_linea", "empresa"]
    ].drop_duplicates()

    kpis = ctx.data.get_raw("kpi_by_day_line")

    try:
        servicios = ctx.data.query("SELECT * FROM services WHERE valid = 1")
    except Exception:
        servicios = pd.DataFrame()

    # Procesamiento de GPS y cálculo de flota
    gps["fecha"] = pd.to_datetime(gps["fecha"], unit="s")
    gps["dia"] = gps["fecha"].dt.strftime("%Y-%m-%d")

    flota = (
        gps.groupby(["dia", "id_linea"], as_index=False)
        .size()
        .rename(columns={"size": "flota"})
    )

    # Cálculo de velocidad comercial
    vel_comercial_linea = cal_velocidad_comercial(servicios)

    # Procesamiento de transacciones

    kpis_varios = flota.merge(vel_comercial_linea, how="left").merge(kpis, how="left")

    return trx, etapas, gps, servicios, kpis_varios, lineas


@duracion
def agrego_lineas(cols, trx, etapas, gps, servicios, kpis_varios, lineas,
                  etapas_query_fn=None, etapas_source=None, etapas_cte_prefix=""):

    if etapas_query_fn is not None:
        # Push-down: transacciones + genero/tarifa pivots in one query over
        # etapas_proc (the SUM(CASE ...) reproduce groupby+unstack(fill_value=0)
        # for the fixed classifier label sets). This is already `tot` merged with
        # resumen_genero + resumen_tarifas.
        _keys = ", ".join(cols + ["modo"])
        tot = etapas_query_fn(etapas_cte_prefix + f"""
            SELECT {_keys},
                SUM(factor_expansion_linea) AS transacciones,
                SUM(CASE WHEN genero_agregado = 'Femenino'     THEN factor_expansion_linea ELSE 0 END) AS "Femenino",
                SUM(CASE WHEN genero_agregado = 'Masculino'    THEN factor_expansion_linea ELSE 0 END) AS "Masculino",
                SUM(CASE WHEN genero_agregado = 'No informado' THEN factor_expansion_linea ELSE 0 END) AS "No informado",
                SUM(CASE WHEN tarifa_agregada = 'educacion_jubilacion' THEN factor_expansion_linea ELSE 0 END) AS "educacion_jubilacion",
                SUM(CASE WHEN tarifa_agregada = 'tarifa_social'        THEN factor_expansion_linea ELSE 0 END) AS "tarifa_social",
                SUM(CASE WHEN tarifa_agregada = 'sin_descuento'        THEN factor_expansion_linea ELSE 0 END) AS "sin_descuento"
            FROM {etapas_source}
            GROUP BY {_keys}
        """)
    else:
        # Agregado de transacciones (in-RAM)
        resumen_tarifas = (
            etapas.groupby(cols + ["modo"] + ["tarifa_agregada"])["factor_expansion_linea"]
            .sum().unstack(fill_value=0).reset_index()
        )
        resumen_genero = (
            etapas.groupby(cols + ["modo"] + ["genero_agregado"])["factor_expansion_linea"]
            .sum().unstack(fill_value=0).reset_index()
        )
        tot = (
            etapas.groupby(cols + ["modo"])["factor_expansion_linea"]
            .sum().reset_index().rename(columns={"factor_expansion_linea": "transacciones"})
        )
        tot = tot.merge(resumen_genero, how="left", on=cols + ["modo"]).merge(
            resumen_tarifas, how="left", on=cols + ["modo"]
        )

    # Agregado de etapas con medias ponderadas (push-down cuando hay query_fn)
    etapas_agg = (
        utils.calculate_weighted_means(
            etapas,
            aggregate_cols=cols + ["modo"],
            weighted_mean_cols=["distance_od", "travel_time_min", "kmh_od"],
            zero_to_nan=["distance_od", "travel_time_min", "kmh_od"],
            weight_col="factor_expansion_linea",
            var_fex_summed=False,
            query_fn=etapas_query_fn,
            source=etapas_source,
            cte_prefix=etapas_cte_prefix,
        )
        .round(2)
        .rename(columns={"distance_od": "distancia_media_pax"})
    )

    # # Redondear solo columnas numéricas
    for col in tot.select_dtypes(include="float").columns:
        try:
            tot[col] = pd.to_numeric(tot[col], errors="coerce").round().astype("Int64")
        except Exception as e:
            logger.warning("Error en columna %s: %s", col, e)

    etapas_agg = tot.merge(etapas_agg, how="left", on=cols + ["modo"])

    # Agregado de cantidad de internos en transacciones
    internos_agg = (
        trx.groupby(cols + ["interno"], as_index=False)
        .size()
        .groupby(cols, as_index=False)
        .size()
        .rename(columns={"size": "cant_internos_en_trx"})
    )

    # Agregado de cantidad de internos con GPS
    gps_agg = (
        gps.groupby(cols + ["interno"], as_index=False)
        .size()
        .groupby(cols, as_index=False)
        .size()
        .rename(columns={"size": "cant_internos_en_gps"})
    )

    # Agregado de servicios válidos
    serv_agg = (
        servicios[servicios.valid == 1]
        .groupby(cols, as_index=False)
        .agg({"interno": "count", "distance_route": "sum", "min_ts": "sum"})
        .rename(
            columns={
                "interno": "cant_servicios",
                "distance_route": "serv_distance_route",
                "min_ts": "serv_min_ts",
            }
        )
    )

    # Merge de todos los datasets
    all = (
        etapas_agg.merge(internos_agg, how="left")
        .merge(gps_agg, how="left")
        .merge(kpis_varios, how="left")
        .merge(lineas, how="left")
        .merge(serv_agg, how="left")
    )

    # Cálculo de mes
    all["mes"] = all["dia"].str[:7]

    # Redondeo de valores
    all["transacciones"] = all["transacciones"].round(0)
    all["tot_pax"] = all["tot_pax"].round(0).fillna(0)
    all["flota"] = all["flota"].round(0)
    all["serv_min_ts"] = all["serv_min_ts"].round(2)
    all = all.round({col: 2 for col in all.select_dtypes(include="float").columns})

    for i in [
        "Femenino",
        "Masculino",
        "No informado",
        "educacion_jubilacion",
        "tarifa_social",
        "sin_descuento",
    ]:
        if i not in all.columns:
            all[i] = 0

    # vehiculos_operativos: conteo directo de internos con al menos un servicio valid=1
    veh_validos = (
        servicios[servicios.valid == 1]
        .groupby(cols, as_index=False)["interno"]
        .nunique()
        .rename(columns={"interno": "vehiculos_operativos"})
    )
    all = all.drop(columns=["vehiculos_operativos"], errors="ignore").merge(
        veh_validos, on=cols, how="left"
    )

    # tot_km: solo km de servicios con valid=1
    all["tot_km"] = all["serv_distance_route"]

    # tot_veh sincronizado con vehiculos_operativos corregido
    all["tot_veh"] = all["vehiculos_operativos"]

    # Recalcular ratios que dependen de tot_veh y tot_km
    all["pvd"] = (all["tot_pax"] / all["tot_veh"].replace(0, pd.NA)).round(1)
    all["kvd"] = (all["tot_km"] / all["tot_veh"].replace(0, pd.NA)).round(1)
    all["ipk_route"] = (all["tot_pax"] / all["tot_km"].replace(0, pd.NA)).round(1)

    all = all[
        [
            "dia",
            "mes",
            "id_linea",
            "nombre_linea",
            "empresa",
            "modo",
            "transacciones",
            "Femenino",
            "Masculino",
            "No informado",
            "educacion_jubilacion",
            "sin_descuento",
            "tarifa_social",
            "travel_time_min",
            "kmh_od",
            "cant_internos_en_gps",
            "cant_internos_en_trx",
            "flota",
            "vehiculos_operativos",
            "velocidad_comercial",
            "velocidad_comercial_am",
            "velocidad_comercial_pm",
            "distancia_media_veh",
            "tot_km",
            "distancia_media_pax",
            "dmt_mean_od",
            "dmt_mean_route",
            "dmt_mean_route_gps",
            "dmt_median_od",
            "dmt_median_route",
            "dmt_median_route_gps",
            "pvd",
            "kvd",
            "ipk_route",
            "ipk_route_gps",
            "fo_mean_od",
            "fo_mean_route",
            "fo_mean_route_gps",
            "fo_median_od",
            "fo_median_route",
            "fo_median_route_gps",
        ]
    ]

    for i in ["dia", "mes", "id_linea", "nombre_linea", "empresa", "modo"]:
        all[i] = all[i].fillna("").astype(str)
    lista = [
        x
        for x in all.columns.tolist()
        if x not in ["dia", "mes", "id_linea", "nombre_linea", "empresa", "modo"]
    ]
    for i in lista:
        if i in [
            "transacciones",
            "Femenino",
            "Masculino",
            "No informado",
            "educacion_jubilacion",
            "sin_descuento",
            "tarifa_social",
        ]:
            all[i] = all[i].fillna(0).astype(int)
        else:
            all[i] = all[i].astype(float).round(1)

    return all


@duracion
def calculo_kpi_lineas(ctx: StorageContext, etapas=[], viajes=[]):
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, ETAPAS_PROC_MAT,
    )
    materializar_proc_tables(ctx)

    trx, _etapas, gps, servicios, kpis_varios, lineas = levanto_data(ctx)
    kpis = agrego_lineas(
        ["dia", "id_linea"], trx, None, gps, servicios, kpis_varios, lineas,
        etapas_query_fn=ctx.data.query, etapas_source=ETAPAS_PROC_MAT,
    )

    # delete existing rows for these days AND the previous "Promedios" row before
    # re-reading. En corridas incrementales (--step dashboard) la fila "Promedios"
    # vieja sobrevivía, se re-leía, contaminaba la media nueva y se duplicaba al
    # re-appendear (3631 vs 3223 filas). Borrarla acá deja la lectura limpia.
    dias = kpis.dia.unique().tolist()
    _delete_kpis_lineas_if_exists(ctx, dias + ["Promedios"])
    ctx.general.append_raw(kpis, "kpis_lineas")

    df = ctx.general.get_raw("kpis_lineas")
    tot = (
        df.drop(["dia", "mes"], axis=1)
        .groupby(["id_linea", "nombre_linea", "empresa", "modo"], as_index=False)
        .mean()
    )
    tot["dia"] = "Promedios"
    tot["mes"] = ""
    df = pd.concat([df, tot], ignore_index=True)

    # replace the whole table including the new averages row
    all_dias = df.dia.unique().tolist()
    _delete_kpis_lineas_if_exists(ctx, all_dias)
    ctx.general.append_raw(df, "kpis_lineas")

    return df
