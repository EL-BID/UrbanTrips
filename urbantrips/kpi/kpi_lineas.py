import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from urbantrips.utils import utils
from urbantrips.utils.dataframe import (
    combinar_conteo_distintos,
    combinar_suma,
    dias_para_leer_por_dia,
    leer_dia,
)
from urbantrips.utils.utils import duracion
from urbantrips.storage.context import StorageContext
from datetime import datetime

logger = logging.getLogger(__name__)


def _sql_in_values(values):
    escaped = [str(value).replace("'", "''") for value in values]
    return ", ".join(f"'{value}'" for value in escaped)


def _alinear_esquema_kpis_lineas(ctx: StorageContext, df, tabla="kpis_lineas") -> None:
    """Alinea `tabla` al esquema de `df` antes de insertar.

    `append_raw` hace `INSERT INTO t SELECT * FROM df`, que exige que la tabla
    tenga exactamente las mismas columnas y en el mismo orden. Al agregar
    columnas nuevas (`tot_km_route_gps`, `kvd_route_gps`) una corrida
    incremental sobre una base ya escrita fallaría con un mismatch.

    Se reconstruye la tabla proyectada al esquema nuevo: las columnas que ya
    existían se conservan con sus datos, las nuevas quedan en NULL para los
    días viejos, y las que el df ya no produce se descartan. Es no-op cuando el
    esquema coincide, que es el caso normal.
    """
    try:
        info = ctx.general.query(f"SELECT * FROM {tabla} LIMIT 0")
    except Exception as exc:
        if "does not exist" in str(exc) or "not found" in str(exc).lower():
            return  # tabla nueva: la crea append_raw con el esquema del df
        raise

    existentes = list(info.columns)
    nuevas = list(df.columns)
    # sin columnas la tabla no existe todavía (algunos adapters devuelven un df
    # vacío en vez de lanzar): la crea append_raw con el esquema del df
    if not existentes or existentes == nuevas:
        return

    proyeccion = ", ".join(
        f'"{c}"' if c in existentes else f'CAST(NULL AS DOUBLE) AS "{c}"'
        for c in nuevas
    )
    logger.info(
        "Migrando esquema de %s: %s -> %s columnas (nuevas: %s)",
        tabla, len(existentes), len(nuevas), sorted(set(nuevas) - set(existentes)),
    )
    ctx.general.execute(
        f"CREATE OR REPLACE TABLE {tabla}_mig AS SELECT {proyeccion} FROM {tabla}"
    )
    ctx.general.execute(f"DROP TABLE {tabla}")
    ctx.general.execute(f"ALTER TABLE {tabla}_mig RENAME TO {tabla}")


def _delete_kpis_lineas_if_exists(ctx: StorageContext, dias, tabla="kpis_lineas"):
    if len(dias) == 0:
        return

    dias_str = _sql_in_values(dias)
    try:
        ctx.general.execute(f"DELETE FROM {tabla} WHERE dia IN ({dias_str})")
    except Exception as exc:
        if "does not exist" not in str(exc):
            raise


def cal_velocidad_comercial(servicios, entidad=None):
    """Velocidades comerciales y distancia media por vehículo.

    `entidad` son las columnas que definen la unidad de análisis:
    `["id_linea"]` (default) o `["id_linea", "id_ramal"]` para la vista por
    ramal. El cálculo es el mismo, solo cambia el nivel de agregación.
    """
    entidad = entidad or ["id_linea"]
    _g = ["dia"] + entidad
    # Conversión de columnas a datetime
    servicios["min_datetime"] = pd.to_datetime(servicios["min_datetime"])
    servicios["max_datetime"] = pd.to_datetime(servicios["max_datetime"])

    # Cálculo de duración del servicio en minutos
    servicios["diff_minutes"] = (
        servicios["max_datetime"] - servicios["min_datetime"]
    ).dt.total_seconds() / 60

    # Velocidad comercial por servicio, en dos familias homónimas al resto de
    # los KPIs de distancia:
    #   _route     -> ping-based   (distance_route,     reconstruida ping-a-ping)
    #   _route_gps -> odómetro     (distance_route_gps, distancia de servicio)
    horas = servicios["diff_minutes"] / 60
    servicios["velocidad_comercial_route"] = servicios["distance_route"] / horas
    servicios["velocidad_comercial_route_gps"] = servicios["distance_route_gps"] / horas

    _vc_cols = ["velocidad_comercial_route", "velocidad_comercial_route_gps"]
    _dist_cols = ["distance_route", "distance_route_gps"]

    # Extraer hora de finalización del servicio
    servicios["hour"] = servicios["max_datetime"].dt.hour

    filtro_pico_am = (servicios["diff_minutes"] < 180) & (
        servicios["hour"].between(6, 10)
    )

    # Velocidad comercial total (todo el día)
    vel_comercial_linea_all = (
        servicios.groupby(_g, as_index=False)[_vc_cols]
        .mean()
        .round(1)
    )

    # Velocidad comercial AM
    vel_comercial_linea_am = (
        servicios[filtro_pico_am]
        .groupby(_g, as_index=False)[_vc_cols]
        .mean()
        .round(1)
        .rename(columns={
            "velocidad_comercial_route": "velocidad_comercial_am_route",
            "velocidad_comercial_route_gps": "velocidad_comercial_am_route_gps",
        })
    )

    # Velocidad comercial PM (15 a 19 hs)
    filtro_pico_pm = (servicios["diff_minutes"] < 180) & (
        servicios["hour"].between(15, 19)
    )
    vel_comercial_linea_pm = (
        servicios[filtro_pico_pm]
        .groupby(_g, as_index=False)[_vc_cols]
        .mean()
        .round(1)
        .rename(columns={
            "velocidad_comercial_route": "velocidad_comercial_pm_route",
            "velocidad_comercial_route_gps": "velocidad_comercial_pm_route_gps",
        })
    )

    # Consolidar velocidades comerciales
    vel_comercial_linea = vel_comercial_linea_all.merge(
        vel_comercial_linea_am, how="left"
    ).merge(vel_comercial_linea_pm, how="left")

    # Distancia media recorrida por vehículo: primero km por vehículo, después
    # el promedio entre vehículos de la entidad
    km_recorridos_linea = (
        servicios.groupby(_g + ["interno"], as_index=False)[_dist_cols]
        .sum()
        .groupby(_g, as_index=False)[_dist_cols]
        .mean()
        .rename(columns={
            "distance_route": "distancia_media_veh_route",
            "distance_route_gps": "distancia_media_veh_route_gps",
        })
        .round(1)
    )

    vel_comercial_linea = vel_comercial_linea.merge(km_recorridos_linea, how="left")

    return vel_comercial_linea


def levanto_data(ctx: StorageContext, etapas=[], viajes=[], dias=None, entidad=None):
    """Insumos para `agrego_lineas`, agregados al nivel de `entidad`.

    `entidad` es `["id_linea"]` (default) o `["id_linea", "id_ramal"]`. En el
    segundo caso la flota, la velocidad comercial y los KPI de demanda se leen
    y agregan por ramal: los KPI vienen de `kpi_by_day_branch`, que escribe
    `compute_kpi_by_branch_day` cuando `lineas_contienen_ramales` es True.

    Devuelve `(internos_agg, gps_agg, servicios, kpis_varios, lineas)`: los
    dos primeros son los agregados de `transacciones` y `gps` ya cerrados
    (ver abajo), no los frames enteros.
    """
    entidad = entidad or ["id_linea"]
    por_ramal = "id_ramal" in entidad

    # `gps` y `transacciones` son las dos tablas mas grandes de la corrida.
    # Materializarlas enteras cuesta ~20 GB y ~10 GB a escala de un mes (mas la
    # columna `dia` que se deriva de `fecha`, otros 62 B/fila) y no entran en la
    # maquina del cliente. Lo unico que se les pide son tres agregados que
    # llevan `dia` en la clave -- la flota, los internos con GPS y los internos
    # con transacciones -- asi que se leen de a un dia y se combinan los
    # parciales. `dias` acota las lecturas (tablas acumulativas) a los dias del
    # proc-mat: las filas de salida de agrego_lineas se anclan en el mat, leer
    # mas dias es descarte.
    from urbantrips.preparo_dashboard.sql_queries import dias_where_clause

    _where = dias_where_clause(dias)

    lineas = ctx.insumos.get_metadata_lineas()[
        ["id_linea", "nombre_linea", "empresa"]
    ].drop_duplicates()

    _tabla_kpi = "kpi_by_day_branch" if por_ramal else "kpi_by_day_line"
    try:
        kpis = ctx.data.query(f"SELECT * FROM {_tabla_kpi}{_where}")
    except Exception:
        kpis = pd.DataFrame()

    _and = dias_where_clause(dias, prefix="AND")
    try:
        servicios = ctx.data.query(f"SELECT * FROM services WHERE valid = 1{_and}")
    except Exception:
        servicios = pd.DataFrame()

    cols = ["dia"] + entidad

    # Procesamiento de GPS y calculo de flota. El `dia` se deriva de `fecha`
    # (no es la columna `dia` de la tabla, por la que se chunkea), asi que un
    # grupo puede quedar partido entre dos chunks: combinar_* lo vuelve a unir.
    flota_parc, gps_parc = [], []
    for dia in dias_para_leer_por_dia(ctx.data.query, "gps", dias):
        dia_gps = leer_dia(
            ctx.data.query, "gps", ["fecha", "id_linea", "id_ramal", "interno"], dia,
        )
        dia_gps["fecha"] = pd.to_datetime(dia_gps["fecha"], unit="s")
        dia_gps["dia"] = dia_gps["fecha"].dt.strftime("%Y-%m-%d")
        flota_parc.append(dia_gps.groupby(cols, as_index=False).size())
        gps_parc.append(dia_gps.groupby(cols + ["interno"], as_index=False).size())
        del dia_gps

    flota = combinar_suma(flota_parc, cols, "size", observed=False).rename(
        columns={"size": "flota"}
    )
    gps_agg = combinar_conteo_distintos(
        gps_parc, cols, "cant_internos_en_gps", observed=False
    )
    del flota_parc, gps_parc

    # Procesamiento de transacciones
    internos_parc = []
    for dia in dias_para_leer_por_dia(ctx.data.query, "transacciones", dias):
        dia_trx = leer_dia(
            ctx.data.query, "transacciones",
            ["dia", "id_linea", "id_ramal", "interno"], dia,
        )
        internos_parc.append(dia_trx.groupby(cols + ["interno"], as_index=False).size())
        del dia_trx

    internos_agg = combinar_conteo_distintos(
        internos_parc, cols, "cant_internos_en_trx", observed=False
    )
    del internos_parc

    # Calculo de velocidad comercial
    vel_comercial_linea = cal_velocidad_comercial(servicios, entidad=entidad)

    kpis_varios = flota.merge(vel_comercial_linea, how="left").merge(kpis, how="left")

    return internos_agg, gps_agg, servicios, kpis_varios, lineas


@duracion
def agrego_lineas(cols, trx, etapas, gps, servicios, kpis_varios, lineas,
                  etapas_query_fn=None, etapas_source=None, etapas_cte_prefix="",
                  internos_agg=None, gps_agg=None):
    """`internos_agg`/`gps_agg` permiten pasar los agregados de `trx` y `gps` ya
    calculados (leidos de a un dia por `levanto_data`) en lugar de los frames
    enteros, que a escala de un mes no entran en RAM. Sin ellos se calculan aca
    como siempre.
    """

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
    if internos_agg is None:
        internos_agg = (
            trx.groupby(cols + ["interno"], as_index=False)
            .size()
            .groupby(cols, as_index=False)
            .size()
            .rename(columns={"size": "cant_internos_en_trx"})
        )

    # Agregado de cantidad de internos con GPS
    if gps_agg is None:
        gps_agg = (
            gps.groupby(cols + ["interno"], as_index=False)
            .size()
            .groupby(cols, as_index=False)
            .size()
            .rename(columns={"size": "cant_internos_en_gps"})
        )

    # Agregado de servicios válidos.
    # `distance_route_gps` (odómetro del equipo) se suma igual que
    # `distance_route` (recorrido reconstruido ping a ping) para poder guardar
    # `tot_km_route_gps`: sin ese total no hay forma de agregar correctamente
    # los ratios de la familia _route_gps (hay que ponderarlos por sus propios
    # km). Ojo: no todos los insumos traen odómetro — en AMBA 2026-05-14 llega
    # en 0,0 para los 142.494 servicios válidos, y entonces toda la familia
    # queda en cero.
    _serv_aggs = {"interno": "count", "distance_route": "sum", "min_ts": "sum"}
    _serv_ren = {
        "interno": "cant_servicios",
        "distance_route": "serv_distance_route",
        "min_ts": "serv_min_ts",
    }
    if "distance_route_gps" in servicios.columns:
        _serv_aggs["distance_route_gps"] = "sum"
        _serv_ren["distance_route_gps"] = "serv_distance_route_gps"

    serv_agg = (
        servicios[servicios.valid == 1]
        .groupby(cols, as_index=False)
        .agg(_serv_aggs)
        .rename(columns=_serv_ren)
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

    # tot_km_route: solo km de servicios con valid=1
    all["tot_km_route"] = all["serv_distance_route"]
    # idem para la familia _route_gps (odómetro). Si el insumo no lo trae, la
    # columna queda en NaN y toda su familia de indicadores se muestra vacía.
    # Tiene que ser una Serie float y no un escalar pd.NA: un pd.NA escalar deja
    # la columna en dtype object y el astype(float) de más abajo revienta.
    all["tot_km_route_gps"] = (
        all["serv_distance_route_gps"]
        if "serv_distance_route_gps" in all.columns
        else pd.Series(np.nan, index=all.index, dtype="float64")
    )

    # tot_veh sincronizado con vehiculos_operativos corregido
    all["tot_veh"] = all["vehiculos_operativos"]

    # Recalcular ratios que dependen de tot_veh y tot_km_route
    all["pvd"] = (all["tot_pax"] / all["tot_veh"].replace(0, pd.NA)).round(1)
    all["kvd_route"] = (all["tot_km_route"] / all["tot_veh"].replace(0, pd.NA)).round(1)
    all["ipk_route"] = (all["tot_pax"] / all["tot_km_route"].replace(0, pd.NA)).round(1)
    # los mismos ratios sobre los km del odómetro, para que las dos familias de
    # recorrido sean simétricas y comparables
    all["kvd_route_gps"] = (
        all["tot_km_route_gps"] / all["tot_veh"].replace(0, pd.NA)
    ).round(1)
    all["ipk_route_gps"] = (
        all["tot_pax"] / all["tot_km_route_gps"].replace(0, pd.NA)
    ).round(1)

    # `id_ramal` solo aparece cuando se está agregando por ramal (cols lo trae)
    _id_cols = ["dia", "mes", "id_linea"]
    if "id_ramal" in cols:
        _id_cols.append("id_ramal")
    _id_cols += ["nombre_linea", "empresa", "modo"]

    all = all[
        _id_cols
        + [
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
            "velocidad_comercial_route",
            "velocidad_comercial_route_gps",
            "velocidad_comercial_am_route",
            "velocidad_comercial_am_route_gps",
            "velocidad_comercial_pm_route",
            "velocidad_comercial_pm_route_gps",
            "distancia_media_veh_route",
            "distancia_media_veh_route_gps",
            "tot_km_route",
            "tot_km_route_gps",
            "distancia_media_pax",
            "dmt_mean_od",
            "dmt_mean_route",
            "dmt_mean_route_gps",
            "dmt_median_od",
            "dmt_median_route",
            "dmt_median_route_gps",
            "pvd",
            "kvd_route",
            "kvd_route_gps",
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

    # los identificadores van como texto (id_ramal incluido cuando está), el
    # resto a numérico. id_ramal llega como float por ser nullable: pasa por
    # Int64 para que no quede "1234.0".
    if "id_ramal" in all.columns:
        all["id_ramal"] = pd.to_numeric(all["id_ramal"], errors="coerce").astype("Int64")
    for i in _id_cols:
        all[i] = all[i].fillna("").astype(str)
    lista = [x for x in all.columns.tolist() if x not in _id_cols]
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
            # to_numeric y no astype(float): una columna que quedó entera en
            # nulos (p. ej. la familia _route_gps cuando el insumo no trae
            # odómetro) llega con dtype object y NAType, y astype(float) falla
            all[i] = pd.to_numeric(all[i], errors="coerce").astype(float).round(1)

    return all


@duracion
def calculo_kpi_lineas(ctx: StorageContext, etapas=[], viajes=[]):
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, ETAPAS_PROC_MAT, proc_mat_days,
    )
    materializar_proc_tables(ctx)

    dias_mat = proc_mat_days(ctx)
    internos_agg, gps_agg, servicios, kpis_varios, lineas = levanto_data(
        ctx, dias=dias_mat
    )
    kpis = agrego_lineas(
        ["dia", "id_linea"], None, None, None, servicios, kpis_varios, lineas,
        etapas_query_fn=ctx.data.query, etapas_source=ETAPAS_PROC_MAT,
        internos_agg=internos_agg, gps_agg=gps_agg,
    )

    # delete existing rows for these days AND the previous "Promedios" row before
    # re-reading. En corridas incrementales (--step dashboard) la fila "Promedios"
    # vieja sobrevivía, se re-leía, contaminaba la media nueva y se duplicaba al
    # re-appendear (3631 vs 3223 filas). Borrarla acá deja la lectura limpia.
    dias = kpis.dia.unique().tolist()
    _delete_kpis_lineas_if_exists(ctx, dias + ["Promedios"])
    _alinear_esquema_kpis_lineas(ctx, kpis)
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


@duracion
def calculo_kpi_ramales(ctx: StorageContext):
    """Igual que `calculo_kpi_lineas` pero abierto por ramal, en `kpis_ramales`.

    Solo se ejecuta si `lineas_contienen_ramales` está en True; con ramales
    ficticios la tabla sería una copia de la de líneas. Los KPI de demanda
    (dmt, fo, ipk, pvd) salen de `kpi_by_day_branch`, que escribe
    `compute_kpi_by_branch_day` bajo la misma condición: si esa tabla no está,
    esas columnas quedan vacías y se avisa.
    """
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, ETAPAS_PROC_MAT, proc_mat_days,
    )

    entidad = ["id_linea", "id_ramal"]
    materializar_proc_tables(ctx)
    dias_mat = proc_mat_days(ctx)

    internos_agg, gps_agg, servicios, kpis_varios, lineas = levanto_data(
        ctx, dias=dias_mat, entidad=entidad
    )
    if "id_ramal" not in kpis_varios.columns:
        logger.warning(
            "No hay KPI por ramal (falta kpi_by_day_branch); "
            "kpis_ramales va a quedar sin los indicadores de demanda"
        )

    kpis = agrego_lineas(
        ["dia"] + entidad, None, None, None, servicios, kpis_varios, lineas,
        etapas_query_fn=ctx.data.query, etapas_source=ETAPAS_PROC_MAT,
        internos_agg=internos_agg, gps_agg=gps_agg,
    )

    # nombre del ramal, si la metadata lo tiene
    try:
        ramales = ctx.insumos.get_metadata_ramales()
        if "nombre_ramal" in ramales.columns:
            ramales = ramales[["id_ramal", "nombre_ramal"]].drop_duplicates("id_ramal")
            ramales["id_ramal"] = (
                pd.to_numeric(ramales["id_ramal"], errors="coerce")
                .astype("Int64").astype(str)
            )
            kpis = kpis.merge(ramales, on="id_ramal", how="left")
            kpis["nombre_ramal"] = kpis["nombre_ramal"].fillna("")
    except Exception as exc:
        logger.info("Sin metadata de ramales (%s); se sigue sin nombre_ramal", exc)

    dias = kpis.dia.unique().tolist()
    _delete_kpis_lineas_if_exists(ctx, dias + ["Promedios"], tabla="kpis_ramales")
    _alinear_esquema_kpis_lineas(ctx, kpis, tabla="kpis_ramales")
    ctx.general.append_raw(kpis, "kpis_ramales")

    df = ctx.general.get_raw("kpis_ramales")
    _claves = [c for c in ["id_linea", "id_ramal", "nombre_linea", "nombre_ramal",
                           "empresa", "modo"] if c in df.columns]
    tot = df.drop(["dia", "mes"], axis=1).groupby(_claves, as_index=False).mean()
    tot["dia"] = "Promedios"
    tot["mes"] = ""
    df = pd.concat([df, tot], ignore_index=True)

    all_dias = df.dia.unique().tolist()
    _delete_kpis_lineas_if_exists(ctx, all_dias, tabla="kpis_ramales")
    ctx.general.append_raw(df, "kpis_ramales")

    return df
