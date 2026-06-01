"""
Pure DataFrame aggregation and classification helpers for the dashboard preparation layer.
No DB access — all functions take DataFrames and return DataFrames.
"""
import logging

import pandas as pd
import unidecode

from urbantrips.utils.utils import calculate_weighted_means

logger = logging.getLogger(__name__)


def clasificar_tarifa_agregada_social(serie_tarifa_agregada):
    """
    Clasifica categorías de tarifa_agregada social en:
    - 'educacion_jubilacion': incluye escolar, jubilado, pensionado, etc.
    - 'tarifa_social': otras categorías con descuento.
    - 'sin_descuento': valores nulos o vacíos.
    """
    def normalizar(texto):
        if pd.isna(texto):
            return ""
        return unidecode.unidecode(str(texto).strip().lower())

    def clasificar(valor):
        val = normalizar(valor)
        if val in {"", "-"}:
            return "sin_descuento"
        if any(palabra in val for palabra in ["jubilad", "pensionad", "escolar"]):
            return "educacion_jubilacion"
        return "tarifa_social"

    return serie_tarifa_agregada.apply(clasificar)


def clasificar_genero_agregado(serie_genero_agregado):
    def normalizar(val):
        if pd.isna(val):
            return ""
        return str(val).strip().lower()

    def mapear(val):
        val = normalizar(val)
        if val in ["m", "masculino", "varón", "varon", "hombre"]:
            return "Masculino"
        elif val in ["f", "femenino", "mujer"]:
            return "Femenino"
        else:
            return "No informado"

    return serie_genero_agregado.apply(mapear)


def format_values(row):
    if row["type_val"] == "int":
        return f"{int(row['Valor']):,}".replace(",", ".")
    elif row["type_val"] == "float":
        return (
            f"{row['Valor']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
    elif row["type_val"] == "percentage":
        return f"{row['Valor']:.2f}%".replace(".", ",")
    else:
        return str(row["Valor"])


def format_dataframe(df):
    df["Valor_str"] = df.apply(format_values, axis=1)
    return df


def determinar_modo_agregado(grupo):
    modos_unicos = grupo["modo"].unique()
    if len(modos_unicos) == 1:
        if len(grupo) > 1:
            return f"multietapa ({modos_unicos[0]})"
        else:
            return modos_unicos[0]
    else:
        return "multimodal"


def _sql_in_values(values):
    return "', '".join(str(value).replace("'", "''") for value in values)


def agrupar_viajes(
    etapas_agrupadas,
    aggregate_cols,
    weighted_mean_cols,
    weight_col,
    zero_to_nan,
    agg_transferencias=False,
    agg_modo=False,
    agg_hora=False,
    agg_distancia=False,
    agg_genero_agregado=False,
    agg_tarifa_agregada=False,
):
    etapas_agrupadas_zon = etapas_agrupadas.copy()

    if agg_transferencias:
        etapas_agrupadas_zon["transferencia"] = 99
    if agg_modo:
        etapas_agrupadas_zon["modo_agregado"] = 99
    if agg_hora:
        etapas_agrupadas_zon["rango_hora"] = 99
    if agg_distancia:
        etapas_agrupadas_zon["distancia_agregada"] = 99
    if agg_genero_agregado:
        etapas_agrupadas_zon["genero_agregado"] = 99
    if agg_tarifa_agregada:
        etapas_agrupadas_zon["tarifa_agregada"] = 99

    etapas_agrupadas_zon = calculate_weighted_means(
        etapas_agrupadas_zon,
        aggregate_cols=aggregate_cols,
        weighted_mean_cols=weighted_mean_cols,
        weight_col=weight_col,
        zero_to_nan=zero_to_nan,
    )
    return etapas_agrupadas_zon


def construyo_matrices(
    etapas_desagrupadas,
    aggregate_cols,
    zonificaciones,
    agg_transferencias=False,
    agg_modo=False,
    agg_hora=False,
    agg_distancia=False,
    agg_genero_agregado=False,
    agg_tarifa_agregada=False,
):
    matriz = etapas_desagrupadas.copy()

    if agg_transferencias:
        matriz["transferencia"] = 99
    if agg_modo:
        matriz["modo_agregado"] = 99
    if agg_hora:
        matriz["rango_hora"] = 99
    if agg_distancia:
        matriz["distancia_agregada"] = 99
    if agg_genero_agregado:
        matriz["genero_agregado"] = 99
    if agg_tarifa_agregada:
        matriz["tarifa_agregada"] = 99

    matriz = calculate_weighted_means(
        matriz,
        aggregate_cols=aggregate_cols,
        weighted_mean_cols=[
            "lat1", "lon1", "lat4", "lon4",
            "distance_od", "travel_time_min", "kmh_od",
        ],
        weight_col="factor_expansion_linea",
        zero_to_nan=["lat1", "lon1", "lat4", "lon4", "travel_time_min", "kmh_od"],
    )

    zonificaciones["orden"] = zonificaciones["orden"].fillna(0)
    matriz = matriz.merge(
        zonificaciones[["zona", "id", "orden"]].rename(
            columns={"id": "inicio", "orden": "orden_origen"}
        ),
        on=["zona", "inicio"],
    )
    matriz = matriz.merge(
        zonificaciones[["zona", "id", "orden"]].rename(
            columns={"id": "fin", "orden": "orden_destino"}
        ),
        on=["zona", "fin"],
    )
    matriz["Origen"] = (
        matriz.orden_origen.astype(int).astype(str).str.zfill(3) + "_" + matriz.inicio
    )
    matriz["Destino"] = (
        matriz.orden_destino.astype(int).astype(str).str.zfill(3) + "_" + matriz.fin
    )
    return matriz


def agg_matriz(
    df,
    aggregate_cols=None,
    weight_col=None,
    weight_var="factor_expansion_linea",
    agg_transferencias=False,
    agg_modo=False,
    agg_hora=False,
    agg_distancia=False,
):
    if aggregate_cols is None:
        aggregate_cols = [
            "id_polygon", "zona", "Origen", "Destino",
            "transferencia", "modo_agregado", "rango_hora", "distancia_agregada",
        ]
    if weight_col is None:
        weight_col = ["distance_od", "travel_time_min", "kmh_od"]

    if len(df) > 0:
        if agg_transferencias:
            df["transferencia"] = 99
        if agg_modo:
            df["modo_agregado"] = 99
        if agg_hora:
            df["rango_hora"] = 99
        if agg_distancia:
            df["distancia_agregada"] = 99

        df1 = df.groupby(aggregate_cols, as_index=False, observed=True)[weight_var].sum()
        df2 = calculate_weighted_means(
            df,
            aggregate_cols=aggregate_cols,
            weighted_mean_cols=weight_col,
            weight_col=weight_var,
        )
        df = df1.merge(df2)
    return df


def agrego_lineas(cols, trx, etapas, gps, servicios, kpis, lineas):
    trx_agg = (
        trx.groupby(cols + ["modo"], as_index=False, observed=True)
        .factor_expansion.sum()
        .rename(columns={"factor_expansion": "transacciones"})
    )
    lineas_agg = lineas[["id_linea", "nombre_linea", "empresa"]].drop_duplicates()
    etapas_agg = (
        calculate_weighted_means(
            etapas,
            aggregate_cols=cols + ["modo"],
            weighted_mean_cols=["distance_od", "travel_time_min", "kmh_od"],
            zero_to_nan=["distance_od", "travel_time_min", "kmh_od"],
            weight_col="factor_expansion_linea",
            var_fex_summed=False,
        )
        .round(2)
        .rename(columns={"modo": "modo_new"})
        .rename(columns={"distance_od": "distancia_media"})
    )
    internos_agg = (
        trx.groupby(cols + ["interno"], as_index=False, observed=True)
        .size()
        .groupby(cols, as_index=False, observed=True)
        .size()
        .rename(columns={"size": "cant_internos_en_trx"})
    )
    gps_agg = (
        gps.groupby(cols + ["interno"], as_index=False, observed=True)
        .size()
        .groupby(cols, as_index=False, observed=True)
        .size()
        .rename(columns={"size": "cant_internos_en_gps"})
    )
    serv_agg = (
        servicios[servicios.valid == 1]
        .groupby(cols, as_index=False, observed=True)
        .agg({"interno": "count", "distance_route": "sum", "min_ts": "sum"})
        .rename(columns={
            "interno": "cant_servicios",
            "distance_route": "serv_distance_route",
            "min_ts": "serv_min_ts",
        })
    )

    all = (
        trx_agg.merge(etapas_agg, how="left")
        .merge(internos_agg, how="left")
        .merge(gps_agg, how="left")
        .merge(kpis, how="left")
        .merge(lineas_agg, how="left")
        .merge(serv_agg, how="left")
        .round(2)
    )
    all = all[
        cols + [
            "nombre_linea", "empresa", "modo", "transacciones",
            "distancia_media", "travel_time_min", "kmh_od",
            "cant_internos_en_trx", "cant_internos_en_gps",
            "tot_veh", "tot_km", "tot_pax", "dmt_mean_od", "dmt_median_od",
            "pvd", "kvd", "ipk_route", "fo_mean_od", "fo_median_od",
        ]
    ]
    all["transacciones"] = all["transacciones"].round(0)
    all["tot_pax"] = all["tot_pax"].round(0)
    return all
