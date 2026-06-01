import logging
import gc
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import unidecode
from shapely import wkt
from shapely.geometry import MultiPolygon, Point

from urbantrips.carto import carto
from urbantrips.carto.carto import guardo_zonificaciones
from urbantrips.geo.geo import (
    create_h3_gdf,
    h3_to_geodataframe,
    h3_to_lat_lon,
    h3toparent,
    normalizo_lat_lon,
    point_to_h3,
)
from urbantrips.kpi.kpi_lineas import calculo_kpi_lineas
from urbantrips.preparo_dashboard.aggregation import (  # noqa: F401 — re-exported
    agg_matriz,
    agrego_lineas,
    agrupar_viajes,
    clasificar_genero_agregado,
    clasificar_tarifa_agregada_social,
    construyo_matrices,
    determinar_modo_agregado,
    format_dataframe,
    format_values,
    _sql_in_values,
)
from urbantrips.preparo_dashboard.geo import (  # noqa: F401 — re-exported
    creo_h3_equivalencias,
    ensure_geodataframe,
    fix_mixed_polygons,
    normalizo_zona,
    select_cases_from_polygons,
    select_h3_from_polygon,
)
from urbantrips.storage.context import StorageContext
from urbantrips.utils.check_configs import check_config
from urbantrips.utils.utils import calculate_weighted_means, duracion, leer_alias, leer_configs_generales

logger = logging.getLogger(__name__)

pd.set_option("future.no_silent_downcasting", True)

import warnings

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
    module=r".*urbantrips.*preparo_dashboard",
)




def load_and_process_data(ctx: StorageContext):
    """
    Devuelve los DataFrames `etapas` y `viajes` procesados
    sin alterar la lógica de negocio pero con pasos internos
    más rápidos y ordenados.
    """

    # ── 1. leer etapas y viajes (filtrados) ─────────────────────────
    etapas = ctx.data.query(
        """
        SELECT e.id, e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa, e.tiempo, e.hora,
               e.modo, e.id_linea, e.id_ramal, e.interno, e.genero, e.tarifa,
               e.latitud, e.longitud, e.h3_o, e.h3_d, e.od_validado,
               e.factor_expansion_original, e.factor_expansion_linea,
               e.factor_expansion_tarjeta,
               tt.travel_time_min, tt.distance_od, tt.distance_route,
               tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
        FROM etapas e
        LEFT JOIN travel_times_legs tt ON e.id = tt.id
        WHERE e.od_validado = 1
        """
    )

    viajes = ctx.data.query(
        """
        SELECT v.*, tt.travel_time_min, tt.distance_od, tt.distance_route,
               tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
        FROM viajes v
        LEFT JOIN travel_times_trips tt
        ON v.dia = tt.dia
        AND v.id_tarjeta = tt.id_tarjeta
        AND v.id_viaje = tt.id_viaje
        WHERE v.od_validado = 1
        """
    )

    etapas["tarifa_agregada"] = clasificar_tarifa_agregada_social(etapas["tarifa"])
    etapas["genero_agregado"] = clasificar_genero_agregado(etapas["genero"])
    viajes["tarifa_agregada"] = clasificar_tarifa_agregada_social(viajes["tarifa"])
    viajes["genero_agregado"] = clasificar_genero_agregado(viajes["genero"])

    # ── 2. incorporar travel_time_min y velocidades ─────────────────

    etapas[["kmh_od"]] = np.nan
    viajes[["kmh_od"]] = np.nan

    etapas["travel_time_min"] = (
        pd.to_numeric(etapas["travel_time_min"], errors="coerce").fillna(0).astype(int)
    )

    etapas["kmh_od"] = np.where(
        etapas["travel_time_min"] > 0,
        (etapas["distance_od"] / (etapas["travel_time_min"] / 60)).round(1),
        np.nan,
    )

    viajes["kmh_od"] = np.where(
        viajes["travel_time_min"] > 0,
        (viajes["distance_od"] / (viajes["travel_time_min"] / 60)).round(1),
        np.nan,
    )
    viajes.loc[viajes["kmh_od"] >= 80, "kmh_od"] = np.nan

    # ── 3. flags y rangos horarios (vectorizado) ────────────────────
    viajes["transferencia"] = (viajes["cant_etapas"] > 1).astype(int)

    cond_rh = [viajes["hora"].between(13, 16), viajes["hora"].between(17, 24)]
    viajes["rango_hora"] = np.select(cond_rh, ["13-16", "17-24"], default="0-12")

    viajes["distancia_agregada"] = np.where(
        viajes["distance_od"] > 5, "Viajes largos (>5kms)", "Viajes cortos (<=5kms)"
    )
    etapas["distancia_agregada"] = np.where(
        etapas["distance_od"] > 5, "Etapa larga (>5kms)", "Etapa corta (<=5kms)"
    )

    viajes["tipo_dia"] = np.where(
        pd.to_datetime(viajes["dia"]).dt.dayofweek >= 5, "Fin de Semana", "Hábil"
    )

    viajes["mes"] = pd.to_datetime(viajes["dia"]).dt.to_period("M").astype(str)

    viajes["Fecha"] = pd.to_datetime(viajes["dia"] + " " + viajes["tiempo"])
    viajes["Fecha_next"] = viajes.groupby(["dia", "id_tarjeta"])["Fecha"].shift(-1)
    viajes["diff_time"] = (
        (viajes["Fecha_next"] - viajes["Fecha"]).dt.seconds / 60
    ).round()

    # mismas transformaciones para etapas
    etapas["tipo_dia"] = np.where(
        pd.to_datetime(etapas["dia"]).dt.dayofweek >= 5, "Fin de Semana", "Hábil"
    )
    etapas["mes"] = pd.to_datetime(etapas["dia"]).dt.to_period("M").astype(str)

    cond_rh_e = [etapas["hora"].between(13, 16), etapas["hora"].between(17, 24)]
    etapas["rango_hora"] = np.select(cond_rh_e, ["13-16", "17-24"], default="0-12")

    etapas = etapas.merge(
        viajes[["dia", "id_tarjeta", "id_viaje", "transferencia"]], how="left"
    )

    # ── 4. partición modal (vectorizada) ────────────────────────────

    keys_mod = ["dia", "id_tarjeta", "id_viaje"]
    tmp = etapas.groupby(keys_mod, sort=False).agg(
        n_unique=("modo", "nunique"),
        etapas_tot=("modo", "size"),
        modo_ref=("modo", "first"),
    )
    tmp["modo_agregado"] = np.where(
        tmp["n_unique"] == 1,
        np.where(
            tmp["etapas_tot"] > 1,
            "multietapa (" + tmp["modo_ref"] + ")",
            tmp["modo_ref"],
        ),
        "multimodal",
    )
    etapas = etapas.merge(tmp["modo_agregado"].reset_index(), on=keys_mod)
    viajes = viajes.merge(
        etapas.groupby(keys_mod + ["modo_agregado"], as_index=False)
        .size()
        .drop(columns="size"),
        how="left",
    )

    # ── 5. eliminar registros sin distancias ────────────────────────
    etapas = etapas[etapas["distance_od"].notna()]
    viajes = viajes[viajes["distance_od"].notna()]

    # ── 6. rellenar nulos finales ───────────────────────────────────
    for df in (etapas, viajes):
        df["travel_time_min"] = df["travel_time_min"].fillna(0).astype("float32")
        df["kmh_od"] = df["kmh_od"].fillna(0).astype("float32")

    # ── 7. columnas finales ─────────────────────────────────────────
    etapas = etapas[
        [
            "id",
            "dia",
            "mes",
            "tipo_dia",
            "id_tarjeta",
            "id_viaje",
            "id_etapa",
            "tiempo",
            "hora",
            "modo",
            "id_linea",
            "id_ramal",
            "interno",
            "h3_o",
            "h3_d",
            "latitud",
            "longitud",
            "od_validado",
            "factor_expansion_original",
            "factor_expansion_linea",
            "factor_expansion_tarjeta",
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            "transferencia",
            "genero_agregado",
            "tarifa_agregada",
        ]
    ]

    viajes = viajes[
        [
            "dia",
            "mes",
            "tipo_dia",
            "id_tarjeta",
            "id_viaje",
            "Fecha",
            "tiempo",
            "hora",
            "cant_etapas",
            "modo",
            "autobus",
            "tren",
            "metro",
            "tranvia",
            "brt",
            "cable",
            "lancha",
            "otros",
            "h3_o",
            "h3_d",
            "od_validado",
            "factor_expansion_linea",
            "factor_expansion_tarjeta",
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "diff_time",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
            "transferencia",
            "genero_agregado",
            "tarifa_agregada",
        ]
    ]

    return etapas, viajes




def construyo_indicadores(ctx: StorageContext, viajes, poligonos=False):

    if poligonos:
        nombre_tabla = "poly_indicadores"
    else:
        nombre_tabla = "agg_indicadores"

    if "id_polygon" not in viajes.columns:
        viajes["id_polygon"] = "NONE"

    ind1 = (
        viajes.groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False)
        .Valor.mean()
        .round()
    )
    ind1["Indicador"] = "Cantidad de Viajes"
    ind1["Valor"] = ind1.Valor.astype(int)
    ind1["Tipo"] = "General"
    ind1["type_val"] = "int"

    ind2 = (
        viajes[viajes.transferencia == 1]
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False)
        .Valor.mean()
        .round()
    )
    ind2["Indicador"] = "Cantidad de Viajes con Transferencia"
    ind2 = ind2.merge(
        ind1[["id_polygon", "dia", "mes", "tipo_dia", "Valor"]].rename(
            columns={"Valor": "Tot"}
        ),
        how="left",
    )
    ind2["Valor"] = (ind2["Valor"] / ind2["Tot"] * 100).round(2)
    ind2["Tipo"] = "General"
    ind2["type_val"] = "percentage"

    ind3 = (
        viajes.groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "rango_hora"], as_index=False
        )
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia", "rango_hora"], as_index=False)
        .Valor.mean()
        .round()
    )
    ind3["Indicador"] = "Cantidad de Según Rango Horas"
    ind3["Tot"] = ind3.groupby(
        ["id_polygon", "dia", "mes", "tipo_dia"]
    ).Valor.transform("sum")
    ind3["Valor"] = (ind3["Valor"] / ind3["Tot"] * 100).round(2)
    ind3["Indicador"] = "Cantidad de Viajes de " + ind3["rango_hora"] + "hs"
    ind3["Tipo"] = "General"
    ind3["type_val"] = "percentage"

    ind4 = (
        viajes.groupby(["id_polygon", "dia", "mes", "tipo_dia", "modo"], as_index=False)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia", "modo"], as_index=False)
        .Valor.mean()
        .round()
    )
    ind4["Indicador"] = "Partición Modal"
    ind4["Tot"] = ind4.groupby(
        ["id_polygon", "dia", "mes", "tipo_dia"]
    ).Valor.transform("sum")
    ind4["Valor"] = (ind4["Valor"] / ind4["Tot"] * 100).round(2)
    ind4 = ind4.sort_values(["id_polygon", "Valor"], ascending=False)
    ind4["Indicador"] = ind4["modo"]
    ind4["Tipo"] = "Modal"
    ind4["type_val"] = "percentage"

    ind9 = (
        viajes.groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "distancia_agregada"],
            as_index=False,
        )
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "distancia_agregada"],
            as_index=False,
        )
        .Valor.mean()
        .round()
    )
    ind9["Indicador"] = "Partición Modal"
    ind9["Tot"] = ind9.groupby(
        ["id_polygon", "dia", "mes", "tipo_dia"]
    ).Valor.transform("sum")
    ind9["Valor"] = (ind9["Valor"] / ind9["Tot"] * 100).round(2)
    ind9 = ind9.sort_values(["id_polygon", "Valor"], ascending=False)
    ind9["Indicador"] = "Cantidad de " + ind9["distancia_agregada"]
    ind9["Tipo"] = "General"
    ind9["type_val"] = "percentage"

    ind5 = (
        viajes.groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "id_tarjeta"], as_index=False
        )
        .factor_expansion_linea.first()
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False)
        .factor_expansion_linea.sum()
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False)
        .factor_expansion_linea.mean()
        .round()
        .rename(columns={"factor_expansion_linea": "Valor"})
    )
    ind5["Indicador"] = "Cantidad de Usuarios"
    ind5["Tipo"] = "General"
    ind5["type_val"] = "int"

    ind6 = (
        calculate_weighted_means(
            viajes,
            aggregate_cols=["id_polygon", "dia", "mes", "tipo_dia"],
            weighted_mean_cols=["distance_od"],
            weight_col="factor_expansion_linea",
        )
        .rename(columns={"distance_od": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False)
        .Valor.mean()
        .round(2)
    )
    ind6["Tipo"] = "Distancias"
    ind6["Indicador"] = "Distancia Promedio (kms)"
    ind6["type_val"] = "float"

    ind7 = (
        calculate_weighted_means(
            viajes,
            aggregate_cols=["id_polygon", "dia", "mes", "tipo_dia", "modo"],
            weighted_mean_cols=["distance_od"],
            weight_col="factor_expansion_linea",
        )
        .rename(columns={"distance_od": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia", "modo"], as_index=False)
        .Valor.mean()
        .round(2)
    )
    ind7["Tipo"] = "Distancias"
    ind7["Indicador"] = "Distancia Promedio (" + ind7.modo + ") (kms)"
    ind7["type_val"] = "float"

    ind8 = (
        calculate_weighted_means(
            viajes,
            aggregate_cols=[
                "id_polygon",
                "dia",
                "mes",
                "tipo_dia",
                "distancia_agregada",
            ],
            weighted_mean_cols=["distance_od"],
            weight_col="factor_expansion_linea",
        )
        .rename(columns={"distance_od": "Valor"})
        .groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "distancia_agregada"],
            as_index=False,
        )
        .Valor.mean()
        .round(2)
    )
    ind8["Tipo"] = "Distancias"
    ind8["Indicador"] = "Distancia Promedio " + ind8.distancia_agregada
    ind8["type_val"] = "float"

    indicadores = pd.concat([ind1, ind5, ind2, ind3, ind6, ind9, ind7, ind8, ind4])

    tupla_dia = tuple(indicadores.dia.unique().tolist() + ["Todos"])
    if len(tupla_dia) == 1:
        query = f"""
            SELECT *
            FROM {nombre_tabla}
            WHERE dia != '{tupla_dia[0]}'
        """
    else:
        query = f"""
            SELECT *
            FROM {nombre_tabla}
            WHERE dia NOT IN {tupla_dia}
        """

    try:
        indicadores_ant = ctx.dash.get_raw(nombre_tabla)
        if len(indicadores_ant) > 0:
            indicadores_ant = indicadores_ant[
                ~indicadores_ant.dia.isin(indicadores.dia.unique().tolist() + ["Todos"])
            ]
    except Exception:
        indicadores_ant = pd.DataFrame([])

    indicadores = pd.concat(
        [
            indicadores[
                [
                    "id_polygon",
                    "dia",
                    "mes",
                    "tipo_dia",
                    "Tipo",
                    "Indicador",
                    "type_val",
                    "Valor",
                ]
            ],
            indicadores_ant,
        ],
        ignore_index=True,
    )

    indicadores_todos = (
        indicadores.groupby(
            ["id_polygon", "Tipo", "Indicador", "type_val"], as_index=False
        )
        .Valor.mean()
        .round(2)
    )
    indicadores_todos["dia"] = "Todos"
    indicadores_todos["tipo_dia"] = ""
    indicadores_todos["mes"] = ""
    indicadores = pd.concat([indicadores, indicadores_todos])

    indicadores = format_dataframe(indicadores)
    indicadores = indicadores[
        ["id_polygon", "dia", "mes", "tipo_dia", "Tipo", "Indicador", "Valor_str"]
    ].rename(columns={"Valor_str": "Valor"})

    indicadores = indicadores.sort_values(
        ["id_polygon", "dia", "mes", "tipo_dia", "Tipo", "Indicador"]
    )

    if poligonos:
        tabla_destino = "poly_indicadores"
    else:
        tabla_destino = "agg_indicadores"

    replace_dash_partition(ctx, indicadores, tabla_destino, ["dia"])






def replace_dash_partition(ctx: StorageContext, df, table_name, partition_cols):
    if len(df) == 0:
        return

    filters = []
    for col in partition_cols:
        values = _sql_in_values(df[col].dropna().unique().tolist())
        filters.append(f"{col} IN ('{values}')")

    try:
        ctx.dash.execute(f"DELETE FROM {table_name} WHERE {' AND '.join(filters)}")
    except Exception as exc:
        if "does not exist" not in str(exc):
            raise

    ctx.dash.append_raw(df, table_name)






def imprimo_matrices_od(ctx: StorageContext):
    logger.info("Imprimo matrices OD")
    alias = leer_alias()

    matrices_all = ctx.dash.query(
        "SELECT id_polygon, tipo_dia, zona, inicio, fin, transferencia, "
        "modo_agregado, rango_hora, genero_agregado, tarifa_agregada, "
        "distancia_agregada, orden_origen, orden_destino, Origen, Destino, "
        "lat1, lon1, lat4, lon4, distance_od, travel_time_min, kmh_od, "
        "factor_expansion_linea, dia "
        "FROM agg_matrices"
    )

    agg_transferencias = True
    agg_modo = True
    agg_hora = True
    agg_distancia = True

    matrices_all.loc[matrices_all.travel_time_min == 0, "travel_time_min"] = np.nan
    matrices_all.loc[matrices_all.kmh_od == 0, "kmh_od"] = np.nan
    matrices = (
        matrices_all.groupby(
            [
                "id_polygon",
                "tipo_dia",
                "zona",
                "inicio",
                "fin",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "distancia_agregada",
                "orden_origen",
                "orden_destino",
                "Origen",
                "Destino",
            ],
            as_index=False,
        )[
            [
                "lat1",
                "lon1",
                "lat4",
                "lon4",
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "factor_expansion_linea",
            ]
        ]
        .mean()
        .round(2)
    )

    matriz_ = agg_matriz(
        matrices,
        aggregate_cols=[
            "id_polygon",
            "zona",
            "Origen",
            "Destino",
            "transferencia",
            "modo_agregado",
            "rango_hora",
            "distancia_agregada",
        ],
        weight_col=["distance_od", "travel_time_min", "kmh_od"],
        weight_var="factor_expansion_linea",
        agg_transferencias=agg_transferencias,
        agg_modo=agg_modo,
        agg_hora=agg_hora,
        agg_distancia=agg_distancia,
    )

    for var_zona in matriz_.zona.unique():

        savefile = f"{alias}matriz_{var_zona}"

        matriz = matriz_[matriz_.zona == var_zona]
        var_matriz = "factor_expansion_linea"
        normalize = False

        od_heatmap = pd.crosstab(
            index=matriz["Origen"],
            columns=matriz["Destino"],
            values=matriz[var_matriz],
            aggfunc="sum",
            normalize=False,
        )

        od_heatmap = od_heatmap.reset_index()
        od_heatmap["Origen"] = od_heatmap["Origen"].str[4:]
        od_heatmap = od_heatmap.set_index("Origen")
        od_heatmap.columns = [i[4:] for i in od_heatmap.columns]

        db_path = os.path.join("resultados", "matrices", f"{savefile}.xlsx")
        od_heatmap.reset_index().fillna("").to_excel(db_path, index=False)

        od_heatmap = pd.crosstab(
            index=matriz["Origen"],
            columns=matriz["Destino"],
            values=matriz[var_matriz],
            aggfunc="sum",
            normalize=True,
        )

        od_heatmap = od_heatmap.reset_index()
        od_heatmap["Origen"] = od_heatmap["Origen"].str[4:]
        od_heatmap = od_heatmap.set_index("Origen")
        od_heatmap.columns = [i[4:] for i in od_heatmap.columns]

        db_path2 = os.path.join(
            "resultados", "matrices", f"{savefile}_normalizada.xlsx"
        )
        od_heatmap.reset_index().fillna("").to_excel(db_path2, index=False)

        logger.debug("Saved %s --- %s", db_path, db_path2)


def crea_socio_indicadores(ctx: StorageContext, etapas, viajes):
    logger.info("Creo indicadores de género y tarifa_agregada")

    socio_indicadores = pd.DataFrame([])
    viajes.loc[viajes.travel_time_min == 0, "travel_time_min"] = np.nan
    viajes.loc[viajes.kmh_od == 0, "kmh_od"] = np.nan
    etapas.loc[etapas.travel_time_min == 0, "travel_time_min"] = np.nan
    etapas.loc[etapas.kmh_od == 0, "kmh_od"] = np.nan

    viajesx = calculate_weighted_means(
        viajes,
        aggregate_cols=["dia", "mes", "tipo_dia", "genero_agregado", "tarifa_agregada"],
        weighted_mean_cols=[
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "cant_etapas",
            "diff_time",
        ],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
    )

    viajesx = calculate_weighted_means(
        viajesx,
        aggregate_cols=["dia", "mes", "tipo_dia", "genero_agregado", "tarifa_agregada"],
        weighted_mean_cols=[
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "cant_etapas",
            "diff_time",
        ],
        weight_col="factor_expansion_linea",
        var_fex_summed=False,
    ).round(3)

    etapasx = calculate_weighted_means(
        etapas,
        aggregate_cols=[
            "dia",
            "mes",
            "tipo_dia",
            "genero_agregado",
            "tarifa_agregada",
            "modo",
        ],
        weighted_mean_cols=["distance_od", "travel_time_min", "kmh_od"],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
    )

    etapasx = calculate_weighted_means(
        etapasx,
        aggregate_cols=[
            "dia",
            "mes",
            "tipo_dia",
            "genero_agregado",
            "tarifa_agregada",
            "modo",
        ],
        weighted_mean_cols=["distance_od", "travel_time_min", "kmh_od"],
        weight_col="factor_expansion_linea",
        var_fex_summed=False,
    ).round(3)

    # calcular tabla de indicadores
    etapasxx = calculate_weighted_means(
        etapasx,
        aggregate_cols=["dia", "mes", "tipo_dia", "genero_agregado", "modo"],
        weighted_mean_cols=["distance_od", "travel_time_min", "kmh_od"],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
    ).round(3)

    etapasxx["tabla"] = "etapas-genero_agregado-modo"
    socio_indicadores = pd.concat([socio_indicadores, etapasxx], ignore_index=True)

    etapasxx = calculate_weighted_means(
        etapasx,
        aggregate_cols=["dia", "mes", "tipo_dia", "tarifa_agregada", "modo"],
        weighted_mean_cols=["distance_od", "travel_time_min", "kmh_od"],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
    ).round(3)

    etapasxx["tabla"] = "etapas-tarifa_agregada-modo"
    socio_indicadores = pd.concat([socio_indicadores, etapasxx], ignore_index=True)

    viajesxx = calculate_weighted_means(
        viajesx,
        aggregate_cols=["dia", "mes", "tipo_dia", "genero_agregado", "tarifa_agregada"],
        weighted_mean_cols=[
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "cant_etapas",
            "diff_time",
        ],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
    ).round(3)

    viajesxx["tabla"] = "viajes-genero_agregado-tarifa_agregada"
    socio_indicadores = pd.concat([socio_indicadores, viajesxx], ignore_index=True)

    # Calculo viajes promedio por día por género y tarifa_agregada
    userx = viajes.copy()
    userx["tarifa_agregada"] = userx["tarifa_agregada"].str.replace("-", "")
    userx = (
        userx.groupby(["dia", "id_tarjeta"])["tarifa_agregada"]
        .apply(lambda x: "-".join(x.unique()))
        .reset_index()
    )
    userx.loc[userx.tarifa_agregada.str[-1] == "-", "tarifa_agregada"] = userx.loc[
        userx.tarifa_agregada.str[-1] == "-", :
    ].tarifa_agregada.str[:-1]
    userx.loc[userx.tarifa_agregada.str[:1] == "-", "tarifa_agregada"] = userx.loc[
        userx.tarifa_agregada.str[:1] == "-", :
    ].tarifa_agregada.str[1:]
    userx = userx.rename(columns={"tarifa_agregada": "tarifa_agregada_agg"})
    userx.loc[userx.tarifa_agregada_agg == "", "tarifa_agregada_agg"] = "-"
    userx = viajes.merge(userx, how="left")
    userx = (
        userx.groupby(
            [
                "dia",
                "mes",
                "tipo_dia",
                "id_tarjeta",
                "genero_agregado",
                "tarifa_agregada_agg",
            ],
            as_index=False,
        )
        .agg({"factor_expansion_tarjeta": "count", "factor_expansion_linea": "mean"})
        .rename(columns={"factor_expansion_tarjeta": "cant_viajes"})
        .rename(columns={"tarifa_agregada_agg": "tarifa_agregada"})
    )

    userx = calculate_weighted_means(
        userx,
        aggregate_cols=["dia", "mes", "tipo_dia", "genero_agregado", "tarifa_agregada"],
        weighted_mean_cols=["cant_viajes"],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
    ).round(3)

    userx = calculate_weighted_means(
        userx,
        aggregate_cols=["dia", "mes", "tipo_dia", "genero_agregado", "tarifa_agregada"],
        weighted_mean_cols=["cant_viajes"],
        weight_col="factor_expansion_linea",
        var_fex_summed=False,
    ).round(3)

    userx["tabla"] = "usuario-genero_agregado-tarifa_agregada"
    socio_indicadores = pd.concat([socio_indicadores, userx], ignore_index=True)

    # Preparo socioindicadores final
    socio_indicadores = socio_indicadores[
        [
            "tabla",
            "dia",
            "mes",
            "tipo_dia",
            "genero_agregado",
            "tarifa_agregada",
            "modo",
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "cant_etapas",
            "cant_viajes",
            "diff_time",
            "factor_expansion_linea",
        ]
    ]
    socio_indicadores.columns = [
        "tabla",
        "dia",
        "mes",
        "tipo_dia",
        "genero_agregado",
        "tarifa_agregada",
        "Modo",
        "distance_od",
        "Tiempo de viaje",
        "Velocidad",
        "Etapas promedio",
        "Viajes promedio",
        "Tiempo entre viajes",
        "factor_expansion_linea",
    ]

    socio_indicadores["genero_agregado"] = socio_indicadores["genero_agregado"].fillna(
        ""
    )
    socio_indicadores["tarifa_agregada"] = socio_indicadores["tarifa_agregada"].fillna(
        ""
    )

    # socio_indicadores['genero_agregado'] = socio_indicadores['genero_agregado'].fillna('')
    # socio_indicadores['tarifa_agregada'] = socio_indicadores['tarifa_agregada'].fillna('')

    socio_indicadores["Modo"] = socio_indicadores["Modo"].fillna("")

    socio_indicadores = socio_indicadores.sort_values(
        ["tabla", "dia", "mes", "tipo_dia"]
    )

    replace_dash_partition(ctx, socio_indicadores, "socio_indicadores", ["dia"])

    hora = (
        etapas.groupby(["dia", "modo", "hora"], as_index=False)
        .factor_expansion_linea.sum()
        .round()
        .fillna(0)
        .rename(columns={"factor_expansion_linea": "viajes"})
    )
    hora["viajes"] = hora["viajes"].astype(int)

    horaT = hora.groupby(["dia", "hora"], as_index=False).viajes.sum()
    horaT["modo"] = "Todos"

    hora = pd.concat([hora, horaT], ignore_index=True)

    etapas["dist"] = etapas.distance_od.round(0).astype(int)
    dist = (
        etapas.groupby(["dia", "modo", "dist"], as_index=False)
        .factor_expansion_linea.sum()
        .round()
        .fillna(0)
        .rename(columns={"factor_expansion_linea": "viajes"})
    )
    dist["viajes"] = dist["viajes"].astype(int)

    distT = dist.groupby(["dia", "dist"], as_index=False).viajes.sum()
    distT["modo"] = "Todos"

    dist = pd.concat([dist, distT], ignore_index=True)

    dist.columns = ["Día", "Modo", "Distancia (kms)", "Viajes"]
    hora.columns = ["Día", "Modo", "Hora", "Viajes"]

    ctx.dash.save_indicator(dist, "distribucion")
    ctx.dash.save_indicator(hora, "viajes_hora")


def preparo_etapas_agregadas(ctx: StorageContext, etapas, viajes, equivalencias_zonas):

    e_agg = etapas.groupby(
        ["dia", "mes", "tipo_dia", "h3_o", "h3_d", "modo", "id_linea"], as_index=False
    ).factor_expansion_linea.sum()

    e_agg = e_agg[e_agg.h3_o != e_agg.h3_d]
    lineas = ctx.insumos.get_metadata_lineas()
    e_agg = e_agg.merge(lineas[["id_linea", "nombre_linea"]])

    v_agg = viajes.groupby(
        ["dia", "mes", "tipo_dia", "h3_o", "h3_d", "modo"], as_index=False
    ).factor_expansion_linea.sum()
    # v_agg = v_agg.groupby(['dia', 'mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo'], as_index=False).factor_expansion_linea.mean()
    v_agg = v_agg[v_agg.h3_o != v_agg.h3_d]

    etapas = etapas.assign(
        etapas_max=etapas.groupby(["dia", "id_tarjeta", "id_viaje"]).id_etapa.transform("max")
    )

    transfers = etapas.loc[
        :,
        [
            "dia",
            "id_tarjeta",
            "id_viaje",
            "id_etapa",
            "etapas_max",
            "id_linea",
            "h3_o",
            "h3_d",
            "factor_expansion_linea",
        ],
    ]  # (etapas.etapas_max>1)
    transfers = transfers.merge(lineas[["id_linea", "nombre_linea"]], how="left")
    transfers = (
        transfers.pivot(
            index=["dia", "id_tarjeta", "id_viaje"],
            columns="id_etapa",
            values="nombre_linea",
        )
        .reset_index()
        .fillna("")
    )
    transfers["seq_lineas"] = ""
    for i in range(1, etapas.etapas_max.max() + 1):
        transfers["seq_lineas"] += transfers[i] + " -- "
        transfers["seq_lineas"] = transfers["seq_lineas"].str.replace(" --  -- ", "")

    transfers.loc[transfers.seq_lineas.str[-4:] == " -- ", "seq_lineas"] = (
        transfers.loc[transfers.seq_lineas.str[-4:] == " -- ", "seq_lineas"].str[:-4]
    )
    transfers = viajes.merge(transfers[["dia", "id_tarjeta", "id_viaje", "seq_lineas"]])
    transfers = transfers.groupby(
        ["dia", "mes", "tipo_dia", "h3_o", "h3_d", "modo", "seq_lineas"], as_index=False
    ).factor_expansion_linea.sum()
    # transfers = transfers.groupby(['dia', 'mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo', 'seq_lineas'], as_index=False).factor_expansion_linea.mean()

    if len(equivalencias_zonas) > 0:
        zonas_cols = equivalencias_zonas.columns.tolist()
        zonas_cols = [
            item for item in zonas_cols if item not in ["fex", "latitud", "longitud"]
        ]
        equivalencias_zonas = equivalencias_zonas[zonas_cols]

        zonas_cols_o = [f"{item}_o" for item in zonas_cols]
        zonas_cols_d = [f"{item}_d" for item in zonas_cols]

        eq_o = equivalencias_zonas.rename(columns=dict(zip(zonas_cols, zonas_cols_o)))
        e_agg = e_agg.merge(eq_o, how="left")
        v_agg = v_agg.merge(eq_o, how="left")
        transfers = transfers.merge(eq_o, how="left")

        eq_d = equivalencias_zonas.rename(columns=dict(zip(zonas_cols, zonas_cols_d)))
        e_agg = e_agg.merge(eq_d, how="left")
        v_agg = v_agg.merge(eq_d, how="left")
        transfers = transfers.merge(eq_d, how="left")

    ctx.dash.save_raw(e_agg, "etapas_agregadas")
    ctx.dash.save_raw(v_agg, "viajes_agregados")
    ctx.dash.save_raw(transfers, "transferencias_agregadas")


def preparo_lineas_deseo(
    ctx: StorageContext,
    etapas_selec,
    viajes_selec,
    polygons_h3="",
    poligonos="",
    res=6,
    zonificaciones=[],
):

    zonificaciones = ensure_geodataframe(zonificaciones)

    if len(polygons_h3) == 0:
        id_polygon = "NONE"
        polygons_h3 = pd.DataFrame([["NONE"]], columns=["id_polygon"])
        poligonos = pd.DataFrame([["NONE", "NONE"]], columns=["id", "tipo"])
        etapas_selec = etapas_selec.assign(id_polygon="NONE", coincidencias="NONE")
        viajes_selec = viajes_selec.assign(id_polygon="NONE", coincidencias="NONE")

    # Traigo zonas
    zonas_data = ctx.insumos.get_raw("equivalencias_zonas")
    zonas_cols = (
        [c for c in zonas_data.columns if c not in ["h3", "latitud", "longitud"]]
        if len(zonas_data) > 0
        else []
    )

    if type(res) == int:
        res = [res]

    res_vars = []
    for i in res:
        res_vars += [f"res_{i}"]
        if not f"res_{i}" in zonificaciones.zona.unique().tolist():
            h3_vals = pd.concat(
                [
                    etapas_selec.loc[etapas_selec.h3_o.notna(), ["h3_o"]].rename(
                        columns={"h3_o": "h3"}
                    ),
                    etapas_selec.loc[etapas_selec.h3_d.notna(), ["h3_d"]].rename(
                        columns={"h3_d": "h3"}
                    ),
                ]
            ).drop_duplicates()
            h3_vals["h3_res"] = h3_vals["h3"].apply(h3toparent, res=i)

            h3_zona = (
                create_h3_gdf(h3_vals.h3_res.tolist())
                .rename(columns={"hexagon_id": "id"})
                .drop_duplicates()
            )
            h3_zona["zona"] = f"res_{i}"
            zonificaciones = ensure_geodataframe(
                pd.concat([zonificaciones, h3_zona], ignore_index=True)
            )

    zonas_cols = [x for x in zonas_cols if "res" not in x]
    zonas = zonas_cols + res_vars
    logger.debug("Zonas: %s", zonas)

    for id_polygon in polygons_h3.id_polygon.unique():

        poly_h3 = polygons_h3[polygons_h3.id_polygon == id_polygon]
        poly = poligonos[poligonos.id == id_polygon]
        tipo_poly = poly.tipo.values[0]

        # Preparo Etapas con inicio, transferencias y fin del viaje
        etapas_all = etapas_selec.loc[
            (etapas_selec.id_polygon == id_polygon),
            [
                "dia",
                "id_tarjeta",
                "id_viaje",
                "id_etapa",
                "h3_o",
                "h3_d",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "transferencia",
                "distancia_agregada",
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "coincidencias",
                "factor_expansion_linea",
            ],
        ].copy()
        etapas_all["etapa_max"] = etapas_all.groupby(
            ["dia", "id_tarjeta", "id_viaje"]
        ).id_etapa.transform("max")

        # Borro los casos que tienen 3 transferencias o más
        _excess = etapas_all[etapas_all.etapa_max > 3]
        if len(_excess) > 0:
            nborrar = (
                len(_excess[["id_tarjeta", "id_viaje"]].value_counts())
                / len(etapas_all[["id_tarjeta", "id_viaje"]].value_counts())
                * 100
            )
            logger.info(
                "Borrando viajes con más de 3 etapas: %.2f%% del polígono %s",
                nborrar, id_polygon,
            )
            etapas_all = etapas_all[etapas_all.etapa_max <= 3].copy()
            del _excess

        etapas_all["ultimo_viaje"] = 0
        etapas_all.loc[etapas_all.etapa_max == etapas_all.id_etapa, "ultimo_viaje"] = 1

        ultimo_viaje = etapas_all[etapas_all.ultimo_viaje == 1]

        etapas_all["h3"] = etapas_all["h3_o"]
        etapas_all = etapas_all[
            [
                "dia",
                "id_tarjeta",
                "id_viaje",
                "id_etapa",
                "h3",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "transferencia",
                "distancia_agregada",
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "coincidencias",
                "factor_expansion_linea",
            ]
        ]
        etapas_all["ultimo_viaje"] = 0

        ultimo_viaje["h3"] = ultimo_viaje["h3_d"]
        ultimo_viaje["id_etapa"] += 1
        ultimo_viaje = ultimo_viaje[
            [
                "dia",
                "id_tarjeta",
                "id_viaje",
                "id_etapa",
                "h3",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "transferencia",
                "distancia_agregada",
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "coincidencias",
                "factor_expansion_linea",
                "ultimo_viaje",
            ]
        ]

        etapas_all = (
            pd.concat([etapas_all, ultimo_viaje])
            .sort_values(["dia", "id_tarjeta", "id_viaje", "id_etapa"])
            .reset_index(drop=True)
        )

        etapas_all["tipo_viaje"] = "Transfer_" + (etapas_all["id_etapa"] - 1).astype(
            str
        )
        etapas_all.loc[etapas_all.ultimo_viaje == 1, "tipo_viaje"] = "Fin"
        etapas_all.loc[etapas_all.id_etapa == 1, "tipo_viaje"] = "Inicio"

        etapas_all["polygon"] = ""
        if id_polygon != "NONE":
            etapas_all.loc[etapas_all.h3.isin(poly_h3.h3.unique()), "polygon"] = (
                id_polygon
            )

        etapas_all = etapas_all.drop(["ultimo_viaje"], axis=1)

        # Guardo las coordenadas de los H3
        h3_coords = (
            etapas_all.groupby("h3", as_index=False)
            .id_viaje.count()
            .drop(["id_viaje"], axis=1)
        )
        # h3_coords[['lat', 'lon']] = h3_coords.h3.apply(h3_to_lat_lon)
        h3_coords[["lat", "lon"]] = h3_coords.h3.apply(
            lambda x: pd.Series(h3_to_lat_lon(x))
        )

        # Preparo cada etapa de viaje para poder hacer la agrupación y tener inicio, transferencias y destino en un mismo registro
        inicio = etapas_all.loc[
            etapas_all.tipo_viaje == "Inicio",
            [
                "dia",
                "id_tarjeta",
                "id_viaje",
                "h3",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "transferencia",
                "distancia_agregada",
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "coincidencias",
                "factor_expansion_linea",
                "polygon",
            ],
        ].rename(columns={"h3": "h3_inicio", "polygon": "poly_inicio"})

        fin = etapas_all.loc[
            etapas_all.tipo_viaje == "Fin",
            ["dia", "id_tarjeta", "id_viaje", "h3", "polygon"],
        ].rename(columns={"h3": "h3_fin", "polygon": "poly_fin"})
        transfer1 = etapas_all.loc[
            etapas_all.tipo_viaje == "Transfer_1",
            ["dia", "id_tarjeta", "id_viaje", "h3", "polygon"],
        ].rename(columns={"h3": "h3_transfer1", "polygon": "poly_transfer1"})
        transfer2 = etapas_all.loc[
            etapas_all.tipo_viaje == "Transfer_2",
            ["dia", "id_tarjeta", "id_viaje", "h3", "polygon"],
        ].rename(columns={"h3": "h3_transfer2", "polygon": "poly_transfer2"})

        etapas_agrupadas = (
            inicio.merge(transfer1, how="left")
            .merge(transfer2, how="left")
            .merge(fin, how="left")
            .fillna("")
        )

        etapas_agrupadas = etapas_agrupadas[
            [
                "dia",
                "id_tarjeta",
                "id_viaje",
                "h3_inicio",
                "h3_transfer1",
                "h3_transfer2",
                "h3_fin",
                "poly_inicio",
                "poly_transfer1",
                "poly_transfer2",
                "poly_fin",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "transferencia",
                "distancia_agregada",
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "coincidencias",
                "factor_expansion_linea",
            ]
        ]

        for zona in zonas:
            logger.debug("Polígono %s - Tipo: %s - Zona: %s", id_polygon, tipo_poly, zona)

            if id_polygon != "NONE":
                # print(id_polygon, zona)
                h3_equivalencias = creo_h3_equivalencias(
                    polygons_h3[polygons_h3.id_polygon == id_polygon].copy(),
                    poligonos[poligonos.id == id_polygon],
                    zona,
                    zonificaciones[zonificaciones.zona == zona].copy(),
                )

            # Preparo para agrupar por líneas de deseo y cambiar de resolución si es necesario

            etapas_agrupadas_zon = etapas_agrupadas.copy()

            etapas_agrupadas_zon["id_polygon"] = id_polygon
            etapas_agrupadas_zon["zona"] = zona

            etapas_agrupadas_zon["inicio_norm"] = etapas_agrupadas_zon["h3_inicio"]
            etapas_agrupadas_zon["transfer1_norm"] = etapas_agrupadas_zon[
                "h3_transfer1"
            ]
            etapas_agrupadas_zon["transfer2_norm"] = etapas_agrupadas_zon[
                "h3_transfer2"
            ]
            etapas_agrupadas_zon["fin_norm"] = etapas_agrupadas_zon["h3_fin"]
            etapas_agrupadas_zon["poly_inicio_norm"] = etapas_agrupadas_zon[
                "poly_inicio"
            ]
            etapas_agrupadas_zon["poly_transfer1_norm"] = etapas_agrupadas_zon[
                "poly_transfer1"
            ]
            etapas_agrupadas_zon["poly_transfer2_norm"] = etapas_agrupadas_zon[
                "poly_transfer2"
            ]
            etapas_agrupadas_zon["poly_fin_norm"] = etapas_agrupadas_zon["poly_fin"]

            n = 1
            for i in ["inicio_norm", "transfer1_norm", "transfer2_norm", "fin_norm"]:

                etapas_agrupadas_zon = etapas_agrupadas_zon.merge(
                    h3_coords.rename(columns={"h3": i}), how="left", on=i
                )

                etapas_agrupadas_zon[f"lon{n}"] = etapas_agrupadas_zon["lon"]
                etapas_agrupadas_zon[f"lat{n}"] = etapas_agrupadas_zon["lat"]
                etapas_agrupadas_zon = etapas_agrupadas_zon.drop(["lon", "lat"], axis=1)

                # Selecciono el centroide del polígono en vez del centroide de cada hexágono

                if tipo_poly == "poligono":
                    poly_centroid = poly.geometry.to_crs(4326).centroid
                    etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon[i].isin(poly_h3.h3.unique()), f"lat{n}"
                    ] = poly_centroid.y.iloc[0]
                    etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon[i].isin(poly_h3.h3.unique()), f"lon{n}"
                    ] = poly_centroid.x.iloc[0]

                if "res_" in zona:
                    resol = int(zona.replace("res_", ""))
                    etapas_agrupadas_zon[i] = etapas_agrupadas_zon[i].apply(
                        h3toparent, res=resol
                    )

                else:
                    # zonas_data_ = zonas_data.groupby(
                    #     ['h3', 'latitud', 'longitud'], as_index=False)[zona].first()

                    etapas_agrupadas_zon = etapas_agrupadas_zon.merge(
                        zonas_data[["h3", zona]].rename(
                            columns={"h3": i, zona: "zona_tmp"}
                        ),
                        how="left",
                    )

                    etapas_agrupadas_zon[i] = etapas_agrupadas_zon["zona_tmp"]
                    etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                        ["zona_tmp"], axis=1
                    )

                    if (
                        len(
                            etapas_agrupadas_zon[
                                (etapas_agrupadas_zon.inicio_norm.isna())
                                | (etapas_agrupadas_zon.fin_norm.isna())
                            ]
                        )
                        > 0
                    ) & (i == "fin_norm"):
                        cant_etapas = len(
                            etapas_agrupadas_zon[
                                (etapas_agrupadas_zon.inicio_norm.isna())
                                | (etapas_agrupadas_zon.fin_norm.isna())
                            ]
                        )
                        logger.warning(
                            "Hay %d registros a los que no se les pudo asignar %s",
                            cant_etapas, zona,
                        )

                    etapas_agrupadas_zon = etapas_agrupadas_zon[
                        ~(
                            (etapas_agrupadas_zon.inicio_norm.isna())
                            | (etapas_agrupadas_zon.fin_norm.isna())
                        )
                    ]

                # Si es cuenca modifico las latitudes longitudes donde coincide el polígono de cuenca con el h3
                if tipo_poly == "cuenca":
                    # reemplazo latitudes y longitudes de cuenca para normalizar
                    poly_var = i.replace("h3_", "").replace("_norm", "")


                    
                    h3_equivalencias_agg = (
                        h3_equivalencias.groupby(
                            [f"zona_{zona}", f"lat_{zona}", f"lon_{zona}"],
                            as_index=False,
                        )
                        ['h3_o'].count()
                        .drop(["h3_o"], axis=1)
                    )

                    etapas_agrupadas_zon = etapas_agrupadas_zon.merge(
                        h3_equivalencias_agg[
                            [f"zona_{zona}", f"lat_{zona}", f"lon_{zona}"]
                        ].rename(columns={f"zona_{zona}": i}),
                        how="left",
                        on=i,
                    )

                    etapas_agrupadas_zon.loc[
                        (etapas_agrupadas_zon[f"lat_{zona}"].notna())
                        & (etapas_agrupadas_zon[f"poly_{poly_var}"] != ""),
                        f"lat{n}",
                    ] = etapas_agrupadas_zon.loc[
                        (etapas_agrupadas_zon[f"lat_{zona}"].notna())
                        & (etapas_agrupadas_zon[f"poly_{poly_var}"] != ""),
                        f"lat_{zona}",
                    ]

                    etapas_agrupadas_zon.loc[
                        (etapas_agrupadas_zon[f"lon_{zona}"].notna())
                        & (etapas_agrupadas_zon[f"poly_{poly_var}"] != ""),
                        f"lon{n}",
                    ] = etapas_agrupadas_zon.loc[
                        (etapas_agrupadas_zon[f"lon_{zona}"].notna())
                        & (etapas_agrupadas_zon[f"poly_{poly_var}"] != ""),
                        f"lon_{zona}",
                    ]

                    etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                        [f"lon_{zona}", f"lat_{zona}"], axis=1
                    )

                etapas_agrupadas_zon[i] = etapas_agrupadas_zon[i].fillna("")
                n += 1

            # INICIO - Normalizo variables (variables _norm)
            etapas_agrupadas_zon["inicio"] = etapas_agrupadas_zon["inicio_norm"]
            etapas_agrupadas_zon["transfer1"] = etapas_agrupadas_zon["transfer1_norm"]
            etapas_agrupadas_zon["transfer2"] = etapas_agrupadas_zon["transfer2_norm"]
            etapas_agrupadas_zon["fin"] = etapas_agrupadas_zon["fin_norm"]
            etapas_agrupadas_zon["poly_inicio"] = etapas_agrupadas_zon[
                "poly_inicio_norm"
            ]
            etapas_agrupadas_zon["poly_transfer1"] = etapas_agrupadas_zon[
                "poly_transfer1_norm"
            ]
            etapas_agrupadas_zon["poly_transfer2"] = etapas_agrupadas_zon[
                "poly_transfer2_norm"
            ]
            etapas_agrupadas_zon["poly_fin"] = etapas_agrupadas_zon["poly_fin_norm"]
            etapas_agrupadas_zon["lat1_norm"] = etapas_agrupadas_zon["lat1"]
            etapas_agrupadas_zon["lat2_norm"] = etapas_agrupadas_zon["lat2"]
            etapas_agrupadas_zon["lat3_norm"] = etapas_agrupadas_zon["lat3"]
            etapas_agrupadas_zon["lat4_norm"] = etapas_agrupadas_zon["lat4"]
            etapas_agrupadas_zon["lon1_norm"] = etapas_agrupadas_zon["lon1"]
            etapas_agrupadas_zon["lon2_norm"] = etapas_agrupadas_zon["lon2"]
            etapas_agrupadas_zon["lon3_norm"] = etapas_agrupadas_zon["lon3"]
            etapas_agrupadas_zon["lon4_norm"] = etapas_agrupadas_zon["lon4"]

            et1 = etapas_agrupadas_zon[
                etapas_agrupadas_zon.inicio <= etapas_agrupadas_zon.fin
            ].copy()
            et2 = etapas_agrupadas_zon[
                (etapas_agrupadas_zon.inicio > etapas_agrupadas_zon.fin)
            ].copy()

            et2["inicio_norm"] = et2["fin"]
            et2["fin_norm"] = et2["inicio"]
            et2["poly_inicio_norm"] = et2["poly_fin"]
            et2["poly_fin_norm"] = et2["poly_inicio"]

            et2["lat1_norm"] = et2["lat4"]
            et2["lon1_norm"] = et2["lon4"]
            et2["lat4_norm"] = et2["lat1"]
            et2["lon4_norm"] = et2["lon1"]

            et2.loc[et2.transfer2 != "", "transfer1_norm"] = et2.loc[
                et2.transfer2 != "", "transfer2"
            ]
            et2.loc[et2.transfer2 != "", "transfer2_norm"] = et2.loc[
                et2.transfer2 != "", "transfer1"
            ]
            et2.loc[et2.transfer2 != "", "poly_transfer1_norm"] = et2.loc[
                et2.transfer2 != "", "poly_transfer2"
            ]
            et2.loc[et2.transfer2 != "", "poly_transfer2_norm"] = et2.loc[
                et2.transfer2 != "", "poly_transfer1"
            ]
            et2.loc[et2.transfer2 != "", "lat2_norm"] = et2.loc[
                et2.transfer2 != "", "lat3"
            ]
            et2.loc[et2.transfer2 != "", "lon2_norm"] = et2.loc[
                et2.transfer2 != "", "lon3"
            ]
            et2.loc[et2.transfer2 != "", "lat3_norm"] = et2.loc[
                et2.transfer2 != "", "lat2"
            ]
            et2.loc[et2.transfer2 != "", "lon3_norm"] = et2.loc[
                et2.transfer2 != "", "lon2"
            ]

            etapas_agrupadas_zon = pd.concat([et1, et2], ignore_index=True)

            # FIN - Normalizo variables (variables _norm)

            ### etapas_agrupadas_zon = normalizo_zona(etapas_agrupadas_zon,
            ###                                       zonificaciones[zonificaciones.zona == zona].copy())

            etapas_agrupadas_zon["tipo_dia_"] = (
                pd.to_datetime(etapas_agrupadas_zon.dia).dt.weekday.astype(str).copy()
            )
            etapas_agrupadas_zon["tipo_dia"] = "Hábil"
            etapas_agrupadas_zon.loc[
                etapas_agrupadas_zon.tipo_dia_.astype(int) >= 5, "tipo_dia"
            ] = "Fin de Semana"
            etapas_agrupadas_zon = etapas_agrupadas_zon.drop(["tipo_dia_"], axis=1)
            etapas_agrupadas_zon["mes"] = etapas_agrupadas_zon.dia.str[:7]

            etapas_agrupadas_zon = etapas_agrupadas_zon[
                [
                    "id_polygon",
                    "zona",
                    "dia",
                    "mes",
                    "tipo_dia",
                    "id_tarjeta",
                    "id_viaje",
                    "h3_inicio",
                    "h3_transfer1",
                    "h3_transfer2",
                    "h3_fin",
                    "inicio",
                    "transfer1",
                    "transfer2",
                    "fin",
                    "poly_inicio",
                    "poly_transfer1",
                    "poly_transfer2",
                    "poly_fin",
                    "inicio_norm",
                    "transfer1_norm",
                    "transfer2_norm",
                    "fin_norm",
                    "poly_inicio_norm",
                    "poly_transfer1_norm",
                    "poly_transfer2_norm",
                    "poly_fin_norm",
                    "lon1",
                    "lat1",
                    "lon2",
                    "lat2",
                    "lon3",
                    "lat3",
                    "lon4",
                    "lat4",
                    "lon1_norm",
                    "lat1_norm",
                    "lon2_norm",
                    "lat2_norm",
                    "lon3_norm",
                    "lat3_norm",
                    "lon4_norm",
                    "lat4_norm",
                    "transferencia",
                    "modo_agregado",
                    "rango_hora",
                    "genero_agregado",
                    "tarifa_agregada",
                    "coincidencias",
                    "distancia_agregada",
                    "distance_od",
                    "travel_time_min",
                    "kmh_od",
                    "factor_expansion_linea",
                ]
            ]

            aggregate_cols = [
                "id_polygon",
                "dia",
                "mes",
                "tipo_dia",
                "zona",
                "inicio",
                "fin",
                "poly_inicio",
                "poly_fin",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "coincidencias",
                "distancia_agregada",
            ]

            viajes_matrices = construyo_matrices(
                etapas_agrupadas_zon,
                aggregate_cols,
                zonificaciones,
                False,
                False,
                False,
            )

            # Agrupación de viajes
            aggregate_cols = [
                "id_polygon",
                "dia",
                "mes",
                "tipo_dia",
                "zona",
                "inicio_norm",
                "transfer1_norm",
                "transfer2_norm",
                "fin_norm",
                "poly_inicio_norm",
                "poly_transfer1_norm",
                "poly_transfer2_norm",
                "poly_fin_norm",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "coincidencias",
                "distancia_agregada",
            ]

            weighted_mean_cols = [
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "lat1_norm",
                "lon1_norm",
                "lat2_norm",
                "lon2_norm",
                "lat3_norm",
                "lon3_norm",
                "lat4_norm",
                "lon4_norm",
            ]

            weight_col = "factor_expansion_linea"

            zero_to_nan = [
                "lat1_norm",
                "lon1_norm",
                "lat2_norm",
                "lon2_norm",
                "lat3_norm",
                "lon3_norm",
                "lat4_norm",
                "lon4_norm",
                "travel_time_min",
                "kmh_od",
            ]

            etapas_agrupadas_zon = agrupar_viajes(
                etapas_agrupadas_zon,
                aggregate_cols,
                weighted_mean_cols,
                weight_col,
                zero_to_nan,
                agg_transferencias=False,
                agg_modo=False,
                agg_hora=False,
                agg_distancia=False,
            )

            zonificaciones["lat"] = zonificaciones.geometry.representative_point().y
            zonificaciones["lon"] = zonificaciones.geometry.representative_point().x

            n = 1
            poly_lst = ["poly_inicio", "poly_transfer1", "poly_transfer2", "poly_fin"]
            for i in ["inicio", "transfer1", "transfer2", "fin"]:
                etapas_agrupadas_zon = etapas_agrupadas_zon.merge(
                    zonificaciones[["zona", "id", "lat", "lon"]].rename(
                        columns={
                            "id": f"{i}_norm",
                            "lat": f"lat{n}_norm_tmp",
                            "lon": f"lon{n}_norm_tmp",
                        }
                    ),
                    how="left",
                    on=["zona", f"{i}_norm"],
                )
                etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon[f"{poly_lst[n-1]}_norm"] == "", f"lat{n}_norm"
                ] = etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon[f"{poly_lst[n-1]}_norm"] == "",
                    f"lat{n}_norm_tmp",
                ]
                etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon[f"{poly_lst[n-1]}_norm"] == "", f"lon{n}_norm"
                ] = etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon[f"{poly_lst[n-1]}_norm"] == "",
                    f"lon{n}_norm_tmp",
                ]

                etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                    [f"lat{n}_norm_tmp", f"lon{n}_norm_tmp"], axis=1
                )

                if (n == 1) | (n == 4):
                    viajes_matrices = viajes_matrices.merge(
                        zonificaciones[["zona", "id", "lat", "lon"]].rename(
                            columns={
                                "id": f"{i}",
                                "lat": f"lat{n}_tmp",
                                "lon": f"lon{n}_tmp",
                            }
                        ),
                        how="left",
                        on=["zona", f"{i}"],
                    )
                    viajes_matrices.loc[
                        viajes_matrices[f"{poly_lst[n-1]}"] == "", f"lat{n}"
                    ] = viajes_matrices.loc[
                        viajes_matrices[f"{poly_lst[n-1]}"] == "", f"lat{n}_tmp"
                    ]
                    viajes_matrices.loc[
                        viajes_matrices[f"{poly_lst[n-1]}"] == "", f"lon{n}"
                    ] = viajes_matrices.loc[
                        viajes_matrices[f"{poly_lst[n-1]}"] == "", f"lon{n}_tmp"
                    ]
                    viajes_matrices = viajes_matrices.drop(
                        [f"lat{n}_tmp", f"lon{n}_tmp"], axis=1
                    )

                n += 1

            # # Agrupar a nivel de mes y corregir factor de expansión
            sum_viajes = (
                etapas_agrupadas_zon.groupby(
                    ["dia", "mes", "tipo_dia", "zona"], as_index=False
                )
                .factor_expansion_linea.sum()
                .groupby(["dia", "mes", "tipo_dia", "zona"], as_index=False)
                .factor_expansion_linea.mean()
                .round()
            )

            aggregate_cols = [
                "dia",
                "mes",
                "tipo_dia",
                "id_polygon",
                "poly_inicio_norm",
                "poly_transfer1_norm",
                "poly_transfer2_norm",
                "poly_fin_norm",
                "zona",
                "inicio_norm",
                "transfer1_norm",
                "transfer2_norm",
                "fin_norm",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "coincidencias",
                "distancia_agregada",
            ]
            weighted_mean_cols = [
                "distance_od",
                "travel_time_min",
                "kmh_od",
                "lat1_norm",
                "lon1_norm",
                "lat2_norm",
                "lon2_norm",
                "lat3_norm",
                "lon3_norm",
                "lat4_norm",
                "lon4_norm",
            ]

            etapas_agrupadas_zon = calculate_weighted_means(
                etapas_agrupadas_zon,
                aggregate_cols=aggregate_cols,
                weighted_mean_cols=weighted_mean_cols,
                weight_col="factor_expansion_linea",
                zero_to_nan=zero_to_nan,
                var_fex_summed=False,
            )

            sum_viajes["factor_expansion_linea"] = 1 - (
                sum_viajes["factor_expansion_linea"]
                / etapas_agrupadas_zon.groupby(
                    ["dia", "mes", "tipo_dia", "zona"], as_index=False
                )
                .factor_expansion_linea.sum()
                .factor_expansion_linea
            )
            sum_viajes = sum_viajes.rename(
                columns={"factor_expansion_linea": "factor_correccion"}
            )

            etapas_agrupadas_zon = etapas_agrupadas_zon.merge(sum_viajes)
            etapas_agrupadas_zon["factor_expansion_linea2"] = (
                etapas_agrupadas_zon["factor_expansion_linea"]
                * etapas_agrupadas_zon["factor_correccion"]
            )
            etapas_agrupadas_zon["factor_expansion_linea2"] = (
                etapas_agrupadas_zon["factor_expansion_linea"]
                - etapas_agrupadas_zon["factor_expansion_linea2"]
            )
            etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                ["factor_correccion", "factor_expansion_linea"], axis=1
            )
            etapas_agrupadas_zon = etapas_agrupadas_zon.rename(
                columns={"factor_expansion_linea2": "factor_expansion_linea"}
            )

            # # Agrupar a nivel de dia y corregir factor de expansión
            sum_viajes = (
                viajes_matrices.groupby(
                    ["dia", "mes", "tipo_dia", "zona"], as_index=False
                )
                .factor_expansion_linea.sum()
                .groupby(["dia", "mes", "tipo_dia", "zona"], as_index=False)
                .factor_expansion_linea.mean()
            )

            aggregate_cols = [
                "id_polygon",
                "poly_inicio",
                "poly_fin",
                "dia",
                "mes",
                "tipo_dia",
                "zona",
                "inicio",
                "fin",
                "transferencia",
                "modo_agregado",
                "rango_hora",
                "genero_agregado",
                "tarifa_agregada",
                "coincidencias",
                "distancia_agregada",
                "orden_origen",
                "orden_destino",
                "Origen",
                "Destino",
            ]
            weighted_mean_cols = [
                "lat1",
                "lon1",
                "lat4",
                "lon4",
                "distance_od",
                "travel_time_min",
                "kmh_od",
            ]
            zero_to_nan = [
                "lat1",
                "lon1",
                "lat4",
                "lon4",
                "kmh_od",
                "travel_time_min",
            ]

            viajes_matrices = calculate_weighted_means(
                viajes_matrices,
                aggregate_cols=aggregate_cols,
                weighted_mean_cols=weighted_mean_cols,
                weight_col="factor_expansion_linea",
                zero_to_nan=zero_to_nan,
                var_fex_summed=False,
            )

            sum_viajes["factor_expansion_linea"] = 1 - (
                sum_viajes["factor_expansion_linea"]
                / viajes_matrices.groupby(
                    ["dia", "mes", "tipo_dia", "zona"], as_index=False
                )
                .factor_expansion_linea.sum()
                .factor_expansion_linea
            )
            sum_viajes = sum_viajes.rename(
                columns={"factor_expansion_linea": "factor_correccion"}
            )

            viajes_matrices = viajes_matrices.merge(sum_viajes)
            viajes_matrices["factor_expansion_linea2"] = (
                viajes_matrices["factor_expansion_linea"]
                * viajes_matrices["factor_correccion"]
            )
            viajes_matrices["factor_expansion_linea2"] = (
                viajes_matrices["factor_expansion_linea"]
                - viajes_matrices["factor_expansion_linea2"]
            )
            viajes_matrices = viajes_matrices.drop(
                ["factor_correccion", "factor_expansion_linea"], axis=1
            )
            viajes_matrices = viajes_matrices.rename(
                columns={"factor_expansion_linea2": "factor_expansion_linea"}
            )

            if len(poligonos[poligonos.tipo == "cuenca"]) > 0:

                etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon.poly_inicio_norm.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "inicio_norm",
                ] = (
                    etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_inicio_norm.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "inicio_norm",
                    ]
                    + " (cuenca)"
                )
                etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon.poly_transfer1_norm.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "transfer1_norm",
                ] = (
                    etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_transfer1_norm.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "transfer1_norm",
                    ]
                    + " (cuenca)"
                )
                etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon.poly_transfer2_norm.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "transfer2_norm",
                ] = (
                    etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_transfer2_norm.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "transfer2_norm",
                    ]
                    + " (cuenca)"
                )
                etapas_agrupadas_zon.loc[
                    etapas_agrupadas_zon.poly_fin_norm.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "fin_norm",
                ] = (
                    etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_fin_norm.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "fin_norm",
                    ]
                    + " (cuenca)"
                )
                viajes_matrices.loc[
                    viajes_matrices.poly_inicio.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "Origen",
                ] = (
                    viajes_matrices.loc[
                        viajes_matrices.poly_inicio.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "Origen",
                    ]
                    + " (cuenca)"
                )
                viajes_matrices.loc[
                    viajes_matrices.poly_fin.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "Destino",
                ] = (
                    viajes_matrices.loc[
                        viajes_matrices.poly_fin.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "Destino",
                    ]
                    + " (cuenca)"
                )
                viajes_matrices.loc[
                    viajes_matrices.poly_inicio.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "inicio",
                ] = (
                    viajes_matrices.loc[
                        viajes_matrices.poly_inicio.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "inicio",
                    ]
                    + " (cuenca)"
                )
                viajes_matrices.loc[
                    viajes_matrices.poly_fin.isin(
                        poligonos[poligonos.tipo == "cuenca"].id.unique()
                    ),
                    "fin",
                ] = (
                    viajes_matrices.loc[
                        viajes_matrices.poly_fin.isin(
                            poligonos[poligonos.tipo == "cuenca"].id.unique()
                        ),
                        "fin",
                    ]
                    + " (cuenca)"
                )

            etapas_agrupadas_zon = etapas_agrupadas_zon.fillna(0)

            if id_polygon == "NONE":

                etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                    [
                        "id_polygon",
                        "poly_inicio_norm",
                        "poly_transfer1_norm",
                        "poly_transfer2_norm",
                        "poly_fin_norm",
                    ],
                    axis=1,
                )

                viajes_matrices = viajes_matrices.drop(
                    ["poly_inicio", "poly_fin"], axis=1
                )

                replace_dash_partition(
                    ctx, etapas_agrupadas_zon, "agg_etapas", ["dia", "zona"]
                )

                replace_dash_partition(
                    ctx, viajes_matrices, "agg_matrices", ["dia", "zona"]
                )
            else:
                replace_dash_partition(
                    ctx,
                    etapas_agrupadas_zon,
                    "poly_etapas",
                    ["dia", "zona", "id_polygon"],
                )

                replace_dash_partition(
                    ctx,
                    viajes_matrices,
                    "poly_matrices",
                    ["dia", "zona", "id_polygon"],
                )



def guarda_particion_modal(ctx: StorageContext, etapas):

    df_dummies = pd.get_dummies(etapas.modo)
    etapas = pd.concat([etapas, df_dummies], axis=1)
    cols_dummies = df_dummies.columns.tolist()

    etapas_modos = (
        etapas.groupby(
            ["dia", "mes", "tipo_dia", "genero_agregado", "id_tarjeta", "id_viaje"],
            as_index=False,
        )
        .factor_expansion_linea.mean()
        .merge(
            etapas.groupby(["dia", "id_tarjeta", "id_viaje"], as_index=False)[
                cols_dummies
            ].sum(),
            how="left",
        )
    )

    cols = [
        "dia",
        "mes",
        "tipo_dia",
        "genero_agregado",
    ] + cols_dummies
    etapas_modos = (
        etapas_modos.groupby(cols, as_index=False).factor_expansion_linea.sum().copy()
    )
    for i in cols_dummies:
        etapas_modos = etapas_modos.rename(columns={i: i.capitalize()})

    replace_dash_partition(ctx, etapas_modos, "datos_particion_modal", ["dia"])



def resumen_x_linea(ctx: StorageContext, etapas, viajes):

    gps = ctx.data.get_raw("gps")
    gps["fecha"] = pd.to_datetime(gps["fecha"], unit="s")
    lineas = ctx.insumos.get_metadata_lineas()
    kpis = ctx.data.get_raw("kpi_by_day_line")
    servicios = ctx.data.get_raw("services")
    lineas = lineas[["id_linea", "nombre_linea", "empresa"]].sort_values(["id_linea"])

    trx = ctx.data.get_raw("transacciones")
    if "tarifa_agregada" in trx.columns:
        trx["tarifa_agregada"] = trx["tarifa_agregada"].fillna("")
    if "genero_agregado" in trx.columns:
        trx["genero_agregado"] = trx["genero_agregado"].fillna("")

    metric_cols = [
        "transacciones",
        "distancia_media", "travel_time_min", "kmh_od",
        "cant_internos_en_trx", "cant_internos_en_gps",
        "tot_veh", "tot_km", "tot_pax",
        "dmt_mean_od", "dmt_median_od",
        "pvd", "kvd", "ipk_route", "fo_mean_od", "fo_median_od",
    ]

    # Resumen por línea
    all_linea = agrego_lineas(["dia", "id_linea"], trx, etapas, gps, servicios, kpis, lineas)
    all_linea["mes"] = all_linea["dia"].str[:7]
    metric_cols_linea = [c for c in metric_cols if c in all_linea.columns]
    all_linea = (
        all_linea
        .groupby(["dia", "mes", "id_linea", "nombre_linea", "empresa", "modo"], as_index=False)
        [metric_cols_linea]
        .mean()
        .round(2)
    )
    replace_dash_partition(ctx, all_linea, "resumen_lineas", ["dia"])

    # Resumen por línea y ramal
    all_ramal = agrego_lineas(["dia", "id_linea", "id_ramal"], trx, etapas, gps, servicios, kpis, lineas)
    all_ramal["mes"] = all_ramal["dia"].str[:7]
    metric_cols_ramal = [c for c in metric_cols if c in all_ramal.columns]
    all_ramal = (
        all_ramal
        .groupby(["dia", "mes", "id_linea", "id_ramal", "nombre_linea", "empresa", "modo"], as_index=False)
        [metric_cols_ramal]
        .mean()
        .round(2)
    )
    replace_dash_partition(ctx, all_ramal, "resumen_lineas_ramal", ["dia"])


@duracion
def proceso_poligonos(
    ctx: StorageContext,
    etapas=[],
    viajes=[],
    zonificaciones=[],
    resoluciones=[6],
    poligon_id="",
):
    logger.info("Procesa polígonos")
    poligonos = ensure_geodataframe(ctx.insumos.get_raw("poligonos"))
    if (len(poligonos) > 0) & (poligon_id != ""):
        poligonos = poligonos[poligonos.id == poligon_id]
    if len(poligonos) > 0:

        configs = leer_configs_generales(autogenerado=False)
        res = configs["resolucion_h3"]

        # Select cases based fron polygon
        etapas_selec, viajes_selec, polygons, polygons_h3 = select_cases_from_polygons(
            etapas, viajes, poligonos, res=res
        )

        preparo_lineas_deseo(
            ctx,
            etapas_selec,
            viajes_selec,
            polygons_h3,
            poligonos=poligonos,
            res=resoluciones,
            zonificaciones=zonificaciones,
        )

        construyo_indicadores(ctx, viajes_selec, poligonos=True)


def _table_exists(port, table: str) -> bool:
    try:
        df = port.query(f"SELECT * FROM {table} LIMIT 0")
        return len(df.columns) > 0
    except Exception:
        return False


def _table_has_cols(port, table: str, cols: list) -> bool:
    try:
        df = port.query(f"SELECT * FROM {table} LIMIT 0")
        return all(c in df.columns for c in cols)
    except Exception:
        return False


def crear_indices_unificados(ctx: StorageContext):
    """
    Crea índices en las bases SQLite usadas por el pipeline UrbanTrips:
    - data (etapas, viajes, transacciones, gps, services, kpi_by_day_line)
    - dash (agg_etapas, agg_matrices, poly_*, indicadores, resumen_lineas, etc.)
    - insumos (metadata_lineas, poligonos, zonificaciones, equivalencias_zonas)
    Aplica PRAGMAs de rendimiento y ANALYZE/optimize.
    """

    def _maybe_create(port, table, spec_list):
        if not _table_exists(port, table):
            return
        for name, cols in spec_list:
            if _table_has_cols(port, table, cols):
                try:
                    cols_sql = ", ".join(cols)
                    port.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols_sql});")
                except Exception as e:
                    logger.debug("[índice omitido] %s.%s: %s", table, name, e)

    def _speed_pragmas(port):
        for sql in [
            "PRAGMA journal_mode=WAL;",
            "PRAGMA synchronous=NORMAL;",
            "PRAGMA temp_store=MEMORY;",
            "PRAGMA mmap_size=134217728;",  # 128MB
        ]:
            try:
                port.execute(sql)
            except Exception as e:
                logger.debug("[pragma omitido] %s: %s", sql.strip(), e)

    def _analyze_optimize(port):
        for sql in ["ANALYZE;", "PRAGMA optimize;"]:
            try:
                port.execute(sql)
            except Exception as e:
                logger.debug("[optimizacion omitida] %s: %s", sql.strip(), e)

    # ---- PRAGMAs de rendimiento ----
    for port in (ctx.data, ctx.dash, ctx.insumos):
        _speed_pragmas(port)

    # =================
    #   DATA
    # =================
    _maybe_create(
        ctx.data,
        "etapas",
        [
            ("idx_etapas_od_dia", ["od_validado", "dia"]),
            ("idx_etapas_dia_tarjeta", ["dia", "id_tarjeta"]),
            ("idx_etapas_tarjeta_viaje", ["id_tarjeta", "id_viaje"]),
            ("idx_etapas_h3o", ["h3_o"]),
            ("idx_etapas_h3d", ["h3_d"]),
            ("idx_etapas_linea", ["id_linea"]),
            ("idx_etapas_ramal", ["id_ramal"]),
            ("idx_etapas_hora", ["hora"]),
        ],
    )

    _maybe_create(
        ctx.data,
        "viajes",
        [
            ("idx_viajes_od_dia", ["od_validado", "dia"]),
            ("idx_viajes_dia_tarjeta", ["dia", "id_tarjeta"]),
            ("idx_viajes_tarjeta_viaje", ["id_tarjeta", "id_viaje"]),
            ("idx_viajes_h3o", ["h3_o"]),
            ("idx_viajes_h3d", ["h3_d"]),
            ("idx_viajes_modo", ["modo"]),
            ("idx_viajes_hora", ["hora"]),
        ],
    )

    _maybe_create(
        ctx.data,
        "transacciones",
        [
            ("idx_trx_dia_linea", ["dia", "id_linea"]),
            ("idx_trx_linea_modo", ["id_linea", "modo"]),
            ("idx_trx_interno", ["interno"]),
        ],
    )

    _maybe_create(
        ctx.data,
        "gps",
        [
            ("idx_gps_interno", ["interno"]),
            ("idx_gps_fecha", ["fecha"]),
        ],
    )

    _maybe_create(
        ctx.data,
        "services",
        [
            ("idx_services_valid", ["valid"]),
            ("idx_services_interno", ["interno"]),
        ],
    )

    _maybe_create(
        ctx.data,
        "kpi_by_day_line",
        [
            ("idx_kpi_dia_linea", ["dia", "id_linea"]),
        ],
    )

    # =================
    #   DASH
    # =================
    _maybe_create(
        ctx.dash,
        "agg_etapas",
        [
            ("idx_agg_etapas_dia_zona", ["dia", "zona"]),
            ("idx_agg_etapas_zona_if", ["zona", "inicio_norm", "fin_norm"]),
            ("idx_agg_etapas_modo", ["modo_agregado"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "agg_matrices",
        [
            ("idx_agg_matrices_dia_zona", ["dia", "zona"]),
            ("idx_agg_matrices_od", ["Origen", "Destino"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "poly_etapas",
        [
            ("idx_poly_etapas_poly_zona_dia", ["id_polygon", "zona", "dia"]),
            ("idx_poly_etapas_if", ["inicio_norm", "fin_norm"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "poly_matrices",
        [
            ("idx_poly_matrices_poly_zona_dia", ["id_polygon", "zona", "dia"]),
            ("idx_poly_matrices_od", ["Origen", "Destino"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "socio_indicadores",
        [
            ("idx_socio_tabla_dia", ["tabla", "dia"]),
            ("idx_socio_genero_tarifa", ["genero_agregado", "tarifa_agregada"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "resumen_lineas",
        [
            ("idx_res_lineas_dia_linea", ["dia", "id_linea"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "resumen_lineas_ramal",
        [
            ("idx_res_lineas_ramal_dia", ["dia", "id_linea", "id_ramal"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "agg_indicadores",
        [
            ("idx_agg_ind_dia_tipo", ["dia", "tipo_dia"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "poly_indicadores",
        [
            ("idx_poly_ind_poly_dia", ["id_polygon", "dia"]),
        ],
    )

    _maybe_create(
        ctx.dash,
        "datos_particion_modal",
        [
            ("idx_part_modal_dia_genero", ["dia", "genero_agregado"]),
        ],
    )

    # =================
    #   INSUMOS
    # =================
    _maybe_create(
        ctx.insumos,
        "metadata_lineas",
        [
            ("idx_meta_lineas_id", ["id_linea"]),
        ],
    )

    _maybe_create(
        ctx.insumos,
        "poligonos",
        [
            ("idx_poligonos_id", ["id"]),
            ("idx_poligonos_tipo", ["tipo"]),
        ],
    )

    _maybe_create(
        ctx.insumos,
        "zonificaciones",
        [
            ("idx_zonif_zona_id", ["zona", "id"]),
        ],
    )

    _maybe_create(
        ctx.insumos,
        "equivalencias_zonas",
        [
            ("idx_eqz_h3", ["h3"]),
        ],
    )

    # ---- analizar y optimizar ----
    for port in (ctx.data, ctx.dash, ctx.insumos):
        _analyze_optimize(port)


@duracion
def proceso_lineas_deseo(
    ctx: StorageContext,
    etapas=[],
    viajes=[],
    zonificaciones=[],
    equivalencias_zonas=[],
    resoluciones=[6],
):

    preparo_etapas_agregadas(ctx, etapas, viajes, equivalencias_zonas)

    preparo_lineas_deseo(
        ctx,
        etapas,
        viajes,
        res=resoluciones,
        zonificaciones=zonificaciones,
    )  # , 8

    resumen_x_linea(ctx, etapas, viajes)

    construyo_indicadores(ctx, viajes, poligonos=False)

    crea_socio_indicadores(ctx, etapas, viajes)

    guarda_particion_modal(ctx, etapas)

    # imprimo_matrices_od(ctx))


@duracion
def preparo_indicadores_dash(
    ctx: StorageContext,
    corrida="",
    lineas_deseo=True,
    poligonos=True,
    kpis=True,
    resoluciones=[6],
    poligon_id="",
):
    guardo_zonificaciones(ctx)

    zonificaciones = ctx.insumos.get_raw("zonificaciones")
    equivalencias_zonas = ctx.insumos.get_raw("equivalencias_zonas")

    etapas, viajes = load_and_process_data(ctx)

    if lineas_deseo:
        # print("Proceso lineas de deseo")
        proceso_lineas_deseo(
            ctx,
            etapas=etapas.copy(),
            viajes=viajes.copy(),
            zonificaciones=zonificaciones.copy(),
            equivalencias_zonas=equivalencias_zonas.copy(),
            resoluciones=resoluciones,
        )
    if poligonos:
        # print("Proceso Polígonos")
        proceso_poligonos(
            ctx,
            etapas=etapas.copy(),
            viajes=viajes.copy(),
            zonificaciones=zonificaciones.copy(),
            resoluciones=resoluciones,
            poligon_id=poligon_id,
        )

    if kpis:
        # print("Proceso kpis")
        kpis = calculo_kpi_lineas(
            ctx, etapas=etapas.copy(), viajes=viajes.copy()
        )

    crear_indices_unificados(ctx)

    gc.collect()
