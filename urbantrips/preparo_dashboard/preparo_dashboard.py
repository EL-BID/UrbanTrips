import logging
import gc
import os
from datetime import datetime
from itertools import product

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import unidecode
from shapely import wkt
from shapely.geometry import MultiPolygon, Point

from urbantrips.carto import carto
from urbantrips.carto.carto import guardo_zonificaciones
from urbantrips.carto.equivalencias import (
    migrar_equivalencias_zonas,
    sincronizar_equivalencias_dash,
)
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
    calcular_modo_agregado,
    clasificar_distancia_agregada,
    clasificar_genero_agregado,
    clasificar_mes,
    clasificar_rango_hora,
    clasificar_tarifa_agregada_social,
    clasificar_tipo_dia,
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
from urbantrips.utils.paths import get_paths
from urbantrips.utils.utils import (
    calculate_weighted_means,
    duracion,
    leer_alias,
    leer_configs_generales,
    VELOCIDAD_MAXIMA_KMH,
)

logger = logging.getLogger(__name__)

pd.set_option("future.no_silent_downcasting", True)

import warnings

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
    module=r".*urbantrips.*preparo_dashboard",
)




@duracion
def load_and_process_data(ctx: StorageContext):
    """
    Devuelve los DataFrames `etapas` y `viajes` procesados
    sin alterar la lógica de negocio pero con pasos internos
    más rápidos y ordenados.
    """

    # ── 1. leer etapas y viajes (filtrados) ─────────────────────────
    logger.info("load_and_process_data: leyendo etapas desde DB")
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
    logger.info("load_and_process_data: etapas cargadas (%d filas)", len(etapas))

    # derived filter columns via the shared classifiers (same logic as chains)
    etapas["tipo_dia"] = clasificar_tipo_dia(etapas["dia"])
    etapas["mes"] = clasificar_mes(etapas["dia"])
    etapas["rango_hora"] = clasificar_rango_hora(etapas["hora"])
    etapas["distancia_agregada"] = clasificar_distancia_agregada(
        etapas["distance_od"], nivel="etapa"
    )

    logger.info("load_and_process_data: leyendo viajes desde DB")
    viajes = ctx.data.query(
        """
        SELECT v.*, tt.travel_time_min, tt.distance_od, tt.distance_route,
               tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps,
               CAST(v.cant_etapas > 1 AS INTEGER)                      AS transferencia
        FROM viajes v
        LEFT JOIN travel_times_trips tt
        ON v.dia = tt.dia
        AND v.id_tarjeta = tt.id_tarjeta
        AND v.id_viaje = tt.id_viaje
        WHERE v.od_validado = 1
        """
    )
    logger.info("load_and_process_data: viajes cargados (%d filas)", len(viajes))

    viajes["tipo_dia"] = clasificar_tipo_dia(viajes["dia"])
    viajes["mes"] = clasificar_mes(viajes["dia"])
    viajes["rango_hora"] = clasificar_rango_hora(viajes["hora"])
    viajes["distancia_agregada"] = clasificar_distancia_agregada(
        viajes["distance_od"], nivel="viaje"
    )

    logger.info("load_and_process_data: clasificando tarifa y género")
    etapas["tarifa_agregada"] = clasificar_tarifa_agregada_social(etapas["tarifa"])
    etapas["genero_agregado"] = clasificar_genero_agregado(etapas["genero"])
    viajes["tarifa_agregada"] = clasificar_tarifa_agregada_social(viajes["tarifa"])
    viajes["genero_agregado"] = clasificar_genero_agregado(viajes["genero"])

    # ── 2. incorporar travel_time_min y velocidades ─────────────────
    logger.info("load_and_process_data: calculando velocidades y tiempos de viaje")

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
    viajes.loc[viajes["kmh_od"] >= VELOCIDAD_MAXIMA_KMH, "kmh_od"] = np.nan

    # ── 3. datetime window columns (stay in pandas) ────────────────
    logger.info("load_and_process_data: construyendo columnas de ventana temporal")
    viajes["Fecha"] = pd.to_datetime(viajes["dia"] + " " + viajes["tiempo"], format="%Y-%m-%d %H:%M:%S")
    viajes["Fecha_next"] = viajes.groupby(["dia", "id_tarjeta"], observed=True)["Fecha"].shift(-1)
    viajes["diff_time"] = (
        (viajes["Fecha_next"] - viajes["Fecha"]).dt.seconds / 60
    ).round()

    etapas = etapas.merge(
        viajes[["dia", "id_tarjeta", "id_viaje", "transferencia"]], how="left"
    )

    # ── 4. partición modal (clasificador compartido con chains) ─────
    logger.info("load_and_process_data: calculando modo agregado")
    keys_mod = ["dia", "id_tarjeta", "id_viaje"]
    tmp = calcular_modo_agregado(etapas, keys_mod)
    etapas = etapas.merge(tmp[keys_mod + ["modo_agregado"]], on=keys_mod)
    viajes = viajes.merge(
        etapas.groupby(keys_mod + ["modo_agregado"], as_index=False, observed=True)
        .size()
        .drop(columns="size"),
        how="left",
    )

    # ── 5. eliminar registros sin distancias ────────────────────────
    logger.info("load_and_process_data: filtrando registros sin distancias")
    etapas = etapas[etapas["distance_od"].notna()]
    viajes = viajes[viajes["distance_od"].notna()]
    logger.info(
        "load_and_process_data: tras filtro — etapas %d, viajes %d", len(etapas), len(viajes)
    )

    # ── 6. rellenar nulos finales ───────────────────────────────────
    for df in (etapas, viajes):
        df["travel_time_min"] = df["travel_time_min"].fillna(0).astype("float32")
        df["kmh_od"] = df["kmh_od"].fillna(0).astype("float32")

    # ── 7. columnas finales ─────────────────────────────────────────
    logger.info("load_and_process_data: seleccionando columnas finales")
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




def _construyo_indicadores_pandas(ctx: StorageContext, viajes, poligonos=False):

    if poligonos:
        nombre_tabla = "poly_indicadores"
    else:
        nombre_tabla = "agg_indicadores"

    if "id_polygon" not in viajes.columns:
        viajes["id_polygon"] = "NONE"

    ind1 = (
        viajes.groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False, observed=True)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False, observed=True)
        .Valor.mean()
        .round()
    )
    ind1["Indicador"] = "Cantidad de Viajes"
    ind1["Valor"] = ind1.Valor.astype(int)
    ind1["Tipo"] = "General"
    ind1["type_val"] = "int"

    ind2 = (
        viajes[viajes.transferencia == 1]
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False, observed=True)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False, observed=True)
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
        , observed=True)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia", "rango_hora"], as_index=False, observed=True)
        .Valor.mean()
        .round()
    )
    ind3["Indicador"] = "Cantidad de Según Rango Horas"
    ind3["Tot"] = ind3.groupby(
        ["id_polygon", "dia", "mes", "tipo_dia"]
    , observed=True).Valor.transform("sum")
    ind3["Valor"] = (ind3["Valor"] / ind3["Tot"] * 100).round(2)
    ind3["Indicador"] = "Cantidad de Viajes de " + ind3["rango_hora"] + "hs"
    ind3["Tipo"] = "General"
    ind3["type_val"] = "percentage"

    ind4 = (
        viajes.groupby(["id_polygon", "dia", "mes", "tipo_dia", "modo"], as_index=False, observed=True)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(["id_polygon", "dia", "mes", "tipo_dia", "modo"], as_index=False, observed=True)
        .Valor.mean()
        .round()
    )
    ind4["Indicador"] = "Partición Modal"
    ind4["Tot"] = ind4.groupby(
        ["id_polygon", "dia", "mes", "tipo_dia"]
    , observed=True).Valor.transform("sum")
    ind4["Valor"] = (ind4["Valor"] / ind4["Tot"] * 100).round(2)
    ind4 = ind4.sort_values(["id_polygon", "Valor"], ascending=False)
    ind4["Indicador"] = ind4["modo"]
    ind4["Tipo"] = "Modal"
    ind4["type_val"] = "percentage"

    ind9 = (
        viajes.groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "distancia_agregada"],
            as_index=False, observed=True)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Valor"})
        .groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "distancia_agregada"],
            as_index=False, observed=True)
        .Valor.mean()
        .round()
    )
    ind9["Indicador"] = "Partición Modal"
    ind9["Tot"] = ind9.groupby(
        ["id_polygon", "dia", "mes", "tipo_dia"]
    , observed=True).Valor.transform("sum")
    ind9["Valor"] = (ind9["Valor"] / ind9["Tot"] * 100).round(2)
    ind9 = ind9.sort_values(["id_polygon", "Valor"], ascending=False)
    ind9["Indicador"] = "Cantidad de " + ind9["distancia_agregada"]
    ind9["Tipo"] = "General"
    ind9["type_val"] = "percentage"

    ind5 = (
        viajes.groupby(
            ["id_polygon", "dia", "mes", "tipo_dia", "id_tarjeta"], as_index=False
        , observed=True)
        .factor_expansion_linea.first()
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False, observed=True)
        .factor_expansion_linea.sum()
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False, observed=True)
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
        .groupby(["id_polygon", "dia", "mes", "tipo_dia"], as_index=False, observed=True)
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
        .groupby(["id_polygon", "dia", "mes", "tipo_dia", "modo"], as_index=False, observed=True)
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
            as_index=False, observed=True)
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
        , observed=True)
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


def _viajes_poligonos_desde_chains(ctx: StorageContext):
    """Build a trips-like frame per analysis polygon from chains_norm.

    For every polygon in the long-format equivalencias_zonas (tipo
    'poligono' or 'cuenca'), trips are selected by membership of their
    normalized origin/destination cells in the polygon's H3 set:

    - tipo 'poligono' (area): origin OR destination inside the polygon.
    - tipo 'cuenca' (basin):  origin AND destination inside the polygon.

    The polygon id is assigned as the id_polygon column so the regular
    indicator computation can run unchanged (distance_od is the trip-level
    value stored in chains_norm).
    """
    try:
        equivalencias = ctx.insumos.query(
            "SELECT h3, zona, tipo FROM equivalencias_zonas "
            "WHERE tipo IN ('poligono', 'cuenca')"
        )
    except Exception:
        logger.warning(
            "construyo_indicadores: no se pudo leer equivalencias_zonas "
            "(¿falta migrar a formato long?)."
        )
        return pd.DataFrame([])

    if len(equivalencias) == 0:
        return pd.DataFrame([])

    try:
        chains = ctx.dash.query(
            "SELECT dia, mes, tipo_dia, id_tarjeta, id_viaje, "
            "h3_inicio_norm, h3_fin_norm, modo_agregado, rango_hora, "
            "transferencia, distancia_agregada, distance_od, "
            "factor_expansion_linea "
            "FROM chains_norm"
        )
    except Exception:
        logger.warning("construyo_indicadores: la tabla chains_norm no existe en dash.")
        return pd.DataFrame([])

    if len(chains) == 0:
        return pd.DataFrame([])

    frames = []
    for (zona, tipo), grupo in equivalencias.groupby(["zona", "tipo"], observed=True):
        h3_poly = set(grupo["h3"])
        en_origen = chains["h3_inicio_norm"].isin(h3_poly)
        en_destino = chains["h3_fin_norm"].isin(h3_poly)
        mask = (en_origen & en_destino) if tipo == "cuenca" else (en_origen | en_destino)
        if not mask.any():
            continue
        seleccion = chains.loc[mask].drop(columns=["h3_inicio_norm", "h3_fin_norm"])
        seleccion = seleccion.assign(id_polygon=zona)
        frames.append(seleccion)
        logger.info(
            "construyo_indicadores: polígono %s (%s) — %s viajes.",
            zona, tipo, f"{int(mask.sum()):,}",
        )

    if not frames:
        return pd.DataFrame([])

    viajes = pd.concat(frames, ignore_index=True)
    viajes = viajes.rename(columns={"modo_agregado": "modo"})
    return viajes


@duracion
def construyo_indicadores(ctx: StorageContext, viajes=None, poligonos=False):
    """Compute dashboard indicators using DuckDB for single-pass aggregations.

    When ``viajes`` is omitted and ``poligonos=True``, trips are selected
    on the fly from chains_norm + equivalencias_zonas per analysis polygon
    instead of receiving a pre-filtered frame.
    """
    nombre_tabla = "poly_indicadores" if poligonos else "agg_indicadores"

    desde_chains = False
    if poligonos and viajes is None:
        viajes = _viajes_poligonos_desde_chains(ctx)
        desde_chains = True
        if len(viajes) == 0:
            logger.info(
                "construyo_indicadores: sin polígonos en equivalencias_zonas "
                "o sin datos en chains_norm — no se generan poly_indicadores."
            )
            return

    # Source setup. The five aggregation blocks below all read FROM _vproc; the
    # only thing that changes is where _vproc comes from:
    #  - non-polygon production (viajes is None): materialise the proc-CTE once
    #    into a DuckDB temp table in the data DB. RAM is bounded by memory_limit;
    #    the 26M-row viajes frame is never built in pandas.
    #  - polygon path / explicit frame (tests): register the in-RAM frame as _vproc.
    _con = None
    if viajes is None:
        from urbantrips.preparo_dashboard.sql_queries import (
            materializar_proc_tables, VIAJES_PROC_MAT,
        )
        materializar_proc_tables(ctx)
        ctx.data.execute(
            f"CREATE OR REPLACE TEMP TABLE _vproc AS "
            f"SELECT *, 'NONE' AS id_polygon FROM {VIAJES_PROC_MAT}"
        )
        run = ctx.data.query
    else:
        if "id_polygon" not in viajes.columns:
            viajes = viajes.copy()
            viajes["id_polygon"] = "NONE"
        _con = duckdb.connect()
        _con.register("_vproc", viajes)
        run = lambda body: _con.execute(body).df()  # noqa: E731

    KEYS = ["id_polygon", "dia", "mes", "tipo_dia"]

    # ── 1. Base group ─────────────────────────────────────────────────────────
    logger.info("construyo_indicadores: calculando indicadores base")
    base = run("""
        SELECT
            id_polygon, dia, mes, tipo_dia,
            ROUND(SUM(factor_expansion_linea))                                        AS total_viajes,
            ROUND(SUM(CASE WHEN transferencia = 1 THEN factor_expansion_linea
                          ELSE 0 END))                                                AS con_transferencia,
            ROUND(SUM(distance_od * factor_expansion_linea)
                  / NULLIF(SUM(factor_expansion_linea), 0), 2)                        AS dist_prom
        FROM _vproc
        GROUP BY id_polygon, dia, mes, tipo_dia
    """)

    ind1 = base[KEYS + ["total_viajes"]].rename(columns={"total_viajes": "Valor"})
    ind1["Indicador"] = "Cantidad de Viajes"
    ind1["Valor"] = ind1.Valor.astype(int)
    ind1["Tipo"] = "General"
    ind1["type_val"] = "int"

    ind2 = base[KEYS + ["con_transferencia", "total_viajes"]].copy()
    ind2["Valor"] = (
        ind2["con_transferencia"] / ind2["total_viajes"].replace(0, float("nan")) * 100
    ).round(2)
    ind2 = ind2[KEYS + ["Valor"]]
    ind2["Indicador"] = "Cantidad de Viajes con Transferencia"
    ind2["Tipo"] = "General"
    ind2["type_val"] = "percentage"

    ind6 = base[KEYS + ["dist_prom"]].rename(columns={"dist_prom": "Valor"})
    ind6["Tipo"] = "Distancias"
    ind6["Indicador"] = "Distancia Promedio (kms)"
    ind6["type_val"] = "float"

    # ── 2. Usuarios ───────────────────────────────────────────────────────────
    logger.info("construyo_indicadores: calculando usuarios únicos")
    usuarios = run("""
        SELECT id_polygon, dia, mes, tipo_dia,
            ROUND(SUM(first_fex)) AS Valor
        FROM (
            -- one expansion factor per card. arg_min(fex, id_viaje) = the card's
            -- first trip, matching the pandas oracle's groupby(...).first() and,
            -- unlike ANY_VALUE, deterministic + independent of source row order.
            SELECT id_polygon, dia, mes, tipo_dia,
                arg_min(factor_expansion_linea, id_viaje) AS first_fex
            FROM _vproc
            GROUP BY id_polygon, dia, mes, tipo_dia, id_tarjeta
        )
        GROUP BY id_polygon, dia, mes, tipo_dia
    """)
    ind5 = usuarios[KEYS + ["Valor"]]
    ind5["Indicador"] = "Cantidad de Usuarios"
    ind5["Tipo"] = "General"
    ind5["type_val"] = "int"

    # ── 3. By rango_hora ──────────────────────────────────────────────────────
    logger.info("construyo_indicadores: calculando distribución por rango hora")
    by_hora = run("""
        WITH totals AS (
            SELECT id_polygon, dia, mes, tipo_dia,
                SUM(factor_expansion_linea) AS total
            FROM _vproc
            GROUP BY id_polygon, dia, mes, tipo_dia
        )
        SELECT v.id_polygon, v.dia, v.mes, v.tipo_dia, v.rango_hora,
            ROUND(SUM(v.factor_expansion_linea) / t.total * 100, 2) AS Valor
        FROM _vproc v
        JOIN totals t USING (id_polygon, dia, mes, tipo_dia)
        GROUP BY v.id_polygon, v.dia, v.mes, v.tipo_dia, v.rango_hora, t.total
    """)

    ind3 = by_hora.copy()
    ind3["Indicador"] = "Cantidad de Viajes de " + ind3["rango_hora"] + "hs"
    ind3["Tipo"] = "General"
    ind3["type_val"] = "percentage"
    ind3 = ind3.drop(columns=["rango_hora"])

    # ── 4. By modo ────────────────────────────────────────────────────────────
    logger.info("construyo_indicadores: calculando distribución modal")
    by_modo = run("""
        WITH totals AS (
            SELECT id_polygon, dia, mes, tipo_dia,
                SUM(factor_expansion_linea) AS total
            FROM _vproc
            GROUP BY id_polygon, dia, mes, tipo_dia
        )
        SELECT v.id_polygon, v.dia, v.mes, v.tipo_dia, v.modo,
            ROUND(SUM(v.factor_expansion_linea) / t.total * 100, 2)           AS pct,
            ROUND(SUM(v.distance_od * v.factor_expansion_linea)
                  / NULLIF(SUM(v.factor_expansion_linea), 0), 2)              AS dist_prom_modo
        FROM _vproc v
        JOIN totals t USING (id_polygon, dia, mes, tipo_dia)
        GROUP BY v.id_polygon, v.dia, v.mes, v.tipo_dia, v.modo, t.total
    """)

    ind4 = by_modo[KEYS + ["modo", "pct"]].copy()
    ind4 = ind4.sort_values(KEYS + ["pct"], ascending=[True] * 4 + [False])
    ind4["Indicador"] = ind4["modo"]
    ind4["Tipo"] = "Modal"
    ind4["type_val"] = "percentage"
    ind4 = ind4.rename(columns={"pct": "Valor"}).drop(columns=["modo"])

    ind7 = by_modo[KEYS + ["modo", "dist_prom_modo"]].copy()
    ind7["Indicador"] = "Distancia Promedio (" + ind7["modo"] + ") (kms)"
    ind7["Tipo"] = "Distancias"
    ind7["type_val"] = "float"
    ind7 = ind7.rename(columns={"dist_prom_modo": "Valor"}).drop(columns=["modo"])

    # ── 5. By distancia_agregada ──────────────────────────────────────────────
    logger.info("construyo_indicadores: calculando distribución por distancia")
    by_dist = run("""
        WITH totals AS (
            SELECT id_polygon, dia, mes, tipo_dia,
                SUM(factor_expansion_linea) AS total
            FROM _vproc
            GROUP BY id_polygon, dia, mes, tipo_dia
        )
        SELECT v.id_polygon, v.dia, v.mes, v.tipo_dia, v.distancia_agregada,
            ROUND(SUM(v.factor_expansion_linea) / t.total * 100, 2)           AS pct,
            ROUND(SUM(v.distance_od * v.factor_expansion_linea)
                  / NULLIF(SUM(v.factor_expansion_linea), 0), 2)              AS dist_prom_dist
        FROM _vproc v
        JOIN totals t USING (id_polygon, dia, mes, tipo_dia)
        GROUP BY v.id_polygon, v.dia, v.mes, v.tipo_dia, v.distancia_agregada, t.total
    """)

    ind9 = by_dist[KEYS + ["distancia_agregada", "pct"]].copy()
    ind9 = ind9.sort_values(KEYS + ["pct"], ascending=[True] * 4 + [False])
    ind9["Indicador"] = "Cantidad de " + ind9["distancia_agregada"]
    ind9["Tipo"] = "General"
    ind9["type_val"] = "percentage"
    ind9 = ind9.rename(columns={"pct": "Valor"}).drop(columns=["distancia_agregada"])

    ind8 = by_dist[KEYS + ["distancia_agregada", "dist_prom_dist"]].copy()
    ind8["Indicador"] = "Distancia Promedio " + ind8["distancia_agregada"]
    ind8["Tipo"] = "Distancias"
    ind8["type_val"] = "float"
    ind8 = ind8.rename(columns={"dist_prom_dist": "Valor"}).drop(columns=["distancia_agregada"])

    # release the source — every aggregate is now materialised in pandas
    if _con is not None:
        _con.close()
    else:
        ctx.data.execute("DROP TABLE IF EXISTS _vproc")

    # ── 6. Combine, merge history, add "Todos" aggregate ─────────────────────
    logger.info("construyo_indicadores: consolidando y guardando indicadores")
    indicadores = pd.concat(
        [ind1, ind5, ind2, ind3, ind6, ind9, ind7, ind8, ind4], ignore_index=True
    )

    if desde_chains:
        # drop groups whose metric had no valid values (all-NaN weighted means)
        indicadores = indicadores[indicadores["Valor"].notna()]

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
            indicadores[["id_polygon", "dia", "mes", "tipo_dia",
                          "Tipo", "Indicador", "type_val", "Valor"]],
            indicadores_ant,
        ],
        ignore_index=True,
    )

    indicadores_todos = (
        indicadores.groupby(
            ["id_polygon", "Tipo", "Indicador", "type_val"], as_index=False, observed=True
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

    tabla_destino = "poly_indicadores" if poligonos else "agg_indicadores"
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
        "lat1, lon1, lat4, lon4, distancia, travel_time_min, travel_speed, "
        "factor_expansion_linea, dia "
        "FROM agg_matrices"
    )

    agg_transferencias = True
    agg_modo = True
    agg_hora = True
    agg_distancia = True

    matrices_all.loc[matrices_all.travel_time_min == 0, "travel_time_min"] = np.nan
    matrices_all.loc[matrices_all.travel_speed == 0, "travel_speed"] = np.nan
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
            as_index=False, observed=True)[
            [
                "lat1",
                "lon1",
                "lat4",
                "lon4",
                "distancia",
                "travel_time_min",
                "travel_speed",
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
        weight_col=["distancia", "travel_time_min", "travel_speed"],
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

        db_path = str(get_paths().output_dir / "matrices" / f"{savefile}.xlsx")
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

        db_path2 = str(get_paths().output_dir / "matrices" / f"{savefile}_normalizada.xlsx")
        od_heatmap.reset_index().fillna("").to_excel(db_path2, index=False)

        logger.debug("Saved %s --- %s", db_path, db_path2)


@duracion
def crea_socio_indicadores(ctx: StorageContext):
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, ETAPAS_PROC_MAT, VIAJES_PROC_MAT,
    )
    materializar_proc_tables(ctx)

    logger.info("crea_socio_indicadores: calculando medias ponderadas de viajes")
    socio_indicadores = pd.DataFrame([])

    # First-pass weighted means run inside the data DB over the proc-CTEs (the
    # 26M/30M-row frames are never materialised). zero_to_nan=[travel_time_min,
    # kmh_od] reproduces the legacy pre-null of those two columns (the proc-CTE
    # 0-fills them); distance_od/cant_etapas/diff_time keep their zeros, as before.
    viajesx = calculate_weighted_means(
        None,
        aggregate_cols=["dia", "mes", "tipo_dia", "genero_agregado", "tarifa_agregada"],
        weighted_mean_cols=[
            "distance_od",
            "travel_time_min",
            "kmh_od",
            "cant_etapas",
            "diff_time",
        ],
        zero_to_nan=["travel_time_min", "kmh_od"],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
        query_fn=ctx.data.query,
        source=VIAJES_PROC_MAT,
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
        None,
        aggregate_cols=[
            "dia",
            "mes",
            "tipo_dia",
            "genero_agregado",
            "tarifa_agregada",
            "modo",
        ],
        weighted_mean_cols=["distance_od", "travel_time_min", "kmh_od"],
        zero_to_nan=["travel_time_min", "kmh_od"],
        weight_col="factor_expansion_linea",
        var_fex_summed=True,
        query_fn=ctx.data.query,
        source=ETAPAS_PROC_MAT,
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
    logger.info("crea_socio_indicadores: calculando medias ponderadas de etapas")
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
    logger.info("crea_socio_indicadores: calculando viajes promedio por usuario")
    # Per-card trip counts. The folded single-SQL version over the WIDE materialised
    # table with STRING_AGG(DISTINCT ... ORDER BY) was ~13x slower (37 min vs ~3 min):
    # the ORDER BY sort per card-group + scanning the 22-col table twice dominated.
    # Revert to the original pandas path (STRING_AGG over a NARROW 3-col frame +
    # pandas groupby), fed by a narrow 8-col projection of the materialised table
    # (~26M x 8 ≈ 1.5 GB). The multi-tariff label order is non-deterministic again
    # (as in the legacy); affects only the ~35 multi-tariff rows of this table.
    viajes_user = ctx.data.query(
        f"SELECT dia, mes, tipo_dia, id_tarjeta, genero_agregado, tarifa_agregada, "
        f"factor_expansion_tarjeta, factor_expansion_linea FROM {VIAJES_PROC_MAT}"
    )
    _userx_clean = viajes_user[["dia", "id_tarjeta"]].copy()
    _userx_clean["tarifa_agregada"] = viajes_user["tarifa_agregada"].str.replace("-", "")
    _tarifa_agg = duckdb.sql("""
        SELECT dia, id_tarjeta,
               COALESCE(STRING_AGG(DISTINCT NULLIF(tarifa_agregada, ''), '-'), '-') AS tarifa_agregada_agg
        FROM _userx_clean
        GROUP BY dia, id_tarjeta
    """).df()
    userx = viajes_user[
        ["dia", "mes", "tipo_dia", "id_tarjeta", "genero_agregado",
         "factor_expansion_tarjeta", "factor_expansion_linea"]
    ].merge(_tarifa_agg, how="left")
    userx = (
        userx.groupby(
            ["dia", "mes", "tipo_dia", "id_tarjeta", "genero_agregado", "tarifa_agregada_agg"],
            as_index=False, observed=True)
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
    logger.info("crea_socio_indicadores: guardando socio_indicadores, distribución y viajes_hora")
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

    # round_even == pandas .round() half-to-even; CAST to BIGINT == .astype(int).
    hora = ctx.data.query(f"""
        SELECT dia, modo, hora,
               CAST(round_even(SUM(factor_expansion_linea), 0) AS BIGINT) AS viajes
        FROM {ETAPAS_PROC_MAT}
        GROUP BY dia, modo, hora
    """)

    horaT = hora.groupby(["dia", "hora"], as_index=False, observed=True).viajes.sum()
    horaT["modo"] = "Todos"

    hora = pd.concat([hora, horaT], ignore_index=True)

    dist = ctx.data.query(f"""
        SELECT dia, modo, CAST(round_even(distance_od, 0) AS BIGINT) AS dist,
               CAST(round_even(SUM(factor_expansion_linea), 0) AS BIGINT) AS viajes
        FROM {ETAPAS_PROC_MAT}
        GROUP BY dia, modo, CAST(round_even(distance_od, 0) AS BIGINT)
    """)

    distT = dist.groupby(["dia", "dist"], as_index=False, observed=True).viajes.sum()
    distT["modo"] = "Todos"

    dist = pd.concat([dist, distT], ignore_index=True)

    dist.columns = ["Día", "Modo", "Distancia (kms)", "Viajes"]
    hora.columns = ["Día", "Modo", "Hora", "Viajes"]

    ctx.dash.save_indicator(dist, "distribucion")
    ctx.dash.save_indicator(hora, "viajes_hora")


@duracion
def preparo_etapas_agregadas(ctx: StorageContext, etapas, viajes, equivalencias_zonas):
    """Deprecated stub kept for signature compatibility.

    The etapas_agregadas / viajes_agregados / transferencias_agregadas tables
    were replaced by on-the-fly dashboard aggregation over chains_norm +
    equivalencias_zonas (see urbantrips.preparo_dashboard.chains).
    """
    logger.info(
        "preparo_etapas_agregadas: skipped — replaced by on-the-fly "
        "aggregation over chains_norm + equivalencias_zonas."
    )


@duracion
def preparo_lineas_deseo(
    ctx: StorageContext,
    etapas_selec,
    viajes_selec,
    polygons_h3="",
    poligonos="",
    res=6,
    zonificaciones=[],
):
    """Deprecated stub kept for signature compatibility.

    The precomputed agg_etapas / agg_matrices / poly_etapas / poly_matrices
    tables were replaced by on-the-fly dashboard aggregation over chains_norm
    + equivalencias_zonas (see urbantrips.preparo_dashboard.chains).
    """
    logger.info(
        "preparo_lineas_deseo: skipped — replaced by on-the-fly aggregation "
        "over chains_norm + equivalencias_zonas."
    )


@duracion
def guarda_particion_modal(ctx: StorageContext):
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, ETAPAS_PROC_MAT,
    )
    materializar_proc_tables(ctx)

    # modo set, sorted — matches pandas get_dummies(modo) column order.
    modos = ctx.data.query(
        f"SELECT DISTINCT modo FROM {ETAPAS_PROC_MAT} ORDER BY modo"
    )["modo"].tolist()

    # SUM(CASE WHEN modo='X' ...) per modo == get_dummies(modo) summed per trip.
    dummy_cols = ", ".join(
        f"SUM(CASE WHEN modo = '{m}' THEN 1 ELSE 0 END) AS \"{m}\"" for m in modos
    )
    dummy_names = ", ".join(f'"{m}"' for m in modos)

    # Reproduces the legacy two-level groupby + left-merge, pushed into the DB:
    #   g1: one fex per (dia,mes,tipo_dia,genero,trip)  [groupby.mean]
    #   g2: modo-count vector per (dia,trip)            [groupby[dummies].sum]
    #   final: SUM(fex) per (dia,mes,tipo_dia,genero, modo-vector)
    etapas_modos = ctx.data.query(
        f"""
        WITH g1 AS (
            SELECT dia, mes, tipo_dia, genero_agregado, id_tarjeta, id_viaje,
                   AVG(factor_expansion_linea) AS factor_expansion_linea
            FROM {ETAPAS_PROC_MAT}
            GROUP BY dia, mes, tipo_dia, genero_agregado, id_tarjeta, id_viaje
        ),
        g2 AS (
            SELECT dia, id_tarjeta, id_viaje, {dummy_cols}
            FROM {ETAPAS_PROC_MAT}
            GROUP BY dia, id_tarjeta, id_viaje
        ),
        joined AS (
            SELECT g1.dia, g1.mes, g1.tipo_dia, g1.genero_agregado,
                   g1.factor_expansion_linea, {dummy_names}
            FROM g1 LEFT JOIN g2 USING (dia, id_tarjeta, id_viaje)
        )
        SELECT dia, mes, tipo_dia, genero_agregado, {dummy_names},
               SUM(factor_expansion_linea) AS factor_expansion_linea
        FROM joined
        GROUP BY dia, mes, tipo_dia, genero_agregado, {dummy_names}
        """
    )

    etapas_modos = etapas_modos.rename(columns={m: m.capitalize() for m in modos})
    replace_dash_partition(ctx, etapas_modos, "datos_particion_modal", ["dia"])



@duracion
def resumen_x_linea(ctx: StorageContext):
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, ETAPAS_PROC_MAT,
    )
    materializar_proc_tables(ctx)

    # Only the columns agrego_lineas reads — gps and transacciones are the two
    # largest tables in the run; loading them whole multiplies peak RSS.
    logger.info("resumen_x_linea: cargando gps, lineas, kpis, servicios, transacciones")
    gps = ctx.data.query("SELECT dia, id_linea, id_ramal, interno FROM gps")
    lineas = ctx.insumos.get_metadata_lineas()
    kpis = ctx.data.get_raw("kpi_by_day_line")
    servicios = ctx.data.get_raw("services")
    lineas = lineas[["id_linea", "nombre_linea", "empresa"]].sort_values(["id_linea"])

    trx = ctx.data.query(
        "SELECT dia, id_linea, id_ramal, modo, interno, factor_expansion FROM transacciones"
    )

    metric_cols = [
        "transacciones",
        "distancia_media", "travel_time_min", "kmh_od",
        "cant_internos_en_trx", "cant_internos_en_gps",
        "tot_veh", "tot_km", "tot_pax",
        "dmt_mean_od", "dmt_median_od",
        "pvd", "kvd", "ipk_route", "fo_mean_od", "fo_median_od",
    ]

    # Resumen por línea
    logger.info("resumen_x_linea: agregando por línea")
    all_linea = agrego_lineas(
        ["dia", "id_linea"], trx, None, gps, servicios, kpis, lineas,
        etapas_query_fn=ctx.data.query, etapas_source=ETAPAS_PROC_MAT,
    )
    all_linea["mes"] = all_linea["dia"].str[:7]
    metric_cols_linea = [c for c in metric_cols if c in all_linea.columns]
    all_linea = (
        all_linea
        .groupby(["dia", "mes", "id_linea", "nombre_linea", "empresa", "modo"], as_index=False, observed=True)
        [metric_cols_linea]
        .mean()
        .round(2)
    )
    replace_dash_partition(ctx, all_linea, "resumen_lineas", ["dia"])

    # Resumen por línea y ramal
    logger.info("resumen_x_linea: agregando por línea y ramal")
    all_ramal = agrego_lineas(
        ["dia", "id_linea", "id_ramal"], trx, None, gps, servicios, kpis, lineas,
        etapas_query_fn=ctx.data.query, etapas_source=ETAPAS_PROC_MAT,
    )
    all_ramal["mes"] = all_ramal["dia"].str[:7]
    metric_cols_ramal = [c for c in metric_cols if c in all_ramal.columns]
    all_ramal = (
        all_ramal
        .groupby(["dia", "mes", "id_linea", "id_ramal", "nombre_linea", "empresa", "modo"], as_index=False, observed=True)
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
    """Deprecated stub kept for signature compatibility.

    Polygon filtering moved to on-the-fly dashboard queries over chains_norm
    + equivalencias_zonas (tipo 'poligono'/'cuenca'). Polygon indicators are
    produced by construyo_indicadores(ctx, poligonos=True).
    """
    logger.info(
        "proceso_poligonos: skipped — replaced by on-the-fly aggregation "
        "over chains_norm + equivalencias_zonas."
    )


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

@duracion
def crear_indices_unificados(ctx: StorageContext):
    """
    Crea índices en las bases SQLite usadas por el pipeline UrbanTrips:
    - data (etapas, viajes, transacciones, gps, services, kpi_by_day_line)
    - dash (agg_etapas, agg_matrices, poly_*, indicadores, resumen_lineas, etc.)
    - insumos (metadata_lineas, poligonos, zonificaciones, equivalencias_zonas)
    Aplica PRAGMAs de rendimiento y ANALYZE/optimize.
    """

    def _maybe_create(port, table, spec_list):
        # Auditoría empírica 2026-07-18: los índices ART son puro costo en DuckDB (no
        # aceleran ninguna query — los sirve el zonemap/hash join — y se mantienen en
        # cada escritura). Vestigio SQLite: no se crean más. Se conserva el
        # ANALYZE/optimize de esta función (sí actualiza estadísticas del optimizador).
        return

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
    # NOTA (2026-06-24): índices ART secundarios sobre las tablas GRANDES de `data`
    # (etapas, viajes, transacciones, gps, services) REMOVIDOS. Evidencia con EXPLAIN
    # sobre la corrida real (63M etapas): DuckDB hace SEQ_SCAN para las queries del
    # dashboard — incluida la única consumidora de etapas en runtime (pág. 8
    # "Estimar demanda", filtros h3_o/h3_d/od_validado/hora/dia) — y NUNCA usa estos
    # índices (ni siquiera para una igualdad simple `h3_o = '...'`). Construirlos
    # costaba ~57 min al final de la corrida, sin beneficio. Además viajes/
    # transacciones/gps/services no tienen ningún consumidor en runtime. Los índices
    # del schema (idx_etapas_batch, idx_etapas_dia_*, etc.) se mantienen aparte; las
    # tablas `dash` (chicas) conservan los suyos más abajo.

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

    # chains_norm (26.6M filas): índices ART REMOVIDOS (2026-06-26). EXPLAIN sobre la
    # corrida real confirma que DuckDB hace SEQ_SCAN para todos los filtros del
    # dashboard (dia / h3_inicio_norm / modo_agregado / transferencia / ...) y nunca
    # usa estos índices —ni para una igualdad muy selectiva como h3_inicio_norm = '...'—.
    # Construir los 8 índices era el grueso de los ~15 min de esta función, sin
    # beneficio, y además frenaba los inserts/upserts a chains_norm.

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

    # equivalencias_zonas (7.3M filas): índices ART REMOVIDOS (2026-06-26). El comentario
    # previo los daba por "critical" para el join chains_norm × equivalencias y los
    # IN-subqueries por zona, pero EXPLAIN confirma SEQ_SCAN tanto para `WHERE zona = x`
    # como para `WHERE h3 = x` — DuckDB no los usa (es columnar; los joins son hash).

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

    resumen_x_linea(ctx)

    construyo_indicadores(ctx, viajes, poligonos=False)

    crea_socio_indicadores(ctx)

    guarda_particion_modal(ctx)

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
    """Prepare dashboard inputs.

    Builds equivalencias_zonas (config resolution + res 10 for chains_norm),
    runs the chains_norm pipeline day by day, then computes the non-spatial
    indicator tables. The lineas_deseo / resoluciones / poligon_id parameters
    are kept for signature compatibility but no longer drive precomputed tables.
    """
    from urbantrips.preparo_dashboard.chains import (
        procesar_pipeline_por_dia,
        RES_CHAINS_NORM,
    )
    from urbantrips.datamodel.trips import verificar_integridad_viajes_etapas

    # fail fast if viajes is stale relative to etapas: indicators (from
    # viajes) and chains_norm/maps (from etapas) would silently diverge
    verificar_integridad_viajes_etapas(ctx)

    guardo_zonificaciones(ctx)

    # one-shot wide -> long migration; no-op when already migrated
    migrar_equivalencias_zonas(ctx=ctx)

    # refresh the dash copy so dashboards join chains_norm x equivalencias
    # in a single SQL connection
    sincronizar_equivalencias_dash(ctx=ctx)

    # build chains_norm (trip-level OD table at res 10 used by all dashboard pages)
    procesar_pipeline_por_dia(res=RES_CHAINS_NORM, guardar=True, ctx=ctx)

    # All indicator consumers below self-source from the data DB via the proc-CTEs
    # (sql_queries.py); the 30.9M-etapas / 26.6M-viajes frames are never built in
    # pandas — RAM stays bounded by DuckDB's memory_limit, so this scales to a
    # full month. load_and_process_data() is retained only as a parity oracle for
    # the unit tests. The proc relations are materialised ONCE here (the heavy
    # etapas⋈travel_times join) and reused by every consumer; dropped at the end.
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, drop_proc_tables,
    )
    materializar_proc_tables(ctx, replace=True)
    try:
        resumen_x_linea(ctx)

        # non-polygon indicators self-source from the data DB (proc-CTE temp table);
        # no longer needs the in-RAM viajes frame.
        construyo_indicadores(ctx, poligonos=False)

        if poligonos:
            # trips per polygon are selected on the fly from chains_norm
            construyo_indicadores(ctx, poligonos=True)

        crea_socio_indicadores(ctx)

        guarda_particion_modal(ctx)

        if kpis:
            kpis = calculo_kpi_lineas(ctx)
    finally:
        drop_proc_tables(ctx)

    crear_indices_unificados(ctx)

    gc.collect()
