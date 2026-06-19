import logging
import numpy as np
import pandas as pd
import h3
from datetime import datetime

from urbantrips.utils.utils import (
    duracion,
    leer_configs_generales,
    agrego_indicador,
    modos_con_ramal,
    id_ramal_efectivo,
    RAMAL_SENTINEL,
)
from urbantrips.storage.context import StorageContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ordenamiento
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Imputación de destino potencial
# ---------------------------------------------------------------------------

def imputar_destino_potencial(etapas):
    """
    Imputa para cada (dia, id_tarjeta) el origen de la etapa siguiente
    como destino potencial. La última etapa del día recibe el origen de
    la primera (hipótesis de viaje circular). Si O == D, se deja como NaN.
    Usa orden_trx cuando está disponible para un ordenamiento más confiable.
    """

    grp = etapas.groupby(["dia", "id_tarjeta"])["h3_o"]

    etapas["h3_d"] = grp.shift(-1)
    etapas["h3_d"] = etapas["h3_d"].fillna(grp.transform("first"))
    etapas.loc[etapas["h3_o"] == etapas["h3_d"], "h3_d"] = np.nan

    return etapas


# ---------------------------------------------------------------------------
# Validación contra matriz_validacion
# ---------------------------------------------------------------------------

def validar_destinos(destinos, ctx: StorageContext):
    """
    Valida destinos potenciales contra la matriz de validación de la DB.
    Delega en _validar_destinos_con_matriz para facilitar el testing.

    El cruce opera por id_linea_agg (red agregada). El id_ramal entra en la clave
    solo para los modos que validan por ramal (modo_valida_ramal); para el resto
    se usa el id_ramal efectivo (centinela), o sea se valida por id_linea_agg.
    """
    configs = leer_configs_generales(autogenerado=False)
    modos_ramal = modos_con_ramal(configs)
    mv = ctx.insumos.get_matrix_validation()
    matriz_validacion = mv[
        ["id_linea_agg", "id_ramal", "area_influencia"]
    ].drop_duplicates()
    return _validar_destinos_con_matriz(destinos, matriz_validacion, modos_ramal)


def _validar_destinos_con_matriz(destinos, matriz_validacion, modos_ramal=None):
    """
    Valida destinos potenciales contra una matriz de validación dada.
    Versión separada de validar_destinos() para facilitar el testing sin DB.

    El join discrimina por id_ramal solo en los modos que validan por ramal
    (modos_ramal); para el resto usa el id_ramal efectivo (centinela), o sea
    valida por id_linea_agg como hasta hoy. La matriz guarda id_ramal NULL en
    esos modos, que aquí se rellena con el centinela para que el merge coincida.

    Agrega drop_duplicates antes del merge para evitar multiplicar filas y
    verifica con un assert que el tamaño del DataFrame no cambia.
    """
    modos_ramal = modos_ramal or set()
    n_orig = len(destinos)

    destinos = destinos.copy()
    destinos["id_ramal"] = id_ramal_efectivo(
        destinos["modo"], destinos["id_ramal"], modos_ramal
    )

    matriz_validacion = matriz_validacion.copy()
    matriz_validacion["id_ramal"] = (
        matriz_validacion["id_ramal"].fillna(RAMAL_SENTINEL).astype("int64")
    )

    join_cols = ["h3_o", "h3_d", "id_linea_agg", "id_ramal"]

    pares_od_linea = (
        destinos.reindex(columns=["h3_o", "h3_d", "id_linea_agg", "id_ramal"])
        .drop_duplicates()
    )

    pares_od_linea = pares_od_linea.merge(
        matriz_validacion.drop_duplicates(),
        how="left",
        left_on=["id_linea_agg", "id_ramal", "h3_d"],
        right_on=["id_linea_agg", "id_ramal", "area_influencia"],
    )
    pares_od_linea["od_validado"] = pares_od_linea["area_influencia"].notna()

    # Si un par od tiene múltiples matches, consolidar con max (True prevalece)
    pares_od_linea = (
        pares_od_linea.groupby(join_cols, as_index=False)["od_validado"].max()
    )

    destinos = destinos.merge(pares_od_linea, how="left", on=join_cols)

    assert len(destinos) == n_orig, (
        f"El merge multiplicó filas: {n_orig} → {len(destinos)}"
    )

    destinos = destinos.reindex(columns=["id", "h3_d", "od_validado"])
    destinos["od_validado"] = destinos["od_validado"].fillna(0).astype(int)

    return destinos


# ---------------------------------------------------------------------------
# Minimización de distancia
# ---------------------------------------------------------------------------

def _build_latlng_maps(cells):
    """Builds {cell: lat} and {cell: lng} dicts for a collection of unique h3 cells."""
    lat_map, lng_map = {}, {}
    for cell in cells:
        if cell and pd.notna(cell):
            try:
                lat, lng = h3.cell_to_latlng(cell)
                lat_map[cell] = lat
                lng_map[cell] = lng
            except Exception:
                pass
    return lat_map, lng_map


def imputar_destino_min_distancia(etapas, ctx: StorageContext):
    """
    Para cada etapa, busca la parada de su línea que minimiza la distancia
    h3 al destino potencial (origen de la etapa siguiente).
    Delega en _imputar_destino_min_distancia_con_matriz para facilitar el testing.

    El cruce opera por id_linea_agg; el id_ramal entra en la clave solo para los
    modos que validan por ramal (modo_valida_ramal), via el id_ramal efectivo.
    """
    configs = leer_configs_generales(autogenerado=False)
    modos_ramal = modos_con_ramal(configs)
    matriz_validacion = ctx.insumos.get_matrix_validation()
    return _imputar_destino_min_distancia_con_matriz(
        etapas, matriz_validacion, modos_ramal
    )


def _imputar_destino_min_distancia_con_matriz(etapas, matriz_validacion, modos_ramal=None):
    """
    Versión testeable sin DB de imputar_destino_min_distancia.

    Approach:
    1. Arma pares únicos (id_linea_agg, id_ramal, h3_d_potencial)
    2. Merge con matriz_validacion para obtener paradas candidatas
    3. Calcula distancia euclidiana sobre lat/lng (vectorizado) en vez de
       h3.grid_distance por par (loop Python). Convierte h3 → lat/lng una vez
       por celda única y luego hace aritmética numpy.
    4. Groupby para quedarse con la de menor distancia
    5. Merge de vuelta a etapas

    El id_ramal se incluye en las llaves solo para los modos que validan por
    ramal (modos_ramal); para el resto se usa el id_ramal efectivo (centinela),
    o sea se imputa sobre toda la red agregada (id_linea_agg) como hasta hoy.
    """
    modos_ramal = modos_ramal or set()

    etapas = etapas.copy()
    etapas["id_ramal"] = id_ramal_efectivo(
        etapas["modo"], etapas["id_ramal"], modos_ramal
    )

    matriz_validacion = matriz_validacion.copy()
    matriz_validacion["id_ramal"] = (
        matriz_validacion["id_ramal"].fillna(RAMAL_SENTINEL).astype("int64")
    )

    clave = ["id_linea_agg", "id_ramal"]

    lag_etapas = (
        etapas.reindex(columns=["id"] + clave + ["h3_d"])
        .rename(columns={"h3_d": "lag_etapa"})
    )

    pares_unicos = (
        lag_etapas.reindex(columns=clave + ["lag_etapa"])
        .drop_duplicates()
    )

    candidatas = pares_unicos.merge(
        matriz_validacion[clave + ["area_influencia", "parada"]],
        left_on=clave + ["lag_etapa"],
        right_on=clave + ["area_influencia"],
        how="left",
    ).rename(columns={"parada": "h3_d"}).drop(columns=["area_influencia"])

    candidatas = candidatas.dropna(subset=["h3_d"])

    if len(candidatas) > 0:
        unique_cells = pd.unique(
            np.concatenate([candidatas["lag_etapa"].values, candidatas["h3_d"].values])
        )
        lat_map, lng_map = _build_latlng_maps(unique_cells)

        dlat = candidatas["lag_etapa"].map(lat_map) - candidatas["h3_d"].map(lat_map)
        dlng = candidatas["lag_etapa"].map(lng_map) - candidatas["h3_d"].map(lng_map)
        candidatas = candidatas.copy()
        # equirectangular correction: a longitude degree spans cos(lat) times
        # a latitude degree, so without it the ranking is biased toward
        # north-south candidates (~20% at lat -34)
        cos_lat = np.cos(np.deg2rad(candidatas["lag_etapa"].map(lat_map).mean()))
        candidatas["distance_od"] = np.sqrt(
            dlat.values ** 2 + (dlng.values * cos_lat) ** 2
        )

        candidatas = (
            candidatas
            .sort_values("distance_od")
            .drop_duplicates(subset=clave + ["lag_etapa"], keep="first")
        )

    resultado = lag_etapas.merge(
        candidatas[clave + ["lag_etapa", "h3_d"]]
        if len(candidatas) > 0
        else pd.DataFrame(columns=clave + ["lag_etapa", "h3_d"]),
        on=clave + ["lag_etapa"],
        how="left",
    )

    out = resultado.reindex(columns=["id", "h3_d"])
    out["od_validado"] = out["h3_d"].notna().astype(int)
    return out


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------

def diagnostico_destinos(etapas):
    """
    Retorna un dict con métricas de la imputación de destinos e imprime
    un resumen en consola.

    Métricas:
    - total_etapas
    - etapas_con_destino (od_validado == 1)
    - tasa_imputacion (%)
    - etapas_od_mismo_h3
    - etapas_tarjeta_unica (tarjetas con 1 sola etapa en el día)
    - tasa_imputacion_por_linea: dict {id_linea: tasa}
    """
    total = len(etapas)
    con_destino = int((etapas["od_validado"] == 1).sum())
    tasa = round(con_destino / total * 100, 2) if total > 0 else 0.0

    od_mismo_h3 = int(
        (etapas["h3_o"] == etapas["h3_d"]).sum()
    )

    tarjetas_unicas = int(
        etapas.groupby(["dia", "id_tarjeta"])["id"].count().eq(1).sum()
    )

    tasa_por_linea = {}
    if "id_linea" in etapas.columns:
        tasa_por_linea = (
            (etapas.groupby("id_linea")["od_validado"].mean() * 100)
            .round(2)
            .to_dict()
        )

    metricas = {
        "total_etapas": total,
        "etapas_con_destino": con_destino,
        "tasa_imputacion": tasa,
        "etapas_od_mismo_h3": od_mismo_h3,
        "etapas_tarjeta_unica": tarjetas_unicas,
        "tasa_imputacion_por_linea": tasa_por_linea,
    }

    print("=" * 50)
    print("DIAGNÓSTICO DE IMPUTACIÓN DE DESTINOS")
    print("=" * 50)
    print(f"  Total etapas:           {total:,}")
    print(f"  Con destino validado:   {con_destino:,} ({tasa}%)")
    print(f"  Etapas O==D eliminadas: {od_mismo_h3:,}")
    print(f"  Tarjetas con 1 etapa:   {tarjetas_unicas:,}")
    print("=" * 50)

    return metricas


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def calcular_indicadores_destinos_etapas(etapas, ctx: StorageContext):
    """Calcula el porcentaje de etapas con destinos imputados y lo sube a la DB."""

    agrego_indicador(
        etapas[etapas["od_validado"] == 1],
        "Cantidad de etapas con destinos validados",
        "etapas",
        0,
        var_fex="",
        ctx=ctx,
    )

# H3 cell of (lat 0, lon 0) at res 8: legs whose coordinates were missing
# (latitud/longitud == 0) geocode to this "null island" cell in the middle of
# the ocean. Cleared from h3_o/h3_d below; applies to any city.
H3_NULL_ISLAND_RES8 = h3.latlng_to_cell(0, 0, 8)


def verif_h3_parent(h3_index, resolution=8):
    if pd.isna(h3_index) or h3_index == "":
        return ""
    try:
        return h3.cell_to_parent(h3_index, resolution)
    except Exception:
        return ""


def _clear_h3_parent(etapas, column, parent_h3):
    unique_h3 = etapas[column].dropna().unique()
    parent_by_h3 = {
        h3_index: verif_h3_parent(h3_index)
        for h3_index in unique_h3
        if h3_index != ""
    }
    mask = etapas[column].map(parent_by_h3).eq(parent_h3)
    etapas.loc[mask, column] = ""
    etapas.loc[mask, "etapa_validada"] = 0
    etapas.loc[mask, "od_validado"] = 0

def _infer_destinations_for_day(dia, destinos_min_dist, ctx):
    """
    Runs the full destination inference pipeline for a single day.
    Returns the processed etapas DataFrame (columns needed for write-back and diagnostics).

    La validación contra matriz_validacion se hace por id_linea_agg (red agregada)
    e id_ramal efectivo según modo_valida_ramal. id_linea_agg se trae mergeando
    metadata_lineas; modo es nativo de etapas. El id_ramal efectivo lo resuelven
    validar_destinos / imputar_destino_min_distancia con modos_con_ramal(configs).
    """
    etapas = ctx.data.query(
        f"""
        SELECT e.id, e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa,
               e.hora, e.tiempo, e.modo, e.id_linea, e.id_ramal, e.h3_o,
               e.etapa_validada
        FROM etapas e
        WHERE e.dia = '{dia}'
        ORDER BY e.id_tarjeta, e.id_viaje, e.id_etapa, e.hora, e.tiempo
        """
    )

    metadata_lineas = ctx.insumos.get_metadata_lineas()[["id_linea", "id_linea_agg"]]
    etapas = etapas.merge(metadata_lineas, how="left", on="id_linea")

    etapas_destinos_potencial = imputar_destino_potencial(etapas)
    del etapas

    if destinos_min_dist:
        destinos = imputar_destino_min_distancia(etapas_destinos_potencial, ctx)
        etapas = etapas_destinos_potencial.drop(columns=["h3_d"]).merge(
            destinos[["id", "h3_d", "od_validado"]], on="id", how="left"
        )
    else:
        destinos = validar_destinos(etapas_destinos_potencial, ctx)
        etapas = etapas_destinos_potencial.merge(
            destinos[["id", "od_validado"]], on="id", how="left"
        )
    del etapas_destinos_potencial, destinos

    etapas_mismo_od = etapas["h3_o"] == etapas["h3_d"]
    logger.info(
        "Dia %s — eliminando destinos con OD mismo h3: %d", dia, etapas_mismo_od.sum()
    )
    etapas.loc[etapas_mismo_od, "h3_d"] = np.nan
    etapas.loc[etapas_mismo_od, "od_validado"] = 0
    del etapas_mismo_od

    etapas["od_validado"] = etapas["od_validado"].fillna(0).astype(int)
    etapas["h3_d"] = etapas["h3_d"].fillna("")

    # etapa_validada = validacion individual del destino de la etapa (independiente
    # de la validez de la cadena de la tarjeta, que se evalua despues en trips.py).
    # Se setea aca para que assign_time_distances (que corre ANTES de create_trips)
    # pueda gatear el calculo de tiempos/distancias por etapa_validada. El
    # _clear_h3_parent siguiente la refina a 0 para los h3 en isla nula.
    etapas["etapa_validada"] = etapas["od_validado"]

    logger.debug("Dia %s — limpiando h3_o con coordenadas (0, 0)", dia)
    _clear_h3_parent(etapas, "h3_o", H3_NULL_ISLAND_RES8)
    logger.debug("Dia %s — limpiando h3_d con coordenadas (0, 0)", dia)
    _clear_h3_parent(etapas, "h3_d", H3_NULL_ISLAND_RES8)

    return etapas


_DIAG_COLS = ["id", "dia", "id_tarjeta", "id_linea", "h3_o", "h3_d", "od_validado"]


@duracion
def infer_destinations(ctx: StorageContext):
    """
    Lee las etapas de la DB un día a la vez, imputa destinos y los valida.
    Procesar por día mantiene el pico de memoria constante (~4-5 GB)
    independientemente de cuántos días tenga la corrida.
    """
    configs = leer_configs_generales(autogenerado=False)
    destinos_min_dist = configs.get("imputar_destinos_min_distancia", False) or False

    if destinos_min_dist:
        logger.info(
            "Utilizando como destino la parada de la linea de origen "
            "que minimiza la distancia con respecto al origen de la siguiente etapa"
        )
    else:
        logger.info("Utilizando como destino el origen de la siguiente etapa")

    dias = ctx.data.get_run_days()["dia"].tolist()
    logger.info("Imputando destinos para %d día(s): %s", len(dias), dias)

    has_selective_update = hasattr(ctx.data, "update_leg_destinations")

    diag_slices = []   # accumulate lightweight slices for end-of-run diagnostics
    fallback_full = [] # only used when update_leg_destinations is unavailable

    # Drop the index on (dia, od_validado) while the per-day UPDATEs run:
    # updating an indexed column makes DuckDB rewrite each row (delete+insert
    # with full ART maintenance), turning the write-back into the dominant
    # cost of the step (~34 min/day vs seconds without the index).
    if has_selective_update and hasattr(ctx.data, "begin_leg_destination_updates"):
        ctx.data.begin_leg_destination_updates()
    try:
        for dia in dias:
            logger.info("Procesando dia %s", dia)
            etapas_dia = _infer_destinations_for_day(
                dia, destinos_min_dist, ctx
            )

            if has_selective_update:
                ctx.data.update_leg_destinations(
                    etapas_dia[["id", "dia", "h3_d", "od_validado", "etapa_validada"]]
                )
                logger.info("Dia %s — destinos guardados", dia)
            else:
                fallback_full.append(etapas_dia)

            diag_slices.append(etapas_dia[_DIAG_COLS].copy())
            del etapas_dia
    finally:
        if has_selective_update and hasattr(ctx.data, "end_leg_destination_updates"):
            ctx.data.end_leg_destination_updates()

    # Fallback write-back for adapters that lack selective UPDATE support
    if not has_selective_update and fallback_full:
        etapas_completas = pd.concat(fallback_full, ignore_index=True)
        del fallback_full
        if hasattr(ctx.data, "replace_legs_for_days"):
            ctx.data.replace_legs_for_days(etapas_completas, dias)
        else:
            dias_str = ", ".join(f"'{d}'" for d in dias)
            ctx.data.execute(f"DELETE FROM etapas WHERE dia IN ({dias_str})")
            ctx.data.save_legs(etapas_completas)
        del etapas_completas

    # Diagnostics over the full run using the accumulated lightweight slices
    etapas_diag = pd.concat(diag_slices, ignore_index=True)
    del diag_slices
    calcular_indicadores_destinos_etapas(etapas_diag, ctx)
    diagnostico_destinos(etapas_diag)

    logger.info("Etapas con destinos imputados guardadas")
    return None
