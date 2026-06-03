import numpy as np
import pandas as pd
import h3
from datetime import datetime

from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    agrego_indicador,
    levanto_tabla_sql,
    guardar_tabla_sql,
)


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

def validar_destinos(destinos):
    """
    Valida destinos potenciales contra la matriz de validación de la DB.
    Delega en _validar_destinos_con_matriz para facilitar el testing.
    """
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    matriz_validacion = pd.read_sql_query(
        "SELECT DISTINCT id_linea_agg, area_influencia FROM matriz_validacion",
        conn_insumos,
    )
    conn_insumos.close()

    return _validar_destinos_con_matriz(destinos, matriz_validacion)


def _validar_destinos_con_matriz(destinos, matriz_validacion):
    """
    Valida destinos potenciales contra una matriz de validación dada.
    Versión separada de validar_destinos() para facilitar el testing sin DB.

    Agrega drop_duplicates antes del merge para evitar multiplicar filas y
    verifica con un assert que el tamaño del DataFrame no cambia.
    """
    n_orig = len(destinos)

    pares_od_linea = (
        destinos.reindex(columns=["h3_o", "h3_d", "id_linea_agg"])
        .drop_duplicates()
    )

    pares_od_linea = pares_od_linea.merge(
        matriz_validacion.drop_duplicates(),
        how="left",
        left_on=["id_linea_agg", "h3_d"],
        right_on=["id_linea_agg", "area_influencia"],
    )
    pares_od_linea["od_validado"] = pares_od_linea["area_influencia"].notna()

    # Si un par od tiene múltiples matches, consolidar con max (True prevalece)
    pares_od_linea = (
        pares_od_linea.groupby(["h3_o", "h3_d", "id_linea_agg"], as_index=False)[
            "od_validado"
        ].max()
    )

    destinos = destinos.merge(
        pares_od_linea, how="left", on=["h3_o", "h3_d", "id_linea_agg"]
    )

    assert len(destinos) == n_orig, (
        f"El merge multiplicó filas: {n_orig} → {len(destinos)}"
    )

    destinos = destinos.reindex(columns=["id", "h3_d", "od_validado"])
    destinos["od_validado"] = destinos["od_validado"].fillna(0).astype(int)

    return destinos


# ---------------------------------------------------------------------------
# Minimización de distancia
# ---------------------------------------------------------------------------

def _h3_grid_distance_safe(h3_a, h3_b):
    """Calcula h3.grid_distance con manejo de errores."""
    try:
        return h3.grid_distance(h3_a, h3_b)
    except Exception:
        return np.inf


def imputar_destino_min_distancia(etapas):
    """
    Para cada etapa, busca la parada de su línea que minimiza la distancia
    h3 al destino potencial (origen de la etapa siguiente).
    Delega en _imputar_destino_min_distancia_con_matriz para facilitar el testing.
    """
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)
    matriz_validacion = pd.read_sql_query(
        "SELECT * FROM matriz_validacion", conn_insumos
    )
    conn_insumos.close()

    return _imputar_destino_min_distancia_con_matriz(etapas, matriz_validacion)


def _imputar_destino_min_distancia_con_matriz(etapas, matriz_validacion):
    """
    Versión testeable sin DB de imputar_destino_min_distancia.

    Approach:
    1. Arma pares únicos (id_linea_agg, h3_d_potencial)
    2. Merge con matriz_validacion para obtener paradas candidatas
    3. Calcula h3.grid_distance (con try/except) para cada candidata
    4. Groupby para quedarse con la de menor distancia
    5. Merge de vuelta a etapas

    Sin multiprocessing, sin conversión a dict, sin loop por chunks.
    """
    lag_etapas = (
        etapas.reindex(columns=["id", "id_linea_agg", "h3_d"])
        .rename(columns={"h3_d": "lag_etapa"})
    )

    pares_unicos = (
        lag_etapas.reindex(columns=["id_linea_agg", "lag_etapa"])
        .drop_duplicates()
    )

    candidatas = pares_unicos.merge(
        matriz_validacion[["id_linea_agg", "area_influencia", "parada"]],
        left_on=["id_linea_agg", "lag_etapa"],
        right_on=["id_linea_agg", "area_influencia"],
        how="left",
    ).rename(columns={"parada": "h3_d"}).drop(columns=["area_influencia"])

    candidatas = candidatas.dropna(subset=["h3_d"])

    if len(candidatas) > 0:
        _dist_vec = np.vectorize(_h3_grid_distance_safe)
        candidatas["distance_od"] = _dist_vec(
            candidatas["lag_etapa"].values, candidatas["h3_d"].values
        )
        candidatas = (
            candidatas
            .sort_values("distance_od")
            .drop_duplicates(subset=["id_linea_agg", "lag_etapa"], keep="first")
        )

    resultado = lag_etapas.merge(
        candidatas[["id_linea_agg", "lag_etapa", "h3_d"]]
        if len(candidatas) > 0
        else pd.DataFrame(columns=["id_linea_agg", "lag_etapa", "h3_d"]),
        on=["id_linea_agg", "lag_etapa"],
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
            etapas.groupby("id_linea")["od_validado"]
            .apply(lambda g: round((g == 1).mean() * 100, 2))
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

def calcular_indicadores_destinos_etapas(etapas):
    """Calcula el porcentaje de etapas con destinos imputados y lo sube a la DB."""

    agrego_indicador(
        etapas[etapas["od_validado"] == 1],
        "Cantidad de etapas con destinos validados",
        "etapas",
        0,
        var_fex="",
    )

def verif_h3(h3_index):
    if pd.isna(h3_index) or h3_index == '':
        return ""
    try:
        lat, lon = h3.cell_to_latlng(h3_index)
        return h3.latlng_to_cell(lat, lon, 8)
    except Exception:
        return ""

@duracion
def infer_destinations():
    """
    Lee las etapas de la DB, imputa destinos potenciales y los valida.
    Mantiene la misma firma y estructura de entrada/salida que destinations.py.
    """
    configs = leer_configs_generales()
    destinos_min_dist = configs.get("imputar_destinos_min_distancia", False) or False

    if destinos_min_dist:
        mensaje = (
            "Utilizando como destino la parada de la linea de origen "
            "que minimiza la distancia con respecto al origen de la siguiente etapa"
        )
    else:
        mensaje = "Utilizando como destino el origen de la siguiente etapa"
    print(mensaje)

    conn_data = iniciar_conexion_db(tipo="data")
    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    dias_ultima_corrida = pd.read_sql_query(
        "SELECT * FROM dias_ultima_corrida", conn_data
    )

    q = """
    SELECT e.*
    FROM etapas e
    JOIN dias_ultima_corrida d ON e.dia = d.dia
    ORDER BY e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa, e.hora, e.tiempo
    """
    etapas = pd.read_sql_query(q, conn_data)

    metadata_lineas = pd.read_sql_query("SELECT * FROM metadata_lineas", conn_insumos)
    conn_insumos.close()

    etapas = etapas.merge(
        metadata_lineas[["id_linea", "id_linea_agg"]], how="left", on="id_linea"
    )

    for col in ("od_validado", "h3_d"):
        if col in etapas.columns:
            etapas = etapas.drop(columns=[col])

    etapas_destinos_potencial = imputar_destino_potencial(etapas)

    if destinos_min_dist:
        destinos = imputar_destino_min_distancia(etapas_destinos_potencial)
        # destinos["h3_d"] es la parada más cercana (snap): reemplaza al destino potencial
        etapas = etapas_destinos_potencial.drop(columns=["h3_d"]).merge(
            destinos[["id", "h3_d", "od_validado"]], on="id", how="left"
        )
    else:
        destinos = validar_destinos(etapas_destinos_potencial)
        # solo valida el destino potencial: se conserva h3_d
        etapas = etapas_destinos_potencial.merge(
            destinos[["id", "od_validado"]], on="id", how="left"
        )

    # # etapas_destinos_potencial ya tiene h3_d (el potencial); destinos aporta od_validado
    # etapas = etapas_destinos_potencial.merge(
    #     destinos[["id", "od_validado"]], on="id", how="left"
    # )

    etapas_mismo_od = etapas["h3_o"] == etapas["h3_d"]
    print("Eliminando destinos de etapas con OD mismo h3:", etapas_mismo_od.sum())
    etapas.loc[etapas_mismo_od, "h3_d"] = np.nan
    etapas.loc[etapas_mismo_od, "od_validado"] = 0

    etapas["od_validado"] = etapas["od_validado"].fillna(0).astype(int)
    etapas["h3_d"] = etapas["h3_d"].fillna("")

    calcular_indicadores_destinos_etapas(etapas)

    diagnostico_destinos(etapas)

    # values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    # query = f"DELETE FROM etapas WHERE dia IN ({values})"
    # conn_data.execute(query)
    # conn_data.commit()

    etapas = etapas.drop(columns=["id_linea_agg"])
    
    mask = etapas['h3_o'].apply(verif_h3).eq('88754e6499fffff')
    etapas.loc[mask, 'h3_o'] = ""
    etapas.loc[mask, 'etapa_validada'] = 0
    etapas.loc[mask, 'od_validado'] = 0
    mask = etapas['h3_d'].apply(verif_h3).eq('88754e6499fffff')
    etapas.loc[mask, 'h3_d'] = ""    
    etapas.loc[mask, 'etapa_validada'] = 0
    etapas.loc[mask, 'od_validado'] = 0

    
    conn_data.close()
    
    dias_ultima_corrida = levanto_tabla_sql("dias_ultima_corrida", "data")
    guardar_tabla_sql(
                        etapas,
                        "etapas",
                        tabla_tipo="data",
                        modo="append",
                        filtros={"dia": dias_ultima_corrida["dia"].tolist()},
                    )

    return None
