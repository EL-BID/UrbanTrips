import pandas as pd
import numpy as np
from math import ceil
from itertools import repeat
import multiprocessing
import h3
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    agrego_indicador)


@duracion
def infer_destinations():
    """
    Esta funcion lee las etapas de la db, imputa destinos potenciales
    y los valida
    """

    configs = leer_configs_generales()
    mensaje = "Utilizando como destino el origen de la siguiente etapa"
    try:
        destinos_min_dist = configs["imputar_destinos_min_distancia"]
        if destinos_min_dist is None:
            destinos_min_dist = False
        else:
            if destinos_min_dist:
                mensaje = "Utilizando como destino la parada de la linea "
                mensaje = mensaje + "de origen que minimiza la distancia con "
                mensaje = mensaje + "respecto al origen de la siguiente etapa"

    except KeyError:
        destinos_min_dist = False

    print(mensaje)

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    dias_ultima_corrida = pd.read_sql_query(
        """
                                SELECT *
                                FROM dias_ultima_corrida
                                """,
        conn_data,
    )

    q = """
    select e.*
    from etapas e
    join dias_ultima_corrida d
    on e.dia = d.dia
    order by e.dia,e.id_tarjeta,e.id_viaje,e.id_etapa,e.hora,e.tiempo
    """

    etapas = pd.read_sql_query(q, conn_data)

    metadata_lineas = pd.read_sql_query(
        """
        SELECT *
        FROM metadata_lineas
        """,
        conn_insumos,
    )

    etapas = etapas.merge(metadata_lineas[['id_linea',
                                           'id_linea_agg']],
                          how='left',
                          on='id_linea')

    if 'od_validado' in etapas.columns:
        etapas = etapas.drop(['od_validado'], axis=1)
    if 'h3_d' in etapas.columns:
        etapas = etapas.drop(['h3_d'], axis=1)

    etapas_destinos_potencial = imputar_destino_potencial(etapas)

    if destinos_min_dist:
        print("Imputando destinos por minimizacion de distancias...")
        destinos = imputar_destino_min_distancia(etapas_destinos_potencial)
        # no usar h3_d que en este caso es el potencial, las coords siguientes
        etapas = etapas.drop('h3_d', axis=1)
        etapas = etapas.merge(destinos, on='id', how='left')

    else:
        print("Imputando destinos por siguiente etapa...")
        destinos = validar_destinos(etapas_destinos_potencial)

        etapas = etapas.merge(
            destinos, on=['id', 'h3_d'], how='left')

    etapas = etapas\
        .sort_values(
            ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa', 'hora', 'tiempo'])\
        .reset_index(drop=True)

    etapas['od_validado'] = etapas['od_validado'].fillna(0).astype(int)
    etapas['h3_d'] = etapas['h3_d'].fillna('')

    # calcular indicador de imputacion de destinos
    calcular_indicadores_destinos_etapas(etapas)

    # borro si ya existen etapas de una corrida anterior
    values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
    query = f"DELETE FROM etapas WHERE dia IN ({values})"
    conn_data.execute(query)
    conn_data.commit()

    etapas = etapas.drop(['id_linea_agg'],
                         axis=1)

    etapas.to_sql("etapas",
                  conn_data,
                  if_exists="append",
                  index=False)

    conn_data.close()
    conn_insumos.close()

    return None


def imputar_destino_potencial(etapas):
    """
    Esta funcion toma un DF de etapas, ordena por
    'dia','id_tarjeta','id_viaje','id_etapa','hora'
    e imputa para cada dia, id_tarjeta el origen de la
    siguiente etapa como destino potencial a validar
    """

    # imputar como destino el origen de la etapa siguiente
    etapas["h3_d"] = (
        etapas.reindex(columns=["dia", "id_tarjeta", "h3_o"])
        .groupby(["dia", "id_tarjeta"])
        .shift(-1)
    )

    # completar la ultima con la primera del dia
    primera_trx = (
        etapas.reindex(columns=["dia", "id_tarjeta", "h3_o"])
        .groupby(["dia", "id_tarjeta"], as_index=False)
        .apply(lambda x: x.h3_o.iloc[0])
    )
    primera_trx.columns = ["dia", "id_tarjeta", "h3_d_primera"]
    primera_trx = (
        etapas.reindex(columns=["dia", "id_tarjeta"])
        .merge(primera_trx, how="left", on=["dia", "id_tarjeta"])
        .h3_d_primera
    )
    etapas.h3_d = etapas.h3_d.combine_first(primera_trx)

    return etapas


def imputar_destino_min_distancia(etapas):
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    matriz_validacion = pd.read_sql_query(
        """SELECT * from matriz_validacion""",
        conn_insumos)
    n_cores = max(int(multiprocessing.cpu_count() / 2), 1)

    # crear un df con el id de cada etapa, la linea que uso y la etapa
    # siguiente
    lag_etapas = etapas.copy().reindex(columns=['id', 'id_linea_agg', 'h3_d'])\
        .rename(columns={'h3_d': 'lag_etapa'})
    del etapas

    # Obtener las paradas candidatas que compartan la misma linea
    # y esten dentro del area de influencia
    lag_etapas_no_dups = lag_etapas.reindex(
        columns=['id_linea_agg', 'lag_etapa']).drop_duplicates()
    lag_etapas_no_dups['id'] = range(len(lag_etapas_no_dups))

    paradas_candidatas = pd.DataFrame()
    numero_corte = 5000000
    iteraciones = ceil(len(lag_etapas_no_dups) / numero_corte)
    iteraciones = list(range(0, iteraciones+1))

    for i in iteraciones:
        filas_principio = i * numero_corte
        filas_fin = (i+1) * numero_corte

        print('Running from row', filas_principio, 'to', filas_fin)

        paradas_candidatas_sample = (
            lag_etapas_no_dups.iloc[filas_principio:filas_fin, :].copy())

        if len(paradas_candidatas_sample) > 0:
            paradas_candidatas_sample = parallelize_dataframe(
                paradas_candidatas_sample,
                minimizar_distancia_parada_candidata,
                matriz_validacion,
                n_cores=n_cores)

            paradas_candidatas = pd.concat([paradas_candidatas,
                                            paradas_candidatas_sample])
            paradas_candidatas = paradas_candidatas.drop(['id'], axis=1)
            paradas_candidatas = lag_etapas.merge(
                paradas_candidatas, on=['id_linea_agg', 'lag_etapa'],
                how='left')

    # Imprimir estadisticos de las distancias
    print("Promedios de distancia entre la etapa siguiente  y ")
    print("la parada de origen. En unidades de h3:")
    print(paradas_candidatas.distancias.describe())

    # Volver a unir con la tabla etapas
    out = paradas_candidatas.reindex(columns=['id', 'h3_d'])
    out['od_validado'] = out.h3_d.notna().astype("int")
    out = out.reindex(columns=["id", "h3_d", "od_validado"])

    print('fin')
    return out


def parallelize_dataframe(df, func, matriz_validacion, n_cores=4):
    """
    This function takes a dataframe of legs with the next origen as possible
    destination, a function that minimices distance to a set of possible stops
    and a validation matrix  of stops' catchment area and returns that
    function in parallel
    """
    df_split = np.array_split(df, n_cores)
    pool = multiprocessing.Pool(n_cores)
    df = pd.concat(pool.starmap(
        func, zip(df_split, repeat(matriz_validacion))))
    pool.close()
    pool.join()
    return df


def minimizar_distancia_parada_candidata(
        paradas_candidatas, matriz_validacion):
    """
    This function takes a dataframe with a set of legs' origins and possible
    destinations and a stops' catchment area validation matrix and returns
    the stops that minimices the distances to all possible stops from that line
    """
    paradas_candidatas_sample = paradas_candidatas\
        .merge(matriz_validacion, left_on=['id_linea_agg', 'lag_etapa'],
               right_on=['id_linea_agg', 'area_influencia'])\
        .rename(columns={'parada': 'h3_d'})\
        .drop(['area_influencia',
               ], axis=1)
    p_dict = paradas_candidatas_sample.to_dict('records')
    # calcular distancia de cada parada posible al lag etapa
    distancias = list(map(h3_distance_stops, p_dict))
    del p_dict
    paradas_candidatas_sample['distancias'] = distancias
    # quedarse con la de menor distancia
    paradas_candidatas_sample = paradas_candidatas_sample\
        .sort_values(['id', 'distancias'])\
        .drop_duplicates(subset=['id'], keep='first')

    return paradas_candidatas_sample


def h3_distance_stops(row):
    return h3.h3_distance(row['lag_etapa'], row['h3_d'])


def validar_destinos(destinos):
    """
    Esta funcion toma una DF con destinos potenciales imputados
    y los evalua contra la matriz de evaluacion que contiene los h3
    de la linea con los adyacentes
    """
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    q = """
    SELECT distinct id_linea_agg, area_influencia from matriz_validacion
    """
    matriz_validacion = pd.read_sql_query(
        q,
        conn_insumos,
    )
    # Crear pares od unicos por linea
    pares_od_linea = destinos.reindex(
        columns=["h3_o", "h3_d", "id_linea_agg"]
    ).drop_duplicates()

    # validar esos pares od con los hrings
    pares_od_linea = pares_od_linea.merge(
        matriz_validacion,
        how="left",
        left_on=["id_linea_agg", "h3_d"],
        right_on=["id_linea_agg", "area_influencia"],
    )
    pares_od_linea["od_validado"] = pares_od_linea['area_influencia'].notna(
    ).fillna(0)

    # Pasar de pares od a cada etapa
    pares_od_linea = pares_od_linea.reindex(
        columns=["h3_o", "h3_d", "id_linea_agg", "od_validado"]
    )
    destinos = destinos.merge(
        pares_od_linea, how="left", on=["h3_o", "h3_d", "id_linea_agg"]
    )
    # Seleccionar columnas y convertir en int
    destinos = destinos.reindex(columns=["id", "h3_d", "od_validado"])
    destinos.od_validado = destinos.od_validado.astype("int")

    return destinos


def calcular_indicadores_destinos_etapas(etapas):
    """
    Esta funcion calcula el % de etapas con destinos imputados
    y lo sube a la db
    """
    print("Calculando indicadores de etapas con destinos")

    agrego_indicador(etapas[etapas.od_validado == 1],
                     'Cantidad de etapas con destinos validados',
                     'etapas',
                     1,
                     var_fex='')
