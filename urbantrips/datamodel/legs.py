import pandas as pd
import itertools
import time
import numpy as np
from urbantrips.geo.geo import referenciar_h3
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    agrego_indicador,
    eliminar_tarjetas_trx_unica)


@duracion
def create_legs_from_transactions(trx_order_params):
    """
    Esta function toma las transacciones de la db
    las estructura en etapas con sus id y id viaje
    y crea la tabla etapas en la db
    """

    conn = iniciar_conexion_db(tipo='data')

    dias_ultima_corrida = pd.read_sql_query(
        """
                                    SELECT *
                                    FROM dias_ultima_corrida
                                    """,
        conn,
    )

    legs = pd.read_sql_query(
        """
                            SELECT t.*
                            FROM transacciones t
                            JOIN dias_ultima_corrida d
                            ON t.dia = d.dia
                            """,
        conn
    )
    # parse dates using local timezone
    legs['fecha'] = pd.to_datetime(legs.fecha, unit='s',
                                   errors='coerce')

    # asignar id h3
    configs = leer_configs_generales()
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
    legs = change_card_id_for_concurrent_trx(
        legs, trx_order_params, dias_ultima_corrida)

    # elminar casos de nuevas tarjetas con trx unica
    # legs = eliminar_tarjetas_trx_unica(legs)
    # No borrar transacciones únicas (quedan en estas con fex=0)

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
            "latitud",
            "longitud",
            "h3_o",
            "factor_expansion"
        ]
    )

    legs = legs.rename(
        columns={"factor_expansion": "factor_expansion_original"})

    print(f"Subiendo {len(legs)} registros a la tabla etapas en la db")

    # borro si ya existen etapas de una corrida anterior
    values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
    query = f"DELETE FROM etapas WHERE dia IN ({values})"
    conn.execute(query)
    conn.commit()

    legs.to_sql("etapas", conn, if_exists="append", index=False)

    print("Fin subir etapas")
    agrego_indicador(legs,
                     'Cantidad de etapas pre imputacion de destinos',
                     'etapas',
                     0,
                     var_fex='')
    conn.close()


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

    print("Creando delta de trx")
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
    print("Fin creacion delta de trx")
    return trx


def change_card_id_for_concurrent_trx(trx,
                                      trx_order_params, dias_ultima_corrida):
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
    conn = iniciar_conexion_db(tipo='data')

    print("Creando nuevos id tajetas para trx simultaneas")
    trx_c = trx.copy()

    trx_c, tarjetas_duplicadas = pago_doble_tarjeta(trx_c, trx_order_params)

    print(f"Subiendo {len(tarjetas_duplicadas)} tarjetas duplicadas a la db")
    if len(tarjetas_duplicadas) > 0:

        # borro si ya existen etapas de una corrida anterior
        values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
        query = f"DELETE FROM tarjetas_duplicadas WHERE dia IN ({values})"
        conn.execute(query)
        conn.commit()

        tarjetas_duplicadas.to_sql(
            "tarjetas_duplicadas", conn, if_exists="append", index=False
        )
    print("Fin subir tarjetas duplicadas")
    print("Fin creacion nuevos id tajetas para trx simultaneas")

    return trx_c


def pago_doble_tarjeta(trx, trx_order_params):
    """
    Takes a transaction dataframe with a time delta and
    a time window for duplciates in minutes,
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
        diff_segundos = ventana_duplicado*60

        trx['fecha_aux'] = trx['fecha'].astype(str).str[-8:]

        trx['fecha_aux'] = trx['fecha_aux'].apply(lambda x: sum(
            int(i) * 60 ** j for j, i in enumerate(x.split(":")[::-1])))

    elif trx_order_params["criterio"] == "orden_trx":
        trx.loc[:, ['fecha_aux']] = trx['hora']
        diff_segundos = 1

    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    trx['interno2'] = trx['interno']
    trx['interno2'] = trx['interno2'].fillna(0)
    
    trx = trx.sort_values(['dia', 'id_tarjeta', 'id_linea',
                          'interno2', 'fecha_aux', 'orden_trx']).reset_index(drop=True)

    trx["datetime_proximo"] = trx["fecha_aux"].shift(-1)

    trx['diff_datetime'] = (trx.fecha_aux - trx.datetime_proximo).abs()

    trx['diff_datetime2'] = trx.groupby(
        ['dia', 'id_tarjeta', 'id_linea', 'interno2']).diff_datetime.shift(+1)

    trx['nro'] = np.nan
    trx.loc[(trx.diff_datetime2.isna()) | (
        trx.diff_datetime2 > diff_segundos), 'nro'] = 0

    while len(trx[trx.nro.isna()]) > 0:
        trx['nro2'] = trx.groupby(
            ['dia', 'id_tarjeta', 'id_linea', 'interno2']).nro.shift(+1)+1
        trx.loc[trx.nro.isna() & (trx.nro2.notna()),
                'nro'] = trx.loc[trx.nro.isna() & (trx.nro2.notna()), 'nro2']

    trx['id_tarjeta_nuevo'] = trx['id_tarjeta'] + \
        '_' + trx['nro'].astype(int).astype(str)

    tarjetas_duplicadas = trx.loc[trx['nro'] > 0]\
        .reindex(columns=['dia', 'id_tarjeta', 'id_tarjeta_nuevo'])\
        .rename(columns={'id_tarjeta': 'id_tarjeta_original'})\
        .drop_duplicates()

    trx = trx\
        .drop('id_tarjeta', axis=1)\
        .rename(columns={'id_tarjeta_nuevo': 'id_tarjeta'})

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

    print(f"Hay {duplicados.sum()} casos duplicados")

    if duplicados.sum() > 0:
        # crear una tabla de registro de cambio de id tarjeta
        tarjetas_duplicadas = trx.loc[nro_duplicado.index, [
            "dia", "id_tarjeta"]]
        tarjetas_duplicadas = tarjetas_duplicadas.rename(
            columns={"id_tarjeta": "id_tarjeta_original"}
        )
        tarjetas_duplicadas["id_tarjeta_nuevo"] = (
            tarjetas_duplicadas.id_tarjeta_original + '_' + nro_duplicado
        )

        # crear un nuevo vector con los incrementales y concatenarlos
        nuevo_id_tarjeta = pd.Series(["0"] * len(trx))
        nuevo_id_tarjeta.loc[nro_duplicado.index] = nro_duplicado
        trx.id_tarjeta = trx.id_tarjeta.map(str) + '_' + nuevo_id_tarjeta
    else:
        tarjetas_duplicadas = pd.DataFrame()
    trx = trx.drop("duplicados_ventana", axis=1)

    print("Fin creacion de nuevos id tarjetas para duplicados con delta")
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

    print(f"Hay {duplicados.sum()} casos duplicados")
    # crear un nuevo vector con los incrementales y concatenarlos

    if duplicados.sum() > 0:
        nuevo_id_tarjeta = pd.Series(["0"] * len(trx))
        nuevo_id_tarjeta.loc[nro_duplicado.index] = nro_duplicado
        trx.id_tarjeta = trx.id_tarjeta + '_' + nuevo_id_tarjeta

    print("Fin creacion de nuevos id tarjetas para duplicados con orden trx")
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

    print("Crear un id para viajes y etapas")

    if trx_order_params["criterio"] == "orden_trx":
        print("Utilizando orden_trx")
        trx = asignar_id_viaje_etapa_orden_trx(trx)

    elif trx_order_params["criterio"] == "fecha_completa":
        print("Utilizando fecha_completa")
        ventana_viajes = trx_order_params["ventana_viajes"]
        trx = asignar_id_viaje_etapa_fecha_completa(trx, ventana_viajes)

    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    print("Fin creacion de un id para viajes y etapas")
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

    trx = trx.sort_values(["dia", "id_tarjeta", "fecha"])

    # Calcular los id_viajes
    id_viajes = (
        trx.reindex(columns=["dia", "id_tarjeta", "delta"])
        .groupby(["dia", "id_tarjeta"])
        .apply(crear_viaje_id_acumulada, ventana_viajes)
    )

    trx["id_viaje"] = list(itertools.chain(*id_viajes.values))
    lista = ["dia", "id_tarjeta", "id_viaje"]
    trx["id_etapa"] = trx.groupby(lista).cumcount() + 1
    return trx


def asignar_id_viaje_etapa_orden_trx(trx):
    """
    Esta funcion toma un DF de trx y asigna id_viaje y id_etapa
    en base al dia, hora y orden_trx
    """
    variables_secuencia = [
        "dia",
        "id_tarjeta",
        "tiempo",
        "hora",
        "orden_trx",
        "modo",
        "id_linea",
    ]
    # ordenar transacciones
    trx = trx.sort_values(variables_secuencia)
    trx["secuencia"] = trx.groupby(["dia", "id_tarjeta"]).cumcount() + 1

    # calcular id viaje restando a secuencia cada vez que hay trasbordo
    trx["nro_viaje_temp"] = trx.secuencia - trx["orden_trx"]

    # calcular un id_viaje unico y secuencial
    temp = trx.groupby(["id_tarjeta", "nro_viaje_temp"]).size().reset_index()
    temp["id_viaje"] = temp.groupby(["id_tarjeta"]).cumcount() + 1
    temp = temp.reindex(columns=["id_tarjeta", "nro_viaje_temp", "id_viaje"])

    # volver a unir a tabla trx
    trx = trx.merge(temp, on=["id_tarjeta", "nro_viaje_temp"], how="left")
    trx = trx.drop(["secuencia", "nro_viaje_temp"], axis=1)

    # asignar id_etapa
    sort = ["dia", "id_tarjeta", "id_viaje", "hora", "orden_trx"]
    trx = trx.sort_values(sort)
    g = ["dia", "id_tarjeta", "id_viaje"]
    trx["id_etapa"] = trx.groupby(g).cumcount() + 1

    return trx


def crear_viaje_id_acumulada(df, ventana_viajes=120):
    """
    Esta funcion toma un df y una ventana de tiempo
    y agrupa en un mismo viaje id los que caigan dentro de esa
    ventana
    """

    cumulativa = 0
    viaje_id = 1
    viajes = []
    for i in df.delta:
        cumulativa += i

        if cumulativa <= ventana_viajes:
            pass
        else:
            cumulativa = 0
            viaje_id += 1

        viajes.append(viaje_id)

    return viajes
