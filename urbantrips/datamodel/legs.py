import pandas as pd
import itertools
import numpy as np
import h3
from urbantrips.geo.geo import referenciar_h3, convert_h3_to_resolution
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    agrego_indicador
)


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


@duracion
def assign_gps_origin():
    """
    This function read legs data and if there is gps table 
    assigns a gps to the leg origin
    """
    print("Clasificando etapas en su gps de origen")
    conn_data = iniciar_conexion_db(tipo='data')
    # get legs data
    legs = pd.read_sql_query(
        """
        SELECT dia,id_linea,id_ramal,interno,id, tiempo
        FROM etapas
        WHERE od_validado==1
        order by dia,id_tarjeta,id_viaje,id_etapa, id_linea,id_ramal,interno
        """,
        conn_data,
    )
    legs['fecha'] = pd.to_datetime(legs['dia']+' '+legs['tiempo'])

    # get gps data
    q = """
    select dia,id_linea,id_ramal,interno,fecha,id 
    from gps
    order by dia, id_linea,id_ramal,interno,fecha;
    """
    gps = pd.read_sql(q, conn_data)
    gps.loc[:, ['fecha']] = gps.fecha.map(
        lambda ts: pd.Timestamp(ts, unit='s'))
    cols = ['dia', 'id_linea', 'id_ramal', 'interno', 'fecha', 'id']
    legs_to_join = legs.reindex(columns=cols).sort_values('fecha')
    gps_to_join = gps.reindex(columns=cols).sort_values('fecha')

    # Join on closest date
    legs_to_gps_o = pd.merge_asof(
        legs_to_join,
        gps_to_join,
        on='fecha', by=['dia', 'id_linea', 'id_ramal', 'interno'],
        direction='nearest', tolerance=pd.Timedelta('7 minutes'),
        suffixes=('_legs', '_gps'))

    legs_to_gps_o = legs_to_gps_o\
        .reindex(columns=['dia', 'id_legs', 'id_gps']).dropna()

    print(f"Subiendo {len(legs_to_gps_o)} etapas con id gps a la DB")
    legs_to_gps_o.to_sql("legs_to_gps_origin", conn_data,
                         if_exists='append', index=False)
    conn_data.close()

    return legs_to_gps_o


@duracion
def assign_gps_destination():
    """
    This function read legs data and if there is gps table 
    assigns a gps to the leg origin
    """
    print("Clasificando etapas en su gps de destino")
    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    configs = leer_configs_generales()
    legs_h3_res = configs['resolucion_h3']

    # read stops zone of incluence
    q = """
    select distinct parada,area_influencia
    from matriz_validacion;
    """
    matriz = pd.read_sql(q, conn_insumos)
    matriz['ring'] = matriz.apply(lambda row: h3.h3_distance(
        row.parada, row.area_influencia), axis=1)
    matriz = matriz[matriz.ring < 3]

    print("Leyendo datos de etapas con GPS")
    legs = pd.read_sql_query(
        """
        SELECT e.*
        FROM etapas e
        JOIN dias_ultima_corrida d
        ON e.dia = d.dia
        WHERE od_validado==1
        and modo = 'autobus'
        order by dia,id_tarjeta,id_viaje,id_etapa, id_linea,id_ramal,interno
        ;
        """,
        conn_data,
    )

    # Add distances to legs
    q = """
    select h3_o, h3_d, distance_osm_drive
    from distancias
    where distance_osm_drive is not null;
    """
    distances = pd.read_sql(q, conn_insumos)

    legs = legs.merge(distances, how='inner', on=['h3_o', 'h3_d'])
    del distances
    legs['fecha'] = pd.to_datetime(legs['dia']+' '+legs['tiempo'])

    print("Leyendo datos de GPS")
    q = """
    select * 
    from gps g
    JOIN dias_ultima_corrida d
    ON g.dia = d.dia
    order by dia, id_linea,id_ramal,interno,fecha
    ;
    """
    gps = pd.read_sql(q, conn_data)

    # get h3 res for gps
    gps_h3_res = h3.h3_get_resolution(gps['h3'].sample().item())

    # geocode gps with same h3 res than legs
    gps = referenciar_h3(gps, res=legs_h3_res,
                         nombre_h3='h3_legs_res', lat='latitud', lon='longitud')
    gps.loc[:, ['fecha_gps']] = gps.fecha.map(
        lambda ts: pd.Timestamp(ts, unit='s'))
    gps.loc[:, ['hora']] = gps.fecha_gps.dt.hour

    # Geocode legs destination in the same h3 resolution than gps
    legs['h3_d_gps_res'] = legs['h3_d'].apply(
        lambda x: convert_h3_to_resolution(x, gps_h3_res))

    # Lista para acumular resultados parciales
    etapas_result_list = []

    # Iteración por cada hora del día
    legs_hours = legs.hora.unique()
    legs_hours.sort()

    print("Imputando GPS de destino")

    for hora in legs_hours:
        # Filtrar las etapas por la hora específica y eliminar valores nulos en 'h3_d'
        etapas_tx = legs.loc[legs['hora'] == hora, [
            'dia', 'id', 'id_linea', 'id_ramal', 'interno',
            'h3_o', 'h3_d', 'h3_d_gps_res', 'distance_osm_drive', 'fecha']].copy()

        # Agregar anillos a las etapas
        etapas_tx = etapas_tx.merge(
            matriz, how='left', left_on='h3_d', right_on='parada')

        # Determinar horas consecutivas para el filtrado de datos GPS
        hora_filtro = [hora + i for i in range(0, 4)]
        gps_tx = gps.loc[gps['hora'].isin(hora_filtro), :].copy()

        # Renombrar y seleccionar columnas relevantes en los datos GPS
        gps_tx = gps_tx.reindex(columns=['id', 'id_linea', 'id_ramal',
                                         'interno', 'h3_legs_res',
                                         'h3', 'fecha_gps']
                                ).rename(columns={'h3_legs_res': 'area_influencia'})

        # Join gps to legs destination rings dataframe by the same resolution (legs resolution)
        etapas_tx = etapas_tx.merge(gps_tx, how='inner',
                                    on=['id_linea', 'id_ramal',
                                        'interno', 'area_influencia'],
                                    suffixes=('_legs', '_gps'),)

        # Calcular la diferencia de tiempo entre cada punto de gps y cada etapa
        etapas_tx['fecha_dif'] = (
            etapas_tx['fecha_gps'] - etapas_tx['fecha']).dt.total_seconds() / 60

        # Filtrar por diferencia de fecha positiva y ordenar por id, anillo y fecha_dif
        etapas_tx = etapas_tx.loc[etapas_tx.fecha_dif > 0, :]

        if len(etapas_tx) > 0:

            # Calcular la distancia entre h3 del destino de la etapa y h3 del gps

            gps_dict = etapas_tx.reindex(
                columns=['h3_d_gps_res', 'h3']).to_dict("records")
            etapas_tx.loc[:, ["distancia_h3"]] = list(
                map(distancia_h3_gps_leg, gps_dict))

            # Calcular el tiempo mínimo de destino por id
            etapas_tx['min_fecha_d'] = etapas_tx.groupby(
                ['id_legs']).fecha_gps.transform('min')
            etapas_tx['min_fecha_d'] = round(
                (etapas_tx.fecha_gps -
                 etapas_tx['min_fecha_d']).dt.total_seconds() / 60,
                1)

            # Filtrar por tiempo mínimo de destino menor a 20 minutos y ordenar por distancia_h3
            etapas_tx = etapas_tx.loc[etapas_tx.min_fecha_d < 20, :]
            etapas_tx = etapas_tx.sort_values(
                ['id_legs', 'ring', 'distancia_h3', 'min_fecha_d'])

            # Obtener la primera ocurrencia por id - elijo el gps que se encuentra más cerca del destino
            etapas_tx = etapas_tx.groupby('id_legs', as_index=False).first()

            # Agregar resultado a la lista
            etapas_result_list.append(etapas_tx)

    # Concatenar todos los resultados acumulados
    etapas_result = pd.concat(etapas_result_list, ignore_index=True)

    legs_to_gps_d = etapas_result.reindex(
        columns=['dia', 'id_legs', 'id_gps'])

    legs_to_gps_d.to_sql("legs_to_gps_destination", conn_data,
                         if_exists='append', index=False)

    # Unir los resultados con el DataFrame original de etapas
    travel_times = legs.reindex(columns=['dia', 'id', 'fecha', 'distance_osm_drive'])\
        .merge(
            etapas_result.reindex(columns=['id_legs', 'fecha_gps']),
            how='left', left_on=['id'], right_on=['id_legs'])

    # Calcular el tiempo de viaje en minutos y velocidad comercial
    travel_times['travel_time_min'] = round(
        (travel_times['fecha_gps'] - travel_times['fecha']).dt.total_seconds() / 60, 1)

    travel_times = travel_times.loc[travel_times.travel_time_min > 0, :]
    travel_times.loc[:, 'commercial_speed'] = (
        travel_times['distance_osm_drive'] /
        (travel_times['travel_time_min']/60)
    ).round(1)

    tot_gps = len(travel_times)
    tot_gps_asig = travel_times.travel_time_min.notna().sum()
    print('% imputado', round(tot_gps_asig / tot_gps * 100, 1))

    travel_times.to_sql("travel_times_gps", conn_data,
                        if_exists='append', index=False)
    conn_data.close()

    return etapas_result


def distancia_h3_gps_leg(row):
    return h3.h3_distance(row["h3_d_gps_res"], row["h3"])
