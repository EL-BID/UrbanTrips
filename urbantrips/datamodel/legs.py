import pandas as pd
import itertools
from urbantrips.geo.geo import referenciar_h3
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    agrego_indicador,
    eliminar_tarjetas_trx_unica)


def create_legs_from_transactions(criterio_orden_transacciones):
    """
    Esta function toma las transacciones de la db
    las estructura en etapas con sus id y id viaje
    y crea la tabla etapas en la db
    """
    print("ESTRUCTURANDO TRANSACCIONES EN ETAPAS")
    print("Estableciendo conexion con la db")

    conn = iniciar_conexion_db(tipo='data')
    q = """
    SELECT * from transacciones t
    where t.id > (select    coalesce(max(id),-1) from etapas)
    """
    legs = pd.read_sql_query(
        q,
        conn,
        parse_dates={"fecha": "%Y-%m-%d %H:%M:%S"},
    )

    # asignar id h3
    configs = leer_configs_generales()
    res = configs["resolucion_h3"]
    legs = referenciar_h3(df=legs, res=res, nombre_h3="h3_o")

    # crear columna delta
    if criterio_orden_transacciones["criterio"] == "orden_trx":
        legs["delta"] = None
    elif criterio_orden_transacciones["criterio"] == "fecha_completa":
        legs = crear_delta_trx(legs)
    else:
        raise ValueError("ordenamiento_transacciones mal especificado")
    # asignar nuevo id tarjeta trx simultaneas
    legs = cambiar_id_tarjeta_trx_simul(
        legs, criterio_orden_transacciones, conn)
    # elminar casos de nuevas tarjetas con trx unica
    legs = eliminar_tarjetas_trx_unica(legs)
    # asignar ids de viajes y etapas
    legs = asignar_id_viaje_etapa(legs, criterio_orden_transacciones)

    print("Creando factores de expansion...")
    # Actualizo factores de expansión
    # Crear un factor de expansion de las trx que no se subieron a etapas
    factores_expansion = legs\
        .groupby(['dia', 'id_tarjeta'], as_index=False)\
        .agg(
            cant_trx=('id', 'count'),
            factor_expansion_original=('factor_expansion', 'min')
        )

    factores_expansion['factor_expansion'] = 0
    factores_expansion['factor_calibracion'] = 0
    factores_expansion = factores_expansion.reindex(
        columns=['dia',
                 'id_tarjeta',
                 'factor_expansion',
                 'factor_expansion_original',
                 'factor_calibracion',
                 'cant_trx',
                 'id_tarjeta_valido'
                 ])

    # Subir a la base
    factores_expansion.to_sql("factores_expansion",
                              conn, if_exists="append", index=False)

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
            "h3_o"
        ]
    )
    print(f"Subiendo {len(legs)} registros a la tabla etapas en la db")

    legs.to_sql("etapas", conn, if_exists="append", index=False)
    print("Fin subir etapas")
    agrego_indicador(legs,
                     'Cantidad de etapas pre imputacion de destinos',
                     'etapas',
                     0,
                     var_fex='')
    conn.close()


@duracion
def crear_delta_trx(trx):
    """
    Esta funcion toma una tabla trx con un campo de fecha completo
    con horas y minutos y calcula un delta en segundos para cada trx
    con respecto a la trx anterior
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


@duracion
def cambiar_id_tarjeta_trx_simul(trx, criterio_orden_transacciones, conn):
    """
    Esta funcion toma un DF de trx y asigna un nuevo id_tarjeta
    a las transacciones simultaneas
    """
    print("Creando nuevos id tajetas para trx simultaneas")
    trx_c = trx.copy()
    if criterio_orden_transacciones["criterio"] == "orden_trx":
        print("Utilizando orden_trx")
        trx_c, tarjetas_duplicadas = cambiar_id_tarjeta_trx_simul_orden_trx(
            trx_c)
    elif criterio_orden_transacciones["criterio"] == "fecha_completa":
        print("Utilizando fecha completa")
        ventana_duplicado = criterio_orden_transacciones["ventana_duplicado"]
        trx_c, tarjetas_duplicadas = cambiar_id_tarjeta_trx_simul_fecha(
            trx_c, ventana_duplicado
        )
    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    print(f"Subiendo {len(tarjetas_duplicadas)} tarjetas duplicadas a la db")
    if len(tarjetas_duplicadas) > 0:
        tarjetas_duplicadas.to_sql(
            "tarjetas_duplicadas", conn, if_exists="append", index=False
        )
    print("Fin subir tarjetas duplicadas")
    print("Fin creacion nuevos id tajetas para trx simultaneas")

    return trx_c


@duracion
def cambiar_id_tarjeta_trx_simul_fecha(trx, ventana_duplicado):
    """
    Esta funcion toma un DF de trx y una ventana de tiempo en minutos
    para detectar duplicados y asigna un nuevo id_tarjeta a estos en
    base al delta de tiempo con respecto a la trx anterior
    """
    # convertir ventana en segundos
    ventana_duplicado = ventana_duplicado * 60
    # seleccinar atributos para considerar duplicados
    subset_dup = ["dia", "id_tarjeta", "id_linea", "interno"]

    # detectar duplicados por criterio de delta y atributos
    duplicados_ventana = (trx.delta > 0) & (trx.delta <= ventana_duplicado)
    duplicados_atributos = trx.duplicated(subset=subset_dup)
    duplicados = duplicados_ventana & duplicados_atributos
    trx["duplicados_ventana"] = duplicados_ventana

    subset_dup = subset_dup + ["duplicados_ventana"]
    # crear para duplicado por delta dentro de dia tarjeta linea interno
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
    interno y ubicacion
    """
    subset_dup = [
        "dia",
        "id_tarjeta",
        "hora",
        "orden_trx",
        "id_linea",
        "interno",
        "h3_o",
    ]

    # detectar duplicados por criterio de atributos
    duplicados = trx.duplicated(subset=subset_dup)

    # crear para duplicado por dia tarjeta linea interno
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


@duracion
def asignar_id_viaje_etapa(trx, criterio_orden_transacciones):
    """
    Esta funcion toma un DF de trx
    un dict con el criterio de asignar ids viajes-etapa y ventana de tiempo
    y asigna id_viaje y id_etapa
    en base a ventana de tiempo o a hora y orden_trx
    """
    print("Crear un id para viajes y etapas")

    if criterio_orden_transacciones["criterio"] == "orden_trx":
        print("Utilizando orden_trx")
        trx = asignar_id_viaje_etapa_orden_trx(trx)

    elif criterio_orden_transacciones["criterio"] == "fecha_completa":
        print("Utilizando fecha_completa")
        ventana_viajes = criterio_orden_transacciones["ventana_viajes"]
        trx = asignar_id_viaje_etapa_fecha_completa(trx, ventana_viajes)

    else:
        raise ValueError("ordenamiento_transacciones mal especificado")

    print("Fin creacion de un id para viajes y etapas")
    return trx


def asignar_id_viaje_etapa_fecha_completa(trx, ventana_viajes):
    """
    Esta funcion toma un DF de trx y asigna id_viaje y id_etapa
    en base al dia, delta de trx y a una ventana de tiempo
    """

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
