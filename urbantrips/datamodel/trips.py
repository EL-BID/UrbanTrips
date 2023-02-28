
import pandas as pd
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db)


def create_trips_from_legs():
    """
    Esta función toma la tabla de etapas y produce la de viajes y usuarios
    """

    # Leer etapas que no esten en ya viajes por id_tarjeta, id_viaje, dia
    conn = iniciar_conexion_db(tipo='data')

    etapas = pd.read_sql_query(
        """
        with etapas_not_viajes as (
        select e.*
        from etapas e
        LEFT JOIN viajes v
        USING (id_tarjeta,id_viaje,dia)
        where v.id_tarjeta is null
        )
        SELECT e.*, d.h3_d,d.od_validado
        FROM etapas_not_viajes e
        LEFT JOIN destinos d
        ON e.id = d.id
        ORDER BY dia,id_tarjeta,id_viaje,id_etapa,hora
        """,
        conn,
    )

    # Traer los factores de expansion de las tarjeta dia
    # que no hayan sido procesados en viajes
    factores_expansion = pd.read_sql_query(
        """
        select f.*
        from factores_expansion f
        LEFT JOIN viajes v
        USING (dia,id_tarjeta)
        where v.id_tarjeta is null
        """,
        conn,
    )
    # Leer los indicadores de los dias presentes en etapas
    dias_etapa = etapas.dia.unique()
    string_dias_etapa = "','".join(dias_etapa)

    total_trx_dia = pd.read_sql_query(
        f"""
        SELECT *
        FROM indicadores
        where detalle = 'Cantidad de transacciones totales'
        and dia in ('{string_dias_etapa}')
        """,
        conn,
    )

    print(f'Creando tabla de viajes de {len(etapas)} etapas')
    # Crear tabla viajes
    etapas = pd.concat([etapas, pd.get_dummies(etapas.modo)], axis=1)

    # Corrijo factor de expansion
    tmp_fex = etapas\
        .groupby(['dia', 'id_tarjeta'], as_index=False)\
        .agg(cant_etapas=('id', 'count'))

    factores_expansion = factores_expansion.merge(
        tmp_fex, how='left', on=['dia', 'id_tarjeta'])

    tarj_val = factores_expansion.cant_trx == factores_expansion.cant_etapas
    factores_expansion = factores_expansion.loc[tarj_val, :]
    factores_expansion['id_tarjeta_valido'] = 1

    # Sumar a etapas el factor de expansion original
    etapas = etapas.merge(factores_expansion.loc[:,
                                                 ['dia', 'id_tarjeta',
                                                  'factor_expansion_original']
                                                 ], on=['dia', 'id_tarjeta'])

    # PARA CADA DIA
    for _, i in total_trx_dia.iterrows():
        # Calcular el total de transacciones originales del dia
        total_transacciones = i.indicador
        # El total de etapas considerando el factor de expansion original
        total_etapas = etapas[(etapas.dia == i.dia)
                              ].factor_expansion_original.sum()
        # Cantidad de registros del dia
        len_etapas = len(etapas[(etapas.dia == i.dia)])

        # Calcular la calibracion
        calib_trx = round((total_transacciones-total_etapas)/len_etapas, 2)

        # Aplicar la calibracion sobre la expansion original
        mask = (factores_expansion.dia == i.dia)

        factores_expansion.loc[mask, 'factor_calibracion'] = calib_trx

        factores_expansion.loc[mask, 'factor_expansion'] = (
            factores_expansion.loc[mask,
                                   'factor_expansion_original'] + calib_trx)

    # Guardar factores de expansión
    factores_expansion = factores_expansion.drop(['cant_etapas'], axis=1)

    # Borrar los factores de expansión previos de los dias de etapas
    q = f"""
    delete from factores_expansion
    where dia in ('{string_dias_etapa}')
    """
    cur = conn.cursor()
    cur.execute(q)
    conn.commit()

    # Subir a la db
    factores_expansion.to_sql("factores_expansion",
                              conn, if_exists="append", index=False)

    # Guarda viajes y usuarios en sqlite
    # TODO: remove od_validado
    agg_func_dict = {
        "tiempo": "first",
        "hora": "first",
        "h3_o": "first",
        "h3_d": "last",
        "od_validado": "min"
    }
    viajes = etapas.groupby(
        ["dia", "id_tarjeta", "id_viaje"],
        as_index=False,
    ).agg(agg_func_dict)

    cols = pd.get_dummies(etapas.modo).columns.tolist()

    viajes = viajes.merge(
        etapas.groupby(
            ["dia", "id_tarjeta", "id_viaje"],
            as_index=False,
        )[cols].max()
    )
    # Sumar cantidad de etapas por modo
    viajes["tmp_cant_modos"] = viajes[cols].sum(axis=1)

    for i in cols:
        viajes.loc[viajes[i] == 1, "modo"] = i
    viajes.loc[viajes.tmp_cant_modos > 1, "modo"] = "Multimodal"
    viajes = viajes.drop(cols, axis=1)

    viajes = viajes.merge(
        etapas.groupby(
            ["dia", "id_tarjeta", "id_viaje"],
            as_index=False,
        )[cols].sum()
    )
    viajes["cant_etapas"] = viajes[cols].sum(axis=1)

    print("Clasificando modalidad...")
    # Clasificar los viajes como Multimodal o Multietapa
    viajes.loc[
        (viajes.cant_etapas > 1) & (viajes.modo != "Multimodal"), "modo"
    ] = "Multietapa"

    # TODO: remove od_validado
    viajes_cols = ['id_tarjeta', 'id_viaje', 'dia', 'tiempo', 'hora',
                   'cant_etapas', 'modo', 'autobus', 'tren', 'metro',
                   'tranvia', 'brt', 'otros', 'h3_o', 'h3_d', 'od_validado']
    viajes = viajes.reindex(columns=viajes_cols)

    print('Subiendo tabla de viajes a la db...')
    viajes.to_sql("viajes", conn, if_exists="append", index=False)

    print('Creando tabla de usuarios...')
    # Crear tabla usuarios
    usuarios = viajes.groupby(["dia", "id_tarjeta"], as_index=False).agg(
        {"od_validado": "min",
         "id_viaje": 'count'}).rename(columns={'id_viaje': 'cant_viajes'})

    print('Subiendo tabla de usuarios a la db...')
    # Guarda viajes y usuarios en sqlite
    usuarios.to_sql("usuarios", conn, if_exists="append", index=False)
    print('Fin de creacion de tablas viajes y usuarios')
    conn.close()


@duracion
def rearrange_trip_id_same_od():
    """
    Esta funcion toma la tabla de etapas y altera los ids de etapas que
    al eslabonarse en viajes, resultan con un mismo par od
    """
    print("Corrigiendo ids de etapas y viajes con mismo od")
    conn_data = iniciar_conexion_db(tipo='data')

    # Traer etapas que no esten ya procesadas en viajes
    print("Leer etapas")
    etapas = pd.read_sql_query(
        """
        with etapas_not_viajes as (
        select e.*
        from etapas e
        LEFT JOIN viajes v
        USING (id_tarjeta,id_viaje,dia)
        where v.id_tarjeta is null
        )
        SELECT e.*,d.h3_d,d.od_validado
        FROM etapas_not_viajes e
        LEFT JOIN destinos d
        ON e.id = d.id
        ORDER BY dia,id_tarjeta,id_viaje,id_etapa,hora
        """,
        conn_data,
    )
    # Crear tabla viajes
    etapas['od_validado'] = etapas['od_validado'].fillna(0)

    print("Crear nuevos ids")
    # crear nuevos ids
    nuevos_ids_etapas_viajes = cambia_id_viajes_etapas_tarjeta_dia(etapas)

    nuevos_ids_etapas_viajes.to_sql(
        "nuevos_ids_etapas_viajes", conn_data, if_exists="replace",
        index=False,)

    n = len(nuevos_ids_etapas_viajes)
    print(
        f'Reemplazando id viaje y etapa de {n} etapas')

    print('Actualizando nuevos ids en etapas')
    q = """
        REPLACE INTO etapas (id,id_tarjeta,dia,id_viaje,id_etapa,tiempo,hora,
                modo,id_linea,id_ramal,interno,latitud,longitud,h3_o)
        SELECT id,id_tarjeta,dia,id_viaje,id_etapa,tiempo,hora,modo,id_linea,
                id_ramal,interno,latitud,longitud,h3_o
        FROM nuevos_ids_etapas_viajes;
        """
    cur = conn_data.cursor()
    cur.execute(q)
    conn_data.commit()

    print("borrando casos de tabla insumo")
    q = "delete from nuevos_ids_etapas_viajes"
    cur = conn_data.cursor()
    cur.execute(q)
    conn_data.commit()

    conn_data.close()

    print("Fin correxión de ids de etapas y viajes con mismo od")


def cambia_id_viajes_etapas_tarjeta_dia(df):
    """
    Esta funcion toma un df de etapas con destinos
    evalua si a nivel de viaje coincide el par od
    y para esos casos cambia los ids de viaje y etapa
    tomando la ultima etapa de cada viaje y convirtiendolo
    en un viaje en si mismo
    """
    # produce una tabla temporal de viajes
    # para detectar los que tienen mismo par od
    viajes_temp = crear_viaje_temp(df)

    # dejar solamente con las tarjeta dia que tienen problemas
    tarjeta_dia_problemas = viajes_temp.reindex(
        columns=['dia', 'id_tarjeta']).drop_duplicates()
    df = df.merge(tarjeta_dia_problemas,
                  on=['dia', 'id_tarjeta'], how='inner')

    # sumar la informacion de la ultima etapa
    df = df.merge(viajes_temp, how='left', on=[
                  'dia', 'id_tarjeta', 'id_viaje', 'id_etapa'])

    cumsum_mismo_od = df\
        .reindex(columns=['id_tarjeta', 'dia', 'mismo_od'])\
        .groupby(['id_tarjeta', 'dia'])\
        .apply(crear_cumsum_mismo_od)['mismo_od']
    df['cumsum'] = cumsum_mismo_od

    # crear nuevos id viaje y etapa
    df['nuevo_id_viaje'] = (df.id_viaje + df['cumsum']).map(int)
    df['nuevo_id_etapa'] = df.id_etapa.copy()
    df.loc[df.mismo_od == 1, 'nuevo_id_etapa'] = 1

    df = df\
        .drop(['h3_d', 'od_validado', 'mismo_od', 'cumsum',
               'id_viaje', 'id_etapa'], axis=1)\
        .rename(columns={'nuevo_id_viaje': 'id_viaje',
                         'nuevo_id_etapa': 'id_etapa'})\
        .reindex(columns=['id', 'id_tarjeta', 'dia', 'id_viaje', 'id_etapa',
                          'tiempo', 'hora', 'modo', 'id_linea', 'id_ramal',
                          'interno', 'latitud', 'longitud', 'h3_o'])
    return df


def crear_viaje_temp(df):
    """
    Esta funcion toma un df de etapas y produce
    un df de viaje temporal para detectar viajes
    con el mismo od
    """
    df = df.groupby(
        ["dia", "id_tarjeta", "id_viaje"],
        as_index=False,
    ).agg(
        h3_o=('h3_o', 'first'),
        h3_d=('h3_d', 'last'),
        od_validado=('od_validado', 'min'),
        cant_etapas=('id_etapa', 'count'),
        id_etapa=('id_etapa', 'last')
    )
    mask = (df.h3_o == df.h3_d) & (df.od_validado == 1) & (
        df.cant_etapas > 1) & (df.od_validado == 1)
    df = df.loc[mask, ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa']]
    df['mismo_od'] = 1
    return df


def crear_cumsum_mismo_od(s):
    return s.cumsum().fillna(method='ffill').fillna(0)
