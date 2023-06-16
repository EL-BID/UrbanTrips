
import pandas as pd
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db)




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
    df['cumsum'] = cumsum_mismo_od.values

    # crear nuevos id viaje y etapa
    df['nuevo_id_viaje'] = (df.id_viaje + df['cumsum']).map(int)
    df['nuevo_id_etapa'] = df.id_etapa.copy()
    df.loc[df.mismo_od == 1, 'nuevo_id_etapa'] = 1

    df = df\
        .drop(['mismo_od', 'cumsum',
               'id_viaje', 'id_etapa'], axis=1)\
        .rename(columns={'nuevo_id_viaje': 'id_viaje',
                         'nuevo_id_etapa': 'id_etapa'})\
        .reindex(columns=['id', 'id_tarjeta', 'dia', 'id_viaje', 'id_etapa',
                          'tiempo', 'hora', 'modo', 'id_linea', 'id_ramal',
                          'interno', 'latitud', 'longitud', 'h3_o', 'h3_d', 'od_validado',
                          'factor_expansion_original', 'factor_expansion_linea', 'factor_expansion_tarjeta'])
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


def create_trips_from_legs():
    """
    Esta función toma la tabla de etapas y produce la de viajes y usuarios
    """

    # Leer etapas que no esten en ya viajes por id_tarjeta, id_viaje, dia
    conn = iniciar_conexion_db(tipo='data')

    # etapas = pd.read_sql_query(
    #     """
    #     with etapas_not_viajes as (
    #     select e.*
    #     from etapas e
    #     LEFT JOIN viajes v
    #     USING (id_tarjeta,id_viaje,dia)
    #     where v.id_tarjeta is null and e.od_validado==1
    #     )
    #     SELECT e.*
    #     FROM etapas_not_viajes e
    #     ORDER BY dia,id_tarjeta,id_viaje,id_etapa,hora
    #     """,
    #     conn,
    # )
    
    dias_ultima_corrida = pd.read_sql_query(
                                """
                                SELECT *
                                FROM dias_ultima_corrida
                                """,
                                conn,
                                )    
    
    etapas_ant = pd.read_sql_query(
                                    """
                                    SELECT *
                                    FROM etapas e     
                                    JOIN dias_ultima_corrida d
                                    ON e.dia = d.dia
                                    """,
                                    conn,
                                    )
    
    
    etapas = etapas_ant[(etapas_ant.od_validado==1)&(etapas_ant.factor_expansion_linea.isna())].copy()
        
    if len(etapas) > 0:
    
        print("Creando factores de expansion...")
        # Actualizo factores de expansión
        # Crear un factor de expansion de las trx que no se subieron a etapas
        factores_expansion = etapas\
            .groupby(['dia', 'id_tarjeta'], as_index=False)\
            .agg(
                cant_trx=('id', 'count'),
                factor_expansion_original=('factor_expansion_original', 'min')
            )

        factores_expansion['factor_expansion_tarjeta'] = 0
        factores_expansion['factor_calibracion'] = 0
        factores_expansion = factores_expansion.reindex(
            columns=['dia',
                     'id_tarjeta',                 
                     'factor_expansion_original',
                     'factor_expansion_tarjeta',
                     'factor_calibracion',
                     'cant_trx',
                     'id_tarjeta_valido'
                     ])

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

            factores_expansion.loc[mask, 'factor_expansion_tarjeta'] = (
                factores_expansion.loc[mask,
                                       'factor_expansion_original'] + calib_trx)

        # Cálculo de factores de expansion por línea
        transacciones_linea_ant = pd.read_sql_query(
                                """
                                SELECT *
                                FROM transacciones_linea t
                                JOIN dias_ultima_corrida d
                                ON t.dia = d.dia
                                """,
                                conn,
                                )

        etapas = etapas.drop(['factor_expansion_linea', 
                              'factor_expansion_tarjeta'], axis=1)

        transacciones_linea = transacciones_linea_ant[['dia', 'id_linea', 'transacciones']].merge(
                                etapas.groupby(
                                    ['dia', 'id_linea'], 
                                    as_index=False).agg(
                                    {'factor_expansion_original':'sum'}
                                )).rename(columns={'factor_expansion_original':'transacciones_validas'})

        transacciones_linea['factor_expansion_linea'] = transacciones_linea['transacciones'] / transacciones_linea['transacciones_validas']

        transacciones_linea_ant = transacciones_linea_ant[~(
                   (transacciones_linea_ant.dia.isin(transacciones_linea.dia.unique().tolist()))
                  )]

        transacciones_linea=pd.concat([transacciones_linea_ant, transacciones_linea], ignore_index=True)
        
        # borro si ya existen etapas de una corrida anterior
        values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
        query = f"DELETE FROM transacciones_linea WHERE dia IN ({values})"
        conn.execute(query)
        conn.commit()

        transacciones_linea.to_sql("transacciones_linea", conn, if_exists="append", index=False)

        factores_expansion_etapas = etapas[['id', 
                                            'dia', 
                                            'id_etapa', 
                                            'id_viaje', 
                                            'id_tarjeta', 
                                            'id_linea', 
                                            'factor_expansion_original'
                                              ]].merge(
                                                        transacciones_linea[['id_linea', 'factor_expansion_linea']]
                                                                      , how='left')

        factores_expansion_etapas['factor_expansion_linea'] = factores_expansion_etapas['factor_expansion_linea'] * factores_expansion_etapas['factor_expansion_original']

        factores_expansion_etapas = factores_expansion_etapas.merge(factores_expansion[['dia', 'id_tarjeta', 'factor_expansion_tarjeta']], how='left')

        etapas = etapas.merge(factores_expansion_etapas[['id', 'factor_expansion_linea', 'factor_expansion_tarjeta']])

        # Guarda viajes y usuarios en sqlite
        agg_func_dict = {
            "tiempo": "first",
            "hora": "first",
            "h3_o": "first",
            "h3_d": "last",
            "od_validado": "min",
            "factor_expansion_linea":"mean",
            "factor_expansion_tarjeta":"mean"
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
        viajes_cols = ['id_tarjeta', 
                       'id_viaje', 
                       'dia', 'tiempo', 
                       'hora',
                       'cant_etapas', 
                       'modo', 
                       'autobus', 
                       'tren', 
                       'metro',
                       'tranvia', 
                       'brt', 
                       'otros', 
                       'h3_o', 
                       'h3_d', 
                       'od_validado', 
                       'factor_expansion_linea', 
                       'factor_expansion_tarjeta']
        viajes = viajes.reindex(columns=viajes_cols)

        print('Subiendo tabla de viajes a la db...')
        
        # borro si ya existen viajes de una corrida anterior
        values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
        query = f"DELETE FROM viajes WHERE dia IN ({values})"
        conn.execute(query)
        conn.commit()        
        viajes.to_sql("viajes", conn, if_exists="append", index=False)

        print('Creando tabla de usuarios...')
        # Crear tabla usuarios
        usuarios = viajes.groupby(["dia", "id_tarjeta"], as_index=False).agg(
            {"od_validado": "min",
             "id_viaje": 'count', 
             "factor_expansion_linea":"mean",
             "factor_expansion_tarjeta":"mean"
            }).rename(columns={'id_viaje': 'cant_viajes'})

        print('Subiendo tabla de usuarios a la db...')
        
        # borro si ya existen etapas de una corrida anterior
        values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
        query = f"DELETE FROM usuarios WHERE dia IN ({values})"
        conn.execute(query)
        conn.commit()        
        # Guarda viajes y usuarios en sqlite
        usuarios.to_sql("usuarios", conn, if_exists="append", index=False)
        print('Fin de creacion de tablas viajes y usuarios')

        etapas = etapas[['id', 
                         'id_tarjeta', 
                         'dia', 
                         'id_viaje', 
                         'id_etapa', 
                         'tiempo', 
                         'hora',
                         'modo', 
                         'id_linea', 
                         'id_ramal', 
                         'interno', 
                         'latitud', 
                         'longitud',
                         'h3_o', 
                         'h3_d',
                         'od_validado',
                         'factor_expansion_original', 
                         'factor_expansion_linea',
                         'factor_expansion_tarjeta']]

        etapas_ant = etapas_ant[~(
            (etapas_ant.id.isin(etapas.id.unique())) &
            (etapas_ant.dia.isin(etapas.dia.unique())) 
                                )]

        etapas = pd.concat(
            [etapas_ant, etapas], ignore_index=True)
        
        etapas['factor_expansion_linea'] = etapas['factor_expansion_linea'].fillna(0)
        etapas['factor_expansion_tarjeta'] = etapas['factor_expansion_tarjeta'].fillna(0)

        # borro si ya existen etapas de una corrida anterior
        values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
        query = f"DELETE FROM etapas WHERE dia IN ({values})"
        conn.execute(query)
        conn.commit()        
        etapas.to_sql("etapas", conn, if_exists="append", index=False)

    conn.close()
    
    
@duracion
def rearrange_trip_id_same_od():
    """
    Esta funcion toma la tabla de etapas y altera los ids de etapas que
    al eslabonarse en viajes, resultan con un mismo par od
    """

    print("Corrigiendo ids de etapas y viajes con mismo od")
    conn_data = iniciar_conexion_db(tipo='data')

    print("Leer etapas")
    # Traer etapas que no esten ya procesadas en viajes    
    # etapas = pd.read_sql_query(
    #     """
    #     with etapas_not_viajes as (
    #     select e.*
    #     from etapas e
    #     LEFT JOIN viajes v
    #     USING (id_tarjeta,id_viaje,dia)
    #     where v.id_tarjeta is null 
    #     )
    #     SELECT e.*
    #     FROM etapas_not_viajes e
    #     ORDER BY dia,id_tarjeta,id_viaje,id_etapa,hora
    #     """,
    #     conn_data,
    # )
    
    dias_ultima_corrida = pd.read_sql_query(
                                """
                                SELECT *
                                FROM dias_ultima_corrida
                                """,
                                conn_data,
                                )    
    
    etapas = pd.read_sql_query(
                                """
                                SELECT *
                                FROM etapas e
                                JOIN dias_ultima_corrida d
                                ON e.dia = d.dia
                                """,
                                conn_data,
                                )

    print("Crear nuevos ids")
    # crear nuevos ids
    nuevos_ids_etapas_viajes = cambia_id_viajes_etapas_tarjeta_dia(etapas)

    print('Actualizando nuevos ids en etapas')
    
    etapas = etapas[~(etapas.id.isin(nuevos_ids_etapas_viajes.id.unique()))]
    etapas = pd.concat([etapas, nuevos_ids_etapas_viajes])
    etapas = etapas.sort_values('id').reset_index(drop=True)
    
    # borro si ya existen etapas de una corrida anterior
    values = ', '.join([f"'{val}'" for val in dias_ultima_corrida['dia']])
    query = f"DELETE FROM etapas WHERE dia IN ({values})"
    conn_data.execute(query)
    conn_data.commit()

    etapas.to_sql("etapas", conn_data,
                           if_exists="append", index=False)
    
    print('len etapas final', len(etapas))

    conn_data.close()

    print("Fin correxión de ids de etapas y viajes con mismo od")
