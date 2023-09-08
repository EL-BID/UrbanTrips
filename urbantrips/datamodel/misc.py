import pandas as pd
import os
from pandas.io.sql import DatabaseError
from urbantrips.utils.utils import (iniciar_conexion_db,
                                    leer_alias, agrego_indicador, duracion)

@duracion
def persist_datamodel_tables():
    """
    Esta funcion lee los datos de etapas, viajes y usuarios
    le suma informacion de distancias y de zonas
    y las guarda en csv
    """

    
    alias = leer_alias()
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    conn_data = iniciar_conexion_db(tipo='data')

    # Leer tablas etapas, viajes, usuarios
    q = """
    select
        h3_o,h3_d,distance_osm_drive,distance_osm_walk,distance_h3
    from
        distancias
    """
    distancias = pd.read_sql_query(q, conn_insumos)

    zonas = pd.read_sql_query("""
                            select * from zonas
                              """,
                              conn_insumos)
    zonas = zonas.drop(['fex', 'latitud', 'longitud'], axis=1)
    zonas_o = zonas.copy()
    zonas_d = zonas.copy()
    cols_o = zonas.columns.copy() + '_o'
    cols_d = zonas.columns.copy() + '_d'
    zonas_o.columns = cols_o
    zonas_d.columns = cols_d

    q = """
        SELECT *
        from etapas e
        where e.od_validado==1
    """
    etapas = pd.read_sql_query(q, conn_data)
    etapas = etapas.merge(zonas_o, how='left').merge(zonas_d, how='left')
    etapas = etapas.merge(distancias, how='left')

    viajes = pd.read_sql_query("""
                                select *
                                from viajes
                               """, conn_data)
    viajes = viajes.merge(zonas_o, how='left').merge(zonas_d, how='left')
    viajes = viajes.merge(distancias, how='left')

    print("Leyendo informacion de usuarios...")
    usuarios = pd.read_sql_query("""
                                SELECT *
                                from usuarios
                                """, conn_data)

    # Grabo resultados en tablas .csv
    print("Guardando informacion de etapas...")
    etapas.to_csv(
        os.path.join("resultados", "data", f"{alias}etapas.csv"),
        index=False,
    )
    print("Guardando informacion de viajes...")
    viajes.to_csv(
        os.path.join("resultados", "data", f"{alias}viajes.csv"),
        index=False,
    )

    print("Guardando informacion de usuarios...")
    usuarios.to_csv(
        os.path.join("resultados", "data", f"{alias}usuarios.csv"),
        index=False,
    )

    agrego_indicador(etapas[etapas.od_validado == 1],
                     'Cantidad total de etapas',
                     'etapas_expandidas',
                     0)

    for i in etapas[etapas.od_validado == 1].modo.unique():
        agrego_indicador(
            etapas.loc[(etapas.od_validado == 1) &
                       (etapas.modo == i)],
            f'Etapas {i}', 'etapas_expandidas', 1)

    agrego_indicador(viajes,
                     'Cantidad de registros en viajes',
                     'viajes',
                     0,
                     var_fex='')

    agrego_indicador(viajes[viajes.od_validado == 1],
                     'Cantidad total de viajes expandidos',
                     'viajes expandidos',
                     0)
    agrego_indicador(viajes[(viajes.od_validado == 1) &
                            (viajes.distance_osm_drive <= 5)],
                     'Cantidad de viajes cortos (<5kms)',
                     'viajes expandidos',
                     1)
    agrego_indicador(viajes[(viajes.od_validado == 1) &
                            (viajes.cant_etapas > 1)],
                     'Cantidad de viajes con transferencia',
                     'viajes expandidos',
                     1)

    agrego_indicador(viajes[viajes.od_validado == 1],
                     'Cantidad total de viajes expandidos',
                     'modos viajes',
                     0)

    for i in viajes[viajes.od_validado == 1].modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.od_validado == 1) &
                       (viajes.modo == i)],
            f'Viajes {i}', 'modos viajes', 1)

    agrego_indicador(viajes[viajes.od_validado == 1],
                     'Distancia de los viajes (promedio en kms)',
                     'avg',
                     0,
                     var='distance_osm_drive',
                     aggfunc='mean')

    agrego_indicador(viajes[viajes.od_validado == 1],
                     'Distancia de los viajes (mediana en kms)',
                     'avg',
                     0,
                     var='distance_osm_drive',
                     aggfunc='median')

    for i in viajes[viajes.od_validado == 1].modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.od_validado == 1) &
                       (viajes.modo == i)],
            f'Distancia de los viajes (promedio en kms) - {i}',
            'avg', 0,
            var='distance_osm_drive',
            aggfunc='mean')

    for i in viajes[viajes.od_validado == 1].modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.od_validado == 1) &
                       (viajes.modo == i)],
            f'Distancia de los viajes (mediana en kms) - {i}',
            'avg', 0,
            var='distance_osm_drive',
            aggfunc='median')

    agrego_indicador(viajes[viajes.od_validado == 1],
                     'Etapas promedio de los viajes',
                     'avg',
                     0,
                     var='cant_etapas',
                     aggfunc='mean')

    agrego_indicador(usuarios,
                     'Cantidad promedio de viajes por tarjeta',
                     'avg',
                     0,
                     var='cant_viajes',
                     aggfunc='mean')

    agrego_indicador(etapas.groupby(['dia', 'id_tarjeta'],
                                    as_index=False).
                     factor_expansion_linea.sum(),
                     'Cantidad de tarjetas finales',
                     'usuarios',
                     0,
                     var_fex='')

    agrego_indicador(etapas[etapas.od_validado == 1].groupby(
        ['dia', 'id_tarjeta'],
        as_index=False).factor_expansion_linea.min(),
        'Cantidad total de usuarios',
        'usuarios expandidos',
        0)

    print(
        "Resultados:",
        "{:,}".format(len(etapas)).replace(",", "."),
        "(etapas)",
        "{:,}".format(len(viajes)).replace(",", "."),
        "(viajes)",
        "{:,}".format(len(usuarios)).replace(",", "."),
        "(usuarios)",
    )

    print(
        "Validados :",
        "{:,}".format(len(etapas[etapas.od_validado == 1])).replace(",", "."),
        "(etapas)",
        "{:,}".format(len(viajes[viajes.od_validado == 1])).replace(",", "."),
        "(viajes)",
        "{:,}".format(
            len(usuarios[usuarios.od_validado == 1])).replace(",", "."),
        "(usuarios)",
    )

    print(
        "    %     :",
        "{:,}".format(
            round(len(etapas[etapas.od_validado == 1]) / len(etapas) * 100)
        ).replace(",", ".")
        + "%",
        "(etapas)",
        "{:,}".format(
            round(len(viajes[viajes.od_validado == 1]) / len(viajes) * 100)
        ).replace(",", ".")
        + "%",
        "(viajes)",
        "{:,}".format(
            round(len(usuarios[usuarios.od_validado == 1]) /
                  len(usuarios) * 100)
        ).replace(",", ".")
        + "%",
        "(usuarios)",
    )

    conn_data.close()
    conn_insumos.close()
    tabla_indicadores()


def tabla_indicadores():
    alias = leer_alias()
    conn_data = iniciar_conexion_db(tipo='data')

    try:
        indicadores = pd.read_sql_query(
            """
            SELECT *
            FROM indicadores
            """,
            conn_data,
        )
    except DatabaseError as e:
        indicadores = pd.DataFrame([])

    db_path = os.path.join("resultados", "tablas",
                           f"{alias}indicadores.xlsx")
    indicadores[['dia', 'detalle', 'indicador', 'porcentaje']
                ].to_excel(db_path, index=False)

    conn_data.close()
