import pandas as pd
import os
from pandas.io.sql import DatabaseError
from urbantrips.utils.utils import (leer_configs_generales,
                                    duracion,
                                    iniciar_conexion_db,
                                    leer_alias, agrego_indicador)


@duracion
def create_line_and_branches_metadata():
    """
    Esta funcion lee la ubicacion del archivo de informacion de las lineas
    de la ciudad y crea una tabla en la db
    """

    conn_insumos = iniciar_conexion_db(tipo='insumos')
    configs = leer_configs_generales()
    try:
        tabla_lineas = configs["nombre_archivo_informacion_lineas"]
        con_ramales = configs["lineas_contienen_ramales"]

        if tabla_lineas is not None:
            crear_tabla_metadata_lineas(conn_insumos)

            print('Leyendo tabla con informacion de lineas')
            ruta = os.path.join("data", "data_ciudad", tabla_lineas)
            info = pd.read_csv(ruta)

            # chequear que tengan todas las columnas de id nombre y modo
            if con_ramales:
                crear_tabla_metadata_ramales(conn_insumos)

                cols = ['id_linea', 'nombre_linea',
                                    'id_ramal', 'nombre_ramal', 'modo']
            else:
                cols = ['id_linea', 'nombre_linea', 'modo']

            assert pd.Series(cols).isin(info.columns).all()

            # chequear que modo coincida con el config de modos homologados
            try:
                modos_homologados = configs["modos"]
                zipped = zip(modos_homologados.values(),
                             modos_homologados.keys())
                modos_homologados = {k: v for k, v in zipped}

                assert pd.Series(info.modo.unique()).isin(
                    modos_homologados.keys()).all()

                info['modo'] = info['modo'].replace(modos_homologados)

            except KeyError:
                pass

            # que no haya missing en id_linea y id_ramal
            assert not info.id_linea.isna().any()
            assert info.dtypes['id_linea'] == int

            lineas_cols = ['id_linea', 'nombre_linea',
                           'modo', 'empresa', 'descripcion']

            info_lineas = info.reindex(columns=lineas_cols)

            if con_ramales:
                info_lineas = info_lineas.drop_duplicates(subset='id_linea')

                ramales_cols = ['id_ramal', 'id_linea',
                                'nombre_ramal', 'modo', 'empresa',
                                'descripcion']

                info_ramales = info.reindex(columns=ramales_cols)
                assert not info_ramales.id_ramal.isna().any()
                assert not info_ramales.id_ramal.duplicated().any()
                assert info_ramales.dtypes['id_ramal'] == int
                info_ramales.to_sql(
                    "metadata_ramales", conn_insumos, if_exists="replace",
                    index=False)

            info_lineas.to_sql(
                "metadata_lineas", conn_insumos, if_exists="replace",
                index=False)

    except KeyError:
        print("No hay tabla con informacion configs")
    conn_insumos.close()


def crear_tabla_metadata_lineas(conn_insumos):

    conn_insumos.execute("DROP TABLE IF EXISTS metadata_lineas;")

    conn_insumos.execute(
        """
        CREATE TABLE metadata_lineas
            (id_linea INT PRIMARY KEY     NOT NULL,
            nombre_linea text not null,
            modo text not null,
            empresa text,
            descripcion text
            )
        ;
        """
    )


def crear_tabla_metadata_ramales(conn_insumos):
    conn_insumos.execute("DROP TABLE IF EXISTS metadata_ramales;")

    conn_insumos.execute(
        """
        CREATE TABLE metadata_ramales
            (id_ramal INT PRIMARY KEY     NOT NULL,
            id_linea int not null,
            nombre_ramal text not null,
            modo text not null,
            empresa text,
            descripcion text
            )
        ;
        """
    )


def persist_datamodel_tables():
    """
    Esta funcion lee los datos de etapas, viajes y usuarios
    le suma informacion de distancias y de zonas
    y las guarda en csv
    """

    print('Guarda tablas de etapas, viajes y usuarios en formato .csv')

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

    zonas = pd.read_sql_query("select * from zonas", conn_insumos)
    zonas = zonas.drop(['fex', 'latitud', 'longitud'], axis=1)
    zonas_o = zonas.copy()
    zonas_d = zonas.copy()
    cols_o = zonas.columns.copy() + '_o'
    cols_d = zonas.columns.copy() + '_d'
    zonas_o.columns = cols_o
    zonas_d.columns = cols_d

    q = """
        SELECT *
        from etapas
        LEFT JOIN destinos
        ON etapas.id = destinos.id
    """
    etapas = pd.read_sql_query(q, conn_data)
    etapas = etapas.merge(zonas_o, how='left').merge(zonas_d, how='left')
    etapas = etapas.merge(distancias, how='left')

    viajes = pd.read_sql_query("select * from viajes", conn_data)
    viajes = viajes.merge(zonas_o, how='left').merge(zonas_d, how='left')
    viajes = viajes.merge(distancias, how='left')

    print("Leyendo informacion de usuarios...")
    usuarios = pd.read_sql_query("select * from usuarios", conn_data)

    factores_expansion = pd.read_sql_query(
        """
        SELECT *
        FROM factores_expansion
        """,
        conn_data,
    )
    factores = factores_expansion.reindex(
        columns=['dia', 'id_tarjeta', 'factor_expansion'])

    etapas = etapas\
        .merge(factores, on=['dia', 'id_tarjeta'])
    viajes = viajes\
        .merge(factores, on=['dia', 'id_tarjeta'])
    usuarios = usuarios\
        .merge(factores, on=['dia', 'id_tarjeta'])

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
                                    as_index=False).factor_expansion.sum(),
                     'Cantidad de tarjetas finales',
                     'usuarios',
                     0,
                     var_fex='')

    agrego_indicador(etapas[etapas.od_validado == 1].groupby(
        ['dia', 'id_tarjeta'],
        as_index=False).factor_expansion.min(),
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
