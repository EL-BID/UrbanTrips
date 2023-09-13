import pandas as pd
import sqlite3
import os
import yaml
import time
from functools import wraps
import h3
import numpy as np
import weightedstats as ws
from pandas.io.sql import DatabaseError
import datetime
os.environ['USE_PYGEOS'] = '0'


def duracion(f):
    @ wraps(f)
    def wrap(*args, **kw):
        print('')
        print(
            f"{f.__name__} ({str(datetime.datetime.now())[:19]})\n", end="",
            flush=True)
        print('-' * (len(f.__name__)+22))

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"Finalizado {f.__name__}. Tardo {te - ts:.2f} segundos")
        print('')
        return result

    return wrap


@ duracion
def create_directories():
    """
    This function creates the basic directory structure
    for Urbantrips to work
    """

    db_path = os.path.join("data", "db")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("data", "data_ciudad")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("configs")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "tablas")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "png")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("docs")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "pdf")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "matrices")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "data")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "html")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "ppts")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "geojson")
    os.makedirs(db_path, exist_ok=True)


def leer_alias(tipo='data'):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve el alias seteado en el archivo de congifuracion
    """
    configs = leer_configs_generales()
    # Setear el tipo de key en base al tipo de datos
    if tipo == 'data':
        key = 'alias_db_data'
    elif tipo == 'insumos':
        key = 'alias_db_insumos'
    elif tipo == 'dash':
        key = 'alias_db_data'
    else:
        raise ValueError('tipo invalido: %s' % tipo)
    # Leer el alias
    try:
        alias = configs[key] + '_'
    except KeyError:
        alias = ''
    return alias


def traigo_db_path(tipo='data'):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve el path a una base de datos con esa informacion
    """
    if tipo not in ('data', 'insumos', 'dash'):
        raise ValueError('tipo invalido: %s' % tipo)

    alias = leer_alias(tipo)
    db_path = os.path.join("data", "db", f"{alias}{tipo}.sqlite")

    return db_path


def iniciar_conexion_db(tipo='data'):
    """"
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve una conexion sqlite a la db
    """
    db_path = traigo_db_path(tipo)
    conn = sqlite3.connect(db_path, timeout=10)
    return conn


@ duracion
def create_db():
    print("Creando bases de datos")

    # create basic tables
    create_basic_data_model_tables()

    # other inpus tables
    create_other_inputs_tables()

    # stops and routes
    create_stops_and_routes_carto_tables()

    # create services and gps tables
    create_gps_table()

    # create KPI tables
    create_kpi_tables()

    # dashborad tables
    create_dash_tables()

    print("Fin crear base")
    print("Todas las tablas creadas")


def create_other_inputs_tables():
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS distancias
        (h3_o text NOT NULL,
        h3_d text NOT NULL,
        h3_o_norm text NOT NULL,
        h3_d_norm text NOT NULL,
        distance_osm_drive float,
        distance_osm_walk float,
        distance_h3 float
        )

        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS matriz_validacion
        (
        id_linea_agg int,
        parada text,
        area_influencia text
        )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS nuevos_ids_etapas_viajes
        (id INT PRIMARY KEY     NOT NULL,
        nuevo_id_viaje int,
        nuevo_id_etapa int,
        factor_expansion float
        )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata_lineas
            (id_linea INT PRIMARY KEY NOT NULL,
            nombre_linea text not null,
            id_linea_agg INT,
            nombre_linea_agg ING
            modo text,
            empresa text,
            descripcion text
            )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata_ramales
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

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS official_lines_geoms
        (id_linea INT PRIMARY KEY     NOT NULL,
        wkt text not null
        )
        ;
        """
    )
    conn_insumos.close()


def create_dash_tables():
    conn_dash = iniciar_conexion_db(tipo='dash')
    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS matrices
        (
        desc_dia text not null,
        tipo_dia text not null,
        var_zona text not null,
        filtro1 text not null,
        Origen text not null,
        Destino text not null,
        Viajes int not null
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS lineas_deseo
        (
        desc_dia text not null,
        tipo_dia text not null,
        var_zona text not null,
        filtro1 text not null,
        Origen text not null,
        Destino text not null,
        Viajes int not null,
        lon_o float,
        lat_o float,
        lon_d float,
        lat_d float
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS viajes_hora
        (
        desc_dia text not null,
        tipo_dia text not null,
        Hora int,
        Viajes int,
        Modo text
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS distribucion
        (
        desc_dia text not null,
        tipo_dia text not null,
        Distancia int,
        Viajes int,
        Modo text
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS indicadores
        (
        desc_dia text not null,
        tipo_dia text not null,
        Titulo text,
        orden int,
        Indicador text,
        Valor text
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS zonas
        (
        zona text not null,
        tipo_zona text not null,
        wkt text
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS particion_modal
        (
        desc_dia str,
        tipo_dia str,
        tipo str,
        modo str,
        modal float
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS ocupacion_por_linea_tramo
        (id_linea int not null,
        nombre_linea str,
        day_type text nor null,
        n_sections int,
        sentido text not null,
        section_id float not null,
        hora_min int,
        hora_max int,
        cantidad_etapas int not null,
        prop_etapas float not null,
        buff_factor float,
        wkt text
        )
        ;
        """
    )
    conn_dash.close()


def create_stops_and_routes_carto_tables():
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS official_branches_geoms
        (id_ramal INT PRIMARY KEY     NOT NULL,
        wkt text not null
        )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS inferred_lines_geoms
        (id_linea INT PRIMARY KEY     NOT NULL,
        wkt text not null
        )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS lines_geoms
        (id_linea INT PRIMARY KEY     NOT NULL,
        wkt text not null
        )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS branches_geoms
        (id_ramal INT PRIMARY KEY     NOT NULL,
        wkt text not null
        )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS stops
        (id_linea INT NOT NULL,
        id_ramal INT NOT NULL,
        node_id INT NOT NULL,
        branch_stop_order INT NOT NULL,
        stop_x float NOT NULL,
        stop_y float NOT NULL,
        node_x float NOT NULL,
        node_y float NOT NULL
        )
        ;
        """
    )
    conn_insumos.close()


def create_basic_data_model_tables():
    conn_data = iniciar_conexion_db(tipo='data')
    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS transacciones
            (id INT NOT NULL,
            fecha datetime NOT NULL,
            id_original text,
            id_tarjeta text,
            dia text,
            tiempo text,
            hora int,
            modo text,
            id_linea int,
            id_ramal int,
            interno int,
            orden_trx int,
            latitud float,
            longitud float,
            factor_expansion float
            )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS dias_ultima_corrida
            (dia INT NOT NULL)
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS etapas
            (id INT PRIMARY KEY NOT NULL,
            id_tarjeta text,
            dia text,
            id_viaje int,
            id_etapa int,
            tiempo text,
            hora int,
            modo text,
            id_linea int,
            id_ramal int,
            interno int,
            latitud float,
            longitud float,
            h3_o text,
            h3_d text,
            od_validado int,
            factor_expansion_original float,
            factor_expansion_linea float,
            factor_expansion_tarjeta float
            )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS viajes
            (
            id_tarjeta text NOT NULL,
            id_viaje int NOT NULL,
            dia text NOT NULL,
            tiempo text,
            hora int,
            cant_etapas int,
            modo text,
            autobus int,
            tren int,
            metro int,
            tranvia int,
            brt int,
            otros int,
            h3_o text,
            h3_d text,
            od_validado int,
            factor_expansion_linea,
            factor_expansion_tarjeta
            )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS usuarios
            (
            id_tarjeta text NOT NULL,
            dia text NOT NULL,
            od_validado int,
            cant_viajes float,
            factor_expansion_linea,
            factor_expansion_tarjeta
            )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS transacciones_linea
            (
            dia text NOT NULL,
            id_linea int NOT NULL,
            transacciones float
            )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS tarjetas_duplicadas
        (dia text,
        id_tarjeta_original text,
        id_tarjeta_nuevo text
        )

        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS ocupacion_por_linea_tramo
        (id_linea int not null,
        day_type text nor null,
        n_sections int,
        section_meters int,
        sentido text not null,
        section_id float not null,
        x float,
        y float,
        hora_min int,
        hora_max int,
        cantidad_etapas int not null,
        prop_etapas float not null
        )
        ;
        """
    )

    conn_data.close()


def leer_configs_generales():
    """
    Esta funcion lee los configs generales
    """
    path = os.path.join("configs", "configuraciones_generales.yaml")

    try:
        with open(path, 'r', encoding="utf8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        print(f'Error al leer el archivo de configuracion: {error}')
        config = {}

    return config


def crear_tablas_geolocalizacion():
    """Esta funcion crea la tablas en la db para albergar los datos de
    gps y transacciones economicas sin latlong"""

    conn_data = iniciar_conexion_db(tipo='data')

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS trx_eco
                (
                id INT PRIMARY KEY NOT NULL,
                id_original int,
                id_tarjeta text,
                fecha datetime,
                dia text,
                tiempo text,
                hora int,
                modo text,
                id_linea int,
                id_ramal int,
                interno int,
                orden int,
                factor_expansion float
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE INDEX IF NOT EXISTS trx_idx ON trx_eco (
                "id_linea","id_ramal","interno","fecha"
                );
            """
    )

    conn_data.execute(
        """
            CREATE INDEX  IF NOT EXISTS gps_idx ON gps (
                "id_linea","id_ramal","interno","fecha"
                );
        """
    )
    conn_data.close()


def create_gps_table():

    conn_data = iniciar_conexion_db(tipo='data')

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS gps
                (
                id INT PRIMARY KEY NOT NULL,
                id_original int,
                dia text,
                id_linea int,
                id_ramal int,
                interno int,
                fecha datetime,
                latitud FLOAT,
                longitud FLOAT,
                velocity float,
                service_type text,
                distance_km float,
                h3 text
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS services_gps_points
                (
                id INT PRIMARY KEY NOT NULL,
                original_service_id int not null,
                new_service_id int not null,
                service_id int not null,
                id_ramal_gps_point int,
                node_id int
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS services
                (
                id_linea int,
                dia text,
                interno int,
                original_service_id int,
                service_id int,
                total_points int,
                distance_km float,
                min_ts int,
                max_ts int,
                min_datetime text,
                max_datetime text,
                prop_idling float,
                valid int
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS services_stats
                (
                id_linea int,
                dia text,
                cant_servicios_originales int,
                cant_servicios_nuevos int,
                cant_servicios_nuevos_validos int,
                n_servicios_nuevos_cortos int ,
                prop_servicos_cortos_nuevos_idling float,
                distancia_recorrida_original float,
                prop_distancia_recuperada float,
                servicios_originales_sin_dividir float
                )
            ;
            """
    )

    conn_data.commit()
    conn_data.close()


def agrego_indicador(df_indicador,
                     detalle,
                     tabla,
                     nivel=0,
                     var='indicador',
                     var_fex='factor_expansion_linea',
                     aggfunc='sum'):
    '''
    Agrego indicadores de tablas utilizadas
    '''

    df = df_indicador.copy()

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
        print("No existe la tabla indicadores, construyendola...")
        indicadores = pd.DataFrame([])

    if var not in df.columns:
        if not var_fex:
            df[var] = 1
        else:
            df[var] = df[var_fex]

    if var != 'indicador':
        df = df.rename(columns={var: 'indicador'})

    df = df[(df.indicador.notna())].copy()

    if (not var_fex) | (aggfunc == 'sum'):
        resultado = df.groupby('dia', as_index=False).agg(
            {'indicador': aggfunc}).round(2)

    elif aggfunc == 'mean':
        resultado = df.groupby('dia')\
            .apply(lambda x: np.average(x['indicador'], weights=x[var_fex]))\
            .reset_index()\
            .rename(columns={0: 'indicador'})\
            .round(2)
    elif aggfunc == 'median':
        resultado = df.groupby('dia')\
            .apply(
                lambda x: ws.weighted_median(
                    x['indicador'].tolist(),
                    weights=x[var_fex].tolist()))\
            .reset_index()\
            .rename(columns={0: 'indicador'})\
            .round(2)

    resultado['detalle'] = detalle
    resultado = resultado[['dia', 'detalle', 'indicador']]
    resultado['tabla'] = tabla
    resultado['nivel'] = nivel

    if len(indicadores) > 0:
        indicadores = indicadores[~(
            (indicadores.dia.isin(resultado.dia.unique())) &
            (indicadores.detalle == detalle) &
            (indicadores.tabla == tabla)
        )]

    indicadores = pd.concat([indicadores,
                            resultado],
                            ignore_index=True)
    if nivel > 0:
        for i in indicadores[(indicadores.tabla == tabla) &
                             (indicadores.nivel == nivel)].dia.unique():
            for x in indicadores.loc[(indicadores.tabla == tabla) &
                                     (indicadores.nivel == nivel) &
                                     (indicadores.dia == i), 'detalle']:
                valores = round(
                    indicadores.loc[(indicadores.tabla == tabla) &
                                    (indicadores.nivel == nivel) &
                                    (indicadores.dia == i) &
                                    (indicadores.detalle == x),
                                    'indicador'].values[0] / indicadores.loc[
                        (indicadores.tabla == tabla) &
                                        (indicadores.nivel == nivel-1) &
                                        (indicadores.dia == i),
                        'indicador'].values[0] * 100, 1)
                indicadores.loc[(indicadores.tabla == tabla) &
                                (indicadores.nivel == nivel) &
                                (indicadores.dia == i) &
                                (indicadores.detalle == x),
                                'porcentaje'] = valores

    indicadores.fillna(0, inplace=True)

    indicadores.to_sql("indicadores", conn_data,
                       if_exists="replace", index=False)
    conn_data.close()


@ duracion
def eliminar_tarjetas_trx_unica(trx):
    """
    Esta funcion toma el DF de trx y elimina las trx de una tarjeta con
    una unica trx en el dia
    """

    tarjetas_dia_multiples = (
        trx.reindex(columns=["id_tarjeta", "dia"])
        .groupby(["dia", "id_tarjeta"], as_index=False)
        .size()
        .query("size > 1")
    )

    pre = len(trx)
    trx = trx.merge(tarjetas_dia_multiples,
                    on=['dia', 'id_tarjeta'],
                    how='inner').drop('size', axis=1)
    post = len(trx)
    print(pre - post, "casos elminados por trx unicas en el dia")
    return trx


def create_kpi_tables():
    """
    Creates KPI tables in the data db
    """

    conn_data = iniciar_conexion_db(tipo='data')
    conn_dash = iniciar_conexion_db(tipo='dash')

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS kpi_by_day_line
                (
                id_linea int not null,
                dia text not null,
                tot_veh int,
                tot_km float,
                tot_pax foat,
                dmt_mean foat,
                dmt_median float,
                pvd float,
                kvd float,
                ipk float,
                fo_mean float,
                fo_median float
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS kpi_by_day_line_service
                (
                id_linea int not null,
                dia text not null,
                interno text not null,
                service_id int not null,
                hora_inicio float,
                hora_fin float,
                tot_km float,
                tot_pax float,
                dmt_mean float,
                dmt_median float,
                ipk float,
                fo_mean float,
                fo_median float
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS services_by_line_hour
                (
                id_linea int not null,
                dia text not null,
                hora int  not null,
                servicios float  not null
                )
            ;
            """
    )
    conn_dash.execute(
        """
            CREATE TABLE IF NOT EXISTS services_by_line_hour
                (
                id_linea int not null,
                dia text not null,
                hora int  not null,
                servicios float  not null
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS basic_kpi_by_vehicle_hr
                (
                dia text not null,
                id_linea int not null,
                interno int not null,
                hora int  not null,
                tot_pax float,
                eq_pax float,
                dmt float,
                of float,
                speed_kmh float
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS basic_kpi_by_line_hr
                (
                dia text not null,
                id_linea int not null,
                hora int  not null,
                veh float,
                pax float,
                dmt float,
                of float,
                speed_kmh float
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS basic_kpi_by_line_day
                (
                dia text not null,
                id_linea int not null,
                veh float,
                pax float,
                dmt float,
                of float,
                speed_kmh float
                )
            ;
            """
    )

    conn_dash.execute(
        """
            CREATE TABLE IF NOT EXISTS basic_kpi_by_line_hr
                (
                dia text not null,
                id_linea int not null,
                nombre_linea text,
                hora int  not null,
                veh float,
                pax float,
                dmt float,
                of float,
                speed_kmh float
                )
            ;
            """
    )

    conn_data.close()
    conn_dash.close()


def check_table_in_db(table_name, tipo_db):
    """
    Checks if a tbale exists in a db

    Parameters
    ----------
    table_name : str
        Name of table to check for
    tipo_db : str
        db where to check. Must be data or insumos

    Returns
    -------
    bool
        if that table exists in that db
    """
    conn = iniciar_conexion_db(tipo=tipo_db)
    cur = conn.cursor()

    q = f"""
        SELECT tbl_name FROM sqlite_master
        WHERE type='table'
        AND tbl_name='{table_name}';
    """
    listOfTables = cur.execute(q).fetchall()

    if listOfTables == []:
        print(f"No existe la tabla {table_name} en la base")
        return False
    else:
        return True
