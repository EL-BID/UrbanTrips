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


def duracion(f):
    @ wraps(f)
    def wrap(*args, **kw):
        # print(f"{f.__name__} [{args}, {kw}] ", end="", flush=True)
        print(f"{f.__name__} ", end="", flush=True)
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f" Finalizado. Tardo {te - ts:.2f} segundos")
        return result

    return wrap


def create_directories():
    """
    This function creates the basic directory structure
    for Urbantrips to work
    """
    db_path = os.path.join("data", "db")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "tablas")
    os.makedirs(db_path, exist_ok=True)

    db_path = os.path.join("resultados", "png")
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
    # Crear conexion con bases de data e insumos
    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    conn_dash = iniciar_conexion_db(tipo='dash')

    print("Bases abiertas con exito")

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS transacciones
            (id INT PRIMARY KEY     NOT NULL,
            id_original text,
            id_tarjeta text,
            fecha datetime,
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
        CREATE TABLE IF NOT EXISTS etapas
            (id INT PRIMARY KEY     NOT NULL,
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
            h3_o text
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
            od_validado int
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
            cant_viajes float
            )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS factores_expansion
            (
            dia text NOT NULL,
            id_tarjeta text NOT NULL,
            factor_expansion float,
            factor_expansion_original float,
            factor_calibracion float,
            cant_trx int,
            id_tarjeta_valido int
            )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS destinos
        (id INT PRIMARY KEY     NOT NULL,
        h3_d text,
        od_validado int
        )

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
        hora_min int,
        hora_max int,
        cantidad_etapas int not null,
        prop_etapas float not null
        )
        ;
        """
    )

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
        id_linea int,
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
            (id_linea INT PRIMARY KEY     NOT NULL,
            nombre_linea text not null,
            modo text not null,
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
        Viajes text not null
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
        Viajes text not null,
        lon_o float,
        lat_o float,
        lon_d float,
        lat_d float
        )
        ;
        """
    )

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

    conn_data.close()
    conn_insumos.close()

    print("Todas las tablas creadas")


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

    crear_tabla_gps(conn_data)

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


def crear_tabla_gps(conn_data):

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
                cum_distance int,
                service_type text
                )
            ;
            """
    )


def agrego_indicador(df_indicador,
                     detalle,
                     tabla,
                     nivel=0,
                     var='indicador',
                     var_fex='factor_expansion',
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


def crear_tablas_indicadores_operativos():
    """Esta funcion crea la tablas en la db para albergar los datos de
    los indicadores operativos"""

    conn_data = iniciar_conexion_db(tipo='data')

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS indicadores_operativos_linea
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
                fo float
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS indicadores_operativos_interno
                (
                id_linea int not null,
                dia text not null,
                interno text not null,
                kvd float,
                pvd float,
                dmt_mean float,
                dmt_median float,
                ipk float,
                fo float
                )
            ;
            """
    )

    conn_data.close()


def check_config():
    """
    This function takes a configuration file in yaml format
    and read its content. Then check for any inconsistencies
    in the file, printing an error message if one is found.

    Args:
    None

    Returns:
    None

    """
    print("Chequeando archivo de configuracion")
    configs = leer_configs_generales()
    nombre_archivo_trx = configs["nombre_archivo_trx"]

    ruta = os.path.join("data", "data_ciudad", nombre_archivo_trx)
    trx = pd.read_csv(ruta, nrows=1000)

    # chequear que esten los atributos obligatorios
    configs_obligatorios = [
        'geolocalizar_trx', 'resolucion_h3', 'tolerancia_parada_destino',
        'nombre_archivo_trx', 'nombres_variables_trx',
        'imputar_destinos_min_distancia', 'formato_fecha', 'columna_hora',
        'ordenamiento_transacciones', 'lineas_contienen_ramales']

    for param in configs_obligatorios:
        if param not in configs:
            raise KeyError(
                f'Error: El archivo de configuracion no especifica el parámetro {param}')

    # check branch param
    mensaje = "Debe especificarse `lineas_contienen_ramales`"
    assert 'lineas_contienen_ramales' in configs, mensaje
    assert configs['lineas_contienen_ramales'] is not None, mensaje
    assert isinstance(configs['lineas_contienen_ramales'],
                      bool), '`lineas_contienen_ramales` debe ser True o False'

    # Chequear que los parametros tengan valor correcto
    assert isinstance(configs['geolocalizar_trx'],
                      bool), "El parámetro geolocalizar_trx debe ser True o False"

    assert isinstance(configs['columna_hora'],
                      bool), "El parámetro columna_hora debe ser True o False"

    assert isinstance(configs['resolucion_h3'], int) and configs['resolucion_h3'] >= 0 and configs[
        'resolucion_h3'] <= 15, "El parámetro resolucion_h3 debe ser un entero entre 0 y 16"

    assert isinstance(configs['tolerancia_parada_destino'], int) and configs['tolerancia_parada_destino'] >= 0 and configs[
        'tolerancia_parada_destino'] <= 10000, "El parámetro tolerancia_parada_destino debe ser un entero entre 0 y 10000"

    assert not isinstance(configs['nombre_archivo_trx'], type(
        None)), "El parámetro nombre_archivo_trx no puede estar vacío"

    # chequear nombres de variables en archivo trx
    nombres_variables_trx = configs['nombres_variables_trx']
    assert isinstance(nombres_variables_trx,
                      dict), "El parámetro nombres_variables_trx debe especificarse como un diccionario"

    nombres_variables_trx = pd.DataFrame(
        {'trx_name': nombres_variables_trx.keys(), 'csv_name': nombres_variables_trx.values()})

    nombres_variables_trx_s = nombres_variables_trx.csv_name.dropna()
    nombres_var_config_en_trx = nombres_variables_trx_s.isin(trx.columns)

    if not nombres_var_config_en_trx.all():
        raise KeyError('Algunos nombres de atributos especificados en el archivo de configuración no están en el archivo csv de transacciones: ' +
                       ','.join(nombres_variables_trx_s[~nombres_var_config_en_trx]))

    # check mandatory attributes for trx
    atributos_trx_obligatorios = pd.Series(
        ['fecha_trx', 'id_tarjeta_trx', 'id_linea_trx'])

    if not configs['geolocalizar_trx']:
        trx_coords = pd.Series(['latitud_trx', 'longitud_trx'])
        atributos_trx_obligatorios = pd.concat(
            [atributos_trx_obligatorios, trx_coords])
    else:
        # if geocoding vehicle id but be present
        interno_col = pd.Series(['interno_trx'])
        atributos_trx_obligatorios = pd.concat(
            [atributos_trx_obligatorios, interno_col])

    if configs['lineas_contienen_ramales']:
        # if branches branch id must be present
        ramal_trx = pd.Series(['id_ramal_trx'])
        atributos_trx_obligatorios = pd.concat(
            [atributos_trx_obligatorios, ramal_trx])

    attr_obligatorios_en_csv = atributos_trx_obligatorios.isin(
        nombres_variables_trx.dropna().trx_name)

    assert attr_obligatorios_en_csv.all(), "Algunos atributos obligatorios no tienen un atributo correspondiente en el csv de transacionnes: " + \
        ','.join(atributos_trx_obligatorios[~attr_obligatorios_en_csv])

    # check date
    columns_with_date = configs['nombres_variables_trx']['fecha_trx']
    date_format = configs['formato_fecha']
    check_config_fecha(
        df=trx, columns_with_date=columns_with_date, date_format=date_format)

    # check consistency in params

    if configs['ordenamiento_transacciones'] == 'fecha_completa':

        assert isinstance(configs['ventana_viajes'], int) and configs['ventana_viajes'] >= 1 and configs[
            'ventana_viajes'] <= 1000, "Cuando el parametro ordenamiento_transacciones es 'fecha_completa', el parámetro 'ventana_viajes' debe ser un entero mayor a 0"

        assert isinstance(configs['ventana_duplicado'],
                          int) and configs['ventana_duplicado'] >= 1, "Cuando el parametro ordenamiento_transacciones es 'fecha_completa', el parámetro 'ventana_duplicado' debe ser un entero mayor a 0"

    # check consistency in geocoding

    if configs['geolocalizar_trx']:
        mensaje = "Si geolocalizar_trx = True entonces se debe especificar un archivo con informacion gps" + \
            " con los parámetros `nombre_archivo_gps` y `nombres_variables_gps`"
        assert 'nombre_archivo_gps' in configs, mensaje
        assert configs['nombre_archivo_gps'] is not None, mensaje

        assert 'nombres_variables_gps' in configs, mensaje
        nombres_variables_gps = configs['nombres_variables_gps']

        assert isinstance(nombres_variables_gps,
                          dict), "El parámetro nombres_variables_gps debe especificarse como un diccionario"

        ruta = os.path.join("data", "data_ciudad",
                            configs['nombre_archivo_gps'])
        gps = pd.read_csv(ruta, nrows=1000)

        nombres_variables_gps = pd.DataFrame(
            {
                'trx_name': nombres_variables_gps.keys(),
                'csv_name': nombres_variables_gps.values()
            }
        )

        nombres_variables_gps_s = nombres_variables_gps.csv_name.dropna()
        nombres_var_config_en_gps = nombres_variables_gps_s.isin(gps.columns)

        if not nombres_var_config_en_gps.all():
            raise KeyError('Algunos nombres de atributos especificados en el archivo de configuración no están en el archivo de transacciones',
                           nombres_variables_gps_s[~nombres_var_config_en_gps])

        # chequear que todos los atributos obligatorios de trx
        # tengan un atributo en el csv
        atributos_gps_obligatorios = pd.Series(
            ['id_linea_gps',
             'interno_gps',
             'fecha_gps',
             'latitud_gps',
             'longitud_gps'])

        if configs['lineas_contienen_ramales']:
            ramal_gps = pd.Series(['id_ramal_gps'])
            atributos_gps_obligatorios = pd.concat(
                [atributos_gps_obligatorios, ramal_gps])

        attr_obligatorios_en_csv = atributos_gps_obligatorios.isin(
            nombres_variables_gps.trx_name)

        assert attr_obligatorios_en_csv.all(), "Algunos atributos obligatorios no tienen un atributo correspondiente en el csv de transacionnes" + \
            ','.join(atributos_gps_obligatorios[~attr_obligatorios_en_csv])

        # chequear validez de fecha
        columns_with_date = configs['nombres_variables_gps']['fecha_gps']
        check_config_fecha(
            df=gps,
            columns_with_date=columns_with_date, date_format=date_format)

    # Checkear que existan los archivos de zonficación especificados config
    assert 'zonificaciones' in configs, "Debe haber un atributo " +\
        "`zonificaciones` en config aunque este vacío"

    if configs['zonificaciones']:
        for i in configs['zonificaciones']:
            if 'geo' in i:
                geo_file = os.path.join(
                    "data", "data_ciudad", configs['zonificaciones'][i])
                assert os.path.exists(
                    geo_file), f"File {geo_file} does not exist"

    # check epsg in meters
    assert 'epsg_m' in configs, "Debe haber un atributo `epsg_m` en config " +\
        "especificando un id de EPSG para una proyeccion en metros"

    assert isinstance(
        configs['epsg_m'], int), "Debe haber un id de EPSG en metros en" +\
        " configs['epsg_m'] "

    print("Proceso de chequeo de archivo de configuración concluido con éxito")
    return None


def check_config_fecha(df, columns_with_date, date_format):
    """
    Esta funcion toma un dataframe, una columna donde se guardan fechas,
    un formato de fecha, intenta parsear las fechas y arroja un error
    si mas del 80% de las fechas no pueden parsearse
    """
    fechas = pd.to_datetime(
        df[columns_with_date], format=date_format, errors="coerce"
    )

    # Chequear si el formato funciona
    checkeo = fechas.isna().sum() / len(df)
    string = "Corrija el formato de fecha en config. Actualmente se pierden" +\
        f"{round((checkeo * 100),2)} por ciento de registros"
    assert checkeo < 0.8, string
