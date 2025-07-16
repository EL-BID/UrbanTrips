import pandas as pd
import geopandas as gpd
import sqlite3
import os
import yaml
import time
from functools import wraps
import re
import numpy as np
import weightedstats as ws
from pandas.io.sql import DatabaseError
import datetime
from shapely import wkt


def duracion(f):
    @wraps(f)
    def wrap(*args, **kw):
        print("")
        print(
            f"{f.__name__} ({str(datetime.datetime.now())[:19]})\n", end="", flush=True
        )
        print("-" * (len(f.__name__) + 22))

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"Finalizado {f.__name__}. Tardo {te - ts:.2f} segundos")
        print("")
        return result

    return wrap


@duracion
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

    db_path = os.path.join("resultados", "geojson")
    os.makedirs(db_path, exist_ok=True)


def leer_alias(tipo="data"):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve el alias seteado en el archivo de congifuracion
    """
    configs = leer_configs_generales()
    # Setear el tipo de key en base al tipo de datos
    if tipo == "data":
        key = "alias_db_data"
    elif tipo == "insumos":
        key = "alias_db_insumos"
    elif tipo == "dash":
        key = "alias_db_dashboard"
    else:
        raise ValueError("tipo invalido: %s" % tipo)
    # Leer el alias
    try:
        alias = configs[key] + "_"
    except KeyError:
        alias = ""
    return alias


def traigo_db_path(tipo="data", alias_db=""):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve el path a una base de datos con esa informacion
    """
    if tipo not in ("data", "insumos", "dash", "general"):
        raise ValueError("tipo invalido: %s" % tipo)
    if len(alias_db) == 0:
        alias_db = leer_alias(tipo)
    if not alias_db.endswith("_"):
        alias_db += "_"
    db_path = os.path.join("data", "db", f"{alias_db}{tipo}.sqlite")

    return db_path


def iniciar_conexion_db(tipo="data", alias_db=""):
    """ "
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve una conexion sqlite a la db
    """
    if len(alias_db) == 0:
        alias_db = leer_alias(tipo)
    if not alias_db.endswith("_"):
        alias_db += "_"
    db_path = traigo_db_path(tipo, alias_db)
    print("DB_PATH:", db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    return conn


@duracion
def create_insumos_general_dbs():
    print("Creando bases para insumos")
    configs_usuario = leer_configs_generales(autogenerado=False)
    alias_db = configs_usuario.get("alias_db", "")

    # Recorridos y paradas
    create_stops_and_routes_carto_tables(alias_db)

    # Otros insumos
    create_other_inputs_tables(alias_db)

    # Crear una tabla general para todas las corridas
    create_general_db(alias_db)


def create_general_db(alias_db):
    """
    Crea la base de datos general para UrbanTrips
    """
    conn_general = iniciar_conexion_db(tipo="general", alias_db=alias_db)
    conn_general.execute(
        """
        CREATE TABLE IF NOT EXISTS corridas
        (corrida text PRIMARY KEY NOT NULL,
         process text NOT NULL,
         date text NOT NULL
        )
        ;
        """
    )
    conn_general.close()


@duracion
def create_data_dash_dbs(alias_db):
    print("Creando bases para data para {alias_db}".format(alias_db=alias_db))

    # create basic tables
    create_basic_data_model_tables(alias_db)

    # create services and gps tables
    create_gps_table(alias_db)

    # create KPI tables
    create_kpi_tables(alias_db)

    # dashborad tables
    create_dash_tables(alias_db)

    print("Fin crear base")
    print("Todas las tablas creadas")


def create_other_inputs_tables(alias_db):

    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_db)

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

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS zonificaciones
        (zona text NOT NULL,
         id text NOT NULL,
         orden int,
         wkt text
        )
        ;
        """
    )
    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS poligonos
        (id text NOT NULL,
         wkt text
        )
        ;
        """
    )

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS travel_times_stations
        (id_o int NOT NULL,
         id_linea_o int NOT NULL,
         id_ramal_o int,
         lat_o float NOT NULL,
         lon_o float NOT NULL,
         id_d int NOT NULL,
         lat_d float NOT NULL,
         lon_d float NOT NULL,
         id_linea_d int NOT NULL,
         id_ramal_d int,
         travel_time_min float NOT NULL
        )
        ;
        """
    )

    conn_insumos.close()


def create_dash_tables(alias_db):
    conn_dash = iniciar_conexion_db(tipo="dash", alias_db=alias_db)

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
        yr_mo text,
        nombre_linea str,
        day_type text not null,
        n_sections int,
        section_meters int,
        sentido text not null,
        section_id int not null,
        hour_min int,
        hour_max int,
        legs int not null,
        prop float not null,
        buff_factor float,
        wkt text
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS lines_od_matrix_by_section
        (id_linea int not null,
        yr_mo text,
        day_type text nor null,
        n_sections int,
        hour_min int,
        hour_max int,
        Origen int not null,
        Destino int not null,
        legs int not null,
        prop float not null,
        nombre_linea text
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS matrices_linea_carto
        (id_linea INT NOT NULL,
        n_sections INT NOT NULL,
        section_id INT NOT NULL,
        wkt text,
        x float,
        y float,
        nombre_linea text
        )
        ;
        """
    )

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS matrices_linea
        (id_linea INT NOT NULL,
        yr_mo text,
        day_type text not null,
        n_sections INT NOT NULL,
        hour_min int,
        hour_max int,
        section_id INT,
        Origen int ,
        Destino int ,
        legs int,
        prop float,
        nombre_linea text
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

    conn_dash.execute(
        """
            CREATE TABLE IF NOT EXISTS basic_kpi_by_line_hr
                (
                dia text not null,
                yr_mo text,
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

    conn_dash.execute(
        """
        CREATE TABLE IF NOT EXISTS supply_stats_by_section_id
        (id_linea int not null,
        yr_mo text,
        nombre_linea str,
        day_type text not null,
        n_sections int,
        section_meters int,
        sentido text not null,
        section_id int not null,
        hour_min int,
        hour_max int,
        n_vehicles int,
        avg_speed float,
        median_speed float,
        speed_interval float,
        frequency float,
        frequency_interval text,
        buff_factor float,
        wkt text
        )
        ;
        """
    )

    conn_dash.close()


def create_stops_and_routes_carto_tables(alias_db):

    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_db)

    # Crear tablas de insumos para paradas y rutas
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

    conn_insumos.execute(
        """
        CREATE TABLE IF NOT EXISTS routes_section_id_coords
        (id_linea INT NOT NULL,
        n_sections INT NOT NULL,
        section_id INT NOT NULL,
        section_lrs float NOT NULL,
        x float NOT NULL,
        y float NOT NULL
        )
        ;
        """
    )

    conn_insumos.close()


def create_basic_data_model_tables(alias_db):
    conn_data = iniciar_conexion_db(tipo="data", alias_db=alias_db)
    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS transacciones
            (id INT NOT NULL,
            fecha int NOT NULL,
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
            genero text,
            tarifa text,
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
            genero text,
            tarifa text,
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
            cable int,
            lancha int,
            otros int,
            h3_o text,
            h3_d text,
            genero text,
            tarifa text,
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
        yr_mo text,
        day_type text nor null,
        n_sections int,
        section_meters int,
        sentido text not null,
        section_id int not null,
        hour_min int,
        hour_max int,
        legs int not null,
        prop float not null
        )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS legs_to_gps_origin
        (
        dia text,
        id_legs int not null,
        id_gps int not null
        )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS legs_to_gps_destination
        (
        dia text,
        id_legs int not null,
        id_gps int not null
        )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS legs_to_station_origin
        (
        dia text,
        id_legs int not null,
        id_station int not null
        )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS legs_to_station_destination
        (
        dia text,
        id_legs int not null,
        id_station int not null
        )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS travel_times_gps
        (
        dia text,
        id int not null,
        travel_time_min float,
        travel_speed float
        )
        ;
        """
    )
    conn_data.execute(
        """
        CREATE INDEX  IF NOT EXISTS travel_times_gps_idx ON travel_times_gps (
        "id"
        );
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS travel_times_stations
        (
        dia text,
        id int not null,
        travel_time_min float,
        travel_speed float
        )
        ;
        """
    )
    conn_data.execute(
        """
        CREATE INDEX  IF NOT EXISTS travel_times_ts_idx ON travel_times_stations (
        "id"
        );
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS travel_times_legs
        (
        dia text,
        id int not null,
        id_etapa int,
        id_viaje int,
        id_tarjeta text,
        travel_time_min float
        )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS travel_times_trips
        (
        dia text,
        id_tarjeta text,
        id_viaje int,
        travel_time_min float
        )
        ;
        """
    )

    conn_data.close()


def leer_configs_generales(autogenerado=True):
    """
    Esta funcion lee los configs generales
    """
    if autogenerado:
        path = os.path.join("configs", "configuraciones_generales_autogenerado.yaml")
    else:
        path = os.path.join("configs", "configuraciones_generales.yaml")
    try:
        with open(path, "r", encoding="utf8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        print(f"Error al leer el archivo de configuracion: {error}")
        config = {}

    return config


def crear_tablas_geolocalizacion():
    """Esta funcion crea la tablas en la db para albergar los datos de
    gps y transacciones economicas sin latlong"""

    conn_data = iniciar_conexion_db(tipo="data")

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS trx_eco
                (
                id INT PRIMARY KEY NOT NULL,
                id_original int,
                id_tarjeta text,
                fecha int,
                dia text,
                tiempo text,
                hora int,
                modo text,
                id_linea int,
                id_ramal int,
                interno int,
                orden int,
                genero text,
                tarifa text,
                factor_expansion float
                )
            ;
            """
    )

    conn_data.execute(
        """
            CREATE INDEX IF NOT EXISTS trx_idx_r ON trx_eco (
                "dia","id_linea","id_ramal","interno","fecha"
                );
            """
    )

    conn_data.execute(
        """
            CREATE INDEX  IF NOT EXISTS gps_idx_r ON gps (
                "dia","id_linea","id_ramal","interno","fecha"
                );
        """
    )

    conn_data.execute(
        """
            CREATE INDEX IF NOT EXISTS trx_idx_l ON trx_eco (
                "dia","id_linea","interno","fecha"
                );
            """
    )

    conn_data.execute(
        """
            CREATE INDEX  IF NOT EXISTS gps_idx_l ON gps (
                "dia","id_linea","interno","fecha"
                );
        """
    )
    conn_data.close()


def create_gps_table(alias_db):

    conn_data = iniciar_conexion_db(tipo="data", alias_db=alias_db)

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
                fecha int,
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
                id_linea int not null,
                id_ramal int,
                interno int,
                dia text,
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
                id_ramal int,
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
                id_ramal int,
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

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS vehicle_expansion_factors
                (
             	id_linea int,
                dia text,
                unique_vehicles int,
                broken_gps_veh int,
                veh_exp float
                )
            ;
            """
    )

    conn_data.commit()
    conn_data.close()


def agrego_indicador(
    df_indicador,
    detalle,
    tabla,
    nivel=0,
    var="indicador",
    var_fex="factor_expansion_linea",
    aggfunc="sum",
):
    """
    Agrego indicadores de tablas utilizadas
    """

    df = df_indicador.copy()

    conn_data = iniciar_conexion_db(tipo="data")

    try:
        indicadores = pd.read_sql_query(
            """
            SELECT *
            FROM indicadores
            """,
            conn_data,
        )
    except DatabaseError as e:
        print("No existe la tabla indicadores, construyendola...", e)
        indicadores = pd.DataFrame([])

    if var not in df.columns:
        if not var_fex:
            df[var] = 1
        else:
            df[var] = df[var_fex]

    if var != "indicador":
        df = df.rename(columns={var: "indicador"})

    df = df[(df.indicador.notna())].copy()

    if (not var_fex) | (aggfunc == "sum"):
        resultado = (
            df.groupby("dia", as_index=False).agg({"indicador": aggfunc}).round(2)
        )

    elif aggfunc == "mean":
        resultado = (
            df.groupby("dia")
            .apply(lambda x: np.average(x["indicador"], weights=x[var_fex]))
            .reset_index()
            .rename(columns={0: "indicador"})
            .round(2)
        )

    elif aggfunc == "median":
        resultado = (
            df.groupby("dia")
            .apply(
                lambda x: ws.weighted_median(
                    x["indicador"].tolist(), weights=x[var_fex].tolist()
                )
            )
            .reset_index()
            .rename(columns={0: "indicador"})
            .round(2)
        )

    resultado["detalle"] = detalle
    resultado = resultado[["dia", "detalle", "indicador"]]
    resultado["tabla"] = tabla
    resultado["nivel"] = nivel

    if len(indicadores) > 0:
        indicadores = indicadores[
            ~(
                (indicadores.dia.isin(resultado.dia.unique()))
                & (indicadores.detalle == detalle)
                & (indicadores.tabla == tabla)
            )
        ]

    indicadores = pd.concat([indicadores, resultado], ignore_index=True)
    if nivel > 0:
        for i in indicadores[
            (indicadores.tabla == tabla) & (indicadores.nivel == nivel)
        ].dia.unique():
            for x in indicadores.loc[
                (indicadores.tabla == tabla)
                & (indicadores.nivel == nivel)
                & (indicadores.dia == i),
                "detalle",
            ]:
                valores = round(
                    indicadores.loc[
                        (indicadores.tabla == tabla)
                        & (indicadores.nivel == nivel)
                        & (indicadores.dia == i)
                        & (indicadores.detalle == x),
                        "indicador",
                    ].values[0]
                    / indicadores.loc[
                        (indicadores.tabla == tabla)
                        & (indicadores.nivel == nivel - 1)
                        & (indicadores.dia == i),
                        "indicador",
                    ].values[0]
                    * 100,
                    1,
                )
                indicadores.loc[
                    (indicadores.tabla == tabla)
                    & (indicadores.nivel == nivel)
                    & (indicadores.dia == i)
                    & (indicadores.detalle == x),
                    "porcentaje",
                ] = valores

    indicadores.fillna(0, inplace=True)

    indicadores.to_sql("indicadores", conn_data, if_exists="replace", index=False)
    conn_data.close()


@duracion
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
    trx = trx.merge(tarjetas_dia_multiples, on=["dia", "id_tarjeta"], how="inner").drop(
        "size", axis=1
    )
    post = len(trx)
    print(pre - post, "casos elminados por trx unicas en el dia")
    return trx


def create_kpi_tables(alias_db):
    """
    Creates KPI tables in the data db
    """

    conn_data = iniciar_conexion_db(tipo="data", alias_db=alias_db)

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
                id_ramal int,
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

    conn_data.execute(
        """
            CREATE TABLE IF NOT EXISTS basic_kpi_by_vehicle_hr
                (
                dia text not null,
                id_linea int not null,
                id_ramal int,
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
                yr_mo text,
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
                yr_mo text,
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

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS lines_od_matrix_by_section
        (id_linea int not null,
        yr_mo text,
        day_type text nor null,
        n_sections int,
        hour_min int,
        hour_max int,
        section_id_o int not null,
        section_id_d int not null,
        legs int not null,
        prop float not null
        )
        ;
        """
    )

    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS overlapping_by_route
        (
        dia text not null,
        base_line_id int not null,
        base_branch_id int,
        comp_line_id int not null,
        comp_branch_id int,
        res_h3 int,
        overlap float,
        type_overlap text
        )
        ;
        """
    )
    conn_data.execute(
        """
        CREATE TABLE IF NOT EXISTS supply_stats_by_section_id
        (
        id_linea int not null,
        yr_mo text not null,
        day_type text not null,
        n_sections int not null,
        section_meters int,
        sentido text not null,
        section_id int not null,
        hour_min int,
        hour_max int,
        n_vehicles int,
        avg_speed float,
        median_speed float,
        speed_interval float,
        frequency float,
        frequency_interval text
        )
        ;
        """
    )

    conn_data.close()


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


def is_date_string(input_str):
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if pattern.match(input_str):
        return True
    else:
        return False


def check_date_type(day_type):
    """Checks if a day_type param is formated in the right way"""
    day_type_is_a_date = is_date_string(day_type)

    # check day type format
    day_type_format_ok = (day_type in ["weekday", "weekend"]) or day_type_is_a_date

    if not day_type_format_ok:
        raise Exception("dat_type debe ser `weekday`, `weekend` o fecha 'YYYY-MM-DD'")


def create_line_ids_sql_filter(line_ids):
    """
    Takes a set of line ids and returns a where clause
    to filter in sqlite
    """
    if line_ids is not None:
        if isinstance(line_ids, int):
            line_ids = [line_ids]
        lines_str = ",".join(map(str, line_ids))
        line_ids_where = f" where id_linea in ({lines_str})"

    else:
        lines_str = ""
        line_ids_where = " where id_linea is not NULL"
    return line_ids_where


def create_branch_ids_sql_filter(branch_ids):
    """
    Takes a set of branch ids and returns a where clause
    to filter in sqlite
    """
    if branch_ids is not None:
        if isinstance(branch_ids, int):
            branch_ids = [branch_ids]
        branches_str = ",".join(map(str, branch_ids))
        branch_ids_where = f" where id_ramal in ({branches_str})"

    else:
        branches_str = ""
        branch_ids_where = " where id_ramal is not NULL"
    return branch_ids_where


def traigo_tabla_zonas():

    zonas = levanto_tabla_sql("equivalencias_zonas", "insumos")
    zonas_cols = []
    if len(zonas) > 0:
        zonas_cols = [
            i for i in zonas.columns if i not in ["h3", "latitud", "longitud"]
        ]

    return zonas, zonas_cols


def normalize_vars(tabla):
    if "day_type" in tabla.columns:
        tabla.loc[tabla.day_type == "weekday", "day_type"] = "Día hábil"
        tabla.loc[tabla.day_type == "weekend", "day_type"] = "Fin de semana"

    if "nombre_linea" in tabla.columns:
        tabla["nombre_linea"] = tabla["nombre_linea"].str.replace(" -", "")
    if "Modo" in tabla.columns:
        tabla["Modo"] = tabla["Modo"].str.capitalize()
    if "modo" in tabla.columns:
        tabla["modo"] = tabla["modo"].str.capitalize()
    return tabla


def levanto_tabla_sql(tabla_sql, tabla_tipo="dash", query="", alias_db=""):

    if alias_db and not alias_db.endswith("_"):
        alias_db += "_"

    conn = iniciar_conexion_db(tipo=tabla_tipo, alias_db=alias_db)

    try:
        if len(query) == 0:
            query = f"SELECT * FROM {tabla_sql}"
        tabla = pd.read_sql_query(query, conn)
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        if "no such table" in str(e):
            print(f"La tabla '{tabla_sql}' no existe.")
            tabla = pd.DataFrame([])
        else:
            raise

    conn.close()

    if "wkt" in tabla.columns and not tabla.empty:
        tabla["geometry"] = tabla.wkt.apply(wkt.loads)
        tabla = gpd.GeoDataFrame(tabla, crs=4326)
        tabla = tabla.drop(["wkt"], axis=1)

    tabla = normalize_vars(tabla)

    return tabla


def calculate_weighted_means(
    df_,
    aggregate_cols,
    weighted_mean_cols,
    weight_col,
    zero_to_nan=[],
    var_fex_summed=True,
):
    df = df_.copy()
    for i in zero_to_nan:
        df.loc[df[i] == 0, i] = np.nan

    # calculate_weighted_means  # Validate inputs
    if not set(aggregate_cols + weighted_mean_cols + [weight_col]).issubset(df.columns):
        raise ValueError("One or more columns specified do not exist in the DataFrame.")
    result = pd.DataFrame([])
    # Calculate the product of the value and its weight for weighted mean calculation
    for col in weighted_mean_cols:
        df.loc[df[col].notna(), f"{col}_weighted"] = (
            df.loc[df[col].notna(), col] * df.loc[df[col].notna(), weight_col]
        )
        grouped = (
            df.loc[df[col].notna()]
            .groupby(aggregate_cols, as_index=False)[[f"{col}_weighted", weight_col]]
            .sum()
        )
        grouped[col] = grouped[f"{col}_weighted"] / grouped[weight_col]
        grouped = grouped.drop([f"{col}_weighted", weight_col], axis=1)

        if len(result) == 0:
            result = grouped.copy()
        else:
            result = result.merge(grouped, how="left", on=aggregate_cols)

    if var_fex_summed:
        fex_summed = df.groupby(aggregate_cols, as_index=False)[weight_col].sum()
        result = result.merge(fex_summed, how="left", on=aggregate_cols)
    else:
        fex_mean = df.groupby(aggregate_cols, as_index=False)[weight_col].mean()
        result = result.merge(fex_mean, how="left", on=aggregate_cols)

    return result


def delete_data_from_table_run_days(table_name):

    conn_data = iniciar_conexion_db(tipo="data")

    dias_ultima_corrida = pd.read_sql_query(
        """
                                    SELECT *
                                    FROM dias_ultima_corrida
                                    """,
        conn_data,
    )
    # delete data from same day if exists
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM {table_name} WHERE dia IN ({values})"
    conn_data.execute(query)
    conn_data.commit()
    conn_data.close()


def tabla_existe(conn, table_name):
    try:
        conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        return True
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return False
        else:
            raise


def guardar_tabla_sql(df, table_name, tabla_tipo="dash", filtros=None):
    """
    Guarda un DataFrame en una base de datos SQLite.

    Parámetros:
    df (pd.DataFrame): DataFrame que se desea guardar.
    conn (sqlite3.Connection): Conexión a la base de datos SQLite.
    table_name (str): Nombre de la tabla en la base de datos.
    filtros (dict, optional): Diccionario de filtros para eliminar registros. Las claves son los nombres
                              de los campos y los valores pueden ser un valor único o una lista de valores.
    """
    # Verifica si la tabla existe en la base de datos

    conn = iniciar_conexion_db(tipo=tabla_tipo)
    cursor = conn.cursor()
    table_exists = tabla_existe(conn, table_name)

    # Si la tabla existe y se han proporcionado filtros, elimina los registros que coincidan
    if table_exists and filtros:
        condiciones = []
        valores = []

        # Construir las condiciones y los valores para cada filtro
        for campo, valor in filtros.items():
            if isinstance(valor, list):
                # Si el valor es una lista, usamos la cláusula IN
                condiciones.append(f"{campo} IN ({','.join(['?'] * len(valor))})")
                valores.extend(valor)
            else:
                # Si el valor es único, usamos una condición simple
                condiciones.append(f"{campo} = ?")
                valores.append(valor)

        # Ejecutar la eliminación con las condiciones construidas
        where_clause = " AND ".join(condiciones)
        cursor.execute(f"DELETE FROM {table_name} WHERE {where_clause}", valores)
        conn.commit()

    # Guarda el DataFrame en la base de datos, crea la tabla si no existe
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()
    print(f"Datos guardados exitosamente {table_name}.")
