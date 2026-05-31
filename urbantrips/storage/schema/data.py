# urbantrips/storage/schema/data.py

TRANSACCIONES = """
CREATE TABLE IF NOT EXISTS transacciones (
    id               INT PRIMARY KEY NOT NULL,
    batch_id         INT,
    fecha            INT NOT NULL,
    id_original      TEXT,
    id_tarjeta       TEXT,
    dia              TEXT,
    tiempo           TEXT,
    hora             INT,
    modo             TEXT,
    id_linea         BIGINT,
    id_ramal         BIGINT,
    interno          INT,
    orden_trx        INT,
    genero           TEXT,
    tarifa           TEXT,
    latitud          FLOAT,
    longitud         FLOAT,
    factor_expansion FLOAT
)
"""

DIAS_ULTIMA_CORRIDA = """
CREATE TABLE IF NOT EXISTS dias_ultima_corrida (
    dia TEXT NOT NULL
)
"""

ETAPAS = """
CREATE TABLE IF NOT EXISTS etapas (
    id                        INT PRIMARY KEY NOT NULL,
    batch_id                  INT,
    id_tarjeta                TEXT,
    dia                       TEXT,
    id_viaje                  INT,
    id_etapa                  INT,
    tiempo                    TEXT,
    hora                      INT,
    modo                      TEXT,
    id_linea                  BIGINT,
    id_ramal                  BIGINT,
    interno                   INT,
    genero                    TEXT,
    tarifa                    TEXT,
    latitud                   FLOAT,
    longitud                  FLOAT,
    h3_o                      TEXT,
    h3_d                      TEXT,
    od_validado               INT,
    etapa_validada            INT,
    factor_expansion_original FLOAT,
    factor_expansion_linea    FLOAT,
    factor_expansion_tarjeta  FLOAT,
    factor_expansion_etapa    FLOAT,
    distancia                 FLOAT,
    travel_time_min           FLOAT
)
"""

VIAJES = """
CREATE TABLE IF NOT EXISTS viajes (
    id_tarjeta               TEXT NOT NULL,
    id_viaje                 INT NOT NULL,
    dia                      TEXT NOT NULL,
    tiempo                   TEXT,
    hora                     INT,
    cant_etapas              INT,
    modo                     TEXT,
    autobus                  INT,
    tren                     INT,
    metro                    INT,
    tranvia                  INT,
    brt                      INT,
    cable                    INT,
    lancha                   INT,
    otros                    INT,
    h3_o                     TEXT,
    h3_d                     TEXT,
    genero                   TEXT,
    tarifa                   TEXT,
    od_validado              INT,
    factor_expansion_linea   FLOAT,
    factor_expansion_tarjeta FLOAT,
    distancia                FLOAT,
    travel_time_min          FLOAT
)
"""

USUARIOS = """
CREATE TABLE IF NOT EXISTS usuarios (
    id_tarjeta               TEXT NOT NULL,
    dia                      TEXT NOT NULL,
    od_validado              INT,
    cant_viajes              FLOAT,
    factor_expansion_linea   FLOAT,
    factor_expansion_tarjeta FLOAT
)
"""

GPS = """
CREATE TABLE IF NOT EXISTS gps (
    id           INT PRIMARY KEY NOT NULL,
    id_original  TEXT,
    dia          TEXT,
    id_linea     BIGINT,
    id_ramal     BIGINT,
    interno      INT,
    fecha        INT,
    latitud      FLOAT,
    longitud     FLOAT,
    velocity     FLOAT,
    service_type TEXT,
    distance_km  FLOAT,
    h3           TEXT
)
"""

VEHICLE_EXPANSION_FACTORS = """
CREATE TABLE IF NOT EXISTS vehicle_expansion_factors (
    id_linea        BIGINT,
    dia             TEXT,
    unique_vehicles INT,
    broken_gps_veh  INT,
    veh_exp         FLOAT
)
"""

LEGS_TO_GPS_ORIGIN = """
CREATE TABLE IF NOT EXISTS legs_to_gps_origin (
    dia     TEXT,
    id_legs INT NOT NULL,
    id_gps  INT NOT NULL
)
"""

LEGS_TO_GPS_DESTINATION = """
CREATE TABLE IF NOT EXISTS legs_to_gps_destination (
    dia     TEXT,
    id_legs INT NOT NULL,
    id_gps  INT NOT NULL
)
"""

LEGS_TO_STATION_ORIGIN = """
CREATE TABLE IF NOT EXISTS legs_to_station_origin (
    dia        TEXT,
    id_legs    INT NOT NULL,
    id_station INT NOT NULL
)
"""

LEGS_TO_STATION_DESTINATION = """
CREATE TABLE IF NOT EXISTS legs_to_station_destination (
    dia        TEXT,
    id_legs    INT NOT NULL,
    id_station INT NOT NULL
)
"""

TRAVEL_TIMES_GPS = """
CREATE TABLE IF NOT EXISTS travel_times_gps (
    dia             TEXT,
    id              INT NOT NULL,
    travel_time_min FLOAT,
    travel_speed    FLOAT
)
"""

TRAVEL_TIMES_STATIONS = """
CREATE TABLE IF NOT EXISTS travel_times_stations (
    dia             TEXT,
    id              INT NOT NULL,
    travel_time_min FLOAT,
    travel_speed    FLOAT
)
"""

TRAVEL_TIMES_LEGS = """
CREATE TABLE IF NOT EXISTS travel_times_legs (
    dia                 TEXT,
    id                  INT NOT NULL,
    id_tarjeta          TEXT,
    id_viaje            INT,
    id_etapa            INT,
    travel_time_min     FLOAT,
    distance_od         FLOAT,
    distance_route      FLOAT,
    distance_route_gps  FLOAT,
    kmh_od              FLOAT,
    kmh_route           FLOAT,
    kmh_route_gps       FLOAT
)
"""

TRAVEL_TIMES_TRIPS = """
CREATE TABLE IF NOT EXISTS travel_times_trips (
    dia                 TEXT,
    id_tarjeta          TEXT,
    id_viaje            INT,
    travel_time_min     FLOAT,
    distance_od         FLOAT,
    distance_route      FLOAT,
    distance_route_gps  FLOAT,
    kmh_od              FLOAT,
    kmh_route           FLOAT,
    kmh_route_gps       FLOAT
)
"""

TRANSACCIONES_LINEA = """
CREATE TABLE IF NOT EXISTS transacciones_linea (
    dia           TEXT NOT NULL,
    id_linea      BIGINT NOT NULL,
    transacciones FLOAT
)
"""

TARJETAS_DUPLICADAS = """
CREATE TABLE IF NOT EXISTS tarjetas_duplicadas (
    dia                 TEXT,
    id_tarjeta_original TEXT,
    id_tarjeta_nuevo    TEXT
)
"""

OCUPACION_POR_LINEA_TRAMO = """
CREATE TABLE IF NOT EXISTS ocupacion_por_linea_tramo (
    id_linea       BIGINT NOT NULL,
    yr_mo          TEXT,
    day_type       TEXT NOT NULL,
    n_sections     INT,
    section_meters INT,
    sentido        TEXT NOT NULL,
    section_id     INT NOT NULL,
    hour_min       INT,
    hour_max       INT,
    legs           INT NOT NULL,
    prop           FLOAT NOT NULL
)
"""

SERVICES_GPS_POINTS = """
CREATE TABLE IF NOT EXISTS services_gps_points (
    id                 INT PRIMARY KEY NOT NULL,
    id_linea           BIGINT NOT NULL,
    id_ramal           BIGINT,
    interno            INT,
    dia                TEXT,
    original_service_id INT NOT NULL,
    new_service_id     INT NOT NULL,
    service_id         INT NOT NULL,
    id_ramal_gps_point BIGINT,
    node_id            INT
)
"""

SERVICES = """
CREATE TABLE IF NOT EXISTS services (
    id_linea            BIGINT,
    id_ramal            BIGINT,
    dia                 TEXT,
    interno             INT,
    original_service_id INT,
    service_id          INT,
    total_points        INT,
    distance_km         FLOAT,
    min_ts              INT,
    max_ts              INT,
    min_datetime        TEXT,
    max_datetime        TEXT,
    prop_idling         FLOAT,
    valid               INT
)
"""

KPI_BY_DAY_LINE = """
CREATE TABLE IF NOT EXISTS kpi_by_day_line (
    id_linea   BIGINT NOT NULL,
    dia        TEXT NOT NULL,
    tot_veh    INT,
    tot_km     FLOAT,
    tot_pax    FLOAT,
    dmt_mean   FLOAT,
    dmt_median FLOAT,
    pvd        FLOAT,
    kvd        FLOAT,
    ipk        FLOAT,
    fo_mean    FLOAT,
    fo_median  FLOAT
)
"""

KPI_BY_DAY_LINE_SERVICE = """
CREATE TABLE IF NOT EXISTS kpi_by_day_line_service (
    id_linea   BIGINT NOT NULL,
    dia        TEXT NOT NULL,
    id_ramal   BIGINT,
    interno    TEXT NOT NULL,
    service_id INT NOT NULL,
    hora_inicio FLOAT,
    hora_fin   FLOAT,
    tot_km     FLOAT,
    tot_pax    FLOAT,
    dmt_mean   FLOAT,
    dmt_median FLOAT,
    ipk        FLOAT,
    fo_mean    FLOAT,
    fo_median  FLOAT
)
"""

SERVICES_STATS = """
CREATE TABLE IF NOT EXISTS services_stats (
    id_linea                          BIGINT,
    id_ramal                          BIGINT,
    dia                               TEXT,
    cant_servicios_originales         INT,
    cant_servicios_nuevos             INT,
    cant_servicios_nuevos_validos     INT,
    n_servicios_nuevos_cortos         INT,
    prop_servicos_cortos_nuevos_idling FLOAT,
    distancia_recorrida_original      FLOAT,
    prop_distancia_recuperada         FLOAT,
    servicios_originales_sin_dividir  FLOAT
)
"""

TRANSACCIONES_RAW = """
CREATE TABLE IF NOT EXISTS transacciones_raw (
    id_original          TEXT,
    id_tarjeta           TEXT,
    dia                  TEXT,
    tiempo               TEXT,
    hora                 INT,
    modo                 TEXT,
    id_linea             BIGINT,
    id_ramal             BIGINT,
    interno              INT,
    orden_trx            INT,
    genero               TEXT,
    tarifa               TEXT,
    latitud              FLOAT,
    longitud             FLOAT,
    fecha_ts             BIGINT,
    factor_expansion_raw FLOAT
)
"""

IDX_TRX_BATCH    = "CREATE INDEX IF NOT EXISTS idx_trx_batch ON transacciones(batch_id)"
IDX_ETAPAS_BATCH = "CREATE INDEX IF NOT EXISTS idx_etapas_batch ON etapas(batch_id)"
IDX_GPS_LINE_DAY = (
    "CREATE INDEX IF NOT EXISTS idx_gps_line_day ON gps(id_linea, dia)"
)
IDX_ETAPAS_DIA_OD_VALIDADO = (
    "CREATE INDEX IF NOT EXISTS idx_etapas_dia_od_validado "
    "ON etapas(dia, od_validado)"
)
IDX_ETAPAS_DIA_LINE_RAMAL_INTERNO = (
    "CREATE INDEX IF NOT EXISTS idx_etapas_dia_line_ramal_interno "
    "ON etapas(dia, id_linea, id_ramal, interno)"
)
IDX_GPS_DIA_LINE_RAMAL_INTERNO_FECHA = (
    "CREATE INDEX IF NOT EXISTS idx_gps_dia_line_ramal_interno_fecha "
    "ON gps(dia, id_linea, id_ramal, interno, fecha)"
)
IDX_TRAVEL_TIMES_GPS_ID = (
    "CREATE INDEX IF NOT EXISTS idx_travel_times_gps_id "
    "ON travel_times_gps(id)"
)
IDX_TRAVEL_TIMES_STATIONS_ID = (
    "CREATE INDEX IF NOT EXISTS idx_travel_times_stations_id "
    "ON travel_times_stations(id)"
)
IDX_SERVICES_STATS_LINE_DAY = (
    "CREATE INDEX IF NOT EXISTS idx_services_stats_line_day "
    "ON services_stats(id_linea, dia)"
)

ALL_INDEXES = [
    IDX_TRX_BATCH,
    IDX_ETAPAS_BATCH,
    IDX_GPS_LINE_DAY,
    IDX_ETAPAS_DIA_OD_VALIDADO,
    IDX_ETAPAS_DIA_LINE_RAMAL_INTERNO,
    IDX_GPS_DIA_LINE_RAMAL_INTERNO_FECHA,
    IDX_TRAVEL_TIMES_GPS_ID,
    IDX_TRAVEL_TIMES_STATIONS_ID,
    IDX_SERVICES_STATS_LINE_DAY,
]

ALL_TABLES = [
    TRANSACCIONES, TRANSACCIONES_RAW, DIAS_ULTIMA_CORRIDA, ETAPAS, VIAJES, USUARIOS,
    GPS, VEHICLE_EXPANSION_FACTORS,
    LEGS_TO_GPS_ORIGIN, LEGS_TO_GPS_DESTINATION,
    LEGS_TO_STATION_ORIGIN, LEGS_TO_STATION_DESTINATION,
    TRAVEL_TIMES_GPS, TRAVEL_TIMES_STATIONS, TRAVEL_TIMES_LEGS, TRAVEL_TIMES_TRIPS,
    TRANSACCIONES_LINEA, TARJETAS_DUPLICADAS, OCUPACION_POR_LINEA_TRAMO,
    SERVICES_GPS_POINTS, SERVICES, SERVICES_STATS,
    KPI_BY_DAY_LINE, KPI_BY_DAY_LINE_SERVICE,
]
