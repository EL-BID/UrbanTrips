# urbantrips/storage/schema/dash.py

MATRICES = """
CREATE TABLE IF NOT EXISTS matrices (
    desc_dia TEXT NOT NULL,
    tipo_dia TEXT NOT NULL,
    var_zona TEXT NOT NULL,
    filtro1  TEXT NOT NULL,
    Origen   TEXT NOT NULL,
    Destino  TEXT NOT NULL,
    Viajes   INT  NOT NULL
)
"""

LINEAS_DESEO = """
CREATE TABLE IF NOT EXISTS lineas_deseo (
    desc_dia TEXT NOT NULL,
    tipo_dia TEXT NOT NULL,
    var_zona TEXT NOT NULL,
    filtro1  TEXT NOT NULL,
    Origen   TEXT NOT NULL,
    Destino  TEXT NOT NULL,
    Viajes   INT  NOT NULL,
    lon_o    FLOAT,
    lat_o    FLOAT,
    lon_d    FLOAT,
    lat_d    FLOAT
)
"""

VIAJES_HORA = """
CREATE TABLE IF NOT EXISTS viajes_hora (
    desc_dia TEXT NOT NULL,
    tipo_dia TEXT NOT NULL,
    Hora     INT,
    Viajes   INT,
    Modo     TEXT
)
"""

DISTRIBUCION = """
CREATE TABLE IF NOT EXISTS distribucion (
    desc_dia  TEXT NOT NULL,
    tipo_dia  TEXT NOT NULL,
    Distancia INT,
    Viajes    INT,
    Modo      TEXT
)
"""

INDICADORES = """
CREATE TABLE IF NOT EXISTS indicadores (
    desc_dia  TEXT NOT NULL,
    tipo_dia  TEXT NOT NULL,
    Titulo    TEXT,
    orden     INT,
    Indicador TEXT,
    Valor     TEXT
)
"""

# NOTA: el DDL de `particion_modal` se eliminó (2026-07-27). Nadie la escribía:
# la tabla real del dashboard es `datos_particion_modal`, que crea
# preparo_dashboard.py y leen las páginas (1_Datos Generales).

OCUPACION_POR_LINEA_TRAMO = """
CREATE TABLE IF NOT EXISTS ocupacion_por_linea_tramo (
    id_linea       BIGINT NOT NULL,
    yr_mo          TEXT,
    nombre_linea   TEXT,
    day_type       TEXT NOT NULL,
    n_sections     INT,
    section_meters INT,
    sentido        TEXT NOT NULL,
    section_id     INT NOT NULL,
    hour_min       INT,
    hour_max       INT,
    legs           INT NOT NULL,
    prop           FLOAT NOT NULL,
    buff_factor    FLOAT,
    wkt            TEXT
)
"""

# NOTA: el DDL de `lines_od_matrix_by_section` se eliminó del esquema de dash
# (2026-07-27). Ese nombre existe en la base `data`, pero la copia del dash se
# llama `matrices_linea` (ver la nota en kpi/line_od_matrix.py), así que acá
# nunca se escribía nada.

MATRICES_LINEA_CARTO = """
CREATE TABLE IF NOT EXISTS matrices_linea_carto (
    id_linea     BIGINT NOT NULL,
    n_sections   INT NOT NULL,
    section_id   INT NOT NULL,
    wkt          TEXT,
    x            FLOAT,
    y            FLOAT,
    nombre_linea TEXT
)
"""

SERVICES_BY_LINE_HOUR = """
CREATE TABLE IF NOT EXISTS services_by_line_hour (
    id_linea  BIGINT  NOT NULL,
    dia       TEXT NOT NULL,
    hora      INT  NOT NULL,
    servicios FLOAT NOT NULL
)
"""

CHAINS_NORM = """
CREATE TABLE IF NOT EXISTS chains_norm (
    dia                    TEXT,
    id_tarjeta             TEXT,
    id_viaje               BIGINT,
    h3_inicio              TEXT,
    h3_transfer1           TEXT,
    h3_transfer2           TEXT,
    h3_fin                 TEXT,
    h3_inicio_norm         TEXT,
    h3_transfer1_norm      TEXT,
    h3_transfer2_norm      TEXT,
    h3_fin_norm            TEXT,
    modo_agregado          TEXT,
    rango_hora             TEXT,
    genero_agregado        TEXT,
    tarifa_agregada        TEXT,
    transferencia          INTEGER,
    distancia_agregada     TEXT,
    travel_speed           DOUBLE,
    factor_expansion_linea DOUBLE,
    tipo_dia               TEXT,
    mes                    TEXT,
    seq_lineas             TEXT,
    distance_od            DOUBLE,
    travel_time_min        DOUBLE
)
"""

ALL_TABLES = [
    MATRICES, LINEAS_DESEO, VIAJES_HORA, DISTRIBUCION, INDICADORES,
    OCUPACION_POR_LINEA_TRAMO,
    MATRICES_LINEA_CARTO, SERVICES_BY_LINE_HOUR, CHAINS_NORM,
]

# Explicit set of valid table names for DashPort adapter validation
VALID_TABLE_NAMES: frozenset[str] = frozenset({
    "matrices", "lineas_deseo", "viajes_hora", "distribucion", "indicadores",
    "ocupacion_por_linea_tramo",
    "matrices_linea_carto", "services_by_line_hour", "chains_norm",
    "equivalencias_zonas",
})
