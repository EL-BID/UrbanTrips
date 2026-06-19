# urbantrips/storage/schema/insumos.py

DISTANCIAS = """
CREATE TABLE IF NOT EXISTS distancias (
    h3_o               TEXT NOT NULL,
    h3_d               TEXT NOT NULL,
    h3_o_norm          TEXT NOT NULL,
    h3_d_norm          TEXT NOT NULL,
    distance_osm_drive FLOAT,
    distance_osm_walk  FLOAT,
    distance_h3        FLOAT
)
"""

MATRIZ_VALIDACION = """
CREATE TABLE IF NOT EXISTS matriz_validacion (
    id_linea_agg    BIGINT,
    id_ramal        BIGINT,
    parada          TEXT,
    area_influencia TEXT
)
"""

POLIGONOS = """
CREATE TABLE IF NOT EXISTS poligonos (
    id              TEXT PRIMARY KEY NOT NULL,
    tipo            TEXT,
    wkt             TEXT NOT NULL
)
"""

METADATA_LINEAS = """
CREATE TABLE IF NOT EXISTS metadata_lineas (
    id_linea         BIGINT PRIMARY KEY NOT NULL,
    nombre_linea     TEXT NOT NULL,
    id_linea_agg     BIGINT,
    nombre_linea_agg TEXT,
    modo             TEXT,
    empresa          TEXT,
    descripcion      TEXT
)
"""

METADATA_RAMALES = """
CREATE TABLE IF NOT EXISTS metadata_ramales (
    id_ramal     BIGINT PRIMARY KEY NOT NULL,
    id_linea     BIGINT NOT NULL,
    nombre_ramal TEXT NOT NULL,
    modo         TEXT NOT NULL,
    empresa      TEXT,
    descripcion  TEXT
)
"""

OFFICIAL_BRANCHES_GEOMS = """
CREATE TABLE IF NOT EXISTS official_branches_geoms (
    id_ramal BIGINT PRIMARY KEY NOT NULL,
    wkt      TEXT NOT NULL
)
"""

INFERRED_LINES_GEOMS = """
CREATE TABLE IF NOT EXISTS inferred_lines_geoms (
    id_linea BIGINT PRIMARY KEY NOT NULL,
    wkt      TEXT NOT NULL
)
"""

LINES_GEOMS = """
CREATE TABLE IF NOT EXISTS lines_geoms (
    id_linea BIGINT PRIMARY KEY NOT NULL,
    wkt      TEXT NOT NULL
)
"""

BRANCHES_GEOMS = """
CREATE TABLE IF NOT EXISTS branches_geoms (
    id_ramal BIGINT PRIMARY KEY NOT NULL,
    wkt      TEXT NOT NULL
)
"""

STOPS = """
CREATE TABLE IF NOT EXISTS stops (
    id_linea          BIGINT NOT NULL,
    id_ramal          BIGINT NOT NULL,
    node_id           INT NOT NULL,
    branch_stop_order INT NOT NULL,
    stop_x            FLOAT NOT NULL,
    stop_y            FLOAT NOT NULL,
    node_x            FLOAT NOT NULL,
    node_y            FLOAT NOT NULL
)
"""

ROUTES_SECTION_ID_COORDS = """
CREATE TABLE IF NOT EXISTS routes_section_id_coords (
    id_linea    BIGINT NOT NULL,
    n_sections  INT NOT NULL,
    section_id  INT NOT NULL,
    section_lrs FLOAT NOT NULL,
    x           FLOAT NOT NULL,
    y           FLOAT NOT NULL
)
"""

OFFICIAL_BRANCHES_GEOMS_H3 = """
CREATE TABLE IF NOT EXISTS official_branches_geoms_h3 (
    id_ramal   BIGINT PRIMARY KEY NOT NULL,
    section_id INT,
    h3         TEXT,
    wkt        TEXT NOT NULL
)
"""

TRAVEL_TIMES_STATIONS = """
CREATE TABLE IF NOT EXISTS travel_times_stations (
    id_o         INT,
    id_d         INT,
    id_linea_o   BIGINT,
    id_ramal_o   BIGINT,
    lat_o        FLOAT,
    lon_o        FLOAT,
    id_linea_d   BIGINT,
    id_ramal_d   BIGINT,
    lat_d        FLOAT,
    lon_d        FLOAT,
    travel_time_min FLOAT
)
"""

ALL_TABLES = [
    DISTANCIAS, MATRIZ_VALIDACION, POLIGONOS,METADATA_LINEAS, METADATA_RAMALES,
    OFFICIAL_BRANCHES_GEOMS, INFERRED_LINES_GEOMS, LINES_GEOMS, BRANCHES_GEOMS,
    STOPS, ROUTES_SECTION_ID_COORDS, OFFICIAL_BRANCHES_GEOMS_H3,
    TRAVEL_TIMES_STATIONS,
]
