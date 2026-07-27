# urbantrips/storage/schema/insumos.py

# NOTA: la tabla `distancias` se eliminó (2026-07-27). El cache real de
# distancias OD es el archivo aparte `od_distances` (o_norm, d_norm, distance_m),
# que crea y consulta carto/compute_distances.py. `distancias` no tenía productor
# fuera de los tests y su único lector estaba en código sin llamadores.

MATRIZ_VALIDACION = """
CREATE TABLE IF NOT EXISTS matriz_validacion (
    id_linea_agg    BIGINT,
    id_ramal        BIGINT,
    parada          TEXT,
    area_influencia TEXT
)
"""

MATRIZ_PARADAS = """
CREATE TABLE IF NOT EXISTS matriz_paradas (
    id_linea BIGINT,
    id_ramal BIGINT,
    parada   TEXT,
    n_trx    BIGINT,
    n_gps    BIGINT,
    valido   INTEGER
)
"""

MATRIZ_PARADAS_DIAS = """
CREATE TABLE IF NOT EXISTS matriz_paradas_dias (
    dia TEXT
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
    id_ramal  BIGINT NOT NULL,
    direction INT NOT NULL,
    wkt       TEXT NOT NULL
)
"""

# Faltaba en ALL_TABLES (agregada 2026-07-27): la escribe process_routes_geoms
# con save_raw (que la autocrea), pero esa función corta antes si el config no
# trae `recorridos_geojson` (carto/routes.py:270-274). En ese caso la tabla no
# existía y build_routes_from_official_inferred, que la LEE sin try/except en
# run_outputs, tiraba CatalogException. Declarándola, _apply_schema la crea vacía
# y el LEFT JOIN simplemente no aporta filas, que es la semántica buscada.
OFFICIAL_LINES_GEOMS = """
CREATE TABLE IF NOT EXISTS official_lines_geoms (
    id_linea  BIGINT NOT NULL,
    direction INT NOT NULL,
    wkt       TEXT NOT NULL
)
"""

INFERRED_LINES_GEOMS = """
CREATE TABLE IF NOT EXISTS inferred_lines_geoms (
    id_linea  BIGINT NOT NULL,
    direction INT NOT NULL,
    wkt       TEXT NOT NULL
)
"""

LINES_GEOMS = """
CREATE TABLE IF NOT EXISTS lines_geoms (
    id_linea  BIGINT NOT NULL,
    direction INT NOT NULL,
    wkt       TEXT NOT NULL
)
"""

BRANCHES_GEOMS = """
CREATE TABLE IF NOT EXISTS branches_geoms (
    id_ramal  BIGINT NOT NULL,
    direction INT NOT NULL,
    wkt       TEXT NOT NULL
)
"""

STOPS = """
CREATE TABLE IF NOT EXISTS stops (
    id_linea          BIGINT NOT NULL,
    id_ramal          BIGINT NOT NULL,
    direction         INT NOT NULL,
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
    id_ramal   BIGINT NOT NULL,
    direction  INT NOT NULL,
    section_id INT,
    h3         TEXT,
    wkt        TEXT NOT NULL
)
"""

OFFICIAL_BRANCHES_GEOMS_H3_PARENT = """
CREATE TABLE IF NOT EXISTS official_branches_geoms_h3_parent (
    id_ramal   BIGINT NOT NULL,
    direction  INT NOT NULL,
    section_id INT,
    h3         TEXT,
    resolution INT,
    wkt        TEXT NOT NULL
)
"""

OFFICIAL_LINES_GEOMS_H3 = """
CREATE TABLE IF NOT EXISTS official_lines_geoms_h3 (
    id_linea   BIGINT NOT NULL,
    direction  INT NOT NULL,
    section_id INT,
    h3         TEXT,
    wkt        TEXT NOT NULL
)
"""

OFFICIAL_LINES_GEOMS_H3_PARENT = """
CREATE TABLE IF NOT EXISTS official_lines_geoms_h3_parent (
    id_linea   BIGINT NOT NULL,
    direction  INT NOT NULL,
    section_id INT,
    h3         TEXT,
    resolution INT,
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
    MATRIZ_VALIDACION,
    MATRIZ_PARADAS,
    MATRIZ_PARADAS_DIAS,
    POLIGONOS,
    METADATA_LINEAS,
    METADATA_RAMALES,
    OFFICIAL_BRANCHES_GEOMS,
    OFFICIAL_LINES_GEOMS,
    INFERRED_LINES_GEOMS,
    LINES_GEOMS,
    BRANCHES_GEOMS,
    STOPS,
    ROUTES_SECTION_ID_COORDS,
    OFFICIAL_BRANCHES_GEOMS_H3,
    OFFICIAL_BRANCHES_GEOMS_H3_PARENT,
    OFFICIAL_LINES_GEOMS_H3,
    OFFICIAL_LINES_GEOMS_H3_PARENT,
    TRAVEL_TIMES_STATIONS,
]
