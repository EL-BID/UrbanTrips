# urbantrips/tests/integration/adapters/test_schema_migration_dead_columns.py
"""Migración de las columnas y tablas muertas quitadas el 2026-07-27.

Las métricas de distancia y tiempo se movieron de etapas/viajes a
travel_times_legs/travel_times_trips durante el refactor, pero las columnas
viejas quedaron declaradas (100% NULL). CREATE TABLE IF NOT EXISTS no las saca
de una base ya creada, así que los adapters las dropean al conectar en modo
escritura. Estos tests fijan ese comportamiento y, sobre todo, que sea
idempotente y que no toque las filas.
"""
import duckdb


# DDL previo al 2026-07-27, con las columnas y tablas que hoy ya no existen.
_ETAPAS_LEGACY = """
CREATE TABLE etapas (
    id INT, batch_id INT, id_tarjeta TEXT, dia TEXT, id_viaje INT, id_etapa INT,
    tiempo TEXT, hora INT, modo TEXT, id_linea BIGINT, id_ramal BIGINT,
    interno INT, genero TEXT, tarifa TEXT, latitud FLOAT, longitud FLOAT,
    h3_o TEXT, h3_d TEXT, od_validado INT, etapa_validada INT,
    factor_expansion_original FLOAT, factor_expansion_linea FLOAT,
    factor_expansion_tarjeta FLOAT, factor_expansion_etapa FLOAT,
    distancia FLOAT, travel_time_min FLOAT
)
"""

_VIAJES_LEGACY = """
CREATE TABLE viajes (
    id_tarjeta TEXT, id_viaje INT, dia TEXT, tiempo TEXT, hora INT,
    cant_etapas INT, modo TEXT, autobus INT, tren INT, metro INT, tranvia INT,
    brt INT, cable INT, lancha INT, otros INT, h3_o TEXT, h3_d TEXT,
    genero TEXT, tarifa TEXT, od_validado INT, factor_expansion_linea FLOAT,
    factor_expansion_tarjeta FLOAT, distancia FLOAT, travel_time_min FLOAT
)
"""

_SERVICES_GPS_POINTS_LEGACY = """
CREATE TABLE services_gps_points (
    id INT PRIMARY KEY, id_linea BIGINT, id_ramal BIGINT, interno INT,
    dia TEXT, original_service_id INT, new_service_id INT, service_id INT,
    id_ramal_gps_point BIGINT, node_id INT
)
"""


def _columns(path, table) -> list[str]:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
        ]
    finally:
        conn.close()


def _tables(path) -> set[str]:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
    finally:
        conn.close()


def _build_legacy_data_db(path):
    conn = duckdb.connect(str(path))
    conn.execute(_ETAPAS_LEGACY)
    # una fila con valores en las columnas muertas (bases pre-refactor) y otra
    # sin ellos: el drop no debe depender del contenido
    conn.execute(
        "INSERT INTO etapas (id, dia, od_validado, distancia, travel_time_min) "
        "VALUES (1, '2026-03-09', 1, 9.9, 7.7), (2, '2026-03-10', 1, NULL, NULL)"
    )
    conn.execute(_VIAJES_LEGACY)
    conn.execute(
        "INSERT INTO viajes (id_tarjeta, id_viaje, dia, distancia, travel_time_min) "
        "VALUES ('A', 1, '2026-03-09', 3.3, 12.0)"
    )
    conn.execute(_SERVICES_GPS_POINTS_LEGACY)
    conn.execute(
        "INSERT INTO services_gps_points (id, id_linea, dia, id_ramal_gps_point, node_id) "
        "VALUES (1, 10, '2026-03-09', 55, 77)"
    )
    conn.execute(
        "CREATE TABLE travel_times_gps "
        "(dia TEXT, id INT, travel_time_min FLOAT, travel_speed FLOAT)"
    )
    conn.execute("INSERT INTO travel_times_gps VALUES ('2026-03-09', 1, 5.0, 20.0)")
    conn.close()


def test_data_adapter_drops_dead_columns_and_table(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    db = tmp_path / "legacy_data.duckdb"
    _build_legacy_data_db(db)

    DuckDBDataAdapter(db).close()

    assert "distancia" not in _columns(db, "etapas")
    assert "travel_time_min" not in _columns(db, "etapas")
    assert "distancia" not in _columns(db, "viajes")
    assert "travel_time_min" not in _columns(db, "viajes")
    assert "id_ramal_gps_point" not in _columns(db, "services_gps_points")
    assert "node_id" not in _columns(db, "services_gps_points")
    assert "travel_times_gps" not in _tables(db)


def test_data_migration_preserves_rows_and_surviving_columns(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    db = tmp_path / "legacy_data.duckdb"
    _build_legacy_data_db(db)

    DuckDBDataAdapter(db).close()

    conn = duckdb.connect(str(db), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM etapas").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM viajes").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM services_gps_points").fetchone()[0] == 1
        )
        # las columnas que sobreviven conservan sus valores
        assert conn.execute(
            "SELECT dia FROM etapas ORDER BY id"
        ).fetchall() == [("2026-03-09",), ("2026-03-10",)]
        assert conn.execute(
            "SELECT id_linea, service_id FROM services_gps_points"
        ).fetchone()[0] == 10
    finally:
        conn.close()


def test_data_migration_is_idempotent(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    db = tmp_path / "legacy_data.duckdb"
    _build_legacy_data_db(db)

    # Se dispara en CADA conexión de escritura, así que la segunda pasada tiene
    # que ser un no-op y no romper por columna inexistente.
    DuckDBDataAdapter(db).close()
    DuckDBDataAdapter(db).close()
    DuckDBDataAdapter(db).close()

    assert "distancia" not in _columns(db, "etapas")
    conn = duckdb.connect(str(db), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM etapas").fetchone()[0] == 2
    finally:
        conn.close()


def test_data_migration_on_fresh_db_is_noop(tmp_path):
    """Una base nueva no tiene las tablas: la migración no debe romper."""
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    db = tmp_path / "fresh_data.duckdb"
    DuckDBDataAdapter(db).close()

    assert "distancia" not in _columns(db, "etapas")
    assert "travel_times_gps" not in _tables(db)
    assert "etapas" in _tables(db)


def test_insumos_adapter_drops_distancias(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    db = tmp_path / "legacy_insumos.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute(
        "CREATE TABLE distancias (h3_o TEXT, h3_d TEXT, h3_o_norm TEXT, "
        "h3_d_norm TEXT, distance_osm_drive FLOAT, distance_osm_walk FLOAT, "
        "distance_h3 FLOAT)"
    )
    conn.execute("INSERT INTO distancias VALUES ('a', 'b', 'a', 'b', 1, 2, 3)")
    conn.close()

    DuckDBInsumoAdapter(db).close()
    assert "distancias" not in _tables(db)

    # idempotente, y las tablas vivas siguen ahí
    DuckDBInsumoAdapter(db).close()
    assert "distancias" not in _tables(db)
    assert "matriz_validacion" in _tables(db)


def test_dash_adapter_drops_dead_ddl_tables(tmp_path):
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter

    db = tmp_path / "legacy_dash.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute(
        "CREATE TABLE particion_modal (desc_dia TEXT, tipo_dia TEXT, tipo TEXT, "
        "modo TEXT, modal FLOAT)"
    )
    conn.execute(
        "CREATE TABLE lines_od_matrix_by_section (id_linea BIGINT, yr_mo TEXT, "
        "day_type TEXT, n_sections INT, hour_min INT, hour_max INT, Origen INT, "
        "Destino INT, legs INT, prop FLOAT, nombre_linea TEXT)"
    )
    conn.close()

    DuckDBDashAdapter(db).close()

    tables = _tables(db)
    assert "particion_modal" not in tables
    assert "lines_od_matrix_by_section" not in tables
    # la tabla real del dashboard no se toca
    assert "chains_norm" in tables


def test_read_only_connection_does_not_migrate(tmp_path):
    """El dashboard abre en solo lectura: no debe intentar ningún ALTER/DROP."""
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    db = tmp_path / "legacy_data.duckdb"
    _build_legacy_data_db(db)

    DuckDBDataAdapter(db, read_only=True).close()

    assert "distancia" in _columns(db, "etapas")
    assert "travel_times_gps" in _tables(db)
