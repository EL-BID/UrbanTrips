"""Migración de `services`: columnas horas_servicio y motivo_no_valido.

Se agregaron junto con la regla de validez por tiempo detenido. CREATE TABLE IF
NOT EXISTS no las agrega a una base ya creada, así que el adapter las suma al
conectar en modo escritura, sin tocar las filas existentes.
"""
import duckdb
import pandas as pd

_SERVICES_LEGACY = """
CREATE TABLE services (
    id_linea BIGINT, id_ramal BIGINT, dia TEXT, interno INT,
    original_service_id INT, service_id INT, total_points INT,
    distance_route FLOAT, distance_route_gps FLOAT, min_ts INT, max_ts INT,
    min_datetime TEXT, max_datetime TEXT, prop_idling FLOAT, valid INT
)
"""


def _build_legacy(path):
    conn = duckdb.connect(str(path))
    conn.execute(_SERVICES_LEGACY)
    conn.execute(
        "INSERT INTO services (id_linea, dia, interno, service_id, distance_route, valid) "
        "VALUES (10, '2026-03-09', 1, 0, 12.5, 1)"
    )
    conn.close()


def _columns(path):
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'services' ORDER BY ordinal_position"
            ).fetchall()
        ]
    finally:
        conn.close()


def test_services_recibe_las_columnas_nuevas_al_final_y_conserva_filas(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    db = tmp_path / "legacy_data.duckdb"
    _build_legacy(db)

    DuckDBDataAdapter(db).close()
    DuckDBDataAdapter(db).close()   # idempotente

    cols = _columns(db)
    assert cols[-2:] == ["horas_servicio", "motivo_no_valido"]
    assert cols.count("horas_servicio") == 1
    conn = duckdb.connect(str(db), read_only=True)
    try:
        fila = conn.execute(
            "SELECT id_linea, distance_route, valid, horas_servicio, motivo_no_valido "
            "FROM services"
        ).fetchall()
    finally:
        conn.close()
    assert fila == [(10, 12.5, 1, None, None)]


def test_base_nueva_y_base_migrada_tienen_las_mismas_columnas(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    legacy, nueva = tmp_path / "legacy.duckdb", tmp_path / "nueva.duckdb"
    _build_legacy(legacy)
    DuckDBDataAdapter(legacy).close()
    DuckDBDataAdapter(nueva).close()

    assert _columns(legacy) == _columns(nueva)


def test_save_services_inserta_por_nombre(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    # columnas en un orden distinto al de la tabla
    df = pd.DataFrame(
        {
            "motivo_no_valido": ["detenido"], "horas_servicio": [1.5], "valid": [False],
            "prop_idling": [0.7], "max_datetime": ["b"], "min_datetime": ["a"],
            "max_ts": [20], "min_ts": [10], "distance_route_gps": [None],
            "distance_route": [3.2], "total_points": [9], "service_id": [0],
            "original_service_id": [1], "interno": [7], "dia": ["2026-03-09"],
            "id_ramal": [100], "id_linea": [10],
        }
    )
    adapter.save_services(df)
    out = adapter.get_services()
    adapter.close()

    fila = out.iloc[0]
    assert (fila["id_linea"], fila["interno"], fila["total_points"]) == (10, 7, 9)
    assert fila["valid"] == 0
    assert fila["horas_servicio"] == 1.5
    assert fila["motivo_no_valido"] == "detenido"
