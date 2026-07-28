# urbantrips/tests/integration/adapters/test_official_lines_geoms_ddl.py
"""`official_lines_geoms` faltaba en insumos.ALL_TABLES.

La escribe `process_routes_geoms` con `save_raw` (que la autocrea), pero esa
función corta antes si el config no trae `recorridos_geojson`
(carto/routes.py:270-274). En ese caso la tabla no existía y
`build_routes_from_official_inferred` —que la LEE sin try/except y corre
incondicionalmente en run_outputs— tiraba `CatalogException`.
"""
import duckdb
import pytest


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


def test_official_lines_geoms_is_declared_in_schema():
    from urbantrips.storage.schema import insumos as schema

    assert hasattr(schema, "OFFICIAL_LINES_GEOMS")
    assert "official_lines_geoms" in schema.OFFICIAL_LINES_GEOMS
    assert schema.OFFICIAL_LINES_GEOMS in schema.ALL_TABLES


def test_fresh_insumos_db_has_official_lines_geoms(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    db = tmp_path / "insumos.duckdb"
    DuckDBInsumoAdapter(db).close()

    assert "official_lines_geoms" in _tables(db)
    # mismo shape que sus hermanas inferred_lines_geoms / lines_geoms
    conn = duckdb.connect(str(db), read_only=True)
    try:
        cols = [
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'official_lines_geoms' ORDER BY ordinal_position"
            ).fetchall()
        ]
    finally:
        conn.close()
    assert cols == ["id_linea", "direction", "wkt"]


def test_build_routes_no_crashea_sin_recorridos_oficiales(tmp_path):
    """El caso que crasheaba: nunca se escribió official_lines_geoms.

    Con la tabla declarada, el LEFT JOIN no aporta filas y lines_geoms queda con
    las geometrías inferidas — que es la semántica buscada por el COALESCE.
    """
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.carto.routes import build_routes_from_official_inferred

    db = tmp_path / "insumos.duckdb"
    adapter = DuckDBInsumoAdapter(db)
    # solo geometrías inferidas: es el escenario "config sin recorridos_geojson"
    adapter.execute(
        "INSERT INTO inferred_lines_geoms VALUES "
        "(1, 0, 'LINESTRING (0 0, 1 1)'), (2, 0, 'LINESTRING (2 2, 3 3)')"
    )

    ctx = type("Ctx", (), {"insumos": adapter})()
    build_routes_from_official_inferred(ctx)

    rows = adapter.query(
        "SELECT id_linea, direction, wkt FROM lines_geoms ORDER BY id_linea"
    )
    adapter.close()

    assert len(rows) == 2
    assert rows["wkt"].tolist() == [
        "LINESTRING (0 0, 1 1)",
        "LINESTRING (2 2, 3 3)",
    ]


def test_build_routes_prefiere_la_geometria_oficial(tmp_path):
    """Cuando SÍ hay recorrido oficial, el COALESCE lo prefiere sobre el inferido."""
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.carto.routes import build_routes_from_official_inferred

    db = tmp_path / "insumos.duckdb"
    adapter = DuckDBInsumoAdapter(db)
    adapter.execute(
        "INSERT INTO inferred_lines_geoms VALUES "
        "(1, 0, 'LINESTRING (0 0, 1 1)'), (2, 0, 'LINESTRING (2 2, 3 3)')"
    )
    adapter.execute(
        "INSERT INTO official_lines_geoms VALUES (1, 0, 'LINESTRING (9 9, 8 8)')"
    )

    ctx = type("Ctx", (), {"insumos": adapter})()
    build_routes_from_official_inferred(ctx)

    rows = adapter.query(
        "SELECT id_linea, wkt FROM lines_geoms ORDER BY id_linea"
    )
    adapter.close()

    assert rows["wkt"].tolist() == [
        "LINESTRING (9 9, 8 8)",   # oficial gana
        "LINESTRING (2 2, 3 3)",   # sin oficial, queda la inferida
    ]
