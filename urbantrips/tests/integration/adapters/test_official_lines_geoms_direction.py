# urbantrips/tests/integration/adapters/test_official_lines_geoms_direction.py
"""En las ciudades SIN ramales, `official_lines_geoms.direction` quedaba NULL.

`process_routes_geoms` armaba el dataframe de líneas con
`reindex(columns=["id_linea", "geometry"])`, que dejaba `direction` afuera; el
reindex siguiente la volvía a crear vacía. Como `save_raw` hace
CREATE OR REPLACE TABLE AS SELECT, el NOT NULL del DDL no lo frenaba y la tabla
quedaba con direction DOUBLE, toda NULL (704 filas así en la base de Mendoza).

Dos consecuencias: el join por (id_linea, direction) nunca matcheaba, así que
los recorridos oficiales se ignoraban en silencio y lines_geoms se armaba sólo
con los inferidos; y desde que build_routes_from_official_inferred hace FULL
JOIN, el NULL llega a lines_geoms.direction —que sí es NOT NULL— y la corrida
muere con ConstraintException.
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely import wkt as shapely_wkt


@pytest.fixture
def geojson_sin_direction(tmp_path):
    """Como el recorridos_mza.geojson real: sin columna `direction`."""
    gdf = gpd.GeoDataFrame(
        {
            "id_linea": [1, 2],
            "id_ramal": [10, 20],
            "geometry": [
                LineString([(-68.85, -32.89), (-68.84, -32.88)]),
                LineString([(-68.80, -32.90), (-68.79, -32.91)]),
            ],
        },
        crs=4326,
    )
    path = tmp_path / "recorridos.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


def _correr(monkeypatch, tmp_path, geojson_path, lineas_contienen_ramales=False):
    from urbantrips.carto import routes
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    monkeypatch.setattr(
        routes,
        "leer_configs_generales",
        lambda autogenerado=False: {
            "resolucion_h3": 8,
            "recorridos_geojson": geojson_path.name,
            "lineas_contienen_ramales": lineas_contienen_ramales,
        },
    )
    monkeypatch.setattr(
        routes,
        "get_paths",
        lambda: type("P", (), {"input_dir": geojson_path.parent})(),
    )
    # El H3 no es lo que se está probando y levanta un pool de procesos.
    monkeypatch.setattr(
        routes,
        "process_routes_into_h3_parallel",
        lambda routes_gdf, route_id_column, res=10: pd.DataFrame(
            columns=[route_id_column, "direction", "section_id", "h3", "wkt"]
        ),
    )
    monkeypatch.setattr(
        routes,
        "process_parent_h3_parallel",
        lambda **kw: pd.DataFrame(),
    )

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    ctx = type("Ctx", (), {"insumos": adapter})()
    routes.process_routes_geoms(ctx)
    return adapter


def test_official_lines_geoms_conserva_direction(
    monkeypatch, tmp_path, geojson_sin_direction
):
    adapter = _correr(monkeypatch, tmp_path, geojson_sin_direction)
    try:
        oficiales = adapter.get_raw("official_lines_geoms")
    finally:
        adapter.close()

    assert not oficiales.empty
    assert oficiales["direction"].notna().all(), "direction quedó NULL"
    # check_directions_on_geoms completa la dirección faltante: 2 líneas x 2
    assert sorted(oficiales["direction"].astype(int).tolist()) == [0, 0, 1, 1]
    assert sorted(oficiales["id_linea"].astype(int).tolist()) == [1, 1, 2, 2]


def test_el_oficial_gana_en_ciudades_sin_ramales(
    monkeypatch, tmp_path, geojson_sin_direction
):
    """El bug de fondo: con direction NULL el join no matcheaba nunca y el
    recorrido oficial se perdía. Además el FULL JOIN ya no puede crashear."""
    from urbantrips.carto.routes import build_routes_from_official_inferred

    adapter = _correr(monkeypatch, tmp_path, geojson_sin_direction)
    try:
        adapter.execute(
            "INSERT INTO inferred_lines_geoms VALUES "
            "(1, 0, 'LINESTRING (0 0, 1 1)'), (1, 1, 'LINESTRING (1 1, 0 0)')"
        )
        ctx = type("Ctx", (), {"insumos": adapter})()
        build_routes_from_official_inferred(ctx)

        rows = adapter.query(
            "SELECT id_linea, direction, wkt FROM lines_geoms "
            "ORDER BY id_linea, direction"
        )
    finally:
        adapter.close()

    # 2 líneas x 2 direcciones, ninguna duplicada por un join que no matchea
    assert len(rows) == 4
    assert rows["direction"].notna().all()
    # la línea 1 tiene oficial e inferido: gana el oficial
    assert "LINESTRING (0 0, 1 1)" not in rows["wkt"].tolist()


def test_la_z_del_geojson_no_llega_a_la_base(monkeypatch, tmp_path):
    """Los geojson de GIS traen LINESTRING Z con z=0 (el de Mendoza: las 352).

    Esa tercera coordenada no aporta nada y rompe a todo el que haga
    `for lon, lat in geom.coords`: el mapa de recorrido de Herramientas
    interactivas moría con "too many values to unpack (expected 2)". Se aplana
    en el ingest para que ninguna tabla guarde geometrías 3D.
    """
    gdf = gpd.GeoDataFrame(
        {
            "id_linea": [1, 2],
            "geometry": [
                LineString([(-68.85, -32.89, 0), (-68.84, -32.88, 0)]),
                LineString([(-68.80, -32.90, 0), (-68.79, -32.91, 0)]),
            ],
        },
        crs=4326,
    )
    path = tmp_path / "recorridos_z.geojson"
    gdf.to_file(path, driver="GeoJSON")
    assert gpd.read_file(path).geometry.has_z.all(), "el fixture debe entrar en 3D"

    adapter = _correr(monkeypatch, tmp_path, path)
    try:
        oficiales = adapter.get_raw("official_lines_geoms")
        lineas = adapter.get_raw("lines_geoms")
    finally:
        adapter.close()

    for nombre, tabla in (("official_lines_geoms", oficiales), ("lines_geoms", lineas)):
        if tabla.empty:
            continue
        for w in tabla["wkt"]:
            g = shapely_wkt.loads(w)
            assert not g.has_z, f"quedó una geometría 3D en {nombre}: {w[:60]}"
            # lo que hace el mapa del dashboard
            assert all(len(c) == 2 for c in g.coords)


def test_id_linea_string_del_geojson_no_rompe_el_coalesce(monkeypatch, tmp_path):
    """El geojson de AMBA trae id_linea como texto ("1001000015"); etapas (de
    donde sale inferred_lines_geoms) lo tiene como BIGINT en toda la base.

    geopandas lee esas properties como VARCHAR, y como `save_raw` hace
    CREATE OR REPLACE TABLE AS SELECT, ese es el tipo que queda en
    `official_lines_geoms` — no el BIGINT que dice el DDL. El
    `coalesce(o.id_linea, i.id_linea)` de `build_routes_from_official_inferred`
    mezclaba VARCHAR y BIGINT y DuckDB lo rechazaba con:
        BinderException: Cannot mix values of type VARCHAR and BIGINT in COALESCE

    Sólo aparece con ramales (branches_present=True): sin ramales el recorrido de
    línea sale directo del geojson vía otro camino que no tiene este problema en
    los fixtures de arriba, que ya usan id_linea numérico.
    """
    from urbantrips.carto.routes import build_routes_from_official_inferred

    # Un ramal por línea con las dos direcciones explícitas: así
    # `check_directions_on_geoms` no tiene nada que completar y
    # `create_line_geom_from_branches` toma la geometría directo (un solo branch
    # por (id_linea, direction), sin necesitar el lowess). Lo que se está
    # probando es el tipo de `id_linea`, no el armado de líneas con ramales.
    gdf = gpd.GeoDataFrame(
        {
            # texto, como el geojson real de AMBA — no [1, 2] como los fixtures de arriba.
            "id_linea": ["1001000015", "1001000015", "2001000348", "2001000348"],
            "id_ramal": [
                "1001000015001", "1001000015001",
                "2001000348001", "2001000348001",
            ],
            "direction": [0, 1, 0, 1],
            "geometry": [
                LineString([(-58.45, -34.61), (-58.44, -34.60)]),
                LineString([(-58.44, -34.60), (-58.45, -34.61)]),
                LineString([(-58.50, -34.65), (-58.49, -34.64)]),
                LineString([(-58.49, -34.64), (-58.50, -34.65)]),
            ],
        },
        crs=4326,
    )
    path = tmp_path / "recorridos_amba.geojson"
    gdf.to_file(path, driver="GeoJSON")
    assert gpd.read_file(path)["id_linea"].dtype == object, (
        "el fixture debe entrar como texto, igual que el geojson real"
    )

    adapter = _correr(monkeypatch, tmp_path, path, lineas_contienen_ramales=True)
    try:
        oficiales = adapter.get_raw("official_lines_geoms")
        assert pd.api.types.is_integer_dtype(oficiales["id_linea"]), (
            f"id_linea quedó {oficiales['id_linea'].dtype}, no numérico"
        )

        # Con un inferido BIGINT de verdad (como sale de etapas), el COALESCE no
        # puede fallar por tipos aunque los dos lados vengan de fuentes distintas.
        adapter.execute(
            "INSERT INTO inferred_lines_geoms VALUES "
            "(2001000348, 1, 'LINESTRING (0 0, 1 1)'), "
            "(9999999999, 0, 'LINESTRING (2 2, 3 3)')"
        )
        ctx = type("Ctx", (), {"insumos": adapter})()
        build_routes_from_official_inferred(ctx)  # no debe tirar BinderException

        rows = adapter.query(
            "SELECT id_linea, direction FROM lines_geoms ORDER BY id_linea, direction"
        )
    finally:
        adapter.close()

    # `.tolist()`, no `.astype(int)`: en Windows numpy mapea el `int` de Python al
    # `long` de C (32 bits), así que un id como 9999999999 se truncaría a
    # 1410065407 en la VERIFICACIÓN sin que el dato real esté mal — ya pasó acá.
    ids = sorted(rows["id_linea"].tolist())
    assert ids == [1001000015, 1001000015, 2001000348, 2001000348, 9999999999]
