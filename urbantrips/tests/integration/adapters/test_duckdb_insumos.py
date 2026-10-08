# urbantrips/tests/integration/adapters/test_duckdb_insumos.py
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString


def _sample_routes() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id_linea": [1, 2], "nombre": ["L1", "L2"], "direction": [0, 1]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs=4326,
    )


def _sample_stops() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_linea": [1, 1],
            "id_ramal": [10, 10],
            "direction": [0, 0],
            "node_id": [100, 101],
            "branch_stop_order": [0, 1],
            "stop_x": [0.0, 1.0],
            "stop_y": [0.0, 1.0],
            "node_x": [0.0, 1.0],
            "node_y": [0.0, 1.0],
            "h3": ["", ""],
        }
    )


def test_routes_roundtrip(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    routes = _sample_routes()
    adapter.save_routes(routes)
    result = adapter.get_routes()
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 2
    assert set(result["id_linea"]) == {1, 2}
    assert result.crs.to_epsg() == 4326


def test_stops_roundtrip(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    adapter.save_stops(_sample_stops())
    result = adapter.get_stops()
    assert len(result) == 2
    assert "id_linea" in result.columns


def test_satisfies_insumo_port(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.storage.ports import InsumoPort

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    assert isinstance(adapter, InsumoPort)


# ── zonificaciones ────────────────────────────────────────────────────────────
# get_zones() fue un stub que devolvia GeoDataFrame() sin consultar nada, y como
# sus dos consumidores (bbox del filtro geografico y bbox de la red OSM) tienen
# fallback silencioso al config, el bug no daba senal. Estos tests atacan eso
# desde el adaptador: un mock del ctx no lo detectaria.

def _zonificaciones_como_las_guarda_carto() -> pd.DataFrame:
    """Replica lo que escribe `guardo_zonificaciones`: WKT en la columna
    `geometry` (via `_with_wkt_geometry`), NO en una columna `wkt`."""
    from shapely.geometry import Polygon

    poly = Polygon([(-60, -36), (-57, -36), (-57, -34), (-60, -34)])
    return pd.DataFrame(
        {"zona": ["Partido"], "id": ["X"], "orden": [0], "geometry": [poly.wkt]}
    )


def test_get_zones_lee_la_tabla_zonificaciones(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    adapter.save_raw(_zonificaciones_como_las_guarda_carto(), "zonificaciones")

    result = adapter.get_zones()

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    # total_bounds es lo que consume bbox_area_estudio; si la geometria quedara
    # como string el GeoDataFrame no lo expondria
    assert tuple(result.total_bounds) == (-60.0, -36.0, -57.0, -34.0)
    assert result.crs is not None and result.crs.to_epsg() == 4326


def test_get_zones_sin_tabla_devuelve_vacio(tmp_path):
    """Antes de la primera `guardo_zonificaciones` la tabla no existe: el bbox
    tiene que poder caer al config sin que esto explote."""
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    assert len(adapter.get_zones()) == 0


def test_get_zones_tabla_vacia_devuelve_vacio(tmp_path):
    """Proyecto sin zonificaciones declaradas: tabla presente pero sin filas."""
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    adapter.save_raw(_zonificaciones_como_las_guarda_carto().iloc[0:0], "zonificaciones")
    assert len(adapter.get_zones()) == 0
