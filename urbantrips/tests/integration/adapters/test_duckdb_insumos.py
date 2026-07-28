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
