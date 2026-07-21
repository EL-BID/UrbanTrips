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


def _sample_distances() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "h3_o": ["882a100d2bfffff", "882a100d2bfffff"],
            "h3_d": ["882a100d3bfffff", "882a100d4bfffff"],
            "h3_o_norm": ["882a100d2bfffff", "882a100d2bfffff"],
            "h3_d_norm": ["882a100d3bfffff", "882a100d4bfffff"],
            "distance_osm_drive": [500.0, 750.0],
            "distance_osm_walk": [600.0, 900.0],
            "distance_h3": [450.0, 700.0],
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


def test_distances_roundtrip(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    adapter.save_distances(_sample_distances())
    result = adapter.get_distances()
    assert len(result) == 2


def test_distances_filter_by_h3_ids(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    adapter.save_distances(_sample_distances())
    result = adapter.get_distances(h3_ids=["882a100d2bfffff"])
    assert len(result) == 2  # both rows have h3_o matching the filter


def test_satisfies_insumo_port(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.storage.ports import InsumoPort

    adapter = DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")
    assert isinstance(adapter, InsumoPort)
