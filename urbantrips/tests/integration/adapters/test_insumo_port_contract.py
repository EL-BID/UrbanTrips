import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from urbantrips.storage.ports import InsumoPort


def _sample_routes() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id_linea": [1, 2]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs=4326,
    )


def _sample_stops() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_linea": [1, 1],
            "id_ramal": [10, 10],
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
            "h3_o": ["882a100d2bfffff", "882a100d6bfffff"],
            "h3_d": ["882a100d3bfffff", "882a100d4bfffff"],
            "h3_o_norm": ["882a100d2bfffff", "882a100d6bfffff"],
            "h3_d_norm": ["882a100d3bfffff", "882a100d4bfffff"],
            "distance_osm_drive": [500.0, 750.0],
            "distance_osm_walk": [600.0, 900.0],
            "distance_h3": [450.0, 700.0],
        }
    )


def _sample_metadata_lineas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_linea": [1],
            "nombre_linea": ["L1"],
            "id_linea_agg": [1],
            "nombre_linea_agg": ["L1"],
            "modo": ["bus"],
            "empresa": ["E1"],
            "descripcion": ["main line"],
        }
    )


def _sample_metadata_ramales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_ramal": [10],
            "id_linea": [1],
            "nombre_ramal": ["R1"],
            "modo": ["bus"],
            "empresa": ["E1"],
            "descripcion": ["main branch"],
        }
    )


def _sample_matrix_validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_linea_agg": [1],
            "id_ramal": [10],
            "parada": ["P1"],
            "area_influencia": ["A1"],
        }
    )


def _sample_travel_times_stations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_o": [100],
            "id_d": [101],
            "id_linea_o": [1],
            "id_ramal_o": [10],
            "lat_o": [-34.6],
            "lon_o": [-58.4],
            "id_linea_d": [1],
            "id_ramal_d": [10],
            "lat_d": [-34.7],
            "lon_d": [-58.5],
            "travel_time_min": [15.0],
        }
    )


@pytest.fixture(params=["memory", "duckdb"])
def insumo_adapter(request, tmp_path) -> InsumoPort:
    if request.param == "memory":
        from urbantrips.storage.adapters.memory.adapters import InMemoryInsumoAdapter

        return InMemoryInsumoAdapter()

    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    return DuckDBInsumoAdapter(tmp_path / "contract_insumos.duckdb")


def test_insumo_port_contract_reference_roundtrips(insumo_adapter):
    assert isinstance(insumo_adapter, InsumoPort)
    assert not insumo_adapter.has_routes()

    routes = _sample_routes()
    insumo_adapter.save_routes(routes)
    route_result = insumo_adapter.get_routes()
    assert len(route_result) == len(routes)
    assert set(route_result["id_linea"]) == {1, 2}
    assert insumo_adapter.has_routes()

    stops = _sample_stops()
    insumo_adapter.save_stops(stops)
    assert len(insumo_adapter.get_stops()) == len(stops)

    distances = _sample_distances()
    insumo_adapter.save_distances(distances)
    assert len(insumo_adapter.get_distances()) == len(distances)
    filtered = insumo_adapter.get_distances(h3_ids=["882a100d2bfffff"])
    assert filtered["h3_o"].tolist() == ["882a100d2bfffff"]


def test_insumo_port_contract_metadata_roundtrips(insumo_adapter):
    lineas = _sample_metadata_lineas()
    ramales = _sample_metadata_ramales()
    validation = _sample_matrix_validation()
    travel_times = _sample_travel_times_stations()

    insumo_adapter.save_metadata_lineas(lineas)
    insumo_adapter.save_metadata_ramales(ramales)
    insumo_adapter.save_matrix_validation(validation)
    insumo_adapter.save_travel_times_stations(travel_times)

    assert insumo_adapter.get_metadata_lineas()["nombre_linea"].tolist() == ["L1"]
    assert insumo_adapter.get_metadata_ramales()["nombre_ramal"].tolist() == ["R1"]
    assert insumo_adapter.get_matrix_validation()["parada"].tolist() == ["P1"]
    assert insumo_adapter.get_travel_times_stations()["travel_time_min"].tolist() == [15.0]


def test_insumo_port_contract_raw_helpers(insumo_adapter):
    first = pd.DataFrame({"id": [1], "value": ["a"]})
    second = pd.DataFrame({"id": [2], "value": ["b"]})

    insumo_adapter.save_raw(first, "contract_raw")
    insumo_adapter.append_raw(second, "contract_raw")

    result = insumo_adapter.get_raw("contract_raw").sort_values("id").reset_index(drop=True)
    expected = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(result[expected.columns], expected)

    missing = insumo_adapter.get_raw("missing_raw")
    assert isinstance(missing, pd.DataFrame)
    assert missing.empty


def test_insumo_port_contract_query_reads_existing_table(insumo_adapter):
    raw = pd.DataFrame({"id": [1], "value": ["a"]})
    insumo_adapter.save_raw(raw, "contract_raw")

    result = insumo_adapter.query("SELECT * FROM contract_raw")

    pd.testing.assert_frame_equal(result[raw.columns], raw)


def test_insumo_port_contract_rejects_invalid_raw_table_names(insumo_adapter):
    raw = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError, match="Invalid table name"):
        insumo_adapter.save_raw(raw, "contract_raw; DROP TABLE stops")
