import inspect
import pytest
import pandas as pd
from urbantrips.storage.context import StorageContext
from urbantrips.storage.adapters.memory.adapters import (
    InMemoryDataAdapter, InMemoryInsumoAdapter,
    InMemoryDashAdapter, InMemoryGeneralAdapter,
)


@pytest.fixture
def ctx():
    return StorageContext(
        data=InMemoryDataAdapter(),
        insumos=InMemoryInsumoAdapter(),
        dash=InMemoryDashAdapter(),
        general=InMemoryGeneralAdapter(),
    )


def test_write_transactions_to_db_signature():
    from urbantrips.datamodel.transactions import write_transactions_to_db
    params = list(inspect.signature(write_transactions_to_db).parameters)
    assert params[0] == "ctx"
    assert "corrida" in params


def test_write_transactions_to_db_registers_run(ctx):
    from urbantrips.datamodel.transactions import write_transactions_to_db
    write_transactions_to_db(ctx, "run01")
    runs = ctx.general.get_completed_runs()
    assert len(runs) == 1
    assert runs["corrida"].iloc[0] == "run01"


def test_agrego_factor_expansion_signature():
    from urbantrips.datamodel.transactions import agrego_factor_expansion
    params = list(inspect.signature(agrego_factor_expansion).parameters)
    assert params[0] == "trx"
    assert params[1] == "ctx"


def test_process_routes_geoms_accepts_ctx_param():
    from urbantrips.carto.routes import process_routes_geoms
    params = list(inspect.signature(process_routes_geoms).parameters)
    assert params[0] == "ctx"


def test_create_stops_table_accepts_ctx_param():
    from urbantrips.carto.stops import create_stops_table
    params = list(inspect.signature(create_stops_table).parameters)
    assert params[0] == "ctx"


def test_get_route_geoms_with_sections_data_accepts_ctx_param():
    from urbantrips.carto.routes import get_route_geoms_with_sections_data
    params = list(inspect.signature(get_route_geoms_with_sections_data).parameters)
    assert "ctx" in params


def test_check_exists_route_section_accepts_ctx_param():
    from urbantrips.carto.routes import check_exists_route_section_points_table
    params = list(inspect.signature(check_exists_route_section_points_table).parameters)
    assert "ctx" in params


def test_insumo_adapter_has_execute_save_raw_get_raw():
    from urbantrips.storage.adapters.memory.adapters import InMemoryInsumoAdapter
    adapter = InMemoryInsumoAdapter()
    assert hasattr(adapter, "execute")
    assert hasattr(adapter, "save_raw")
    assert hasattr(adapter, "get_raw")
    assert hasattr(adapter, "append_raw")


def test_insumo_save_raw_get_raw_roundtrip():
    from urbantrips.storage.adapters.memory.adapters import InMemoryInsumoAdapter
    adapter = InMemoryInsumoAdapter()
    df = pd.DataFrame({"id_linea": [1, 2], "n_sections": [10, 10]})
    adapter.save_raw(df, "routes_section_id_coords")
    result = adapter.get_raw("routes_section_id_coords")
    assert len(result) == 2


def test_insumo_append_raw_accumulates():
    from urbantrips.storage.adapters.memory.adapters import InMemoryInsumoAdapter
    adapter = InMemoryInsumoAdapter()
    df1 = pd.DataFrame({"id_linea": [1], "n_sections": [10]})
    df2 = pd.DataFrame({"id_linea": [2], "n_sections": [20]})
    adapter.append_raw(df1, "routes_section_id_coords")
    adapter.append_raw(df2, "routes_section_id_coords")
    result = adapter.get_raw("routes_section_id_coords")
    assert len(result) == 2


def test_run_all_signature_accepts_ctx():
    from urbantrips.utils.run_process import run_all
    params = list(inspect.signature(run_all).parameters)
    assert "ctx" in params


def test_procesar_transacciones_accepts_ctx():
    from urbantrips.utils.run_process import procesar_transacciones
    params = list(inspect.signature(procesar_transacciones).parameters)
    assert params[0] == "ctx"


def test_general_adapter_run_exists_and_clear():
    from urbantrips.storage.adapters.memory.adapters import InMemoryGeneralAdapter
    adapter = InMemoryGeneralAdapter()
    assert adapter.run_exists("corrida1") is False
    adapter.register_run("corrida1", "transactions_completed")
    assert adapter.run_exists("corrida1") is True
    adapter.clear_runs()
    assert adapter.run_exists("corrida1") is False


def test_insumo_adapter_has_routes():
    from urbantrips.storage.adapters.memory.adapters import InMemoryInsumoAdapter
    import geopandas as gpd
    from shapely.geometry import LineString
    adapter = InMemoryInsumoAdapter()
    assert adapter.has_routes() is False
    gdf = gpd.GeoDataFrame(
        {"id_linea": [1]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs=4326,
    )
    adapter.save_routes(gdf)
    assert adapter.has_routes() is True
