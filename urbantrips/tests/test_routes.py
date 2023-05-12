from urbantrips.carto import routes
from urbantrips.utils import utils
import pandas as pd


def test_routes():
    routes.process_routes_geoms()

    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')

    lines_routes = pd.read_sql(
        "select * from official_lines_geoms", conn_insumos)
    branches_routes = pd.read_sql(
        "select * from official_branches_geoms", conn_insumos)

    assert len(lines_routes) == 2
    assert len(branches_routes) == 8
