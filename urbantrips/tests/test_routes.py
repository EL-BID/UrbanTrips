from urbantrips.carto import routes, stops
from urbantrips.utils import utils
import pandas as pd


def test_routes():
    # Create basic dir structure:
    utils.create_directories()
    utils.create_db()
    routes.process_routes_geoms()

    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')

    lines_routes = pd.read_sql(
        "select * from official_lines_geoms", conn_insumos)
    branches_routes = pd.read_sql(
        "select * from official_branches_geoms", conn_insumos)

    assert len(lines_routes) == 2
    assert len(branches_routes) == 8


def test_create_line_g():
    # Create basic dir structure:
    utils.create_directories()
    utils.create_db()
    stops.create_stops_table()
    G = routes.create_line_g(line_id=1)
    assert len(G.nodes) == 4
