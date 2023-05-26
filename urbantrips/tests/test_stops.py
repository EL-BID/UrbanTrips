import os
from urbantrips.carto import stops
from urbantrips.utils import utils
import pandas as pd


def test_temp_stops():
    configs = utils.leer_configs_generales()
    data_path = os.path.join("data", "data_ciudad",)
    geojson_path = os.path.join(data_path,
                                configs['recorridos_geojson'])

    # Create temporary stops with onde_id
    stops.create_temporary_stops_csv_with_node_id(geojson_path)

    # Check node id for constitucion has several stops within

    temp_stops = pd.read_csv(os.path.join(data_path, 'temporary_stops.csv'))

    # Get line and node_id
    temp_stops = temp_stops.loc[(temp_stops.id_linea == 143) & (
        temp_stops.node_id == 9), :]

    # For Constitucion, line id 143 (linea name 12) should have 4 stops
    # within each branch

    assert len(temp_stops == 8)


def test_create_stops_table():
    # Create basic dir structure:
    utils.create_directories()

    # Create DB
    utils.create_db()

    # Create stops table
    stops.create_stops_table()
    conn = utils.iniciar_conexion_db(tipo='insumos')
    query = f"select * from stops where id_linea = 1"
    line_stops = pd.read_sql(query, conn)
    assert len(line_stops) == 8
