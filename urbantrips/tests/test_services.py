import os
from urbantrips.datamodel import services
from urbantrips.utils import utils
import pandas as pd
import pytest
import geopandas as gpd


@pytest.fixture
def path_test_data():
    path = os.path.join(os.getcwd(), "urbantrips", "tests", "data")
    return path


@pytest.fixture
def gps_points_test_data(path_test_data):
    path = os.path.join(path_test_data, "service_id_gps_test.csv")
    df = pd.read_csv(path, dtype={"id_linea": int})
    return df


@pytest.fixture
def stops_test_data(path_test_data):
    path = os.path.join(path_test_data, "service_id_stops_test.csv")
    df = pd.read_csv(path, dtype={"id_linea": int})
    return df


def test_service_id(gps_points_test_data, stops_test_data):

    configs = utils.leer_configs_generales()

    gps_points = gps_points_test_data
    stops = stops_test_data
    stops = stops.drop_duplicates(subset=['id_linea', 'id_ramal', 'node_id'])

    gps_points = gpd.GeoDataFrame(gps_points,
                                  geometry=gpd.GeoSeries.from_xy(
                                      x=gps_points.longitud,
                                      y=gps_points.latitud,
                                      crs='EPSG:4326'),
                                  crs='EPSG:4326'
                                  )

    stops = gpd.GeoDataFrame(stops,
                             geometry=gpd.GeoSeries.from_xy(
                                 x=stops.node_x, y=stops.node_y,
                                 crs='EPSG:4326'),
                             crs='EPSG:4326'
                             )

    gps_points = gps_points.to_crs(epsg=configs['epsg_m'])
    gps_points['distance_km'] = 0.2

    stops = stops.to_crs(epsg=configs['epsg_m'])

    gps_points_with_new_service_id = gps_points\
        .groupby(['dia', 'interno'], as_index=False)\
        .apply(services.classify_line_gps_points_into_services,
               line_stops_gdf=stops)

    gps_points_with_new_service_id = gps_points_with_new_service_id.droplevel(
        0)

    service_ids = gps_points_with_new_service_id.service_id.unique()
    assert all(service_ids == [0, 1, 2, 3, 4, 5, 6, 7])
    assert gps_points_with_new_service_id.loc[
        gps_points_with_new_service_id.service_o == 99,
        'service_id'].unique() == 4
