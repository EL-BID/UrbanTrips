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
    check_amount_new_services = service_ids == [0, 1, 2, 3, 4, 5, 6]
    assert all(check_amount_new_services)

    # check service 2 doesn't get cut when joinin short branch
    service_2 = gps_points_with_new_service_id\
        .loc[gps_points_with_new_service_id.service_o == 2, 'service_id']
    check_service_2 = service_2.value_counts()[1] == 18
    assert (check_service_2)


def test_find_change_in_direction():
    idling_up = pd.Series([5, 5, 5, 6, 7, 8])
    idling_down = pd.Series([5, 5, 5, 4, 3, 2])
    idling_down_up = pd.Series([5, 5, 5, 4, 3, 2, 3, 4, 5])
    idling_down_idling_up = pd.Series([5, 5, 5, 4, 3, 2, 2, 2, 3, 4, 5])

    # test idling and then up no change
    df = pd.DataFrame({'branch_stop_order': idling_up})
    change = services.find_change_in_direction(df)
    assert not change.any()

    # test idling and then down no change
    df = pd.DataFrame({'branch_stop_order': idling_down})
    change = services.find_change_in_direction(df)
    assert not change.any()

    # test idling, then down  and up no idling between
    df = pd.DataFrame({'branch_stop_order': idling_down_up})
    change = services.find_change_in_direction(df)
    # only one change in index 6
    assert change.sum() == 1
    assert change.loc[6]

    # test idling, then down  and up no idling between
    df = pd.DataFrame({'branch_stop_order': idling_down_idling_up})
    change = services.find_change_in_direction(df)
    # only one change in index 6
    assert change.sum() == 1
    assert change.loc[8]
