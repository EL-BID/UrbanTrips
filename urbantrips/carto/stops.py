import os
import pandas as pd
import geopandas as gpd
from shapely import line_interpolate_point
import libpysal
from urbantrips.carto import carto
from urbantrips.geo import geo
from urbantrips.utils.utils import (
    duracion, iniciar_conexion_db, leer_configs_generales)


@duracion
def create_stops_table():
    """
    Reads stops.csv file if present and uploads it to
    stops table in the db
    """
    configs = leer_configs_generales()
    stops_file_name = 'stops.csv'

    if 'nombre_archivo_paradas' in configs:
        if configs['nombre_archivo_paradas'] is not None:
            stops_file_name = configs['nombre_archivo_paradas']

    stops_path = os.path.join("data", "data_ciudad", stops_file_name)

    if os.path.isfile(stops_path):
        stops = pd.read_csv(stops_path)
        upload_stops_table(stops)
    else:
        print("No existe un archivo de stops. Puede utilizar "
              "notebooks/stops_creation_with_node_id_helper.ipynb"
              "para crearlo a partir de los recorridos"
              )


def upload_stops_table(stops):
    """
    Reads a stops table, checks it and uploads it to db
    """
    conn = iniciar_conexion_db(tipo='insumos')
    cols = ['id_linea', 'id_ramal', 'node_id', 'branch_stop_order',
            'stop_x', 'stop_y', 'node_x', 'node_y']
    stops = stops.reindex(columns=cols)
    assert not stops.isna().any().all(), "Hay datos faltantes en stops"

    print("Subiendo paradas a stops")
    stops.to_sql("stops", conn, if_exists="replace", index=False)


def create_temporary_stops_csv_with_node_id(geojson_path):
    """
    Takes a geojson with a LineString for each line and or branch
    and creates a stops dataframe with x, y, branch_stop_order, and node_id
    every [stops_distance] meters.

    Parameters
    ----------
    geojson_path : str
        Path to the geojson file containing the LineStrings for each line
        and/or branch, a `stops_distance` attribute with the distance
        in meters and a `line_stops_buffer` attribute for each line id
        with the distance in meters between stops to be aggregated in a single
        node

    Returns
    -------
    pandas.DataFrame
        DataFrame containing stops information (x, y, branch_stop_order,
        and node_id) for each line and/or branch and saves it in data directory
    """

    # create stops with order but no node_id
    stops_gdf = create_line_stops_equal_interval(geojson_path)

    # aggregate at node_id
    stops_df = aggregate_line_stops_to_node_id(stops_gdf)

    data_path = os.path.join("data", "data_ciudad")
    stops_df.to_csv(os.path.join(data_path,
                                 "temporary_stops.csv"), index=False)


def create_line_stops_equal_interval(geojson_path):
    """
    Takes a geojson with a LineString for each line and or branch
    and creates a stops dataframe with x, y, and branch_stop_order

    Parameters
    ----------
    geojson_path : str
        Path to the geojson file containing the LineStrings for each line
        and/or branch and a `stops_distance` attribute with the distance
        in meters

    Returns
    -------
    pandas.DataFrame
        DataFrame containing stops information (x, y, and branch_stop_order)
        for each line and/or branch
    """
    # Read geojson
    geojson_data = gpd.read_file(geojson_path)

    # Check geometry type
    geo.check_all_geoms_linestring(geojson_data)

    # if there is no branch_id create
    if 'id_ramal' not in geojson_data.columns:
        geojson_data['id_ramal'] = None

    # Project in meters
    epsg_m = geo.get_epsg_m()

    geojson_data = geojson_data.to_crs(epsg=epsg_m)
    stops_gdf = interpolate_stops_every_x_meters(geojson_data)

    stops_gdf = stops_gdf.reindex(
        columns=['id_linea', 'id_ramal', 'branch_stop_order',
                 'line_stops_buffer', 'x', 'y', 'geometry'])

    stops_gdf = stops_gdf.to_crs(epsg=4326)
    return stops_gdf


def interpolate_stops_every_x_meters(gdf):
    """
    Takes a gdf in proyected crs in meters with linestrings and
    interpolates points every x meters returning a data frame with
    those points
    """
    # Initialize list to store stops data
    stops_data = []

    # Iterate over each LineString in the geojson data
    for i, row in gdf.iterrows():

        # Create stops for the LineString
        stops_distance = row.stops_distance
        route_geom = row.geometry
        line_stops_buffer = row.line_stops_buffer

        line_stops_data = create_stops_from_route_geom(
            route_geom=route_geom,
            stops_distance=stops_distance
        )

        # Add line_id to the stops data
        line_stops_data['id_linea'] = row.id_linea
        line_stops_data['id_ramal'] = row.id_ramal
        line_stops_data['line_stops_buffer'] = line_stops_buffer

        # Add the stops data to the overall stops data list
        stops_data.append(line_stops_data)

    # Concatenate the stops data for all lines and return as a DataFrame
    stops_gdf = pd.concat(stops_data, ignore_index=True)
    return stops_gdf


def aggregate_line_stops_to_node_id(stops_gdf):
    """
    Takes a geojson with stops for each line/branch and aggregates stops
    closed together using `line_stops_buffer` attribute

    Parameters
    ----------
    geopandas.GeoDataFrame
        GeoDataFrame containing a branch_stop_order
        for each line and/or branch and `line_stops_buffer` attribute

    Returns
    -------
    pandas.DataFrame
        DataFrame containing stops information (x, y, branch_stop_order
        and node_id) for each line and/or branch
    """

    # Add node_id for each line
    stops_df = stops_gdf\
        .groupby('id_linea', as_index=False)\
        .apply(create_node_id)\
        .reset_index(drop=True)

    return stops_df


def create_stops_from_route_geom(route_geom, stops_distance):
    """
    Takes a LineString projected in meters and interpolates stops
    every x meters

    Parameters
    ----------
    route_geom : shapely.LineString
        LineString for which to create stops
    stops_distance : int
        Distance in meters between stops

    Returns
    -------
    pandas.DataFrame
        DataFrame containing stops information (x, y, and branch_stop_order)
        for the given LineString
    """
    epsg_m = geo.get_epsg_m()

    ranges = list(range(0, int(route_geom.length), stops_distance))
    stop_points = line_interpolate_point(route_geom, ranges).tolist()

    stops_df = pd.DataFrame(range(len(stop_points)),
                            columns=['branch_stop_order'])
    stops_df = gpd.GeoDataFrame(
        stops_df, geometry=stop_points,
        crs=f"EPSG:{epsg_m}")

    geom_wgs84 = stops_df.geometry.to_crs(epsg=4326)
    stops_df['x'] = geom_wgs84.x
    stops_df['y'] = geom_wgs84.y

    return stops_df


def create_node_id(line_stops_gdf):
    """
    Adds a node_id column to the given DataFrame based on the x, y,
    and order columns using fuzzy contiguity.

    Parameters
    ----------
    df : pandas.DataFrame
        line stops DataFrame to add node_id column.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with an additional node_id column
    """
    buffer = line_stops_gdf.line_stops_buffer.unique()[0]
    epsg_m = geo.get_epsg_m()

    line_stops_gdf = line_stops_gdf.to_crs(epsg=epsg_m)

    gdf = line_stops_gdf.copy()
    connectivity = libpysal.weights.fuzzy_contiguity(
        gdf=gdf,
        buffering=True,
        drop=False,
        buffer=buffer,
        predicate='intersects')

    gdf.loc[:, 'node_id'] = connectivity.component_labels
    gdf = gdf.to_crs(epsg=4326)

    # geocode new position based on new node_id
    gdf.loc[:, ['stop_x']] = gdf.geometry.x
    gdf.loc[:, ['stop_y']] = gdf.geometry.y

    x_new_long = gdf.groupby('node_id').apply(
        lambda df: df.stop_x.mean()).to_dict()
    y_new_long = gdf.groupby('node_id').apply(
        lambda df: df.stop_y.mean()).to_dict()

    gdf.loc[:, 'node_y'] = gdf['node_id'].replace(y_new_long)
    gdf.loc[:, 'node_x'] = gdf['node_id'].replace(x_new_long)

    cols = ['id_linea', 'id_ramal', 'node_id',
            'branch_stop_order', 'stop_x', 'stop_y', 'node_x', 'node_y']
    gdf = gdf.reindex(columns=cols)

    return gdf
