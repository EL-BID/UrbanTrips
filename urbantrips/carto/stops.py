import os
import pandas as pd
import geopandas as gpd
from shapely import line_interpolate_point
import libpysal
from urbantrips.carto import carto
from urbantrips.geo import geo


def create_temprary_stops_csv_with_node_id(geojson_path):
    """
    Takes a geojson with a LineString for each line and or branch
    and creates a stops dataframe with x, y, order, and node_id
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
        DataFrame containing stops information (x, y, order, and node_id)
        for each line and/or branch and saves it in data directory
    """

    # create stops with order but no node_id
    stops_gdf = create_line_stops_equal_interval(geojson_path)

    # aggregate at node_id
    stops = aggregate_line_stops_to_node_id(stops_gdf)

    data_path = os.path.join("data", "data_ciudad")
    stops.to_csv(os.path.join(data_path,
                 "temporary_stops.csv"), index=False)


def create_line_stops_equal_interval(geojson_path):
    """
    Takes a geojson with a LineString for each line and or branch
    and creates a stops dataframe with x, y, and order

    Parameters
    ----------
    geojson_path : str
        Path to the geojson file containing the LineStrings for each line
        and/or branch and a `stops_distance` attribute with the distance
        in meters

    Returns
    -------
    pandas.DataFrame
        DataFrame containing stops information (x, y, and order)
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
    epsg_m = carto.get_epsg_m()

    geojson_data = geojson_data.to_crs(epsg=epsg_m)

    # Initialize list to store stops data
    stops_data = []

    # Iterate over each LineString in the geojson data
    for i, row in geojson_data.iterrows():

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
    stops_gdf = stops_gdf.reindex(
        columns=['id_linea', 'id_ramal', 'order', 'line_stops_buffer',
                 'x', 'y', 'geometry'])

    return stops_gdf


def aggregate_line_stops_to_node_id(stops_gdf):
    """
    Takes a geojson with stops for each line/branch and aggregates stops
    closed together using `line_stops_buffer` attribute

    Parameters
    ----------
    geopandas.GeoDataFrame
        GeoDataFrame containing stops information (x, y, and order)
        for each line and/or branch and `line_stops_buffer` attribute

    Returns
    -------
    pandas.DataFrame
        DataFrame containing stops information (x, y, order and node_id)
        for each line and/or branch
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
        DataFrame containing stops information (x, y, and order)
        for the given LineString
    """
    epsg_m = carto.get_epsg_m()

    ranges = list(range(0, int(route_geom.length), stops_distance))
    stop_points = line_interpolate_point(route_geom, ranges).tolist()

    stops_df = pd.DataFrame(range(len(stop_points)), columns=['order'])
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

    gdf = line_stops_gdf.copy()
    connectivity = libpysal.weights.fuzzy_contiguity(
        gdf=gdf,
        buffering=True,
        drop=False,
        buffer=buffer,
        predicate='intersects')

    gdf.loc[:, 'node_id'] = connectivity.component_labels

    # geocode new position based on new node_id
    x_new_long = gdf.groupby('node_id').apply(
        lambda df: df.x.mean()).to_dict()
    y_new_long = gdf.groupby('node_id').apply(
        lambda df: df.y.mean()).to_dict()

    gdf.loc[:, 'x_original'] = gdf.x.copy()
    gdf.loc[:, 'y_original'] = gdf.y.copy()

    gdf.loc[:, 'y'] = gdf['node_id'].replace(y_new_long)
    gdf.loc[:, 'x'] = gdf['node_id'].replace(x_new_long)

    cols = ['id_linea', 'id_ramal', 'node_id', 'order', 'x', 'y']
    gdf = gdf.reindex(columns=cols)

    return gdf
