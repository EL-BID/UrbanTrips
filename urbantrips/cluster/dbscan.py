from requests.exceptions import ConnectionError as r_ConnectionError
from PIL import UnidentifiedImageError
from urbantrips.kpi import kpi
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import geopandas as gpd
from sklearn.metrics import silhouette_score
from shapely.geometry import LineString, Point
from shapely import wkt
import re
import contextily as cx
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import seaborn as sns
import os
from urbantrips.utils.utils import iniciar_conexion_db


def get_legs_and_route_geoms(id_linea, rango_hrs, day_type):
    """
    Computes the load per route section.

    Parameters
    ----------
    id_linea : int, list of ints or bool
        route id or list of route ids present in the legs dataset. Route
        section load will be computed for that subset of lines. If False, it
        will run with all routes.
    rango_hrs : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24. Route section
        load will be computed for legs happening within tat time range.
        If False it won't filter by hour.
    day_type: str
        type of day on which the section load is to be computed. It can take
        `weekday`, `weekend` or a specific day in format 'YYYY-MM-DD'

    Returns
    ----------

    legs: pandas DataFrame
        dataframe with all legs for that route, day and time

    route_geom: shapely LineString
        route geom

    """

    # Basic query for legs and route geoms
    q_rec = f"select * from lines_geoms"
    q_main_etapas = f"select * from etapas"
    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    # If line, hour and/or day type, get that subset
    if id_linea:
        if type(id_linea) == int:
            id_linea = [id_linea]

        lineas_str = ",".join(map(str, id_linea))

        id_linea_where = f" where id_linea in ({lineas_str})"
        q_rec = q_rec + id_linea_where
        q_main_etapas = q_main_etapas + id_linea_where

    if rango_hrs:
        rango_hrs_where = (
            f" and hora >= {rango_hrs[0]} and hora <= {rango_hrs[1]}"
        )
        q_main_etapas = q_main_etapas + rango_hrs_where

    day_type_is_a_date = is_date_string(day_type)

    if day_type_is_a_date:
        q_main_etapas = q_main_etapas + f" and dia = '{day_type}'"

    # query legs
    q_etapas = f"""
        select e.*,d.h3_d, f.factor_expansion
        from ({q_main_etapas}) e
        left join destinos d
        on d.id = e.id
        left join factores_expansion f
        on e.dia = f.dia
        and e.id_tarjeta = f.id_tarjeta
    """

    print("Obteniendo datos de etapas y rutas")

    # get data for legs and route geoms
    legs = pd.read_sql(q_etapas, conn_data)
    route_geom = pd.read_sql(q_rec, conn_insumos)

    route_geom["geometry"] = route_geom.wkt.apply(wkt.loads)
    route_geom = route_geom.loc[route_geom.id_linea ==
                                id_linea, "geometry"].item()

    return legs, route_geom


def cluster_legs_lrs(legs, route_geom):
    """
    Cluster legs with DBSCAN using LRS on the route geom

    Parameters
    ----------
    legs : pandas.DataFrame
        A dataframe with legs.
    route_geom: shapely LineString
        A route geom

    Returns
    -------
    tuple
        A tuple of dataframes, with for each direction, with legs
         classified into clusters.

    """
    # Clasiffy direction and lrs over route geom
    legs = classify_legs_into_directions(legs=legs, route_geom=route_geom)

    legs = legs\
        .reindex(columns=['o_proj', 'd_proj', 'sentido', 'factor_expansion'])\
        .groupby(['o_proj', 'd_proj', 'sentido'], as_index=False).sum()

    # direction 0
    X_ida = legs.loc[legs.sentido == 'ida', ['o_proj', 'd_proj']]
    w_ida = legs.loc[legs.sentido == 'ida', 'factor_expansion']
    clustered_legs_d0 = cluster_legs(X_ida, w_ida, type_k='lrs')

    # direction 1
    X_vuelta = legs.loc[legs.sentido == 'vuelta', ['o_proj', 'd_proj']]
    w_vuelta = legs.loc[legs.sentido == 'vuelta', 'factor_expansion']
    clustered_legs_d1 = cluster_legs(X_vuelta, w_vuelta, type_k='lrs')

    return clustered_legs_d0, clustered_legs_d1


def cluster_legs_4d(legs, route_geom, epsg_m=9265):
    """
    Cluster legs with DBSCAN using 4 coordinate points in meters

    Parameters
    ----------
    legs : pandas.DataFrame
        A dataframe with legs.
    route_geom: shapely LineString
        A route geom

    Returns
    -------
    tuple
        A tuple of dataframes, with for each direction, with legs
         classified into clusters.

    """
    legs = classify_legs_into_directions(legs=legs, route_geom=route_geom)

    o_meters = gpd.GeoSeries(legs.o, crs='EPSG:4326').to_crs(epsg=epsg_m)
    d_meters = gpd.GeoSeries(legs.d, crs='EPSG:4326').to_crs(epsg=epsg_m)

    legs = legs.reindex(columns=['o', 'd', 'factor_expansion', 'sentido'])
    legs['o_x_m'] = o_meters.x
    legs['o_y_m'] = o_meters.y
    legs['d_x_m'] = d_meters.x
    legs['d_y_m'] = d_meters.y

    legs = legs\
        .reindex(
            columns=['o_x_m', 'o_y_m', 'd_x_m', 'd_y_m',
                     'sentido', 'factor_expansion']
        )\
        .groupby(
            ['o_x_m', 'o_y_m', 'd_x_m', 'd_y_m', 'sentido'],
            as_index=False).sum()

    X_d0 = legs.loc[legs.sentido == 'ida', [
        'o_x_m', 'o_y_m', 'd_x_m', 'd_y_m']]
    w_d0 = legs.loc[legs.sentido == 'ida', 'factor_expansion']

    X_d1 = legs.loc[legs.sentido == 'vuelta',
                    ['o_x_m', 'o_y_m', 'd_x_m', 'd_y_m']]
    w_d1 = legs.loc[legs.sentido == 'vuelta', 'factor_expansion']

    print('Direction 0')
    clustered_legs_d0 = cluster_legs(X_d0, w_d0, type_k='4d')
    print('Direction 1')
    clustered_legs_d1 = cluster_legs(X_d1, w_d1, type_k='4d')

    return clustered_legs_d0, clustered_legs_d1


def cluster_legs(X, w, type_k='lrs'):
    """
    Classifies legs into clusters

    Parameters
    ----------
    X : pandas DataFrame
        legs data grouped by coordinates (lrs or latlong in meters)
        for a single direction

    w : pandas Series
        weights for the legs data after grouping

    type_k: str
        type of cluster techinque to use.
        * 'lrs': uses a single dimension (Line Reference System) using the
                 positions in the route geom
        * '4d':  uses the 4 data points for origin and destination coordinates
                 in meters

    Returns
    ----------

    X: pandas DataFrame
        grouped legs data classified into clusters

    """

    # set initial benchmarks
    best_num_clusters = 0
    best_num_noise = float('inf')
    best_silhouette_score = -1

    # placeholder for dbscan params
    max_groups_params = None
    max_silhouette_params = None
    min_noise_params = None

    # Create a placeholder for the clustering results
    clusters_table = pd.DataFrame([], index=X.index)

    # set ranges for min samples on from 1% to 50% total legs
    min_samples_range = list(map(int, w.sum() * np.linspace(0.01, .5, 20)))

    # set distance as % of route geom length
    if type_k == 'lrs':
        eps_range = np.linspace(0.01, 0.5, 20)
    elif type_k == '4d':
        eps_range = np.arange(100, 1000, 50)
    else:
        pass

    for eps in eps_range:
        for min_samples in min_samples_range:
            params = (eps, min_samples)

            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(
                X, sample_weight=w)

            labels = dbscan.labels_

            # Subtract 1 if there are any noise points
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Get weighted noise points
            num_noise = pd.DataFrame({'labels': labels, 'w': w})
            num_noise = num_noise.groupby('labels').sum()

            try:
                num_noise = num_noise.loc[-1, 'w']
            except KeyError:
                num_noise = 0

            if num_clusters > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = -1
                num_noise = float('inf')

            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                max_silhouette_params = params

            # Update best parameters and scores
            if num_clusters > best_num_clusters:
                best_num_clusters = num_clusters
                max_groups_params = params

            if num_noise < best_num_noise:
                best_num_noise = num_noise
                min_noise_params = params

    # Print results
    print(
        f"Max number of clusters ({best_num_clusters}) "
        f"found with eps={max_groups_params[0]} and "
        f"min_samples={max_groups_params[1]}")
    print(
        f"Min number of noise points ({best_num_noise}) found with"
        f" eps={min_noise_params[0]} and min_samples={min_noise_params[1]}")
    print(
        f"Max silhouette score ({best_silhouette_score}) found with "
        f"eps={max_silhouette_params[0]} and "
        f"min_samples={max_silhouette_params[1]}")

    params = {
        'max_groups': max_groups_params,
        'max_silhouette': max_silhouette_params,
        'min_noise': min_noise_params,
    }
    for p in params:
        eps, min_samples = params[p]
        # print(eps,min_samples)
        clustering = DBSCAN(eps=eps, min_samples=min_samples)\
            .fit(X, sample_weight=w)

        clusters_table[f'k_{p}'] = reassign_labels(
            labels=clustering.labels_, weights=w)
    X['factor_expansion'] = w
    X = X.join(clusters_table)

    return X


def reassign_labels(labels, weights):
    """
    Reassigns DBSCAN cluster labels based on cluster weight.

    Given an array of DBSCAN cluster labels and an array of weights
    corresponding to each data point, this function reassigns the
    cluster labels so that the cluster with the highest weight is
    assigned the label 0, the next highest is assigned the label 1,
    and so on. The -1 label is left unchanged.

    Parameters
    ----------
    labels : array_like
        An array of DBSCAN cluster labels.
    weights : array_like
        An array of weights corresponding to each data point.

    Returns
    -------
    array_like
        An array of re-assigned cluster labels, with the same shape
        as `labels`.

    Examples
    --------
    >>> labels = np.array([0, 1, 1, 0, -1, 2, 2, 2])
    >>> weights = np.array([0.2, 0.3, 0.1, 0.4, 0.0, 0.2, 0.1, 0.3])
    >>> new_labels = reassign_labels(labels, weights)
    >>> print(new_labels)
    [1 0 0 1 -1 2 2 2]
    """
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        unique_labels = unique_labels[1:]
    sorted_labels = sorted(unique_labels, key=lambda label: np.sum(
        weights[labels == label]), reverse=True)
    new_labels = np.zeros_like(labels)
    for i, label in enumerate(sorted_labels):
        new_labels[labels == label] = i
    new_labels[labels == -1] = -1
    return new_labels


def classify_legs_into_directions(legs, route_geom):
    """
    Classifies legs into directions based on a route geom and LRS

    Parameters
    ----------
    legs : pandas.DataFrame
        A dataframe with legs.
    route_geom: shapely LineString
        A route geom

    Returns
    -------
    pandas.DataFrame
        A legs dataframe with a direction attribute for each leg.

    """

    legs = kpi.add_od_lrs_to_legs_from_route(
        legs_df=legs, route_geom=route_geom)
    legs["sentido"] = [
        "ida" if row.o_proj <= row.d_proj
        else "vuelta" for _, row in legs.iterrows()
    ]
    return legs


def is_date_string(input_str):
    """ Checks a tring inputs for a date format"""
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if pattern.match(input_str):
        return True
    else:
        return False


def plot_cluster_legs_lrs(clustered_legs_d0, clustered_legs_d1,
                          id_linea, rango_hrs, day_type):
    alpha = .5
    cm = plt.get_cmap('tab20')
    colors_list = ['#000000']+[rgb2hex(c) for c in cm.colors]

    colors_dict = {k: rgb2hex(v) for k, v in zip(range(20), cm.colors)}
    colors_dict.update({-1: '#000000'})

    # For each direction
    for direction in [0, 1]:
        if direction == 0:
            direction_str = 'IDA'
            clustered_legs_direction = clustered_legs_d0
        else:
            direction_str = 'VUELTA'
            clustered_legs_direction = clustered_legs_d1

        # plot scatter plot for clusters
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            figsize=(24, 12), ncols=3, nrows=2)

        sns.scatterplot(
            data=clustered_legs_direction.query("k_max_groups>-1"),
            x="o_proj", y="d_proj",
            hue='k_max_groups', alpha=alpha,
            size='factor_expansion', palette="tab20",
            legend=False, ax=ax1)
        sns.scatterplot(
            data=clustered_legs_direction.query("k_max_silhouette>-1"),
            x="o_proj", y="d_proj",
            hue='k_max_silhouette', alpha=alpha,
            size='factor_expansion', palette="tab20",
            legend=False, ax=ax2)
        sns.scatterplot(
            data=clustered_legs_direction.query("k_min_noise>-1"),
            x="o_proj", y="d_proj",
            hue='k_min_noise', alpha=alpha,
            size='factor_expansion', palette="tab20",
            legend=False, ax=ax3)

        hr_str = ' - '.join(map(str, rango_hrs))

        # plot barplot for demand
        data_g = clustered_legs_direction.groupby(
            'k_max_groups').agg(cluster=('factor_expansion', 'sum'))
        data_s = clustered_legs_direction.groupby(
            'k_max_silhouette').agg(cluster=('factor_expansion', 'sum'))
        data_n = clustered_legs_direction.groupby(
            'k_min_noise').agg(cluster=('factor_expansion', 'sum'))

        sns.barplot(x=data_g.index, y=data_g.cluster,
                    palette=colors_list, ax=ax4)
        sns.barplot(x=data_s.index, y=data_s.cluster,
                    palette=colors_list, ax=ax5)
        sns.barplot(x=data_n.index, y=data_n.cluster,
                    palette=colors_list, ax=ax6)

        for ax in [ax1, ax2, ax3]:
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.set_xlabel('Origen LRS')
            ax.set_ylabel('Destino LRS')

        for ax in [ax4, ax5, ax6]:
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.set_ylabel("Etapas")

        route_name = get_route_name(route_id=id_linea)
        title = f"Cluster LRS - {route_name} (id {id_linea}) - " +\
            f"Rango {hr_str} hrs - {day_type} - {direction_str}"

        f.suptitle(title, fontsize=24)

        for frm in ['png', 'pdf']:
            file_name = f'cluster_LRS_id_linea_{id_linea}_{hr_str}' +\
                f'_{day_type}_{direction_str}.{frm}'
            file_path = os.path.join("resultados", frm, file_name)
            f.savefig(file_path, dpi=300)


def create_gdf_clustered_legs_direction(legs, clustered_legs_tuple):

    clusters = pd.concat(clustered_legs_tuple)\
        .reindex(columns=[
            'o_proj', 'd_proj',
            'k_max_groups', 'k_max_silhouette', 'k_min_noise'
        ])
    legs = legs.merge(clusters, on=['o_proj', 'd_proj'], how='left')
    geoms = legs.apply(lambda row: LineString([row.o, row.d]), axis=1)
    legs_gdf = gpd.GeoDataFrame(legs, geometry=geoms, crs='EPSG:4326')
    legs_gdf = legs_gdf\
        .reindex(
            columns=['factor_expansion', 'sentido',
                     'k_max_groups', 'k_max_silhouette', 'k_min_noise',
                     'geometry'])
    return legs_gdf


def plot_cluster_legs_4d(
    clustered_legs_d0, clustered_legs_d1,
        route_geom, epsg_m, id_linea, rango_hrs, day_type, factor):

    alpha = .7
    # For each direction
    route_gs = gpd.GeoSeries(route_geom, crs='EPSG:4326').to_crs(epsg=epsg_m)

    for direction in [0, 1]:
        if direction == 0:
            direction_str = 'IDA'
            clustered_legs_direction = clustered_legs_d0
            route_end = route_gs.item().coords[-1]
            arrow_start = route_gs.geometry.iloc[0].interpolate(
                .95, normalized=True).coords[0]

        else:
            direction_str = 'VUELTA'
            clustered_legs_direction = clustered_legs_d1
            route_end = route_gs.item().coords[0]
            arrow_start = route_gs.geometry.iloc[0].interpolate(
                .05, normalized=True).coords[0]

        cm = plt.get_cmap('tab20')
        colors_list = ['#000000']+[rgb2hex(c) for c in cm.colors]

        colors_dict = {k: rgb2hex(v) for k, v in zip(range(20), cm.colors)}
        colors_dict.update({-1: '#000000'})

        # Create figs and ax
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            figsize=(24, 12), ncols=3, nrows=2)

        # Create geodataframe for each cluster technique
        gdf_max_groups = create_gdf_cluster_technique(
            clustered_legs_direction, technique='k_max_groups',
            factor=factor, epsg_m=epsg_m)
        gdf_max_silhouette = create_gdf_cluster_technique(
            clustered_legs_direction, technique='k_max_silhouette',
            factor=factor, epsg_m=epsg_m)
        gdf_min_noise = create_gdf_cluster_technique(
            clustered_legs_direction, technique='k_min_noise',
            factor=factor, epsg_m=epsg_m)

        # map cluster techiniques
        gdf_max_groups.plot(ax=ax1,
                            color=gdf_max_groups['k_max_groups']
                            .map(colors_dict).fillna("#ffffff"),
                            legend=True, alpha=alpha)

        gdf_max_silhouette.plot(ax=ax2,
                                color=gdf_max_silhouette['k_max_silhouette']
                                .map(colors_dict).fillna("#ffffff"),
                                legend=True, alpha=alpha)

        gdf_min_noise.plot(ax=ax3, color=gdf_min_noise['k_min_noise']
                           .map(colors_dict).fillna("#ffffff"),
                           legend=True, alpha=alpha)

        for ax in [ax1, ax2, ax3]:
            ax.set_axis_off()

            ax.annotate('', xy=(route_end[0],
                                route_end[1]),
                        xytext=(arrow_start[0],
                                arrow_start[1]),
                        arrowprops=dict(facecolor='black',
                                        edgecolor='black'),
                        )
            # plot route geom

            route_gs.plot(ax=ax, color='black')

        prov = cx.providers.Stamen.TonerLite
        crs_string = gdf_max_groups.crs.to_string()
        try:
            cx.add_basemap(ax1, crs=crs_string, source=prov)
        except (UnidentifiedImageError):
            prov = cx.providers.CartoDB.Positron
            cx.add_basemap(ax1, crs=crs_string, source=prov)
        except (r_ConnectionError):
            pass

        cx.add_basemap(ax2, crs=crs_string, source=prov)
        cx.add_basemap(ax3, crs=crs_string, source=prov)

        data_g = clustered_legs_direction.groupby(
            'k_max_groups').agg(cluster=('factor_expansion', 'sum'))
        data_s = clustered_legs_direction.groupby(
            'k_max_silhouette').agg(cluster=('factor_expansion', 'sum'))
        data_n = clustered_legs_direction.groupby(
            'k_min_noise').agg(cluster=('factor_expansion', 'sum'))

        sns.barplot(x=data_g.index, y=data_g.cluster,
                    palette=colors_list, ax=ax4)
        sns.barplot(x=data_s.index, y=data_s.cluster,
                    palette=colors_list, ax=ax5)
        sns.barplot(x=data_n.index, y=data_n.cluster,
                    palette=colors_list, ax=ax6)

        for ax in [ax4, ax5, ax6]:
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.set_ylabel("Etapas")

        hr_str = ' - '.join(map(str, rango_hrs))

        route_name = get_route_name(route_id=id_linea)
        title = f"Cluster 4D - {route_name} (id {id_linea}) - " +\
            f"Rango {hr_str} hrs - {day_type} - {direction_str}"
        f.suptitle(title, fontsize=24)

        for frm in ['png', 'pdf']:
            file_name = f'cluster_4D_id_linea_{id_linea}_{hr_str}' +\
                f'_{day_type}_{direction_str}.{frm}'
            file_path = os.path.join("resultados", frm, file_name)
            f.savefig(file_path, dpi=300)


def create_gdf_cluster_technique(clustered_legs, technique, factor, epsg_m):
    noise_filter = clustered_legs[technique] > -1
    cols = [technique, 'factor_expansion', 'o_x_m', 'o_y_m', 'd_x_m', 'd_y_m']
    clustered_legs_technique = clustered_legs.loc[noise_filter, cols]
    lines = clustered_legs_technique.groupby(technique).apply(
        lambda df:
            np.average(df.loc[:, ['o_x_m', 'o_y_m', 'd_x_m', 'd_y_m']].values,
                       axis=0,
                       weights=df.factor_expansion)
    )

    totals = clustered_legs_technique\
        .groupby(technique, as_index=False)\
        .agg(
            total=('factor_expansion', 'sum'),
        )

    geoms = [LineString([Point([line[0], line[1]]), Point([line[2], line[3]])])
             for line in lines]
    gdf_technique = gpd.GeoDataFrame(
        lines.index, geometry=geoms, crs=f'EPSG:{epsg_m}')
    gdf_technique = gdf_technique.merge(totals)
    gdf_technique.geometry = gdf_technique.geometry.buffer(
        gdf_technique.total * factor)
    return gdf_technique


def get_route_name(route_id):
    """
    Gets the route name based on the route id
    """
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    cur = conn_insumos.cursor()
    q = f"""
    SELECT nombre_linea FROM metadata_lineas
    where id_linea = {route_id}
    """
    cur.execute(q)

    route_name = cur.fetchall()[0][0]

    return route_name
