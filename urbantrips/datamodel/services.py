import numpy as np
import pandas as pd
import geopandas as gpd
from math import ceil
import h3
from urbantrips.utils import utils
from urbantrips.kpi import kpi
from urbantrips.geo import geo


def process_services():
    """
    Download unprocessed gps data and classify them into services
    """
    print("Descargando paradas y puntos gps")
    gps_points, stops = get_stops_and_gps_data()

    if gps_points is None:
        print("Todos los puntos gps ya fueron procesados en servicios ")
    else:
        print("Clasificando puntos gps en servicios")
        gps_points.groupby('id_linea').apply(
            process_line_services, stops=stops)


def get_stops_and_gps_data():
    """
    Download unprocessed gps data
    """
    configs = utils.leer_configs_generales()

    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')
    conn_data = utils.iniciar_conexion_db(tipo='data')

    # check if data is present in the db
    if not utils.check_table_in_db(table_name='gps', tipo_db='data'):
        print("No existe la tabla gps. Asegurese de tener datos gps y ")
        print("correr datamodel.transactions.process_and_upload_gps_table()")
    else:
        cur = conn_data.cursor()
        q = "select count(*) from gps;"
        records = cur.execute(q).fetchall()[0][0]
        if records == 0:
            print("La tabla gps no tiene registros.Asegurese de tetener datos")
            print("gps y correr")
            print("datamodel.transactions.process_and_upload_gps_table()")

    # check if data is present in the db
    if not utils.check_table_in_db(table_name='gps', tipo_db='data'):
        print("No hay tabla services_stats. Asegurese de tener datos gps ")
        print("y correr datamodel.transactions.process_and_upload_gps_table()")

    gps_query = """
        select g.*
        from gps g
        left join services_stats ss
        on g.id_linea = ss.id_linea
        and g.dia = ss.dia
        where ss.id_linea is null
    """
    gps_points = pd.read_sql(gps_query, conn_data)

    gps_points = gpd.GeoDataFrame(
        gps_points,
        geometry=gpd.GeoSeries.from_xy(
            x=gps_points.longitud, y=gps_points.latitud, crs='EPSG:4326'),
        crs='EPSG:4326'
    )

    gps_lines = ','.join(map(str, gps_points.id_linea.unique()))

    if len(gps_lines) == 0:
        return None, None

    else:
        # check if data is present in the db
        if not utils.check_table_in_db(table_name='stops', tipo_db='insumos'):
            print("No existe la tabla stops. Asegurese de tener datos de ")
            print("stops y correr carto.stops.create_stops_table()")
        else:
            cur = conn_data.cursor()
            q = "select count(*) from gps;"
            records = cur.execute(q).fetchall()[0][0]
            if records == 0:
                print("La tabla stops no tiene registros. Asegurese de tener")
                print("datos de stops y correr")
                print("carto.stops.create_stops_table()")

        stops_query = f"""
            select *
            from stops
            where id_linea in ({gps_lines})
        """

        stops = pd.read_sql(stops_query, conn_insumos)

        # use only nodes as stops
        stops = stops.drop_duplicates(
            subset=['id_linea', 'id_ramal', 'node_id'])

        stops = gpd.GeoDataFrame(
            stops,
            geometry=gpd.GeoSeries.from_xy(
                x=stops.node_x, y=stops.node_y, crs='EPSG:4326'),
            crs='EPSG:4326'
        )
        gps_points = gps_points.to_crs(epsg=configs['epsg_m'])
        stops = stops.to_crs(epsg=configs['epsg_m'])

        return gps_points, stops


def process_line_services(gps_points, stops):
    """
    Takes gps points and stops for a given line
    and classifies each point into a services
    and produces services tables and daily stats
    for that line

    Parameters
    ----------
    gps_points : geopandas.GeoDataFrame
        GeoDataFrame with gps points for a given line

    stops : geopandas.GeoDataFrame
        GeoDataFrame with stops for a given line
    Returns
    -------
    pandas.DataFrame
        DataFrame containing statistics for services by line and day
    """
    line_id = gps_points.id_linea.unique()[0]

    conn_data = utils.iniciar_conexion_db(tipo='data')
    print(f"Procesando servicios en base a gps para id_linea {line_id}")

    # select only stops for that line
    line_stops_gdf = stops.loc[stops.id_linea == line_id, :]

    print("Asignando servicios")
    gps_points_with_new_service_id = gps_points\
        .groupby(['dia', 'interno'], as_index=False)\
        .apply(classify_line_gps_points_into_services,
               line_stops_gdf=line_stops_gdf)\
        .droplevel(0)

    print("Subiendo servicios a la db")
    # save result to services table
    services_gps_points = gps_points_with_new_service_id\
        .reindex(columns=['id', 'original_service_id', 'new_service_id',
                          'service_id', 'id_ramal_gps_point', 'node_id'])
    services_gps_points.to_sql("services_gps_points",
                               conn_data, if_exists='append', index=False)

    print("Creando tabla de servicios")
    # process services gps points into services table
    line_services = create_line_services_table(
        gps_points_with_new_service_id)
    line_services.to_sql("services", conn_data,
                         if_exists='append', index=False)

    print("Creando estadisticos de servicios")
    # create stats for each line and day
    stats = line_services\
        .groupby(['id_linea', 'dia'], as_index=False)\
        .apply(compute_new_services_stats)

    stats.to_sql("services_stats", conn_data, if_exists='append', index=False)
    return stats


def create_line_services_table(line_day_gps_points):
    # get  basic stats for each service
    line_services = line_day_gps_points\
        .groupby(['id_linea', 'dia', 'interno',
                  'original_service_id', 'service_id'], as_index=False)\
        .agg(
            is_idling=('idling', 'sum'),
            total_points=('idling', 'count'),
            distance_km=('distance_km', 'sum'),
            min_ts=('fecha', 'min'),
            max_ts=('fecha', 'max'),
        )

    line_services.loc[:, ['min_datetime']] = line_services.min_ts.map(
        lambda ts: str(pd.Timestamp(ts, unit='s')))
    line_services.loc[:, ['max_datetime']] = line_services.max_ts.map(
        lambda ts: str(pd.Timestamp(ts, unit='s')))

    # compute idling proportion for each service
    line_services['prop_idling'] = line_services.is_idling / \
        line_services['total_points']
    line_services = line_services.drop(['is_idling'], axis=1)

    # stablish valid services
    line_services['valid'] = (line_services.prop_idling < .5) & (
        line_services.total_points > 5)

    return line_services


def infer_service_id_stops(line_gps_points, line_stops_gdf, debug=False):
    """
    Takes gps points and stops for a given line and classifies each point into
    services whenever the order of passage across stops switches from
    increasing to decreasing order in the majority of active branches in that
    line.

    Parameters
    ----------
    line_gps_points : geopandas.GeoDataFrame
        GeoDataFrame with gps points for a given line

    line_stops_gdf : geopandas.GeoDataFrame
        GeoDataFrame with stops for a given line

    debug: bool
        If the attributes concerning services classification
        should be added

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame points classified into services
    """

    n_original_services_ids = len(
        line_gps_points['original_service_id'].unique())

    branches = line_stops_gdf.id_ramal.unique()

    n_original_services_ids = len(
        line_gps_points['original_service_id'].unique())

    majority_by_node_id = line_stops_gdf\
        .drop_duplicates(['id_ramal', 'node_id'])\
        .groupby('node_id', as_index=False)\
        .agg(branch_mayority=('id_ramal', 'count'))

    gps_all_branches = pd.DataFrame()
    debug_df = pd.DataFrame()

    for branch in branches:
        stops_to_join = line_stops_gdf\
            .loc[line_stops_gdf.id_ramal == branch,
                 ['branch_stop_order', 'geometry']]

        # Not use max_distance. Far away stops will appear as
        # still on the same stop and wont be active branches
        gps_branch = gpd.sjoin_nearest(
            line_gps_points,
            stops_to_join,
            how='left',
            max_distance=1500,
            distance_col=f'distance_to_stop')
        gps_branch['id_ramal'] = branch

        # Evaluate change on stops order for each branch
        temp_change = gps_branch\
            .groupby(['interno', 'original_service_id'])\
            .apply(find_change_in_direction)

        if n_original_services_ids > 1:
            # when vehicle is always too far away from this branch

            if isinstance(temp_change, type(pd.Series())):
                temp_change = temp_change.droplevel([0, 1])
            else:
                temp_change = False

        # when there is only one original service per vehicle
        else:
            temp_change = pd.Series(
                temp_change.iloc[0].values, index=temp_change.columns)

        gps_branch[f'temp_change'] = temp_change

        window = 5
        gps_branch['consistent_post'] = (
            gps_branch[f'temp_change']
            .shift(-window).fillna(False)
            .rolling(window=window, center=False, min_periods=3).sum() == 0
        )
        # gps_branch[f'consistent_pre'] = (
        #    gps_branch[f'temp_change']
        #    .fillna(False).shift(1) == False)  # noqa

        # Accept there is a change in direction when consistent
        gps_branch[f'change'] = (
            gps_branch[f'temp_change'] &
            gps_branch[f'consistent_post']
            # & gps_branch[f'consistent_pre']
        )
        if debug:
            debug_branch = gps_branch\
                .reindex(columns=['id', 'branch_stop_order', 'id_ramal',
                                  'temp_change', 'consistent_post',
                                  'consistent_pre', 'change'])

            debug_df = pd.concat([debug_df, debug_branch])

        gps_branch = gps_branch.drop(
            ['index_right', 'temp_change', 'consistent_post',
             # 'consistent_pre'
             ], axis=1)
        gps_all_branches = pd.concat([gps_all_branches, gps_branch])

    # para cada punto gps necesito el node id del branch mas cercano
    branches_distances_table = gps_all_branches\
        .reindex(columns=['id', 'id_ramal', 'distance_to_stop',
                          'branch_stop_order'])\
        .sort_values(['id', 'distance_to_stop'])\
        .drop_duplicates(subset=['id'], keep='first')\
        .drop(['distance_to_stop'], axis=1)

    gps_node_ids = branches_distances_table\
        .merge(
            line_stops_gdf.reindex(
                columns=['node_id', 'branch_stop_order', 'id_ramal']),
            on=['id_ramal', 'branch_stop_order'],
            how='left')\
        .reindex(columns=['id', 'id_ramal', 'node_id'])

    total_changes_by_gps = gps_all_branches\
        .groupby(['id'], as_index=False)\
        .agg(total_changes=('change', 'sum'))

    gps_points_changes = gps_node_ids\
        .merge(total_changes_by_gps, how='left', on='id')\
        .merge(majority_by_node_id, how='left', on='node_id')
    gps_points_changes['change'] = (
        gps_points_changes.total_changes >= gps_points_changes.branch_mayority
    )
    gps_points_changes = gps_points_changes.reindex(
        columns=['id', 'id_ramal', 'node_id', 'change'])

    gps_points_changes = gps_points_changes\
        .rename(columns={'id_ramal': 'id_ramal_gps_point'})

    line_gps_points = line_gps_points.merge(
        gps_points_changes, on='id', how='left')

    if n_original_services_ids > 1:

        # Within each original service id, classify services within
        new_services_ids = line_gps_points\
            .groupby('original_service_id')\
            .apply(lambda df: df['change'].cumsum().fillna(method='ffill'))\
            .droplevel(0)
    else:
        new_services_ids = line_gps_points\
            .groupby('original_service_id')\
            .apply(lambda df: df['change'].cumsum().fillna(method='ffill'))

        new_services_ids = pd.Series(
            new_services_ids.iloc[0].values, index=new_services_ids.columns)

    line_gps_points['new_service_id'] = new_services_ids

    if debug:
        debug_df = debug_df\
            .pivot(index="id", columns='id_ramal',
                   values=["branch_stop_order", "temp_change",
                           'consistent_post', 'consistent_pre', 'change'])\
            .reset_index()

        cols = [c[0]+'_'+str(c[1]) if c[0] != 'id'
                else c[0] for c in debug_df.columns]

        debug_df.columns = cols
        line_gps_points = line_gps_points.merge(debug_df, on='id', how='left')

    return line_gps_points


def classify_line_gps_points_into_services(line_gps_points, line_stops_gdf,
                                           *args, **kwargs):
    """
    Takes gps points and stops for a given line and classifies each point into
    services based on original gps data or infered basd on stops whenever the
    order of passage across stops switches from increasing to decreasing order
    in the majority of active branches in that line.

    Parameters
    ----------
    line_gps_points : geopandas.GeoDataFrame
        GeoDataFrame with gps points for a given line

    line_stops_gdf : geopandas.GeoDataFrame
        GeoDataFrame with stops for a given line
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame points classified into services
    """
    # create original service id
    original_service_id = line_gps_points\
        .reindex(columns=['dia', 'interno', 'service_type'])\
        .groupby(['dia', 'interno'])\
        .apply(create_original_service_id)
    original_service_id = original_service_id.service_type
    original_service_id = original_service_id.droplevel([0, 1])
    line_gps_points.loc[:, ['original_service_id']] = original_service_id

    # check configs if trust in service type gps
    configs = utils.leer_configs_generales()

    if 'utilizar_servicios_gps' in configs:
        trust_service_type_gps = configs['utilizar_servicios_gps']

    else:
        trust_service_type_gps = False

    if trust_service_type_gps:
        # classify services based on gps dataset attribute
        line_gps_points['new_service_id'] = (
            line_gps_points['original_service_id'].copy()
        )
    else:
        # classify services based on stops
        line_gps_points = infer_service_id_stops(
            line_gps_points, line_stops_gdf)

    # Classify idling points when there is no movement
    line_gps_points['idling'] = line_gps_points.distance_km < 0.1

    # create a unique id from both old and new
    new_ids = line_gps_points\
        .reindex(columns=['original_service_id', 'new_service_id'])\
        .drop_duplicates()
    new_ids['service_id'] = range(len(new_ids))

    line_gps_points = line_gps_points\
        .merge(new_ids, how='left',
               on=['original_service_id', 'new_service_id'])

    return line_gps_points


def compute_new_services_stats(line_day_services):
    """
    Takes a gps tracking points for a line in a given day
    with service id and computes stats for services

    Parameters
    ----------
    df : pandas.DataFrame
        line_day_services stats table for a given day

    Returns
    -------
    pandas.DataFrame
        DataFrame with stats for each line and day
    """
    id_linea = line_day_services.id_linea.unique()
    dia = line_day_services.dia.unique()

    n_original_services = line_day_services\
        .drop_duplicates(subset=['interno', 'original_service_id'])\
        .shape[0]

    n_new_services = len(line_day_services)
    n_new_valid_services = line_day_services.valid.sum()
    n_services_short = (line_day_services.total_points <= 5).sum()

    prop_short_idling = ((line_day_services.prop_idling >= .5) & (
        line_day_services.total_points <= 5)).sum() / n_services_short

    original_services_distance = round(line_day_services.distance_km.sum())
    new_services_distance = round(line_day_services
                                  .loc[line_day_services['valid'],
                                       'distance_km']
                                  .sum() / original_services_distance, 2)

    sub_services = line_day_services\
        .loc[line_day_services['valid'], :]\
        .groupby(['interno', 'original_service_id'])\
        .apply(lambda df: len(df.service_id.unique()))

    if len(sub_services):
        sub_services = sub_services.value_counts(normalize=True)

        if 1 in sub_services.index:
            original_service_no_change = round(sub_services[1], 2)
        else:
            original_service_no_change = 0
    else:
        original_service_no_change = None

    day_line_stats = pd.DataFrame({
        'id_linea': id_linea,
        'dia': dia,
        'cant_servicios_originales': n_original_services,
        'cant_servicios_nuevos': n_new_services,
        'cant_servicios_nuevos_validos': n_new_valid_services,
        'n_servicios_nuevos_cortos': n_services_short,
        'prop_servicos_cortos_nuevos_idling': prop_short_idling,
        'distancia_recorrida_original': original_services_distance,
        'prop_distancia_recuperada': new_services_distance,
        'servicios_originales_sin_dividir': original_service_no_change,
    }, index=[0])
    return day_line_stats


def find_change_in_direction(df):
    # Create a new series with the differences between consecutive elements
    series = df['branch_stop_order'].copy()

    # check diference against previous stop
    diff_series = series.diff().dropna()

    # select only where change happens
    changes_in_series = diff_series.loc[diff_series != 0]

    # checks for change in a decreasing manner
    decreasing_change = changes_in_series.map(lambda x: x < 0)

    decreasing_to_increasing = decreasing_change.diff().fillna(False)

    return decreasing_to_increasing


def create_original_service_id(service_type_series):
    return (service_type_series == 'start_service').cumsum()


def delete_services_data(id_linea):
    "this function deletes data for a given line in servicies tables"

    conn_data = utils.iniciar_conexion_db(tipo='data')
    print(f"Borrando datos en tablas de servicios para id linea {id_linea};")
    conn_data.execute(f"DELETE FROM services where id_linea = {id_linea};")
    conn_data.execute(
        f"DELETE FROM services_stats where id_linea = {id_linea};")
    query = f"""
    DELETE FROM services_gps_points
    where id in (
        select id
        from gps
        where id_linea = {id_linea}
    );
    """
    conn_data.execute(query)
    conn_data.commit()

    print("Servicios borrados")
