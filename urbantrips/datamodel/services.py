import pandas as pd
import geopandas as gpd
from math import ceil
import h3
from urbantrips.utils import utils
from urbantrips.kpi import kpi
from urbantrips.geo import geo


def classify_gps_points_into_services(line_gps_points, line_stops_gdf, *args, **kwargs):
    """
    Takes a gps tracking points from a line and
    a interpolated points over line's route geoms
    and classifies every gps point into a service within declared services

    Parameters
    ----------
    line_gps_points : GeoPandas.GeoDataFrame
        gps tracking points from a line
    line_stops_gdf : GeoPandas.GeoDataFrame
        points interpolated over the line's route geom


    Returns
    -------
    geopandas.geoDataFrame
        GeoDataFrame with gps tracking points classified in new service id
        within original ones
    """
    line_id = int(line_gps_points.id_linea.unique().item())
    print(f"Procesando servicios en base a gps para id_linea {line_id}")

    line_gps_points = line_gps_points.sort_values(['interno', 'fecha'])

    # Create a service id based on original gps tracking data
    line_gps_points['original_service_id'] = (
        line_gps_points['service_type'] == 'start_service').cumsum()

    n_original_services_ids = len(
        line_gps_points['original_service_id'].unique())

    # select only stops for that line
    line_stops_gdf = line_stops_gdf.loc[line_stops_gdf.id_linea == line_id, :]

    # Set branches from routes and majority criteria for service detection
    branches = line_stops_gdf.id_ramal.unique()

    check_branches_consistency = pd.Series(
        line_gps_points.id_ramal.unique()).isin(branches).all()
    mssg_text = "No todos los ramales en gps estan presentes "\
        + "en la cartografia de recorridos"

    assert check_branches_consistency, mssg_text

    majority = ceil(len(branches) / 2)

    # For each branch
    for branch in branches:
        print("Procesando ramal", branch)

        # assign a stop to each gps point
        stops_to_join = line_stops_gdf\
            .loc[line_stops_gdf.id_ramal == branch, ['order', 'geometry']]
        stops_to_join = stops_to_join.rename(
            columns={'order': f'order_{branch}'})

        line_gps_points = gpd.sjoin_nearest(
            line_gps_points,
            stops_to_join,
            how='left',
            # max_distance= 1000,
            lsuffix='gps',
            rsuffix=str(branch),
            distance_col=f'distance_to_stop_{branch}')

        # Check if there is a change in direction using line stops order,
        # within each vehicle and original service

        # If df is too short, temp change may be len1, so index becames the col
        temp_change = line_gps_points\
            .groupby(['interno', 'original_service_id'])\
            .apply(find_change_in_direction, branch=branch)

        if n_original_services_ids > 1:
            temp_change = temp_change.droplevel([0, 1])
        else:

            temp_change = pd.Series(
                temp_change.iloc[0].values, index=temp_change.columns)

        line_gps_points[f'temp_change_{branch}'] = temp_change

        # Check if change is consistent over the next 5 gps positions
        window = 5
        line_gps_points[f'consistent_{branch}'] = line_gps_points[f'temp_change_{branch}']\
            .shift(-5).rolling(window=window, center=False, min_periods=3).sum() == 0

        # Accept there is a change in direction when consistent
        line_gps_points[f'change_{branch}'] = line_gps_points[f'temp_change_{branch}'] & line_gps_points[f'consistent_{branch}']

    # Accept change when a majority of branches registers one
    line_gps_points['change'] = line_gps_points\
        .reindex(columns=[f'change_{branch}' for branch in branches])\
        .sum(axis=1) >= majority

    # Classify idling points when there is no movement in most branches
    line_gps_points['idling'] = line_gps_points\
        .reindex(columns=[f'temp_change_{branch}' for branch in branches])\
        .isna().sum(axis=1) >= majority

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

    line_gps_points = line_gps_points.reindex(columns=[
        'id', 'id_linea', 'id_ramal', 'interno', 'fecha', 'latitud', 'longitud',
        'direction', 'velocity',  'dia', 'tiempo',
        'geometry', 'change', 'idling', 'service_type', 'cum_distance', 'original_service_id', 'new_service_id'])

    # add service id using original or a unique id based on new and original
    configs = utils.leer_configs_generales()
    confiar_service_type_gps = configs['confiar_service_type_gps']

    if confiar_service_type_gps:
        line_gps_points['service_id'] = line_gps_points['original_service_id'].copy()
    else:
        service_id_df = line_gps_points\
            .reindex(columns=['original_service_id', 'new_service_id'])\
            .drop_duplicates()
        service_id_df['service_id'] = range(len(service_id_df))
        line_gps_points = line_gps_points\
            .merge(service_id_df, how='left', on=['original_service_id', 'new_service_id'])

    valid_services_df = line_gps_points\
        .reindex(columns=['service_id', 'idling'])\
        .groupby(['service_id'])\
        .agg(is_idling=('idling', 'sum'), total_points_new_service=('idling', 'count'))

    valid_services_df['prop_idling'] = valid_services_df.is_idling / \
        valid_services_df['total_points_new_service']

    # stablish valid services
    valid_services = (valid_services_df.prop_idling < .5) & (
        valid_services_df.total_points_new_service > 4)
    valid_services_df.loc[:, ['valid_service']] = valid_services
    valid_services_df = valid_services_df.reset_index().reindex(
        columns=['service_id', 'valid_service'])

    line_gps_points = line_gps_points.merge(
        valid_services_df, on=['service_id'], how='left')

    # BORRAR DE ACA EN MAS
    # conn = sqlite3.connect('/home/pipe/proyectos/urbantrips/desarrollo/notebooks/servicios.sqlite')
    # valid_services_df['id_linea'] = line_id
    # valid_services_df.to_sql("valid_services", conn,if_exists="append",index=False,)

    return line_gps_points


def compute_service_stats(gdf):
    """
    Takes a gps tracking points for a line with service id
    and computes stats for services

    Parameters
    ----------
    gdf : GeoPandas.GeoDataFrame
        gps tracking points with service id

    Returns
    -------
    pandas.DataFrame
        DataFrame with stats for each line and day
    """

    line_id = gdf.id_linea.unique().item()
    dia = gdf.dia.unique().item()
    print(f"Procesando service stats para id_linea {line_id} dia {dia}")

    # Estimate valid services and idling points
    idling_points = gdf.idling.sum() / len(gdf) * 100
    # print("{:.1f} % de puntos en la linea estan detenidos".format(idling_points))

    # New valid services
    amount_new_valid_servicies = len(
        gdf.loc[gdf.valid_service, 'service_id'].unique())
    # print("Cantidad de servicios nuevos validos:",amount_new_valid_servicies)

    # if there is no cumulative distance
    no_cum_distance = gdf['cum_distance'].isna().all()

    if no_cum_distance:
        res = 11
        distancia_entre_hex = h3.edge_length(resolution=res, unit="km")
        distancia_entre_hex = distancia_entre_hex * 2

        # Compute original gps tracking distance
        gdf["h3"] = gdf.apply(geo.h3_from_row, axis=1,
                              args=(res, "latitud", "longitud"))

        gdf["h3_lag"] = (
            gdf.reindex(columns=["dia", "id_linea", "interno", "h3"])
            .groupby(["dia", "id_linea", "interno"])
            .shift(-1)
        )

        gdf = gdf.dropna(subset=["h3", "h3_lag"])
        gps_dict = gdf.to_dict("records")
        dist_h3 = pd.Series(map(kpi.distancia_h3, gps_dict))
        gdf['h3_distance'] = dist_h3 * distancia_entre_hex
        del gps_dict

        original_distance = gdf['h3_distance'].sum()

        distance_valid_services = gdf\
            .loc[gdf['valid_service'], 'h3_distance'].sum()
    else:
        # use original distance in meters turn into km
        original_distance = gdf.groupby('original_service_id').apply(
            lambda df: df.iloc[-1])['cum_distance'].sum()/1000
        # use original distance in meters turn into km

        distance_valid_services = gdf\
            .loc[gdf['valid_service'], :]\
            .groupby('original_service_id')\
            .apply(lambda df: df.iloc[-1])['cum_distance'].sum()/1000

    # print('Distancia original recorrida por los servicios',original_distance)

    # Compute amount of services
    amount_original_servicies = len(gdf.original_service_id.unique())
    # print("Cantidad de servicios originales:",amount_original_servicies)

    amount_new_total_servicies = len(gdf.service_id.unique())
    # print("Cantidad de servicios nuevos totales:",amount_new_total_servicies)

    # print('Distancia recorrida por servicios validos',distance_valid_services)

    sub_services = gdf\
        .loc[gdf['valid_service'], :]\
        .groupby(['original_service_id']).apply(lambda df: len(df.service_id.unique()))

    if len(sub_services):
        sub_services = sub_services.value_counts(normalize=True)

        if 1 in sub_services.index:
            original_service_no_change = sub_services[1]
        else:
            original_service_no_change = 0
    else:
        original_service_no_change = None

    # compute amount of tack points within each valid service
    points_within_new_services = gdf\
        .loc[gdf['valid_service'], :]\
        .groupby(['service_id']).size().describe()

    stats = pd.DataFrame({
        # 'id_linea':line_id,
        # 'dia':,
        'cant_servicios_originales': amount_original_servicies,
        'cant_servicios_nuevos': amount_new_total_servicies,
        'cant_servicios_nuevos_validos': amount_new_valid_servicies,
        'porcentaje_detencion': idling_points,
        'distancia_recorrida_original': original_distance,
        'distancia_recorrida_validos': distance_valid_services,
        'prop_distancia_recuperada': distance_valid_services/original_distance,

        'servicios_originales_sin_dividir': original_service_no_change,
        'puntos_por_servicio_mean': points_within_new_services['mean'],
        'puntos_por_servicio_cv': points_within_new_services['std']/points_within_new_services['mean'],
        'puntos_por_servicio_median': points_within_new_services['50%'],
        'puntos_por_servicio_min': points_within_new_services['min'],
        'puntos_por_servicio_max': points_within_new_services['max'],
    }, index=[0])

    return stats


def find_change_in_direction(df, branch):
    # Create a new series with the differences between consecutive elements
    series = df[f'order_{branch}'].copy()

    # check diference against previous stop
    diff_series = series.diff()
    # select only where change happens
    diff_series = diff_series.loc[diff_series != 0]

    # checks for change in a decreasing manner
    change_indexes = diff_series.map(lambda x: x < 0).diff().fillna(False)
    return change_indexes


def compute_distance_km_gps(gps_df):

    res = 11
    distancia_entre_hex = h3.edge_length(resolution=res, unit="km")
    distancia_entre_hex = distancia_entre_hex * 2

    # Georeferenciar con h3
    gps_df["h3"] = gps_df.apply(geo.h3_from_row, axis=1,
                                args=(res, "latitud", "longitud"))

    # Producir un lag con respecto al siguiente posicionamiento gps
    gps_df["h3_lag"] = (
        gps_df.reindex(columns=["dia", "id_linea", "interno", "h3"])
        .groupby(["dia", "id_linea", "interno"])
        .shift(-1)
    )

    # Calcular distancia h3
    gps_df = gps_df.dropna(subset=["h3", "h3_lag"])
    gps_dict = gps_df.to_dict("records")
    gps_df.loc[:, ["distance_km"]] = list(map(geo.distancia_h3, gps_dict))
    gps_df.loc[:, ["distance_km"]] = gps_df["distance_km"] * \
        distancia_entre_hex
    gps_df = gps_df.drop(['h3_lag'], axis=1)
    return gps_df
