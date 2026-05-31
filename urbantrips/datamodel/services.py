import logging
import pandas as pd
import geopandas as gpd
from urbantrips.utils import utils
from urbantrips.utils.utils import duracion
from urbantrips.storage.context import StorageContext

logger = logging.getLogger(__name__)


@duracion
def process_services(ctx: StorageContext, line_ids=None):
    """
    Download unprocessed gps data and classify them into services
    for all days and all lines or a set of specified set of line ids

    Parameters
    ----------
    ctx : StorageContext
    line_ids : int, list
        line id or ids to process services for

    Returns
    -------
        None. Updates services_gps_points, services,
        and services_stats tables in db

    """
    configs = utils.leer_configs_generales()
    nombre_archivo_gps = configs["nombre_archivo_gps"]

    if nombre_archivo_gps is not None:
        logger.info("Procesando servicios en base a tabla gps")
        # check line id type and turn it into list if is a single line id
        if line_ids is not None:
            if isinstance(line_ids, int):
                line_ids = [line_ids]

            line_ids_str = ",".join(map(str, line_ids))
        else:
            line_ids_str = None

        delete_old_services_data(ctx, line_ids_str)

        if line_ids is not None:
            logger.info("Descargando paradas y puntos gps para id lineas %s", line_ids_str)
        else:
            logger.info("Descargando paradas y puntos gps para todas las lineas")

        gps_points, stops = get_stops_and_gps_data(ctx, line_ids_str)

        if gps_points is None:
            logger.info("Todos los puntos gps ya fueron procesados en servicios")
        else:
            logger.info("Clasificando puntos gps en servicios")
            gps_points.groupby("id_linea").apply(process_line_services, stops=stops, ctx=ctx)


def delete_old_services_data(ctx: StorageContext, line_ids_str):
    """
    Deletes data from services tables for all lines or
    a specified set of line ids
    """
    tables = ["services_gps_points", "services", "services_stats"]
    for table in tables:
        if line_ids_str is not None:
            q = f"DELETE FROM {table} WHERE id_linea IN ({line_ids_str})"
        else:
            q = f"DELETE FROM {table}"
        ctx.data.execute(q)


def get_stops_and_gps_data(ctx: StorageContext, line_ids_str):
    """
    Download unprocessed gps data and stops for all lines
    or for a specified set of line ids and all days
    """
    configs = utils.leer_configs_generales()

    gps_exists = ctx.data.query("SELECT 1 AS gps_exists FROM gps LIMIT 1")
    if gps_exists.empty:
        logger.warning(
            "La tabla gps no tiene registros. Asegurese de tener datos gps y "
            "correr datamodel.transactions.process_and_upload_gps_table()"
        )
        return None, None

    gps_query = """
        SELECT g.*
        FROM gps g
        LEFT JOIN services_stats ss
        ON g.id_linea = ss.id_linea AND g.dia = ss.dia
        WHERE ss.id_linea IS NULL
    """
    if line_ids_str is not None:
        gps_query = gps_query + f" AND g.id_linea IN ({line_ids_str})"

    gps_query = (
        gps_query
        + " ORDER BY g.id_linea, g.dia, g.id_ramal, g.interno, g.fecha, g.id"
    )
    gps_points = ctx.data.query(gps_query)

    if gps_points.empty:
        return None, None

    gps_lines = gps_points.id_linea.drop_duplicates()
    gps_lines_str = ",".join(gps_lines.map(str))

    if len(gps_lines_str) == 0:
        return None, None

    if configs["utilizar_servicios_gps"]:
        return gps_points, None

    gps_points = gpd.GeoDataFrame(
        gps_points,
        geometry=gpd.GeoSeries.from_xy(
            x=gps_points.longitud, y=gps_points.latitud, crs="EPSG:4326"
        ),
        crs="EPSG:4326",
    )
    gps_points = gps_points.to_crs(epsg=configs["epsg_m"])

    all_stops = ctx.insumos.get_stops()
    if all_stops.empty:
        logger.warning(
            "No existe la tabla stops. Asegurese de tener datos de "
            "stops y correr carto.stops.create_stops_table()"
        )

    stops = all_stops[all_stops["id_linea"].isin(gps_lines)]

    # check all gps points have stops for that line
    line_no_stops_mask = ~gps_lines.isin(stops.id_linea.drop_duplicates())
    if line_no_stops_mask.any():
        line_no_stops = gps_lines[line_no_stops_mask]
        line_no_stops_str = ",".join(line_no_stops.map(str))

        logger.warning("Hay lineas con GPS que no tienen paradas: %s — No se procesaran", line_no_stops_str)

        gps_points = gps_points.loc[~gps_points.id_linea.isin(line_no_stops)]

    # use only nodes as stops
    stops = stops.drop_duplicates(subset=["id_linea", "id_ramal", "node_id"])

    stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.GeoSeries.from_xy(
            x=stops.node_x, y=stops.node_y, crs="EPSG:4326"
        ),
        crs="EPSG:4326",
    )

    stops = stops.to_crs(epsg=configs["epsg_m"])

    return gps_points, stops


def process_line_services(gps_points, stops, ctx: StorageContext):
    """
    Takes gps points and stops for a given line,
    classifies each point into services and produces services tables
    and daily stats for that line.
    """
    line_id = gps_points.id_linea.unique()[0]

    if stops is not None:
        line_stops_gdf = stops.loc[stops.id_linea == line_id, :]
    else:
        line_stops_gdf = None

    trust_service_type_gps = utils.leer_configs_generales()["utilizar_servicios_gps"]

    if trust_service_type_gps:
        gps_points_with_new_service_id = classify_line_gps_points_into_services(
            gps_points,
            line_stops_gdf=line_stops_gdf,
            trust_service_type_gps=trust_service_type_gps,
        )
    else:
        gps_points_with_new_service_id = (
            gps_points.groupby(["dia", "id_ramal", "interno"], as_index=False)
            .apply(
                classify_line_gps_points_into_services,
                line_stops_gdf=line_stops_gdf,
                trust_service_type_gps=trust_service_type_gps,
            )
            .droplevel(0)
        )

    services_gps_points = gps_points_with_new_service_id.reindex(
        columns=[
            "id", "id_linea", "id_ramal", "interno", "dia",
            "original_service_id", "new_service_id", "service_id",
            "id_ramal_gps_point", "node_id",
        ]
    )
    ctx.data.append_raw(services_gps_points, "services_gps_points")

    line_services = create_line_services_table(gps_points_with_new_service_id)
    ctx.data.save_services(line_services)

    stats = line_services.groupby(
        ["id_linea", "id_ramal", "dia"], as_index=False
    ).apply(compute_new_services_stats)

    ctx.data.append_raw(stats, "services_stats")
    return stats


def create_line_services_table(line_day_gps_points):
    # get  basic stats for each service
    
    line_services = line_day_gps_points.groupby(
        ["id_linea", "id_ramal", "dia", "interno", "original_service_id", "service_id"],
        as_index=False,
    ).agg(
        is_idling=("idling", "sum"),
        total_points=("idling", "count"),
        distance_route=("distance_route", "sum"),
        distance_route_gps=("distance_route_gps", "sum"),
        min_ts=("fecha", "min"),
        max_ts=("fecha", "max"),
    )
    
    line_services.loc[:, ["min_datetime"]] = line_services.min_ts.map(
        lambda ts: str(pd.Timestamp(ts, unit="s"))
    )
    line_services.loc[:, ["max_datetime"]] = line_services.max_ts.map(
        lambda ts: str(pd.Timestamp(ts, unit="s"))
    )

    # compute idling proportion for each service
    line_services["prop_idling"] = (
        line_services.is_idling / line_services["total_points"]
    ).round(2)
    line_services = line_services.drop(["is_idling"], axis=1)

    # stablish valid services
    line_services["valid"] = (line_services.prop_idling < 0.5) & (
        line_services.total_points > 5
    )

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
    # get amount of original serviices
    n_original_services_ids = len(line_gps_points["original_service_id"].unique())

    # get unique branches in the gps points
    branches = line_stops_gdf.id_ramal.unique()

    # get how many branches pass through that node
    majority_by_node_id = (
        line_stops_gdf.drop_duplicates(["id_ramal", "node_id"])
        .groupby("node_id", as_index=False)
        .agg(branch_mayority=("id_ramal", "count"))
    )

    # go through all branches
    gps_all_branches = pd.DataFrame()
    debug_df = pd.DataFrame()

    for branch in branches:
        # select stops for that branch
        stops_to_join = line_stops_gdf.loc[
            line_stops_gdf.id_ramal == branch, ["branch_stop_order", "geometry"]
        ]

        # get nearest stop for that branch within 1.5 km
        # Not use max_distance. Far away stops will appear as
        # still on the same stop and wont be active branches
        gps_branch = gpd.sjoin_nearest(
            line_gps_points,
            stops_to_join,
            how="left",
            max_distance=1500,
            distance_col="distance_to_stop",
        )
        gps_branch["id_ramal"] = branch

        # Evaluate change on stops order for each branch
        temp_change = gps_branch.groupby(["interno", "original_service_id"]).apply(
            find_change_in_direction
        )

        # when vehicle is always too far away from this branch
        if n_original_services_ids > 1:

            if isinstance(temp_change, type(pd.Series())):
                temp_change = temp_change.droplevel([0, 1])
            else:
                temp_change = False

        # when there is only one original service per vehicle
        else:
            temp_change = pd.Series(
                temp_change.iloc[0].values, index=temp_change.columns
            )

        # eval if temporary change is conssitent 5 points ahead
        gps_branch["temp_change"] = temp_change
        window = 5
        gps_branch["consistent_post"] = (
            gps_branch["temp_change"]
            .shift(-window)
            .fillna(False)
            .rolling(window=window, center=False, min_periods=3)
            .sum()
            == 0
        )

        # Accept there is a change in direction when consistent
        gps_branch["change"] = gps_branch["temp_change"] & gps_branch["consistent_post"]

        # add debugging attributes
        if debug:
            debug_branch = gps_branch.reindex(
                columns=[
                    "id",
                    "branch_stop_order",
                    "id_ramal",
                    "temp_change",
                    "consistent_post",
                    "distance_to_stop",
                    "change",
                ]
            )

            debug_df = pd.concat([debug_df, debug_branch])

        gps_branch = gps_branch.drop(
            [
                "index_right",
                "temp_change",
                "consistent_post",
            ],
            axis=1,
        )
        gps_all_branches = pd.concat([gps_all_branches, gps_branch])

    # for each gps point get the node id form the nearest branch
    branches_distances_table = (
        gps_all_branches.reindex(
            columns=["id", "id_ramal", "distance_to_stop", "branch_stop_order"]
        )
        .sort_values(["id", "distance_to_stop"])
        .drop_duplicates(subset=["id"], keep="first")
        .drop(["distance_to_stop"], axis=1)
    )

    gps_node_ids = branches_distances_table.merge(
        line_stops_gdf.reindex(columns=["node_id", "branch_stop_order", "id_ramal"]),
        on=["id_ramal", "branch_stop_order"],
        how="left",
    ).reindex(columns=["id", "id_ramal", "node_id"])

    # count how many branches see a change in that node
    total_changes_by_gps = gps_all_branches.groupby(["id"], as_index=False).agg(
        total_changes=("change", "sum")
    )

    gps_points_changes = gps_node_ids.merge(
        total_changes_by_gps, how="left", on="id"
    ).merge(majority_by_node_id, how="left", on="node_id")

    # set change when passes the mayority
    gps_points_changes["change"] = (
        gps_points_changes.total_changes >= gps_points_changes.branch_mayority
    )

    # set schema
    cols = ["id", "id_ramal", "node_id", "change"]
    if debug:
        cols = cols + ["branch_mayority", "total_changes"]

    gps_points_changes = gps_points_changes.reindex(columns=cols)

    gps_points_changes = gps_points_changes.rename(
        columns={"id_ramal": "id_ramal_gps_point"}
    )

    line_gps_points = line_gps_points.merge(gps_points_changes, on="id", how="left")

    if n_original_services_ids > 1:

        # Within each original service id, classify services within
        new_services_ids = (
            line_gps_points.groupby("original_service_id")
            .apply(lambda df: df["change"].cumsum().ffill())
            .droplevel(0)
        )
    else:
        new_services_ids = line_gps_points.groupby("original_service_id").apply(
            lambda df: df["change"].cumsum().ffill()
        )

        new_services_ids = pd.Series(
            new_services_ids.iloc[0].values, index=new_services_ids.columns
        )

    line_gps_points["new_service_id"] = new_services_ids

    if debug:
        debug_df = debug_df.pivot(
            index="id",
            columns="id_ramal",
            values=[
                "branch_stop_order",
                "temp_change",
                "consistent_post",
                "change",
                "distance_to_stop",
            ],
        ).reset_index()

        cols = [
            c[0] + "_" + str(c[1]) if c[0] != "id" else c[0] for c in debug_df.columns
        ]

        debug_df.columns = cols
        line_gps_points = line_gps_points.merge(debug_df, on="id", how="left")

    return line_gps_points


def classify_line_gps_points_into_services(
    line_gps_points,
    line_stops_gdf,
    debug=False,
    trust_service_type_gps=None,
    *args,
    **kwargs,
):
    """
    Takes gps points and stops for a given line and classifies each point into
    services based on original gps data or infered basd on stops whenever the
    order of passage across stops switches from increasing to decreasing order
    in the majority of active branches in that line.

    Parameters
    ----------
    line_gps_points : geopandas.GeoDataFrame
        GeoDataFrame with gps points for a given line, branch and vehicle

    line_stops_gdf : geopandas.GeoDataFrame
        GeoDataFrame with stops for a given line
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame points classified into services
    """
    # check configs if trust in service type gps
    if trust_service_type_gps is None:
        configs = utils.leer_configs_generales()
        trust_service_type_gps = configs["utilizar_servicios_gps"]

    group_cols = ["dia", "id_ramal", "interno"]
    sort_cols = [
        c for c in group_cols + ["fecha", "id"] if c in line_gps_points.columns
    ]
    line_gps_points = line_gps_points.sort_values(sort_cols).copy()

    if trust_service_type_gps:
        starts = line_gps_points["service_type"].eq("start_service")
        line_gps_points["original_service_id"] = (
            starts.groupby([line_gps_points[c] for c in group_cols])
            .cumsum()
            .astype(int)
        )
        line_gps_points["new_service_id"] = line_gps_points["original_service_id"]
    else:
        # create original service id
        original_service_id = (
            line_gps_points.reindex(
                columns=["dia", "id_ramal", "interno", "service_type"]
            )
            .groupby(["dia", "id_ramal", "interno"])
            .apply(create_original_service_id)
        )
        original_service_id = original_service_id.service_type
        original_service_id = original_service_id.droplevel([0, 1, 2])
        line_gps_points.loc[:, ["original_service_id"]] = original_service_id

        # classify services based on stops
        line_gps_points = infer_service_id_stops(
            line_gps_points, line_stops_gdf, debug=debug
        )

    # Classify idling points when there is no movement
    line_gps_points.loc[:, ["idling"]] = line_gps_points.distance_route < 0.1

    # create a unique id from both old and new
    if trust_service_type_gps:
        line_gps_points["service_id"] = (
            line_gps_points.groupby(group_cols, sort=False)["new_service_id"]
            .transform(lambda s: pd.factorize(s, sort=False)[0])
            .astype(int)
        )
    else:
        new_ids = line_gps_points.reindex(
            columns=["original_service_id", "new_service_id"]
        ).drop_duplicates()
        new_ids["service_id"] = range(len(new_ids))

        line_gps_points = line_gps_points.merge(
            new_ids, how="left", on=["original_service_id", "new_service_id"]
        )

    return line_gps_points


import numpy as np


def compute_services_stats(line_services):
    group_cols = ["id_linea", "id_ramal", "dia"]

    base_stats = (
        line_services
        .assign(
            servicio_original_key=lambda df: (
                df["interno"].astype(str) + "_" + df["original_service_id"].astype(str)
            ),
            servicio_corto=lambda df: df["total_points"] <= 5,
            servicio_corto_idling=lambda df: (
                (df["prop_idling"] >= 0.5) & (df["total_points"] <= 5)
            ),
            distancia_valida=lambda df: np.where(
                df["valid"], df["distance_route"], 0
            ),
        )
        .groupby(group_cols, as_index=False)
        .agg(
            cant_servicios_originales=("servicio_original_key", "nunique"),
            cant_servicios_nuevos=("service_id", "count"),
            cant_servicios_nuevos_validos=("valid", "sum"),
            n_servicios_nuevos_cortos=("servicio_corto", "sum"),
            n_servicios_cortos_idling=("servicio_corto_idling", "sum"),
            distance_route=("distance_route", "sum"),
            distance_route_gps=("distance_route_gps", "sum"),
            distancia_recorrida_valida=("distancia_valida", "sum"),
        )
    )

    base_stats["prop_servicos_cortos_nuevos_idling"] = np.where(
        base_stats["n_servicios_nuevos_cortos"] > 0,
        (
            base_stats["n_servicios_cortos_idling"]
            / base_stats["n_servicios_nuevos_cortos"]
        ).round(2),
        np.nan,
    )

    base_stats["distance_route"] = (
        base_stats["distance_route"].round()
    )

    base_stats["prop_distancia_recuperada"] = np.where(
        base_stats["distance_route"] > 0,
        (
            base_stats["distancia_recorrida_valida"]
            / base_stats["distance_route"]
        ).round(2),
        np.nan,
    )

    valid_services = line_services.loc[line_services["valid"]].copy()

    if not valid_services.empty:
        sub_services = (
            valid_services
            .groupby(group_cols + ["interno", "original_service_id"])["service_id"]
            .nunique()
            .reset_index(name="n_subservicios")
        )

        no_change_stats = (
            sub_services
            .assign(original_sin_dividir=lambda df: df["n_subservicios"] == 1)
            .groupby(group_cols, as_index=False)
            .agg(
                servicios_originales_sin_dividir=(
                    "original_sin_dividir",
                    "mean",
                )
            )
        )

        no_change_stats["servicios_originales_sin_dividir"] = (
            no_change_stats["servicios_originales_sin_dividir"].round(2)
        )

        base_stats = base_stats.merge(
            no_change_stats,
            on=group_cols,
            how="left",
        )
    else:
        base_stats["servicios_originales_sin_dividir"] = np.nan

    original_services_distance_raw = line_day_services.distance_km.sum()
    original_services_distance = round(original_services_distance_raw)
    if pd.notna(original_services_distance_raw) and original_services_distance_raw > 0:
        new_services_distance = round(
            line_day_services.loc[line_day_services["valid"], "distance_km"].sum()
            / original_services_distance_raw,
            2,
        )
    else:
        new_services_distance = None

    sub_services = (
        line_day_services.loc[line_day_services["valid"], :]
        .groupby(["interno", "original_service_id"])
        .service_id.nunique()
    )

    if len(sub_services):
        sub_services = sub_services.value_counts(normalize=True)

        if 1 in sub_services.index:
            original_service_no_change = round(sub_services[1], 2)
        else:
            original_service_no_change = 0
    else:
        original_service_no_change = None

    day_line_stats = pd.DataFrame(
        {
            "id_linea": id_linea,
            "id_ramal": id_ramal,
            "dia": dia,
            "cant_servicios_originales": n_original_services,
            "cant_servicios_nuevos": n_new_services,
            "cant_servicios_nuevos_validos": n_new_valid_services,
            "n_servicios_nuevos_cortos": n_services_short,
            "prop_servicos_cortos_nuevos_idling": prop_short_idling,
            "distancia_recorrida_original": original_services_distance,
            "prop_distancia_recuperada": new_services_distance,
            "servicios_originales_sin_dividir": original_service_no_change,
        },
        index=[0],
    )
    return day_line_stats


def find_change_in_direction(df):
    # Create a new series with the differences between consecutive elements
    series = df["branch_stop_order"].copy()

    # check diference against previous stop
    diff_series = series.diff().dropna()

    # select only where change happens
    changes_in_series = diff_series.loc[diff_series != 0]

    # checks for change in a decreasing manner
    decreasing_change = changes_in_series.map(lambda x: x < 0)

    decreasing_to_increasing = decreasing_change.diff().fillna(False)

    return decreasing_to_increasing


def create_original_service_id(service_type_series):
    return (service_type_series == "start_service").cumsum()


def delete_services_data(ctx: StorageContext, id_linea):
    "this function deletes data for a given line in services tables"

    logger.debug("Borrando datos en tablas de servicios para id linea %s", id_linea)
    ctx.data.execute(f"DELETE FROM services WHERE id_linea = {id_linea}")
    ctx.data.execute(f"DELETE FROM services_stats WHERE id_linea = {id_linea}")
    ctx.data.execute(
        f"""
        DELETE FROM services_gps_points
        WHERE id IN (SELECT id FROM gps WHERE id_linea = {id_linea})
        """
    )
    logger.debug("Servicios borrados")
