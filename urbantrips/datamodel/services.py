import pandas as pd
import geopandas as gpd
from urbantrips.utils import utils
from urbantrips.utils.utils import (guardar_tabla_sql, levanto_tabla_sql, duracion)
import numpy as np


import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
    module=r".*urbantrips.*services",  # tu módulo
)

@duracion
def process_services(line_ids=None):
    """
    Download unprocessed gps data and classify them into services
    for all days and all lines or a set of specified set of line ids

    Parameters
    ----------
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
        print("Procesando servicios en base a tabla gps")
        # check line id type and turn it into list if is a single line id
        if line_ids is not None:
            if isinstance(line_ids, int):
                line_ids = [line_ids]

            line_ids_str = ",".join(map(str, line_ids))
        else:
            line_ids_str = None

        delete_old_services_data(line_ids_str)

        if line_ids is not None:
            print(f"Descargando paradas y puntos gps para id lineas {line_ids_str}")
        else:
            print("Descargando paradas y puntos gps para todas las lineas")

        gps_points, stops = get_stops_and_gps_data(line_ids_str)

        if gps_points is None:
            print("Todos los puntos gps ya fueron procesados en servicios ")
        else:
            print("Clasificando puntos gps en servicios")
            gps_points.groupby("id_linea").apply(process_line_services, stops=stops)


def delete_old_services_data(line_ids_str):
    """
    Deletes data from services tables for all lines or
    a specified set of line ids
    """
    conn_data = utils.iniciar_conexion_db(tipo="data")
    tables = ["services_gps_points", "services", "services_stats"]
    for table in tables:
        q = f"""
        DELETE FROM {table}
        """
        if line_ids_str is not None:
            q = q + f"where id_linea in ({line_ids_str});"
        else:
            q = q + ";"
        conn_data.execute(q)
        conn_data.commit()

    conn_data.close()


def get_stops_and_gps_data(line_ids_str):
    """
    Download unprocessed gps data and stops for all lines
    or ofr a specified set of line ids and all days
    """
    configs = utils.leer_configs_generales()

    conn_insumos = utils.iniciar_conexion_db(tipo="insumos")
    conn_data = utils.iniciar_conexion_db(tipo="data")

    # check if data is present in the db
    if not utils.check_table_in_db(table_name="gps", tipo_db="data"):
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
    if not utils.check_table_in_db(table_name="gps", tipo_db="data"):
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
    if line_ids_str is not None:
        gps_query = gps_query + f"and g.id_linea in ({line_ids_str});"
    else:
        gps_query = gps_query + ";"

    gps_points = pd.read_sql(gps_query, conn_data)
    
    gps_points = gps_points.rename(columns={"distance_km": "distance_route", "distance_servicio_mts": "distance_route_gps"})
    gps_points["distance_route_gps"] = (gps_points["distance_route_gps"] / 1000).round(2)

    gps_points = gpd.GeoDataFrame(
        gps_points,
        geometry=gpd.GeoSeries.from_xy(
            x=gps_points.longitud, y=gps_points.latitud, crs="EPSG:4326"
        ),
        crs="EPSG:4326",
    )
    gps_points = gps_points.to_crs(epsg=configs["epsg_m"])

    gps_lines = gps_points.id_linea.drop_duplicates()
    gps_lines_str = ",".join(gps_lines.map(str))

    if len(gps_lines_str) == 0:
        return None, None

    elif configs["utilizar_servicios_gps"]:
        return gps_points, None

    else:
        # check if data is present in the db
        if not utils.check_table_in_db(table_name="stops", tipo_db="insumos"):
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
            where id_linea in ({gps_lines_str})
        """

        stops = pd.read_sql(stops_query, conn_insumos)
        # check all gps points have stops for that line
        line_no_stops_mask = ~gps_lines.isin(stops.id_linea.drop_duplicates())
        if line_no_stops_mask.any():
            line_no_stops = gps_lines[line_no_stops_mask]
            line_no_stops_str = ",".join(line_no_stops.map(str))

            print(
                f"""Hay lineas con GPS que no tienen paradas:
                  {line_no_stops_str}
                  No se procesaran"""
            )

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


def process_line_services(gps_points, stops):
    """
    Takes gps points and stops for a given line,
    classifies each point into a services
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

    conn_data = utils.iniciar_conexion_db(tipo="data")
    #   print(f"Procesando servicios en base a gps para id_linea {line_id}")

    # if there are stops select only stops for that line
    if stops is not None:
        line_stops_gdf = stops.loc[stops.id_linea == line_id, :]
    else:
        line_stops_gdf = None

    # print("Asignando servicios")
    gps_points_with_new_service_id = (
        gps_points.groupby(["dia", "id_ramal", "interno"], as_index=False)
        .apply(classify_line_gps_points_into_services, line_stops_gdf=line_stops_gdf)
        .droplevel(0)
    )

    # save result to services table
    services_gps_points = gps_points_with_new_service_id.reindex(
        columns=[
            "id",
            "id_linea",
            "id_ramal",
            "interno",
            "dia",
            "original_service_id",
            "new_service_id",
            "service_id",
            "id_ramal_gps_point",
            "node_id",
        ]
    )
    # services_gps_points.to_sql(
    #     "services_gps_points",
    #     conn_data,
    #     if_exists="append",
    #     index=False,
    #     method="multi",
    #     chunksize=40,
    # )
    
    dias_ultima_corrida = levanto_tabla_sql("dias_ultima_corrida", "data")    
    guardar_tabla_sql(
        services_gps_points,
        "services_gps_points",
        tabla_tipo="data",
        modo="append",
        filtros={"dia": dias_ultima_corrida["dia"].tolist(), "id_linea": [line_id]},
    )

    # print("Creando tabla de servicios")
    # process services gps points into services table
    line_services = create_line_services_table(gps_points_with_new_service_id)
    
    # line_services.to_sql("services", conn_data, if_exists="append", index=False)
    
    dt = (
        pd.to_datetime(
            line_services["max_datetime"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )
        -
        pd.to_datetime(
            line_services["min_datetime"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )
    ).dt.total_seconds() / 3600

    # evitar divisiones por cero
    dt = dt.replace(0, np.nan)

    line_services["kmh_route"] = (
        line_services["distance_route"] / dt
    ).round(2)

    line_services["kmh_route_gps"] = (
        line_services["distance_route_gps"] / dt
    ).round(2)
    
    guardar_tabla_sql(
        line_services,
        "services",
        tabla_tipo="data",
        modo="append",
        filtros={"dia": dias_ultima_corrida["dia"].tolist(), "id_linea": [line_id]},
    )
    
    stats = compute_services_stats(line_services)
    
    guardar_tabla_sql(
        stats,
        "services_stats",
        tabla_tipo="data",
        modo="append",
        filtros={"dia": dias_ultima_corrida["dia"].tolist(), "id_linea": [line_id]},
    )
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
    
    line_services['distance_route_gps'] = (line_services['distance_route_gps'] / 1000).round(2)
    
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
    line_gps_points, line_stops_gdf, debug=False, *args, **kwargs
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
    # create original service id
    original_service_id = (
        line_gps_points.reindex(columns=["dia", "id_ramal", "interno", "service_type"])
        .groupby(["dia", "id_ramal", "interno"])
        .apply(create_original_service_id)
    )
    original_service_id = original_service_id.service_type
    original_service_id = original_service_id.droplevel([0, 1, 2])
    line_gps_points.loc[:, ["original_service_id"]] = original_service_id

    # check configs if trust in service type gps
    configs = utils.leer_configs_generales()

    trust_service_type_gps = configs["utilizar_servicios_gps"]

    if trust_service_type_gps:
        # classify services based on gps dataset attribute
        line_gps_points.loc[:, ["new_service_id"]] = line_gps_points[
            "original_service_id"
        ].copy()
    else:
        # classify services based on stops
        line_gps_points = infer_service_id_stops(
            line_gps_points, line_stops_gdf, debug=debug
        )

    # Classify idling points when there is no movement
    line_gps_points.loc[:, ["idling"]] = line_gps_points.distance_route < 0.1

    # create a unique id from both old and new
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

    stats = base_stats.drop(
        columns=[
            "n_servicios_cortos_idling",
            "distancia_recorrida_valida",
        ]
    )

    return stats


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


def delete_services_data(id_linea):
    "this function deletes data for a given line in servicies tables"

    conn_data = utils.iniciar_conexion_db(tipo="data")
    print(f"Borrando datos en tablas de servicios para id linea {id_linea};")
    conn_data.execute(f"DELETE FROM services where id_linea = {id_linea};")
    conn_data.execute(f"DELETE FROM services_stats where id_linea = {id_linea};")
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
