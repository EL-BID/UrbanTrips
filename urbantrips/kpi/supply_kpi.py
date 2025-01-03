import pandas as pd
from urbantrips.carto.routes import (
    get_route_geoms_with_sections_data,
    check_exists_route_section_points_table,
    upload_route_section_points_table,
    floor_rounding,
    create_route_section_ids,
    build_gps_route_sections_df,
)
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    check_date_type,
    create_line_ids_sql_filter,
    is_date_string,
)


import geopandas as gpd


@duracion
def compute_route_section_supply(
    line_ids=False,
    hour_range=False,
    n_sections=10,
    section_meters=None,
    day_type="weekday",
):
    """
    Computes the load per route section.

    Parameters
    ----------
    line_ids : int, list of ints or bool
        route id or list of route ids present in the legs dataset. Route
        section load will be computed for that subset of lines. If False, it
        will run with all routes.
    hour_range : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24. Route section
        load will be computed for legs happening within tat time range.
        If False it won't filter by hour.
    n_sections: int
        number of sections to split the route geom
    section_meters: int
        section lenght in meters to split the route geom. If specified,
        this will be used instead of n_sections.
    day_type: str
        type of day on which the section load is to be computed. It can take
        `weekday`, `weekend` or a specific day in format 'YYYY-MM-DD'
    """

    check_date_type(day_type)

    line_ids_where = create_line_ids_sql_filter(line_ids)

    if n_sections is not None:
        if n_sections > 1000:
            raise Exception("No se puede utilizar una cantidad de secciones > 1000")

    # read legs data
    gps = read_gps_data_by_line_hours_and_day(line_ids_where, hour_range, day_type)

    # read routes geoms
    route_geoms = get_route_geoms_with_sections_data(
        line_ids_where, section_meters, n_sections
    )

    # check which section geoms are already crated
    new_route_geoms = check_exists_route_section_points_table(route_geoms)

    # create the line and n sections pair missing and upload it to the db
    if len(new_route_geoms) > 0:

        upload_route_section_points_table(new_route_geoms, delete_old_data=False)

    # delete old seciton load data
    yr_mos = gps.yr_mo.unique()

    # ESTO SERIA GENERALIZABLE
    # delete_old_route_section_load_data(
    #    route_geoms, hour_range, day_type, yr_mos, db_type="data"
    # )
    # compute section load
    print("Computing supply stats per route section ...")

    if (len(route_geoms) > 0) and (len(gps) > 0):
        """
        section_supply_stats_table = gps.groupby(["id_linea", "yr_mo"]).apply(
            compute_section_supply_stats,
            route_geoms=route_geoms,
            hour_range=hour_range,
            day_type=day_type,
        )
        """
    return gps


def compute_section_supply_stats(gps, route_geoms):
    # compute_section_load_table TOMAR ESTA
    # EN BASE A LO QUE CALCULE VER LO DE delete_old_route_section_load_data

    # id_linea|yr_mo  |day_type|n_sections|section_meters|sentido|section_id|hour_min|hour_max|n_vehicles|avg_speed

    # DISTANCIA: si un punto gps esta demasiado lejos de la seccion, no se considera
    # def compute_section_supply_stats(gps, route_geoms, hour_range, day_type):
    line_id = gps.id_linea.unique()[0]
    print(f"Calculando estadisticos de oferta por tramo para linea id {line_id}")

    if (route_geoms.id_linea == line_id).any():
        route = route_geoms.loc[route_geoms.id_linea == line_id, :]

        route_geom = route.geometry.item()
        n_sections = route.n_sections.item()

        gps["lrs"] = route_geom.project(
            gpd.GeoSeries.from_xy(gps.longitud, gps.latitud), normalized=True
        ).map(floor_rounding)

        lags = (
            gps.reindex(columns=["id_ramal", "interno", "lrs", "fecha"])
            .groupby(["id_ramal", "interno"])
            .shift(-1)
            .rename(columns={"lrs": "lrs_next", "fecha": "fecha_next"})
        )

        gps = pd.concat([gps, lags], axis=1)
        gps["sentido"] = [
            "ida" if row.lrs <= row.lrs_next else "vuelta" for _, row in gps.iterrows()
        ]

        # compute section ids based on amount of sections
        section_ids_LRS = create_route_section_ids(n_sections)
        # remove 0 form cuts so 0 gets included in bin
        section_ids_LRS_cut = section_ids_LRS.copy()
        section_ids_LRS_cut.loc[0] = -0.001

        # For each leg, build traversed route segments ids
        section_ids = list(range(1, len(section_ids_LRS_cut)))

        gps["section_id"] = pd.cut(
            gps["lrs"], bins=section_ids_LRS_cut, labels=section_ids, right=True
        )
        gps["section_id_next"] = pd.cut(
            gps["lrs_next"], bins=section_ids_LRS_cut, labels=section_ids, right=True
        )

        # remove legs with no origin or destination projected
        gps = gps.dropna(subset=["lrs", "lrs_next"])

        gps_dict = (
            gps.reindex(
                columns=[
                    "sentido",
                    "dia",
                    "id_ramal",
                    "interno",
                    "section_id",
                    "section_id_next",
                ]
            )
            .drop_duplicates()
            .to_dict("records")
        )
        gps_route_sections_df = pd.concat(map(build_gps_route_sections_df, gps_dict))

        gps["delta_hours"] = (gps.fecha_next - gps.fecha) / 60 / 60
        gps["kmh"] = gps.distance_km / gps.delta_hours
    return gps, gps_route_sections_df


def read_gps_data_by_line_hours_and_day(line_ids_where, hour_range, day_type):
    """
    Reads legs data by line id, hour range and type of day

    Parameters
    ----------
    line_ids_where : str
        where clause in a sql query with line ids .
    hour_range : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24. Route section
        load will be computed for legs happening within tat time range.
        If False it won't filter by hour.
    day_type: str
        type of day on which the section load is to be computed. It can take
        `weekday`, `weekend` or a specific day in format 'YYYY-MM-DD'

    Returns
    -------
    legs : pandas.DataFrame
        dataframe with legs data by line id, hour range and type of day

    """

    # Read legs data by line id, hours, day type
    #
    q_main = """
    select *
    from gps
    """
    q_main = q_main + line_ids_where

    day_type_is_a_date = is_date_string(day_type)

    if day_type_is_a_date:
        q_main = q_main + f" and dia = '{day_type}'"

    print("Obteniendo datos de GPS")

    # get data for gps
    conn_data = iniciar_conexion_db(tipo="data")
    gps = pd.read_sql(q_main, conn_data)
    conn_data.close()

    gps["yr_mo"] = gps.dia.str[:7]

    if not day_type_is_a_date:
        # create a weekday_filter
        weekday_filter = pd.to_datetime(gps.dia, format="%Y-%m-%d").dt.dayofweek < 5

        if day_type == "weekday":
            gps = gps.loc[weekday_filter, :]
        else:
            gps = gps.loc[~weekday_filter, :]

    if hour_range:
        gps.loc[:, ["hora"]] = gps.fecha.map(
            lambda ts: pd.Timestamp(ts, unit="s")
        ).dt.hour
        gps = gps.loc[(gps.hora >= hour_range[0]) & (gps.hora <= hour_range[1]), :]

    return gps
