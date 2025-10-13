import pandas as pd
import geopandas as gpd
from urbantrips.carto.carto import floor_rounding, create_route_section_ids
from urbantrips.carto.routes import (
    get_route_geoms_with_sections_data,
    check_exists_route_section_points_table,
    upload_route_section_points_table,
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
    if gps is None:
        print("No existen datos de GPS para los filtros aplicados")
        return None

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
    delete_old_supply_stats_by_section_id(
        route_geoms, hour_range, day_type, yr_mos, db_type="data"
    )

    # compute section load
    print("Computing supply stats per route section ...")

    if (len(route_geoms) > 0) and (len(gps) > 0):

        section_supply_stats_table = gps.groupby(["id_linea", "yr_mo"]).apply(
            compute_section_supply_stats, route_geoms=route_geoms
        )

        section_supply_stats_table = section_supply_stats_table.droplevel(
            2, axis=0
        ).reset_index()

        # Add hourly range
        if hour_range:
            section_supply_stats_table["hour_min"] = hour_range[0]
            section_supply_stats_table["hour_max"] = hour_range[1]
        else:
            section_supply_stats_table["hour_min"] = None
            section_supply_stats_table["hour_max"] = None

        section_supply_stats_table["day_type"] = day_type

        section_supply_stats_table = section_supply_stats_table.reindex(
            columns=[
                "id_linea",
                "yr_mo",
                "day_type",
                "n_sections",
                "section_meters",
                "sentido",
                "section_id",
                "hour_min",
                "hour_max",
                "n_vehicles",
                "avg_speed",
                "median_speed",
                "speed_interval",
                "frequency",
                "frequency_interval",
            ]
        )

        conn_data = iniciar_conexion_db(tipo="data")
        print("Uploading data to db...")
        section_supply_stats_table.to_sql(
            "supply_stats_by_section_id",
            conn_data,
            if_exists="append",
            index=False,
        )
        conn_data.close()
    else:
        print("No existen recorridos o etapas para las líneas")
        print("Cantidad de lineas:", len(line_ids))
        print("Cantidad de recorridos", len(route_geoms))
        print("Cantidad de puntos gps", len(gps))
    return section_supply_stats_table


def compute_section_supply_stats(gps, route_geoms):
    line_id = gps.id_linea.unique()[0]
    print(f"Calculando estadisticos de oferta por tramo para linea id {line_id}")

    if (route_geoms.id_linea == line_id).any():
        route = route_geoms.loc[route_geoms.id_linea == line_id, :]

        route_geom = route.geometry.item()
        n_sections = route.n_sections.item()
        section_meters = route.section_meters.item()

        # use more granular lrs to classify direction
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
        # remove duplicates for same vehicle in same section
        gps_route_sections_df = gps_route_sections_df.drop_duplicates()

        gps["delta_hours"] = (gps.fecha_next - gps.fecha) / 60 / 60

        # remove any gps with less than 4 minutes
        gps = gps.loc[gps.delta_hours > (4 / 60), :]

        gps["kmh"] = gps.distance_km / gps.delta_hours
        average_speed_table = (
            gps.reindex(columns=["dia", "section_id", "sentido", "kmh"])
            .groupby(["dia", "sentido", "section_id"])
            .agg(avg_speed=("kmh", "mean"), median_speed=("kmh", "median"))
            .reset_index()
        )
        average_speed_table.avg_speed = average_speed_table.avg_speed.round()
        average_speed_table.median_speed = average_speed_table.median_speed.round()

        n_vehicles_table = (
            gps_route_sections_df.groupby(["dia", "sentido", "section_id"])
            .size()
            .reset_index()
            .rename(columns={0: "n_vehicles"})
        )

        section_supply_stats = n_vehicles_table.merge(
            average_speed_table, on=["dia", "sentido", "section_id"], how="left"
        )
        section_supply_stats["n_sections"] = n_sections
        section_supply_stats["section_meters"] = section_meters
        # Create the frequency column
        section_supply_stats["frequency"] = 60 / section_supply_stats["n_vehicles"]

        labels = [
            f"{str(i).zfill(2)} - {str(i+5).zfill(2)} min" for i in range(0, 60, 5)
        ]

        # Group the frequency into intervals of 5 minutes
        section_supply_stats["frequency_interval"] = pd.cut(
            section_supply_stats["frequency"],
            bins=range(0, 65, 5),
            right=False,
            labels=labels,
        )

        labels = [
            f"{str(i).zfill(2)} - {str(i+5).zfill(2)} kmh" for i in range(0, 60, 5)
        ]
        # Group the speed into intervals of 5 kmh
        section_supply_stats["speed_interval"] = pd.cut(
            section_supply_stats["avg_speed"],
            bins=range(0, 65, 5),
            right=False,
            labels=labels,
        )

    return section_supply_stats


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

    if len(gps) == 0:
        return None

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


def delete_old_supply_stats_by_section_id(
    route_geoms, hour_range, day_type, yr_mos, db_type="data"
):
    """
    Deletes old data in table ocupacion_por_linea_tramo
    """
    table_name = "supply_stats_by_section_id"

    if db_type == "data":
        conn = iniciar_conexion_db(tipo="data")
    else:
        conn = iniciar_conexion_db(tipo="dash")

    # hour range filter
    if hour_range:
        hora_min_filter = f"= {hour_range[0]}"
        hora_max_filter = f"= {hour_range[1]}"
    else:
        hora_min_filter = "is NULL"
        hora_max_filter = "is NULL"

    # create a df with n sections for each line
    delete_df = route_geoms.reindex(columns=["id_linea", "n_sections"])
    for yr_mo in yr_mos:
        for _, row in delete_df.iterrows():
            # Delete old data for those parameters
            print(f"Borrando datos antiguos de {table_name}")
            print(row.id_linea)
            print(f"{row.n_sections} secciones")
            print(yr_mo)
            if hour_range:
                print(f"y horas desde {hour_range[0]} a {hour_range[1]}")

            q_delete = f"""
                delete from {table_name}
                where id_linea = {row.id_linea} 
                and hour_min {hora_min_filter}
                and hour_max {hora_max_filter}
                and day_type = '{day_type}'
                and n_sections = {row.n_sections}
                and yr_mo = '{yr_mo}';
                """

            cur = conn.cursor()
            cur.execute(q_delete)
            conn.commit()

    conn.close()
    print("Fin borrado datos previos")
