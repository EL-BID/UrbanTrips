import warnings
import pandas as pd
import geopandas as gpd
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    check_date_type,
    is_date_string
)
from urbantrips.kpi.kpi import (
    create_route_section_ids,
    add_od_lrs_to_legs_from_route
)
from urbantrips.kpi.kpi import upload_route_section_points_table

from urbantrips.geo import geo


@duracion
def compute_lines_od_matrix(
    line_ids=None,
    hour_range=False,
    n_sections=10,
    section_meters=None,
    day_type="weekday",
):
    """
    Computes leg od matrix for a line or set of lines using route sections

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

    # check inputs
    check_date_type(day_type)
    if line_ids is not None:

        if isinstance(line_ids, int):
            line_ids = [line_ids]
        lines_str = ",".join(map(str, line_ids))
        line_ids_where = f" where id_linea in ({lines_str})"

    else:
        lines_str = ''
        line_ids_where = " where id_linea is not NULL"

    if n_sections is not None:
        if n_sections > 1000:
            raise Exception(
                "No se puede utilizar una cantidad de secciones > 1000")

    conn_data = iniciar_conexion_db(tipo="data")
    conn_insumos = iniciar_conexion_db(tipo="insumos")

    # read legs data
    legs = read_legs_data_by_line_hours_and_day(
        lines_str, hour_range, day_type)

    # read routes data
    q_route_geoms = "select * from lines_geoms"
    q_route_geoms = q_route_geoms + line_ids_where

    route_geoms = pd.read_sql(q_route_geoms, conn_insumos)
    if line_ids:
        route_geoms = route_geoms.loc[route_geoms.id_linea.isin(line_ids)]

    route_geoms["geometry"] = gpd.GeoSeries.from_wkt(route_geoms.wkt)
    route_geoms = gpd.GeoDataFrame(
        route_geoms, geometry="geometry", crs="EPSG:4326"
    )
    # Set which parameter to use to slit route geoms
    if section_meters:
        epsg_m = geo.get_epsg_m()
        # project geoms and get for each geom a n_section
        route_geoms = route_geoms.to_crs(epsg=epsg_m)
        new_n_sections = (
            route_geoms.geometry.length / section_meters).astype(int)
        route_geoms["n_sections"] = new_n_sections

        if (route_geoms.n_sections > 1000).any():
            warnings.warn(
                "Algunos recorridos tienen mas de 1000 segmentos"
                "Puede arrojar resultados imprecisos "
            )
        n_sections = new_n_sections
        route_geoms = route_geoms.to_crs(epsg=4326)

    route_geoms["n_sections"] = n_sections

    upload_route_section_points_table(route_geoms)

    # delete old data
    yr_mos = legs.yr_mo.unique()
    delete_old_lines_od_matrix_by_section_data_q(
        route_geoms, hour_range, day_type, yr_mos)

    print("Computing section load per route ...")

    if (len(route_geoms) > 0) and (len(legs) > 0):

        line_od_matrix = legs.groupby(["id_linea", "yr_mo"]).apply(
            compute_line_od_matrix,
            route_geoms=route_geoms,
            hour_range=hour_range,
            day_type=day_type
        )

        line_od_matrix = line_od_matrix.droplevel(
            2, axis=0).reset_index()

        line_od_matrix = line_od_matrix.reindex(
            columns=[
                "id_linea",
                "yr_mo",
                "day_type",
                "n_sections",
                "hour_min",
                "hour_max",
                "section_id_o",
                "section_id_d",
                "legs",
                "prop"
            ]
        )

        print("Uploading data to db...")
        line_od_matrix.to_sql(
            "lines_od_matrix_by_section", conn_data, if_exists="append",
            index=False,)

        return line_od_matrix

    else:
        print('No existen recorridos o etapas para las líneas')
        print("Cantidad de lineas:", len(line_ids))
        print("Cantidad de recorridos", len(route_geoms))
        print("Cantidad de etapas", len(legs))


def read_legs_data_by_line_hours_and_day(lines_str, hour_range, day_type):
    """
    Reads legs data by line id, hour range and type of day

    Parameters
    ----------
    line_id : int, list of ints or bool
        route id or list of route ids present in the legs dataset. Route
        section load will be computed for that subset of lines. If False, it
        will run with all routes.
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
    q_main_legs = """
    select id_linea, dia, factor_expansion_linea,h3_o,h3_d, od_validado
    from etapas
    """

    # If line and hour, get that subset
    if len(lines_str) > 0:
        line_id_where = f" where id_linea in ({lines_str})"
    else:
        line_id_where = " where id_linea is not NULL"

    q_main_legs = q_main_legs + line_id_where

    if hour_range:
        hour_range_where = (
            f" and hora >= {hour_range[0]} and hora <= {hour_range[1]}"
        )
        q_main_legs = q_main_legs + hour_range_where

    day_type_is_a_date = is_date_string(day_type)

    if day_type_is_a_date:
        q_main_legs = q_main_legs + f" and dia = '{day_type}'"

    q_legs = f"""
        select id_linea, dia, factor_expansion_linea,h3_o,h3_d
        from ({q_main_legs}) e
        where e.od_validado==1
    """
    print("Obteniendo datos de etapas")

    # get data for legs and route geoms
    conn_data = iniciar_conexion_db(tipo="data")
    legs = pd.read_sql(q_legs, conn_data)
    conn_data.close()

    legs['yr_mo'] = legs.dia.str[:7]

    if not day_type_is_a_date:
        # create a weekday_filter
        weekday_filter = pd.to_datetime(
            legs.dia, format="%Y-%m-%d").dt.dayofweek < 5

        if day_type == "weekday":
            legs = legs.loc[weekday_filter, :]
        else:
            legs = legs.loc[~weekday_filter, :]

    return legs


def delete_old_lines_od_matrix_by_section_data_q(
    route_geoms, hour_range, day_type, yr_mos
):
    """
    Deletes old data in table lines_od_matrix_by_section
    """
    conn_data = iniciar_conexion_db(tipo="data")

    # hour range filter
    if hour_range:
        hora_min_filter = f"= {hour_range[0]}"
        hora_max_filter = f"= {hour_range[1]}"
    else:
        hora_min_filter = "is NULL"
        hora_max_filter = "is NULL"

    # create a df with n sections for each line
    delete_df = route_geoms.reindex(columns=['id_linea', 'n_sections'])
    for yr_mo in yr_mos:
        for _, row in delete_df.iterrows():
            # Delete old data for those parameters
            print("Borrando datos antiguos de Matriz OD para linea")
            print(row.id_linea)
            print(f"{row.n_sections} secciones")
            print(yr_mo)
            if hour_range:
                print(f"y horas desde {hour_range[0]} a {hour_range[1]}")

            q_delete = f"""
                delete from lines_od_matrix_by_section
                where id_linea = {row.id_linea} 
                and hour_min {hora_min_filter}
                and hour_max {hora_max_filter}
                and day_type = '{day_type}'
                and n_sections = {row.n_sections}
                and yr_mo = '{yr_mo}';
                """

            cur = conn_data.cursor()
            cur.execute(q_delete)
            conn_data.commit()

    conn_data.close()
    print("Fin borrado datos previos")


def round_section_id_series(series, section_ids):
    """
    Takes an original 3 digits interpolation point on a route
    and rounds to the nearest section id for that route
    """
    original_points = series.drop_duplicates()
    rounded_points = original_points.map(
        lambda p: section_ids[(p - section_ids).abs().idxmin()])
    replace_dict = {k: v for (k, v) in zip(original_points, rounded_points)}
    out = series.replace(replace_dict)
    return out


def compute_line_od_matrix(df, route_geoms, hour_range, day_type):
    """
    Computes leg od matrix for a line or set of lines using route sections

    Parameters
    ----------
    df : pandas DataFrame
        A legs DataFrame with OD data 
    route_geoms: geopandas GeoDataFrame
        A routes GeoDataFrame with routes and number of sections to slice it
    hour_range : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24. Route section
        load will be computed for legs happening within tat time range.
        If False it won't filter by hour.
    day_type: str
        type of day on which the section load is to be computed. It can take
        `weekday`, `weekend` or a specific day in format 'YYYY-MM-DD'

    Returns
    ----------
    pandas DataFrame
        A OD matrix with OD by route section id, number of legs for that
        pair and percentaje of legs for that hour range

    """

    line_id = df.id_linea.unique()[0]
    n_sections = route_geoms.loc[route_geoms.id_linea ==
                                 line_id, 'n_sections'].item()

    print(f"Computing section load id_route {line_id}")

    if (route_geoms.id_linea == line_id).any():

        route_geom = route_geoms.loc[route_geoms.id_linea ==
                                     line_id, "geometry"].item()

        df = add_od_lrs_to_legs_from_route(legs_df=df, route_geom=route_geom)

        # Assign a direction based on line progression
        df = df.reindex(
            columns=["dia", "o_proj", "d_proj", "factor_expansion_linea"])

        # Compute total legs per direction
        # First totals per day
        totals_by_day = df\
            .groupby(["dia"], as_index=False)\
            .agg(daily_legs=("factor_expansion_linea", "sum"))

        totals_by_typeday = totals_by_day.daily_legs.mean()

        # round section ids
        section_ids = create_route_section_ids(n_sections)

        df.o_proj = round_section_id_series(df.o_proj, section_ids)
        df.d_proj = round_section_id_series(df.d_proj, section_ids)

        totals_by_day_section_id = df\
            .groupby(["dia", "o_proj", "d_proj"], as_index=False)\
            .agg(legs=("factor_expansion_linea", "sum"))

        # then average for type of day
        totals_by_typeday_section_id = totals_by_day_section_id\
            .groupby(["o_proj", "d_proj"], as_index=False)\
            .agg(legs=("legs", "mean"))

        totals_by_typeday_section_id['prop'] = (
            totals_by_typeday_section_id.legs / totals_by_typeday * 100
        )

        totals_by_typeday_section_id["day_type"] = day_type
        totals_by_typeday_section_id["n_sections"] = n_sections

        # Add hourly range
        if hour_range:
            totals_by_typeday_section_id["hour_min"] = hour_range[0]
            totals_by_typeday_section_id["hour_max"] = hour_range[1]
        else:
            totals_by_typeday_section_id["hour_min"] = None
            totals_by_typeday_section_id["hour_max"] = None

        totals_by_typeday_section_id = totals_by_typeday_section_id.rename(
            columns={'o_proj': 'section_id_o',
                     'd_proj': 'section_id_d'})
        totals_by_typeday_section_id = totals_by_typeday_section_id.reindex(
            columns=[
                'day_type', 'n_sections',
                'hour_min', 'hour_max', 'section_id_o', 'section_id_d', 'legs', 'prop'
            ]
        )
    else:
        print("No existe recorrido para id_linea:", line_id)

    return totals_by_typeday_section_id
