import itertools
import geopandas as gpd
import warnings
import pandas as pd
import numpy as np
import weightedstats as ws
from math import floor
import re
import h3
from urbantrips.geo import geo
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales
)

# KPI WRAPPER


@duracion
def compute_kpi():
    """
    Esta funcion toma los datos de oferta de la tabla gps
    los datos de demanda de la tabla trx
    y produce una serie de indicadores operativos por
    dia y linea y por dia, linea, interno
    """

    print("Produciendo indicadores operativos...")
    conn_data = iniciar_conexion_db(tipo="data")

    cur = conn_data.cursor()
    q = """
        SELECT tbl_name FROM sqlite_master
        WHERE type='table'
        AND tbl_name='gps';
    """
    listOfTables = cur.execute(q).fetchall()

    if listOfTables == []:
        print("No existe tabla GPS en la base")
        print("No se pudeden computar indicadores de oferta")
        return None

    # read data
    legs, gps = read_data_for_daily_kpi()

    if (len(legs) > 0) & (len(gps) > 0):
        # compute KPI per line and date
        compute_kpi_by_line_day(legs=legs, gps=gps)

        # compute KPI per line and type of day
        compute_kpi_by_line_typeday()

    # Run KPI at service level
    cur = conn_data.cursor()
    q = "select count(*) from services where valid = 1;"
    valid_services = cur.execute(q).fetchall()[0][0]

    if valid_services > 0:

        # compute KPI by service and day
        compute_kpi_by_service()

        # compute amount of hourly services by line and day
        compute_dispatched_services_by_line_hour_day()

        # compute amount of hourly services by line and type of day
        compute_dispatched_services_by_line_hour_typeday()
    else:

        print("No hay servicios procesados.")
        print("Correr la funcion services.process_services()")


# SECTION LOAD KPI

@duracion
def compute_route_section_load(
    id_linea=False,
    rango_hrs=False,
    n_sections=10,
    section_meters=None,
    day_type="weekday",
):
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
    n_sections: int
        number of sections to split the route geom
    section_meters: int
        section lenght in meters to split the route geom. If specified,
        this will be used instead of n_sections.
    day_type: str
        type of day on which the section load is to be computed. It can take
        `weekday`, `weekend` or a specific day in format 'YYYY-MM-DD'

    """

    dat_type_is_a_date = is_date_string(day_type)

    # check day type format
    day_type_format_ok = (
        day_type in ["weekday", "weekend"]) or dat_type_is_a_date

    if not day_type_format_ok:
        raise Exception(
            "dat_type debe ser `weekday`, `weekend` o fecha 'YYYY-MM-DD'"
        )

    if n_sections is not None:
        if n_sections > 1000:
            raise Exception(
                "No se puede utilizar una cantidad de secciones > 1000")

    conn_data = iniciar_conexion_db(tipo="data")
    conn_insumos = iniciar_conexion_db(tipo="insumos")

    # delete old data
    q_delete = delete_old_route_section_load_data_q(
        id_linea, rango_hrs, n_sections, section_meters, day_type
    )

    print(
        f"Eliminando datos de carga por tramo para linea {id_linea} "
        f"horas {rango_hrs} tipo de dia {day_type} n_sections  {n_sections}"
        f"section meters {section_meters}"
    )

    print(q_delete)

    cur = conn_data.cursor()
    cur.execute(q_delete)
    conn_data.commit()

    # Read data from legs and route geoms
    q_rec = f"select * from lines_geoms"
    q_main_etapas = f"select * from etapas"

    # If line and hour, get that subset
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

    if dat_type_is_a_date:
        q_main_etapas = q_main_etapas + f" and dia = '{day_type}'"

    q_etapas = f"""
        select e.*
        from ({q_main_etapas}) e
        where e.od_validado==1
    """

    print("Obteniendo datos de etapas y rutas")

    # get data for legs and route geoms
    etapas = pd.read_sql(q_etapas, conn_data)

    if not dat_type_is_a_date:
        # create a weekday_filter
        weekday_filter = pd.to_datetime(
            etapas.dia, format="%Y-%m-%d").dt.dayofweek < 5

        if day_type == "weekday":
            etapas = etapas.loc[weekday_filter, :]
        else:
            etapas = etapas.loc[~weekday_filter, :]

    recorridos = pd.read_sql(q_rec, conn_insumos)
    recorridos["geometry"] = gpd.GeoSeries.from_wkt(recorridos.wkt)

    # Set which parameter to use to slit route geoms
    if section_meters:
        epsg_m = geo.get_epsg_m()
        # project geoms and get for each geom a n_section
        recorridos = gpd.GeoDataFrame(
            recorridos, geometry="geometry", crs="EPSG:4326"
        ).to_crs(epsg=epsg_m)
        recorridos["n_sections"] = (
            recorridos.geometry.length / section_meters).astype(int)

        if (recorridos.n_sections > 1000).any():
            warnings.warn(
                "Algunos recorridos tienen mas de 1000 segmentos"
                "Puede arrojar resultados imprecisos "
            )

        recorridos = recorridos.to_crs(epsg=4326)
    else:
        recorridos["n_sections"] = n_sections

    print("Computing section load per route ...")

    if len(recorridos) > 0:

        section_load_table = etapas.groupby("id_linea").apply(
            compute_section_load_table,
            recorridos=recorridos,
            rango_hrs=rango_hrs,
            day_type=day_type
        )

        section_load_table = section_load_table.reset_index(drop=True)

        # Add section meters to table
        section_load_table["section_meters"] = section_meters

        section_load_table = section_load_table.reindex(
            columns=[
                "id_linea",
                "day_type",
                "n_sections",
                "section_meters",
                "sentido",
                "section_id",
                "x",
                "y",
                "hora_min",
                "hora_max",
                "cantidad_etapas",
                "prop_etapas",
            ]
        )

        print("Uploading data to db...")

        section_load_table.to_sql(
            "ocupacion_por_linea_tramo", conn_data, if_exists="append",
            index=False,)
    else:
        print('No existen recorridos para las líneas')


def is_date_string(input_str):
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if pattern.match(input_str):
        return True
    else:
        return False


def delete_old_route_section_load_data_q(
    id_linea, rango_hrs, n_sections, section_meters, day_type
):

    # hour range filter
    if rango_hrs:
        hora_min_filter = f"= {rango_hrs[0]}"
        hora_max_filter = f"= {rango_hrs[1]}"
    else:
        hora_min_filter = "is NULL"
        hora_max_filter = "is NULL"

    q_delete = f"""
        delete from ocupacion_por_linea_tramo
        where hora_min {hora_min_filter}
        and hora_max {hora_max_filter}
        and day_type = '{day_type}'
        """

    # route id filter
    if id_linea:

        if type(id_linea) == int:
            id_linea = [id_linea]

        lineas_str = ",".join(map(str, id_linea))

        q_delete = q_delete + f" and id_linea in ({lineas_str})"

    if section_meters:
        q_delete = q_delete + f" and  section_meters = {section_meters}"
    else:
        q_delete = (
            q_delete +
            f" and n_sections = {n_sections} and section_meters is NULL"
        )
    q_delete = q_delete + ";"
    return q_delete


def add_od_lrs_to_legs_from_route(legs_df, route_geom):
    """
    Computes for a legs df with origin and destinarion in h3 (h3_o and h3_d)
    the proyected lrs over a route geom

    Parameters
    ----------
    legs : pandas.DataFrame
        table of legs in a route with columns h3_o and h3_d
    route_geom : shapely LineString
        route geom

    Returns
    ----------
    legs_df : pandas.DataFrame
        table of legs with projected od

    """
    # create Points for origins and destination
    legs_df["o"] = legs_df['h3_o'].map(geo.create_point_from_h3)
    legs_df["d"] = legs_df['h3_d'].map(geo.create_point_from_h3)

    # Assign a route section id
    legs_df["o_proj"] = list(
        map(get_route_section_id, legs_df["o"], itertools.repeat(route_geom))
    )
    legs_df["d_proj"] = list(
        map(get_route_section_id, legs_df["d"], itertools.repeat(route_geom))
    )

    return legs_df


def compute_section_load_table(
        df, recorridos, rango_hrs, day_type, *args, **kwargs):
    """
    Computes for a route a table with the load per section

    Parameters
    ----------
    df : pandas.DataFrame
        table of legs in a route
    recorridos : geopandas.GeoDataFrame
        routes geoms
    rango_hrs : tuple
        tuple holding hourly range (from,to).

    Returns
    ----------
    legs_by_sections_full : pandas.DataFrame
        table of section load stats per route id, hour range
        and day type

    """

    id_linea = df.id_linea.unique()[0]
    n_sections = recorridos.n_sections.unique()[0]

    print(f"Computing section load id_route {id_linea}")

    if (recorridos.id_linea == id_linea).any():

        route_geom = recorridos.loc[recorridos.id_linea ==
                                    id_linea, "geometry"].item()

        df = add_od_lrs_to_legs_from_route(legs_df=df, route_geom=route_geom)

        # Assign a direction based on line progression
        df = df.reindex(
            columns=["dia", "o_proj", "d_proj", "factor_expansion_linea"])
        df["sentido"] = [
            "ida" if row.o_proj <= row.d_proj
            else "vuelta" for _, row in df.iterrows()
        ]

        # Compute total legs per direction
        # First totals per day
        totals_by_direction = df\
            .groupby(["dia", "sentido"], as_index=False)\
            .agg(cant_etapas_sentido=("factor_expansion_linea", "sum"))

        # then average for weekdays
        totals_by_direction = totals_by_direction\
            .groupby(["sentido"], as_index=False)\
            .agg(cant_etapas_sentido=("cant_etapas_sentido", "mean"))

        # compute section ids based on amount of sections
        section_ids = create_route_section_ids(n_sections)

        # For each leg, build traversed route segments ids
        legs_dict = df.to_dict("records")
        leg_route_sections_df = pd.concat(
            map(build_leg_route_sections_df, legs_dict,
                itertools.repeat(section_ids))
        )

        # compute total legs by section and direction
        # first adding totals per day
        legs_by_sections = leg_route_sections_df\
            .groupby(["dia", "sentido", "section_id"], as_index=False)\
            .agg(size=("factor_expansion_linea", "sum"))

        # then computing average across days
        legs_by_sections = legs_by_sections\
            .groupby(["sentido", "section_id"], as_index=False)\
            .agg(size=("size", "mean"))

        # If there is no information for all sections in both directions
        if len(legs_by_sections) < len(section_ids) * 2:
            section_direction_full_set = pd.DataFrame(
                {
                    "sentido": ["ida", "vuelta"] * len(section_ids),
                    "section_id": np.repeat(section_ids, 2),
                    "size": [0] * len(section_ids) * 2,
                }
            )

            legs_by_sections_full = section_direction_full_set.merge(
                legs_by_sections, how="left", on=["sentido", "section_id"]
            )
            legs_by_sections_full["cantidad_etapas"] = (
                legs_by_sections_full.size_y.combine_first(
                    legs_by_sections_full.size_x)
            )

            legs_by_sections_full = legs_by_sections_full.reindex(
                columns=["sentido", "section_id", "cantidad_etapas"]
            )

        else:
            legs_by_sections_full = legs_by_sections.rename(
                columns={"size": "cantidad_etapas"}
            )

        # sum totals per direction and compute prop_etapas
        legs_by_sections_full = legs_by_sections_full.merge(
            totals_by_direction, how="left", on="sentido"
        )

        legs_by_sections_full["prop_etapas"] = (
            legs_by_sections_full["cantidad_etapas"]
            / legs_by_sections_full.cant_etapas_sentido
        )

        legs_by_sections_full.prop_etapas = (
            legs_by_sections_full.prop_etapas.fillna(0)
        )

        legs_by_sections_full = legs_by_sections_full.drop(
            "cant_etapas_sentido", axis=1
        )
        legs_by_sections_full["id_linea"] = id_linea

        # Add hourly range
        if rango_hrs:
            legs_by_sections_full["hora_min"] = rango_hrs[0]
            legs_by_sections_full["hora_max"] = rango_hrs[1]
        else:
            legs_by_sections_full["hora_min"] = None
            legs_by_sections_full["hora_max"] = None

        # Add data for type of day and n sections

        legs_by_sections_full["day_type"] = day_type
        legs_by_sections_full["n_sections"] = n_sections

        # Add section geom reference
        geom = [route_geom.interpolate(section_id, normalized=True)
                for section_id in section_ids]
        x = [g.x for g in geom]
        y = [g.y for g in geom]
        section_ids_coords = pd.DataFrame({
            'section_id': section_ids,
            'x': x,
            'y': y
        })
        legs_by_sections_full = legs_by_sections_full.merge(
            section_ids_coords,
            on='section_id',
            how='left'
        )
        # Set db schema
        legs_by_sections_full = legs_by_sections_full.reindex(
            columns=[
                "id_linea",
                "day_type",
                "n_sections",
                "sentido",
                "section_id",
                "x",
                "y",
                "hora_min",
                "hora_max",
                "cantidad_etapas",
                "prop_etapas",
            ]
        )

        return legs_by_sections_full
    else:
        print("No existe recorrido para id_linea:", id_linea)


def create_route_section_ids(n_sections):
    step = 1 / n_sections
    sections = np.arange(0, 1 + step, step)
    section_ids = pd.Series(map(floor_rounding, sections))
    return section_ids


def build_leg_route_sections_df(row, section_ids):
    """
    Computes for a route a table with the load per section

    Parameters
    ----------
    row : dict
        row in a legs df with origin, destination and direction
    section_ids : list
        list of sections ids into which classify legs trajectory

    Returns
    ----------
    leg_route_sections_df: pandas.DataFrame
        a dataframe with all section ids traversed by the leg's
        trajectory

    """

    sentido = row["sentido"]
    dia = row["dia"]
    f_exp = row["factor_expansion_linea"]

    if sentido == "ida":
        point_o = row["o_proj"]
        point_d = row["d_proj"]
    else:
        point_o = row["d_proj"]
        point_d = row["o_proj"]

    o_id = section_ids - point_o
    o_id = o_id[o_id <= 0]
    o_id = o_id.idxmax()

    d_id = section_ids - point_d
    d_id = d_id[d_id >= 0]
    d_id = d_id.idxmin()

    leg_route_sections = section_ids[o_id: d_id + 1]
    leg_route_sections_df = pd.DataFrame(
        {
            "dia": [dia] * len(leg_route_sections),
            "sentido": [sentido] * len(leg_route_sections),
            "section_id": leg_route_sections,
            "factor_expansion_linea": [f_exp] * len(leg_route_sections),
        }
    )
    return leg_route_sections_df


def floor_rounding(num):
    """
    Rounds a number to the floor at 3 digits to use as route section id
    """
    return floor(num * 1000) / 1000


def get_route_section_id(point, route_geom):
    """
    Computes the route section id as a 3 digit float projecing
    a point on to the route geom in a normalized way

    Parameters
    ----------
    point : shapely Point
        a Point for the leg's origin or destination
    route_geom : shapely Linestring
        a Linestring representing the leg's route geom
    """
    return floor_rounding(route_geom.project(point, normalized=True))

# GENERAL PURPOSE KPIS


def read_data_for_daily_kpi():
    """
    Read legs and gps micro data from db and
    merges distances to legs

    Parameters
    ----------
    None

    Returns
    -------
    legs: pandas.DataFrame
        data frame with legs data

    gps: pandas.DataFrame
        gps vehicle tracking data
    """

    conn_data = iniciar_conexion_db(tipo="data")

    # get day with stats computed
    processed_days_q = """
    select distinct dia
    from indicadores_operativos_linea
    """
    processed_days = pd.read_sql(processed_days_q, conn_data)
    processed_days = processed_days.dia
    processed_days = ', '.join([f"'{val}'" for val in processed_days])

    print("Leyendo datos de oferta")
    q = f"""
    select * from gps
    where dia not in ({processed_days})
    order by dia, id_linea, interno, fecha
    """
    gps = pd.read_sql(q, conn_data)

    print("Leyendo datos de demanda")
    q = f"""
        SELECT e.dia,e.id_linea,e.interno,e.id_tarjeta,e.h3_o,
        e.h3_d, e.factor_expansion_linea
        from etapas e
        where e.od_validado==1
        and dia not in ({processed_days})
    """
    legs = pd.read_sql(q, conn_data)

    if (len(gps) > 0) & (len(legs) > 0):
        # add distances
        legs = add_distances_to_legs(legs)
    else:
        print("No hay datos sin KPI procesados")
        legs = pd.DataFrame()
        gps = pd.DataFrame()
    print("Fin carga de datos de oferta y demanda")
    return legs, gps


def add_distances_to_legs(legs):
    """
    Takes legs data and add distances to each leg

    Parameters
    ----------
    legs : pandas.DataFrame
        DataFrame with legs data

    Returns
    -------
    legs : pandas.DataFrame
        DataFrame with legs and distances data

    """
    configs = leer_configs_generales()
    h3_original_res = configs['resolucion_h3']
    min_distance = h3.edge_length(resolution=h3_original_res, unit="km")

    conn_insumos = iniciar_conexion_db(tipo="insumos")

    print("Leyendo distancias")
    distances = pd.read_sql_query(
        """
        SELECT *
        FROM distancias
        """,
        conn_insumos,
    )

    # TODO: USE DIFERENT DISTANCES, GRAPH

    print("Sumando distancias a etapas")
    # use distances h3 when osm missing
    distances.loc[:, ['distance']] = (
        distances.distance_osm_drive.combine_first(distances.distance_h3)
    )
    distances = distances.reindex(columns=["h3_o", "h3_d", "distance"])

    # add legs' distances
    legs = legs.merge(distances, how="left", on=["h3_o", "h3_d"])

    # add minimum distance in km as length of h3
    legs.distance = legs.distance.map(lambda x: max(x, min_distance))

    no_distance = legs.distance.isna().sum()/len(legs) * 100
    print("Hay un {:.2f} % de etapas sin distancias ".format(no_distance))
    conn_insumos.close()

    return legs


@duracion
def compute_kpi_by_line_day(legs, gps):
    """
    Takes data for supply and demand and computes KPI at line level
    for each day

    Parameters
    ----------
    legs : pandas.DataFrame
        DataFrame with legs data

    gps : pandas.DataFrame
        DataFrame with vehicle gps data

    Returns
    -------
    None

    """
    conn_data = iniciar_conexion_db(tipo="data")

    # demand data
    day_demand_stats = legs\
        .groupby(['id_linea', 'dia'], as_index=False)\
        .apply(demand_stats)

    # supply data
    day_supply_stats = gps\
        .groupby(['id_linea', 'dia'], as_index=False)\
        .apply(supply_stats)

    day_stats = day_demand_stats\
        .merge(day_supply_stats,
               how='inner', on=['id_linea', 'dia'])

    # compute KPI
    day_stats['pvd'] = day_stats.tot_pax / \
        day_stats.tot_veh
    day_stats['kvd'] = day_stats.tot_km / \
        day_stats.tot_veh
    day_stats['ipk'] = day_stats.tot_pax / \
        day_stats.tot_km

    # Calcular espacios-km ofertados (EKO) y los espacios-km demandados (EKD).
    day_stats['ekd_mean'] = day_stats.tot_pax * \
        day_stats.dmt_mean
    day_stats['ekd_median'] = day_stats.tot_pax * \
        day_stats.dmt_median
    day_stats['eko'] = day_stats.tot_km * 60

    day_stats['fo_mean'] = day_stats.ekd_mean / \
        day_stats.eko
    day_stats['fo_median'] = day_stats.ekd_median / \
        day_stats.eko

    cols = [
        "id_linea",
        "dia",
        "tot_veh",
        "tot_km",
        "tot_pax",
        "dmt_mean",
        "dmt_median",
        "pvd",
        "kvd",
        "ipk",
        "fo_mean",
        "fo_median"
    ]

    day_stats = day_stats.reindex(columns=cols)

    day_stats.to_sql(
        "indicadores_operativos_linea",
        conn_data,
        if_exists="append",
        index=False,
    )

    return day_stats


@duracion
def compute_kpi_by_line_typeday():
    """
    Takes data for supply and demand and computes KPI at line level
    for weekday and weekend

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    conn_data = iniciar_conexion_db(tipo="data")

    # delete old data
    delete_q = """
    DELETE FROM indicadores_operativos_linea
    where dia in ('weekday','weekend')
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    # read daily data
    q = """
    select * from indicadores_operativos_linea
    """
    daily_data = pd.read_sql(q, conn_data)

    # get day of the week
    weekend = pd.to_datetime(daily_data['dia'].copy()).dt.dayofweek > 4
    daily_data.loc[:, ['dia']] = 'weekday'
    daily_data.loc[weekend, ['dia']] = 'weekend'

    # compute aggregated stats
    type_of_day_stats = daily_data\
        .groupby(['id_linea', 'dia'], as_index=False)\
        .mean()

    print("Subiendo indicadores por linea a la db")

    cols = [
        "id_linea",
        "dia",
        "tot_veh",
        "tot_km",
        "tot_pax",
        "dmt_mean",
        "dmt_median",
        "pvd",
        "kvd",
        "ipk",
        "fo_mean",
        "fo_median"
    ]

    type_of_day_stats = type_of_day_stats.reindex(columns=cols)

    type_of_day_stats.to_sql(
        "indicadores_operativos_linea",
        conn_data,
        if_exists="append",
        index=False,
    )

    return type_of_day_stats


@duracion
def compute_kpi_by_service():
    """
    Reads supply and demand data and computes KPI at service level
    for each day

    Parameters
    ----------
    legs : pandas.DataFrame
        DataFrame with legs data

    gps : pandas.DataFrame
        DataFrame with vehicle gps data

    Returns
    -------
    None

    """

    conn_data = iniciar_conexion_db(tipo="data")

    print("Leyendo demanda por servicios validos")
    q_valid_services = """
        with fechas_procesadas as (
            select distinct dia from indicadores_operativos_servicio
        ),
        demand as (
            select e.id_tarjeta, e.id, id_linea, dia, interno,
              cast(strftime('%s',(dia||' '||tiempo)) as int) as ts, tiempo,
            e.h3_o,
            e.h3_d, e.factor_expansion_linea
            from etapas e
            where od_validado = 1
            and id_linea in (select distinct id_linea from gps)
            and dia not in fechas_procesadas
        ),
        valid_services as (
            select id_linea,dia,interno, service_id, min_ts, max_ts
            from services
            where valid = 1
        ),
        valid_demand as (
            select d.*, s.service_id
            from demand d
            join valid_services s
            on d.id_linea = s.id_linea
            and d.dia = s.dia
            and d.interno = s.interno
            and d.ts >= s.min_ts
            and d.ts <= s.max_ts
            )
            select * from valid_demand
        ;
        """

    valid_demand = pd.read_sql(q_valid_services, conn_data)

    print("Leyendo demanda por servicios invalidos")
    q_invalid_services = """
        with fechas_procesadas as (
            select distinct dia from indicadores_operativos_servicio
        ),
        demand as (
            select e.id_tarjeta, e.id, id_linea, dia, interno,
              cast(strftime('%s',(dia||' '||tiempo)) as int) as ts, tiempo,
            e.h3_o,
            e.h3_d, e.factor_expansion_linea
            from etapas e
            where od_validado = 1
            and id_linea in (select distinct id_linea from gps)
            and dia not in fechas_procesadas
        ),
        valid_services as (
            select id_linea,dia,interno, service_id, min_ts, max_ts
            from services
            where valid = 1
        ),
        invalid_demand as (
            select d.*, s.service_id
            from demand d
            left join valid_services s
            on d.id_linea = s.id_linea
            and d.dia = s.dia
            and d.interno = s.interno
            and d.ts >= s.min_ts
            and d.ts <= s.max_ts
            ),
        legs_no_service as (
            select e.id_tarjeta, e.id, id_linea, dia, interno, ts,
              tiempo, h3_o, h3_d,factor_expansion_linea
            from invalid_demand e
            where service_id is null
        )
        select d.*, s.service_id
        from legs_no_service d
        left join valid_services s
        on d.id_linea = s.id_linea
        and d.dia = s.dia
        and d.interno = s.interno
        and d.ts <= s.min_ts -- valid services begining after the leg start
        order by d.id_tarjeta,d.dia,d.id_linea,d.interno, s.min_ts asc
        ;
        """

    invalid_demand_dups = pd.read_sql(q_invalid_services, conn_data)

    # remove duplicates leaving the first, i.e. next valid service in time
    invalid_demand = invalid_demand_dups.drop_duplicates(
        subset=['id'], keep='first')
    invalid_demand = invalid_demand.dropna(subset=['service_id'])

    # create single demand by service df
    service_demand = pd.concat([valid_demand, invalid_demand])

    # add distances to demand data
    service_demand = add_distances_to_legs(legs=service_demand)

    # TODO: remove this line when factor is corrected
    service_demand['factor_expansion_linea'] = (
        service_demand['factor_expansion_linea'].replace(0, 1)
    )
    # compute demand stats
    service_demand_stats = service_demand\
        .groupby(['dia', 'id_linea', 'interno', 'service_id'], as_index=False)\
        .apply(demand_stats)

    # read supply service data
    service_supply_q = """
        select
            dia,id_linea,interno,service_id,
            distance_km as tot_km, min_datetime,max_datetime
        from
            services where valid = 1
        """
    service_supply = pd.read_sql(service_supply_q, conn_data)

    # merge supply and demand data
    service_stats = service_supply\
        .merge(service_demand_stats, how='left',
               on=['dia', 'id_linea', 'interno', 'service_id'])
    service_stats.tot_pax = service_stats.tot_pax.fillna(0)

    # compute stats
    service_stats['ipk'] = service_stats['tot_pax'] / service_stats['tot_km']
    service_stats['ekd_mean'] = service_stats['tot_pax'] * \
        service_stats['dmt_mean']
    service_stats['ekd_median'] = service_stats['tot_pax'] * \
        service_stats['dmt_median']
    service_stats['eko'] = service_stats['tot_km'] * 60
    service_stats['fo_mean'] = service_stats['ekd_mean'] / service_stats['eko']
    service_stats['fo_median'] = service_stats['ekd_median'] / \
        service_stats['eko']

    service_stats['hora_inicio'] = service_stats.min_datetime.str[10:13].map(
        int)
    service_stats['hora_fin'] = service_stats.max_datetime.str[10:13].map(int)

    # reindex to meet schema
    cols = ['id_linea', 'dia', 'interno', 'service_id',
            'hora_inicio', 'hora_fin', 'tot_km', 'tot_pax', 'dmt_mean',
            'dmt_median', 'ipk', 'fo_mean', 'fo_median']

    service_stats = service_stats.reindex(columns=cols)

    service_stats.to_sql(
        "indicadores_operativos_servicio",
        conn_data,
        if_exists="append",
        index=False,
    )

    return service_stats


def demand_stats(df):
    d = {}
    d["tot_pax"] = df["factor_expansion_linea"].sum()
    d["dmt_mean"] = np.average(
        a=df['distance'], weights=df.factor_expansion_linea)
    d["dmt_median"] = ws.weighted_median(
        data=df['distance'].tolist(),
        weights=df.factor_expansion_linea.tolist()
    )

    return pd.Series(d, index=["tot_pax", "dmt_mean", "dmt_median"])


def supply_stats(df):
    d = {}
    d["tot_veh"] = len(df.interno.unique())
    d["tot_km"] = df.distance_km.sum()

    return pd.Series(d, index=["tot_veh", "tot_km"])


# SERVICES' KPIS

@duracion
def compute_dispatched_services_by_line_hour_day():
    """
    Reads services' data and computes how many services
    by line, day and hour

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    conn_data = iniciar_conexion_db(tipo="data")
    conn_dash = iniciar_conexion_db(tipo="dash")

    processed_days_q = """
    select distinct dia
    from services_by_line_hour
    """
    processed_days = pd.read_sql(processed_days_q, conn_data)
    processed_days = processed_days.dia
    processed_days = ', '.join([f"'{val}'" for val in processed_days])

    print("Leyendo datos de servicios")

    daily_services_q = f"""
    select
        id_linea, dia, min_datetime
    from
        services
    where
        valid = 1
    and dia not in ({processed_days})
    ;
    """

    daily_services = pd.read_sql(daily_services_q, conn_data)

    if len(daily_services) > 0:

        print("Procesando servicios por hora")

        daily_services['hora'] = daily_services.min_datetime.str[10:13].map(
            int)

        daily_services = daily_services.drop(['min_datetime'], axis=1)

        # computing services by hour
        dispatched_services_stats = daily_services\
            .groupby(['id_linea', 'dia', 'hora'], as_index=False)\
            .agg(servicios=('hora', 'count'))

        print("Fin procesamiento servicios por hora")

        print("Subiendo datos a la DB")

        cols = [
            "id_linea",
            "dia",
            "hora",
            "servicios"
        ]

        dispatched_services_stats = dispatched_services_stats.reindex(
            columns=cols)

        dispatched_services_stats.to_sql(
            "services_by_line_hour",
            conn_data,
            if_exists="append",
            index=False,
        )

        dispatched_services_stats.to_sql(
            "services_by_line_hour",
            conn_dash,
            if_exists="append",
            index=False,
        )
        conn_data.close()
        conn_dash.close()

        print("Datos subidos a la DB")
    else:
        print("Todos los dias fueron procesados")


@duracion
def compute_dispatched_services_by_line_hour_typeday():
    """
    Reads services' data and computes how many services
    by line, type of day (weekday weekend), and hour

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    conn_data = iniciar_conexion_db(tipo="data")
    conn_dash = iniciar_conexion_db(tipo="dash")

    # delete old data
    delete_q = """
    DELETE FROM services_by_line_hour
    where dia in ('weekday','weekend')
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    # read daily data
    q = """
    select * from services_by_line_hour
    """
    daily_data = pd.read_sql(q, conn_data)

    if len(daily_data) > 0:

        print("Procesando servicios por tipo de dia")

        # get day of the week
        weekend = pd.to_datetime(daily_data['dia'].copy()).dt.dayofweek > 4
        daily_data.loc[:, ['dia']] = 'weekday'
        daily_data.loc[weekend, ['dia']] = 'weekend'

        # compute aggregated stats
        type_of_day_stats = daily_data\
            .groupby(['id_linea', 'dia', 'hora'], as_index=False)\
            .mean()

        print("Subiendo datos a la DB")

        cols = [
            "id_linea",
            "dia",
            "hora",
            "servicios"
        ]

        type_of_day_stats = type_of_day_stats.reindex(columns=cols)

        type_of_day_stats.to_sql(
            "services_by_line_hour",
            conn_data,
            if_exists="append",
            index=False,
        )

        # delete old dash data
        delete_q = """
        DELETE FROM services_by_line_hour
        where dia in ('weekday','weekend')
        """
        conn_dash.execute(delete_q)
        conn_dash.commit()

        type_of_day_stats.to_sql(
            "services_by_line_hour",
            conn_dash,
            if_exists="append",
            index=False,
        )
        conn_data.close()
        conn_dash.close()

        print("Datos subidos a la DB")

    else:
        print("No hay datos de servicios por hora")
        print("Correr la funcion kpi.compute_services_by_line_hour_day()")

    return type_of_day_stats
