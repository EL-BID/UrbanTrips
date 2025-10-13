import itertools
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import weightedstats as ws
import h3
from urbantrips.geo import geo
from urbantrips.carto.routes import (
    get_route_geoms_with_sections_data,
    check_exists_route_section_points_table,
    upload_route_section_points_table,
    get_route_section_id,
    build_leg_route_sections_df,
)
from urbantrips.carto.carto import create_route_section_ids
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    leer_configs_generales,
    is_date_string,
    check_date_type,
    create_line_ids_sql_filter,
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
        print("Se calcularán KPI básicos en base a datos de demanda")

    # runing basic kpi
    run_basic_kpi()

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
        print("Computando estadisticos por servicio")
        # compute KPI by service and day
        compute_kpi_by_service()

        # compute amount of hourly services by line and day
        compute_dispatched_services_by_line_hour_day()

        # compute amount of hourly services by line and type of day
        compute_dispatched_services_by_line_hour_typeday()

    else:

        print("No hay servicios procesados.")
        print("Puede correr la funcion services.process_services()")
        print("si cuenta con una tabla de gps que indique servicios")


# SECTION LOAD KPI


@duracion
def compute_route_section_load(
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

    conn_data = iniciar_conexion_db(tipo="data")

    # read legs data
    legs = read_legs_data_by_line_hours_and_day(line_ids_where, hour_range, day_type)

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
    yr_mos = legs.yr_mo.unique()

    delete_old_route_section_load_data(
        route_geoms, hour_range, day_type, yr_mos, db_type="data"
    )

    # compute section load
    print("Computing section load per route ...")

    if (len(route_geoms) > 0) and (len(legs) > 0):

        section_load_table = legs.groupby(["id_linea", "yr_mo"]).apply(
            compute_section_load_table,
            route_geoms=route_geoms,
            hour_range=hour_range,
            day_type=day_type,
        )

        section_load_table = section_load_table.droplevel(2, axis=0).reset_index()

        # Add section meters to table
        section_load_table["legs"] = section_load_table["legs"].map(int)
        section_load_table = section_load_table.reindex(
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
                "legs",
                "prop",
            ]
        )

        print("Uploading data to db...")
        section_load_table.to_sql(
            "ocupacion_por_linea_tramo",
            conn_data,
            if_exists="append",
            index=False,
        )

        return section_load_table
    else:
        print("No existen recorridos o etapas para las líneas")
        print("Cantidad de lineas:", len(line_ids))
        print("Cantidad de recorridos", len(route_geoms))
        print("Cantidad de etapas", len(legs))


def delete_old_route_section_load_data(
    route_geoms, hour_range, day_type, yr_mos, db_type="data"
):
    """
    Deletes old data in table ocupacion_por_linea_tramo
    """
    table_name = "ocupacion_por_linea_tramo"

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
            print("Borrando datos antiguos de ocupacion_por_linea_tramo")
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
    legs_df["o"] = legs_df["h3_o"].map(geo.create_point_from_h3)
    legs_df["d"] = legs_df["h3_d"].map(geo.create_point_from_h3)

    # Assign a route section id
    legs_df["o_proj"] = list(
        map(get_route_section_id, legs_df["o"], itertools.repeat(route_geom))
    )
    legs_df["d_proj"] = list(
        map(get_route_section_id, legs_df["d"], itertools.repeat(route_geom))
    )

    return legs_df


def compute_section_load_table(legs, route_geoms, hour_range, day_type):
    """
    Computes for a route a table with the load per section

    Parameters
    ----------
    legs : pandas.DataFrame
        table of legs in a route
    route_geoms : geopandas.GeoDataFrame
        routes geoms
    hour_range : tuple
        tuple holding hourly range (from,to).

    Returns
    ----------
    pandas.DataFrame
        table of section load stats per route id, hour range
        and day type

    """

    line_id = legs.id_linea.unique()[0]
    print(f"Calculando carga por tramo para linea id {line_id}")

    if (route_geoms.id_linea == line_id).any():
        route = route_geoms.loc[route_geoms.id_linea == line_id, :]

        route_geom = route.geometry.item()
        n_sections = route.n_sections.item()
        section_meters = route.section_meters.item()

        df = add_od_lrs_to_legs_from_route(legs_df=legs, route_geom=route_geom)

        # Assign a direction based on line progression
        df = df.reindex(columns=["dia", "o_proj", "d_proj", "factor_expansion_linea"])
        df["sentido"] = [
            "ida" if row.o_proj <= row.d_proj else "vuelta" for _, row in df.iterrows()
        ]

        # Compute total legs per direction
        # First totals per day
        totals_by_direction = df.groupby(["dia", "sentido"], as_index=False).agg(
            cant_etapas_sentido=("factor_expansion_linea", "sum")
        )

        # then average for weekdays
        totals_by_direction = totals_by_direction.groupby(
            ["sentido"], as_index=False
        ).agg(cant_etapas_sentido=("cant_etapas_sentido", "mean"))

        # compute section ids based on amount of sections
        section_ids_LRS = create_route_section_ids(n_sections)
        # remove 0 form cuts so 0 gets included in bin
        section_ids_LRS_cut = section_ids_LRS.copy()
        section_ids_LRS_cut.loc[0] = -0.001

        # For each leg, build traversed route segments ids
        section_ids = list(range(1, len(section_ids_LRS_cut)))

        df["o_proj"] = pd.cut(
            df.o_proj, bins=section_ids_LRS_cut, labels=section_ids, right=True
        )
        df["d_proj"] = pd.cut(
            df.d_proj, bins=section_ids_LRS_cut, labels=section_ids, right=True
        )

        # remove legs with no origin or destination projected
        df = df.dropna(subset=["o_proj", "d_proj"])

        legs_dict = df.to_dict("records")
        leg_route_sections_df = pd.concat(map(build_leg_route_sections_df, legs_dict))

        # compute total legs by section and direction
        # first adding totals per day
        legs_by_sections = leg_route_sections_df.groupby(
            ["dia", "sentido", "section_id"], as_index=False
        ).agg(size=("factor_expansion_linea", "sum"))

        # then computing average across days
        legs_by_sections = legs_by_sections.groupby(
            ["sentido", "section_id"], as_index=False
        ).agg(size=("size", "mean"))

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
            legs_by_sections_full["legs"] = legs_by_sections_full.size_y.combine_first(
                legs_by_sections_full.size_x
            )

            legs_by_sections_full = legs_by_sections_full.reindex(
                columns=["sentido", "section_id", "legs"]
            )

        else:
            legs_by_sections_full = legs_by_sections.rename(columns={"size": "legs"})

        # sum totals per direction and compute prop_etapas
        legs_by_sections_full = legs_by_sections_full.merge(
            totals_by_direction, how="left", on="sentido"
        )

        legs_by_sections_full["prop"] = (
            legs_by_sections_full["legs"] / legs_by_sections_full.cant_etapas_sentido
        )

        legs_by_sections_full["prop"] = legs_by_sections_full["prop"].fillna(0)

        legs_by_sections_full["id_linea"] = line_id

        # Add hourly range
        if hour_range:
            legs_by_sections_full["hour_min"] = hour_range[0]
            legs_by_sections_full["hour_max"] = hour_range[1]
        else:
            legs_by_sections_full["hour_min"] = None
            legs_by_sections_full["hour_max"] = None

        # Add data for type of day and n sections

        legs_by_sections_full["day_type"] = day_type
        legs_by_sections_full["n_sections"] = n_sections
        legs_by_sections_full["section_meters"] = section_meters

        # Set db schema
        legs_by_sections_full = legs_by_sections_full.reindex(
            columns=[
                "day_type",
                "n_sections",
                "section_meters",
                "sentido",
                "section_id",
                "hour_min",
                "hour_max",
                "legs",
                "prop",
            ]
        )

        return legs_by_sections_full
    else:
        print("No existe recorrido para id_linea:", line_id)


# GENERAL PURPOSE KPIS WITH GPS


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

    cur = conn_data.cursor()
    q = """
        SELECT tbl_name FROM sqlite_master
        WHERE type='table'
        AND tbl_name='gps';
    """
    listOfTables = cur.execute(q).fetchall()

    if listOfTables == []:
        print("No existe tabla GPS en la base")
        print("No se pudeden computar indicadores de oferta usando GPS")

        legs = pd.DataFrame()
        gps = pd.DataFrame()

        return legs, gps

    print("Leyendo datos de oferta")
    q = f"""
    select g.* from gps g
    JOIN dias_ultima_corrida d
    ON g.dia = d.dia
    order by g.dia, id_linea, interno, fecha
    """
    gps = pd.read_sql(q, conn_data)

    print("Leyendo datos de demanda")
    q = f"""
        SELECT e.dia,e.id_linea,e.interno,e.id_tarjeta,e.h3_o,
        e.h3_d, e.factor_expansion_linea
        from etapas e
        JOIN dias_ultima_corrida d
        ON e.dia = d.dia
        where e.od_validado==1
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
    h3_original_res = configs["resolucion_h3"]
    min_distance = h3.average_hexagon_edge_length(res=h3_original_res, unit="km")

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

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
    distances.loc[:, ["distance"]] = distances.distance_osm_drive.combine_first(
        distances.distance_h3
    )
    distances = distances.reindex(columns=["h3_o", "h3_d", "distance"])

    # add legs' distances
    legs = legs.merge(distances, how="left", on=["h3_o", "h3_d"])

    # add minimum distance in km as length of h3
    legs.distance = legs.distance.map(lambda x: max(x, min_distance))

    no_distance = legs.distance.isna().sum() / len(legs) * 100
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

    # get veh expansion factors for supply data
    q = "select id_linea,dia,veh_exp from vehicle_expansion_factors"
    vehicle_expansion_factor = pd.read_sql(q, conn_data)
    gps = gps.merge(vehicle_expansion_factor, on=["dia", "id_linea"], how="left")

    # demand data
    day_demand_stats = (
        legs.dropna(subset=["distance", "factor_expansion_linea"])
        .groupby(["id_linea", "dia"], as_index=False)
        .apply(demand_stats)
    )

    # supply data
    day_supply_stats = gps.groupby(["id_linea", "dia"], as_index=False).apply(
        supply_stats
    )

    day_stats = day_demand_stats.merge(
        day_supply_stats, how="inner", on=["id_linea", "dia"]
    )

    # compute KPI
    day_stats["pvd"] = day_stats.tot_pax / day_stats.tot_veh
    day_stats["kvd"] = day_stats.tot_km / day_stats.tot_veh
    day_stats["ipk"] = day_stats.tot_pax / day_stats.tot_km

    # Calcular espacios-km ofertados (EKO) y los espacios-km demandados (EKD).
    day_stats["ekd_mean"] = day_stats.tot_pax * day_stats.dmt_mean
    day_stats["ekd_median"] = day_stats.tot_pax * day_stats.dmt_median
    day_stats["eko"] = day_stats.tot_km * 60

    day_stats["fo_mean"] = day_stats.ekd_mean / day_stats.eko
    day_stats["fo_median"] = day_stats.ekd_median / day_stats.eko

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
        "fo_median",
    ]

    day_stats = day_stats.reindex(columns=cols)

    # get last processed days
    dias_ultima_corrida = pd.read_sql_query(
        """SELECT * FROM dias_ultima_corrida""",
        conn_data,
    )
    # borro si ya existen etapas de una corrida anterior
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM kpi_by_day_line WHERE dia IN ({values})"
    conn_data.execute(query)
    conn_data.commit()

    day_stats.to_sql(
        "kpi_by_day_line",
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
    DELETE FROM kpi_by_day_line
    where dia in ('weekday','weekend')
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    # read daily data
    q = """
    select * from kpi_by_day_line
    """
    daily_data = pd.read_sql(q, conn_data)

    # get day of the week
    weekend = pd.to_datetime(daily_data["dia"].copy()).dt.dayofweek > 4
    daily_data.loc[:, ["dia"]] = "weekday"
    daily_data.loc[weekend, ["dia"]] = "weekend"

    # compute aggregated stats
    type_of_day_stats = daily_data.groupby(["id_linea", "dia"], as_index=False).mean()

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
        "fo_median",
    ]

    type_of_day_stats = type_of_day_stats.reindex(columns=cols)

    type_of_day_stats.to_sql(
        "kpi_by_day_line",
        conn_data,
        if_exists="append",
        index=False,
    )

    return type_of_day_stats


# KPIS BY SERVICE


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
        WITH demand AS (
        SELECT
            e.id_tarjeta, e.id, e.id_linea, e.dia, e.id_ramal, e.interno,
            CAST(strftime('%s',(e.dia||' '||e.tiempo)) AS INTEGER) AS ts,
            e.tiempo, e.h3_o, e.h3_d, e.factor_expansion_linea
        FROM etapas e
        JOIN dias_ultima_corrida d
            ON e.dia = d.dia
        WHERE e.od_validado = 1
            AND EXISTS (SELECT 1 FROM gps g WHERE g.id_linea = e.id_linea)
        ),
        valid_services AS (
        SELECT id_linea, dia, id_ramal, interno, service_id, min_ts, max_ts
        FROM services
        WHERE valid = 1
        )
        SELECT d.*, s.service_id
        FROM demand d
        JOIN valid_services s
        ON d.id_linea = s.id_linea
        AND d.dia      = s.dia
        AND d.id_ramal = s.id_ramal
        AND d.interno  = s.interno
        AND d.ts BETWEEN s.min_ts AND s.max_ts;
        """

    valid_demand = pd.read_sql(q_valid_services, conn_data)

    print("Leyendo demanda por servicios invalidos")
    q_invalid_services = """
        WITH demand as (
            select e.id_tarjeta, e.id, e.id_linea, e.dia, e.id_ramal, e.interno,
                cast(strftime('%s',(e.dia||' '||e.tiempo)) as int) as ts, e.tiempo,
            e.h3_o,
            e.h3_d, e.factor_expansion_linea
        FROM etapas e
        JOIN dias_ultima_corrida d
            ON e.dia = d.dia
        WHERE od_validado = 1
        AND EXISTS (SELECT 1 FROM gps g WHERE g.id_linea = e.id_linea)

        ),
        valid_services as (
            select id_linea,dia,id_ramal,interno, service_id, min_ts, max_ts
            from services
            where valid = 1
        ),
        invalid_demand as (
            select d.*, s.service_id
            from demand d
            left join valid_services s
            on d.id_linea = s.id_linea
            and d.dia = s.dia
            and d.id_ramal = s.id_ramal
            and d.interno = s.interno
            and d.ts >= s.min_ts
            and d.ts <= s.max_ts
            ),
        legs_no_service as (
            select e.id_tarjeta, e.id, id_linea, dia,id_ramal, interno, ts,
                tiempo, h3_o, h3_d,factor_expansion_linea
            from invalid_demand e
            where service_id is null
        )
        select d.*, s.service_id
        from legs_no_service d
        left join valid_services s
        on d.id_linea = s.id_linea
        and d.dia = s.dia
        and d.id_ramal = s.id_ramal
        and d.interno = s.interno
        and d.ts <= s.min_ts -- valid services begining after the leg start
        order by d.id_tarjeta,d.dia,d.id_linea,d.interno, s.min_ts asc
        ;
        """

    invalid_demand_dups = pd.read_sql(q_invalid_services, conn_data)

    # remove duplicates leaving the first, i.e. next valid service in time
    invalid_demand = invalid_demand_dups.drop_duplicates(subset=["id"], keep="first")
    invalid_demand = invalid_demand.dropna(subset=["service_id"])

    # create single demand by service df
    service_demand = pd.concat([valid_demand, invalid_demand])

    # add distances to demand data
    service_demand = add_distances_to_legs(legs=service_demand)

    # compute demand stats
    service_demand_stats = service_demand.groupby(
        ["dia", "id_linea", "id_ramal", "interno", "service_id"], as_index=False
    ).apply(demand_stats)

    # read supply service data
    service_supply_q = """
        select
            dia,id_linea,id_ramal,interno,service_id,
            distance_km as tot_km, min_datetime,max_datetime
        from
            services where valid = 1
        """
    service_supply = pd.read_sql(service_supply_q, conn_data)

    # merge supply and demand data
    service_stats = service_supply.merge(
        service_demand_stats,
        how="left",
        on=["dia", "id_linea", "id_ramal", "interno", "service_id"],
    )
    service_stats.tot_pax = service_stats.tot_pax.fillna(0)

    # compute stats
    service_stats["ipk"] = service_stats["tot_pax"] / service_stats["tot_km"]
    service_stats["ekd_mean"] = service_stats["tot_pax"] * service_stats["dmt_mean"]
    service_stats["ekd_median"] = service_stats["tot_pax"] * service_stats["dmt_median"]
    service_stats["eko"] = service_stats["tot_km"] * 60
    service_stats["fo_mean"] = service_stats["ekd_mean"] / service_stats["eko"]
    service_stats["fo_median"] = service_stats["ekd_median"] / service_stats["eko"]

    service_stats["hora_inicio"] = service_stats.min_datetime.str[10:13].map(int)
    service_stats["hora_fin"] = service_stats.max_datetime.str[10:13].map(int)

    # reindex to meet schema
    cols = [
        "id_linea",
        "dia",
        "id_ramal",
        "interno",
        "service_id",
        "hora_inicio",
        "hora_fin",
        "tot_km",
        "tot_pax",
        "dmt_mean",
        "dmt_median",
        "ipk",
        "fo_mean",
        "fo_median",
    ]

    service_stats = service_stats.reindex(columns=cols)

    # get last processed days
    dias_ultima_corrida = pd.read_sql_query(
        """SELECT * FROM dias_ultima_corrida""",
        conn_data,
    )
    # borro si ya existen etapas de una corrida anterior
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM kpi_by_day_line_service WHERE dia IN ({values})"
    conn_data.execute(query)
    conn_data.commit()

    service_stats.to_sql(
        "kpi_by_day_line_service",
        conn_data,
        if_exists="append",
        index=False,
    )

    return service_stats


def demand_stats(df):
    d = {}
    d["tot_pax"] = df["factor_expansion_linea"].sum()
    d["dmt_mean"] = np.average(a=df["distance"], weights=df.factor_expansion_linea)
    d["dmt_median"] = ws.weighted_median(
        data=df["distance"].tolist(), weights=df.factor_expansion_linea.tolist()
    )

    return pd.Series(d, index=["tot_pax", "dmt_mean", "dmt_median"])


def supply_stats(df):
    d = {}
    d["tot_veh"] = len(df.interno.unique()) * df.veh_exp.unique()[0]
    d["tot_km"] = df.distance_km.sum() * df.veh_exp.unique()[0]

    return pd.Series(d, index=["tot_veh", "tot_km"])


# GENERAL PURPOSE KPI WITH NO GPS


def compute_speed_by_day_veh_hour():
    """
    This function read gps data and computes
    average speed by veh for each day and line
    """
    conn_data = iniciar_conexion_db(tipo="data")
    processed_days = get_processed_days(table_name="basic_kpi_by_line_day")

    # read data
    q = f"""
    select dia,id_linea,id_ramal,fecha,interno,velocity,distance_km 
    from gps
    where dia not in ({processed_days})
    ;
    """
    gps_df = pd.read_sql(q, conn_data)
    conn_data.close()

    # create a lag in date
    gps_df = gps_df.sort_values(["dia", "id_linea", "id_ramal", "interno", "fecha"])
    gps_df["fecha_lag"] = (
        gps_df.reindex(columns=["dia", "id_linea", "id_ramal", "interno", "fecha"])
        .groupby(["dia", "id_linea", "id_ramal", "interno"])
        .shift(-1)
    )

    # compute delta in time
    gps_df = gps_df.dropna(subset=["fecha", "fecha_lag"])
    gps_df.loc[:, ["delta_hr"]] = (gps_df.fecha_lag - gps_df.fecha) / (60 * 60)
    gps_df = gps_df.loc[gps_df.delta_hr > 0, :]

    # compute speed in kmr
    gps_df.loc[:, ["speed_kmh_veh_h"]] = gps_df.distance_km / gps_df.delta_hr
    gps_df["hora"] = pd.to_datetime(gps_df["fecha"], unit="s").dt.hour

    # get mean speed by day, linea, veh
    speed_vehicle_hour_gps = (
        gps_df.reindex(
            columns=[
                "dia",
                "id_linea",
                "id_ramal",
                "interno",
                "hora",
                "speed_kmh_veh_h",
            ]
        )
        .groupby(["dia", "id_linea", "id_ramal", "interno", "hora"], as_index=False)
        .mean()
    )

    speed_vehicle_hour_gps = speed_vehicle_hour_gps.loc[
        speed_vehicle_hour_gps.speed_kmh_veh_h > 0, :
    ]

    return speed_vehicle_hour_gps


def gps_table_exists():
    conn_data = iniciar_conexion_db(tipo="data")
    cur = conn_data.cursor()
    q = """
        SELECT tbl_name FROM sqlite_master
        WHERE type='table'
        AND tbl_name='gps';
    """
    listOfTables = cur.execute(q).fetchall()
    conn_data.close()
    if listOfTables == []:
        print("No existe tabla GPS en la base")
        print("Se calcularán KPI básicos en base a datos de demanda")
        return False
    else:
        return True


@duracion
def run_basic_kpi(id_linea=[]):
    conn_data = iniciar_conexion_db(tipo="data")

    # read already process days
    processed_days = get_processed_days(table_name="basic_kpi_by_line_day")

    # read unprocessed data from legs

    q = f"""
        select *
        from etapas
        where od_validado = 1
        and dia not in ({processed_days})
    """
    if len(id_linea) > 0:
        id_linea_str = ", ".join(map(str, id_linea))
        q += f" and id_linea in ({id_linea_str})"
    q += ";"
    print("Leyendo datos de demanda")
    legs = pd.read_sql(q, conn_data)

    if len(legs) < 5:
        return None

    legs = add_distances_to_legs(legs=legs)

    # if there is no full timestamp
    if legs["tiempo"].isna().all():

        unique_line_ids = legs.id_linea.unique()
        id_lines = np.repeat(unique_line_ids, 24)
        hours = list(range(0, 24)) * len(unique_line_ids)

        # fix commercial speed at 15kmh for all veh
        speed_vehicle_hour = pd.DataFrame(
            {
                "id_linea": id_lines,
                "hora": hours,
                "speed_kmh_veh_h": [15] * 24 * len(unique_line_ids),
            }
        )
        speed_vehicle_hour = (
            legs.reindex(columns=["dia", "id_linea", "id_ramal", "interno"])
            .drop_duplicates()
            .merge(speed_vehicle_hour, on=["id_linea"], how="left")
        )

    # else compute commercial speed based on gps or demand
    else:
        if gps_table_exists():
            print("Calculando velocidades comerciales usando tabla gps")
            speed_vehicle_hour = compute_speed_by_day_veh_hour()
        else:
            # compute mean veh speed using demand data
            legs.loc[:, ["datetime"]] = legs.dia + " " + legs.tiempo

            legs.loc[:, ["time"]] = pd.to_datetime(
                legs.loc[:, "datetime"], format="%Y-%m-%d %H:%M:%S"
            )

            print("Calculando velocidades comerciales")
            # compute vehicle speed per hour
            speed_vehicle_hour = legs.groupby(
                ["dia", "id_linea", "id_ramal", "interno"]
            ).apply(compute_speed_by_veh_hour)

            speed_vehicle_hour = speed_vehicle_hour.droplevel(4).reset_index()

    # set a max speed te remove outliers
    speed_max = 60
    speed_vehicle_hour.loc[
        speed_vehicle_hour.speed_kmh_veh_h > speed_max, "speed_kmh_veh_h"
    ] = speed_max

    speed_vehicle_hour = speed_vehicle_hour.dropna()

    print("Eliminando casos atipicos en velocidades comerciales")

    # compute standard deviation to remove low speed outliers
    speed_dev = speed_vehicle_hour.groupby(
        ["dia", "id_linea", "id_ramal"], as_index=False
    ).agg(mean=("speed_kmh_veh_h", "mean"), std=("speed_kmh_veh_h", "std"))
    speed_dev["speed_min"] = speed_dev["mean"] - (2 * speed_dev["std"]).map(
        lambda x: max(1, x)
    )
    speed_dev = speed_dev.reindex(columns=["dia", "id_linea", "id_ramal", "speed_min"])

    speed_vehicle_hour = speed_vehicle_hour.merge(
        speed_dev, on=["dia", "id_linea", "id_ramal"], how="left"
    )

    speed_mask = (speed_vehicle_hour.speed_kmh_veh_h < speed_max) & (
        speed_vehicle_hour.speed_kmh_veh_h > speed_vehicle_hour.speed_min
    )

    speed_vehicle_hour = speed_vehicle_hour.loc[
        speed_mask,
        ["dia", "id_linea", "id_ramal", "interno", "hora", "speed_kmh_veh_h"],
    ]

    # compute by hour to fill nans in vehicle speed
    speed_line_hour = (
        speed_vehicle_hour.drop(["id_ramal", "interno"], axis=1)
        .groupby(["dia", "id_linea", "hora"], as_index=False)
        .mean()
        .rename(columns={"speed_kmh_veh_h": "speed_kmh_line_h"})
    )

    speed_line_day = (
        speed_vehicle_hour.drop(["id_ramal", "interno", "hora"], axis=1)
        .groupby(["dia", "id_linea"], as_index=False)
        .mean()
        .rename(columns={"speed_kmh_veh_h": "speed_kmh_line_day"})
    )

    # add commercial speed to demand data
    legs = legs.merge(
        speed_vehicle_hour,
        on=["dia", "id_linea", "id_ramal", "interno", "hora"],
        how="left",
    ).merge(speed_line_hour, on=["dia", "id_linea", "hora"], how="left")

    legs["speed_kmh"] = legs.speed_kmh_veh_h.combine_first(legs.speed_kmh_line_h)

    print("Calculando pasajero equivalente otros KPI por dia" ", linea, interno y hora")

    # get an vehicle space equivalent passenger
    legs["eq_pax"] = (legs.distance / legs.speed_kmh) * legs.factor_expansion_linea

    # COMPUTE KPI BY DAY LINE VEHICLE HOUR
    kpi_by_veh = (
        legs.reindex(
            columns=[
                "dia",
                "id_linea",
                "id_ramal",
                "interno",
                "hora",
                "factor_expansion_linea",
                "eq_pax",
                "distance",
                "speed_kmh",
            ]
        )
        .groupby(["dia", "id_linea", "id_ramal", "interno", "hora"], as_index=False)
        .agg(
            tot_pax=("factor_expansion_linea", "sum"),
            eq_pax=("eq_pax", "sum"),
            dmt=("distance", "mean"),
            speed_kmh=("speed_kmh", "mean"),
        )
    )

    # compute ocupation factor
    kpi_by_veh["of"] = kpi_by_veh.eq_pax / 60 * 100

    # remove outliers
    print("Eliminando casos atipicos en pasajeros equivalentes")

    of_threshold = 120
    of_mask = kpi_by_veh["of"] > of_threshold
    print(
        f"Hay un {round(of_mask.sum()/ len(kpi_by_veh) * 100,1)} % de vehiculos con OF atipicos"
    )

    kpi_by_veh.loc[of_mask, "of"] = None

    print("Subiendo a la base de datos")
    # set schema and upload to db
    cols = [
        "dia",
        "id_linea",
        "id_ramal",
        "interno",
        "hora",
        "tot_pax",
        "eq_pax",
        "dmt",
        "of",
        "speed_kmh",
    ]

    kpi_by_veh = kpi_by_veh.reindex(columns=cols)

    kpi_by_veh.to_sql(
        "basic_kpi_by_vehicle_hr",
        conn_data,
        if_exists="append",
        index=False,
    )

    print("Calculando pasajero equivalente otros KPI por dia, linea y hora")

    # COMPUTE KPI BY DAY LINE HOUR

    # compute ocupation factor
    ocupation_factor_line_hour = (
        kpi_by_veh.reindex(columns=["dia", "id_linea", "hora", "of"])
        .groupby(["dia", "id_linea", "hora"], as_index=False)
        .mean()
    )

    # compute supply as unique vehicles day per hour
    supply = (
        legs.reindex(columns=["dia", "id_linea", "id_ramal", "interno", "hora"])
        .drop_duplicates()
        .groupby(["dia", "id_linea", "hora"])
        .size()
        .reset_index()
        .rename(columns={0: "veh"})
    )

    # compute demand as total legs per hour and DMT
    demand = (
        legs.reindex(
            columns=["dia", "id_linea", "hora", "factor_expansion_linea", "distance"]
        )
        .groupby(["dia", "id_linea", "hora"], as_index=False)
        .agg(pax=("factor_expansion_linea", "sum"), dmt=("distance", "mean"))
    )

    # compute line kpi table
    kpi_by_line_hr = supply.merge(
        demand, on=["dia", "id_linea", "hora"], how="left"
    ).merge(ocupation_factor_line_hour, on=["dia", "id_linea", "hora"], how="left")

    kpi_by_line_hr = kpi_by_line_hr.merge(
        speed_line_hour, on=["dia", "id_linea", "hora"], how="left"
    )
    kpi_by_line_hr = kpi_by_line_hr.rename(columns={"speed_kmh_line_h": "speed_kmh"})

    print("Subiendo a la base de datos")

    # create month
    kpi_by_line_hr["yr_mo"] = kpi_by_line_hr.dia.str[:7]

    # set schema and upload to db
    cols = ["dia", "yr_mo", "id_linea", "hora", "veh", "pax", "dmt", "of", "speed_kmh"]

    kpi_by_line_hr = kpi_by_line_hr.reindex(columns=cols)

    kpi_by_line_hr.to_sql(
        "basic_kpi_by_line_hr",
        conn_data,
        if_exists="append",
        index=False,
    )

    # COMPUTE KPI BY DAY AND LINE
    print("Calculando pasajero equivalente otros KPI por dia y linea")

    # compute daily stats
    ocupation_factor_line = (
        kpi_by_veh.reindex(columns=["dia", "id_linea", "of"])
        .groupby(["dia", "id_linea"], as_index=False)
        .mean()
    )

    # compute supply as unique vehicles day
    daily_supply = (
        legs.reindex(columns=["dia", "id_linea", "id_ramal", "interno"])
        .drop_duplicates()
        .groupby(["dia", "id_linea"])
        .size()
        .reset_index()
        .rename(columns={0: "veh"})
    )

    # compute demand as total legs per hour and DMT
    daily_demand = (
        legs.reindex(columns=["dia", "id_linea", "factor_expansion_linea", "distance"])
        .groupby(["dia", "id_linea"], as_index=False)
        .agg(
            pax=("factor_expansion_linea", "sum"),
            dmt=("distance", "mean"),
        )
    )

    # compute line kpi table
    kpi_by_line_day = daily_supply.merge(
        daily_demand, on=["dia", "id_linea"], how="left"
    ).merge(ocupation_factor_line, on=["dia", "id_linea"], how="left")

    kpi_by_line_day = kpi_by_line_day.merge(
        speed_line_day, on=["dia", "id_linea"], how="left"
    )
    kpi_by_line_day = kpi_by_line_day.rename(
        columns={"speed_kmh_line_day": "speed_kmh"}
    )

    kpi_by_line_day["yr_mo"] = kpi_by_line_day.dia.str[:7]

    print("Subiendo a la base de datos")
    # set schema and upload to db
    cols = ["dia", "yr_mo", "id_linea", "veh", "pax", "dmt", "of", "speed_kmh"]

    kpi_by_line_day = kpi_by_line_day.reindex(columns=cols)

    kpi_by_line_day.to_sql(
        "basic_kpi_by_line_day",
        conn_data,
        if_exists="append",
        index=False,
    )

    # compute aggregated stats by weekday and weekend
    compute_basic_kpi_line_typeday()
    compute_basic_kpi_line_hr_typeday()

    conn_data.close()


def compute_basic_kpi_line_typeday():
    conn_data = iniciar_conexion_db(tipo="data")

    print("Borrando datos desactualizados por tipo de dia")

    # delete old type of day data data
    delete_q = """
    DELETE FROM basic_kpi_by_line_day
    where dia in ('weekday','weekend')
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    print("Calculando KPI basicos por tipo de dia")
    q = """
    select * from basic_kpi_by_line_day;
    """
    kpi_by_line_day = pd.read_sql(q, conn_data)

    # get day of the week
    weekend = pd.to_datetime(kpi_by_line_day["dia"].copy()).dt.dayofweek > 4
    kpi_by_line_day.loc[:, ["dia"]] = "weekday"
    kpi_by_line_day.loc[weekend, ["dia"]] = "weekend"

    # compute aggregated stats
    kpi_by_line_typeday = kpi_by_line_day.groupby(
        [
            "dia",
            "yr_mo",
            "id_linea",
        ],
        as_index=False,
    ).mean()

    print("Subiendo a la base de datos")
    # set schema and upload to db
    cols = ["dia", "yr_mo", "id_linea", "veh", "pax", "dmt", "of", "speed_kmh"]

    kpi_by_line_typeday = kpi_by_line_typeday.reindex(columns=cols)

    kpi_by_line_typeday.to_sql(
        "basic_kpi_by_line_day",
        conn_data,
        if_exists="append",
        index=False,
    )

    conn_data.close()


def compute_basic_kpi_line_hr_typeday():
    conn_data = iniciar_conexion_db(tipo="data")

    print("Borrando datos desactualizados por tipo de dia")

    # delete old type of day data data
    delete_q = """
    DELETE FROM basic_kpi_by_line_hr
    where dia in ('weekday','weekend')
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    print("Calculando KPI basicos por tipo de dia")
    q = """
    select * from basic_kpi_by_line_hr;
    """

    kpi_by_line_hr = pd.read_sql(q, conn_data)

    # get day of the week
    weekend = pd.to_datetime(kpi_by_line_hr["dia"].copy()).dt.dayofweek > 4
    kpi_by_line_hr.loc[:, ["dia"]] = "weekday"
    kpi_by_line_hr.loc[weekend, ["dia"]] = "weekend"

    # compute aggregated stats
    kpi_by_line_typeday = kpi_by_line_hr.groupby(
        ["dia", "yr_mo", "id_linea", "hora"], as_index=False
    ).mean()

    print("Subiendo a la base de datos")
    # set schema and upload to db
    cols = ["dia", "yr_mo", "id_linea", "hora", "veh", "pax", "dmt", "of", "speed_kmh"]

    kpi_by_line_typeday = kpi_by_line_typeday.reindex(columns=cols)

    kpi_by_line_typeday.to_sql(
        "basic_kpi_by_line_hr",
        conn_data,
        if_exists="append",
        index=False,
    )

    conn_data.close()


def compute_speed_by_veh_hour(legs_vehicle):
    try:
        if len(legs_vehicle) < 2:
            return None

        res = 11
        distance_between_hex = h3.average_hexagon_edge_length(res=res, unit="m")
        distance_between_hex = distance_between_hex * 2

        speed = legs_vehicle.reindex(
            columns=["interno", "hora", "time", "latitud", "longitud"]
        )
        speed["h3"] = speed.apply(
            geo.h3_from_row, axis=1, args=(res, "latitud", "longitud")
        )

        # get only one h3 per vehicle hour
        speed = speed.drop_duplicates(subset=["interno", "hora", "h3"])
        if len(speed) < 2:
            return None
        speed = speed.sort_values("time")

        # compute meters between h3
        speed["h3_lag"] = speed["h3"].shift(1)
        speed["time_lag"] = speed["time"].shift(1)

        speed = speed.dropna(subset=["h3_lag", "time_lag"])

        speed["seconds"] = (speed["time"] - speed["time_lag"]).map(
            lambda x: x.total_seconds()
        )

        speed["meters"] = (
            speed.apply(lambda row: h3.grid_distance(row["h3"], row["h3_lag"]), axis=1)
            * distance_between_hex
        )

        speed_by_hour = (
            speed.reindex(columns=["hora", "seconds", "meters"])
            .groupby("hora", as_index=False)
            .agg(
                meters=("meters", "sum"),
                seconds=("seconds", "sum"),
                n=("hora", "count"),
            )
        )
        # remove vehicles with less than 2 pax

        speed_by_hour = speed_by_hour.loc[speed_by_hour.n > 2, :]
        speed_by_hour["speed_kmh_veh_h"] = (
            speed_by_hour.meters / speed_by_hour.seconds * 3.6
        )
        speed_by_hour = speed_by_hour.reindex(columns=["hora", "speed_kmh_veh_h"])

        return speed_by_hour
    except:
        return None


def get_processed_days(table_name):
    """
    Takes a table name and returns all days present in
    that table

    Parameters
    ----------
    table_name : str
        name of the table with processed data

    Returns
    -------
    str
        processed days in a coma separated str


    """
    conn_data = iniciar_conexion_db(tipo="data")

    # get processed days in basic data
    processed_days_q = f"""
    select distinct dia
    from {table_name}
    """
    processed_days = pd.read_sql(processed_days_q, conn_data)
    processed_days = processed_days.dia
    processed_days = ", ".join([f"'{val}'" for val in processed_days])

    return processed_days


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
    processed_days = ", ".join([f"'{val}'" for val in processed_days])

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

        daily_services["hora"] = daily_services.min_datetime.str[10:13].map(int)

        daily_services = daily_services.drop(["min_datetime"], axis=1)

        # computing services by hour
        dispatched_services_stats = daily_services.groupby(
            ["id_linea", "dia", "hora"], as_index=False
        ).agg(servicios=("hora", "count"))

        print("Fin procesamiento servicios por hora")

        print("Subiendo datos a la DB")

        cols = ["id_linea", "dia", "hora", "servicios"]

        dispatched_services_stats = dispatched_services_stats.reindex(columns=cols)

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
        weekend = pd.to_datetime(daily_data["dia"].copy()).dt.dayofweek > 4
        daily_data.loc[:, ["dia"]] = "weekday"
        daily_data.loc[weekend, ["dia"]] = "weekend"

        # compute aggregated stats
        type_of_day_stats = daily_data.groupby(
            ["id_linea", "dia", "hora"], as_index=False
        ).mean()

        print("Subiendo datos a la DB")

        cols = ["id_linea", "dia", "hora", "servicios"]

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


def read_legs_data_by_line_hours_and_day(line_ids_where, hour_range, day_type):
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
    q_main_legs = """
    select id_linea, dia, factor_expansion_linea,h3_o,h3_d, od_validado
    from etapas
    """
    q_main_legs = q_main_legs + line_ids_where

    if hour_range:
        hour_range_where = f" and hora >= {hour_range[0]} and hora <= {hour_range[1]}"
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

    legs["yr_mo"] = legs.dia.str[:7]

    if not day_type_is_a_date:
        # create a weekday_filter
        weekday_filter = pd.to_datetime(legs.dia, format="%Y-%m-%d").dt.dayofweek < 5

        if day_type == "weekday":
            legs = legs.loc[weekday_filter, :]
        else:
            legs = legs.loc[~weekday_filter, :]

    return legs
