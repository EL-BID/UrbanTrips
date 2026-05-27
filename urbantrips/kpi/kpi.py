import itertools
import warnings
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

pd.set_option('future.no_silent_downcasting', True)

# KPI WRAPPER

def _weighted_avg(values, weights):
    """Weighted average ignoring NaN in values or weights."""
    mask = ~(pd.isna(values) | pd.isna(weights))
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])

def _weighted_median(values, weights):
    """Weighted median ignoring NaN in values or weights."""
    mask = ~(pd.isna(values) | pd.isna(weights))
    if mask.sum() == 0:
        return np.nan
    return ws.weighted_median(
        data=values[mask].tolist(), weights=weights[mask].tolist()
    )


def _compute_demand_stats_vectorized(df, group_cols):
    """
    Versión vectorizada de demand_stats sobre un DataFrame agrupado por group_cols.

    Computa para cada grupo:
      - tot_pax: suma de factor_expansion_linea
      - dmt_mean_od / dmt_mean_route / dmt_mean_route_gps: medias ponderadas
      - dmt_median_od / dmt_median_route / dmt_median_route_gps: medianas ponderadas

    Equivalente exacto al groupby().apply(demand_stats) pero mucho más rápido:
    las medias ponderadas se calculan con groupby+sum nativos de pandas (en C)
    en lugar de un loop de Python por grupo. Las medianas ponderadas siguen
    requiriendo apply (no son vectorizables), pero se hacen en una pasada
    única por grupo en lugar de tres llamadas separadas.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas distance_od, distance_route, distance_route_gps,
        factor_expansion_linea.
    group_cols : list of str
        Columnas por las que agrupar.

    Returns
    -------
    pandas.DataFrame
        DataFrame con group_cols + tot_pax + 3 dmt_mean_* + 3 dmt_median_*.
    """
    w = df["factor_expansion_linea"]
    df = df.assign(
        _w_od=df["distance_od"] * w,
        _w_route=df["distance_route"] * w,
        _w_route_gps=df["distance_route_gps"] * w,
        _w_od_valid=np.where(df["distance_od"].notna() & w.notna(), w, np.nan),
        _w_route_valid=np.where(df["distance_route"].notna() & w.notna(), w, np.nan),
        _w_route_gps_valid=np.where(df["distance_route_gps"].notna() & w.notna(), w, np.nan),
    )

    # Medias ponderadas y tot_pax: agregaciones vectorizadas en C
    agg = df.groupby(group_cols, as_index=False).agg(
        tot_pax=("factor_expansion_linea", "sum"),
        _sum_w_od=("_w_od", "sum"),
        _sum_w_route=("_w_route", "sum"),
        _sum_w_route_gps=("_w_route_gps", "sum"),
        _sum_weights_od=("_w_od_valid", "sum"),
        _sum_weights_route=("_w_route_valid", "sum"),
        _sum_weights_route_gps=("_w_route_gps_valid", "sum"),
    )

    # División segura para obtener medias (NaN cuando suma de pesos es 0)
    agg["dmt_mean_od"] = agg["_sum_w_od"] / agg["_sum_weights_od"].replace(0, np.nan)
    agg["dmt_mean_route"] = agg["_sum_w_route"] / agg["_sum_weights_route"].replace(0, np.nan)
    agg["dmt_mean_route_gps"] = agg["_sum_w_route_gps"] / agg["_sum_weights_route_gps"].replace(0, np.nan)

    # Medianas ponderadas: no vectorizables, una sola apply por grupo que computa las 3
    def _three_medians(g):
        w_arr = g["factor_expansion_linea"].values
        return pd.Series({
            "dmt_median_od": _weighted_median(g["distance_od"].values, w_arr),
            "dmt_median_route": _weighted_median(g["distance_route"].values, w_arr),
            "dmt_median_route_gps": _weighted_median(g["distance_route_gps"].values, w_arr),
        })

    medians = (
        df.groupby(group_cols, as_index=False)
        .apply(_three_medians)
    )

    # Combinar
    result = agg.merge(medians, on=group_cols, how="left")
    result = result.drop(columns=[
        "_sum_w_od", "_sum_w_route", "_sum_w_route_gps",
        "_sum_weights_od", "_sum_weights_route", "_sum_weights_route_gps",
    ])
    return result

@duracion
def compute_kpi():
    """
    Esta funcion toma los datos de oferta de la tabla gps
    los datos de demanda de la tabla trx
    y produce una serie de indicadores operativos por
    dia y linea y por dia, linea, interno
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
        print("Se calcularán KPI básicos en base a datos de demanda")

    # runing basic kpi
    run_basic_kpi()

    # read data
    legs, gps = read_data_for_daily_kpi()

    if (len(legs) > 0) & (len(gps) > 0):
        # compute KPI per line and date
        compute_kpi_by_line_day(legs=legs)

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

    else:

        print("No hay servicios procesados. Puede correr la funcion services.process_services() si cuenta con una tabla de gps que indique servicios")


# SECTION LOAD KPI

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
        df["sentido"] = np.where(df["o_proj"] <= df["d_proj"], "ida", "vuelta")

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

        # Vectorized expansion: cada etapa se replica una vez por cada sección
        # que atraviesa. Equivalente a build_leg_route_sections_df aplicado fila
        # por fila, pero sin crear un DataFrame por etapa.
        # Lógica: en "ida" se va de o_proj a d_proj; en "vuelta" se invierte
        # para que el rango sea siempre creciente.
        df_v = df.reset_index(drop=True)
        o_proj = df_v["o_proj"].astype(int).values
        d_proj = df_v["d_proj"].astype(int).values
        sentido = df_v["sentido"].values

        # rangos en orden creciente según sentido
        section_start = np.where(sentido == "ida", o_proj, d_proj)
        section_end   = np.where(sentido == "ida", d_proj, o_proj)
        lengths = section_end - section_start + 1

        # secciones expandidas para todas las etapas
        section_ids_expanded = np.concatenate([
            np.arange(s, e + 1) for s, e in zip(section_start, section_end)
        ])

        # índice de fila origen para cada sección expandida
        row_idx = np.repeat(np.arange(len(df_v)), lengths)

        leg_route_sections_df = pd.DataFrame({
            "dia":                    df_v["dia"].values[row_idx],
            "sentido":                sentido[row_idx],
            "section_id":             section_ids_expanded,
            "factor_expansion_linea": df_v["factor_expansion_linea"].values[row_idx],
        })

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
            legs_by_sections_full["legs"]
            / legs_by_sections_full.cant_etapas_sentido.replace(0, np.nan)
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

    # print("Leyendo datos de oferta")
    q = f"""
    select g.* from gps g
    JOIN dias_ultima_corrida d
    ON g.dia = d.dia
    order by g.dia, id_linea, interno, fecha
    """
    gps = pd.read_sql(q, conn_data)

    q = f"""
            SELECT e.dia, e.id_linea, e.interno, e.id_tarjeta, e.h3_o,
                e.h3_d, e.factor_expansion_linea,
                tt.travel_time_min, tt.distance_od, tt.distance_route,
                tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
            FROM etapas e
            JOIN dias_ultima_corrida d ON e.dia = d.dia
            LEFT JOIN travel_times_legs tt ON e.id = tt.id
            WHERE e.od_validado = 1
        """
    legs = pd.read_sql(q, conn_data)
    
    if not ((len(gps) > 0) & (len(legs) > 0)):
        print("No hay datos sin KPI procesados")
        legs = pd.DataFrame()
        gps = pd.DataFrame()
    # print("Fin carga de datos de oferta y demanda")
    return legs, gps

def compute_kpi_by_line_day(legs):
    """
    Takes demand data and computes KPI at line level for each day.
    Supply metrics (tot_veh, tot_km, tot_km_gps) are read directly
    from services WHERE valid = 1, without any vehicle expansion factor.

    Parameters
    ----------
    legs : pandas.DataFrame
        DataFrame with legs data

    Returns
    -------
    None

    """
    conn_data = iniciar_conexion_db(tipo="data")

    # demand data
    day_demand_stats = (
        legs.dropna(subset=["distance_od", "factor_expansion_linea"])
        .groupby(["id_linea", "dia"], as_index=False)
        .apply(demand_stats)
    )
    day_stats = day_demand_stats.copy()
        
    # supply: read from services filtered to valid=1 (no expansion factor)
    services_data = pd.read_sql(
        "SELECT dia, id_linea, interno, distance_route, distance_route_gps"
        " FROM services WHERE valid = 1",
        conn_data,
    )
    services_tot_veh = (
        services_data
        .groupby(["dia", "id_linea"], as_index=False)["interno"]
        .nunique()
        .rename(columns={"interno": "tot_veh"})
    )
    services_tot_km = (
        services_data
        .groupby(["dia", "id_linea"], as_index=False)
        .agg(
            tot_km=("distance_route", "sum"),
            tot_km_gps=("distance_route_gps", "sum"),
        )
        .round(2)
    )
    day_stats = (
        day_stats
        .merge(services_tot_veh, on=["dia", "id_linea"], how="left")
        .merge(services_tot_km, on=["dia", "id_linea"], how="left")
    )

    # Safe division: replace 0 with NaN in denominators
    tot_veh_safe = day_stats.tot_veh.replace(0, np.nan)
    tot_km_safe = day_stats.tot_km.replace(0, np.nan)
    tot_km_gps_safe = day_stats.tot_km_gps.replace(0, np.nan)

    # compute KPI
    day_stats["pvd"] = day_stats.tot_pax / tot_veh_safe
    day_stats["kvd"] = day_stats.tot_km / tot_veh_safe
    day_stats["kvd_gps"] = day_stats.tot_km_gps / tot_veh_safe
    
    day_stats["ipk_route"] = day_stats.tot_pax / tot_km_safe
    day_stats["ipk_route_gps"] = day_stats.tot_pax / tot_km_gps_safe

    # EKD y FO para las tres distancias
    day_stats["ekd_mean_od"] = day_stats.tot_pax * day_stats.dmt_mean_od
    day_stats["ekd_mean_route"] = day_stats.tot_pax * day_stats.dmt_mean_route
    day_stats["ekd_mean_route_gps"] = day_stats.tot_pax * day_stats.dmt_mean_route_gps
    day_stats["ekd_median_od"] = day_stats.tot_pax * day_stats.dmt_median_od
    day_stats["ekd_median_route"] = day_stats.tot_pax * day_stats.dmt_median_route
    day_stats["ekd_median_route_gps"] = day_stats.tot_pax * day_stats.dmt_median_route_gps

    day_stats["eko"] = (day_stats.tot_km * 60).replace(0, np.nan)
    day_stats["eko_gps"] = (day_stats.tot_km_gps * 60).replace(0, np.nan)

    day_stats["fo_mean_od"] = day_stats.ekd_mean_od / day_stats.eko
    day_stats["fo_mean_route"] = day_stats.ekd_mean_route / day_stats.eko
    day_stats["fo_mean_route_gps"] = day_stats.ekd_mean_route_gps / day_stats.eko_gps
    day_stats["fo_median_od"] = day_stats.ekd_median_od / day_stats.eko
    day_stats["fo_median_route"] = day_stats.ekd_median_route / day_stats.eko
    day_stats["fo_median_route_gps"] = day_stats.ekd_median_route_gps / day_stats.eko_gps

    cols = [
        "id_linea", "dia",
        "tot_veh", "tot_km", "tot_km_gps", "tot_pax",
        "dmt_mean_od", "dmt_mean_route", "dmt_mean_route_gps",
        "dmt_median_od", "dmt_median_route", "dmt_median_route_gps",
        "pvd", "kvd", "kvd_gps", "ipk_route", "ipk_route_gps",
        "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
        "fo_median_od", "fo_median_route", "fo_median_route_gps",
    ]

    day_stats = day_stats.reindex(columns=cols)

    # get last processed days
    dias_ultima_corrida = pd.read_sql_query(
        """SELECT * FROM dias_ultima_corrida""",
        conn_data,
    )
    # borro filas de corridas anteriores y registros de tipo de día agregado
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM kpi_by_day_line WHERE dia IN ({values})"
    conn_data.execute(query)
    conn_data.execute("DELETE FROM kpi_by_day_line WHERE dia IN ('weekday','weekend')")
    conn_data.commit()
    ratio_cols = [
        "pvd", "kvd", "kvd_gps", "ipk_route", "ipk_route_gps",
        "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
        "fo_median_od", "fo_median_route", "fo_median_route_gps",
    ]
    for col in ratio_cols:
        day_stats[col] = day_stats[col].replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).round(2)
    
    day_stats['tot_pax'] = day_stats['tot_pax'].fillna(0).round(0).astype(int)  
    
    day_stats.to_sql(
        "kpi_by_day_line",
        conn_data,
        if_exists="append",
        index=False,
    )

    # return day_stats

def compute_kpi_by_line_typeday():
    """
    Reads daily KPI data from kpi_by_day_line and computes average KPI
    at line level for weekday and weekend.

    Totals (tot_veh, tot_km, tot_km_gps, tot_pax) and distance metrics
    (dmt_mean_*, dmt_median_*) are averaged across days. Ratios (ipk, pvd,
    kvd, fo) are then recomputed from those averaged totals to avoid the
    statistical bias of averaging ratios directly.

    Parameters
    ----------
    None

    Returns
    -------
    type_of_day_stats : pandas.DataFrame
        DataFrame with averaged KPI by line and type of day (weekday/weekend),
        uploaded to kpi_by_day_line table.
    """
    
    conn_data = iniciar_conexion_db(tipo="data")

    delete_q = """
    DELETE FROM kpi_by_day_line
    where dia in ('weekday','weekend')
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    daily_data = pd.read_sql("select * from kpi_by_day_line", conn_data)

    weekend = pd.to_datetime(daily_data["dia"].copy()).dt.dayofweek > 4
    daily_data.loc[:, ["dia"]] = "weekday"
    daily_data.loc[weekend, ["dia"]] = "weekend"

    # average totals by type of day — ratios are recomputed from these
    totals_cols = [
        "id_linea", "dia",
        "tot_veh", "tot_km", "tot_km_gps", "tot_pax",
        "dmt_mean_od", "dmt_mean_route", "dmt_mean_route_gps",
        "dmt_median_od", "dmt_median_route", "dmt_median_route_gps",
    ]
    type_of_day_stats = daily_data[totals_cols].groupby(
        ["id_linea", "dia"], as_index=False
    ).mean()

    # recompute ratios from averaged totals — safe division
    tot_veh_safe = type_of_day_stats.tot_veh.replace(0, np.nan)
    tot_km_safe = type_of_day_stats.tot_km.replace(0, np.nan)
    tot_km_gps_safe = type_of_day_stats.tot_km_gps.replace(0, np.nan)

    type_of_day_stats["pvd"] = type_of_day_stats.tot_pax / tot_veh_safe
    type_of_day_stats["kvd"] = type_of_day_stats.tot_km / tot_veh_safe
    type_of_day_stats["kvd_gps"] = type_of_day_stats.tot_km_gps / tot_veh_safe
    type_of_day_stats["ipk_route"] = type_of_day_stats.tot_pax / tot_km_safe
    type_of_day_stats["ipk_route_gps"] = type_of_day_stats.tot_pax / tot_km_gps_safe

    type_of_day_stats["eko"] = (type_of_day_stats.tot_km * 60).replace(0, np.nan)
    type_of_day_stats["eko_gps"] = (type_of_day_stats.tot_km_gps * 60).replace(0, np.nan)

    type_of_day_stats["fo_mean_od"] = (type_of_day_stats.tot_pax * type_of_day_stats.dmt_mean_od) / type_of_day_stats.eko
    type_of_day_stats["fo_mean_route"] = (type_of_day_stats.tot_pax * type_of_day_stats.dmt_mean_route) / type_of_day_stats.eko
    type_of_day_stats["fo_mean_route_gps"] = (type_of_day_stats.tot_pax * type_of_day_stats.dmt_mean_route_gps) / type_of_day_stats.eko_gps
    type_of_day_stats["fo_median_od"] = (type_of_day_stats.tot_pax * type_of_day_stats.dmt_median_od) / type_of_day_stats.eko
    type_of_day_stats["fo_median_route"] = (type_of_day_stats.tot_pax * type_of_day_stats.dmt_median_route) / type_of_day_stats.eko
    type_of_day_stats["fo_median_route_gps"] = (type_of_day_stats.tot_pax * type_of_day_stats.dmt_median_route_gps) / type_of_day_stats.eko_gps

    cols = [
        "id_linea", "dia",
        "tot_veh", "tot_km", "tot_km_gps", "tot_pax",
        "dmt_mean_od", "dmt_mean_route", "dmt_mean_route_gps",
        "dmt_median_od", "dmt_median_route", "dmt_median_route_gps",
        "pvd", "kvd", "kvd_gps", "ipk_route", "ipk_route_gps",
        "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
        "fo_median_od", "fo_median_route", "fo_median_route_gps",
    ]
    type_of_day_stats = type_of_day_stats.reindex(columns=cols)
    
    cols_float = ['tot_veh', 'tot_km', 'tot_km_gps', 'dmt_mean_od', 'dmt_mean_route', 'dmt_mean_route_gps', 'dmt_median_od',
       'dmt_median_route', 'dmt_median_route_gps', 'pvd', 'kvd', 'kvd_gps',
       'ipk_route', 'ipk_route_gps', 'fo_mean_od', 'fo_mean_route',
       'fo_mean_route_gps', 'fo_median_od', 'fo_median_route', 
       'fo_median_route_gps']
    for i in cols_float:
        type_of_day_stats[i] = type_of_day_stats[i].replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).round(2)

    dias_ultima_corrida = levanto_tabla_sql("dias_ultima_corrida", 'data')
    print("Subiendo indicadores por linea a la db")
    type_of_day_stats.to_sql(
        "kpi_by_day_line", conn_data, if_exists="append", index=False, filtros={"dia": dias_ultima_corrida["dia"].tolist()} 
    )
    
    ratio_cols = [
        "pvd", "kvd", "kvd_gps", "ipk_route", "ipk_route_gps",
        "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
        "fo_median_od", "fo_median_route", "fo_median_route_gps",
    ]
    for col in ratio_cols:
        type_of_day_stats[col] = type_of_day_stats[col].replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).round(2)

    return type_of_day_stats


# KPIS BY SERVICE


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

    import time
    t0 = time.time()
    print("\n[compute_kpi_by_service] inicio")

    # print("Leyendo demanda por servicios validos")

    q_valid_services = """
        WITH demand AS (
        SELECT
            e.id_tarjeta, e.id, e.id_linea, e.dia, e.id_ramal, e.interno,
            CAST(strftime('%s',(e.dia||' '||e.tiempo)) AS INTEGER) AS ts,
            e.tiempo, e.h3_o, e.h3_d, e.factor_expansion_linea,
            tt.distance_od, tt.distance_route, tt.distance_route_gps
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        LEFT JOIN travel_times_legs tt ON e.id = tt.id
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
    print(f"  [{time.time()-t0:>6.1f}s] query valid_demand        | {len(valid_demand):>10,} filas")
    t1 = time.time()

    # print("Leyendo demanda por servicios invalidos")
    q_invalid_services = """
        WITH demand AS (
        SELECT
            e.id_tarjeta, e.id, e.id_linea, e.dia, e.id_ramal, e.interno,
            CAST(strftime('%s',(e.dia||' '||e.tiempo)) AS INTEGER) AS ts,
            e.tiempo, e.h3_o, e.h3_d, e.factor_expansion_linea,
            tt.distance_od, tt.distance_route, tt.distance_route_gps
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        LEFT JOIN travel_times_legs tt ON e.id = tt.id
        WHERE e.od_validado = 1
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
            select e.id_tarjeta, e.id, id_linea, dia, id_ramal, interno, ts,
                tiempo, h3_o, h3_d, factor_expansion_linea,
                distance_od, distance_route, distance_route_gps
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
    print(f"  [{time.time()-t0:>6.1f}s] query invalid_demand      | {len(invalid_demand_dups):>10,} filas | {time.time()-t1:.1f}s")
    t1 = time.time()

    # remove duplicates leaving the first, i.e. next valid service in time
    invalid_demand = invalid_demand_dups.drop_duplicates(subset=["id"], keep="first")
    invalid_demand = invalid_demand.dropna(subset=["service_id"])

    # create single demand by service df
    
    dfs = [
        df for df in [valid_demand, invalid_demand]
        if not df.empty
    ]
    
    service_demand = pd.concat(dfs, ignore_index=True)
    
    print(f"  [{time.time()-t0:>6.1f}s] concat demand             | {len(service_demand):>10,} filas")
    t1 = time.time()

    # compute demand stats
    n_grupos = service_demand.groupby(
        ["dia", "id_linea", "id_ramal", "interno", "service_id"]
    ).ngroups
    print(f"  [{time.time()-t0:>6.1f}s] grupos a procesar         | {n_grupos:>10,} grupos")
    t1 = time.time()

    service_demand_stats = _compute_demand_stats_vectorized(
        service_demand,
        group_cols=["dia", "id_linea", "id_ramal", "interno", "service_id"],
    )
    print(f"  [{time.time()-t0:>6.1f}s] groupby+apply demand_stats| {len(service_demand_stats):>10,} filas | {time.time()-t1:.1f}s")
    t1 = time.time()

    # read supply service data
    service_supply_q = """
        select
            dia,id_linea,id_ramal,interno,service_id,
            distance_route as tot_km, distance_route_gps as tot_km_gps, min_datetime,max_datetime
        from
            services where valid = 1
        """
    service_supply = pd.read_sql(service_supply_q, conn_data)
    print(f"  [{time.time()-t0:>6.1f}s] query supply              | {len(service_supply):>10,} filas | {time.time()-t1:.1f}s")
    t1 = time.time()

    # merge supply and demand data
    service_stats = service_supply.merge(
        service_demand_stats,
        how="left",
        on=["dia", "id_linea", "id_ramal", "interno", "service_id"],
    )
    service_stats.tot_pax = service_stats.tot_pax.fillna(0)

    # Safe division: replace 0 with NaN in denominators
    tot_km_safe = service_stats["tot_km"].replace(0, np.nan)
    tot_km_gps_safe = service_stats["tot_km_gps"].replace(0, np.nan)

    # compute stats
    service_stats["ipk_route"] = service_stats["tot_pax"] / tot_km_safe
    service_stats["ipk_route_gps"] = service_stats["tot_pax"] / tot_km_gps_safe
    service_stats["ekd_mean_od"] = service_stats["tot_pax"] * service_stats["dmt_mean_od"]
    service_stats["ekd_mean_route"] = service_stats["tot_pax"] * service_stats["dmt_mean_route"]
    service_stats["ekd_mean_route_gps"] = service_stats["tot_pax"] * service_stats["dmt_mean_route_gps"]
    service_stats["ekd_median_od"] = service_stats["tot_pax"] * service_stats["dmt_median_od"]
    service_stats["ekd_median_route"] = service_stats["tot_pax"] * service_stats["dmt_median_route"]
    service_stats["ekd_median_route_gps"] = service_stats["tot_pax"] * service_stats["dmt_median_route_gps"]

    service_stats["eko"] = (service_stats["tot_km"] * 60).replace(0, np.nan)
    service_stats["eko_gps"] = (service_stats["tot_km_gps"] * 60).replace(0, np.nan)

    service_stats["fo_mean_od"] = service_stats["ekd_mean_od"] / service_stats["eko"]
    service_stats["fo_mean_route"] = service_stats["ekd_mean_route"] / service_stats["eko"]
    service_stats["fo_mean_route_gps"] = service_stats["ekd_mean_route_gps"] / service_stats["eko_gps"]
    service_stats["fo_median_od"] = service_stats["ekd_median_od"] / service_stats["eko"]
    service_stats["fo_median_route"] = service_stats["ekd_median_route"] / service_stats["eko"]
    service_stats["fo_median_route_gps"] = service_stats["ekd_median_route_gps"] / service_stats["eko_gps"]

    service_stats["hora_inicio"] = service_stats.min_datetime.str[11:16]
    service_stats["hora_fin"] = service_stats.max_datetime.str[11:16]

    # reindex to meet schema
    cols = [
        "id_linea", "dia", "id_ramal", "interno", "service_id",
        "hora_inicio", "hora_fin",
        "tot_km", "tot_km_gps", "tot_pax",
        "dmt_mean_od", "dmt_mean_route", "dmt_mean_route_gps",
        "dmt_median_od", "dmt_median_route", "dmt_median_route_gps",
        "ipk_route", "ipk_route_gps",
        "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
        "fo_median_od", "fo_median_route", "fo_median_route_gps",
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
    
    ratio_cols = [
        "ipk_route", "ipk_route_gps",
        "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
        "fo_median_od", "fo_median_route", "fo_median_route_gps",
    ]
    for col in ratio_cols:
        service_stats[col] = service_stats[col].replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).round(2)

    print(f"  [{time.time()-t0:>6.1f}s] cálculos KPI completos    | {time.time()-t1:.1f}s")
    t1 = time.time()

    service_stats['tot_pax'] = service_stats['tot_pax'].fillna(0).round(0).astype(int)
    
    service_stats.to_sql(
        "kpi_by_day_line_service",
        conn_data,
        if_exists="append",
        index=False,
    )
    print(f"  [{time.time()-t0:>6.1f}s] escritura DB              | {len(service_stats):>10,} filas | {time.time()-t1:.1f}s")
    print(f"[compute_kpi_by_service] FIN — total {time.time()-t0:.1f}s\n")

    return service_stats


def demand_stats(df):
    d = {}
    d["tot_pax"] = df["factor_expansion_linea"].sum()
    d["dmt_mean_od"] = _weighted_avg(df["distance_od"], df["factor_expansion_linea"])
    d["dmt_mean_route"] = _weighted_avg(df["distance_route"], df["factor_expansion_linea"])
    d["dmt_mean_route_gps"] = _weighted_avg(df["distance_route_gps"], df["factor_expansion_linea"])
    d["dmt_median_od"] = _weighted_median(df["distance_od"], df["factor_expansion_linea"])
    d["dmt_median_route"] = _weighted_median(df["distance_route"], df["factor_expansion_linea"])
    d["dmt_median_route_gps"] = _weighted_median(df["distance_route_gps"], df["factor_expansion_linea"])
    return pd.Series(d)


def _build_speed_aggregates(legs, distance_col, speed_leg_col,
                             svh_precomputed=None,
                             speed_veh_h_col="kmh_route_veh_h",
                             gps_distance_for_compute=None):
    """
    Construye speed_vehicle_hour, speed_line_hour y speed_line_day
    aplicando filtros (cap 60 km/h, 2σ) sobre velocidad veh-hora.

    Parameters
    ----------
    legs : DataFrame
        etapas con columnas dia, id_linea, id_ramal, interno, hora, tiempo.
    distance_col : str
        nombre de la columna de distancia en legs ('distance_route' o
        'distance_route_gps'). Documenta el pipeline.
    speed_leg_col : str
        nombre de la columna leg-level de velocidad en legs
        ('kmh_route_leg' o 'kmh_route_gps_leg'). Documenta el pipeline.
    svh_precomputed : DataFrame or None
        Si se provee, se usa como punto de partida en vez de llamar a
        compute_speed_by_day_veh_hour() o reconstruir desde demanda.
        Debe traer la columna `speed_veh_h_col` con la velocidad a usar
        en este pipeline.
    speed_veh_h_col : str
        Nombre de la columna en svh_precomputed que contiene la velocidad
        veh-hora a usar en este pipeline. Default 'kmh_route_veh_h' (ping
        based). Para el pipeline GPS pasar 'kmh_route_gps_veh_h'.
    gps_distance_for_compute : str or None
        legacy, no se usa.

    Returns
    -------
    (speed_vehicle_hour, speed_line_hour, speed_line_day) : tuple of DataFrames
        speed_vehicle_hour con columna 'kmh_veh_h';
        speed_line_hour con columna 'kmh_line_h';
        speed_line_day con columna 'kmh_line_day'.
    """
    if svh_precomputed is not None:
        # Tomar solo las claves + la columna de velocidad pedida,
        # y renombrarla al nombre interno 'kmh_veh_h'
        keep_cols = ["dia", "id_linea", "id_ramal", "interno", "hora"]
        if speed_veh_h_col not in svh_precomputed.columns:
            raise ValueError(
                f"svh_precomputed no tiene la columna esperada {speed_veh_h_col}. "
                f"Columnas disponibles: {svh_precomputed.columns.tolist()}"
            )
        svh = svh_precomputed[keep_cols + [speed_veh_h_col]].copy()
        svh = svh.rename(columns={speed_veh_h_col: "kmh_veh_h"})
    elif legs["tiempo"].isna().all():
        # fallback 15 km/h cuando no hay timestamps
        unique_line_ids = legs.id_linea.unique()
        id_lines = np.repeat(unique_line_ids, 24)
        hours = list(range(0, 24)) * len(unique_line_ids)
        svh = pd.DataFrame({
            "id_linea": id_lines,
            "hora": hours,
            "kmh_veh_h": [15] * 24 * len(unique_line_ids),
        })
        svh = (
            legs.reindex(columns=["dia", "id_linea", "id_ramal", "interno"])
            .drop_duplicates()
            .merge(svh, on=["id_linea"], how="left")
        )
    else:
        if gps_table_exists():
            # Esta rama se mantiene por compatibilidad pero idealmente
            # se llega acá solo cuando svh_precomputed=None y no hay
            # razón para no precomputar
            svh_full = compute_speed_by_day_veh_hour()
            svh = svh_full[["dia", "id_linea", "id_ramal", "interno", "hora",
                            speed_veh_h_col]].copy()
            svh = svh.rename(columns={speed_veh_h_col: "kmh_veh_h"})
        else:
            legs2 = legs.copy()
            legs2.loc[:, "datetime"] = legs2.dia + " " + legs2.tiempo
            legs2.loc[:, "time"] = pd.to_datetime(
                legs2.loc[:, "datetime"], format="%Y-%m-%d %H:%M:%S"
            )
            svh = legs2.groupby(
                ["dia", "id_linea", "id_ramal", "interno"]
            ).apply(compute_speed_by_veh_hour)
            svh = svh.droplevel(4).reset_index().rename(
                columns={"kmh_route_veh_h": "kmh_veh_h"}
            )

    # Filtros outliers
    speed_max = 60
    svh.loc[svh.kmh_veh_h > speed_max, "kmh_veh_h"] = speed_max
    svh = svh.dropna()

    speed_dev = svh.groupby(
        ["dia", "id_linea", "id_ramal"], as_index=False
    ).agg(mean=("kmh_veh_h", "mean"), std=("kmh_veh_h", "std"))
    speed_dev["speed_min"] = speed_dev["mean"] - (2 * speed_dev["std"]).map(
        lambda x: max(1, x)
    )
    speed_dev = speed_dev.reindex(
        columns=["dia", "id_linea", "id_ramal", "speed_min"]
    )

    svh = svh.merge(speed_dev, on=["dia", "id_linea", "id_ramal"], how="left")
    mask = (svh.kmh_veh_h < speed_max) & (svh.kmh_veh_h > svh.speed_min)
    svh = svh.loc[
        mask, ["dia", "id_linea", "id_ramal", "interno", "hora", "kmh_veh_h"]
    ]

    slh = (
        svh.drop(["id_ramal", "interno"], axis=1)
        .groupby(["dia", "id_linea", "hora"], as_index=False)
        .mean()
        .rename(columns={"kmh_veh_h": "kmh_line_h"})
    )
    sld = (
        svh.drop(["id_ramal", "interno", "hora"], axis=1)
        .groupby(["dia", "id_linea"], as_index=False)
        .mean()
        .rename(columns={"kmh_veh_h": "kmh_line_day"})
    )

    return svh, slh, sld

# GENERAL PURPOSE KPI WITH NO GPS
def compute_speed_by_day_veh_hour():
    """
    Reads GPS data and computes average vehicle speed by (day, line, ramal,
    interno, hour) for each day.

    Returns two parallel speed series, one per source of distance:
      - kmh_route_veh_h:     based on distance_km (ping-based, computed by
                             UrbanTrips from GPS ping positions).
      - kmh_route_gps_veh_h: based on distance_servicio_mts (odometer
                             reading from the validator, converted to km).

    Both share the same time deltas (delta_hr) because they refer to the
    same vehicle over the same intervals; what differs is the distance
    measure each interval reports.

    Returns
    -------
    pandas.DataFrame
        Columns: dia, id_linea, id_ramal, interno, hora,
        kmh_route_veh_h, kmh_route_gps_veh_h.
        Rows where both speeds are non-positive are dropped.
    """
    conn_data = iniciar_conexion_db(tipo="data")
    processed_days = get_processed_days(table_name="basic_kpi_by_line_day")

    q = f"""
    select dia, id_linea, id_ramal, fecha, interno, velocity,
           distance_km, distance_servicio_mts
    from gps
    where dia not in ({processed_days})
    ;
    """
    gps_df = pd.read_sql(q, conn_data)
    conn_data.close()

    # Crear lag de fecha por vehículo
    gps_df = gps_df.sort_values(["dia", "id_linea", "id_ramal", "interno", "fecha"])
    gps_df["fecha_lag"] = (
        gps_df.reindex(columns=["dia", "id_linea", "id_ramal", "interno", "fecha"])
        .groupby(["dia", "id_linea", "id_ramal", "interno"])
        .shift(-1)
    )

    # Delta de tiempo
    gps_df = gps_df.dropna(subset=["fecha", "fecha_lag"])
    gps_df["delta_hr"] = (gps_df.fecha_lag - gps_df.fecha) / 3600
    gps_df = gps_df.loc[gps_df.delta_hr > 0, :]

    # Dos velocidades en paralelo, una por cada distancia
    gps_df["distance_km_gps"] = gps_df.distance_servicio_mts / 1000
    gps_df["kmh_route_veh_h"] = gps_df.distance_km / gps_df.delta_hr
    gps_df["kmh_route_gps_veh_h"] = gps_df.distance_km_gps / gps_df.delta_hr
    gps_df["hora"] = pd.to_datetime(gps_df["fecha"], unit="s").dt.hour

    # Promediar ambas por veh-hora
    speed_vehicle_hour = (
        gps_df.reindex(
            columns=[
                "dia", "id_linea", "id_ramal", "interno", "hora",
                "kmh_route_veh_h", "kmh_route_gps_veh_h",
            ]
        )
        .groupby(["dia", "id_linea", "id_ramal", "interno", "hora"], as_index=False)
        .mean()
    )

    # Conservar filas donde al menos una de las dos velocidades sea válida
    keep = (
        (speed_vehicle_hour.kmh_route_veh_h > 0)
        | (speed_vehicle_hour.kmh_route_gps_veh_h > 0)
    )
    speed_vehicle_hour = speed_vehicle_hour.loc[keep, :]

    # Velocidades 0 o negativas se convierten en NaN para que no envenenen
    # los promedios ni el cap de outliers aguas abajo
    speed_vehicle_hour.loc[
        speed_vehicle_hour.kmh_route_veh_h <= 0, "kmh_route_veh_h"
    ] = np.nan
    speed_vehicle_hour.loc[
        speed_vehicle_hour.kmh_route_gps_veh_h <= 0, "kmh_route_gps_veh_h"
    ] = np.nan

    return speed_vehicle_hour


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



def run_basic_kpi(id_linea=[]):
    """
    Computes basic KPI at vehicle-hour, line-hour, and line-day level
    from legs data joined with travel_times_legs.

    Speed is estimated from GPS data when available, from demand data
    otherwise, or fixed at 15 km/h when no timestamp is present.
    Outlier speeds are capped at 60 km/h and low-speed outliers are
    removed using a 2-standard-deviation filter.

    Both kmh_route and kmh_route_gps follow the same three-layer fallback:
    leg-level (travel_times_legs) > vehicle-hour > line-hour. The two
    metrics differ at every layer: at the leg level they come from
    travel_times_legs columns computed with distance_route vs
    distance_route_gps; at the vehicle-hour layer they come from two
    parallel speed series computed from gps.distance_km vs
    gps.distance_servicio_mts (see compute_speed_by_day_veh_hour);
    at the line-hour layer they aggregate these veh-hour series.

    The occupation factor (of) is computed from eq_pax = distance_route /
    kmh_route * factor_expansion_linea. A parallel eq_pax_gps uses
    distance_route_gps / kmh_route_gps. Coverage of eq_pax_gps may still
    be slightly lower because distance_route_gps can be None when the
    validator does not report odometer data.

    Parameters
    ----------
    id_linea : list of int, optional
        If provided, only process legs for those line ids.

    Returns
    -------
    None
        Results are written to basic_kpi_by_vehicle_hr,
        basic_kpi_by_line_hr, and basic_kpi_by_line_day.
    """
    conn_data = iniciar_conexion_db(tipo="data")

    processed_days = get_processed_days(table_name="basic_kpi_by_line_day")

    q = f"""
            SELECT e.*, tt.travel_time_min, tt.distance_od, tt.distance_route,
                tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
            FROM etapas e
            LEFT JOIN travel_times_legs tt ON e.id = tt.id
            WHERE e.od_validado = 1
            AND e.dia NOT IN ({processed_days})
        """
    if len(id_linea) > 0:
        id_linea_str = ", ".join(map(str, id_linea))
        q += f" and id_linea in ({id_linea_str})"
    q += ";"

    legs = pd.read_sql(q, conn_data)

    if len(legs) < 5:
        return None

    legs = legs.rename(columns={
        "kmh_route": "kmh_route_leg",
        "kmh_route_gps": "kmh_route_gps_leg",
    })

    svh_shared = None
    if not legs["tiempo"].isna().all() and gps_table_exists():
        svh_shared = compute_speed_by_day_veh_hour()

    # Pipeline de velocidad para kmh_route (ping-based)
    speed_vehicle_hour, speed_line_hour, speed_line_day = _build_speed_aggregates(
        legs,
        distance_col="distance_route",
        speed_leg_col="kmh_route_leg",
        svh_precomputed=svh_shared,
        speed_veh_h_col="kmh_route_veh_h",
    )

    # Pipeline de velocidad para kmh_route_gps (odometer-based).
    # Mismo svh_shared, pero consume la otra columna de velocidad.
    speed_vehicle_hour_gps, speed_line_hour_gps, speed_line_day_gps = (
        _build_speed_aggregates(
            legs,
            distance_col="distance_route_gps",
            speed_leg_col="kmh_route_gps_leg",
            svh_precomputed=svh_shared,
            speed_veh_h_col="kmh_route_gps_veh_h",
        )
    )

    legs = (
        legs.merge(
            speed_vehicle_hour.rename(columns={"kmh_veh_h": "kmh_route_veh_h"}),
            on=["dia", "id_linea", "id_ramal", "interno", "hora"], how="left",
        )
        .merge(
            speed_line_hour.rename(columns={"kmh_line_h": "kmh_route_line_h"}),
            on=["dia", "id_linea", "hora"], how="left",
        )
        .merge(
            speed_vehicle_hour_gps.rename(columns={"kmh_veh_h": "kmh_route_gps_veh_h"}),
            on=["dia", "id_linea", "id_ramal", "interno", "hora"], how="left",
        )
        .merge(
            speed_line_hour_gps.rename(columns={"kmh_line_h": "kmh_route_gps_line_h"}),
            on=["dia", "id_linea", "hora"], how="left",
        )
    )


    legs["kmh_route"] = (
        legs.kmh_route_leg
        .combine_first(legs.kmh_route_veh_h)
        .combine_first(legs.kmh_route_line_h)
    )
    legs["kmh_route_gps"] = (
        legs.kmh_route_gps_leg
        .combine_first(legs.kmh_route_gps_veh_h)
        .combine_first(legs.kmh_route_gps_line_h)
    )

    legs["eq_pax"] = (
        legs.distance_route / legs.kmh_route.replace(0, np.nan)
    ) * legs.factor_expansion_linea
    legs["eq_pax_gps"] = (
        legs.distance_route_gps / legs.kmh_route_gps.replace(0, np.nan)
    ) * legs.factor_expansion_linea

    kpi_by_veh = (
        legs.reindex(
            columns=[
                "dia", "id_linea", "id_ramal", "interno", "hora",
                "factor_expansion_linea", "eq_pax", "eq_pax_gps",
                "distance_route", "distance_route_gps",
                "kmh_route", "kmh_route_gps",
            ]
        )
        .groupby(["dia", "id_linea", "id_ramal", "interno", "hora"], as_index=False)
        .agg(
            tot_pax=("factor_expansion_linea", "sum"),
            eq_pax=("eq_pax", "sum"),
            eq_pax_gps=("eq_pax_gps", "sum"),
            dmt_route=("distance_route", "mean"),
            dmt_route_gps=("distance_route_gps", "mean"),
            kmh_route=("kmh_route", "mean"),
            kmh_route_gps=("kmh_route_gps", "mean"),
        )
    )

    kpi_by_veh["of"] = kpi_by_veh.eq_pax / 60 * 100
    of_threshold = 120
    of_mask = kpi_by_veh["of"] > of_threshold
    print(f"Hay un {round(of_mask.sum()/ len(kpi_by_veh) * 100,1)} % de vehiculos con OF atipicos")
    kpi_by_veh.loc[of_mask, "of"] = None

    cols = [
        "dia", "id_linea", "id_ramal", "interno", "hora",
        "tot_pax", "eq_pax", "eq_pax_gps",
        "dmt_route", "dmt_route_gps",
        "of", "kmh_route", "kmh_route_gps",
    ]
    kpi_by_veh = kpi_by_veh.reindex(columns=cols)
    kpi_by_veh.to_sql("basic_kpi_by_vehicle_hr", conn_data, if_exists="append", index=False)

    ocupation_factor_line_hour = (
        kpi_by_veh.reindex(columns=["dia", "id_linea", "hora", "of"])
        .groupby(["dia", "id_linea", "hora"], as_index=False)
        .mean()
    )

    supply = (
        legs.reindex(columns=["dia", "id_linea", "id_ramal", "interno", "hora"])
        .drop_duplicates()
        .groupby(["dia", "id_linea", "hora"], as_index=False)["interno"]
        .nunique()
        .rename(columns={"interno": "tot_veh"})
    )

    demand = (
        legs.reindex(
            columns=["dia", "id_linea", "hora", "factor_expansion_linea",
                     "distance_route", "distance_route_gps",
                     "eq_pax", "eq_pax_gps"]
        )
        .groupby(["dia", "id_linea", "hora"], as_index=False)
        .apply(lambda g: pd.Series({
            "pax": g.factor_expansion_linea.sum(),
            "eq_pax": g.eq_pax.sum(),
            "eq_pax_gps": g.eq_pax_gps.sum(),
            "dmt_route": _weighted_avg(g.distance_route, g.factor_expansion_linea),
            "dmt_route_gps": _weighted_avg(g.distance_route_gps, g.factor_expansion_linea),
        }))
        .reset_index()
    )


    kpi_by_line_hr = (
        supply
        .merge(demand, on=["dia", "id_linea", "hora"], how="left")
        .merge(ocupation_factor_line_hour, on=["dia", "id_linea", "hora"], how="left")
    )

    kpi_by_line_hr = kpi_by_line_hr.merge(
        speed_line_hour.rename(columns={"kmh_line_h": "kmh_route"}),
        on=["dia", "id_linea", "hora"], how="left",
    )

    kpi_by_line_hr = kpi_by_line_hr.merge(
        speed_line_hour_gps.rename(columns={"kmh_line_h": "kmh_route_gps"}),
        on=["dia", "id_linea", "hora"], how="left",
    )

    kpi_by_line_hr["yr_mo"] = kpi_by_line_hr.dia.str[:7]

    cols = ["dia", "yr_mo", "id_linea", "hora", "tot_veh", "pax", "eq_pax", "eq_pax_gps",
            "dmt_route", "dmt_route_gps", "of", "kmh_route", "kmh_route_gps"]
    kpi_by_line_hr = kpi_by_line_hr.reindex(columns=cols)

    for col in ["pax", "eq_pax", "eq_pax_gps", "dmt_route", "dmt_route_gps",
                "of", "kmh_route", "kmh_route_gps"]:
        kpi_by_line_hr[col] = (
            kpi_by_line_hr[col].replace([np.inf, -np.inf], np.nan).round(2)
        )

    conn_data.execute("DELETE FROM basic_kpi_by_line_hr WHERE dia IN ('weekday','weekend')")
    conn_data.commit()
    kpi_by_line_hr.to_sql("basic_kpi_by_line_hr", conn_data, if_exists="append", index=False)

    ocupation_factor_line = (
        kpi_by_veh.reindex(columns=["dia", "id_linea", "of"])
        .groupby(["dia", "id_linea"], as_index=False)
        .mean()
    )

    daily_supply = (
        pd.read_sql(
            f"SELECT dia, id_linea, interno FROM services WHERE valid = 1"
            f" AND dia NOT IN ({processed_days})",
            conn_data,
        )
        .groupby(["dia", "id_linea"], as_index=False)["interno"]
        .nunique()
        .rename(columns={"interno": "tot_veh"})
    )

    daily_demand = (
        legs.reindex(
            columns=["dia", "id_linea", "factor_expansion_linea",
                     "distance_route", "distance_route_gps",
                     "eq_pax", "eq_pax_gps"]
        )
        .groupby(["dia", "id_linea"], as_index=False)
        .apply(lambda g: pd.Series({
            "pax": g.factor_expansion_linea.sum(),
            "eq_pax": g.eq_pax.sum(),
            "eq_pax_gps": g.eq_pax_gps.sum(),
            "dmt_route": _weighted_avg(g.distance_route, g.factor_expansion_linea),
            "dmt_route_gps": _weighted_avg(g.distance_route_gps, g.factor_expansion_linea),
        }))
        .reset_index()
    )

    kpi_by_line_day = (
        daily_supply
        .merge(daily_demand, on=["dia", "id_linea"], how="left")
        .merge(ocupation_factor_line, on=["dia", "id_linea"], how="left")
        .merge(
            speed_line_day.rename(columns={"kmh_line_day": "kmh_route"}),
            on=["dia", "id_linea"], how="left",
        )
        .merge(
            speed_line_day_gps.rename(columns={"kmh_line_day": "kmh_route_gps"}),
            on=["dia", "id_linea"], how="left",
        )
    )
    kpi_by_line_day["yr_mo"] = kpi_by_line_day.dia.str[:7]

    cols = ["dia", "yr_mo", "id_linea", "tot_veh", "pax", "eq_pax", "eq_pax_gps",
            "dmt_route", "dmt_route_gps", "of", "kmh_route", "kmh_route_gps"]
    kpi_by_line_day = kpi_by_line_day.reindex(columns=cols)
    conn_data.execute("DELETE FROM basic_kpi_by_line_day WHERE dia IN ('weekday','weekend')")
    conn_data.commit()
    kpi_by_line_day.to_sql("basic_kpi_by_line_day", conn_data, if_exists="append", index=False)

    conn_data.close()


def compute_basic_kpi_line_typeday():
    """
    Reads daily basic KPI data from basic_kpi_by_line_day and computes
    average KPI at line level for weekday and weekend.

    Totals (veh, pax, eq_pax, eq_pax_gps) and distance metrics
    (dmt_route, dmt_route_gps) are averaged across days. The occupation
    factor (of) is then recomputed from averaged eq_pax and veh totals
    to avoid the statistical bias of averaging ratios directly.

    Parameters
    ----------
    None

    Returns
    -------
    kpi_by_line_typeday : pandas.DataFrame
        DataFrame with averaged basic KPI by line and type of day,
        uploaded to basic_kpi_by_line_day table.
    """
    conn_data = iniciar_conexion_db(tipo="data")

    delete_q = "DELETE FROM basic_kpi_by_line_day where dia in ('weekday','weekend')"
    conn_data.execute(delete_q)
    conn_data.commit()

    kpi_by_line_day = pd.read_sql("select * from basic_kpi_by_line_day;", conn_data)

    weekend = pd.to_datetime(kpi_by_line_day["dia"].copy()).dt.dayofweek > 4
    kpi_by_line_day.loc[:, ["dia"]] = "weekday"
    kpi_by_line_day.loc[weekend, ["dia"]] = "weekend"

    # average totals — of is recomputed from these
    totals_cols = ["dia", "yr_mo", "id_linea", "tot_veh", "pax", "eq_pax", "eq_pax_gps",
                   "dmt_route", "dmt_route_gps", "kmh_route", "kmh_route_gps"]
    kpi_by_line_typeday = kpi_by_line_day[totals_cols].groupby(
        ["dia", "yr_mo", "id_linea"], as_index=False
    ).mean()

    # recompute of from averaged totals
    kpi_by_line_typeday["of"] = kpi_by_line_typeday.eq_pax / 60 * 100

    cols = ["dia", "yr_mo", "id_linea", "tot_veh", "pax", "eq_pax", "eq_pax_gps",
            "dmt_route", "dmt_route_gps", "of", "kmh_route", "kmh_route_gps"]
    kpi_by_line_typeday = kpi_by_line_typeday.reindex(columns=cols)
    kpi_by_line_typeday.to_sql("basic_kpi_by_line_day", conn_data, if_exists="append", index=False)

    conn_data.close()


def compute_basic_kpi_line_hr_typeday():
    """
    Reads hourly basic KPI data from basic_kpi_by_line_hr and computes
    average KPI at line-hour level for weekday and weekend.

    Totals (veh, pax, eq_pax, eq_pax_gps) and distance metrics
    (dmt_route, dmt_route_gps) are averaged across days. The occupation
    factor (of) is then recomputed from averaged eq_pax to avoid the
    statistical bias of averaging ratios directly.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Results are written directly to basic_kpi_by_line_hr table.
    """
    conn_data = iniciar_conexion_db(tipo="data")

    delete_q = "DELETE FROM basic_kpi_by_line_hr where dia in ('weekday','weekend')"
    conn_data.execute(delete_q)
    conn_data.commit()

    kpi_by_line_hr = pd.read_sql("select * from basic_kpi_by_line_hr;", conn_data)

    weekend = pd.to_datetime(kpi_by_line_hr["dia"].copy()).dt.dayofweek > 4
    kpi_by_line_hr.loc[:, ["dia"]] = "weekday"
    kpi_by_line_hr.loc[weekend, ["dia"]] = "weekend"

    # average totals — of is recomputed from these
    totals_cols = ["dia", "yr_mo", "id_linea", "hora",
                   "tot_veh", "pax", "eq_pax", "eq_pax_gps",
                   "dmt_route", "dmt_route_gps", "kmh_route", "kmh_route_gps"]
    kpi_by_line_typeday = kpi_by_line_hr[totals_cols].groupby(
        ["dia", "yr_mo", "id_linea", "hora"], as_index=False
    ).mean()

    # recompute of from averaged totals
    kpi_by_line_typeday["of"] = kpi_by_line_typeday.eq_pax / 60 * 100

    cols = ["dia", "yr_mo", "id_linea", "hora",
            "tot_veh", "pax", "eq_pax", "eq_pax_gps",
            "dmt_route", "dmt_route_gps", "of", "kmh_route", "kmh_route_gps"]
    kpi_by_line_typeday = kpi_by_line_typeday.reindex(columns=cols)
    kpi_by_line_typeday.to_sql("basic_kpi_by_line_hr", conn_data, if_exists="append", index=False)

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
        speed_by_hour["kmh_route_veh_h"] = (
            speed_by_hour.meters / speed_by_hour.seconds * 3.6
        )
        speed_by_hour = speed_by_hour.reindex(columns=["hora", "kmh_route_veh_h"])

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
    try:
        processed_days = pd.read_sql(
            f"select distinct dia from {table_name}", conn_data
        )
    finally:
        conn_data.close()
    processed_days = processed_days.dia
    processed_days = ", ".join([f"'{val}'" for val in processed_days])
    return processed_days


# SERVICES' KPIS

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

    # print("Leyendo datos de servicios")

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

        # print("Procesando servicios por hora")

        daily_services["hora"] = daily_services.min_datetime.str[10:13].map(int)

        daily_services = daily_services.drop(["min_datetime"], axis=1)

        # computing services by hour
        dispatched_services_stats = daily_services.groupby(
            ["id_linea", "dia", "hora"], as_index=False
        ).agg(servicios=("hora", "count"))

        cols = ["id_linea", "dia", "hora", "servicios"]

        dispatched_services_stats = dispatched_services_stats.reindex(columns=cols)

        conn_data.execute("DELETE FROM services_by_line_hour WHERE dia IN ('weekday','weekend')")
        conn_data.commit()
        conn_dash.execute("DELETE FROM services_by_line_hour WHERE dia IN ('weekday','weekend')")
        conn_dash.commit()

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

        # print("Datos subidos a la DB")
    else:
        print("Todos los dias fueron procesados")

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

        # get day of the week
        weekend = pd.to_datetime(daily_data["dia"].copy()).dt.dayofweek > 4
        daily_data.loc[:, ["dia"]] = "weekday"
        daily_data.loc[weekend, ["dia"]] = "weekend"

        # compute aggregated stats
        type_of_day_stats = daily_data.groupby(
            ["id_linea", "dia", "hora"], as_index=False
        ).mean()

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

        # print("Datos subidos a la DB")

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
