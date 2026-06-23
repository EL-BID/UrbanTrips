import itertools
import logging
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
    is_date_string,
    check_date_type,
    create_line_ids_sql_filter,
)
from urbantrips.carto.compute_distances import compute_od_distances
from urbantrips.storage.context import StorageContext

logger = logging.getLogger(__name__)

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
def compute_kpi(ctx: StorageContext):
    """
    Esta funcion toma los datos de oferta de la tabla gps
    los datos de demanda de la tabla trx
    y produce una serie de indicadores operativos por
    dia y linea y por dia, linea, interno
    """

    logger.info("Produciendo indicadores operativos...")

    if not ctx.data.has_rows("gps"):
        logger.info("No existe tabla GPS en la base; se calcularán KPI básicos en base a datos de demanda")

    # runing basic kpi
    run_basic_kpi(ctx)

    # read data
    legs, gps = read_data_for_daily_kpi(ctx)

    if (len(legs) > 0) & (len(gps) > 0):
        # compute KPI per line and date
        compute_kpi_by_line_day(legs=legs, gps=gps, ctx=ctx)

        # compute KPI per line and type of day
        compute_kpi_by_line_typeday(ctx)

    # Run KPI at service level
    try:
        valid_services = ctx.data.query(
            "SELECT COUNT(*) AS n FROM services WHERE valid = 1"
        ).iloc[0, 0]
    except Exception:
        valid_services = 0

    if valid_services > 0:
        logger.info("Computando estadisticos por servicio")
        # compute KPI by service and day
        compute_kpi_by_service(ctx)

        # compute amount of hourly services by line and day
        compute_dispatched_services_by_line_hour_day(ctx)

        # compute amount of hourly services by line and type of day
        compute_dispatched_services_by_line_hour_typeday(ctx)

    else:

        logger.info("No hay servicios procesados. Puede correr services.process_services() si cuenta con GPS")


# SECTION LOAD KPI

def compute_route_section_load(
    ctx: StorageContext,
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
    ctx : StorageContext
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
    legs = read_legs_data_by_line_hours_and_day(line_ids_where, hour_range, day_type, ctx)

    # read routes geoms
    route_geoms = get_route_geoms_with_sections_data(
        line_ids, section_meters, n_sections, ctx
    )

    # check which section geoms are already crated
    new_route_geoms = check_exists_route_section_points_table(route_geoms, ctx)

    # create the line and n sections pair missing and upload it to the db
    if len(new_route_geoms) > 0:

        upload_route_section_points_table(new_route_geoms, ctx, delete_old_data=False)

    # delete old seciton load data
    yr_mos = legs.yr_mo.unique()

    delete_old_route_section_load_data(
        route_geoms, hour_range, day_type, yr_mos, ctx
    )

    # compute section load
    logger.info("Computing section load per route ...")

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

        logger.debug("Uploading data to db...")
        ctx.data.append_raw(section_load_table, "ocupacion_por_linea_tramo")

        return section_load_table
    else:
        logger.warning(
            "No existen recorridos o etapas para las líneas (líneas=%d, recorridos=%d, etapas=%d)",
            len(line_ids), len(route_geoms), len(legs),
        )


def delete_old_route_section_load_data(
    route_geoms, hour_range, day_type, yr_mos, ctx: StorageContext, db: str = "data"
):
    """
    Deletes old data in table ocupacion_por_linea_tramo.

    db : str
        Which storage port to delete from: "data" (pipeline output) or "dash"
        (the copy the dashboard reads). The viz functions populate the dash
        copy and pass db="dash".
    """
    table_name = "ocupacion_por_linea_tramo"
    adapter = getattr(ctx, db)

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
            logger.debug(
                "Borrando ocupacion_por_linea_tramo: linea=%s, secciones=%s, yr_mo=%s%s",
                row.id_linea, row.n_sections, yr_mo,
                f", horas {hour_range[0]}-{hour_range[1]}" if hour_range else "",
            )

            q_delete = f"""
                DELETE FROM {table_name}
                WHERE id_linea = {row.id_linea}
                AND hour_min {hora_min_filter}
                AND hour_max {hora_max_filter}
                AND day_type = '{day_type}'
                AND n_sections = {row.n_sections}
                AND yr_mo = '{yr_mo}'
                """

            # The table only exists once data has been appended; tolerate its
            # absence on the first run (same pattern as kpi_lineas).
            try:
                adapter.execute(q_delete)
            except Exception as exc:
                if "does not exist" not in str(exc):
                    raise

    logger.debug("Fin borrado datos previos")


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
    logger.debug("Calculando carga por tramo para linea id %s", line_id)

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
        logger.warning("No existe recorrido para id_linea: %s", line_id)


# GENERAL PURPOSE KPIS WITH GPS


def read_data_for_daily_kpi(ctx: StorageContext):
    """
    Read legs and gps micro data from db and
    merges distances to legs

    Parameters
    ----------
    ctx : StorageContext

    Returns
    -------
    legs: pandas.DataFrame
        data frame with legs data

    gps: pandas.DataFrame
        gps vehicle tracking data
    """

    if not ctx.data.has_rows("gps"):
        logger.info("No existe tabla GPS en la base; no se pueden computar indicadores de oferta")
        return pd.DataFrame(), pd.DataFrame()

    q = """
    SELECT g.* FROM gps g
    JOIN dias_ultima_corrida d
    ON g.dia = d.dia
    ORDER BY g.dia, id_linea, interno, fecha
    """
    gps = ctx.data.query(q)

    q = """
        SELECT e.dia, e.id_linea, e.interno, e.id_tarjeta, e.h3_o,
            e.h3_d, e.factor_expansion_linea,
            tt.travel_time_min, tt.distance_od, tt.distance_route,
            tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        LEFT JOIN travel_times_legs tt ON e.id = tt.id
        WHERE e.od_validado = 1
    """
    legs = ctx.data.query(q)

    if not ((len(gps) > 0) & (len(legs) > 0)):
        logger.info("No hay datos sin KPI procesados")
        return pd.DataFrame(), pd.DataFrame()
    return legs, gps


@duracion
def compute_kpi_by_line_day(legs, gps, ctx: StorageContext):
    """
    Takes demand data and computes KPI at line level for each day.
    Supply metrics (tot_veh, tot_km, tot_km_gps) are read directly
    from services WHERE valid = 1, without any vehicle expansion factor.

    Parameters
    ----------
    legs : pandas.DataFrame
        DataFrame with legs data

    gps : pandas.DataFrame
        DataFrame with vehicle gps data

    ctx : StorageContext

    Returns
    -------
    None

    """
    # get veh expansion factors for supply data
    vehicle_expansion_factor = ctx.data.query(
        "SELECT id_linea, dia, veh_exp FROM vehicle_expansion_factors"
    )
    gps = gps.merge(vehicle_expansion_factor, on=["dia", "id_linea"], how="left")

    # demand data
    legs_valid = legs.dropna(subset=["distance_od", "factor_expansion_linea"])
    if len(legs_valid) == 0:
        logger.info("No hay etapas con distancia OD válida; no se calculan KPI de demanda")
        return

    day_demand_stats = (
        legs_valid
        .groupby(["id_linea", "dia"])
        .apply(demand_stats, include_groups=False)
        .reset_index()
    )
    day_stats = day_demand_stats.copy()
        
    # supply: read from services filtered to valid=1 (no expansion factor)
    services_data = ctx.data.query(
        "SELECT dia, id_linea, interno, distance_route, distance_route_gps"
        " FROM services WHERE valid = 1"
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

    ratio_cols = [
        "pvd", "kvd", "kvd_gps", "ipk_route", "ipk_route_gps",
        "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
        "fo_median_od", "fo_median_route", "fo_median_route_gps",
    ]
    for col in ratio_cols:
        day_stats[col] = day_stats[col].replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).round(2)
    day_stats["tot_pax"] = day_stats["tot_pax"].fillna(0).round(0).astype(int)

    # get last processed days
    dias_ultima_corrida = ctx.data.get_run_days()
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    ctx.data.execute(f"DELETE FROM kpi_by_day_line WHERE dia IN ({values})")

    ctx.data.append_raw(day_stats, "kpi_by_day_line")

    # return day_stats


@duracion
def compute_kpi_by_line_typeday(ctx: StorageContext):
    """
    Reads daily KPI data from kpi_by_day_line and computes average KPI
    at line level for weekday and weekend.

    Totals (tot_veh, tot_km, tot_km_gps, tot_pax) and distance metrics
    (dmt_mean_*, dmt_median_*) are averaged across days. Ratios (ipk, pvd,
    kvd, fo) are then recomputed from those averaged totals to avoid the
    statistical bias of averaging ratios directly.

    Parameters
    ----------
    ctx : StorageContext

    Returns
    -------
    type_of_day_stats : pandas.DataFrame
        DataFrame with averaged KPI by line and type of day (weekday/weekend),
        uploaded to kpi_by_day_line table.
    """
    # delete old data
    ctx.data.execute("DELETE FROM kpi_by_day_line WHERE dia IN ('weekday','weekend')")

    # read daily data
    daily_data = ctx.data.query("SELECT * FROM kpi_by_day_line")

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

    logger.info("Subiendo indicadores por linea a la db")

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

    ctx.data.append_raw(type_of_day_stats, "kpi_by_day_line")

    return type_of_day_stats


# KPIS BY SERVICE


@duracion
def compute_kpi_by_service(ctx: StorageContext):
    """
    Reads supply and demand data and computes KPI at service level
    for each day

    Parameters
    ----------
    ctx : StorageContext

    Returns
    -------
    None

    """

    logger.debug("Leyendo demanda por servicios validos")

    q_valid_services = """
        WITH demand AS (
        SELECT
            e.id_tarjeta, e.id, e.id_linea, e.dia, e.id_ramal, e.interno,
            epoch(CAST((e.dia||' '||e.tiempo) AS TIMESTAMP))::BIGINT AS ts,
            e.tiempo, e.h3_o, e.h3_d, e.factor_expansion_linea,
            tt.distance_route, tt.distance_route_gps
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

    valid_demand = ctx.data.query(q_valid_services)

    logger.debug("Leyendo demanda por servicios invalidos")
    q_invalid_services = """
        WITH demand AS (
            SELECT e.id_tarjeta, e.id, e.id_linea, e.dia, e.id_ramal, e.interno,
                epoch(CAST((e.dia||' '||e.tiempo) AS TIMESTAMP))::BIGINT AS ts,
                e.tiempo, e.h3_o, e.h3_d, e.factor_expansion_linea,
                tt.distance_route, tt.distance_route_gps
            FROM etapas e
            JOIN dias_ultima_corrida d
                ON e.dia = d.dia
            LEFT JOIN travel_times_legs tt ON e.id = tt.id
            WHERE od_validado = 1
            AND EXISTS (SELECT 1 FROM gps g WHERE g.id_linea = e.id_linea)
        ),
        valid_services AS (
            SELECT id_linea, dia, id_ramal, interno, service_id, min_ts, max_ts
            FROM services
            WHERE valid = 1
        ),
        invalid_demand AS (
            SELECT d.*, s.service_id
            FROM demand d
            LEFT JOIN valid_services s
            ON d.id_linea = s.id_linea
            AND d.dia = s.dia
            AND d.id_ramal = s.id_ramal
            AND d.interno = s.interno
            AND d.ts >= s.min_ts
            AND d.ts <= s.max_ts
        ),
        legs_no_service AS (
            SELECT e.id_tarjeta, e.id, id_linea, dia, id_ramal, interno, ts,
                tiempo, h3_o, h3_d, factor_expansion_linea,
                distance_route, distance_route_gps
            FROM invalid_demand e
            WHERE service_id IS NULL
        )
        SELECT d.*, s.service_id
        FROM legs_no_service d
        LEFT JOIN valid_services s
        ON d.id_linea = s.id_linea
        AND d.dia = s.dia
        AND d.id_ramal = s.id_ramal
        AND d.interno = s.interno
        AND d.ts <= s.min_ts
        ORDER BY d.id_tarjeta, d.dia, d.id_linea, d.interno, s.min_ts ASC
        """

    invalid_demand_dups = ctx.data.query(q_invalid_services)

    # remove duplicates leaving the first, i.e. next valid service in time
    invalid_demand = invalid_demand_dups.drop_duplicates(subset=["id"], keep="first")
    invalid_demand = invalid_demand.dropna(subset=["service_id"])

    # create single demand by service df
    service_demand = pd.concat([valid_demand, invalid_demand])

    # add distances to demand data
    service_demand = compute_od_distances(
        od_df             = service_demand,
        origin_col        = "h3_o",
        dest_col          = "h3_d",
        distance_col      = 'distance_od',
        unit              = 'km',
        db_path           = "data/matriz_distancia/matriz_distancia.duckdb",
        network_cache_dir = "data/matriz_distancia",
        symmetric         = False,
        precompute_dist   = 50_000,
        max_tile_deg      = 99,
        verbose           = False
    )

    # compute demand stats
    service_demand_stats = _compute_demand_stats_vectorized(
        service_demand,
        group_cols=["dia", "id_linea", "id_ramal", "interno", "service_id"],
    )

    # read supply service data
    service_supply_q = """
        SELECT
            dia, id_linea, id_ramal, interno, service_id,
            distance_route AS tot_km, distance_route_gps AS tot_km_gps,
            min_datetime, max_datetime
        FROM
            services WHERE valid = 1
        """
    service_supply = ctx.data.query(service_supply_q)

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
    dias_ultima_corrida = ctx.data.get_run_days()
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    ctx.data.execute(f"DELETE FROM kpi_by_day_line_service WHERE dia IN ({values})")

    ctx.data.append_raw(service_stats, "kpi_by_day_line_service")

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
        if False:  # gps_table_exists not available in this context
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


def compute_speed_by_day_veh_hour(ctx: StorageContext):
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
    processed_days = get_processed_days(ctx, table_name="basic_kpi_by_line_day")

    q = f"""
    SELECT dia, id_linea, id_ramal, fecha, interno, velocity,
           distance_km, distance_servicio_mts
    FROM gps
    WHERE dia NOT IN ({processed_days})
    """
    gps_df = ctx.data.query(q)

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
    # distance_servicio_mts may be NULL when the operator doesn't report odometer
    gps_df["distance_km_gps"] = pd.to_numeric(gps_df["distance_servicio_mts"], errors="coerce") / 1000
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


@duracion
def run_basic_kpi(ctx: StorageContext, id_linea=[]):
    # read already process days
    processed_days = get_processed_days(ctx, table_name="basic_kpi_by_line_day")

    # read unprocessed data from legs
    q = f"""
        SELECT *
        FROM etapas
        WHERE od_validado = 1
        AND dia NOT IN ({processed_days})
    """
    if len(id_linea) > 0:
        id_linea_str = ", ".join(map(str, id_linea))
        q += f" AND id_linea IN ({id_linea_str})"
    q += ";"

    legs = ctx.data.query(q)

    if len(legs) < 5:
        return None

    legs = compute_od_distances(
        od_df             = legs,
        origin_col        = "h3_o",
        dest_col          = "h3_d",
        distance_col      = 'distance',
        unit              = 'km',
        db_path           = "data/matriz_distancia/matriz_distancia.duckdb",
        network_cache_dir = "data/matriz_distancia",
        symmetric         = False,
        precompute_dist   = 50_000,
        max_tile_deg      = 99,
        verbose           = False
    )

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
                "kmh_route_veh_h": [15] * 24 * len(unique_line_ids),
            }
        )
        speed_vehicle_hour = (
            legs.reindex(columns=["dia", "id_linea", "id_ramal", "interno"])
            .drop_duplicates()
            .merge(speed_vehicle_hour, on=["id_linea"], how="left")
        )

    # else compute commercial speed based on gps or demand
    else:
        if ctx.data.has_rows("gps"):
            speed_vehicle_hour = compute_speed_by_day_veh_hour(ctx)
        else:
            # compute mean veh speed using demand data
            legs.loc[:, ["datetime"]] = legs.dia + " " + legs.tiempo

            legs.loc[:, ["time"]] = pd.to_datetime(
                legs.loc[:, "datetime"], format="%Y-%m-%d %H:%M:%S"
            )

            # compute vehicle speed per hour
            speed_vehicle_hour = legs.groupby(
                ["dia", "id_linea", "id_ramal", "interno"]
            ).apply(compute_speed_by_veh_hour)

            speed_vehicle_hour = speed_vehicle_hour.droplevel(4).reset_index()

    # set a max speed te remove outliers
    speed_max = 60
    speed_vehicle_hour.loc[
        speed_vehicle_hour.kmh_route_veh_h > speed_max, "kmh_route_veh_h"
    ] = speed_max

    speed_vehicle_hour = speed_vehicle_hour.dropna()

    # compute standard deviation to remove low speed outliers
    speed_dev = speed_vehicle_hour.groupby(
        ["dia", "id_linea", "id_ramal"], as_index=False
    ).agg(mean=("kmh_route_veh_h", "mean"), std=("kmh_route_veh_h", "std"))
    speed_dev["speed_min"] = speed_dev["mean"] - (2 * speed_dev["std"]).map(
        lambda x: max(1, x)
    )
    speed_dev = speed_dev.reindex(columns=["dia", "id_linea", "id_ramal", "speed_min"])

    speed_vehicle_hour = speed_vehicle_hour.merge(
        speed_dev, on=["dia", "id_linea", "id_ramal"], how="left"
    )

    speed_mask = (speed_vehicle_hour.kmh_route_veh_h < speed_max) & (
        speed_vehicle_hour.kmh_route_veh_h > speed_vehicle_hour.speed_min
    )

    speed_vehicle_hour = speed_vehicle_hour.loc[
        speed_mask,
        ["dia", "id_linea", "id_ramal", "interno", "hora", "kmh_route_veh_h"],
    ]

    # compute by hour to fill nans in vehicle speed
    speed_line_hour = (
        speed_vehicle_hour.drop(["id_ramal", "interno"], axis=1)
        .groupby(["dia", "id_linea", "hora"], as_index=False)
        .mean()
        .rename(columns={"kmh_route_veh_h": "speed_kmh_line_h"})
    )

    speed_line_day = (
        speed_vehicle_hour.drop(["id_ramal", "interno", "hora"], axis=1)
        .groupby(["dia", "id_linea"], as_index=False)
        .mean()
        .rename(columns={"kmh_route_veh_h": "speed_kmh_line_day"})
    )

    # add commercial speed to demand data
    legs = legs.merge(
        speed_vehicle_hour,
        on=["dia", "id_linea", "id_ramal", "interno", "hora"],
        how="left",
    ).merge(speed_line_hour, on=["dia", "id_linea", "hora"], how="left")

    legs["speed_kmh"] = legs.kmh_route_veh_h.combine_first(legs.speed_kmh_line_h)

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
    of_threshold = 120
    of_mask = kpi_by_veh["of"] > of_threshold
    logger.info(
        "Hay un %.1f%% de vehículos con OF atípicos", of_mask.sum() / len(kpi_by_veh) * 100
    )

    kpi_by_veh.loc[of_mask, "of"] = None

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

    ctx.data.append_raw(kpi_by_veh, "basic_kpi_by_vehicle_hr")

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

    # create month
    kpi_by_line_hr["yr_mo"] = kpi_by_line_hr.dia.str[:7]

    # set schema and upload to db
    cols = ["dia", "yr_mo", "id_linea", "hora", "veh", "pax", "dmt", "of", "speed_kmh"]

    kpi_by_line_hr = kpi_by_line_hr.reindex(columns=cols)

    ctx.data.append_raw(kpi_by_line_hr, "basic_kpi_by_line_hr")

    # COMPUTE KPI BY DAY AND LINE

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

    # set schema and upload to db
    cols = ["dia", "yr_mo", "id_linea", "veh", "pax", "dmt", "of", "speed_kmh"]

    kpi_by_line_day = kpi_by_line_day.reindex(columns=cols)

    ctx.data.append_raw(kpi_by_line_day, "basic_kpi_by_line_day")

    # compute aggregated stats by weekday and weekend
    compute_basic_kpi_line_typeday(ctx)
    compute_basic_kpi_line_hr_typeday(ctx)



def compute_basic_kpi_line_typeday(ctx: StorageContext):
    # delete old type of day data data
    ctx.data.execute(
        "DELETE FROM basic_kpi_by_line_day WHERE dia IN ('weekday','weekend')"
    )

    logger.info("Calculando KPI basicos por tipo de dia")
    kpi_by_line_day = ctx.data.query("SELECT * FROM basic_kpi_by_line_day")

    weekend = pd.to_datetime(kpi_by_line_day["dia"].copy()).dt.dayofweek > 4
    kpi_by_line_day.loc[:, ["dia"]] = "weekday"
    kpi_by_line_day.loc[weekend, ["dia"]] = "weekend"

    totals_cols = ["dia", "yr_mo", "id_linea", "veh", "pax", "dmt", "of", "speed_kmh"]
    kpi_by_line_typeday = kpi_by_line_day[totals_cols].groupby(
        ["dia", "yr_mo", "id_linea"], as_index=False
    ).mean()

    cols = ["dia", "yr_mo", "id_linea", "veh", "pax", "dmt", "of", "speed_kmh"]
    kpi_by_line_typeday = kpi_by_line_typeday.reindex(columns=cols)

    ctx.data.append_raw(kpi_by_line_typeday, "basic_kpi_by_line_day")


def compute_basic_kpi_line_hr_typeday(ctx: StorageContext):
    # delete old type of day data data
    ctx.data.execute(
        "DELETE FROM basic_kpi_by_line_hr WHERE dia IN ('weekday','weekend')"
    )

    logger.info("Calculando KPI basicos por tipo de dia")
    kpi_by_line_hr = ctx.data.query("SELECT * FROM basic_kpi_by_line_hr")

    # get day of the week
    weekend = pd.to_datetime(kpi_by_line_hr["dia"].copy()).dt.dayofweek > 4
    kpi_by_line_hr.loc[:, ["dia"]] = "weekday"
    kpi_by_line_hr.loc[weekend, ["dia"]] = "weekend"

    totals_cols = ["dia", "yr_mo", "id_linea", "hora", "veh", "pax", "dmt", "of", "speed_kmh"]
    kpi_by_line_typeday = kpi_by_line_hr[totals_cols].groupby(
        ["dia", "yr_mo", "id_linea", "hora"], as_index=False
    ).mean()

    cols = ["dia", "yr_mo", "id_linea", "hora", "veh", "pax", "dmt", "of", "speed_kmh"]
    kpi_by_line_typeday = kpi_by_line_typeday.reindex(columns=cols)

    ctx.data.append_raw(kpi_by_line_typeday, "basic_kpi_by_line_hr")


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
            np.array([h3.grid_distance(a, b) for a, b in zip(speed["h3"], speed["h3_lag"])])
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


def get_processed_days(ctx: StorageContext, table_name: str) -> str:
    """
    Returns all days present in a table as a comma-separated SQL string.
    Returns \"''\" if the table is empty or does not exist.
    """
    try:
        df = ctx.data.get_raw(table_name)
        if df.empty or "dia" not in df.columns:
            return "''"
        days = ", ".join(f"'{v}'" for v in df["dia"].unique())
        return days or "''"
    except Exception:
        return "''"


# SERVICES' KPIS


@duracion
def compute_dispatched_services_by_line_hour_day(ctx: StorageContext):
    """
    Reads services' data and computes how many services
    by line, day and hour

    Parameters
    ----------
    ctx : StorageContext

    Returns
    -------
    None

    """
    try:
        processed_df = ctx.data.get_raw("services_by_line_hour")
        if processed_df.empty or "dia" not in processed_df.columns:
            processed_days = "''"
        else:
            processed_days = (
                ", ".join(f"'{v}'" for v in processed_df["dia"].unique()) or "''"
            )
    except Exception:
        processed_days = "''"

    daily_services_q = f"""
    SELECT
        id_linea, dia, min_datetime
    FROM
        services
    WHERE
        valid = 1
    AND dia NOT IN ({processed_days})
    """

    daily_services = ctx.data.query(daily_services_q)

    if len(daily_services) > 0:

        daily_services["hora"] = daily_services.min_datetime.str[10:13].map(int)

        daily_services = daily_services.drop(["min_datetime"], axis=1)

        # computing services by hour
        dispatched_services_stats = daily_services.groupby(
            ["id_linea", "dia", "hora"], as_index=False
        ).agg(servicios=("hora", "count"))

        logger.debug("Subiendo datos a la DB")

        cols = ["id_linea", "dia", "hora", "servicios"]

        dispatched_services_stats = dispatched_services_stats.reindex(columns=cols)

        ctx.data.append_raw(dispatched_services_stats, "services_by_line_hour")
        ctx.dash.append_raw(dispatched_services_stats, "services_by_line_hour")

    else:
        logger.info("Todos los dias fueron procesados")


@duracion
def compute_dispatched_services_by_line_hour_typeday(ctx: StorageContext):
    """
    Reads services' data and computes how many services
    by line, type of day (weekday weekend), and hour

    Parameters
    ----------
    ctx : StorageContext

    Returns
    -------
    None

    """
    # delete old data
    ctx.data.execute(
        "DELETE FROM services_by_line_hour WHERE dia IN ('weekday','weekend')"
    )

    # read daily data
    try:
        daily_data = ctx.data.get_raw("services_by_line_hour")
    except Exception:
        daily_data = pd.DataFrame()

    if len(daily_data) > 0:

        logger.info("Procesando servicios por tipo de dia")

        # get day of the week
        weekend = pd.to_datetime(daily_data["dia"].copy()).dt.dayofweek > 4
        daily_data.loc[:, ["dia"]] = "weekday"
        daily_data.loc[weekend, ["dia"]] = "weekend"

        # compute aggregated stats
        type_of_day_stats = daily_data.groupby(
            ["id_linea", "dia", "hora"], as_index=False
        ).mean()

        logger.debug("Subiendo datos a la DB")

        cols = ["id_linea", "dia", "hora", "servicios"]

        type_of_day_stats = type_of_day_stats.reindex(columns=cols)

        ctx.data.append_raw(type_of_day_stats, "services_by_line_hour")

        # delete old dash data and re-upload
        ctx.dash.execute(
            "DELETE FROM services_by_line_hour WHERE dia IN ('weekday','weekend')"
        )
        ctx.dash.append_raw(type_of_day_stats, "services_by_line_hour")

    else:
        logger.info("No hay datos de servicios por hora. Correr kpi.compute_services_by_line_hour_day()")
        type_of_day_stats = pd.DataFrame()

    return type_of_day_stats


def read_legs_data_by_line_hours_and_day(line_ids_where, hour_range, day_type, ctx: StorageContext):
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
    ctx : StorageContext

    Returns
    -------
    legs : pandas.DataFrame
        dataframe with legs data by line id, hour range and type of day

    """

    # Read legs data by line id, hours, day type
    q_main_legs = """
    SELECT id_linea, dia, factor_expansion_linea, h3_o, h3_d, od_validado
    FROM etapas
    """
    q_main_legs = q_main_legs + line_ids_where

    if hour_range:
        hour_range_where = f" AND hora >= {hour_range[0]} AND hora <= {hour_range[1]}"
        q_main_legs = q_main_legs + hour_range_where

    day_type_is_a_date = is_date_string(day_type)

    if day_type_is_a_date:
        q_main_legs = q_main_legs + f" AND dia = '{day_type}'"

    q_legs = f"""
        SELECT id_linea, dia, factor_expansion_linea, h3_o, h3_d
        FROM ({q_main_legs}) e
        WHERE e.od_validado = 1
    """
    logger.debug("Obteniendo datos de etapas")

    legs = ctx.data.query(q_legs)

    legs["yr_mo"] = legs.dia.str[:7]

    if not day_type_is_a_date:
        # create a weekday_filter
        weekday_filter = pd.to_datetime(legs.dia, format="%Y-%m-%d").dt.dayofweek < 5

        if day_type == "weekday":
            legs = legs.loc[weekday_filter, :]
        else:
            legs = legs.loc[~weekday_filter, :]

    return legs
