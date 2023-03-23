import itertools
import geopandas as gpd
import warnings
import pandas as pd
import numpy as np
import h3
import weightedstats as ws
from math import floor
from shapely import wkt
from shapely.geometry import Point
import re
from urbantrips.geo.geo import h3_from_row
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    crear_tablas_indicadores_operativos,
)


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
    q_rec = f"select * from recorridos"
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
        select e.*,d.h3_d, f.factor_expansion
        from ({q_main_etapas}) e
        left join destinos d
        on d.id = e.id
        left join factores_expansion f
        on e.dia = f.dia
        and e.id_tarjeta = f.id_tarjeta
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
    recorridos["geometry"] = recorridos.wkt.apply(wkt.loads)

    # Set which parameter to use to slit route geoms
    if section_meters:

        # project geoms and get for each geom a n_section
        recorridos = gpd.GeoDataFrame(
            recorridos, geometry="geometry", crs="EPSG:4326"
        ).to_crs(epsg=9265)
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
    section_load_table = etapas.groupby("id_linea").apply(
        compute_section_load_table,
        recorridos=recorridos,
        rango_hrs=rango_hrs,
        day_type=day_type,
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


def compute_section_load_table(
        df, recorridos, rango_hrs, day_type, *args, **kwargs):
    """
    Computes for a route a table with the load per section

    Parameters
    ----------
    df : pandas.DataFrame
        table of legs in a route
    recorridos : geopandas.GeoDataFrame
        routes geoms with a n_sections column
    rango_hrs : tuple
        tuple holding hourly range (from,to).

    """

    id_linea = df.id_linea.unique()[0]

    print(f"Computing section load id_route {id_linea}")

    if (recorridos.id_linea == id_linea).any():

        recorrido = recorridos.loc[recorridos.id_linea ==
                                   id_linea, "geometry"].item()
        n_sections = recorridos.loc[
            recorridos.id_linea == id_linea, "n_sections"
        ].item()

        # create Points for origins and destination
        df["o"] = df.h3_o.map(lambda h: Point(h3.h3_to_geo(h)[::-1]))
        df["d"] = df.h3_d.map(lambda h: Point(h3.h3_to_geo(h)[::-1]))

        # Assign a route section id
        df["o_proj"] = list(
            map(get_route_section_id, df["o"], itertools.repeat(recorrido))
        )
        df["d_proj"] = list(
            map(get_route_section_id, df["d"], itertools.repeat(recorrido))
        )

        # Assign a direction based on line progression
        df = df.reindex(columns=["o_proj", "d_proj", "factor_expansion"])
        df["sentido"] = [
            "ida" if row.o_proj <= row.d_proj
            else "vuelta" for _, row in df.iterrows()
        ]

        # Compute total legs per direction
        totals_by_direction = df.groupby("sentido", as_index=False).agg(
            cant_etapas_sentido=("factor_expansion", "sum")
        )

        step = 1 / n_sections
        sections = np.arange(0, 1 + step, step)
        section_ids = pd.Series(map(floor_rounding, sections))

        # For each leg, build traversed route segments ids
        legs_dict = df.to_dict("records")
        leg_route_sections_df = pd.concat(
            map(build_leg_route_sections_df, legs_dict,
                itertools.repeat(section_ids))
        )

        # compute total legs by section and direction
        legs_by_sections = leg_route_sections_df.groupby(
            ["sentido", "section_id"], as_index=False
        ).agg(size=("factor_expansion", "sum"))

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

        # Set db schema
        legs_by_sections_full = legs_by_sections_full.reindex(
            columns=[
                "id_linea",
                "day_type",
                "n_sections",
                "sentido",
                "section_id",
                "hora_min",
                "hora_max",
                "cantidad_etapas",
                "prop_etapas",
            ]
        )

        return legs_by_sections_full
    else:
        print("No existe recorrido para id_linea:", id_linea)


def build_leg_route_sections_df(row, section_ids):
    lim_inf = row["o_proj"]
    lim_sup = row["d_proj"]
    sentido = row["sentido"]
    f_exp = row["factor_expansion"]

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
            "sentido": [sentido] * len(leg_route_sections),
            "section_id": leg_route_sections,
            "factor_expansion": [f_exp] * len(leg_route_sections),
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


@duracion
def compute_kpi():
    """
    Esta funcion toma los datos de oferta de la tabla gps
    los datos de demanda de la tabla trx
    y produce una serie de indicadores operativos por
    dia y linea y por dia, linea, interno
    """
    # crear tablas
    crear_tablas_indicadores_operativos()

    print("Produciendo indicadores operativos...")
    conn_data = iniciar_conexion_db(tipo="data")
    conn_insumos = iniciar_conexion_db(tipo="insumos")

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

    res = 11
    distancia_entre_hex = h3.edge_length(resolution=res, unit="km")
    distancia_entre_hex = distancia_entre_hex * 2

    print("Leyendo datos de oferta")
    q = """
    select * from gps
    order by dia, id_linea, interno, fecha
    """
    gps = pd.read_sql(q, conn_data)

    # Georeferenciar con h3
    gps["h3"] = gps.apply(h3_from_row, axis=1,
                          args=(res, "latitud", "longitud"))

    # Producir un lag con respecto al siguiente posicionamiento gps
    gps["h3_lag"] = (
        gps.reindex(columns=["dia", "id_linea", "interno", "h3"])
        .groupby(["dia", "id_linea", "interno"])
        .shift(-1)
    )

    # Calcular distancia h3
    gps = gps.dropna(subset=["h3", "h3_lag"])
    gps_dict = gps.to_dict("records")
    gps["dist_km"] = list(map(distancia_h3, gps_dict))
    gps["dist_km"] = gps["dist_km"] * distancia_entre_hex

    print("Leyendo datos de demanda")
    q = """
        SELECT e.dia,e.id_linea,e.interno,e.id_tarjeta,e.h3_o,
        d.h3_d, f.factor_expansion
        from etapas e
        LEFT JOIN destinos d
        ON e.id = d.id
        LEFT JOIN factores_expansion f
        ON e.id_tarjeta = f.id_tarjeta
        AND e.dia = f.dia
    """
    etapas = pd.read_sql(q, conn_data)
    distancias = pd.read_sql_query(
        """
        SELECT *
        FROM distancias
        """,
        conn_insumos,
    )
    # usar distancias h3 cuando no hay osm
    distancias.distance_osm_drive = (
        distancias.distance_osm_drive.combine_first(distancias.distance_h3)
    )

    # obtener etapas y sus distancias recorridas
    etapas = etapas.merge(distancias, how="left", on=["h3_o", "h3_d"])

    print("Calculando indicadores de oferta por interno")

    # Calcular kilometros vehiculo dia kvd
    oferta_interno = gps\
        .groupby(["id_linea", "dia", "interno"], as_index=False)\
        .agg(kvd=("dist_km", "sum"))

    # Eliminar los vehiculos que tengan 0 kms recorridos
    oferta_interno = oferta_interno.loc[oferta_interno.kvd > 0]

    print("Calculando indicadores de demanda por interno")

    # calcular pax veh dia (pvd) y distancia media recorrida (dmt)
    demanda_interno = etapas.groupby(
        ["id_linea", "dia", "interno"], as_index=False
    ).apply(indicadores_demanda_interno)

    print("Calculando indicadores operativos por dia e interno")
    indicadores_interno = oferta_interno.merge(
        demanda_interno, how="left", on=["id_linea", "dia", "interno"]
    )
    internos_sin_demanda = indicadores_interno.pvd.isna().sum()
    internos_sin_demanda = round(
        internos_sin_demanda / len(indicadores_interno) * 100, 2
    )
    print(f"Hay {internos_sin_demanda} por ciento de internos sin demanda")

    print("Calculando IPK y FO")
    # calcular indice pasajero kilometros (ipk) y factor de ocupacion (fo)
    indicadores_interno["ipk"] = indicadores_interno.pvd / \
        indicadores_interno.kvd

    # Calcular espacios-km ofertados (EKO) y los espacios-km demandados (EKD).
    eko = indicadores_interno.kvd * 60
    ekd = indicadores_interno.pvd * indicadores_interno.dmt_mean
    indicadores_interno["fo"] = ekd / eko

    print("Subiendo indicadores por interno a la db")
    cols = [
        "id_linea",
        "dia",
        "interno",
        "kvd",
        "pvd",
        "dmt_mean",
        "dmt_median",
        "ipk",
        "fo",
    ]
    indicadores_interno = indicadores_interno.reindex(columns=cols)
    indicadores_interno.to_sql(
        "indicadores_operativos_interno",
        conn_data,
        if_exists="append",
        index=False,
    )

    print("Calculando indicadores de demanda por linea y dia")

    demanda_linea = etapas.groupby(["id_linea", "dia"], as_index=False).apply(
        indicadores_demanda_linea
    )

    print("Calculando indicadores de oferta por linea y dia")

    oferta_linea = oferta_interno\
        .groupby(["id_linea", "dia"], as_index=False)\
        .agg(
            tot_veh=("interno", "count"),
            tot_km=("kvd", "sum"),
        )

    indicadores_linea = oferta_linea.merge(
        demanda_linea, how="left", on=["id_linea", "dia"]
    )
    indicadores_linea["pvd"] = indicadores_linea.tot_pax / \
        indicadores_linea.tot_veh
    indicadores_linea["kvd"] = indicadores_linea.tot_km / \
        indicadores_linea.tot_veh
    indicadores_linea["ipk"] = indicadores_linea.tot_pax / \
        indicadores_linea.tot_km

    # Calcular espacios-km ofertados (EKO) y los espacios-km demandados (EKD).
    eko = indicadores_linea.tot_km * 60
    ekd = indicadores_linea.tot_pax * indicadores_linea.dmt_mean

    indicadores_linea["fo"] = ekd / eko

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
        "fo",
    ]
    indicadores_linea = indicadores_linea.reindex(columns=cols)
    indicadores_linea.to_sql(
        "indicadores_operativos_linea",
        conn_data,
        if_exists="append",
        index=False,
    )


def distancia_h3(row, *args, **kwargs):
    try:
        out = h3.h3_distance(row["h3"], row["h3_lag"])
    except ValueError as e:
        out = None
    return out


def indicadores_demanda_interno(df):
    d = {}
    d["pvd"] = df["factor_expansion"].sum()
    d["dmt_mean"] = np.average(
        a=df.distance_osm_drive, weights=df.factor_expansion)
    d["dmt_median"] = ws.weighted_median(
        data=df.distance_osm_drive.tolist(),
        weights=df.factor_expansion.tolist()
    )
    return pd.Series(d, index=["pvd", "dmt_mean", "dmt_median"])


def indicadores_demanda_linea(df):
    d = {}
    d["tot_pax"] = df["factor_expansion"].sum()
    d["dmt_mean"] = np.average(
        a=df.distance_osm_drive, weights=df.factor_expansion)
    d["dmt_median"] = ws.weighted_median(
        data=df.distance_osm_drive.tolist(),
        weights=df.factor_expansion.tolist()
    )
    return pd.Series(d, index=["tot_pax", "dmt_mean", "dmt_median"])
