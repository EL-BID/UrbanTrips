import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
import h3
import itertools

from urbantrips.geo import geo
from urbantrips.kpi import kpi
from urbantrips.utils import utils
from urbantrips.carto.routes import read_routes


def from_linestring_to_h3(linestring, h3_res=8):
    """
    This function takes a shapely linestring and
    returns all h3 hecgrid cells that intersect that linestring
    """
    lrs = np.arange(0, 1, 0.01)
    points = [linestring.interpolate(i, normalized=True) for i in lrs]
    coords = [(point.x, point.y) for point in points]
    linestring_h3 = pd.Series(
        [
            h3.geo_to_h3(lat=coord[1], lng=coord[0], resolution=h3_res)
            for coord in coords
        ]
    ).drop_duplicates()
    return linestring_h3


def create_coarse_h3_from_line(
    linestring: LineString, h3_res: int, route_id: int
) -> dict:

    # Reference to coarser H3 for those lines
    linestring_h3 = from_linestring_to_h3(linestring, h3_res=h3_res)

    # Creeate geodataframes with hex geoms and index and LRS
    gdf = gpd.GeoDataFrame(
        {"h3": linestring_h3}, geometry=linestring_h3.map(geo.add_geometry), crs=4326
    )
    gdf["route_id"] = route_id

    # Create LRS for each hex index
    gdf["h3_lrs"] = [
        kpi.floor_rounding(linestring.project(Point(p[::-1]), True))
        for p in gdf.h3.map(h3.h3_to_geo)
    ]

    # Create section ids for each line
    df_section_ids_LRS = kpi.create_route_section_ids(len(gdf))

    # Create cut points for each section based on H3 LRS
    df_section_ids_LRS_cut = df_section_ids_LRS.copy()
    df_section_ids_LRS_cut.loc[0] = -0.001

    # Use cut points to come up with a unique integer id
    df_section_ids = list(range(1, len(df_section_ids_LRS_cut)))

    gdf["section_id"] = pd.cut(
        gdf.h3_lrs, bins=df_section_ids_LRS_cut, labels=df_section_ids, right=True
    )

    # ESTO REEMPLAZA PARA ATRAS
    gdf = gdf.sort_values("h3_lrs")
    gdf["section_id"] = range(len(gdf))

    return gdf


def get_demand_data(
    supply_gdf: gpd.GeoDataFrame,
    day_type: str,
    line_id: int,
    hour_range: (
        list | None
    ) = None,  # Not used as % of demand is not related to hour range
) -> pd.DataFrame:

    h3_res = h3.h3_get_resolution(supply_gdf.h3.iloc[0])

    # get demand data
    line_ids_where = kpi.create_line_ids_sql_filter(line_ids=[line_id])
    legs = kpi.read_legs_data_by_line_hours_and_day(
        line_ids_where, hour_range=hour_range, day_type=day_type
    )

    # Add legs to same coarser H3 used in branch routes
    legs["h3_o"] = legs["h3_o"].map(lambda h: geo.h3toparent(h, h3_res))
    legs["h3_d"] = legs["h3_d"].map(lambda h: geo.h3toparent(h, h3_res))

    shared_h3 = supply_gdf.loc[supply_gdf.shared_h3, "h3"]
    legs["leg_in_shared_h3"] = legs.h3_o.isin(shared_h3) & legs.h3_d.isin(shared_h3)

    return legs


def aggregate_demand_data(
    legs: pd.DataFrame,
    supply_gdf: gpd.GeoDataFrame,
    base_line_id: int,
    comp_line_id: int,
    base_branch_id: int | str,
    comp_branch_id: int | str,
) -> dict:

    # Compute total legs by h3 origin and destination
    total_legs_by_h3_od = (
        legs.reindex(
            columns=[
                "dia",
                "id_linea",
                "h3_o",
                "h3_d",
                "leg_in_shared_h3",
                "factor_expansion_linea",
            ]
        )
        .groupby(
            ["dia", "id_linea", "h3_o", "h3_d", "leg_in_shared_h3"], as_index=False
        )
        .sum()
    )

    # Get only legs that could have been done in this branch
    legs_within_branch = (
        total_legs_by_h3_od.merge(
            supply_gdf.drop("geometry", axis=1),
            left_on=["h3_o"],
            right_on=["h3"],
            how="inner",
        )
        .merge(
            supply_gdf.drop("geometry", axis=1),
            left_on=["h3_d"],
            right_on=["h3"],
            how="inner",
        )
        .reindex(
            columns=[
                "dia",
                "id_linea",
                "h3_o",
                "h3_d",
                "leg_in_shared_h3",
                "factor_expansion_linea",
                "h3_lrs_x",
                "h3_lrs_y",
                "section_id_x",
                "section_id_y",
            ]
        )
        .rename(columns={"section_id_x": "o_proj", "section_id_y": "d_proj"})
    )

    total_demand = legs_within_branch.factor_expansion_linea.sum()
    line_id = legs_within_branch.id_linea.iloc[0]
    day = legs_within_branch.dia.iloc[0]

    configs = utils.leer_configs_generales()
    use_branches = configs["lineas_contienen_ramales"]
    conn_insumos = utils.iniciar_conexion_db(tipo="insumos")

    line_metadata = pd.read_sql(
        f"select id_linea, nombre_linea from metadata_lineas where id_linea in ({base_line_id},{comp_line_id})",
        conn_insumos,
        dtype={"id_linea": int},
    )

    base_line_name = line_metadata.loc[
        line_metadata.id_linea == base_line_id, "nombre_linea"
    ].item()
    comp_line_name = line_metadata.loc[
        line_metadata.id_linea == comp_line_id, "nombre_linea"
    ].item()

    if use_branches:
        # get line id base on branch
        metadata = pd.read_sql(
            f"select id_linea,id_ramal,nombre_ramal from metadata_ramales where id_ramal in ({base_branch_id},{comp_branch_id})",
            conn_insumos,
            dtype={"id_linea": int, "id_ramal": int},
        )

        base_branch_name = metadata.loc[
            metadata.id_ramal == base_branch_id, "nombre_ramal"
        ].item()
        comp_branch_name = metadata.loc[
            metadata.id_ramal == comp_branch_id, "nombre_ramal"
        ].item()
        demand_base_branch_str = (
            f"que podria recorrer este ramal {base_branch_name} (id {base_branch_id}) "
        )
        demand_comp_branch_str = f"ramal {comp_branch_name} (id {comp_branch_id})"

    else:
        demand_base_branch_str = " "
        demand_comp_branch_str = " "
    conn_insumos.close()

    print(
        f"La demanda total para la linea {base_line_name} (id {line_id}) {demand_base_branch_str}es: {int(total_demand)} etapas "
    )

    shared_demand = round(
        legs_within_branch.loc[
            legs_within_branch.leg_in_shared_h3, "factor_expansion_linea"
        ].sum()
        / total_demand
        * 100,
        1,
    )
    print(
        f"de las cuales el {shared_demand} % comparte OD con la linea {comp_line_name} (id {comp_line_id}) {demand_comp_branch_str}\n\n"
    )
    update_overlapping_table_demand(
        day,
        base_line_id,
        base_branch_id,
        comp_line_id,
        comp_branch_id,
        res_h3=h3.h3_get_resolution(supply_gdf.h3.iloc[0]),
        base_v_comp=shared_demand,
    )
    return legs_within_branch


def demand_by_section_id(legs_within_branch):
    total_demand = legs_within_branch.factor_expansion_linea.sum()

    # Add direction to use for which sections id traversed
    legs_within_branch["sentido"] = [
        "ida" if row.o_proj <= row.d_proj else "vuelta"
        for _, row in legs_within_branch.iterrows()
    ]

    # remove legs with no origin or destination projected
    legs_within_branch = legs_within_branch.dropna(subset=["o_proj", "d_proj"])

    # Create df with all traversed sections
    legs_dict = legs_within_branch.to_dict("records")
    leg_route_sections_df = pd.concat(map(kpi.build_leg_route_sections_df, legs_dict))

    # Compute total demand by section id
    demand_by_section_id = leg_route_sections_df.groupby(
        ["section_id"], as_index=False
    ).agg(total_legs=("factor_expansion_linea", "sum"))

    demand_by_section_id["prop_demand"] = (
        demand_by_section_id.total_legs / total_demand * 100
    )

    return demand_by_section_id


def update_overlapping_table_supply(
    day,
    base_line_id,
    base_branch_id,
    comp_line_id,
    comp_branch_id,
    res_h3,
    base_v_comp,
    comp_v_base,
):
    conn_data = utils.iniciar_conexion_db(tipo="data")
    # Update db
    delete_q = f"""
        delete from overlapping 
        where dia = '{day}'
        and base_line_id = {base_line_id}
        and base_branch_id = {base_branch_id}
        and comp_line_id = {comp_line_id}
        and comp_branch_id = {comp_branch_id}
        and res_h3 = {res_h3}
        and type_overlap = "oferta"
        ;
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    delete_q = f"""
        delete from overlapping 
        where dia = '{day}'
        and base_line_id = {comp_line_id}
        and base_branch_id = {comp_branch_id}
        and comp_line_id = {base_line_id}
        and comp_branch_id = {base_branch_id}
        and res_h3 = {res_h3}
        and type_overlap = "oferta"
        ;
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    insert_q = f"""
        insert into overlapping (dia,base_line_id,base_branch_id,comp_line_id,comp_branch_id,res_h3,overlap, type_overlap) 
        values
         ('{day}',{base_line_id},{base_branch_id},{comp_line_id},{comp_branch_id},{res_h3},{base_v_comp},'oferta'),
         ('{day}',{comp_line_id},{comp_branch_id},{base_line_id},{base_branch_id},{res_h3},{comp_v_base},'oferta')
        ;
    """

    conn_data.execute(insert_q)
    conn_data.commit()
    conn_data.close()


def update_overlapping_table_demand(
    day, base_line_id, base_branch_id, comp_line_id, comp_branch_id, res_h3, base_v_comp
):
    conn_data = utils.iniciar_conexion_db(tipo="data")
    # Update db
    delete_q = f"""
        delete from overlapping 
        where dia = '{day}'
        and base_line_id = {base_line_id}
        and base_branch_id = {base_branch_id}
        and comp_line_id = {comp_line_id}
        and comp_branch_id = {comp_branch_id}
        and res_h3 = {res_h3}
        and type_overlap = "demanda"
        ;
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    insert_q = f"""
        insert into overlapping (dia,base_line_id,base_branch_id,comp_line_id,comp_branch_id,res_h3,overlap, type_overlap) 
        values
         ('{day}',{base_line_id},{base_branch_id},{comp_line_id},{comp_branch_id},{res_h3},{base_v_comp},'demanda')
        ;
    """
    conn_data.execute(insert_q)
    conn_data.commit()
    conn_data.close()


def normalize_total_legs_to_dot_size(series, min_dot_size, max_dot_size):
    return min_dot_size + (max_dot_size - 1) * (series - series.min()) / (
        series.max() - series.min()
    )


def compute_supply_overlapping(
    day, base_route_id, comp_route_id, route_type, h3_res_comp
):
    # Get route geoms
    route_geoms = read_routes(
        route_ids=[base_route_id, comp_route_id], route_type=route_type
    )

    # Crate linestring for each branch
    base_route_gdf = route_geoms.loc[route_geoms.route_id == base_route_id, "geometry"]
    linestring_base = base_route_gdf.item()

    # Crate linestring for each branch
    comp_route_gdf = route_geoms.loc[route_geoms.route_id == comp_route_id, "geometry"]
    linestring_comp = comp_route_gdf.item()

    # Turn linestring into coarse h3 indexes
    base_h3 = create_coarse_h3_from_line(
        linestring=linestring_base, h3_res=h3_res_comp, route_id=base_route_id
    )
    comp_h3 = create_coarse_h3_from_line(
        linestring=linestring_comp, h3_res=h3_res_comp, route_id=comp_route_id
    )

    # Compute overlapping between those h3 indexes
    branch_overlapping = base_h3.reindex(
        columns=["h3", "route_id", "section_id"]
    ).merge(
        comp_h3.reindex(columns=["h3", "route_id", "section_id"]),
        on="h3",
        how="outer",
        suffixes=("_base", "_comp"),
    )

    # classify each h3 index as shared or not
    overlapping_mask = (branch_overlapping.route_id_base.notna()) & (
        branch_overlapping.route_id_comp.notna()
    )
    overlapping_indexes = overlapping_mask.sum()
    overlapping_h3 = branch_overlapping.loc[overlapping_mask, "h3"]
    base_h3["shared_h3"] = base_h3.h3.isin(overlapping_h3)
    comp_h3["shared_h3"] = comp_h3.h3.isin(overlapping_h3)

    # Compute % of shred h3
    base_v_comp = round(overlapping_indexes / len(base_h3) * 100, 1)
    comp_v_base = round(overlapping_indexes / len(comp_h3) * 100, 1)

    configs = utils.leer_configs_generales()
    use_branches = configs["lineas_contienen_ramales"]

    if use_branches:
        # get line id base on branch
        conn_insumos = utils.iniciar_conexion_db(tipo="insumos")
        metadata = pd.read_sql(
            f"select id_linea,id_ramal,nombre_ramal from metadata_ramales where id_ramal in ({base_route_id},{comp_route_id})",
            conn_insumos,
            dtype={"id_linea": int, "id_ramal": int},
        )
        conn_insumos.close()
        base_line_id = metadata.loc[
            metadata.id_ramal == base_route_id, "id_linea"
        ].item()
        comp_line_id = metadata.loc[
            metadata.id_ramal == comp_route_id, "id_linea"
        ].item()

        base_branch_name = metadata.loc[
            metadata.id_ramal == base_route_id, "nombre_ramal"
        ].item()
        comp_branch_name = metadata.loc[
            metadata.id_ramal == comp_route_id, "nombre_ramal"
        ].item()

        base_branch_id = base_route_id
        comp_branch_id = comp_route_id

        print(
            f"El {base_v_comp} % del recorrido del ramal base {base_branch_name} "
            f" se superpone con el del ramal de comparación {comp_branch_name}\n\n"
        )
        print(
            f"Por otro lado {comp_v_base} % del recorrido del ramal {comp_branch_name} "
            f" se superpone con el del ramal {base_branch_name}\n\n"
        )

    else:
        base_line_id = base_route_id
        comp_line_id = comp_route_id
        base_branch_id = "NULL"
        comp_branch_id = "NULL"

        metadata = pd.read_sql(
            f"select id_linea, nombre_linea from metadata_lineas where id_linea in ({base_route_id},{comp_route_id})",
            conn_insumos,
            dtype={"id_linea": int},
        )

        base_line_name = metadata.loc[
            metadata.id_linea == base_route_id, "nombre_linea"
        ].item()
        comp_line_name = metadata.loc[
            metadata.id_linea == comp_route_id, "nombre_linea"
        ].item()

        print(
            f"El {base_v_comp} % del recorrido de la linea base {base_line_name}"
            " se superpone con el del ramal de comparación {comp_line_name}\n\n"
        )
        print(
            f"Por otro lado {comp_v_base} % del recorrido del ramal {comp_line_name}"
            " se superpone con el del ramal {base_line_name}\n\n"
        )

    update_overlapping_table_supply(
        day=day,
        base_line_id=base_line_id,
        base_branch_id=base_branch_id,
        comp_line_id=comp_line_id,
        comp_branch_id=comp_branch_id,
        res_h3=h3_res_comp,
        base_v_comp=base_v_comp,
        comp_v_base=comp_v_base,
    )

    return {
        "base": {"line": base_route_gdf, "h3": base_h3},
        "comp": {"line": comp_route_gdf, "h3": comp_h3},
    }


def compute_demand_overlapping(
    base_line_id,
    comp_line_id,
    day_type,
    base_route_id,
    comp_route_id,
    base_gdf,
    comp_gdf,
):
    configs = utils.leer_configs_generales()
    comp_h3_resolution = h3.h3_get_resolution(comp_gdf.h3.iloc[0])
    configs_resolution = configs["resolucion_h3"]

    if comp_h3_resolution > configs_resolution:
        print(
            "No puede procesarse la demanda con resolución de H3 mayor a la configurada"
        )
        print("Se recomienda bajar la resolución de H3 de la línea de comparación")
        print(f"Resolucion para solapamiento de demanda {comp_h3_resolution}")
        print(f"Resolucion configurada {configs_resolution}")
        return None, None

    use_branches = configs["lineas_contienen_ramales"]

    if use_branches:
        base_branch_id = base_route_id
        comp_branch_id = comp_route_id
    else:
        base_branch_id = "NULL"
        comp_branch_id = "NULL"

    base_legs = get_demand_data(
        supply_gdf=base_gdf, day_type=day_type, line_id=base_line_id
    )
    comp_legs = get_demand_data(
        supply_gdf=comp_gdf, day_type=day_type, line_id=comp_line_id
    )

    base_demand = aggregate_demand_data(
        legs=base_legs,
        supply_gdf=base_gdf,
        base_line_id=base_line_id,
        comp_line_id=comp_line_id,
        base_branch_id=base_branch_id,
        comp_branch_id=comp_branch_id,
    )
    comp_demand = aggregate_demand_data(
        legs=comp_legs,
        supply_gdf=comp_gdf,
        base_line_id=comp_line_id,
        comp_line_id=base_line_id,
        base_branch_id=comp_branch_id,
        comp_branch_id=base_branch_id,
    )

    return base_demand, comp_demand


def get_route_combinations(base_line_id, comp_line_id):
    """
    Retrieve route ID combinations and metadata based on the given line IDs.
    This function fetches configuration settings to determine whether to use branches or lines.
    It then reads metadata from a database and computes all possible combinations of route IDs
    between the specified base and comparison line IDs.
    Args:
        base_line_id (int): The ID of the base line.
        comp_line_id (int): The ID of the comparison line.
    Returns:
        dict: A dictionary containing:
            - "route_id_combinations" (list of tuples): Combinations of route IDs.
            - "metadata" (DataFrame): Metadata of the routes.
            - "route_type" (str): Type of routes used ("branches" or "lines").
    """

    # Obtiene del archivo de configuración si se deben usar ramales o lineas
    configs = utils.leer_configs_generales()
    use_branches = configs["lineas_contienen_ramales"]
    conn_insumos = utils.iniciar_conexion_db(tipo="insumos")

    # Lee los datos de los ramales
    metadata = pd.read_sql(
        f"select id_linea,id_ramal from metadata_ramales where id_linea in ({base_line_id},{comp_line_id})",
        conn_insumos,
        dtype={"id_linea": int, "id_ramal": int},
    )

    if use_branches:
        route_type = "branches"

        # Computa todas las posibles combinaciones de ramales entre esas dos lineas
        route_id_combinations = list(itertools.combinations(metadata["id_ramal"], 2))
        base_route_id_combinations = list(
            itertools.combinations(
                metadata.loc[metadata.id_linea == base_line_id, "id_ramal"], 2
            )
        )
        comp_line_id_combinations = list(
            itertools.combinations(
                metadata.loc[metadata.id_linea == comp_line_id, "id_ramal"], 2
            )
        )
        route_id_combinations = [
            combination
            for combination in route_id_combinations
            if (
                (combination not in base_route_id_combinations)
                and (combination not in comp_line_id_combinations)
            )
        ]

        metadata = pd.read_sql(
            f"select * from metadata_ramales where id_linea in ({base_line_id},{comp_line_id})",
            conn_insumos,
            dtype={"id_linea": int, "id_ramal": int},
        )

    else:
        route_type = "lines"
        route_id_combinations = [(base_line_id, comp_line_id)]

    return {
        "route_id_combinations": route_id_combinations,
        "metadata": metadata,
        "route_type": route_type,
    }


def get_route_ids_from_combination(base_line_id, comp_line_id, route_id_combination):
    # Obtiene del archivo de configuración si se deben usar ramales o lineas
    configs = utils.leer_configs_generales()
    use_branches = configs["lineas_contienen_ramales"]

    conn_insumos = utils.iniciar_conexion_db(tipo="insumos")

    q = f"""select id_linea,id_ramal from metadata_ramales
      where id_linea in ({base_line_id},{comp_line_id})"""
    # Lee los datos de los ramales
    metadata = pd.read_sql(
        q,
        conn_insumos,
        dtype={"id_linea": int, "id_ramal": int},
    )

    q = f"""select id_linea, nombre_linea from metadata_lineas
      where id_linea in ({base_line_id},{comp_line_id})"""
    # Lee los datos de las lineas
    metadata_lineas = pd.read_sql(
        q,
        conn_insumos,
        dtype={"id_linea": int},
    )

    # crea un id de ruta unico de ramal o linea en funcion de si esta configurado
    # para usar ramales o lineas
    if use_branches:
        q = f"""
            select * from metadata_ramales where id_ramal in {route_id_combination}
            """
        metadata_branches = pd.read_sql(
            q,
            conn_insumos,
            dtype={"id_linea": int, "id_ramal": int},
        )
        if (
            route_id_combination[0]
            in metadata.loc[metadata.id_linea == base_line_id, "id_ramal"].values
        ):
            base_route_id = route_id_combination[0]
            comp_route_id = route_id_combination[1]

        else:
            base_route_id = route_id_combination[1]
            comp_route_id = route_id_combination[0]

        nombre_ramal_base = metadata_branches.loc[
            metadata_branches.id_ramal == base_route_id, "nombre_ramal"
        ].item()
        nombre_ramal_comp = metadata_branches.loc[
            metadata_branches.id_ramal == comp_route_id, "nombre_ramal"
        ].item()

        base_route_str = f"ramal {nombre_ramal_base} (id {base_route_id})"
        comp_route_str = f"ramal {nombre_ramal_comp} (id {comp_route_id})"

    else:
        base_route_id, comp_route_id = route_id_combination
        base_route_str = ""
        comp_route_str = ""

    nombre_linea_base = metadata_lineas.loc[
        metadata_lineas.id_linea == base_line_id, "nombre_linea"
    ].item()
    nombre_linea_comp = metadata_lineas.loc[
        metadata_lineas.id_linea == comp_line_id, "nombre_linea"
    ].item()

    print(
        f"Tomando como linea base la linea {nombre_linea_base} (id {base_line_id}) "
        + base_route_str
    )
    print(
        f"Tomando como linea comparacion la linea {nombre_linea_comp} (id {comp_line_id}) "
        + comp_route_str
    )
    return base_route_id, comp_route_id
