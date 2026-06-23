import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
import h3
import itertools
import shapely
from urbantrips.geo import geo
from urbantrips.kpi import kpi
from urbantrips.utils import utils
from urbantrips.carto.routes import read_routes
from urbantrips.carto.carto import create_coarse_h3_from_line, floor_rounding
from urbantrips.storage.context import StorageContext

logger = logging.getLogger(__name__)


def get_demand_data(
    supply_gdf: gpd.GeoDataFrame,
    day_type: str,
    line_id: int,
    ctx: StorageContext,
    hour_range: (
        list | None
    ) = None,  # Not used as % of demand is not related to hour range
) -> pd.DataFrame:

    h3_res = h3.get_resolution(supply_gdf.h3.iloc[0])

    # get demand data
    line_ids_where = kpi.create_line_ids_sql_filter(line_ids=[line_id])
    legs = kpi.read_legs_data_by_line_hours_and_day(
        line_ids_where, hour_range=hour_range, day_type=day_type, ctx=ctx
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
    ctx: StorageContext,
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
            suffixes=("_x", "_y"),
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

    configs = utils.leer_configs_generales(autogenerado=False)
    use_branches = configs["lineas_contienen_ramales"]

    metadata_lineas = ctx.insumos.get_metadata_lineas()
    metadata_ramales = ctx.insumos.get_metadata_ramales()

    line_metadata = metadata_lineas[
        metadata_lineas.id_linea.isin([base_line_id, comp_line_id])
    ]

    base_line_name = line_metadata.loc[
        line_metadata.id_linea == base_line_id, "nombre_linea"
    ].item()
    comp_line_name = line_metadata.loc[
        line_metadata.id_linea == comp_line_id, "nombre_linea"
    ].item()

    if use_branches:
        metadata = metadata_ramales[
            metadata_ramales.id_ramal.isin([base_branch_id, comp_branch_id])
        ]

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

    shared_demand = round(
        legs_within_branch.loc[
            legs_within_branch.leg_in_shared_h3, "factor_expansion_linea"
        ].sum()
        / total_demand
        * 100,
        1,
    )
    output_text = (
        f"La demanda total para la linea {base_line_name} (id {line_id}) {demand_base_branch_str}es: {int(total_demand)} etapas "
        f"de las cuales el {shared_demand} % comparte OD con la linea {comp_line_name} (id {comp_line_id}) {demand_comp_branch_str}\n\n"
    )
    update_overlapping_table_demand(
        day,
        base_line_id,
        base_branch_id,
        comp_line_id,
        comp_branch_id,
        res_h3=h3.get_resolution(supply_gdf.h3.iloc[0]),
        base_v_comp=shared_demand,
        ctx=ctx,
    )
    return {"data": legs_within_branch, "output_text": output_text}


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
    ctx: StorageContext,
):
    # Update db
    delete_q = f"""
        DELETE FROM overlapping_by_route
        WHERE dia = '{day}'
        AND base_line_id = {base_line_id}
        AND base_branch_id = {base_branch_id}
        AND comp_line_id = {comp_line_id}
        AND comp_branch_id = {comp_branch_id}
        AND res_h3 = {res_h3}
        AND type_overlap = 'oferta'
    """
    ctx.data.execute(delete_q)

    delete_q = f"""
        DELETE FROM overlapping_by_route
        WHERE dia = '{day}'
        AND base_line_id = {comp_line_id}
        AND base_branch_id = {comp_branch_id}
        AND comp_line_id = {base_line_id}
        AND comp_branch_id = {base_branch_id}
        AND res_h3 = {res_h3}
        AND type_overlap = 'oferta'
    """
    ctx.data.execute(delete_q)

    insert_q = f"""
        INSERT INTO overlapping_by_route (dia, base_line_id, base_branch_id, comp_line_id,
            comp_branch_id, res_h3, overlap, type_overlap)
        VALUES
         ('{day}',{base_line_id},{base_branch_id},{comp_line_id},{comp_branch_id},{res_h3},{base_v_comp},'oferta'),
         ('{day}',{comp_line_id},{comp_branch_id},{base_line_id},{base_branch_id},{res_h3},{comp_v_base},'oferta')
    """
    ctx.data.execute(insert_q)


def update_overlapping_table_demand(
    day, base_line_id, base_branch_id, comp_line_id, comp_branch_id, res_h3, base_v_comp,
    ctx: StorageContext,
):
    # Update db
    delete_q = f"""
        DELETE FROM overlapping_by_route
        WHERE dia = '{day}'
        AND base_line_id = {base_line_id}
        AND base_branch_id = {base_branch_id}
        AND comp_line_id = {comp_line_id}
        AND comp_branch_id = {comp_branch_id}
        AND res_h3 = {res_h3}
        AND type_overlap = 'demanda'
    """
    ctx.data.execute(delete_q)

    insert_q = f"""
        INSERT INTO overlapping_by_route (dia, base_line_id, base_branch_id, comp_line_id,
            comp_branch_id, res_h3, overlap, type_overlap)
        VALUES
         ('{day}',{base_line_id},{base_branch_id},{comp_line_id},{comp_branch_id},{res_h3},{base_v_comp},'demanda')
    """
    ctx.data.execute(insert_q)


def normalize_total_legs_to_dot_size(series, min_dot_size, max_dot_size):
    return min_dot_size + (max_dot_size - 1) * (series - series.min()) / (
        series.max() - series.min()
    )


def compute_supply_overlapping(
    day, base_route_id, comp_route_id, route_type, h3_res_comp, ctx: StorageContext
):
    # Get route geoms
    route_geoms = read_routes(
        route_ids=[base_route_id, comp_route_id], route_type=route_type, ctx=ctx
    )

    # Crate linestring for each branch
    base_route_gdf = route_geoms.loc[route_geoms.route_id == base_route_id, "geometry"]
    # Crate linestring for each branch
    comp_route_gdf = route_geoms.loc[route_geoms.route_id == comp_route_id, "geometry"]

    if (len(base_route_gdf) == 0) or (len(comp_route_gdf) == 0):
        error_str = (
            "No es posible la comparación de oferta para esta combinación de rutas. "
        )
        if len(base_route_gdf) == 0:
            error_str += f"Ruta base {base_route_id} no encontrada. "
        if len(comp_route_gdf) == 0:
            error_str += f"Ruta comp {comp_route_id} no encontrada. "
        return {
            "base": {"line": None, "h3": None},
            "comp": {"line": None, "h3": None},
            "text_base_v_comp": error_str,
            "text_comp_v_base": error_str,
        }

    else:
        linestring_base = base_route_gdf.item()
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

    configs = utils.leer_configs_generales(autogenerado=False)
    use_branches = configs["lineas_contienen_ramales"]

    metadata_lineas = ctx.insumos.get_metadata_lineas()
    metadata_ramales = ctx.insumos.get_metadata_ramales()

    if use_branches:
        metadata = metadata_ramales[
            metadata_ramales.id_ramal.isin([base_route_id, comp_route_id])
        ]

        base_line_id = int(
            metadata.loc[metadata.id_ramal == base_route_id, "id_linea"].item()
        )
        comp_line_id = int(
            metadata.loc[metadata.id_ramal == comp_route_id, "id_linea"].item()
        )

        base_branch_name = metadata.loc[
            metadata.id_ramal == base_route_id, "nombre_ramal"
        ].item()
        comp_branch_name = metadata.loc[
            metadata.id_ramal == comp_route_id, "nombre_ramal"
        ].item()

        base_branch_id = base_route_id
        comp_branch_id = comp_route_id

        output_text_base_v_comp = (
            f"El {base_v_comp} % del recorrido del ramal base {base_branch_name} "
            f" se superpone con el del ramal de comparación {comp_branch_name}\n\n"
        )
        output_text_comp_v_base = (
            f"Por otro lado {comp_v_base} % del recorrido del ramal {comp_branch_name} "
            f" se superpone con el del ramal {base_branch_name}\n\n"
        )

    else:
        base_line_id = base_route_id
        comp_line_id = comp_route_id
        base_branch_id = "NULL"
        comp_branch_id = "NULL"

        metadata = metadata_lineas[
            metadata_lineas.id_linea.isin([base_route_id, comp_route_id])
        ]
        base_line_name = metadata.loc[
            metadata.id_linea == base_route_id, "nombre_linea"
        ].item()
        comp_line_name = metadata.loc[
            metadata.id_linea == comp_route_id, "nombre_linea"
        ].item()

        output_text_base_v_comp = (
            f"El {base_v_comp} % del recorrido de la linea base {base_line_name}"
            " se superpone con el del ramal de comparación {comp_line_name}\n\n"
        )
        output_text_comp_v_base = (
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
        ctx=ctx,
    )

    return {
        "base": {"line": base_route_gdf, "h3": base_h3},
        "comp": {"line": comp_route_gdf, "h3": comp_h3},
        "text_base_v_comp": output_text_base_v_comp,
        "text_comp_v_base": output_text_comp_v_base,
    }


def compute_demand_overlapping(
    base_line_id,
    comp_line_id,
    day_type,
    base_route_id,
    comp_route_id,
    base_gdf,
    comp_gdf,
    ctx: StorageContext,
):
    configs = utils.leer_configs_generales(autogenerado=False)
    comp_h3_resolution = h3.get_resolution(comp_gdf.h3.iloc[0])
    configs_resolution = configs["resolucion_h3"]

    if comp_h3_resolution > configs_resolution:
        logger.warning(
            "No puede procesarse la demanda con resolución de H3 mayor a la configurada. "
            "Se recomienda bajar la resolución de H3 de la línea de comparación. "
            "Resolución solapamiento=%s, configurada=%s",
            comp_h3_resolution, configs_resolution,
        )
        return None, None

    use_branches = configs["lineas_contienen_ramales"]

    if use_branches:
        base_branch_id = base_route_id
        comp_branch_id = comp_route_id
    else:
        base_branch_id = "NULL"
        comp_branch_id = "NULL"

    base_legs = get_demand_data(
        supply_gdf=base_gdf, day_type=day_type, line_id=base_line_id, ctx=ctx
    )
    comp_legs = get_demand_data(
        supply_gdf=comp_gdf, day_type=day_type, line_id=comp_line_id, ctx=ctx
    )

    base_demand_dict = aggregate_demand_data(
        legs=base_legs,
        supply_gdf=base_gdf,
        base_line_id=base_line_id,
        comp_line_id=comp_line_id,
        base_branch_id=base_branch_id,
        comp_branch_id=comp_branch_id,
        ctx=ctx,
    )
    comp_demand_dict = aggregate_demand_data(
        legs=comp_legs,
        supply_gdf=comp_gdf,
        base_line_id=comp_line_id,
        comp_line_id=base_line_id,
        base_branch_id=comp_branch_id,
        comp_branch_id=base_branch_id,
        ctx=ctx,
    )

    return {"base": base_demand_dict, "comp": comp_demand_dict}


def get_route_combinations(base_line_id, comp_line_id, ctx: StorageContext):
    """
    Retrieve route ID combinations and metadata based on the given line IDs.
    """

    configs = utils.leer_configs_generales(autogenerado=False)
    use_branches = configs["lineas_contienen_ramales"]

    metadata_ramales = ctx.insumos.get_metadata_ramales()
    metadata = metadata_ramales[
        metadata_ramales.id_linea.isin([base_line_id, comp_line_id])
    ].copy()
    metadata["id_linea"] = metadata["id_linea"].astype(int)
    metadata["id_ramal"] = metadata["id_ramal"].astype(int)

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

    else:
        route_type = "lines"
        route_id_combinations = [(base_line_id, comp_line_id)]

    return {
        "route_id_combinations": route_id_combinations,
        "metadata": metadata,
        "route_type": route_type,
    }


def get_route_ids_from_combination(
    base_line_id, comp_line_id, route_id_combination, ctx: StorageContext
):
    configs = utils.leer_configs_generales(autogenerado=False)
    use_branches = configs["lineas_contienen_ramales"]

    metadata_ramales = ctx.insumos.get_metadata_ramales()
    metadata_lineas = ctx.insumos.get_metadata_lineas()

    metadata = metadata_ramales[
        metadata_ramales.id_linea.isin([base_line_id, comp_line_id])
    ].copy()
    metadata["id_linea"] = metadata["id_linea"].astype(int)
    metadata["id_ramal"] = metadata["id_ramal"].astype(int)

    metadata_lineas_filtered = metadata_lineas[
        metadata_lineas.id_linea.isin([base_line_id, comp_line_id])
    ]

    # crea un id de ruta unico de ramal o linea en funcion de si esta configurado
    # para usar ramales o lineas
    if use_branches:
        metadata_branches = metadata_ramales[
            metadata_ramales.id_ramal.isin(route_id_combination)
        ].copy()
        metadata_branches["id_linea"] = metadata_branches["id_linea"].astype(int)
        metadata_branches["id_ramal"] = metadata_branches["id_ramal"].astype(int)

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

    nombre_linea_base = metadata_lineas_filtered.loc[
        metadata_lineas_filtered.id_linea == base_line_id, "nombre_linea"
    ].item()
    nombre_linea_comp = metadata_lineas_filtered.loc[
        metadata_lineas_filtered.id_linea == comp_line_id, "nombre_linea"
    ].item()

    logger.info(
        "Linea base: %s (id %s) %s | Linea comparacion: %s (id %s) %s",
        nombre_linea_base, base_line_id, base_route_str,
        nombre_linea_comp, comp_line_id, comp_route_str,
    )
    return base_route_id, comp_route_id
