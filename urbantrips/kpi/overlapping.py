import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import h3
from urbantrips.geo import geo
from urbantrips.kpi import kpi
from urbantrips.utils import utils


def from_linestring_to_h3(linestring, h3_res=8):
    """
    This function takes a shapely linestring and
    returns all h3 hecgrid cells that intersect that linestring
    """
    linestring_h3 = pd.Series(
        [
            h3.geo_to_h3(lat=coord[1], lng=coord[0], resolution=h3_res)
            for coord in linestring.coords
        ]
    ).drop_duplicates()
    return linestring_h3


def create_coarse_h3_from_line(
    linestring: LineString, h3_res: int, route_id: int
) -> dict:

    # Reference to coarser H3 for those lines
    linestring_h3 = from_linestring_to_h3(linestring, h3_res=8)

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

    print(
        f"La demanda total para el id linea {line_id} que podria recorrer este ramal en este horario es: {int(total_demand)} etapas"
    )

    shared_demand = round(
        legs_within_branch.loc[
            legs_within_branch.leg_in_shared_h3, "factor_expansion_linea"
        ].sum()
        / total_demand
        * 100,
        1,
    )
    print(f"De las cuales el {shared_demand} % comparte OD con el ramal de comparacion")
    update_overlapping_table_demand(
        day,
        base_line_id,
        base_branch_id,
        comp_line_id,
        comp_branch_id,
        base_v_comp=shared_demand,
    )

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

    return {"demand_by_section_id": demand_by_section_id, "total_demand": total_demand}


def update_overlapping_table_supply(
    day,
    base_line_id,
    base_branch_id,
    comp_line_id,
    comp_branch_id,
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
        and type_overlap = "oferta"
        ;
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    insert_q = f"""
        insert into overlapping (dia,base_line_id,base_branch_id,comp_line_id,comp_branch_id,overlap, type_overlap) 
        values
         ('{day}',{base_line_id},{base_branch_id},{comp_line_id},{comp_branch_id},{base_v_comp},'oferta'),
         ('{day}',{comp_line_id},{comp_branch_id},{base_line_id},{base_branch_id},{comp_v_base},'oferta')
        ;
    """
    conn_data.execute(insert_q)
    conn_data.commit()
    conn_data.close()


def update_overlapping_table_demand(
    day, base_line_id, base_branch_id, comp_line_id, comp_branch_id, base_v_comp
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
        and type_overlap = "demanda"
        ;
    """
    conn_data.execute(delete_q)
    conn_data.commit()

    insert_q = f"""
        insert into overlapping (dia,base_line_id,base_branch_id,comp_line_id,comp_branch_id,overlap, type_overlap) 
        values
         ('{day}',{base_line_id},{base_branch_id},{comp_line_id},{comp_branch_id},{base_v_comp},'demanda')
        ;
    """
    conn_data.execute(insert_q)
    conn_data.commit()
    conn_data.close()
