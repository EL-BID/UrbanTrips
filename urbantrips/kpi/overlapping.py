import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import h3
from urbantrips.geo import geo
from urbantrips.kpi import kpi


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
    linestring: LineString, h3_res: int, line_id: int
) -> dict:

    # Reference to coarser H3 for those lines
    linestring_h3 = from_linestring_to_h3(linestring, h3_res=8)

    # Creeate geodataframes with hex geoms and index and LRS
    gdf = gpd.GeoDataFrame(
        {"h3": linestring_h3}, geometry=linestring_h3.map(geo.add_geometry), crs=4326
    )
    gdf["id_linea"] = line_id

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
    return {
        "gdf": gdf,
        "section_ids": df_section_ids,
        "section_ids_LRS_cut": df_section_ids_LRS_cut,
    }


def get_demand_by_coarse_h3(
    supply_data: dict, hour_range: list | None, day_type: str
) -> pd.DataFrame:

    supply_gdf = supply_data["gdf"]
    line_id = supply_gdf.id_linea.unique()[0]
    section_ids_LRS_cut = supply_data["section_ids_LRS_cut"]
    section_ids = supply_data["section_ids"]

    h3_res = h3.h3_get_resolution(supply_gdf.h3.iloc[0])

    # get demand data
    line_ids_where = kpi.create_line_ids_sql_filter(line_ids=[line_id])
    legs = kpi.read_legs_data_by_line_hours_and_day(
        line_ids_where, hour_range=hour_range, day_type=day_type
    )

    # Add legs to same coarser H3 used in branch routes
    legs["h3_o"] = legs["h3_o"].map(lambda h: geo.h3toparent(h, h3_res))

    # Compute total legs by h3 origin and destination
    total_legs_by_h3_od = (
        legs.reindex(
            columns=["dia", "id_linea", "h3_o", "h3_d", "factor_expansion_linea"]
        )
        .groupby(["dia", "id_linea", "h3_o", "h3_d"], as_index=False)
        .sum()
    )

    # Get only legs that could have been done in this branch
    legs_within_branch = (
        total_legs_by_h3_od.merge(
            supply_gdf.drop("geometry", axis=1),
            left_on=["id_linea", "h3_o"],
            right_on=["id_linea", "h3"],
            how="inner",
        )
        .merge(
            supply_gdf.drop("geometry", axis=1),
            left_on=["id_linea", "h3_d"],
            right_on=["id_linea", "h3"],
            how="inner",
        )
        .reindex(
            columns=[
                "dia",
                "id_linea",
                "h3_o",
                "h3_d",
                "factor_expansion_linea",
                "h3_lrs_x",
                "h3_lrs_y",
            ]
        )
        .rename(columns={"h3_lrs_x": "o_proj", "h3_lrs_y": "d_proj"})
    )

    # Classify LRS in to sectin ids
    legs_within_branch["o_proj"] = pd.cut(
        legs_within_branch.o_proj,
        bins=section_ids_LRS_cut,
        labels=section_ids,
        right=True,
    )
    legs_within_branch["d_proj"] = pd.cut(
        legs_within_branch.d_proj,
        bins=section_ids_LRS_cut,
        labels=section_ids,
        right=True,
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

    return demand_by_section_id
