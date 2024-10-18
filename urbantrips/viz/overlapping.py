import numpy as np
import geopandas as gpd
import folium
from urbantrips.geo import geo
from urbantrips.kpi import overlapping as ovl


def plot_interactive_supply_overlapping(overlapping_dict):
    # extract data from overlapping dict
    base_gdf = overlapping_dict["base"]["h3"]
    base_route_gdf = overlapping_dict["base"]["line"]
    comp_gdf = overlapping_dict["comp"]["h3"]
    comp_route_gdf = overlapping_dict["comp"]["line"]

    # get mean coords to center map
    mean_x = np.mean(base_route_gdf.item().coords.xy[0])
    mean_y = np.mean(base_route_gdf.item().coords.xy[1])

    fig = folium.Figure(width=1000, height=600)
    m = folium.Map(location=(mean_y, mean_x), zoom_start=11, tiles="cartodbpositron")

    base_gdf.explore(color="black", tiles="CartoDB positron", m=m, name="Base H3")
    base_route_gdf.explore(
        color="black", tiles="CartoDB positron", m=m, name="Base route"
    )

    comp_gdf.explore(color="red", tiles="CartoDB positron", m=m, name="Comp")
    comp_route_gdf.explore(
        color="red", tiles="CartoDB positron", m=m, name="Comp route"
    )

    folium.LayerControl(name="Legs").add_to(m)

    fig.add_child(m)
    return fig


def plot_interactive_demand_overlapping(base_demand, comp_demand, overlapping_dict):
    base_gdf = overlapping_dict["base"]["h3"]
    base_route_gdf = overlapping_dict["base"]["line"]
    comp_gdf = overlapping_dict["comp"]["h3"]
    comp_route_gdf = overlapping_dict["comp"]["line"]

    # Points for O and D
    base_origins = (
        base_demand.reindex(columns=["h3_o", "factor_expansion_linea"])
        .groupby("h3_o", as_index=False)
        .agg(total_legs=("factor_expansion_linea", "sum"))
    )
    base_destinations = (
        base_demand.reindex(columns=["h3_d", "factor_expansion_linea"])
        .groupby("h3_d", as_index=False)
        .agg(total_legs=("factor_expansion_linea", "sum"))
    )
    base_origins = gpd.GeoDataFrame(
        base_origins, geometry=base_origins.h3_o.map(geo.create_point_from_h3), crs=4326
    )
    base_destinations = gpd.GeoDataFrame(
        base_destinations,
        geometry=base_destinations.h3_d.map(geo.create_point_from_h3),
        crs=4326,
    )

    comp_origins = (
        comp_demand.reindex(columns=["h3_o", "factor_expansion_linea"])
        .groupby("h3_o", as_index=False)
        .agg(total_legs=("factor_expansion_linea", "sum"))
    )
    comp_destinations = (
        comp_demand.reindex(columns=["h3_d", "factor_expansion_linea"])
        .groupby("h3_d", as_index=False)
        .agg(total_legs=("factor_expansion_linea", "sum"))
    )
    comp_origins = gpd.GeoDataFrame(
        comp_origins, geometry=comp_origins.h3_o.map(geo.create_point_from_h3), crs=4326
    )
    comp_destinations = gpd.GeoDataFrame(
        comp_destinations,
        geometry=comp_destinations.h3_d.map(geo.create_point_from_h3),
        crs=4326,
    )

    # compute demand by section id
    base_demand_by_section = ovl.demand_by_section_id(base_demand)
    comp_demand_by_section = ovl.demand_by_section_id(comp_demand)

    # plot
    base_gdf = base_gdf.merge(base_demand_by_section, on="section_id", how="left")
    base_gdf.total_legs = base_gdf.total_legs.fillna(0)
    base_gdf.prop_demand = base_gdf.prop_demand.fillna(0)

    comp_gdf = comp_gdf.merge(comp_demand_by_section, on="section_id", how="left")
    comp_gdf.total_legs = comp_gdf.total_legs.fillna(0)
    comp_gdf.prop_demand = comp_gdf.prop_demand.fillna(0)

    min_dot_size = 1
    max_dot_size = 20

    base_destinations["total_legs_normalized"] = ovl.normalize_total_legs_to_dot_size(
        base_destinations["total_legs"], min_dot_size, max_dot_size
    )
    comp_destinations["total_legs_normalized"] = ovl.normalize_total_legs_to_dot_size(
        comp_destinations["total_legs"], min_dot_size, max_dot_size
    )
    base_origins["total_legs_normalized"] = ovl.normalize_total_legs_to_dot_size(
        base_origins["total_legs"], min_dot_size, max_dot_size
    )
    comp_origins["total_legs_normalized"] = ovl.normalize_total_legs_to_dot_size(
        comp_origins["total_legs"], min_dot_size, max_dot_size
    )

    fig = folium.Figure(width=1000, height=600)
    m = folium.Map(location=(-34.606, -58.436), zoom_start=11, tiles="cartodbpositron")

    base_gdf.explore(
        column="total_legs",
        tiles="CartoDB positron",
        m=m,
        name="Base",
        cmap="Blues",
        scheme="equalinterval",
    )
    base_destinations.explore(
        color="midnightblue",
        style_kwds={
            "style_function": lambda x: {
                "radius": x["properties"]["total_legs_normalized"]
            }
        },
        name="Base Destinations",
        m=m,
    )
    base_origins.explore(
        color="cornflowerblue",
        style_kwds={
            "style_function": lambda x: {
                "radius": x["properties"]["total_legs_normalized"]
            }
        },
        name="Base Origins",
        m=m,
    )
    base_route_gdf.explore(
        color="midnightblue", tiles="CartoDB positron", m=m, name="Base route"
    )

    comp_gdf.explore(
        column="total_legs",
        tiles="CartoDB positron",
        m=m,
        name="Comp",
        cmap="Greens",
        scheme="equalinterval",
    )
    comp_destinations.explore(
        color="darkgreen",
        style_kwds={
            "style_function": lambda x: {
                "radius": x["properties"]["total_legs_normalized"]
            }
        },
        name="Comp Destinations",
        m=m,
    )
    comp_origins.explore(
        color="limegreen",
        style_kwds={
            "style_function": lambda x: {
                "radius": x["properties"]["total_legs_normalized"]
            }
        },
        name="Comp Origins",
        m=m,
    )
    comp_route_gdf.explore(
        color="darkgreen", tiles="CartoDB positron", m=m, name="Comp route"
    )

    folium.LayerControl(name="Leyenda").add_to(m)

    fig.add_child(m)
    return fig
