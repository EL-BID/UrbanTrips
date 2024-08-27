import numpy as np
import folium


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
