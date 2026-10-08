"""Helpers shared by the H3 pages (11_Indicadores_por_h3, 12_GPS, 13_Grafos)."""

import math

import folium
import geopandas as gpd
import h3
import pandas as pd
import streamlit as st
from folium import Figure
from shapely import wkt
from shapely.geometry import Polygon

from urbantrips.viz import basemaps


def seleccionar_linea(key_input, key_select, metadata_lineas):
    """Select a line using text search and dropdown."""
    texto_a_buscar = st.text_input("Ingrese el texto a buscar en líneas", key=key_input)
    if texto_a_buscar:
        if f"df_filtrado_{texto_a_buscar}" not in st.session_state:
            filtrado = metadata_lineas[
                metadata_lineas.apply(
                    lambda row: row.astype(str)
                    .str.contains(texto_a_buscar, case=False, na=False)
                    .any(),
                    axis=1,
                )
            ]
            st.session_state[f"df_filtrado_{texto_a_buscar}"] = filtrado
        df_filtrado = st.session_state[f"df_filtrado_{texto_a_buscar}"]

        if not df_filtrado.empty:
            opciones = df_filtrado.apply(
                lambda row: f"{row['nombre_linea']}", axis=1
            ).tolist()
            seleccion_texto = st.selectbox(
                "Seleccione una línea de colectivo",
                opciones,
                key=key_select,
            )
            df_seleccionado = df_filtrado.iloc[opciones.index(seleccion_texto)]

            st.session_state["nombre_linea_h3"] = df_seleccionado.nombre_linea
            st.session_state["id_linea_h3"] = df_seleccionado.id_linea

        else:
            st.warning("No se encontró ninguna coincidencia.")


def h3_to_shapely_polygon(h3_address):
    """Convert an h3 cell to a shapely Polygon."""
    boundary = h3.cell_to_boundary(h3_address)
    # H3 returns boundary as (lat, lng), shapely expects (lng, lat)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def create_routes_h3_map(routes_h3_df, route_geoms_df, nombre_linea, id_linea):
    """Create a folium map with h3 cells colored by section_id per ramal."""
    if routes_h3_df.empty:
        return None

    # Calculate map center from h3 cells
    lats = []
    lngs = []
    # Sample first 100 cells for center calculation
    for h3_cell in routes_h3_df["h3"].head(100):
        lat, lng = h3.cell_to_latlng(h3_cell)
        lats.append(lat)
        lngs.append(lng)

    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    # Create map
    fig = Figure(width=1000, height=700)
    m = basemaps.folium_map(
        location=[center_lat, center_lng],
        zoom_start=12,
    )

    # Add title
    title_text = f"Geometrías H3 - {nombre_linea} (ID: {id_linea})"
    title_html = f"""
    <h3 align="center" style="font-size:20px"><b>{title_text}</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Check if we have ramales
    has_ramales = "id_ramal" in routes_h3_df.columns

    # Define colormaps for ramales
    cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "YlOrBr"]

    if has_ramales:
        # Group by ramal and direction
        ramales = sorted(routes_h3_df["id_ramal"].unique())
        directions = sorted(routes_h3_df["direction"].unique())

        cmap_idx = 0
        route_idx = 0
        for ramal in ramales:
            # Get colormap for this ramal
            cmap_name = cmaps[cmap_idx % len(cmaps)]
            cmap_idx += 1

            ramal_data = routes_h3_df[routes_h3_df["id_ramal"] == ramal]

            for direction in directions:
                direction_data = ramal_data[ramal_data["direction"] == direction]

                if direction_data.empty:
                    continue

                # Create geometries from H3 cells
                geometries = []
                for h3_cell in direction_data["h3"]:
                    try:
                        poly = h3_to_shapely_polygon(h3_cell)
                        geometries.append(poly)
                    except Exception:
                        geometries.append(None)

                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    direction_data.reset_index(drop=True),
                    geometry=geometries,
                    crs="EPSG:4326",
                )

                # Remove rows with None geometries
                gdf = gdf[gdf.geometry.notna()]

                if not gdf.empty:
                    # Layer name
                    layer_name = f"Ramal {ramal} - Sentido {direction}"

                    # Use explore to add to map for H3 cells
                    gdf.explore(
                        m=m,
                        column="section_id",
                        cmap=cmap_name,
                        name=f"H3: {layer_name}",
                        tooltip=["id_ramal", "direction", "section_id", "h3"],
                        popup=True,
                        style_kwds={"fillOpacity": 0.6, "weight": 1},
                        legend=False,
                        show=(route_idx == 0),
                    )

                    # Add official route geometry with arrow
                    route_layer_name = f"Ruta: {layer_name}"
                    route_group = folium.FeatureGroup(
                        name=route_layer_name,
                        show=(route_idx == 0),
                    )

                    route_line_geom = None
                    if route_geoms_df is not None and not route_geoms_df.empty:
                        try:
                            route_geoms_df_copy = route_geoms_df.copy()
                            route_geoms_df_copy["geometry"] = route_geoms_df_copy[
                                "wkt"
                            ].apply(lambda x: wkt.loads(x))
                            route_gdf = gpd.GeoDataFrame(
                                route_geoms_df_copy,
                                geometry="geometry",
                                crs="EPSG:4326",
                            )

                            route_id_col = (
                                "id" if "id" in route_gdf.columns else "id_ramal"
                            )

                            # Convert to int for matching
                            route_gdf[route_id_col] = pd.to_numeric(
                                route_gdf[route_id_col], errors="coerce"
                            ).astype("Int64")
                            route_gdf["direction"] = pd.to_numeric(
                                route_gdf["direction"], errors="coerce"
                            ).astype("Int64")
                            ramal_conv = (
                                int(ramal)
                                if pd.notna(pd.to_numeric(ramal, errors="coerce"))
                                else ramal
                            )
                            direction_conv = (
                                int(direction)
                                if pd.notna(pd.to_numeric(direction, errors="coerce"))
                                else direction
                            )

                            route_gdf_subset = route_gdf[
                                (route_gdf[route_id_col] == ramal_conv)
                                & (route_gdf["direction"] == direction_conv)
                            ]

                            if not route_gdf_subset.empty:
                                route_line_geom = route_gdf_subset.iloc[0]["geometry"]
                                if hasattr(route_line_geom, "simplify"):
                                    route_line_geom = route_line_geom.simplify(
                                        0.00005, preserve_topology=True
                                    )
                        except Exception:
                            route_line_geom = None

                    if route_line_geom is not None:
                        try:
                            coords = list(route_line_geom.coords)
                            if len(coords) >= 2:
                                line_color = "#1f1f1f"
                                folium.GeoJson(
                                    route_line_geom.__geo_interface__,
                                    style_function=lambda feature, color=line_color: {
                                        "color": color,
                                        "weight": 3,
                                        "opacity": 0.9,
                                    },
                                    tooltip=route_layer_name,
                                    popup=route_layer_name,
                                ).add_to(route_group)

                                end_coord = coords[-1]
                                end_lng, end_lat = end_coord[0], end_coord[1]

                                start_coord = (
                                    coords[-2] if len(coords) >= 2 else coords[0]
                                )
                                start_lng, start_lat = start_coord[0], start_coord[1]

                                dx = end_lng - start_lng
                                dy = end_lat - start_lat
                                heading = math.degrees(math.atan2(dy, dx))

                                folium.RegularPolygonMarker(
                                    location=[end_lat, end_lng],
                                    number_of_sides=3,
                                    radius=7,
                                    rotation=heading + 90,
                                    color=line_color,
                                    fill_color=line_color,
                                    fill=True,
                                    fill_opacity=0.95,
                                    weight=1,
                                    popup=route_layer_name,
                                ).add_to(route_group)
                        except Exception:
                            pass

                    route_group.add_to(m)
                    route_idx += 1

    else:
        # No ramales - group by direction only
        directions = sorted(routes_h3_df["direction"].unique())

        route_idx = 0
        for direction in directions:
            # Get colormap for this direction
            cmap_name = cmaps[direction % len(cmaps)]

            direction_data = routes_h3_df[routes_h3_df["direction"] == direction]

            # Create geometries from H3 cells
            geometries = []
            for h3_cell in direction_data["h3"]:
                try:
                    poly = h3_to_shapely_polygon(h3_cell)
                    geometries.append(poly)
                except Exception:
                    geometries.append(None)

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                direction_data.reset_index(drop=True),
                geometry=geometries,
                crs="EPSG:4326",
            )

            # Remove rows with None geometries
            gdf = gdf[gdf.geometry.notna()]

            if not gdf.empty:
                # Layer name
                layer_name = f"Sentido {direction}"

                # Use explore to add to map for H3 cells
                gdf.explore(
                    m=m,
                    column="section_id",
                    cmap=cmap_name,
                    name=f"Orden de paso: {layer_name}",
                    tooltip=["direction", "section_id", "h3"],
                    popup=True,
                    style_kwds={"fillOpacity": 0.6, "weight": 1},
                    legend=False,
                    show=(route_idx == 0),
                )

                # Add official route geometry with arrow
                route_layer_name = f"Ruta: {layer_name}"
                route_group = folium.FeatureGroup(
                    name=route_layer_name,
                    show=(route_idx == 0),
                )

                route_line_geom = None
                if route_geoms_df is not None and not route_geoms_df.empty:
                    try:
                        route_geoms_df_copy = route_geoms_df.copy()
                        route_geoms_df_copy["geometry"] = route_geoms_df_copy[
                            "wkt"
                        ].apply(lambda x: wkt.loads(x))
                        route_gdf = gpd.GeoDataFrame(
                            route_geoms_df_copy, geometry="geometry", crs="EPSG:4326"
                        )

                        route_id_col = "id" if "id" in route_gdf.columns else "id_linea"

                        # Convert to int for matching
                        route_gdf["direction"] = pd.to_numeric(
                            route_gdf["direction"], errors="coerce"
                        ).astype("Int64")
                        direction_conv = (
                            int(direction)
                            if pd.notna(pd.to_numeric(direction, errors="coerce"))
                            else direction
                        )

                        route_gdf_subset = route_gdf[
                            route_gdf["direction"] == direction_conv
                        ]

                        if not route_gdf_subset.empty:
                            route_line_geom = route_gdf_subset.iloc[0]["geometry"]
                            if hasattr(route_line_geom, "simplify"):
                                route_line_geom = route_line_geom.simplify(
                                    0.00005, preserve_topology=True
                                )
                    except Exception:
                        route_line_geom = None

                if route_line_geom is not None:
                    try:
                        coords = list(route_line_geom.coords)
                        if len(coords) >= 2:
                            line_color = "#1f1f1f"
                            folium.GeoJson(
                                route_line_geom.__geo_interface__,
                                style_function=lambda feature, color=line_color: {
                                    "color": color,
                                    "weight": 3,
                                    "opacity": 0.9,
                                },
                                tooltip=route_layer_name,
                                popup=route_layer_name,
                            ).add_to(route_group)

                            end_coord = coords[-1]
                            end_lng, end_lat = end_coord[0], end_coord[1]

                            start_coord = coords[-2] if len(coords) >= 2 else coords[0]
                            start_lng, start_lat = start_coord[0], start_coord[1]

                            dx = end_lng - start_lng
                            dy = end_lat - start_lat
                            heading = math.degrees(math.atan2(dy, dx))

                            folium.RegularPolygonMarker(
                                location=[end_lat, end_lng],
                                number_of_sides=3,
                                radius=7,
                                rotation=heading + 90,
                                color=line_color,
                                fill_color=line_color,
                                fill=True,
                                fill_opacity=0.95,
                                weight=1,
                                popup=route_layer_name,
                            ).add_to(route_group)
                    except Exception:
                        pass

                route_group.add_to(m)
                route_idx += 1

    # Add layer control
    folium.LayerControl().add_to(m)

    fig.add_child(m)
    return fig
