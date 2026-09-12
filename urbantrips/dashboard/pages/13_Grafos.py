import math
import streamlit as st
import folium
import h3
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
import plotly.express as px
from matplotlib import colormaps
from matplotlib.colors import to_hex
from shapely.geometry import Polygon, LineString
from shapely import wkt
from streamlit_folium import st_folium
from folium import Figure
from dash_utils import (
    get_logo,
    configurar_selector_dia,
)
from urbantrips.utils import utils
from urbantrips.dashboard import dashboard_ctx
from urbantrips.carto.routes import (
    create_routes_h3_directed_graph,
    create_line_h3_directed_graph,
    process_parent_h3_parallel,
    assign_legs_to_line_h3_graph,
    compute_graph_edge_usage,
)
from urbantrips.storage.access import DatabaseBusyError, write_access
from urbantrips.kpi.leg_direction import (
    assign_direction_for_line_and_hours,
    compute_section_usage,
)


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
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles="cartodbpositron",
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
                    name=f"Intensidad: {layer_name}",
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


def create_section_usage_map(
    section_usage_df, routes_h3_df, route_geoms_df, nombre_linea, id_linea, metric_col
):
    """Create a folium choropleth map showing section usage intensity."""
    if section_usage_df.empty or routes_h3_df.empty:
        return None

    # Make copies to avoid modifying original dataframes
    routes_h3_df = routes_h3_df.copy()
    section_usage_df = section_usage_df.copy()

    # Determine if we have branches
    has_branches = "selected_branch" in section_usage_df.columns

    # Fix data type mismatches - convert route IDs to int64 for consistent comparison
    if "id_ramal" in routes_h3_df.columns:
        routes_h3_df["id_ramal"] = routes_h3_df["id_ramal"].astype("int64")
    if "id_linea" in routes_h3_df.columns:
        routes_h3_df["id_linea"] = routes_h3_df["id_linea"].astype("int64")

    # Ensure direction columns are consistent type
    routes_h3_df["direction"] = routes_h3_df["direction"].astype("int64")
    section_usage_df["direction_inferred"] = section_usage_df[
        "direction_inferred"
    ].astype("int64")

    # Calculate map center
    lats = []
    lngs = []
    for h3_cell in routes_h3_df["h3"].head(5):
        lat, lng = h3.cell_to_latlng(h3_cell)
        lats.append(lat)
        lngs.append(lng)

    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    # Create map
    fig = Figure(width=1000, height=700)
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    # Add title
    title_text = f"Uso de Secciones - {nombre_linea} (ID: {id_linea})"
    title_html = f"""
    <h3 align="center" style="font-size:20px"><b>{title_text}</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Create colormap for usage intensity
    vmin = section_usage_df[metric_col].min()
    vmax = section_usage_df[metric_col].max()

    # Group by route and direction
    if has_branches:
        route_col = "selected_branch"
    else:
        route_col = "id_linea"

    # Define colormaps for different ramales/directions - expanded palette
    cmaps = [
        "Reds",
        "Blues",
        "Greens",
        "Purples",
        "Oranges",
        "YlOrBr",
        "Greys",
        "PuRd",
        "GnBu",
        "OrRd",
        "PuBu",
        "YlGn",
        "BuGn",
        "BuPu",
        "RdPu",
        "YlOrRd",
        "PuBuGn",
        "PuGn",
    ]

    # Create a mapping of route_id to colormap for consistency
    route_cmap_map = {}
    unique_routes = sorted(section_usage_df[route_col].unique())
    for idx, route_id in enumerate(unique_routes):
        route_cmap_map[route_id] = cmaps[idx % len(cmaps)]

    # Get unique route-direction combinations
    groups = section_usage_df.groupby([route_col, "direction_inferred"])

    layers_added = 0

    # Plot each route-direction combination
    ordered_routes = list(groups.groups.keys())
    for idx, ((route_id, direction), group_data) in enumerate(groups):
        # Merge with routes_h3 to get H3 cells for each section
        if has_branches:
            route_h3_subset = routes_h3_df[
                (routes_h3_df["id_ramal"] == route_id)
                & (routes_h3_df["direction"] == direction)
            ].copy()
        else:
            route_h3_subset = routes_h3_df[
                (routes_h3_df["id_linea"] == route_id)
                & (routes_h3_df["direction"] == direction)
            ].copy()

        if route_h3_subset.empty:
            continue

        # Merge section usage with H3 cells
        # Build column list avoiding duplicates (metric_col might be n_legs or n_legs_expanded)
        merge_cols = ["section_id", "n_legs", "n_legs_expanded", metric_col]
        merge_cols = list(
            dict.fromkeys(merge_cols)
        )  # Remove duplicates while preserving order

        merged = route_h3_subset.merge(
            group_data[merge_cols],
            on="section_id",
            how="inner",
        )

        if merged.empty:
            continue

        # Create geometries
        geometries = []
        for h3_cell in merged["h3"]:
            try:
                poly = h3_to_shapely_polygon(h3_cell)
                geometries.append(poly)
            except Exception as e:
                geometries.append(None)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            merged.reset_index(drop=True), geometry=geometries, crs="EPSG:4326"
        )
        gdf = gdf[gdf.geometry.notna()]

        if gdf.empty:
            continue

        # Layer name
        if has_branches:
            layer_name = f"Ramal {route_id} - Sentido {direction}"
        else:
            layer_name = f"Línea {route_id} - Sentido {direction}"

        # Get colormap for this route (same for all directions of this route)
        cmap_name = route_cmap_map.get(route_id, cmaps[0])

        try:
            # Simplified route line per ramal/sentido using the official route geometry,
            # plus a final arrow to indicate travel direction.
            route_layer_name = f"Ruta: {layer_name}"
            route_group = folium.FeatureGroup(
                name=route_layer_name,
                show=(idx == 0),
            )

            # Official route geometry used as the base for the simplified route line.
            route_line_geom = None
            if route_geoms_df is None or route_geoms_df.empty:
                st.warning(
                    "No hay geometrías oficiales en route_geoms_df para "
                    f"{layer_name}. Se va a construir la ruta con centroides H3 como fallback."
                )
            elif route_geoms_df is not None and not route_geoms_df.empty:
                try:
                    route_geoms_df_copy = route_geoms_df.copy()
                    route_geoms_df_copy["geometry"] = route_geoms_df_copy["wkt"].apply(
                        lambda x: wkt.loads(x)
                    )
                    route_gdf = gpd.GeoDataFrame(
                        route_geoms_df_copy, geometry="geometry", crs="EPSG:4326"
                    )

                    route_id_col = "id" if "id" in route_gdf.columns else route_col

                    # Convertir explícitamente a int para evitar mismatch string vs int
                    route_gdf[route_id_col] = pd.to_numeric(
                        route_gdf[route_id_col], errors="coerce"
                    ).astype("Int64")
                    route_gdf["direction"] = pd.to_numeric(
                        route_gdf["direction"], errors="coerce"
                    ).astype("Int64")
                    route_id_conv = (
                        int(route_id)
                        if pd.notna(pd.to_numeric(route_id, errors="coerce"))
                        else route_id
                    )
                    direction_conv = (
                        int(direction)
                        if pd.notna(pd.to_numeric(direction, errors="coerce"))
                        else direction
                    )

                    route_gdf_subset = route_gdf[
                        (route_gdf[route_id_col] == route_id_conv)
                        & (route_gdf["direction"] == direction_conv)
                    ]

                    if route_gdf_subset.empty:
                        st.warning(
                            "No se encontró la geometría oficial del ramal/sentido en route_geoms_df para "
                            f"{layer_name}. Se están usando centroides H3 para construir la ruta. "
                            f"route_id_col={route_id_col}, route_id={route_id}, direction={direction}, "
                            f"total_geoms={len(route_gdf)}"
                        )
                    else:
                        route_line_geom = route_gdf_subset.iloc[0]["geometry"]
                        if hasattr(route_line_geom, "simplify"):
                            try:
                                route_line_geom = route_line_geom.simplify(
                                    0.00005, preserve_topology=True
                                )
                            except Exception as simplify_error:
                                st.warning(
                                    "No se pudo simplificar la geometría oficial de la ruta para "
                                    f"{layer_name}. Esto suele pasar cuando la geometría no es una LineString "
                                    "válida, cuando el ramal/sentido no existe en route_geoms_df o cuando "
                                    "la WKT no pudo transformarse correctamente. Se usará el fallback H3. "
                                    f"Detalle: {simplify_error}"
                                )
                                route_line_geom = None
                except Exception as route_geom_error:
                    st.warning(
                        "Fallo la lectura de geometrías oficiales para "
                        f"{layer_name}. Se va a construir esta ruta con centroides H3. "
                        f"Detalle: {route_geom_error}"
                    )
                    route_line_geom = None

            if route_line_geom is None and not gdf.empty:
                gdf_sorted = gdf.sort_values(["section_id"]).copy()
                centroids = []
                for cell in gdf_sorted["h3"]:
                    try:
                        lat, lng = h3.cell_to_latlng(cell)
                        centroids.append((lng, lat))
                    except Exception:
                        continue
                if len(centroids) >= 2:
                    st.warning(
                        "La ruta oficial no se pudo usar para "
                        f"{layer_name}; se está construyendo la ruta con centroides H3 "
                        f"(cantidad centroides={len(centroids)})."
                    )
                    route_line_geom = LineString(centroids)
                else:
                    st.warning(
                        "No hay suficientes centroides para construir la ruta oficial de "
                        f"{layer_name}; no se pudo dibujar la ruta."
                    )

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

                        # Obtener el último punto de la ruta para la flecha
                        end_coord = coords[-1]
                        end_lng, end_lat = end_coord[0], end_coord[1]

                        # Usar los últimos dos puntos para calcular la dirección
                        start_coord = coords[-2] if len(coords) >= 2 else coords[0]
                        start_lng, start_lat = start_coord[0], start_coord[1]

                        dx = end_lng - start_lng
                        dy = end_lat - start_lat
                        heading = math.degrees(math.atan2(dy, dx))

                        # Agregar flecha SOLO al final, una sola vez
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
                    else:
                        st.warning(
                            f"Insuficientes puntos en route_line_geom para {layer_name} "
                            f"(coords={len(coords)}); no se pudo dibujar la ruta ni la flecha."
                        )
                except Exception as arrow_error:
                    st.warning(
                        f"Error al agregar la flecha final a la ruta {layer_name}: {arrow_error}"
                    )

            # Intensity layer for H3 cells remains available in the layer control too.
            intensity_layer_name = f"Intensidad: {layer_name}"
            gdf.explore(
                m=m,
                column=metric_col,
                cmap=cmap_name,
                name=intensity_layer_name,
                tooltip=["section_id", "n_legs_expanded"],
                popup=["section_id", "n_legs_expanded"],
                style_kwds={"fillOpacity": 0.7, "weight": 1, "color": "gray"},
                legend=False,
                vmin=vmin,
                vmax=vmax,
                show=(idx == 0),
            )

            # Add the custom simplified route group after the intensity layer so the route layer
            # appears at the end of the layer control list.
            route_group.add_to(m)
            layers_added += 1
        except Exception:
            pass

    if layers_added == 0:
        return None

    # Add layer control
    folium.LayerControl().add_to(m)

    fig.add_child(m)
    return fig


def create_routes_h3_graph(routes_h3, id_col, h3_cell_interval, has_branches):
    graphs = []
    for (_, _), route_h3 in routes_h3.groupby([id_col, "direction"]):
        graph = create_routes_h3_directed_graph(
            route_h3,
            h3_col="h3",
            section_id_col="section_id",
            h3_cell_interval=h3_cell_interval,
            has_branches=has_branches,
        )
        if graph.number_of_nodes() > 0:
            graphs.append(graph)

    graph = nx.MultiDiGraph()
    if not graphs:
        return graph

    for route_graph in graphs:
        graph.add_nodes_from(route_graph.nodes(data=True))
        for node_from, node_to, attrs in route_graph.edges(data=True):
            graph.add_edge(node_from, node_to, **attrs)

    graph.graph["crs"] = "epsg:4326"
    graph.graph["simplified"] = True
    return graph


def create_routes_h3_graph_map(
    routes_h3,
    route_geoms,
    id_col,
    h3_cell_interval,
    has_branches,
    nombre_linea,
    id_linea,
    line_graph=None,
):
    graphs = []
    groups = []
    for (route_id, direction), route_h3 in routes_h3.groupby([id_col, "direction"]):
        graph = create_routes_h3_directed_graph(
            route_h3,
            h3_col="h3",
            section_id_col="section_id",
            h3_cell_interval=h3_cell_interval,
            has_branches=has_branches,
        )
        if graph.number_of_nodes() > 0:
            graphs.append(graph)
            groups.append((route_id, direction, graph))

    graph = nx.MultiDiGraph()
    for route_graph in graphs:
        graph.add_nodes_from(route_graph.nodes(data=True))
        for node_from, node_to, attrs in route_graph.edges(data=True):
            graph.add_edge(node_from, node_to, **attrs)

    graph.graph["crs"] = "epsg:4326"
    graph.graph["simplified"] = True

    if graph.number_of_nodes() == 0:
        return None, None, None

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
    if nodes_gdf.empty:
        return None, nodes_gdf, edges_gdf

    fig = Figure(width=1000, height=700)
    m = folium.Map(
        location=[nodes_gdf.geometry.y.mean(), nodes_gdf.geometry.x.mean()],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    title_text = f"Grafo H3 - {nombre_linea} (ID: {id_linea})"
    title_html = f"""
    <h3 align="center" style="font-size:20px"><b>{title_text}</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "YlOrBr"]
    cmap_by_route = {}
    if has_branches:
        route_ids = sorted(routes_h3[id_col].dropna().unique())
        for idx, route_id in enumerate(route_ids):
            cmap_by_route[route_id] = cmaps[idx % len(cmaps)]

    for route_idx, (route_id, direction, route_graph) in enumerate(groups):
        group_nodes_gdf, group_edges_gdf = ox.graph_to_gdfs(route_graph)
        if group_nodes_gdf.empty:
            continue

        if has_branches:
            cmap_name = cmap_by_route.get(route_id, cmaps[0])
            layer_name = f"Ramal {route_id} - Sentido {direction}"
        else:
            cmap_name = cmaps[int(direction) % len(cmaps)]
            layer_name = f"Sentido {direction}"

        show_layer = route_idx == 0

        route_line_geom = get_route_line_geom(
            route_geoms,
            route_id=route_id,
            direction=direction,
            route_id_col=id_col,
        )
        if route_line_geom is not None:
            add_route_line_layer(
                m,
                route_line_geom,
                layer_name=f"Ruta: {layer_name}",
                show_layer=show_layer,
            )

        if not group_edges_gdf.empty:
            group_edges_gdf = group_edges_gdf.copy()
            group_edges_gdf["section_id"] = group_edges_gdf["section_id_from"]
            group_edges_gdf.explore(
                m=m,
                column="section_id",
                cmap=cmap_name,
                name=f"Edges: {layer_name}",
                style_kwds={"weight": 3, "opacity": 0.85},
                tooltip=["section_id_from", "section_id_to"],
                popup=["section_id_from", "section_id_to"],
                legend=False,
                show=show_layer,
            )

        route_h3_lookup = routes_h3[
            (routes_h3[id_col] == route_id) & (routes_h3["direction"] == direction)
        ][["h3", "section_id"]].drop_duplicates(subset="h3")
        section_by_h3 = route_h3_lookup.set_index("h3")["section_id"]
        group_nodes_gdf = group_nodes_gdf.copy()
        group_nodes_gdf["section_id"] = group_nodes_gdf.index.map(section_by_h3)
        group_nodes_gdf.explore(
            m=m,
            column="section_id",
            cmap=cmap_name,
            name=f"Nodos: {layer_name}",
            marker_kwds={"radius": 4},
            tooltip=["section_id"],
            popup=["section_id"],
            legend=False,
            show=show_layer,
        )

    if line_graph is not None and line_graph.number_of_nodes() > 0:
        line_nodes_gdf, line_edges_gdf = ox.graph_to_gdfs(line_graph)
        line_edges_gdf = line_edges_gdf.copy()
        line_edges_gdf["u"] = line_edges_gdf.index.get_level_values(0)
        line_edges_gdf["v"] = line_edges_gdf.index.get_level_values(1)

        # Unique line graph: one layer per sentido, merging all ramales
        for direction in sorted(line_edges_gdf["direction"].dropna().unique()):
            dir_edges_gdf = line_edges_gdf[
                line_edges_gdf["direction"] == direction
            ].copy()
            if dir_edges_gdf.empty:
                continue
            dir_node_ids = set(dir_edges_gdf["u"]) | set(dir_edges_gdf["v"])
            dir_nodes_gdf = line_nodes_gdf.loc[line_nodes_gdf.index.isin(dir_node_ids)]

            layer_name = f"Grafo único de línea - Sentido {int(direction)}"
            dir_edges_gdf["section_id"] = dir_edges_gdf["section_id_from"]
            dir_edges_gdf.explore(
                m=m,
                column="section_id",
                cmap="Greys",
                name=f"Edges: {layer_name}",
                style_kwds={"weight": 4, "opacity": 0.9},
                tooltip=["id_ramal", "section_id_from", "section_id_to"],
                popup=["id_ramal", "section_id_from", "section_id_to"],
                legend=False,
                show=False,
            )
            dir_nodes_gdf.explore(
                m=m,
                color="black",
                name=f"Nodos: {layer_name}",
                marker_kwds={"radius": 4},
                legend=False,
                show=False,
            )

    folium.LayerControl().add_to(m)
    fig.add_child(m)
    return fig, nodes_gdf, edges_gdf


def get_route_line_geom(route_geoms, route_id, direction, route_id_col):
    if route_geoms is None or route_geoms.empty:
        return None
    try:
        route_geoms = route_geoms.copy()
        route_geoms["geometry"] = route_geoms["wkt"].apply(lambda x: wkt.loads(x))
        route_gdf = gpd.GeoDataFrame(
            route_geoms,
            geometry="geometry",
            crs="EPSG:4326",
        )
        route_id_col = "id" if "id" in route_gdf.columns else route_id_col
        route_gdf[route_id_col] = pd.to_numeric(
            route_gdf[route_id_col], errors="coerce"
        ).astype("Int64")
        route_gdf["direction"] = pd.to_numeric(
            route_gdf["direction"], errors="coerce"
        ).astype("Int64")
        route_id = (
            int(route_id)
            if pd.notna(pd.to_numeric(route_id, errors="coerce"))
            else route_id
        )
        direction = (
            int(direction)
            if pd.notna(pd.to_numeric(direction, errors="coerce"))
            else direction
        )
        route_gdf_subset = route_gdf[
            (route_gdf[route_id_col] == route_id)
            & (route_gdf["direction"] == direction)
        ]
        if route_gdf_subset.empty:
            return None
        route_line_geom = route_gdf_subset.iloc[0]["geometry"]
        if hasattr(route_line_geom, "simplify"):
            return route_line_geom.simplify(0.00005, preserve_topology=True)
        return route_line_geom
    except Exception:
        return None


def add_route_line_layer(m, route_line_geom, layer_name, show_layer):
    route_group = folium.FeatureGroup(name=layer_name, show=show_layer)
    coords = list(route_line_geom.coords)
    if len(coords) < 2:
        return

    line_color = "#1f1f1f"
    folium.GeoJson(
        route_line_geom.__geo_interface__,
        style_function=lambda feature, color=line_color: {
            "color": color,
            "weight": 3,
            "opacity": 0.9,
        },
        tooltip=layer_name,
        popup=layer_name,
    ).add_to(route_group)

    end_lng, end_lat = coords[-1][0], coords[-1][1]
    start_lng, start_lat = coords[-2][0], coords[-2][1]
    heading = math.degrees(math.atan2(end_lat - start_lat, end_lng - start_lng))
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
        popup=layer_name,
    ).add_to(route_group)
    route_group.add_to(m)


def _scale_edge_widths(values, min_width_m=10, max_width_m=500):
    """Linearly scale a metric column to a buffer width range, in meters."""
    values = values.astype(float)
    vmin, vmax = values.min(), values.max()
    if vmax <= vmin:
        return pd.Series(min_width_m, index=values.index)
    normalized = (values - vmin) / (vmax - vmin)
    return min_width_m + normalized * (max_width_m - min_width_m)


def create_graph_edge_usage_map(
    edge_usage_gdf,
    nombre_linea,
    id_linea,
    metric_col="n_legs_expanded",
    min_width_m=10,
    max_width_m=500,
):
    """Create a folium map showing how many legs used each graph edge.

    Edges are rendered as buffers (instead of plain lines) whose width is
    scaled by ``metric_col``, so busier edges appear visibly thicker.
    """
    if edge_usage_gdf is None or edge_usage_gdf.empty:
        return None

    edge_usage_gdf = edge_usage_gdf.copy()
    centroids = edge_usage_gdf.geometry.centroid

    fig = Figure(width=1000, height=700)
    m = folium.Map(
        location=[centroids.y.mean(), centroids.x.mean()],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    title_text = f"Uso de Edges del Grafo - {nombre_linea} (ID: {id_linea})"
    title_html = f"""
    <h3 align="center" style="font-size:20px"><b>{title_text}</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    vmin = edge_usage_gdf[metric_col].min()
    vmax = edge_usage_gdf[metric_col].max()

    # Buffer each edge in a metric CRS so its width represents demand.
    edge_widths = _scale_edge_widths(
        edge_usage_gdf[metric_col], min_width_m=min_width_m, max_width_m=max_width_m
    )
    edge_usage_m = edge_usage_gdf.to_crs(epsg=3857)
    edge_usage_m["geometry"] = [
        geom.buffer(width / 2)
        for geom, width in zip(edge_usage_m.geometry, edge_widths)
    ]
    edge_usage_buffered = edge_usage_m.to_crs(edge_usage_gdf.crs)

    layers_added = 0
    for direction in sorted(edge_usage_buffered["direction"].dropna().unique()):
        dir_gdf = edge_usage_buffered[
            edge_usage_buffered["direction"] == direction
        ].copy()
        if dir_gdf.empty:
            continue

        layer_name = f"Uso de edges - Sentido {int(direction)}"
        dir_gdf.explore(
            m=m,
            column=metric_col,
            cmap="OrRd",
            name=layer_name,
            style_kwds={"weight": 0, "fillOpacity": 1},
            tooltip=[
                "id_ramal",
                "section_id_from",
                "section_id_to",
                "n_legs",
                metric_col,
            ],
            popup=[
                "id_ramal",
                "section_id_from",
                "section_id_to",
                "n_legs",
                metric_col,
            ],
            legend=False,
            vmin=vmin,
            vmax=vmax,
            show=(layers_added == 0),
        )
        layers_added += 1

    if layers_added == 0:
        return None

    folium.LayerControl().add_to(m)
    fig.add_child(m)
    return fig


def create_graph_od_desire_lines_map(
    od_lines_gdf, nombre_linea, id_linea, metric_col="n_legs_expanded"
):
    """Create a folium map with desire lines between matched graph nodes.

    Lines are drawn from lowest to highest demand, so busier lines end up on
    top of the layer, and their width/opacity scale with demand (same style
    as the líneas de deseo map in the H3 indicators page).
    """
    if od_lines_gdf is None or od_lines_gdf.empty:
        return None

    centroids = od_lines_gdf.geometry.centroid

    fig = Figure(width=1000, height=700)
    m = folium.Map(
        location=[centroids.y.mean(), centroids.x.mean()],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    title_text = f"Líneas de Deseo por Nodo - {nombre_linea} (ID: {id_linea})"
    title_html = f"""
    <h3 align="center" style="font-size:20px"><b>{title_text}</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    vmin = od_lines_gdf[metric_col].min()
    vmax = od_lines_gdf[metric_col].max()

    cmap_by_direction = {0: "PuBu", 1: "OrRd"}
    layers_added = 0
    for direction in sorted(od_lines_gdf["direction_inferred"].dropna().unique()):
        dir_gdf = od_lines_gdf[od_lines_gdf["direction_inferred"] == direction]
        if dir_gdf.empty:
            continue

        layer_name = f"Líneas de deseo - Sentido {int(direction)}"
        od_group = folium.FeatureGroup(name=layer_name, show=(layers_added == 0))
        cmap = colormaps[cmap_by_direction.get(int(direction), "PuBu")]

        for _, row in dir_gdf.sort_values(metric_col, ascending=True).iterrows():
            intensity = (row[metric_col] - vmin) / (vmax - vmin + 1)
            color = to_hex(cmap(0.3 + 0.7 * intensity))
            coords = [(coord[1], coord[0]) for coord in row.geometry.coords]
            folium.PolyLine(
                coords,
                weight=1 + intensity * 7,
                color=color,
                opacity=0.2 + 0.7 * intensity,
                popup=f"{row[metric_col]:.0f} etapas expandidas",
            ).add_to(od_group)

        od_group.add_to(m)
        layers_added += 1

    if layers_added == 0:
        return None

    folium.LayerControl().add_to(m)
    fig.add_child(m)
    return fig


st.set_page_config(layout="wide")

# Header
logo = get_logo()
st.image(logo)
st.title("Grafos")

alias_seleccionado = configurar_selector_dia()

# Configuration and line selection share the same row.
config_col, line_col = st.columns(2)
with config_col:
    st.subheader("Configuración de Resolución H3")
    h3_resolution_selected = st.slider(
        "Seleccione la resolución H3 para visualización",
        min_value=6,
        max_value=10,
        value=9,
        step=1,
        help="Resolución H3: 10 es la más fina (default), 6 es la más gruesa",
    )
    h3_cell_interval = st.number_input(
        "Intervalo de celdas H3 para el grafo",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        help="Usa una celda cada N celdas H3 del recorrido para construir el grafo",
    )

try:
    # Load configurations
    if "configs" not in st.session_state:
        st.session_state.configs = utils.leer_configs_generales(autogenerado=False)

    configs = st.session_state.configs
    h3_res = h3_resolution_selected  # Use selected resolution from slider
    has_branches = configs.get("lineas_contienen_ramales", False)

    # Determine source tables based on configuration
    if has_branches:
        source_table_h3 = "branches"
        source_table_metadata = "ramales"
        id_col = "id_ramal"
        name_col = "nombre_ramal"
    else:
        source_table_h3 = "lines"
        source_table_metadata = "lineas"
        id_col = "id_linea"
        name_col = "nombre_linea"

    # Load metadata - always from metadata_lineas
    with dashboard_ctx() as ctx:
        metadata_query = """
        SELECT id_linea, nombre_linea
        FROM metadata_lineas
        WHERE modo = 'autobus'
        ORDER BY id_linea
        """
        metadata_lineas = ctx.insumos.query(metadata_query)


except ValueError as e:
    st.error(
        f"Falta una base de datos requerida: {e}. \n"
        "Se requiere full acceso a Urbantrips para correr esta página"
    )
    st.stop()

# Initialize session state variables
for var in ["id_linea_h3", "nombre_linea_h3"]:
    if var not in st.session_state:
        st.session_state[var] = None

# Line selection
with line_col:
    st.subheader("Selección de Línea")
    seleccionar_linea("h3_input", "h3_select", metadata_lineas)

id_linea = st.session_state.get("id_linea_h3")
nombre_linea = st.session_state.get("nombre_linea_h3", "")

# Display map if line is selected
if id_linea is not None:
    st.subheader(f"Grafo H3 - {nombre_linea} (ID: {id_linea})")

    col_load_graph, col_clear_graph = st.columns([1, 1])
    with col_load_graph:
        load_graph_button = st.button("Cargar grafo", key="load_graph_button")
    with col_clear_graph:
        clear_graph_button = st.button("Limpiar grafo", key="clear_graph_button")

    graph_cache_key = (int(id_linea), int(h3_res), int(h3_cell_interval))
    cached_graph_key = st.session_state.get("routes_graph_cache_key")
    if cached_graph_key != graph_cache_key:
        st.session_state.pop("routes_graph_cache", None)

    if clear_graph_button:
        st.session_state.pop("routes_graph_cache", None)
        st.session_state.pop("routes_graph_cache_key", None)

    if load_graph_button or "routes_graph_cache" in st.session_state:
        with st.spinner("Cargando grafo H3..."):
            try:
                table_exists = False
                route_filter = None
                table_name = None
                where_clause = None
                source_table_h3_saved = source_table_h3

                with dashboard_ctx() as ctx:
                    metadata_query = f"""
                    SELECT *
                    FROM metadata_{source_table_metadata}
                    WHERE modo = 'autobus' AND id_linea = {id_linea}
                    """
                    metadata = ctx.insumos.query(metadata_query)

                    if has_branches:
                        if metadata.empty:
                            st.warning(
                                f"No se encontraron ramales para la línea {id_linea}"
                            )
                            st.stop()
                        ramales_list = metadata[id_col].unique().tolist()
                        ramales_str = ",".join(map(str, ramales_list))
                        route_filter = f"{id_col} IN ({ramales_str})"
                    else:
                        route_filter = f"id_linea = {id_linea}"

                    if h3_res == 10:
                        table_name = f"official_{source_table_h3}_geoms_h3"
                        where_clause = route_filter
                    else:
                        table_name = f"official_{source_table_h3}_geoms_h3_parent"
                        where_clause = f"resolution = {h3_res} AND {route_filter}"

                    routes_h3_query = f"""
                    SELECT COUNT(*) as count
                    FROM {table_name}
                    WHERE {where_clause}
                    """
                    try:
                        check_result = ctx.insumos.query(routes_h3_query)
                        table_exists = check_result["count"].iloc[0] > 0
                    except Exception:
                        table_exists = False

                if not table_exists and h3_res < 10:
                    try:
                        with (
                            write_access(f"crear H3 res {h3_res}"),
                            dashboard_ctx() as ctx,
                        ):
                            with st.spinner(
                                f"Creando geometrías H3 en resolución {h3_res}... "
                                "Esto puede tardar unos minutos."
                            ):
                                st.info(
                                    f"La resolución H3={h3_res} no existe "
                                    "en la base de datos. Creando..."
                                )

                                base_table = f"official_{source_table_h3}_geoms_h3"
                                base_query = f"""
                                SELECT *
                                FROM {base_table}
                                WHERE {route_filter}
                                """
                                routes_h3_base = ctx.insumos.query(base_query)

                                if not routes_h3_base.empty:
                                    if has_branches:
                                        geoms_table = "official_branches_geoms"
                                        geoms_query = f"""
                                        SELECT id_ramal, direction, wkt
                                        FROM {geoms_table}
                                        WHERE {route_filter}
                                        """
                                    else:
                                        geoms_table = "official_lines_geoms"
                                        geoms_query = f"""
                                        SELECT id_linea, direction, wkt
                                        FROM {geoms_table}
                                        WHERE {route_filter}
                                        """

                                    geoms_df = ctx.insumos.query(geoms_query)
                                    geoms_df["geometry"] = geoms_df["wkt"].apply(
                                        wkt.loads
                                    )
                                    geoms_gdf = gpd.GeoDataFrame(
                                        geoms_df,
                                        geometry="geometry",
                                        crs="EPSG:4326",
                                    )

                                    parent_h3_result = process_parent_h3_parallel(
                                        routes_h3_df=routes_h3_base,
                                        routes_geoms_gdf=geoms_gdf,
                                        route_id_column=id_col,
                                        parent_res=h3_res,
                                    )

                                    if len(parent_h3_result) > 0:
                                        parent_h3_result = parent_h3_result.rename(
                                            columns={"parent_h3": "h3"}
                                        )
                                        parent_h3_result = parent_h3_result.reindex(
                                            columns=[
                                                id_col,
                                                "direction",
                                                "section_id",
                                                "h3",
                                                "resolution",
                                                "wkt",
                                            ]
                                        )
                                        ctx.insumos.append_raw(
                                            parent_h3_result, table_name
                                        )
                                        st.success(
                                            f"Geometrías H3 creadas exitosamente "
                                            f"para resolución {h3_res}"
                                        )
                                        table_exists = True
                                    else:
                                        st.error(
                                            "No se pudieron crear las geometrías H3"
                                        )
                                        st.stop()
                                else:
                                    st.error(
                                        f"No se encontraron datos base para crear "
                                        f"resolución {h3_res}"
                                    )
                                    st.stop()
                    except DatabaseBusyError as e:
                        st.error(
                            f"{e} Cerrá el otro dashboard o esperá a que "
                            "termine la corrida y volvé a intentar."
                        )
                        st.stop()

                with dashboard_ctx() as ctx:
                    routes_h3_query = f"""
                    SELECT {id_col}, direction, section_id, h3
                    FROM {table_name}
                    WHERE {where_clause}
                    ORDER BY direction, section_id
                    """
                    routes_h3 = ctx.insumos.query(routes_h3_query)

                    if has_branches:
                        geoms_table = "official_branches_geoms"
                        geoms_query = f"""
                        SELECT {id_col} as id, direction, wkt
                        FROM {geoms_table}
                        WHERE {route_filter}
                        """
                    else:
                        geoms_table = "lines_geoms"
                        geoms_query = f"""
                        SELECT id_linea as id, direction, wkt
                        FROM {geoms_table}
                        WHERE {route_filter}
                        """

                    try:
                        route_geoms = ctx.insumos.query(geoms_query)
                    except Exception as e:
                        st.warning(f"No se encontraron geometrías oficiales: {e}")
                        route_geoms = None

                    if routes_h3.empty:
                        st.warning(
                            f"No se encontraron celdas H3 para la línea "
                            f"{nombre_linea} (ID: {id_linea}) en resolución {h3_res}"
                        )
                    else:
                        line_graph = create_line_h3_directed_graph(
                            ctx,
                            id_linea=id_linea,
                            h3_res=h3_res,
                            h3_cell_interval=int(h3_cell_interval),
                            has_branches=has_branches,
                        )
                        fig, nodes_gdf, edges_gdf = create_routes_h3_graph_map(
                            routes_h3,
                            route_geoms,
                            id_col=id_col,
                            h3_cell_interval=int(h3_cell_interval),
                            has_branches=has_branches,
                            nombre_linea=nombre_linea,
                            id_linea=id_linea,
                            line_graph=line_graph,
                        )

                        st.session_state["routes_graph_cache_key"] = graph_cache_key
                        st.session_state["routes_graph_cache"] = {
                            "routes_h3": routes_h3,
                            "nodes_gdf": nodes_gdf,
                            "edges_gdf": edges_gdf,
                            "fig": fig,
                            "nombre_linea": nombre_linea,
                            "id_linea": id_linea,
                        }

                graph_cache = st.session_state.get("routes_graph_cache")
                if graph_cache is not None:
                    routes_h3 = graph_cache["routes_h3"]

                    if "id_ramal" in routes_h3.columns:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            n_ramales = routes_h3["id_ramal"].nunique()
                            st.metric("Ramales", n_ramales)
                        with col2:
                            n_directions = routes_h3["direction"].nunique()
                            st.metric("Direcciones", n_directions)
                        with col3:
                            st.metric("Resolución H3", h3_res)
                        with col4:
                            st.metric("Intervalo", int(h3_cell_interval))
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_directions = routes_h3["direction"].nunique()
                            st.metric("Direcciones", n_directions)
                        with col2:
                            st.metric("Resolución H3", h3_res)
                        with col3:
                            st.metric("Intervalo", int(h3_cell_interval))

                    fig = graph_cache["fig"]
                    if fig is not None:
                        st_folium(
                            fig,
                            width=1000,
                            height=700,
                            key="routes_graph_map",
                        )
                    else:
                        st.error("Error al crear el mapa del grafo")
            except DatabaseBusyError as e:
                st.error(
                    f"{e} Cerrá el otro dashboard o esperá a que "
                    "termine la corrida y volvé a intentar."
                )
            except Exception as e:
                st.error(f"Error al cargar grafo H3: {e}")
                st.exception(e)

    st.markdown("---")
    st.subheader("Análisis de Demanda sobre el Grafo")
    st.markdown(
        "Rutea las etapas clasificadas sobre el grafo único de línea para "
        "estimar el uso de cada edge y armar la matriz OD por nodo."
    )

    col_hora1, col_hora2 = st.columns(2)
    with col_hora1:
        hora_inicio_grafo = st.number_input(
            "Hora inicio (0-23)",
            min_value=0,
            max_value=23,
            value=9,
            step=1,
            key="hora_inicio_grafo",
        )
    with col_hora2:
        hora_fin_grafo = st.number_input(
            "Hora fin (0-23)",
            min_value=0,
            max_value=23,
            value=9,
            step=1,
            key="hora_fin_grafo",
        )

    try:
        with dashboard_ctx() as ctx:
            dias_disponibles_grafo_df = ctx.data.query(f"""
                SELECT DISTINCT dia
                FROM etapas
                WHERE id_linea = {id_linea}
                ORDER BY dia
                """)
        dias_disponibles_grafo = (
            dias_disponibles_grafo_df["dia"].dropna().astype(str).tolist()
        )
    except Exception as e:
        st.error(f"No se pudieron cargar las fechas disponibles: {e}")
        st.stop()

    if not dias_disponibles_grafo:
        st.error(f"No hay fechas disponibles en etapas para la línea {id_linea}.")
        st.stop()

    dia_especifico_grafo = st.selectbox(
        "Fecha disponible",
        options=dias_disponibles_grafo,
        key="dia_especifico_grafo",
    )

    calc_demand_cols = st.columns([3, 1])
    with calc_demand_cols[0]:
        compute_demand_button = st.button(
            "Calcular Demanda sobre el Grafo", type="primary"
        )
    with calc_demand_cols[1]:
        clear_demand_button = st.button(
            "Limpiar Resultados", key="clear_demand_results"
        )

    if clear_demand_button and "graph_demand_results" in st.session_state:
        del st.session_state["graph_demand_results"]
        if "graph_demand_cache_key" in st.session_state:
            del st.session_state["graph_demand_cache_key"]
        st.rerun()

    current_demand_key = (
        int(id_linea),
        int(h3_res),
        int(h3_cell_interval),
        int(hora_inicio_grafo),
        int(hora_fin_grafo),
        dia_especifico_grafo,
    )

    cached_demand_key = st.session_state.get("graph_demand_cache_key")
    if (
        cached_demand_key == current_demand_key
        and "graph_demand_results" in st.session_state
    ):
        st.info("Usando resultados ya calculados para estos filtros.")
        compute_demand_button = False

    if compute_demand_button:
        if hora_inicio_grafo > hora_fin_grafo:
            st.error("La hora de inicio debe ser menor que la hora de fin")
        else:
            with st.spinner(
                "Ruteando etapas sobre el grafo... Esto puede tardar unos minutos."
            ):
                try:
                    with (
                        write_access(f"calcular demanda sobre grafo línea {id_linea}"),
                        dashboard_ctx() as ctx,
                    ):
                        legs_with_dir = assign_direction_for_line_and_hours(
                            ctx,
                            id_linea=id_linea,
                            hora_inicio=hora_inicio_grafo,
                            hora_fin=hora_fin_grafo,
                            dia=dia_especifico_grafo,
                        )

                        if legs_with_dir.empty:
                            st.warning(
                                "No se encontraron etapas para los filtros "
                                "especificados"
                            )
                            st.stop()

                        where_clause_demand = " AND ".join(
                            [
                                f"ldbl.id_linea = {id_linea}",
                                f"ldbl.hora >= {hora_inicio_grafo}",
                                f"ldbl.hora <= {hora_fin_grafo}",
                                f"ldbl.dia = '{dia_especifico_grafo}'",
                            ]
                        )

                        legs_query = f"""
                        SELECT ldbl.*, e.factor_expansion_linea, e.h3_o, e.h3_d
                        FROM legs_direction_branch_line ldbl
                        LEFT JOIN etapas e
                            ON ldbl.id = e.id
                           AND ldbl.dia = e.dia
                        WHERE {where_clause_demand}
                        """
                        legs_for_demand = ctx.data.query(legs_query)

                        if legs_for_demand.empty:
                            st.warning(
                                "No se encontraron etapas con dirección asignada"
                            )
                            st.stop()

                        demand_line_graph = create_line_h3_directed_graph(
                            ctx,
                            id_linea=id_linea,
                            h3_res=h3_res,
                            h3_cell_interval=int(h3_cell_interval),
                            has_branches=has_branches,
                        )

                        if demand_line_graph.number_of_nodes() == 0:
                            st.warning("No se pudo construir el grafo de línea")
                            st.stop()

                        matched_legs = assign_legs_to_line_h3_graph(
                            demand_line_graph,
                            legs_for_demand,
                            ring_size=1,
                        )

                        if matched_legs.empty:
                            st.warning(
                                "Ninguna etapa pudo asignarse a un nodo del grafo"
                            )
                            st.stop()

                        edge_usage = compute_graph_edge_usage(
                            demand_line_graph,
                            matched_legs,
                            ring_size=1,
                        )

                        st.session_state["graph_demand_results"] = {
                            "legs_for_demand": legs_for_demand,
                            "matched_legs": matched_legs,
                            "edge_usage": edge_usage,
                            "line_graph": demand_line_graph,
                            "nombre_linea": nombre_linea,
                            "id_linea": id_linea,
                        }
                        st.session_state["graph_demand_cache_key"] = current_demand_key
                except DatabaseBusyError as e:
                    st.error(
                        f"{e} Cerrá el otro dashboard o esperá a que "
                        "termine la corrida y volvé a intentar."
                    )
                except Exception as e:
                    st.error(f"Error al calcular demanda sobre el grafo: {e}")
                    st.exception(e)

    if "graph_demand_results" in st.session_state:
        demand_results = st.session_state["graph_demand_results"]
        legs_for_demand = demand_results["legs_for_demand"]
        matched_legs = demand_results["matched_legs"]
        edge_usage = demand_results["edge_usage"]
        line_graph_demand = demand_results["line_graph"]
        nombre_linea_demand = demand_results["nombre_linea"]
        id_linea_demand = demand_results["id_linea"]

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Etapas con dirección asignada", len(legs_for_demand))
        with col_m2:
            st.metric("Etapas asignadas a nodos", len(matched_legs))
        with col_m3:
            st.metric("Edges con uso", len(edge_usage))

        if not edge_usage.empty:
            with st.expander("Uso de edges (grafo)", expanded=False):
                if st.button(
                    (
                        "Ocultar datos de uso de edges"
                        if st.session_state.get("show_edge_usage_data", False)
                        else "Mostrar datos de uso de edges"
                    ),
                    key="toggle_edge_usage_data",
                ):
                    st.session_state["show_edge_usage_data"] = not st.session_state.get(
                        "show_edge_usage_data", False
                    )

                if st.session_state.get("show_edge_usage_data", False):
                    edge_usage_display = edge_usage.drop(
                        columns=[
                            "id_ramal",
                            "section_id_from",
                            "section_id_to",
                            "geometry",
                        ],
                        errors="ignore",
                    ).copy()
                    edge_usage_display["wkt"] = edge_usage["geometry"].apply(
                        lambda geometry: geometry.wkt if geometry is not None else None
                    )
                    edge_usage_display["n_legs_expanded"] = (
                        pd.to_numeric(
                            edge_usage_display["n_legs_expanded"], errors="coerce"
                        )
                        .round()
                        .astype("Int64")
                    )
                    st.dataframe(
                        edge_usage_display,
                        use_container_width=True,
                    )

                edge_usage_fig = create_graph_edge_usage_map(
                    edge_usage, nombre_linea_demand, id_linea_demand
                )
                if edge_usage_fig is not None:
                    st_folium(
                        edge_usage_fig,
                        width=1000,
                        height=700,
                        key="edge_usage_map",
                    )
                else:
                    st.error("Error al crear el mapa de uso de edges")

        if not matched_legs.empty:
            with st.expander("Matriz OD por nodo y líneas de deseo", expanded=False):
                od_data = matched_legs.copy()
                od_data["factor_expansion_linea"] = pd.to_numeric(
                    od_data.get("factor_expansion_linea", 1), errors="coerce"
                ).fillna(1)

                od_counts = (
                    od_data.groupby(
                        ["graph_node_o", "graph_node_d", "direction_inferred"]
                    )
                    .agg(n_legs_expanded=("factor_expansion_linea", "sum"))
                    .reset_index()
                )

                node_xy = {
                    node: (data.get("x"), data.get("y"))
                    for node, data in line_graph_demand.nodes(data=True)
                }

                def _od_geometry(row, node_xy=node_xy):
                    xy_o = node_xy.get(row["graph_node_o"])
                    xy_d = node_xy.get(row["graph_node_d"])
                    if (
                        xy_o is None
                        or xy_d is None
                        or xy_o[0] is None
                        or xy_d[0] is None
                    ):
                        return None
                    return LineString([xy_o, xy_d])

                od_counts["geometry"] = od_counts.apply(_od_geometry, axis=1)
                od_counts = od_counts[od_counts["geometry"].notna()]

                if od_counts.empty:
                    st.info("No se pudieron construir líneas de deseo por nodo")
                else:
                    od_lines_gdf = gpd.GeoDataFrame(
                        od_counts, geometry="geometry", crs="EPSG:4326"
                    )

                    od_pivot = od_data.pivot_table(
                        values="factor_expansion_linea",
                        index="graph_node_o",
                        columns="graph_node_d",
                        aggfunc="sum",
                        fill_value=0,
                    )

                    if st.button(
                        (
                            "Ocultar datos de líneas de deseo"
                            if st.session_state.get("show_graph_od_data", False)
                            else "Mostrar datos de líneas de deseo"
                        ),
                        key="toggle_graph_od_data",
                    ):
                        st.session_state["show_graph_od_data"] = (
                            not st.session_state.get("show_graph_od_data", False)
                        )

                    if st.session_state.get("show_graph_od_data", False):
                        od_lines_display = od_lines_gdf.drop(columns="geometry").copy()
                        od_lines_display["n_legs_expanded"] = (
                            pd.to_numeric(
                                od_lines_display["n_legs_expanded"], errors="coerce"
                            )
                            .round()
                            .astype("Int64")
                        )
                        st.dataframe(
                            od_lines_display,
                            use_container_width=True,
                        )
                        st.markdown("**Matriz OD por nodo**")
                        st.dataframe(od_pivot, use_container_width=True)

                    od_fig = create_graph_od_desire_lines_map(
                        od_lines_gdf, nombre_linea_demand, id_linea_demand
                    )
                    if od_fig is not None:
                        st_folium(
                            od_fig,
                            width=1000,
                            height=700,
                            key="graph_od_map",
                        )
                    else:
                        st.error("Error al crear el mapa de líneas de deseo")
