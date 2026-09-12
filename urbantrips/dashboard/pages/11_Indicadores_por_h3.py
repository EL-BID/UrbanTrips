import math
import streamlit as st
import folium
import h3
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
from urbantrips.carto.routes import process_parent_h3_parallel
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


st.set_page_config(layout="wide")

# Header
logo = get_logo()
st.image(logo)
st.title("Indicadores por H3")

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
    st.subheader(f"Geometrías H3 - {nombre_linea} (ID: {id_linea})")

    col_load_h3, col_clear_h3 = st.columns([1, 1])
    with col_load_h3:
        load_h3_button = st.button("Cargar geometrías H3", key="load_h3_button")
    with col_clear_h3:
        clear_h3_button = st.button("Limpiar geometrías H3", key="clear_h3_button")

    if clear_h3_button and "routes_h3_cache" in st.session_state:
        del st.session_state["routes_h3_cache"]

    if load_h3_button or "routes_h3_cache" in st.session_state:
        with st.spinner("Cargando geometrías H3..."):
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
                            f"No se encontraron geometrías H3 para la línea "
                            f"{nombre_linea} (ID: {id_linea}) en resolución {h3_res}"
                        )
                    else:
                        st.session_state["routes_h3_cache"] = {
                            "routes_h3": routes_h3,
                            "route_geoms": route_geoms,
                            "nombre_linea": nombre_linea,
                            "id_linea": id_linea,
                        }

                        if "id_ramal" in routes_h3.columns:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total de celdas H3", len(routes_h3))
                            with col2:
                                n_ramales = routes_h3["id_ramal"].nunique()
                                st.metric("Ramales", n_ramales)
                            with col3:
                                n_directions = routes_h3["direction"].nunique()
                                st.metric("Direcciones", n_directions)
                            with col4:
                                st.metric("Resolución H3", h3_res)
                        else:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total de celdas H3", len(routes_h3))
                            with col2:
                                n_directions = routes_h3["direction"].nunique()
                                st.metric("Direcciones", n_directions)
                            with col3:
                                st.metric("Resolución H3", h3_res)

                        if st.button(
                            (
                                "Ocultar datos"
                                if st.session_state.get("show_routes_h3_data", False)
                                else "Mostrar datos"
                            ),
                            key="toggle_routes_h3_data",
                        ):
                            st.session_state["show_routes_h3_data"] = (
                                not st.session_state.get("show_routes_h3_data", False)
                            )

                        if st.session_state.get("show_routes_h3_data", False):
                            routes_h3_display = routes_h3.copy()
                            routes_h3_display["wkt"] = routes_h3_display["h3"].apply(
                                lambda h3_cell: h3_to_shapely_polygon(h3_cell).wkt
                            )
                            st.dataframe(routes_h3_display)

                        fig = create_routes_h3_map(
                            routes_h3,
                            route_geoms,
                            nombre_linea,
                            id_linea,
                        )

                        if fig is not None:
                            st_folium(fig, width=1000, height=700, key="routes_h3_map")
                        else:
                            st.error("Error al crear el mapa")
            except DatabaseBusyError as e:
                st.error(
                    f"{e} Cerrá el otro dashboard o esperá a que "
                    "termine la corrida y volvé a intentar."
                )
            except Exception as e:
                st.error(f"Error al cargar geometrías H3: {e}")
                st.exception(e)

    st.markdown("---")
    st.subheader("Análisis de Uso de Secciones")
    st.markdown(
        "Visualiza la intensidad de uso de cada sección de la ruta "
        "basado en los viajes realizados."
    )

    col1, col2 = st.columns(2)
    with col1:
        hora_inicio = st.number_input(
            "Hora inicio (0-23)",
            min_value=0,
            max_value=23,
            value=9,
            step=1,
        )
    with col2:
        hora_fin = st.number_input(
            "Hora fin (0-23)",
            min_value=0,
            max_value=23,
            value=9,
            step=1,
        )

    try:
        with dashboard_ctx() as ctx:
            dias_disponibles_df = ctx.data.query(f"""
                SELECT DISTINCT dia
                FROM etapas
                WHERE id_linea = {id_linea}
                ORDER BY dia
                """)
        dias_disponibles = dias_disponibles_df["dia"].dropna().astype(str).tolist()
    except Exception as e:
        st.error(f"No se pudieron cargar las fechas disponibles: {e}")
        st.stop()

    if not dias_disponibles:
        st.error(f"No hay fechas disponibles en etapas para la línea {id_linea}.")
        st.stop()

    dia_especifico = st.selectbox(
        "Fecha disponible",
        options=dias_disponibles,
        key="dia_especifico_h3",
    )

    metric_choice = "n_legs_expanded"
    calc_cols = st.columns([3, 1])
    with calc_cols[0]:
        compute_button = st.button("Calcular Uso de Secciones", type="primary")
    with calc_cols[1]:
        clear_results_button = st.button(
            "Limpiar Resultados", key="clear_usage_results"
        )

    if clear_results_button and "section_usage_results" in st.session_state:
        del st.session_state["section_usage_results"]
        if "section_usage_cache_key" in st.session_state:
            del st.session_state["section_usage_cache_key"]
        st.rerun()

    current_usage_key = (
        int(id_linea),
        int(h3_res),
        int(hora_inicio),
        int(hora_fin),
        dia_especifico,
    )

    cached_key = st.session_state.get("section_usage_cache_key")
    if cached_key == current_usage_key and "section_usage_results" in st.session_state:
        st.info("Usando resultados ya calculados para estos filtros.")
        compute_button = False

    if compute_button:
        if hora_inicio > hora_fin:
            st.error("La hora de inicio debe ser menor que la hora de fin")
        else:
            with st.spinner(
                "Calculando uso de secciones... Esto puede tardar unos minutos."
            ):
                try:
                    with (
                        write_access(f"calcular uso secciones línea {id_linea}"),
                        dashboard_ctx() as ctx,
                    ):
                        legs_with_dir = assign_direction_for_line_and_hours(
                            ctx,
                            id_linea=id_linea,
                            hora_inicio=hora_inicio,
                            hora_fin=hora_fin,
                            dia=dia_especifico,
                        )

                        if legs_with_dir.empty:
                            st.warning(
                                "No se encontraron etapas para los filtros especificados"
                            )
                            st.stop()

                        where_parts = [f"ldbl.id_linea = {id_linea}"]
                        if hora_inicio is not None:
                            where_parts.append(f"ldbl.hora >= {hora_inicio}")
                        if hora_fin is not None:
                            where_parts.append(f"ldbl.hora <= {hora_fin}")
                        where_parts.append(f"ldbl.dia = '{dia_especifico}'")
                        where_clause = " AND ".join(where_parts)

                        legs_query = f"""
                        SELECT ldbl.*, e.factor_expansion_linea, e.h3_o, e.h3_d
                        FROM legs_direction_branch_line ldbl
                        LEFT JOIN etapas e
                            ON ldbl.id = e.id
                           AND ldbl.dia = e.dia
                        WHERE {where_clause}
                        """
                        legs_for_usage = ctx.data.query(legs_query)

                        if legs_for_usage.empty:
                            st.warning(
                                "No se encontraron etapas con dirección asignada"
                            )
                            st.stop()

                        if h3_res != 10:
                            routes_usage_table = (
                                f"official_{source_table_h3_saved}_geoms_h3_parent"
                            )
                            routes_usage_where = (
                                f"resolution = {h3_res} AND {route_filter}"
                            )
                        else:
                            routes_usage_table = (
                                f"official_{source_table_h3_saved}_geoms_h3"
                            )
                            routes_usage_where = route_filter

                        routes_for_usage_query = f"""
                        SELECT *
                        FROM {routes_usage_table}
                        WHERE {routes_usage_where}
                        """
                        routes_for_usage = ctx.insumos.query(routes_for_usage_query)

                        if routes_for_usage.empty:
                            st.error("No se encontraron rutas H3 para la línea")
                            st.stop()

                        route_geoms_for_usage = st.session_state.get(
                            "routes_h3_cache", {}
                        ).get("route_geoms")
                        if route_geoms_for_usage is None:
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
                                route_geoms_for_usage = ctx.insumos.query(geoms_query)
                            except Exception:
                                route_geoms_for_usage = None

                        section_usage = compute_section_usage(
                            legs_df=legs_for_usage,
                            routes_df=routes_for_usage,
                            metadata=metadata,
                            h3_resolution=h3_res,
                        )

                        if section_usage.empty:
                            st.warning("No se pudo calcular el uso de secciones")
                            if "section_usage_results" in st.session_state:
                                del st.session_state["section_usage_results"]
                            st.stop()

                        st.session_state["section_usage_results"] = {
                            "section_usage": section_usage,
                            "routes_h3": routes_for_usage,
                            "route_geoms": route_geoms_for_usage,
                            "nombre_linea": nombre_linea,
                            "id_linea": id_linea,
                            "legs_for_usage": legs_for_usage,
                        }
                        st.session_state["section_usage_cache_key"] = current_usage_key
                except DatabaseBusyError as e:
                    st.error(
                        f"{e} Cerrá el otro dashboard o esperá a que "
                        "termine la corrida y volvé a intentar."
                    )
                except Exception as e:
                    st.error(f"Error al calcular uso de secciones: {e}")
                    st.exception(e)

    if "section_usage_results" in st.session_state:
        results = st.session_state["section_usage_results"]
        section_usage = results["section_usage"]
        routes_h3_for_display = results["routes_h3"]
        route_geoms_for_display = results["route_geoms"]
        nombre_linea_display = results["nombre_linea"]
        id_linea_display = results["id_linea"]
        legs_for_usage = results["legs_for_usage"]
        legs_for_usage_display = legs_for_usage.copy()
        if "factor_expansion_linea" in legs_for_usage_display.columns:
            legs_for_usage_display["factor_expansion_linea"] = (
                pd.to_numeric(
                    legs_for_usage_display["factor_expansion_linea"],
                    errors="coerce",
                )
                .round()
                .astype("Int64")
            )
        section_usage_display = section_usage.copy()
        if "n_legs_expanded" in section_usage_display.columns:
            section_usage_display["n_legs_expanded"] = (
                pd.to_numeric(section_usage_display["n_legs_expanded"], errors="coerce")
                .round()
                .astype("Int64")
            )

        st.markdown("### Estadísticas de Uso")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total secciones usadas", len(section_usage))
        with col2:
            st.metric(
                "Total etapas (sin expandir)",
                int(section_usage["n_legs"].sum()),
            )
        with col3:
            st.metric(
                "Total etapas (expandidas)",
                f"{section_usage['n_legs_expanded'].sum():.0f}",
            )

        if st.button(
            (
                "Ocultar todos los datos de uso"
                if st.session_state.get("show_all_usage_data", False)
                else "Mostrar todos los datos de uso"
            ),
            key="toggle_all_usage_data",
        ):
            st.session_state["show_all_usage_data"] = not st.session_state.get(
                "show_all_usage_data", False
            )

        if st.session_state.get("show_all_usage_data", False):
            st.dataframe(legs_for_usage_display, use_container_width=True)

        if (
            not section_usage.empty
            and routes_h3_for_display is not None
            and not routes_h3_for_display.empty
        ):
            with st.expander(
                "Mapa de intensidad de uso por sección H3", expanded=False
            ):
                st.caption(f"Resolución H3: {h3_res}")
                if st.button(
                    (
                        "Ocultar datos de section_usage"
                        if st.session_state.get("show_section_usage_data", False)
                        else "Mostrar datos de section_usage"
                    ),
                    key="toggle_section_usage_data",
                ):
                    st.session_state["show_section_usage_data"] = (
                        not st.session_state.get("show_section_usage_data", False)
                    )

                if st.session_state.get("show_section_usage_data", False):
                    st.dataframe(section_usage_display, use_container_width=True)
                usage_fig = create_section_usage_map(
                    section_usage_df=section_usage,
                    routes_h3_df=routes_h3_for_display,
                    route_geoms_df=route_geoms_for_display,
                    nombre_linea=nombre_linea_display,
                    id_linea=id_linea_display,
                    metric_col=metric_choice,
                )

                if usage_fig is not None:
                    st_folium(
                        usage_fig,
                        width=1000,
                        height=700,
                        key="section_usage_map",
                    )
                else:
                    st.error("Error al crear el mapa de uso")

            od_matrix = None
            with st.expander("Líneas de deseo por H3", expanded=False):
                st.caption(f"Resolución H3: {h3_res}")
                legs_df = results.get("legs_for_usage")
                routes_h3_od = results.get("routes_h3")
                route_geoms_od = results.get("route_geoms")

                if (
                    legs_df is not None
                    and not legs_df.empty
                    and "h3_o" in legs_df.columns
                    and "h3_d" in legs_df.columns
                ):
                    # Convert h3_o and h3_d to the user-selected resolution
                    legs_od = legs_df[
                        (legs_df["h3_o"].notna()) & (legs_df["h3_d"].notna())
                    ].copy()

                    # Convert origin and destination cells to the selected H3 resolution
                    legs_od["h3_o"] = legs_od["h3_o"].apply(
                        lambda x: h3.cell_to_parent(x, h3_res) if pd.notna(x) else None
                    )
                    legs_od["h3_d"] = legs_od["h3_d"].apply(
                        lambda x: h3.cell_to_parent(x, h3_res) if pd.notna(x) else None
                    )

                    if not legs_od.empty:
                        legs_od["lat_o"] = legs_od["h3_o"].apply(
                            lambda x: h3.cell_to_latlng(x)[0] if pd.notna(x) else None
                        )
                        legs_od["lon_o"] = legs_od["h3_o"].apply(
                            lambda x: h3.cell_to_latlng(x)[1] if pd.notna(x) else None
                        )
                        legs_od["lat_d"] = legs_od["h3_d"].apply(
                            lambda x: h3.cell_to_latlng(x)[0] if pd.notna(x) else None
                        )
                        legs_od["lon_d"] = legs_od["h3_d"].apply(
                            lambda x: h3.cell_to_latlng(x)[1] if pd.notna(x) else None
                        )

                        def create_linestring_safe(row):
                            try:
                                if (
                                    pd.notna(row["lat_o"])
                                    and pd.notna(row["lon_o"])
                                    and pd.notna(row["lat_d"])
                                    and pd.notna(row["lon_d"])
                                ):
                                    return LineString(
                                        [
                                            (row["lon_o"], row["lat_o"]),
                                            (row["lon_d"], row["lat_d"]),
                                        ]
                                    )
                            except:
                                pass
                            return None

                        legs_od["geometry"] = legs_od.apply(
                            create_linestring_safe, axis=1
                        )
                        legs_od = legs_od[legs_od["geometry"].notna()]

                        if not legs_od.empty:
                            od_route_col = (
                                "selected_branch"
                                if "selected_branch" in legs_od.columns
                                else "id_linea"
                            )
                            od_direction_col = (
                                "direction_inferred"
                                if "direction_inferred" in legs_od.columns
                                else "direction"
                            )

                            if od_direction_col not in legs_od.columns:
                                st.info(
                                    "No hay dirección predicha para las líneas de deseo"
                                )
                                st.stop()

                            if (
                                routes_h3_od is None
                                or routes_h3_od.empty
                                or "section_id" not in routes_h3_od.columns
                            ):
                                st.info(
                                    "No hay cartografía H3 para asociar los section_id"
                                )
                                st.stop()

                            cartography_route_col = (
                                "id_ramal"
                                if "id_ramal" in routes_h3_od.columns
                                else "id_linea"
                            )
                            if cartography_route_col not in legs_od.columns:
                                legs_od[cartography_route_col] = legs_od[od_route_col]

                            cartography_h3 = routes_h3_od[
                                [
                                    cartography_route_col,
                                    "direction",
                                    "h3",
                                    "section_id",
                                ]
                            ].drop_duplicates()
                            cartography_h3[cartography_route_col] = (
                                cartography_h3[cartography_route_col]
                                .astype(str)
                                .str.strip()
                            )
                            legs_od[cartography_route_col] = (
                                legs_od[cartography_route_col].astype(str).str.strip()
                            )
                            cartography_h3["direction"] = (
                                cartography_h3["direction"].astype(str).str.strip()
                            )
                            legs_od[od_direction_col] = (
                                legs_od[od_direction_col].astype(str).str.strip()
                            )
                            cartography_h3["h3"] = (
                                cartography_h3["h3"].astype(str).str.strip()
                            )
                            legs_od["h3_o"] = legs_od["h3_o"].astype(str).str.strip()
                            legs_od["h3_d"] = legs_od["h3_d"].astype(str).str.strip()

                            cartography_h3 = cartography_h3.rename(
                                columns={
                                    cartography_route_col: od_route_col,
                                    "direction": od_direction_col,
                                }
                            )
                            legs_od[od_route_col] = (
                                legs_od[od_route_col].astype(str).str.strip()
                            )
                            cartography_h3[od_route_col] = (
                                cartography_h3[od_route_col].astype(str).str.strip()
                            )
                            legs_od[od_direction_col] = (
                                legs_od[od_direction_col].astype(str).str.strip()
                            )
                            cartography_h3[od_direction_col] = (
                                cartography_h3[od_direction_col].astype(str).str.strip()
                            )
                            origin_section_map = cartography_h3.rename(
                                columns={
                                    "h3": "h3_o",
                                    "section_id": "section_id_o_h3",
                                }
                            )
                            destination_section_map = cartography_h3.rename(
                                columns={
                                    "h3": "h3_d",
                                    "section_id": "section_id_d_h3",
                                }
                            )
                            legs_od = legs_od.merge(
                                origin_section_map[
                                    [
                                        od_route_col,
                                        od_direction_col,
                                        "h3_o",
                                        "section_id_o_h3",
                                    ]
                                ],
                                on=[od_route_col, od_direction_col, "h3_o"],
                                how="left",
                            ).merge(
                                destination_section_map[
                                    [
                                        od_route_col,
                                        od_direction_col,
                                        "h3_d",
                                        "section_id_d_h3",
                                    ]
                                ],
                                on=[od_route_col, od_direction_col, "h3_d"],
                                how="left",
                            )

                            od_section_cols = [
                                "section_id_o_h3",
                                "section_id_d_h3",
                            ]
                            legs_od = legs_od.dropna(subset=od_section_cols)
                            if legs_od.empty:
                                st.info(
                                    "No se encontraron section_id para los H3 de origen y destino"
                                )
                                st.stop()

                            legs_od["od_pair"] = list(
                                zip(
                                    legs_od["lon_o"],
                                    legs_od["lat_o"],
                                    legs_od["lon_d"],
                                    legs_od["lat_d"],
                                )
                            )
                            legs_od["factor_expansion_linea"] = pd.to_numeric(
                                legs_od["factor_expansion_linea"], errors="coerce"
                            ).fillna(1)
                            legs_count = (
                                legs_od.groupby(
                                    [
                                        od_route_col,
                                        od_direction_col,
                                        *od_section_cols,
                                        "od_pair",
                                    ]
                                )
                                .agg(
                                    n_legs_expanded=(
                                        "factor_expansion_linea",
                                        "sum",
                                    ),
                                )
                                .reset_index()
                            )
                            legs_count[["lon_o", "lat_o", "lon_d", "lat_d"]] = (
                                pd.DataFrame(
                                    legs_count["od_pair"].tolist(),
                                    columns=["lon_o", "lat_o", "lon_d", "lat_d"],
                                )
                            )
                            legs_count["geometry"] = legs_count.apply(
                                lambda row: LineString(
                                    [
                                        (row["lon_o"], row["lat_o"]),
                                        (row["lon_d"], row["lat_d"]),
                                    ]
                                ),
                                axis=1,
                            )
                            legs_count = legs_count.rename(
                                columns={
                                    "section_id_o_h3": "section_id_o",
                                    "section_id_d_h3": "section_id_d",
                                }
                            )
                            legs_count["wkt"] = legs_count["geometry"].map(
                                lambda geometry: geometry.wkt
                            )

                            legs_gdf = gpd.GeoDataFrame(
                                legs_count, geometry="geometry", crs="EPSG:4326"
                            )
                            legs_gdf = legs_gdf.sort_values(
                                "n_legs_expanded", ascending=False
                            ).reset_index(drop=True)
                            od_matrix = legs_od.pivot_table(
                                values="factor_expansion_linea",
                                index="section_id_o_h3",
                                columns="section_id_d_h3",
                                aggfunc="sum",
                                fill_value=0,
                            )

                            if st.button(
                                (
                                    "Ocultar datos de líneas"
                                    if st.session_state.get("show_od_lines_data", False)
                                    else "Mostrar datos de líneas"
                                ),
                                key="toggle_od_lines_data",
                            ):
                                st.session_state["show_od_lines_data"] = (
                                    not st.session_state.get(
                                        "show_od_lines_data", False
                                    )
                                )

                            if st.session_state.get("show_od_lines_data", False):
                                legs_gdf_display = legs_gdf[
                                    [
                                        "section_id_o",
                                        "section_id_d",
                                        "n_legs_expanded",
                                        "wkt",
                                    ]
                                ].copy()
                                legs_gdf_display["n_legs_expanded"] = (
                                    legs_gdf_display["n_legs_expanded"]
                                    .round()
                                    .astype("Int64")
                                )
                                st.dataframe(
                                    legs_gdf_display,
                                    use_container_width=True,
                                )

                            st.markdown(
                                "**Líneas de deseo (origen → destino) sobre celdas H3 y rutas oficiales**"
                            )

                            fig_od = Figure(width=1000, height=700)
                            center_lats = [g.centroid.y for g in legs_gdf["geometry"]]
                            center_lons = [g.centroid.x for g in legs_gdf["geometry"]]

                            m_od = folium.Map(
                                location=[
                                    sum(center_lats) / len(center_lats),
                                    sum(center_lons) / len(center_lons),
                                ],
                                zoom_start=12,
                                tiles="cartodbpositron",
                            )

                            cmaps_od = [
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
                            ramal_cmap_map_od = {}

                            # Agregar celdas H3 como base
                            if routes_h3_od is not None and not routes_h3_od.empty:
                                if "id_ramal" in routes_h3_od.columns:
                                    ramales_od = sorted(
                                        routes_h3_od["id_ramal"].unique()
                                    )
                                    directions_od = sorted(
                                        routes_h3_od["direction"].unique()
                                    )

                                    ramal_cmap_map_od = {
                                        str(ramal): cmaps_od[idx % len(cmaps_od)]
                                        for idx, ramal in enumerate(ramales_od)
                                    }
                                    route_idx = 0
                                    for ramal_od in ramales_od:
                                        cmap_name_od = ramal_cmap_map_od[str(ramal_od)]

                                        ramal_data_od = routes_h3_od[
                                            routes_h3_od["id_ramal"] == ramal_od
                                        ]

                                        for direction_od in directions_od:
                                            direction_data_od = ramal_data_od[
                                                ramal_data_od["direction"]
                                                == direction_od
                                            ]

                                            if direction_data_od.empty:
                                                continue

                                            geometries_od = []
                                            for h3_cell in direction_data_od["h3"]:
                                                try:
                                                    poly = h3_to_shapely_polygon(
                                                        h3_cell
                                                    )
                                                    geometries_od.append(poly)
                                                except Exception:
                                                    geometries_od.append(None)

                                            gdf_od = gpd.GeoDataFrame(
                                                direction_data_od.reset_index(
                                                    drop=True
                                                ),
                                                geometry=geometries_od,
                                                crs="EPSG:4326",
                                            )
                                            gdf_od = gdf_od[gdf_od.geometry.notna()]

                                            if not gdf_od.empty:
                                                layer_name_od = (
                                                    f"Ramal {ramal_od} - "
                                                    f"Sentido {direction_od}"
                                                )
                                                route_layer_name_od = (
                                                    f"Ruta: {layer_name_od}"
                                                )
                                                route_group_od = folium.FeatureGroup(
                                                    name=route_layer_name_od,
                                                    show=(route_idx == 0),
                                                )

                                                # Celdas H3 del recorrido, dentro de la misma capa de la ruta.
                                                gdf_od.explore(
                                                    m=route_group_od,
                                                    color="gray",
                                                    tooltip=[
                                                        "id_ramal",
                                                        "direction",
                                                    ],
                                                    popup=False,
                                                    style_kwds={
                                                        "fillOpacity": 0.4,
                                                        "weight": 1,
                                                    },
                                                    legend=False,
                                                )

                                                # Rutas oficiales con flechas
                                                route_line_geom_od = None
                                                if (
                                                    route_geoms_od is not None
                                                    and not route_geoms_od.empty
                                                ):
                                                    try:
                                                        route_geoms_df_copy_od = (
                                                            route_geoms_od.copy()
                                                        )
                                                        route_geoms_df_copy_od[
                                                            "geometry"
                                                        ] = route_geoms_df_copy_od[
                                                            "wkt"
                                                        ].apply(
                                                            lambda x: wkt.loads(x)
                                                        )
                                                        route_gdf_od = gpd.GeoDataFrame(
                                                            route_geoms_df_copy_od,
                                                            geometry="geometry",
                                                            crs="EPSG:4326",
                                                        )

                                                        route_id_col_od = (
                                                            "id"
                                                            if "id"
                                                            in route_gdf_od.columns
                                                            else "id_ramal"
                                                        )

                                                        route_gdf_od[
                                                            route_id_col_od
                                                        ] = pd.to_numeric(
                                                            route_gdf_od[
                                                                route_id_col_od
                                                            ],
                                                            errors="coerce",
                                                        ).astype(
                                                            "Int64"
                                                        )
                                                        route_gdf_od["direction"] = (
                                                            pd.to_numeric(
                                                                route_gdf_od[
                                                                    "direction"
                                                                ],
                                                                errors="coerce",
                                                            ).astype("Int64")
                                                        )
                                                        ramal_conv_od = (
                                                            int(ramal_od)
                                                            if pd.notna(
                                                                pd.to_numeric(
                                                                    ramal_od,
                                                                    errors="coerce",
                                                                )
                                                            )
                                                            else ramal_od
                                                        )
                                                        direction_conv_od = (
                                                            int(direction_od)
                                                            if pd.notna(
                                                                pd.to_numeric(
                                                                    direction_od,
                                                                    errors="coerce",
                                                                )
                                                            )
                                                            else direction_od
                                                        )

                                                        route_gdf_subset_od = (
                                                            route_gdf_od[
                                                                (
                                                                    route_gdf_od[
                                                                        route_id_col_od
                                                                    ]
                                                                    == ramal_conv_od
                                                                )
                                                                & (
                                                                    route_gdf_od[
                                                                        "direction"
                                                                    ]
                                                                    == direction_conv_od
                                                                )
                                                            ]
                                                        )

                                                        if (
                                                            not route_gdf_subset_od.empty
                                                        ):
                                                            route_line_geom_od = route_gdf_subset_od.iloc[
                                                                0
                                                            ][
                                                                "geometry"
                                                            ]
                                                            if hasattr(
                                                                route_line_geom_od,
                                                                "simplify",
                                                            ):
                                                                route_line_geom_od = route_line_geom_od.simplify(
                                                                    0.00005,
                                                                    preserve_topology=True,
                                                                )
                                                    except Exception:
                                                        route_line_geom_od = None

                                                if route_line_geom_od is not None:
                                                    try:
                                                        coords_od = list(
                                                            route_line_geom_od.coords
                                                        )
                                                        if len(coords_od) >= 2:
                                                            line_color_od = "#1f1f1f"
                                                            folium.GeoJson(
                                                                route_line_geom_od.__geo_interface__,
                                                                style_function=lambda feature, color=line_color_od: {
                                                                    "color": color,
                                                                    "weight": 3,
                                                                    "opacity": 0.9,
                                                                },
                                                                tooltip=route_layer_name_od,
                                                                popup=route_layer_name_od,
                                                            ).add_to(route_group_od)

                                                            end_coord_od = coords_od[-1]
                                                            end_lng_od, end_lat_od = (
                                                                end_coord_od[0],
                                                                end_coord_od[1],
                                                            )

                                                            start_coord_od = (
                                                                coords_od[-2]
                                                                if len(coords_od) >= 2
                                                                else coords_od[0]
                                                            )
                                                            (
                                                                start_lng_od,
                                                                start_lat_od,
                                                            ) = (
                                                                start_coord_od[0],
                                                                start_coord_od[1],
                                                            )

                                                            dx_od = (
                                                                end_lng_od
                                                                - start_lng_od
                                                            )
                                                            dy_od = (
                                                                end_lat_od
                                                                - start_lat_od
                                                            )
                                                            heading_od = math.degrees(
                                                                math.atan2(dy_od, dx_od)
                                                            )

                                                            folium.RegularPolygonMarker(
                                                                location=[
                                                                    end_lat_od,
                                                                    end_lng_od,
                                                                ],
                                                                number_of_sides=3,
                                                                radius=7,
                                                                rotation=heading_od
                                                                + 90,
                                                                color=line_color_od,
                                                                fill_color=line_color_od,
                                                                fill=True,
                                                                fill_opacity=0.95,
                                                                weight=1,
                                                                popup=route_layer_name_od,
                                                            ).add_to(route_group_od)
                                                    except Exception:
                                                        pass

                                                route_group_od.add_to(m_od)
                                                route_idx += 1

                            # Agregar una capa conmutable de líneas OD por ramal y sentido.
                            vmin_legs = legs_gdf["n_legs_expanded"].min()
                            vmax_legs = legs_gdf["n_legs_expanded"].max()

                            for od_layer_idx, (
                                (ramal_od, direction_od),
                                od_subset,
                            ) in enumerate(
                                legs_gdf.groupby(
                                    [od_route_col, od_direction_col], sort=True
                                )
                            ):
                                od_group = folium.FeatureGroup(
                                    name=(
                                        "Líneas OD: "
                                        f"Ramal {ramal_od} - "
                                        f"Sentido {direction_od}"
                                    ),
                                    show=(od_layer_idx == 0),
                                )
                                cmap_od = colormaps[
                                    ramal_cmap_map_od.get(
                                        str(ramal_od),
                                        cmaps_od[od_layer_idx % len(cmaps_od)],
                                    )
                                ]

                                for _, row in od_subset.sort_values(
                                    "n_legs_expanded", ascending=True
                                ).iterrows():
                                    intensity = (row["n_legs_expanded"] - vmin_legs) / (
                                        vmax_legs - vmin_legs + 1
                                    )
                                    color = to_hex(cmap_od(0.3 + 0.7 * intensity))
                                    coords = [
                                        (coord[1], coord[0])
                                        for coord in row["geometry"].coords
                                    ]
                                    folium.PolyLine(
                                        coords,
                                        weight=1 + intensity * 7,
                                        color=color,
                                        opacity=0.2 + 0.7 * intensity,
                                        popup=(
                                            f"{row['n_legs_expanded']:.0f} "
                                            "etapas expandidas"
                                        ),
                                    ).add_to(od_group)

                                od_group.add_to(m_od)

                            folium.LayerControl().add_to(m_od)
                            fig_od.add_child(m_od)
                            st_folium(
                                fig_od, width=1000, height=700, key="od_lines_map"
                            )
                        else:
                            st.info("No hay líneas de deseo válidas para mostrar")
                    else:
                        st.info("No hay datos de origen-destino disponibles")
                else:
                    st.info("Los datos de origen-destino (H3) no están disponibles")

            with st.expander("Matriz OD por línea", expanded=False):
                if od_matrix is not None and not od_matrix.empty:
                    od_matrix_display = od_matrix.round().astype("int64")
                    od_heatmap = px.imshow(
                        od_matrix_display,
                        text_auto=True,
                        labels={
                            "x": "Sección destino",
                            "y": "Sección origen",
                            "color": "Etapas expandidas",
                        },
                        color_continuous_scale="Blues",
                    )
                    od_heatmap.update_coloraxes(showscale=True)
                    matrix_size = max(od_matrix.shape)
                    od_heatmap.update_layout(
                        height=max(800, min(1600, matrix_size * 28)),
                    )
                    st.plotly_chart(od_heatmap, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para construir la matriz OD")

        else:
            st.info(
                "No hay datos suficientes para mostrar el mapa de uso de secciones."
            )
else:
    st.info("Por favor, seleccione una línea para visualizar sus geometrías H3")
