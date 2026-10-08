import streamlit as st
import folium
import h3
import pandas as pd
import geopandas as gpd
from shapely import wkt
from streamlit_folium import st_folium
from folium import Figure
from dash_utils import (
    get_logo,
    configurar_selector_corrida,
)
from h3_utils import create_routes_h3_map, h3_to_shapely_polygon, seleccionar_linea
from urbantrips.utils import utils
from urbantrips.dashboard import dashboard_ctx
from urbantrips.carto.routes import process_parent_h3_parallel
from urbantrips.storage.access import DatabaseBusyError, write_access
from urbantrips.viz import basemaps

ROUTE_H3_FILL_COLOR = "#3f3f46"
ROUTE_H3_LINE_COLOR = "#18181b"
BUFFER_H3_FILL_COLOR = "#d4d4d8"
BUFFER_H3_LINE_COLOR = "#a1a1aa"


def gps_h3_to_resolution(h3_cell, target_resolution):
    if pd.isna(h3_cell):
        return None
    try:
        cell_resolution = h3.get_resolution(h3_cell)
        if cell_resolution == target_resolution:
            return h3_cell
        if cell_resolution > target_resolution:
            return h3.cell_to_parent(h3_cell, target_resolution)
        return None
    except Exception:
        return None


def build_h3_buffer_cells(route_h3_cells, buffer_size):
    route_h3_cells = [cell for cell in route_h3_cells if pd.notna(cell)]
    if buffer_size == 0:
        return set(route_h3_cells)

    buffer_cells = set()
    for h3_cell in route_h3_cells:
        try:
            buffer_cells.update(h3.grid_disk(h3_cell, buffer_size))
        except Exception:
            pass
    return buffer_cells


def create_gps_outside_buffer_map(gps_outside_by_h3, route_h3_cells, buffer_cells):
    if gps_outside_by_h3.empty:
        return None

    geometries = []
    for h3_cell in gps_outside_by_h3["h3_gps_parent"]:
        geometries.append(h3_to_shapely_polygon(h3_cell))

    gps_gdf = gpd.GeoDataFrame(
        gps_outside_by_h3.copy(), geometry=geometries, crs="EPSG:4326"
    )

    center_lat = gps_gdf.geometry.centroid.y.mean()
    center_lng = gps_gdf.geometry.centroid.x.mean()

    fig = Figure(width=1000, height=700)
    m = basemaps.folium_map(
        location=[center_lat, center_lng],
        zoom_start=12,
    )

    route_display_cells = set(route_h3_cells)
    if route_display_cells:
        route_gdf = gpd.GeoDataFrame(
            {"tipo": ["ramal" for _ in route_display_cells]},
            geometry=[h3_to_shapely_polygon(cell) for cell in route_display_cells],
            crs="EPSG:4326",
        )
        folium.GeoJson(
            route_gdf,
            name="Celdas del ramal",
            style_function=lambda feature: {
                "fillColor": ROUTE_H3_FILL_COLOR,
                "color": ROUTE_H3_LINE_COLOR,
                "weight": 1,
                "fillOpacity": 0.15,
            },
        ).add_to(m)

    buffer_only_cells = set(buffer_cells) - route_display_cells
    if buffer_only_cells:
        buffer_gdf = gpd.GeoDataFrame(
            {"tipo": ["buffer" for _ in buffer_only_cells]},
            geometry=[h3_to_shapely_polygon(cell) for cell in buffer_only_cells],
            crs="EPSG:4326",
        )
        folium.GeoJson(
            buffer_gdf,
            name="Buffer H3",
            style_function=lambda feature: {
                "fillColor": BUFFER_H3_FILL_COLOR,
                "color": BUFFER_H3_LINE_COLOR,
                "weight": 1,
                "fillOpacity": 0.08,
            },
        ).add_to(m)

    gps_gdf.explore(
        m=m,
        column="puntos_fuera_buffer",
        cmap="Reds",
        name="GPS fuera del buffer",
        tooltip=["h3_gps_parent", "puntos_fuera_buffer"],
        popup=["h3_gps_parent", "puntos_fuera_buffer"],
        style_kwds={
            "fillOpacity": 1,
            "opacity": 1,
            "weight": 1,
            "color": "gray",
        },
        legend=True,
    )

    folium.LayerControl().add_to(m)
    fig.add_child(m)
    return fig


def create_service_endpoints_map(
    service_outside_by_h3,
    route_h3_cells,
    endpoint_buffer_cells,
    endpoint_cells,
):
    display_cells = (
        set(route_h3_cells)
        | set(endpoint_buffer_cells)
        | set(service_outside_by_h3.get("h3_gps_parent", []))
    )
    if not display_cells:
        return None

    center_lats = []
    center_lngs = []
    for h3_cell in display_cells:
        try:
            lat, lng = h3.cell_to_latlng(h3_cell)
            center_lats.append(lat)
            center_lngs.append(lng)
        except Exception:
            pass

    if not center_lats:
        return None

    fig = Figure(width=1000, height=700)
    m = basemaps.folium_map(
        location=[
            sum(center_lats) / len(center_lats),
            sum(center_lngs) / len(center_lngs),
        ],
        zoom_start=12,
    )

    if route_h3_cells:
        route_gdf = gpd.GeoDataFrame(
            {"tipo": ["ramal" for _ in route_h3_cells]},
            geometry=[h3_to_shapely_polygon(cell) for cell in route_h3_cells],
            crs="EPSG:4326",
        )
        folium.GeoJson(
            route_gdf,
            name="Recorrido H3",
            style_function=lambda feature: {
                "fillColor": ROUTE_H3_FILL_COLOR,
                "color": ROUTE_H3_LINE_COLOR,
                "weight": 1,
                "fillOpacity": 0.12,
            },
        ).add_to(m)

    endpoint_layers = [
        (
            "Buffer extremos del recorrido",
            endpoint_buffer_cells,
            BUFFER_H3_FILL_COLOR,
            BUFFER_H3_LINE_COLOR,
            0.12,
        ),
        (
            "Extremos del recorrido",
            endpoint_cells,
            ROUTE_H3_FILL_COLOR,
            ROUTE_H3_LINE_COLOR,
            0.35,
        ),
    ]
    for layer_name, cells, fill_color, line_color, fill_opacity in endpoint_layers:
        cells = set(cells)
        if not cells:
            continue
        endpoint_gdf = gpd.GeoDataFrame(
            {"tipo": [layer_name for _ in cells]},
            geometry=[h3_to_shapely_polygon(cell) for cell in cells],
            crs="EPSG:4326",
        )
        folium.GeoJson(
            endpoint_gdf,
            name=layer_name,
            style_function=lambda feature, fc=fill_color, lc=line_color, fo=fill_opacity: {
                "fillColor": fc,
                "color": lc,
                "weight": 1,
                "fillOpacity": fo,
            },
        ).add_to(m)

    service_layers = [
        ("start_service", "GPS start_service fuera de k-ring", "Reds"),
        ("finish_service", "GPS finish_service fuera de k-ring", "Purples"),
    ]
    for service_type, layer_name, cmap_name in service_layers:
        service_data = service_outside_by_h3[
            service_outside_by_h3["service_type"] == service_type
        ].copy()
        if service_data.empty:
            continue
        service_gdf = gpd.GeoDataFrame(
            service_data,
            geometry=[
                h3_to_shapely_polygon(cell) for cell in service_data["h3_gps_parent"]
            ],
            crs="EPSG:4326",
        )
        service_gdf.explore(
            m=m,
            column="puntos_fuera_kring",
            cmap=cmap_name,
            name=layer_name,
            tooltip=["service_type", "h3_gps_parent", "puntos_fuera_kring"],
            popup=["service_type", "h3_gps_parent", "puntos_fuera_kring"],
            style_kwds={
                "fillOpacity": 1,
                "opacity": 1,
                "weight": 1,
                "color": "gray",
            },
            legend=False,
        )

    folium.LayerControl().add_to(m)
    fig.add_child(m)
    return fig


st.set_page_config(layout="wide")

# Header
logo = get_logo()
st.image(logo)
st.title("GPS")

alias_seleccionado = configurar_selector_corrida()

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
    st.subheader("Filtros GPS")

    try:
        with dashboard_ctx() as ctx:
            dias_gps_df = ctx.data.query(f"""
                SELECT DISTINCT dia
                FROM gps
                WHERE id_linea = {int(id_linea)}
                ORDER BY dia
                """)

            ramales_df = pd.DataFrame()
            if has_branches:
                ramales_df = ctx.insumos.query(f"""
                    SELECT id_ramal, nombre_ramal
                    FROM metadata_ramales
                    WHERE modo = 'autobus' AND id_linea = {int(id_linea)}
                    ORDER BY id_ramal
                    """)
    except Exception as e:
        st.error(f"No se pudieron cargar los filtros de GPS: {e}")
        st.stop()

    dias_gps = dias_gps_df["dia"].dropna().astype(str).tolist()
    if not dias_gps:
        st.warning(f"No hay datos GPS disponibles para la línea {id_linea}.")
        st.stop()

    gps_filter_cols = st.columns(3)
    with gps_filter_cols[0]:
        dia_gps = st.selectbox(
            "Día",
            options=dias_gps,
            key="gps_dia",
        )
    with gps_filter_cols[1]:
        gps_hora_inicio = st.number_input(
            "Hora inicio (0-23)",
            min_value=0,
            max_value=23,
            value=9,
            step=1,
            key="gps_hora_inicio",
        )
    with gps_filter_cols[2]:
        gps_hora_fin = st.number_input(
            "Hora fin (0-23)",
            min_value=0,
            max_value=23,
            value=9,
            step=1,
            key="gps_hora_fin",
        )
    selected_ramal = None
    if has_branches:
        if ramales_df.empty:
            st.warning(f"No se encontraron ramales para la línea {id_linea}.")
            st.stop()

        ramales_options = ramales_df.copy()
        ramales_options["label"] = ramales_options.apply(
            lambda row: f"{row['id_ramal']} - {row['nombre_ramal']}", axis=1
        )
        selected_ramal_label = st.selectbox(
            "Ramal",
            options=ramales_options["label"].tolist(),
            key="gps_ramal",
        )
        selected_ramal = ramales_options.loc[
            ramales_options["label"] == selected_ramal_label, "id_ramal"
        ].iloc[0]

    gps_outside_expander = st.expander("GPS fuera del buffer H3")
    gps_buffer_h3 = gps_outside_expander.number_input(
        "Buffer H3 (k-ring)",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
        key="gps_buffer_h3",
    )

    gps_action_cols = gps_outside_expander.columns([3, 1])
    with gps_action_cols[0]:
        compute_gps_button = st.button(
            "Calcular GPS fuera del buffer",
            type="primary",
            key="compute_gps_outside_buffer",
        )
    with gps_action_cols[1]:
        clear_gps_button = st.button(
            "Limpiar resultados",
            key="clear_gps_outside_buffer",
        )

    if clear_gps_button:
        st.session_state.pop("gps_outside_buffer_results", None)

    current_gps_key = (
        int(id_linea),
        int(h3_res),
        str(dia_gps),
        int(gps_hora_inicio),
        int(gps_hora_fin),
        int(gps_buffer_h3),
        int(selected_ramal) if selected_ramal is not None else None,
    )

    if compute_gps_button:
        if gps_hora_inicio > gps_hora_fin:
            st.error("La hora de inicio debe ser menor o igual que la hora de fin")
        else:
            with st.spinner("Calculando puntos GPS fuera del buffer H3..."):
                try:
                    route_filter_gps = f"id_linea = {int(id_linea)}"
                    if has_branches:
                        route_filter_h3 = f"{id_col} = {int(selected_ramal)}"
                        route_filter_gps += f" AND id_ramal = {int(selected_ramal)}"
                    else:
                        route_filter_h3 = f"id_linea = {int(id_linea)}"

                    if h3_res == 10:
                        table_name = f"official_{source_table_h3}_geoms_h3"
                        where_clause = route_filter_h3
                    else:
                        table_name = f"official_{source_table_h3}_geoms_h3_parent"
                        where_clause = (
                            f"resolution = {int(h3_res)} AND {route_filter_h3}"
                        )

                    dia_gps_sql = str(dia_gps).replace("'", "''")

                    with dashboard_ctx() as ctx:
                        routes_h3_query = f"""
                        SELECT {id_col}, direction, section_id, h3
                        FROM {table_name}
                        WHERE {where_clause}
                        ORDER BY direction, section_id
                        """
                        routes_h3_gps = ctx.insumos.query(routes_h3_query)

                        gps_query = f"""
                        SELECT id, dia, id_linea, id_ramal, interno, fecha, h3
                        FROM gps
                        WHERE dia = '{dia_gps_sql}'
                          AND {route_filter_gps}
                          AND h3 IS NOT NULL
                        """
                        gps_points = ctx.data.query(gps_query)

                    if routes_h3_gps.empty:
                        st.warning(
                            "No se encontraron celdas H3 de ruta para los filtros "
                            "seleccionados. "
                            "Cargá o generá primero las geometrías H3 de esa resolución."
                        )
                        st.stop()

                    if gps_points.empty:
                        st.warning(
                            "No se encontraron puntos GPS para los filtros seleccionados."
                        )
                        st.stop()

                    gps_points["hora"] = pd.to_datetime(
                        gps_points["fecha"], unit="s", errors="coerce"
                    ).dt.hour
                    gps_points = gps_points[
                        (gps_points["hora"] >= gps_hora_inicio)
                        & (gps_points["hora"] <= gps_hora_fin)
                    ].copy()

                    if gps_points.empty:
                        st.warning(
                            "No se encontraron puntos GPS en el rango horario seleccionado."
                        )
                        st.stop()

                    gps_points["h3_gps_parent"] = gps_points["h3"].apply(
                        lambda cell: gps_h3_to_resolution(cell, h3_res)
                    )
                    gps_with_parent = gps_points.dropna(subset=["h3_gps_parent"]).copy()

                    if gps_with_parent.empty:
                        st.warning(
                            "No se pudo convertir el H3 de GPS a la resolución "
                            "seleccionada. "
                            "Usá una resolución menor o igual a la resolución H3 de GPS."
                        )
                        st.stop()

                    route_h3_cells = routes_h3_gps["h3"].dropna().astype(str).tolist()
                    buffer_cells = build_h3_buffer_cells(
                        route_h3_cells, int(gps_buffer_h3)
                    )
                    gps_with_parent["fuera_buffer"] = ~gps_with_parent[
                        "h3_gps_parent"
                    ].isin(buffer_cells)
                    gps_outside = gps_with_parent[
                        gps_with_parent["fuera_buffer"]
                    ].copy()

                    total_points = len(gps_points)
                    converted_points = len(gps_with_parent)
                    outside_points = len(gps_outside)
                    inside_points = converted_points - outside_points

                    dropped_points = total_points - converted_points

                    gps_outside_by_h3 = pd.DataFrame(
                        columns=["h3_gps_parent", "puntos_fuera_buffer"]
                    )
                    if not gps_outside.empty:
                        gps_outside_by_h3 = (
                            gps_outside.groupby("h3_gps_parent", as_index=False)
                            .size()
                            .rename(columns={"size": "puntos_fuera_buffer"})
                            .sort_values("puntos_fuera_buffer", ascending=False)
                        )

                    st.session_state["gps_outside_buffer_results"] = {
                        "key": current_gps_key,
                        "total_points": total_points,
                        "converted_points": converted_points,
                        "inside_points": inside_points,
                        "outside_points": outside_points,
                        "dropped_points": dropped_points,
                        "gps_outside_by_h3": gps_outside_by_h3,
                        "route_h3_cells": route_h3_cells,
                        "buffer_cells": buffer_cells,
                    }
                except DatabaseBusyError as e:
                    st.error(
                        f"{e} Cerrá el otro dashboard o esperá a que "
                        "termine la corrida y volvé a intentar."
                    )
                except Exception as e:
                    st.error(f"Error al calcular GPS fuera del buffer H3: {e}")
                    st.exception(e)

    with gps_outside_expander:
        gps_results = st.session_state.get("gps_outside_buffer_results")
        if gps_results is not None:
            if gps_results.get("key") == current_gps_key:
                metric_cols = st.columns(4)
                metric_cols[0].metric("Puntos GPS", gps_results["total_points"])
                metric_cols[1].metric(
                    "Puntos comparados", gps_results["converted_points"]
                )
                metric_cols[2].metric("Dentro del buffer", gps_results["inside_points"])
                metric_cols[3].metric("Fuera del buffer", gps_results["outside_points"])

                if gps_results["dropped_points"] > 0:
                    st.info(
                        f"Se excluyeron {gps_results['dropped_points']} puntos GPS "
                        "porque su H3 no pudo llevarse a la resolución seleccionada."
                    )

                gps_outside_by_h3 = gps_results["gps_outside_by_h3"]
                if gps_outside_by_h3.empty:
                    st.success(
                        "No hay puntos GPS por fuera del ramal y del buffer H3 "
                        "seleccionado."
                    )
                else:
                    gps_map = create_gps_outside_buffer_map(
                        gps_outside_by_h3,
                        gps_results["route_h3_cells"],
                        gps_results["buffer_cells"],
                    )
                    if gps_map is not None:
                        st_folium(
                            gps_map,
                            width=1000,
                            height=700,
                            key="gps_outside_buffer_map",
                        )
                    if st.button(
                        (
                            "Ocultar tabla de resultados"
                            if st.session_state.get(
                                "show_gps_outside_buffer_data", False
                            )
                            else "Mostrar tabla de resultados"
                        ),
                        key="toggle_gps_outside_buffer_data",
                    ):
                        st.session_state["show_gps_outside_buffer_data"] = (
                            not st.session_state.get(
                                "show_gps_outside_buffer_data", False
                            )
                        )

                    if st.session_state.get("show_gps_outside_buffer_data", False):
                        st.dataframe(gps_outside_by_h3, hide_index=True)
            else:
                st.info(
                    "Los filtros cambiaron. Volvé a calcular para actualizar "
                    "el mapa GPS."
                )

    with st.expander("GPS de inicio y fin de servicio fuera de k-ring"):
        service_buffer_h3 = st.number_input(
            "Buffer H3 para inicio y fin de servicio (k-ring)",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            key="gps_service_buffer_h3",
        )

        service_action_cols = st.columns([3, 1])
        with service_action_cols[0]:
            compute_service_button = st.button(
                "Calcular start/finish fuera de k-ring",
                type="primary",
                key="compute_gps_service_endpoints",
            )
        with service_action_cols[1]:
            clear_service_button = st.button(
                "Limpiar resultados",
                key="clear_gps_service_endpoints",
            )

        if clear_service_button:
            st.session_state.pop("gps_service_endpoint_results", None)

        current_service_key = (
            int(id_linea),
            int(h3_res),
            str(dia_gps),
            int(gps_hora_inicio),
            int(gps_hora_fin),
            int(service_buffer_h3),
            int(selected_ramal) if selected_ramal is not None else None,
        )

        if compute_service_button:
            if gps_hora_inicio > gps_hora_fin:
                st.error("La hora de inicio debe ser menor o igual que la hora de fin")
            else:
                with st.spinner(
                    "Calculando start_service y finish_service fuera de k-ring..."
                ):
                    try:
                        route_filter_gps = f"id_linea = {int(id_linea)}"
                        if has_branches:
                            route_filter_h3 = f"{id_col} = {int(selected_ramal)}"
                            route_filter_gps += f" AND id_ramal = {int(selected_ramal)}"
                        else:
                            route_filter_h3 = f"id_linea = {int(id_linea)}"

                        if h3_res == 10:
                            table_name = f"official_{source_table_h3}_geoms_h3"
                            where_clause = route_filter_h3
                        else:
                            table_name = f"official_{source_table_h3}_geoms_h3_parent"
                            where_clause = (
                                f"resolution = {int(h3_res)} AND {route_filter_h3}"
                            )

                        dia_gps_sql = str(dia_gps).replace("'", "''")

                        with dashboard_ctx() as ctx:
                            routes_h3_query = f"""
                            SELECT {id_col}, direction, section_id, h3
                            FROM {table_name}
                            WHERE {where_clause}
                            ORDER BY direction, section_id
                            """
                            routes_h3_service = ctx.insumos.query(routes_h3_query)

                            gps_query = f"""
                            SELECT id, dia, id_linea, id_ramal, interno, fecha,
                                   h3, service_type
                            FROM gps
                            WHERE dia = '{dia_gps_sql}'
                              AND {route_filter_gps}
                              AND h3 IS NOT NULL
                              AND service_type IN ('start_service', 'finish_service')
                            """
                            service_points = ctx.data.query(gps_query)

                        if routes_h3_service.empty:
                            st.warning(
                                "No se encontraron celdas H3 de ruta para los "
                                "filtros seleccionados. Cargá o generá primero "
                                "las geometrías H3 de esa resolución."
                            )
                            st.stop()

                        if service_points.empty:
                            st.warning(
                                "No se encontraron registros start_service o "
                                "finish_service para los filtros seleccionados."
                            )
                            st.stop()

                        service_points["hora"] = pd.to_datetime(
                            service_points["fecha"], unit="s", errors="coerce"
                        ).dt.hour
                        service_points = service_points[
                            (service_points["hora"] >= gps_hora_inicio)
                            & (service_points["hora"] <= gps_hora_fin)
                        ].copy()

                        if service_points.empty:
                            st.warning(
                                "No se encontraron start_service o finish_service "
                                "en el rango horario seleccionado."
                            )
                            st.stop()

                        group_cols = [id_col, "direction"]
                        endpoint_sections = (
                            routes_h3_service.groupby(group_cols, as_index=False)
                            .agg(
                                section_id_inicio=("section_id", "min"),
                                section_id_fin=("section_id", "max"),
                            )
                            .sort_values(group_cols)
                        )

                        routes_with_endpoints = routes_h3_service.merge(
                            endpoint_sections, on=group_cols, how="left"
                        )
                        start_section_cells = (
                            routes_with_endpoints.loc[
                                routes_with_endpoints["section_id"]
                                == routes_with_endpoints["section_id_inicio"],
                                "h3",
                            ]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )
                        finish_section_cells = (
                            routes_with_endpoints.loc[
                                routes_with_endpoints["section_id"]
                                == routes_with_endpoints["section_id_fin"],
                                "h3",
                            ]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )

                        endpoint_cells = sorted(
                            set(start_section_cells) | set(finish_section_cells)
                        )
                        endpoint_buffer_cells = build_h3_buffer_cells(
                            endpoint_cells, int(service_buffer_h3)
                        )
                        route_h3_cells = (
                            routes_h3_service["h3"].dropna().astype(str).tolist()
                        )

                        service_points["h3_gps_parent"] = service_points["h3"].apply(
                            lambda cell: gps_h3_to_resolution(cell, h3_res)
                        )
                        service_with_parent = service_points.dropna(
                            subset=["h3_gps_parent"]
                        ).copy()

                        if service_with_parent.empty:
                            st.warning(
                                "No se pudo convertir el H3 de GPS a la resolución "
                                "seleccionada. Usá una resolución menor o igual a "
                                "la resolución H3 de GPS."
                            )
                            st.stop()

                        start_points = service_with_parent[
                            service_with_parent["service_type"] == "start_service"
                        ].copy()
                        finish_points = service_with_parent[
                            service_with_parent["service_type"] == "finish_service"
                        ].copy()

                        start_outside = start_points[
                            ~start_points["h3_gps_parent"].isin(endpoint_buffer_cells)
                        ].copy()
                        finish_outside = finish_points[
                            ~finish_points["h3_gps_parent"].isin(endpoint_buffer_cells)
                        ].copy()
                        service_outside = pd.concat(
                            [start_outside, finish_outside], ignore_index=True
                        )

                        service_outside_by_h3 = pd.DataFrame(
                            columns=[
                                "service_type",
                                "h3_gps_parent",
                                "puntos_fuera_kring",
                            ]
                        )
                        if not service_outside.empty:
                            service_outside_by_h3 = (
                                service_outside.groupby(
                                    ["service_type", "h3_gps_parent"],
                                    as_index=False,
                                )
                                .size()
                                .rename(columns={"size": "puntos_fuera_kring"})
                                .sort_values(
                                    ["service_type", "puntos_fuera_kring"],
                                    ascending=[True, False],
                                )
                            )

                        st.session_state["gps_service_endpoint_results"] = {
                            "key": current_service_key,
                            "endpoint_sections": endpoint_sections,
                            "route_h3_cells": route_h3_cells,
                            "endpoint_buffer_cells": endpoint_buffer_cells,
                            "endpoint_cells": endpoint_cells,
                            "service_outside_by_h3": service_outside_by_h3,
                            "start_total": len(start_points),
                            "finish_total": len(finish_points),
                            "start_outside": len(start_outside),
                            "finish_outside": len(finish_outside),
                            "dropped_points": len(service_points)
                            - len(service_with_parent),
                        }
                    except DatabaseBusyError as e:
                        st.error(
                            f"{e} Cerrá el otro dashboard o esperá a que "
                            "termine la corrida y volvé a intentar."
                        )
                    except Exception as e:
                        st.error(
                            "Error al calcular GPS de inicio y fin de servicio: " f"{e}"
                        )
                        st.exception(e)

        service_results = st.session_state.get("gps_service_endpoint_results")
        if service_results is not None:
            service_result_keys = {
                "endpoint_buffer_cells",
                "endpoint_cells",
                "service_outside_by_h3",
                "route_h3_cells",
            }
            has_required_keys = service_result_keys.issubset(service_results)
            if service_results.get("key") == current_service_key and has_required_keys:
                service_metric_cols = st.columns(4)
                service_metric_cols[0].metric(
                    "Start service", service_results["start_total"]
                )
                service_metric_cols[1].metric(
                    "Start fuera", service_results["start_outside"]
                )
                service_metric_cols[2].metric(
                    "Finish service", service_results["finish_total"]
                )
                service_metric_cols[3].metric(
                    "Finish fuera", service_results["finish_outside"]
                )

                if service_results["dropped_points"] > 0:
                    st.info(
                        f"Se excluyeron {service_results['dropped_points']} "
                        "puntos porque su H3 no pudo llevarse a la resolución "
                        "seleccionada."
                    )

                st.dataframe(
                    service_results["endpoint_sections"],
                    hide_index=True,
                    use_container_width=True,
                )

                service_map = create_service_endpoints_map(
                    service_results["service_outside_by_h3"],
                    service_results["route_h3_cells"],
                    service_results["endpoint_buffer_cells"],
                    service_results["endpoint_cells"],
                )
                if service_map is not None:
                    st_folium(
                        service_map,
                        width=1000,
                        height=700,
                        key="gps_service_endpoints_map",
                    )

                if service_results["service_outside_by_h3"].empty:
                    st.success(
                        "No hay start_service ni finish_service por fuera de los "
                        "k-rings seleccionados."
                    )
                else:
                    if st.button(
                        (
                            "Ocultar tabla start/finish"
                            if st.session_state.get(
                                "show_gps_service_endpoint_data", False
                            )
                            else "Mostrar tabla start/finish"
                        ),
                        key="toggle_gps_service_endpoint_data",
                    ):
                        st.session_state["show_gps_service_endpoint_data"] = (
                            not st.session_state.get(
                                "show_gps_service_endpoint_data", False
                            )
                        )

                    if st.session_state.get("show_gps_service_endpoint_data", False):
                        st.dataframe(
                            service_results["service_outside_by_h3"],
                            hide_index=True,
                            use_container_width=True,
                        )
            else:
                st.info(
                    "Los filtros cambiaron. Volvé a calcular para actualizar "
                    "el mapa de start/finish."
                )
