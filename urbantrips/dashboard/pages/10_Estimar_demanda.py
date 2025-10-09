import pandas as pd
import geopandas as gpd
import streamlit as st
import numpy as np
import h3
from streamlit_folium import st_folium
import folium
from shapely.geometry import Polygon, shape, LineString
from shapely import wkt
from shapely.ops import unary_union
import contextily as cx
import seaborn as sns
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from requests.exceptions import ConnectionError as r_ConnectionError
from dash_utils import (
    levanto_tabla_sql,
    get_epsg_m,
    iniciar_conexion_db,
    get_logo,
    bring_latlon,
    configurar_selector_dia,
    leer_configs_generales,
    create_squared_polygon,
    h3_to_polygon,
    extract_hex_colors_from_cmap,
)

from urbantrips.utils.utils import guardar_tabla_sql
from urbantrips.preparo_dashboard.preparo_dashboard import preparo_indicadores_dash
from urbantrips.utils.check_configs import check_config
from urbantrips.kpi.line_od_matrix import compute_line_od_matrix
from urbantrips.kpi.kpi import compute_section_load_table
from urbantrips.utils import utils
from urbantrips.carto.carto import create_coarse_h3_from_line, create_route_section_ids
from urbantrips.carto.routes import create_route_section_points
from urbantrips.viz.viz import standarize_size
from urbantrips.geo.geo import get_h3_buffer_ring_size, create_sections_geoms
import mapclassify
from folium import Figure


def crear_mapa_folium(df_agg, cmap, var_fex, savefile="", k_jenks=5):
    location = line_od.geometry.union_all().centroid
    location = [location.y, location.x]
    try:
        bins = [df_agg[var_fex].min() - 1] + mapclassify.FisherJenks(
            df_agg[var_fex], k=k_jenks
        ).bins.tolist()
    except:
        k_jenks = 3
        bins = [df_agg[var_fex].min() - 1] + mapclassify.FisherJenks(
            df_agg[var_fex], k=k_jenks
        ).bins.tolist()

    range_bins = range(0, len(bins) - 1)
    bins_labels = [f"{int(bins[n])} a {int(bins[n+1])} viajes" for n in range_bins]
    df_agg["cuts"] = pd.cut(df_agg[var_fex], bins=bins, labels=bins_labels)

    fig = Figure(width=800, height=800)
    m = folium.Map(
        location=location,
        zoom_start=9,
        tiles="cartodbpositron",
    )

    title_html = """
    <h3 align="center" style="font-size:20px"><b>Your map title</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    line_w = 0.5

    colors = extract_hex_colors_from_cmap(cmap=cmap, n=k_jenks)

    n = 0
    for i in bins_labels:

        df_agg[df_agg.cuts == i].explore(
            m=m,
            color=colors[n],
            style_kwds={"fillOpacity": 0.1, "weight": line_w},
            name=i,
            tooltip=False,
        )
        n += 1
        line_w += 3

    folium.LayerControl(name="xx").add_to(m)

    fig.add_child(m)

    return fig


def levanto_tabla_sql_local(tabla_sql, tabla_tipo="dash", query=""):

    conn = iniciar_conexion_db(tipo=tabla_tipo)

    try:
        if len(query) == 0:
            query = f"""
            SELECT *
            FROM {tabla_sql}
            """

        tabla = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"{tabla_sql} no existe: {e}")
        tabla = pd.DataFrame([])

    conn.close()

    if len(tabla) > 0:
        if "wkt" in tabla.columns:
            tabla["geometry"] = tabla.wkt.apply(wkt.loads)
            tabla = gpd.GeoDataFrame(tabla, crs=4326)
            tabla = tabla.drop(["wkt"], axis=1)

    return tabla


@st.cache_data
def traigo_mes_dia():
    mes_dia = levanto_tabla_sql_local(
        "etapas_agregadas",
        "dash",
        "SELECT DISTINCT mes, tipo_dia FROM etapas_agregadas;",
    )
    mes = mes_dia.mes.values.tolist()
    tipo_dia = mes_dia.tipo_dia.values.tolist()
    return mes, tipo_dia


def get_legs_from_draw_line(route_h3, hour_range, day_type):

    route_h3 = ", ".join([f"'{h3}'" for h3 in route_h3])
    q_main_legs = f"""
    select id_linea, dia,factor_expansion_linea,h3_o,h3_d
    from etapas
    where od_validado==1
    and (h3_o in ({route_h3}) and h3_d in ({route_h3}))
    """

    if hour_range:
        hour_range_where = f" and hora >= {hour_range[0]} and hora <= {hour_range[1]}"
        q_main_legs = q_main_legs + hour_range_where

    day_type_is_a_date = utils.is_date_string(day_type)

    if day_type_is_a_date:
        q_main_legs = q_main_legs + f" and dia = '{day_type}'"

    print("Obteniendo datos de etapas")

    # get data for legs and route geoms
    conn_data = iniciar_conexion_db(tipo="data")
    legs = pd.read_sql(q_main_legs, conn_data)
    conn_data.close()

    legs["yr_mo"] = legs.dia.str[:7]

    if not day_type_is_a_date:
        # create a weekday_filter
        weekday_filter = pd.to_datetime(legs.dia, format="%Y-%m-%d").dt.dayofweek < 5

        if day_type == "weekday":
            legs = legs.loc[weekday_filter, :]
        else:
            legs = legs.loc[~weekday_filter, :]

    return legs


def plot_demand_by_section(lineas):

    gdf_d0 = lineas[(lineas.sentido == "ida")].copy()
    gdf_d1 = lineas[(lineas.sentido == "vuelta")].copy()
    epsg_m = get_epsg_m()
    gdf_d0 = gdf_d0.to_crs(epsg=epsg_m)
    gdf_d1 = gdf_d1.to_crs(epsg=epsg_m)

    # Arrows
    flecha_ida_wgs84 = gdf_d0.loc[
        gdf_d0.section_id == gdf_d0.section_id.min(), "geometry"
    ]

    flecha_ida_wgs84 = list(flecha_ida_wgs84.item().coords)
    flecha_ida_inicio_wgs84 = flecha_ida_wgs84[0]

    flecha_vuelta_wgs84 = gdf_d1.loc[
        gdf_d1.section_id == max(gdf_d1.section_id), "geometry"
    ]
    flecha_vuelta_wgs84 = list(flecha_vuelta_wgs84.item().coords)
    flecha_vuelta_fin_wgs84 = flecha_vuelta_wgs84[1]

    # check if route geom is drawn from west to east
    geom_dir_east = flecha_ida_inicio_wgs84[0] < flecha_vuelta_fin_wgs84[0]
    # Matching bar plot with route direction
    flecha_eo_xy = (0.4, 1.1)
    flecha_eo_text_xy = (0.05, 1.1)
    flecha_oe_xy = (0.6, 1.1)
    flecha_oe_text_xy = (0.95, 1.1)

    labels_eo = [""] * len(gdf_d0)
    labels_eo[0] = "INICIO"
    labels_eo[-1] = "FIN"
    labels_oe = [""] * len(gdf_d0)
    labels_oe[-1] = "INICIO"
    labels_oe[0] = "FIN"

    # Set arrows in barplots based on reout geom direction
    if geom_dir_east:

        flecha_ida_xy = flecha_eo_xy
        flecha_ida_text_xy = flecha_eo_text_xy
        labels_ida = labels_eo

        flecha_vuelta_xy = flecha_oe_xy
        flecha_vuelta_text_xy = flecha_oe_text_xy
        labels_vuelta = labels_oe

        # direction 0 east to west
        gdf_d0 = gdf_d0.sort_values("section_id", ascending=True)
        gdf_d1 = gdf_d1.sort_values("section_id", ascending=True)

    else:
        flecha_ida_xy = flecha_oe_xy
        flecha_ida_text_xy = flecha_oe_text_xy
        labels_ida = labels_oe

        flecha_vuelta_xy = flecha_eo_xy
        flecha_vuelta_text_xy = flecha_eo_text_xy
        labels_vuelta = labels_eo

        gdf_d0 = gdf_d0.sort_values("section_id", ascending=False)
        gdf_d1 = gdf_d1.sort_values("section_id", ascending=False)

    # For direction 0, get the last section of the route geom
    flecha_ida = gdf_d0.loc[gdf_d0.section_id == max(gdf_d0.section_id), "geometry"]
    flecha_ida = list(flecha_ida.item().coords)
    flecha_ida_inicio = flecha_ida[1]
    flecha_ida_fin = flecha_ida[0]

    # For direction 1, get the first section of the route geom
    flecha_vuelta = gdf_d1.loc[gdf_d1.section_id == gdf_d1.section_id.min(), "geometry"]
    flecha_vuelta = list(flecha_vuelta.item().coords)

    # invert the direction of the arrow
    flecha_vuelta_inicio = flecha_vuelta[0]
    flecha_vuelta_fin = flecha_vuelta[1]

    minx, miny, maxx, maxy = gdf_d0.total_bounds
    box = create_squared_polygon(minx, miny, maxx, maxy, epsg_m)

    # st.dataframe(gdf_d0.drop('geometry', axis=1))
    # st.dataframe(gdf_d1.drop('geometry', axis=1))

    # creando buffers en base a
    gdf_d0["geometry"] = gdf_d0.geometry.buffer(gdf_d0.buff_factor)
    gdf_d1["geometry"] = gdf_d1.geometry.buffer(gdf_d1.buff_factor)

    # creating plot

    f = plt.figure(tight_layout=True, figsize=(20, 15))
    gs = f.add_gridspec(nrows=3, ncols=2)
    ax1 = f.add_subplot(gs[0:2, 0])
    ax2 = f.add_subplot(gs[0:2, 1])
    ax3 = f.add_subplot(gs[2, 0])
    ax4 = f.add_subplot(gs[2, 1])

    font_dicc = {"fontsize": 18, "fontweight": "bold"}
    box.plot(ax=ax1, color="#ffffff00")
    box.plot(ax=ax2, color="#ffffff00")

    try:
        gdf_d0.plot(
            ax=ax1, column="legs", cmap="BuPu", scheme="fisherjenks", k=5, alpha=0.6
        )
        gdf_d1.plot(
            ax=ax2, column="legs", cmap="Oranges", scheme="fisherjenks", k=5, alpha=0.6
        )
    except ValueError:
        gdf_d0.plot(
            ax=ax1,
            color="purple",
            alpha=0.7,
            # linewidth=gdf_d0['buff_factor']
        )
        gdf_d1.plot(
            ax=ax2,
            color="orange",
            alpha=0.7,
            # linewidth=gdf_d1['buff_factor']
        )

    ax1.set_axis_off()
    ax2.set_axis_off()

    ax1.set_title("IDA", fontdict=font_dicc)
    ax2.set_title("VUELTA", fontdict=font_dicc)

    title = "Segmentos del recorrido - Porcentaje de etapas totales"
    y_axis_lable = "Porcentaje del total de etapas"

    f.suptitle(title, fontsize=20)

    sns.barplot(
        data=gdf_d0,
        x="section_id",
        y="prop",
        ax=ax3,
        color="Purple",
        order=gdf_d0.section_id.values,
    )

    sns.barplot(
        data=gdf_d1,
        x="section_id",
        y="prop",
        ax=ax4,
        color="Orange",
        order=gdf_d1.section_id.values,
    )

    # Axis
    ax3.set_xticklabels(labels_ida)
    ax4.set_xticklabels(labels_vuelta)

    ax3.set_ylabel(y_axis_lable)
    ax3.set_xlabel("")

    ax4.get_yaxis().set_visible(False)

    ax4.set_ylabel("")
    ax4.set_xlabel("")
    max_y_barplot = max(gdf_d0["prop"].max(), gdf_d1["prop"].max())
    ax3.set_ylim(0, max_y_barplot)
    ax4.set_ylim(0, max_y_barplot)

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax4.spines.left.set_visible(False)
    ax4.spines.right.set_visible(False)
    ax4.spines.top.set_visible(False)

    ax1.annotate(
        "",
        xy=(flecha_ida_inicio[0], flecha_ida_inicio[1]),
        xytext=(flecha_ida_fin[0], flecha_ida_fin[1]),
        arrowprops=dict(facecolor="black", edgecolor="black", shrink=0.2),
    )

    ax2.annotate(
        "",
        xy=(flecha_vuelta_inicio[0], flecha_vuelta_inicio[1]),
        xytext=(flecha_vuelta_fin[0], flecha_vuelta_fin[1]),
        arrowprops=dict(facecolor="black", edgecolor="black", shrink=0.2),
    )

    ax3.annotate(
        "Sentido",
        xy=flecha_ida_xy,
        xytext=flecha_ida_text_xy,
        size=16,
        va="center",
        ha="center",
        xycoords="axes fraction",
        arrowprops=dict(facecolor="Purple", shrink=0.05, edgecolor="Purple"),
    )
    ax4.annotate(
        "Sentido",
        xy=flecha_vuelta_xy,
        xytext=flecha_vuelta_text_xy,
        size=16,
        va="center",
        ha="center",
        xycoords="axes fraction",
        arrowprops=dict(facecolor="Orange", shrink=0.05, edgecolor="Orange"),
    )

    try:
        prov = cx.providers.CartoDB.Positron
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string(), source=prov, attribution_size=7)
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string(), source=prov, attribution_size=7)
    except (UnidentifiedImageError, ValueError):
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string(), attribution_size=7)
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string(), attribution_size=7)
    except r_ConnectionError:
        pass

    plt.close(f)
    return f


# Create unified geometry for each zone
def zona_to_geometry(h3_list):
    polygons = [h3_to_polygon(h3_index) for h3_index in h3_list]
    return unary_union(polygons)


for var in [
    "id_linea_7",
    "nombre_linea_7",
    "day_type_7",
    "yr_mo_7",
    "n_sections_7",
    "section_meters_7",
    "hour_range_7",
    "stat_7",
    "route_h3_buffer",
]:
    if var not in st.session_state:
        st.session_state[var] = None
route_h3 = None
st.set_page_config(layout="wide")
logo = get_logo()
st.image(logo)

alias_seleccionado = configurar_selector_dia()
check_config(corrida=alias_seleccionado)
st.text(f"Alias seleccionado: {alias_seleccionado}")

latlon = bring_latlon()
mes_lst, tipo_dia_lst = traigo_mes_dia()
try:

    # --- Cargar configuraciones y conexiones en session_state ---
    if "configs" not in st.session_state:
        st.session_state.configs = leer_configs_generales(autogenerado=True)

    configs = st.session_state.configs
    h3_legs_res = configs["resolucion_h3"]
    st.write("Resolución h3 para etapas:", h3_legs_res)

    alias = configs["alias_db_data"]
    conn_insumos = iniciar_conexion_db(tipo="insumos")

except ValueError as e:
    st.error(
        f"Falta una base de datos requerida: {e}. \nSe requiere full acceso a Urbantrips para correr esta página"
    )
    st.stop()

with st.expander("Dibujar linea para estimar demanda", expanded=True):
    col1, col2, col3 = st.columns([1, 3, 3])
    with col1:

        st.subheader("Periodo")

        kpi_lineas = levanto_tabla_sql("agg_indicadores")
        if len(kpi_lineas) == 0:
            months = None
        else:
            months = kpi_lineas.mes.unique()

        day_type = col1.selectbox("Tipo de dia  ", options=["weekday", "weekend"])
        st.session_state["day_type_7"] = day_type

        # add month and year
        yr_mo = col1.selectbox("Periodo  ", options=months, key="year_month")
        st.session_state["yr_mo_7"] = yr_mo

        stat = col1.selectbox(
            "Estadístico", options=["Total etapas", "Proporción"], index=1
        )
        if stat == "Total etapas":
            stat = "totals"
        else:
            stat = "proportion"
        st.session_state["stat_7"] = stat

        n_sections = col1.number_input(
            "Numero Secciones", min_value=0, max_value=999, value=None
        )
        st.session_state["n_sections_7"] = n_sections
        (
            col3a,
            col3b,
        ) = st.columns([1, 1])

        rango_desde = col3a.selectbox(
            "Rango horario (desde) ",
            options=range(0, 24),
            key="rango_hora_desde",
            index=9,
        )
        rango_hasta = col3b.selectbox(
            "Rango horario (hasta)",
            options=range(0, 24),
            key="rango_hora_hasta",
            index=9,
        )
        buffer_distance = col1.slider(
            "Distancia area influencia - buffer (metros)",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
        )
        # get_h3_buffer_ring_size
        ring_size = get_h3_buffer_ring_size(h3_legs_res, buffer_distance)

        hour_range = [rango_desde, rango_hasta]
        st.session_state["hour_range_7"] = hour_range

    # Display map with drawing tools
    with col2:
        st.subheader("Dibujar la línea")

        # Initialize Folium map
        m = folium.Map(location=latlon, zoom_start=10)
        draw = folium.plugins.Draw(
            export=False,
            draw_options={
                "polygon": False,
                "rectangle": False,
                "circle": False,
                "circlemarker": False,
                "polyline": True,
                "marker": False,
            },
            edit_options={"edit": True, "remove": True},
        )
        draw.add_to(m)

        output = st_folium(m, width=700, height=700, key="map")

    if output.get("last_active_drawing"):
        geometry_data = output["last_active_drawing"]["geometry"]
        geometry = shape(geometry_data)
        n_sections = st.session_state["n_sections_7"]
        if n_sections is None or n_sections == 0:
            st.text(
                "Por favor ingrese un número de secciones válido. Se usara 10 por defecto"
            )
            n_sections = 10
            st.session_state["n_sections_7"] = n_sections

        day_type = st.session_state["day_type_7"]
        hour_range = st.session_state["hour_range_7"]

        route_geoms = gpd.GeoDataFrame(
            {
                "id_linea": [-1],
                "geometry": [geometry],
                "wkt": [geometry.wkt],
                "n_sections": [n_sections],
                "section_meters": [None],
            },
            crs=4326,
        )
        # create section geoms
        section_geoms = create_route_section_points(route_geoms.iloc[0])
        section_geoms = create_sections_geoms(section_geoms, buffer_meters=False)
        section_geoms = section_geoms.reindex(columns=["section_id", "geometry"])

        route_h3 = create_coarse_h3_from_line(
            geometry, h3_legs_res, -1
        ).h3.values.tolist()

        route_h3_buffer = np.unique(
            np.array([h3.grid_disk(h, ring_size) for h in route_h3]).flatten()
        )
        st.session_state["route_h3_buffer"] = route_h3_buffer
        geometry_buffer = zona_to_geometry(st.session_state["route_h3_buffer"])

        gdf = gpd.GeoDataFrame(
            {
                "zona": ["Geometria buffer"],
                "geometry": [geometry_buffer],
            },
            crs="EPSG:4326",
        )

        # Plot the zones on a new Folium map
        m2 = folium.Map(
            location=[
                gdf.geometry.centroid.y.mean(),
                gdf.geometry.centroid.x.mean(),
            ],
            zoom_start=10,
        )
        folium.GeoJson(gdf, name="GeoData").add_to(m2)
        folium.GeoJson(route_geoms, name="GeoData2").add_to(m2)

        with col3:
            st.subheader("Buffer de la línea")
            output2 = st_folium(m2, width=700, height=700)

with st.expander("Demanda total y por linea"):
    if route_h3 is None:
        st.warning("Por favor dibuje una línea en el mapa para ver la demanda")
        st.stop()
    legs = get_legs_from_draw_line(route_h3_buffer, hour_range, day_type)
    st.text(f"Etapas totales en la zona: {int(legs.factor_expansion_linea.sum())}")
    lineas = legs.id_linea.unique().tolist()
    metadata = pd.read_sql(
        f"""
        SELECT id_linea, nombre_linea
        FROM metadata_lineas
        where id_linea in ({','.join(map(str, lineas))})
        """,
        conn_insumos,
    )

    # Get the default DPI or set a specific one
    dpi = plt.rcParams["figure.dpi"]  # Or set dpi = 100
    px_to_inch = 1 / dpi
    # Desired pixel dimensions
    width_px = 400
    height_px = 300

    resultados = legs.reindex(columns=["id_linea", "factor_expansion_linea"])
    resultados = resultados.groupby("id_linea").sum().reset_index()
    resultados.factor_expansion_linea = resultados.factor_expansion_linea.map(int)
    resultados = resultados.merge(metadata, on="id_linea", how="left")
    resultados = resultados.reindex(columns=["nombre_linea", "factor_expansion_linea"])
    resultados = resultados.rename(columns={"factor_expansion_linea": "Etapas totales"})
    resultados = resultados.sort_values("Etapas totales", ascending=False)

    col1, col2 = st.columns([1, 4])
    with col1:
        st.write(resultados)
    with col2:
        st.bar_chart(resultados, x="nombre_linea", y="Etapas totales")

with st.expander("Demanda por segmento de recorrido"):
    if route_h3 is None:
        st.warning("Por favor dibuje una línea en el mapa para ver la demanda")
        st.stop()
    st.text(
        "A partir de la línea dibujada, se estimará la demanda en función de las etapas agregadas disponibles."
    )
    legs["id_linea"] = -1
    stat = st.session_state["stat_7"]
    factor = 500
    factor_min = 10
    # Set title and plot axis
    if stat == "totals":
        title = "Segmentos del recorrido - Cantidad de etapas"
        y_axis_lable = "Cantidad de etapas por sentido"
        indicator_col = "legs"
    elif stat == "proportion":
        title = "Segmentos del recorrido - Porcentaje de etapas totales"
        y_axis_lable = "Porcentaje del total de etapas"
        indicator_col = "prop"

    else:
        raise Exception("Indicador stat debe ser 'cantidad_etapas' o 'prop_etapas'")

    if len(legs) > 0:
        section_load = compute_section_load_table(
            legs, route_geoms, hour_range, day_type
        )
        section_load = section_geoms.merge(section_load, on="section_id", how="inner")
        section_load["buff_factor"] = standarize_size(
            series=section_load[indicator_col], min_size=factor_min, max_size=factor
        )
        f_lineas = plot_demand_by_section(section_load)
        st.pyplot(f_lineas)
    else:
        st.write("No hay sufiente demanda para computar la demanda por segmento")

with st.expander("Líneas de deseo por linea"):
    if len(legs) > 0:
        n_sections = st.session_state["n_sections_7"]

        line_od_data = legs.copy()

        section_ids = create_route_section_ids(n_sections)

        section_carto = section_load.reindex(
            columns=["section_id", "geometry"]
        ).drop_duplicates(subset="section_id")
        section_carto.geometry = section_carto.geometry.centroid

        labels = list(range(1, len(section_ids)))

        line_od_data["o_proj"] = pd.cut(
            line_od_data.o_proj, bins=section_ids, labels=labels, right=True
        )
        line_od_data["d_proj"] = pd.cut(
            line_od_data.d_proj, bins=section_ids, labels=labels, right=True
        )

        totals_by_day_section_id = (
            line_od_data.groupby(["dia", "o_proj", "d_proj"])
            .agg(legs=("factor_expansion_linea", "sum"))
            .reset_index()
        )
        totals_by_day = totals_by_day_section_id.groupby(["dia"], as_index=False).agg(
            daily_legs=("legs", "sum")
        )

        totals_by_typeday = totals_by_day.daily_legs.mean()

        # then average for type of day
        totals_by_typeday_section_id = (
            totals_by_day_section_id.groupby(["o_proj", "d_proj"])
            .agg(legs=("legs", "mean"))
            .reset_index()
        )
        totals_by_typeday_section_id["legs"] = (
            totals_by_typeday_section_id["legs"].round().map(int)
        )
        totals_by_typeday_section_id["prop"] = (
            totals_by_typeday_section_id.legs / totals_by_typeday * 100
        ).round(1)
        totals_by_typeday_section_id["day_type"] = day_type
        totals_by_typeday_section_id["n_sections"] = n_sections

        totals_by_typeday_section_id = totals_by_typeday_section_id.merge(
            section_carto, left_on="o_proj", right_on="section_id", how="left"
        ).merge(
            section_carto,
            left_on="d_proj",
            right_on="section_id",
            suffixes=("_o", "_d"),
            how="left",
        )
        geometry = [
            LineString([(row.geometry_o), (row.geometry_d)])
            for _, row in totals_by_typeday_section_id.iterrows()
        ]
        line_od = gpd.GeoDataFrame(
            totals_by_typeday_section_id.reindex(
                columns=["o_proj", "d_proj", "legs", "prop", "day_type"]
            ),
            geometry=geometry,
            crs=4326,
        )

        if len(line_od) > 0:
            if st.checkbox("Mostrar datos ", value=False, key="mostrar_datos2"):
                st.write(line_od)

            k_jenks = st.slider("Cantidad de grupos", min_value=1, max_value=5, value=5)
            st.text(f"Hay un total de {line_od.legs.sum()} etapas")
            try:
                map = crear_mapa_folium(
                    line_od, cmap="BuPu", var_fex="legs", k_jenks=k_jenks
                )
                st_map = st_folium(map, width=900, height=700)
            except ValueError as e:
                st.write(
                    "Error al crear el mapa. Verifique los parametros seleccionados "
                )
        else:
            st.write("No hay datos para mostrar")
    else:
        st.write("No hay sufiente demanda para computar las lineas de deseo")

if st.button("Procesar polígono"):
    st.write(
        "Procesando polígono del recorrido para análisis de cuenca. Esto puede tomar unos minutos..."
    )
    # Agregar resultado al geojson de poligonos
    poligonos = levanto_tabla_sql_local("poligonos", "insumos")

    if len(poligonos) > 0:
        poligonos["wkt"] = poligonos.geometry.map(lambda g: g.wkt)

    drawn_poli_id = "estimacion de demanda dibujada"
    drawn_poli = gdf.copy()
    drawn_poli["id"] = drawn_poli_id
    drawn_poli["tipo"] = "cuenca"
    drawn_poli["wkt"] = drawn_poli.geometry.map(lambda g: g.wkt)

    poligonos = poligonos.loc[poligonos["id"] != drawn_poli_id, :]

    drawn_poli = pd.concat([poligonos, drawn_poli], ignore_index=True)
    drawn_poli = drawn_poli.reindex(columns=["id", "tipo", "wkt"])

    st.write("Guardando poligono en base de insumos")
    guardar_tabla_sql(drawn_poli, "poligonos", "insumos", modo="replace")

    corrida = alias_seleccionado
    st.write("Corriendo cuencas para Polígonos ...")

    preparo_indicadores_dash(
        lineas_deseo=False,
        poligonos=True,
        kpis=False,
        corrida=corrida,
        resoluciones=[6],
        poligon_id=drawn_poli_id,
    )

    st.cache_data.clear()
    st.write("Datos procesados, puede ver los patrones en el apartado Polígonos")
