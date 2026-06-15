import pandas as pd
import geopandas as gpd
import streamlit as st
import numpy as np
import h3
from streamlit_folium import st_folium
import folium
from shapely.geometry import Polygon, shape, LineString
from shapely.ops import unary_union
import contextily as cx
import seaborn as sns
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from requests.exceptions import ConnectionError as r_ConnectionError
from dash_storage import leer_configs_generales
from dash_utils import (
    levanto_tabla_sql,
    get_epsg_m,
    get_logo,
    bring_latlon,
    configurar_selector_dia,
    create_squared_polygon,
    h3_to_polygon,
    extract_hex_colors_from_cmap,
)
from shapely.ops import linemerge
from urbantrips.utils.utils import guardar_tabla_sql
from urbantrips.preparo_dashboard.preparo_dashboard import preparo_indicadores_dash
from urbantrips.carto.equivalencias import (
    construir_equivalencias_zonas,
    upsert_equivalencias_zonas,
)
from urbantrips.preparo_dashboard.chains import RES_CHAINS_NORM
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
import json
from shapely.geometry import shape, LineString, MultiLineString


def crear_mapa_folium(df_agg, cmap, var_fex, savefile="", k_jenks=5):
    location = df_agg.geometry.union_all().centroid
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
    return utils.levanto_tabla_sql(
        tabla_sql,
        tabla_tipo=tabla_tipo,
        query=query,
    )


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
    legs = utils.levanto_tabla_sql(
        "etapas",
        tabla_tipo="data",
        query=q_main_legs,
    )

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

# check_config(corrida=alias_seleccionado)
# st.text(f"Alias seleccionado: {alias_seleccionado}")

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

   
    with col2:
        st.subheader("Dibujar la línea")
    
        fuente = st.radio(
            "Fuente de la geometría",
            ["Dibujar en el mapa", "Cargar GeoJSON"],
            horizontal=True,
            key="route_source_7",
        )
    
        geometry = None  # <- acá vamos a dejar la geometría final (LineString)
    
        if fuente == "Dibujar en el mapa":
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
    
        else:
            geojson_file = st.file_uploader(
                "Subí un GeoJSON con una línea (FeatureCollection/Feature) o un archivo exportado por QGIS",
                type=["geojson", "json"],
                key="route_geojson_7",
            )
    
            if geojson_file is not None:
                try:
                    geojson = json.load(geojson_file)
    
                    # Opción A (más robusta): leer con GeoPandas desde un archivo temporal
                    # (Streamlit requiere escribirlo a disco si querés usar gpd.read_file)
                    # Para evitar tocar mucho, hacemos parse directo con shapely:
    
                    # def _extract_line_from_geojson(obj):
                    #     # obj puede ser FeatureCollection, Feature o Geometry
                    #     if obj.get("type") == "FeatureCollection":
                    #         geoms = [shape(f["geometry"]) for f in obj.get("features", []) if f.get("geometry")]
                    #     elif obj.get("type") == "Feature":
                    #         geoms = [shape(obj.get("geometry"))] if obj.get("geometry") else []
                    #     else:
                    #         geoms = [shape(obj)]
    
                    #     geoms = [g for g in geoms if g is not None and not g.is_empty]
                    #     if not geoms:
                    #         return None
    
                    #     # Si hay varias geometrías, las unimos/mergeamos si son líneas
                    #     # (si hubiera polígonos por error, esto no va a ser una línea válida)
                    #     g = linemerge(geoms) if len(geoms) > 1 else geoms[0]
    
                    #     # Normalizar MultiLineString -> LineString cuando sea posible
                    #     if isinstance(g, MultiLineString):
                    #         g = linemerge(g)
    
                    #     return g
                    from shapely.geometry import shape, LineString, MultiLineString, Point
                    from shapely.ops import linemerge
                    import numpy as np
                    
                    def _extract_line_from_geojson(obj):
                        """
                        Acepta:
                        - LineString / MultiLineString
                        - Point o múltiples Point (estaciones)
                        Devuelve:
                        - LineString o None
                        """
                    
                        # -------------------------
                        # Extraer geometrías
                        # -------------------------
                        if obj.get("type") == "FeatureCollection":
                            geoms = [shape(f["geometry"]) for f in obj.get("features", []) if f.get("geometry")]
                        elif obj.get("type") == "Feature":
                            geoms = [shape(obj.get("geometry"))] if obj.get("geometry") else []
                        else:
                            geoms = [shape(obj)]
                    
                        geoms = [g for g in geoms if g is not None and not g.is_empty]
                    
                        if not geoms:
                            return None
                    
                        # -------------------------
                        # Caso 1: ya es línea
                        # -------------------------
                        if all(g.geom_type in ("LineString", "MultiLineString") for g in geoms):
                            g = linemerge(geoms) if len(geoms) > 1 else geoms[0]
                            if g.geom_type == "MultiLineString":
                                g = linemerge(g)
                            return g if g.geom_type == "LineString" else None
                    
                        # -------------------------
                        # Caso 2: son puntos (estaciones)
                        # -------------------------
                        if all(g.geom_type == "Point" for g in geoms):
                            coords = np.array([(p.x, p.y) for p in geoms])
                    
                            # Si vienen ordenados, esto ya alcanza
                            # Si no, hacemos un ordenamiento simple por distancia acumulada
                            if len(coords) > 2:
                                ordered = [coords[0]]
                                remaining = list(coords[1:])
                    
                                while remaining:
                                    last = ordered[-1]
                                    dists = [np.linalg.norm(last - r) for r in remaining]
                                    idx = int(np.argmin(dists))
                                    ordered.append(remaining.pop(idx))
                    
                                coords = np.array(ordered)
                    
                            return LineString(coords)
                    
                        # -------------------------
                        # Caso no soportado
                        # -------------------------
                        return None

    
                    geometry = _extract_line_from_geojson(geojson)
    
                except Exception as e:
                    st.error(f"No pude leer el GeoJSON. Detalle: {e}")
                    geometry = None
    
    # -----------------------------
    # Validación mínima 
    # -----------------------------
    if geometry is not None:
        # Asegurar que sea línea (si te suben polygon por error)
        if geometry.geom_type not in ("LineString", "MultiLineString"):
            st.error(f"La geometría debe ser una línea. Recibí: {geometry.geom_type}")
        else:
            # Si llega MultiLineString, intentamos normalizar
            if geometry.geom_type == "MultiLineString":
                geometry = linemerge(geometry)
                if geometry.geom_type != "LineString":
                    st.error("La geometría es MultiLineString y no pude convertirla a una sola LineString.")
                    geometry = None
    
    if geometry is not None:
        n_sections = st.session_state["n_sections_7"]
        if n_sections is None or n_sections == 0:
            st.text("Por favor ingrese un número de secciones válido. Se usara 10 por defecto")
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
    # st.text(f"Etapas totales en la zona: {int(legs.factor_expansion_linea.sum())}")
    lineas = legs.id_linea.unique().tolist()
    metadata = utils.levanto_tabla_sql(
        "metadata_lineas",
        tabla_tipo="insumos",
        query=f"""
        SELECT id_linea, nombre_linea
        FROM metadata_lineas
        where id_linea in ({','.join(map(str, lineas))})
        """,
    )

    # Get the default DPI or set a specific one
    dpi = plt.rcParams["figure.dpi"]  # Or set dpi = 100
    px_to_inch = 1 / dpi
    # Desired pixel dimensions
    width_px = 400
    height_px = 300

    resultados = legs.reindex(columns=["id_linea", "factor_expansion_linea"])
    resultados = resultados.groupby("id_linea").sum().reset_index()
    resultados.factor_expansion_linea = resultados.factor_expansion_linea.round().map(int)
    resultados = resultados.merge(metadata, on="id_linea", how="left")
    resultados = resultados.reindex(columns=["nombre_linea", "factor_expansion_linea"])
    resultados = resultados.rename(columns={"factor_expansion_linea": "Etapas totales"})
    resultados = resultados.sort_values("Etapas totales", ascending=False)
    st.text(f"Etapas totales en la zona: {int(resultados['Etapas totales'].sum())}")

    col1, col2 = st.columns([1, 4])
    with col1:
        st.write(resultados)
        # st.write(legs)
    with col2:
        # st.bar_chart(resultados, x="nombre_linea", y="Etapas totales")
        import altair as alt

        chart = (
            alt.Chart(resultados)
            .mark_bar()
            .encode(
                x=alt.X(
                    "nombre_linea:N",
                    sort=alt.EncodingSortField(
                        field="Etapas totales",
                        order="descending"
                    )
                ),
                y="Etapas totales:Q",
                tooltip=["nombre_linea", "Etapas totales"]
            )
        )
        
        st.altair_chart(chart, use_container_width=True)


with st.expander("Superposición"):
    if route_h3 is None:
        st.warning("Por favor dibuje una línea en el mapa para ver la superposición")
        st.stop()

    if "resultados" not in locals() or resultados is None or resultados.empty:
        st.warning("No hay resultados por línea para superponer.")
        st.stop()

    # -----------------------------
    # 1) Top N + selección/deselección
    # -----------------------------
    top_n = st.slider("Cantidad de líneas a superponer (Top N por etapas)", 1, 30, 1, key="sup_top_n")

    # Asegurar id_linea en resultados_sup
    if "id_linea" not in resultados.columns:
        name_to_id = (
            metadata.drop_duplicates("nombre_linea")
            .set_index("nombre_linea")["id_linea"]
            .to_dict()
        )
        resultados_sup = resultados.copy()
        resultados_sup["id_linea"] = resultados_sup["nombre_linea"].map(name_to_id)
    else:
        resultados_sup = resultados.copy()

    resultados_sup = resultados_sup.dropna(subset=["id_linea"]).copy()
    resultados_sup["id_linea"] = resultados_sup["id_linea"].astype(int)
    resultados_sup = resultados_sup.head(top_n).copy()

    if resultados_sup.empty:
        st.warning("No pude determinar id_linea para superponer.")
        st.stop()

    # Label legible (OJO: cambia cuando cambian "Etapas totales" -> por eso se desincroniza el default)
    if "Etapas totales" in resultados_sup.columns:
        resultados_sup["label"] = (
            resultados_sup["nombre_linea"].fillna("sin_nombre")
            + " — "
            + resultados_sup["Etapas totales"].astype(int).astype(str)
            + " etapas"
        )
    else:
        resultados_sup["label"] = resultados_sup["nombre_linea"].fillna("sin_nombre")

    sel_key = "superpos_sel_lineas"
    opts = resultados_sup["label"].tolist()

    # --- FIX: defaults siempre deben existir en options ---
    prev = st.session_state.get(sel_key, None)
    if not prev:
        default = opts[:]  # primera vez: todas
    else:
        default = [x for x in prev if x in opts]  # solo válidos
        if len(default) == 0:
            default = opts[:]  # fallback recomendado

    seleccion = st.multiselect(
        "Seleccionar líneas a mostrar",
        options=opts,
        default=default,
        key="sup_multiselect",
    )
    st.session_state[sel_key] = seleccion

    label_to_id = resultados_sup.set_index("label")["id_linea"].to_dict()
    ids = [label_to_id[lbl] for lbl in seleccion if lbl in label_to_id]

    # -----------------------------
    # 2) Traer geometrías de líneas seleccionadas
    # -----------------------------
    gdf_lineas = levanto_tabla_sql("lines_geoms", "insumos")
    if gdf_lineas is None or len(gdf_lineas) == 0:
        st.warning("No pude levantar geometrías de líneas con levanto_tabla_sql('lines_geoms', 'insumos').")
        st.stop()

    if ids:
        gdf_sup = gdf_lineas[gdf_lineas["id_linea"].isin(ids)].copy()
    else:
        gdf_sup = gdf_lineas.iloc[0:0].copy()

    # Asegurar nombre_linea
    if not gdf_sup.empty and "nombre_linea" not in gdf_sup.columns:
        gdf_sup = gdf_sup.merge(metadata, on="id_linea", how="left")

    # Agregar Etapas totales a gdf_sup (si existe)
    if not gdf_sup.empty and "Etapas totales" in resultados_sup.columns:
        gdf_sup = gdf_sup.merge(
            resultados_sup.reindex(columns=["id_linea", "Etapas totales"]),
            on="id_linea",
            how="left",
        )

    # -----------------------------
    # 3) H3 como buffer (hexágono = buffer)
    # -----------------------------
    st.markdown("### Superposición usando H3 como buffer")

    h3_res_sup = st.selectbox(
        "Resolución H3 (hexágonos que representan el buffer)",
        options=list(range(5, 11)),
        index=3,  # -> 8 si opciones=5..10
        key="sup_h3_res",
    )

    def _line_to_h3_set(line_geom, h3_res):
        core = create_coarse_h3_from_line(line_geom, h3_res, -999)["h3"].tolist()
        return set(core)

    drawn_geom = route_geoms.geometry.iloc[0]
    h3_drawn = _line_to_h3_set(drawn_geom, h3_res_sup)

    # -----------------------------
    # 4) Superposición por línea + total
    # -----------------------------
    rows = []
    h3_all_selected = set()

    if not gdf_sup.empty:
        for _, r in gdf_sup.iterrows():
            idl = int(r["id_linea"])
            name = r["nombre_linea"] if "nombre_linea" in r else str(idl)

            h3_line = _line_to_h3_set(r.geometry, h3_res_sup)
            h3_all_selected |= h3_line

            inter = h3_drawn.intersection(h3_line)
            union = h3_drawn.union(h3_line)

            pct_on_drawn = (len(inter) / len(h3_drawn) * 100) if len(h3_drawn) else 0.0
            pct_on_line = (len(inter) / len(h3_line) * 100) if len(h3_line) else 0.0
            jaccard = (len(inter) / len(union)) if len(union) else 0.0

            rows.append(
                {
                    "id_linea": idl,
                    "nombre_linea": name,
                    "hex_dibujada": len(h3_drawn),
                    "hex_linea": len(h3_line),
                    "hex_intersección": len(inter),
                    "% sobre dibujada": round(pct_on_drawn, 1),
                    "% sobre línea": round(pct_on_line, 1),
                    "Jaccard": round(jaccard, 3),
                }
            )

    df_overlap = pd.DataFrame(rows)

    # TOTAL: unión de todas las líneas seleccionadas vs dibujada
    h3_inter_total = h3_drawn.intersection(h3_all_selected)
    h3_union_total = h3_drawn.union(h3_all_selected)

    pct_total_on_drawn = (len(h3_inter_total) / len(h3_drawn) * 100) if len(h3_drawn) else 0.0
    pct_total_on_selected = (len(h3_inter_total) / len(h3_all_selected) * 100) if len(h3_all_selected) else 0.0
    jaccard_total = (len(h3_inter_total) / len(h3_union_total)) if len(h3_union_total) else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Hexágonos (línea dibujada)", f"{len(h3_drawn):,}")
    c2.metric("Hexágonos (unión seleccionadas)", f"{len(h3_all_selected):,}")
    c3.metric("Hexágonos (intersección total)", f"{len(h3_inter_total):,}")

    st.write(
        {
            "% superposición TOTAL sobre dibujada": round(pct_total_on_drawn, 1),
            "% superposición TOTAL sobre seleccionadas": round(pct_total_on_selected, 1),
            "Jaccard TOTAL": round(jaccard_total, 3),
        }
    )

    if not df_overlap.empty:
        df_overlap = df_overlap.sort_values("% sobre dibujada", ascending=False)
        st.dataframe(df_overlap, use_container_width=True)
    else:
        st.info("No hay líneas seleccionadas para calcular superposición (solo se muestra la línea dibujada).")

    # -----------------------------
    # 5) Mapa folium con colores:
    #    - H3 seleccionadas: claro
    #    - H3 dibujada: gris claro
    #    - Intersección: oscuro
    # -----------------------------
    st.markdown("### Mapa de superposición (H3 claro/oscuro)")

    colv1, colv2, colv3 = st.columns(3)
    show_h3_selected = colv1.checkbox("Ver H3 líneas seleccionadas (claro)", value=True, key="sup_show_h3_selected")
    show_h3_drawn = colv2.checkbox("Ver H3 línea dibujada (gris)", value=True, key="sup_show_h3_drawn")
    show_h3_inter_total = colv3.checkbox("Ver H3 superpuesto (oscuro)", value=True, key="sup_show_h3_inter_total")

    show_h3_inter_one = st.checkbox(
        "Ver intersección por línea (además de la total)",
        value=False,
        key="sup_show_h3_inter_one",
    )

    center = drawn_geom.centroid
    m_sup = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron")

    # Línea dibujada (distintiva)
    folium.GeoJson(
        route_geoms.__geo_interface__,
        name="Línea dibujada",
        style_function=lambda x: {
            "color": "#111111",
            "weight": 8,
            "opacity": 0.95,
            "dashArray": "8, 6",
        },
        tooltip=folium.Tooltip("Línea dibujada"),
    ).add_to(m_sup)

    # Líneas seleccionadas (geom)
    if not gdf_sup.empty:
        tooltip_fields = ["nombre_linea", "id_linea"]
        tooltip_aliases = ["Línea", "ID"]
        if "Etapas totales" in gdf_sup.columns:
            tooltip_fields = ["nombre_linea", "id_linea", "Etapas totales"]
            tooltip_aliases = ["Línea", "ID", "Etapas totales"]

        folium.GeoJson(
            gdf_sup.__geo_interface__,
            name="Líneas seleccionadas",
            style_function=lambda x: {"color": "#ff0000", "weight": 3, "opacity": 0.8},
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases),
        ).add_to(m_sup)

    def _add_h3_union_layer(h3_set, layer_name, color, fill_opacity):
        if not h3_set:
            return
        geom_union = zona_to_geometry(list(h3_set))
        folium.GeoJson(
            gpd.GeoSeries([geom_union], crs=4326).__geo_interface__,
            name=layer_name,
            style_function=lambda x: {"color": color, "weight": 2, "fillOpacity": fill_opacity},
        ).add_to(m_sup)

    if show_h3_selected:
        _add_h3_union_layer(h3_all_selected, f"H3 líneas seleccionadas (res {h3_res_sup})", "#ff0000", 0.08)

    if show_h3_drawn:
        _add_h3_union_layer(h3_drawn, f"H3 línea dibujada (res {h3_res_sup})", "#444444", 0.15)

    if show_h3_inter_total:
        _add_h3_union_layer(h3_inter_total, f"H3 superpuesto TOTAL (res {h3_res_sup})", "#1f77b4", 0.35)

    if show_h3_inter_one and not df_overlap.empty and not gdf_sup.empty:
        opciones_vis = df_overlap["nombre_linea"].tolist()
        sel_vis = st.selectbox("Línea para visualizar intersección H3", options=opciones_vis, key="sup_sel_vis")

        id_vis = int(df_overlap.loc[df_overlap["nombre_linea"] == sel_vis, "id_linea"].iloc[0])
        geom_vis = gdf_sup.loc[gdf_sup["id_linea"] == id_vis, "geometry"].iloc[0]
        h3_vis = _line_to_h3_set(geom_vis, h3_res_sup)
        h3_inter_one = h3_drawn.intersection(h3_vis)

        _add_h3_union_layer(h3_inter_one, f"H3 superpuesto — {sel_vis}", "#006400", 0.35)

    folium.LayerControl(collapsed=True).add_to(m_sup)
    st_folium(m_sup, width=900, height=700)

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

# with st.expander("Líneas de deseo por linea"):
#     if len(legs) > 0:
#         n_sections = st.session_state["n_sections_7"]

#         line_od_data = legs.copy()

#         section_ids = create_route_section_ids(n_sections)

#         section_carto = section_load.reindex(
#             columns=["section_id", "geometry"]
#         ).drop_duplicates(subset="section_id")
#         section_carto.geometry = section_carto.geometry.centroid

#         labels = list(range(1, len(section_ids)))

#         line_od_data["o_proj"] = pd.cut(
#             line_od_data.o_proj, bins=section_ids, labels=labels, right=True
#         )
#         line_od_data["d_proj"] = pd.cut(
#             line_od_data.d_proj, bins=section_ids, labels=labels, right=True
#         )

#         totals_by_day_section_id = (
#             line_od_data.groupby(["dia", "o_proj", "d_proj"])
#             .agg(legs=("factor_expansion_linea", "sum"))
#             .reset_index()
#         )
#         totals_by_day = totals_by_day_section_id.groupby(["dia"], as_index=False).agg(
#             daily_legs=("legs", "sum")
#         )

#         totals_by_typeday = totals_by_day.daily_legs.mean()

#         # then average for type of day
#         totals_by_typeday_section_id = (
#             totals_by_day_section_id.groupby(["o_proj", "d_proj"])
#             .agg(legs=("legs", "mean"))
#             .reset_index()
#         )
#         totals_by_typeday_section_id["legs"] = (
#             totals_by_typeday_section_id["legs"].round().map(int)
#         )
#         totals_by_typeday_section_id["prop"] = (
#             totals_by_typeday_section_id.legs / totals_by_typeday * 100
#         ).round(1)
#         totals_by_typeday_section_id["day_type"] = day_type
#         totals_by_typeday_section_id["n_sections"] = n_sections

#         totals_by_typeday_section_id = totals_by_typeday_section_id.merge(
#             section_carto, left_on="o_proj", right_on="section_id", how="left"
#         ).merge(
#             section_carto,
#             left_on="d_proj",
#             right_on="section_id",
#             suffixes=("_o", "_d"),
#             how="left",
#         )
#         geometry = [
#             LineString([(row.geometry_o), (row.geometry_d)])
#             for _, row in totals_by_typeday_section_id.iterrows()
#         ]
#         line_od = gpd.GeoDataFrame(
#             totals_by_typeday_section_id.reindex(
#                 columns=["o_proj", "d_proj", "legs", "prop", "day_type"]
#             ),
#             geometry=geometry,
#             crs=4326,
#         )

#         if len(line_od) > 0:
#             if st.checkbox("Mostrar datos ", value=False, key="mostrar_datos2"):
#                 st.write(line_od)

#             k_jenks = st.slider("Cantidad de grupos", min_value=1, max_value=5, value=5)
#             st.text(f"Hay un total de {line_od.legs.sum()} etapas")
#             try:
#                 map = crear_mapa_folium(
#                     line_od, cmap="BuPu", var_fex="legs", k_jenks=k_jenks
#                 )
#                 st_map = st_folium(map, width=900, height=700)
#             except ValueError as e:
#                 st.write(
#                     "Error al crear el mapa. Verifique los parametros seleccionados "
#                 )
#         else:
#             st.write("No hay datos para mostrar")
#     else:
#         st.write("No hay sufiente demanda para computar las lineas de deseo")

with st.expander("Polígono de análisis de cuenca"):
    st.write(
        "En este apartado se puede dibujar un polígono (o subir un GeoJSON) para estimar la demanda en esa zona, sin necesidad de que esté asociada a una línea de transporte público específica. Esto es útil para analizar zonas sin cobertura o con cobertura deficiente."
    )

    # --- ID del polígono dibujado (editable por el usuario) ---
    default_drawn_poli_id = "estimacion de demanda dibujada"

    if "drawn_poli_id" not in st.session_state or not st.session_state["drawn_poli_id"]:
        st.session_state["drawn_poli_id"] = default_drawn_poli_id

    # corrida = alias_seleccionado

    # correr = st.selectbox(
    #     "Corridas", options=[corrida, 'Todas'], index=0)
    # if correr == 'Todas':
    #     corridas = leer_configs_generales(autogenerado=False)['corridas']
    # else:
    #     corridas = [corrida]


    drawn_poli_id = st.text_input(
        "ID del polígono (para guardar en insumos)",
        value=st.session_state["drawn_poli_id"],
        help="Este identificador se usará para guardar/actualizar el polígono en la tabla 'poligonos'.",
        key="drawn_poli_id_input",
    ).strip()



    # persistir (por si el usuario lo modifica)
    st.session_state["drawn_poli_id"] = drawn_poli_id if drawn_poli_id else default_drawn_poli_id
    drawn_poli_id = st.session_state["drawn_poli_id"]

    if st.button("Procesar polígono"):
        drawn_poli_id = st.session_state["drawn_poli_id"]
        st.write("Procesando polígono de cuenca: " + drawn_poli_id)

        # Geometría del polígono = buffer dibujado de la línea (gdf)
        poly_gdf = gdf.copy()
        poly_gdf["id"] = drawn_poli_id
        poly_gdf["tipo"] = "cuenca"
        poly_gdf = poly_gdf.reindex(columns=["id", "tipo", "geometry"])
        if poly_gdf.crs is None:
            poly_gdf = poly_gdf.set_crs(4326)

        # --- 1) Upsert de la geometría del polígono en la tabla 'poligonos' ---
        # Se preserva el contorno para los mapas. Se reemplaza solo este id.
        poligonos = levanto_tabla_sql_local("poligonos", "insumos")
        if len(poligonos) > 0:
            poligonos = poligonos.loc[poligonos["id"] != drawn_poli_id, :]
        poligonos_out = pd.concat([poligonos, poly_gdf], ignore_index=True)
        st.write("Guardando geometría del polígono en base de insumos")
        guardar_tabla_sql(poligonos_out, "poligonos", "insumos", modo="replace")

        # --- 2) Construir equivalencias_zonas (formato largo) para el polígono ---
        # Mismas resoluciones que el pipeline (resolucion_h3 + RES_CHAINS_NORM),
        # para que los joins del dashboard funcionen en todas las capas.
        resoluciones = sorted({int(configs["resolucion_h3"]), RES_CHAINS_NORM})
        equiv = construir_equivalencias_zonas(
            gdf_poligonos=poly_gdf,
            resoluciones=resoluciones,
            modo_poligonos="overlap",
        )

        # --- 3) Upsert en equivalencias_zonas (borra solo este polígono y lo
        # reescribe, preservando el resto) en insumos y en la copia dash. ---
        st.write(
            f"Actualizando equivalencias_zonas: {len(equiv)} celdas H3 para "
            f"'{drawn_poli_id}'"
        )
        upsert_equivalencias_zonas(equiv, db_path="insumos")
        upsert_equivalencias_zonas(equiv, db_path="dash")

        st.cache_data.clear()
        st.success(
            f"Polígono '{drawn_poli_id}' procesado. "
            "Puede ver los patrones en el apartado Polígonos."
        )
