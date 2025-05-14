import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import mapclassify
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from PIL import UnidentifiedImageError
from requests.exceptions import ConnectionError as r_ConnectionError
from folium import Figure
from shapely.geometry import LineString
from dash_utils import (
    levanto_tabla_sql,
    get_logo,
    create_linestring_od,
    create_squared_polygon,
    get_epsg_m,
    extract_hex_colors_from_cmap,
    levanto_tabla_sql_local,
    configurar_selector_dia
)

try:
    from urbantrips.utils.utils import iniciar_conexion_db
    from urbantrips.utils import utils
except ImportError as e:
    st.error(
        f"Falta una librería requerida: {e}. Algunas funcionalidades no estarán disponibles. \nSe requiere full acceso a Urbantrips para correr esta página"
    )
    st.stop()


def crear_mapa_folium(df_agg, cmap, var_fex, savefile="", k_jenks=5):

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
        location=[df_agg.lat_o.mean(), df_agg.lon_o.mean()],
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


def plot_frequency_by_section(
    lineas, id_linea, nombre_linea, day_type, n_sections, rango
):
    indicator_col = "frequency_interval"
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

    f = plt.figure(tight_layout=True, figsize=(20, 10))
    ax1 = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2)

    font_dicc = {"fontsize": 18, "fontweight": "bold"}
    box.plot(ax=ax1, color="#ffffff00")
    box.plot(ax=ax2, color="#ffffff00")

    try:
        gdf_d0.plot(
            ax=ax1,
            column=indicator_col,
            cmap="PiYG",
            categorical=True,
            alpha=0.8,
            legend=True,
        )
        gdf_d1.plot(
            ax=ax2,
            column=indicator_col,
            cmap="managua",
            categorical=True,
            alpha=0.8,
            legend=True,
        )
    except ValueError:
        gdf_d0.plot(ax=ax1, column=indicator_col, cmap="BuPu", alpha=0.6)
        gdf_d1.plot(ax=ax2, column=indicator_col, cmap="Oranges", alpha=0.6)

    ax1.set_axis_off()
    ax2.set_axis_off()

    ax1.set_title("IDA", fontdict=font_dicc, y=1.0, pad=-20)
    ax2.set_title("VUELTA", fontdict=font_dicc, y=1.0, pad=-20)

    title = "Promedio de frequencia - "

    if nombre_linea == "":
        title = title + f"Id línea {id_linea} - {day_type}\n{rango}"
    else:
        title = (
            title
            + f"Línea: {nombre_linea.replace('Línea ', '')} - Id línea: {id_linea} - {day_type}\n{rango}"
        )

    f.suptitle(title, fontsize=20)

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


def plot_speed_by_section(lineas, id_linea, nombre_linea, day_type, n_sections, rango):
    indicator_col = "speed_interval"

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
            ax=ax1,
            column=indicator_col,
            cmap="RdYlGn",
            categorical=True,
            alpha=0.6,
            legend=True,
        )
        gdf_d1.plot(
            ax=ax2,
            column=indicator_col,
            cmap="RdYlGn",
            categorical=True,
            alpha=0.6,
            legend=True,
        )
    except ValueError:
        gdf_d0.plot(ax=ax1, column=indicator_col, cmap="RdYlGn", alpha=0.6)
        gdf_d1.plot(ax=ax2, column=indicator_col, cmap="RdYlGn", alpha=0.6)

    ax1.set_axis_off()
    ax2.set_axis_off()

    ax1.set_title("IDA", fontdict=font_dicc)
    ax2.set_title("VUELTA", fontdict=font_dicc)

    title = "Promedio de velocidad - "
    y_axis_lable = "Velocidad promedio (kmh)"

    if nombre_linea == "":
        title = title + f"Id línea {id_linea} - {day_type}\n{rango}"
    else:
        title = (
            title
            + f"Línea: {nombre_linea.replace('Línea ', '')} - Id línea: {id_linea} - {day_type}\n{rango}"
        )

    f.suptitle(title, fontsize=20)

    bar_variable = "avg_speed"

    sns.barplot(
        data=gdf_d0,
        x="section_id",
        y=bar_variable,
        ax=ax3,
        color="Grey",
        order=gdf_d0.section_id.values,
    )

    sns.barplot(
        data=gdf_d1,
        x="section_id",
        y=bar_variable,
        ax=ax4,
        color="Grey",
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

    max_y_barplot = max(gdf_d0[bar_variable].max(), gdf_d1[bar_variable].max())
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
        arrowprops=dict(facecolor="Grey", shrink=0.05, edgecolor="Grey"),
    )
    ax4.annotate(
        "Sentido",
        xy=flecha_vuelta_xy,
        xytext=flecha_vuelta_text_xy,
        size=16,
        va="center",
        ha="center",
        xycoords="axes fraction",
        arrowprops=dict(facecolor="Grey", shrink=0.05, edgecolor="Grey"),
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


def plot_demand_by_section(lineas, id_linea, nombre_linea, day_type, n_sections, rango):

    gdf_d0 = lineas[
        (lineas.id_linea == id_linea)
        & (lineas.day_type == day_type)
        & (lineas.n_sections == n_sections)
        & (lineas.sentido == "ida")
    ].copy()

    gdf_d1 = lineas[
        (lineas.id_linea == id_linea)
        & (lineas.day_type == day_type)
        & (lineas.n_sections == n_sections)
        & (lineas.sentido == "vuelta")
    ].copy()
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

    if nombre_linea == "":
        title = f"Id línea {id_linea} - {day_type}\n{rango}"
    else:
        title = f"Línea: {nombre_linea.replace('Línea ', '')} - Id línea: {id_linea} - {day_type}\n{rango}"

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


@st.cache_data
def traigo_nombre_lineas(df):
    return (
        df[(df.nombre_linea.notna()) & (df.nombre_linea != "")]
        .sort_values("nombre_linea")
        .nombre_linea.unique()
    )


# --- Función para levantar tablas SQL y almacenar en session_state ---
def cargar_tabla_sql(tabla_sql, tipo_conexion="dash", query=""):
    if f"{tabla_sql}_{tipo_conexion}" not in st.session_state:
        conn = iniciar_conexion_db(tipo=tipo_conexion)
        try:
            query = query or f"SELECT * FROM {tabla_sql}"
            tabla = pd.read_sql_query(query, conn)
            st.session_state[f"{tabla_sql}_{tipo_conexion}"] = tabla
        except Exception:
            st.error(f"{tabla_sql} no existe")
            st.session_state[f"{tabla_sql}_{tipo_conexion}"] = pd.DataFrame()
        finally:
            conn.close()
    return st.session_state[f"{tabla_sql}_{tipo_conexion}"]


def seleccionar_linea(key_input, key_select):
    texto_a_buscar = st.text_input(
        f"Ingrese el texto a buscar en líneas", key=key_input
    )
    if texto_a_buscar:
        if f"df_filtrado_{texto_a_buscar}" not in st.session_state:
            st.session_state[f"df_filtrado_{texto_a_buscar}"] = metadata_lineas[
                metadata_lineas.apply(
                    lambda row: row.astype(str)
                    .str.contains(texto_a_buscar, case=False, na=False)
                    .any(),
                    axis=1,
                )
            ]
        df_filtrado = st.session_state[f"df_filtrado_{texto_a_buscar}"]

        if not df_filtrado.empty:
            opciones = df_filtrado.apply(
                lambda row: f"{row['nombre_linea']}", axis=1
            ).tolist()
            seleccion_texto = st.selectbox(
                f"Seleccione una línea de colectivo",
                opciones,
                key=key_select,
            )
            df_seleccionado = df_filtrado.iloc[opciones.index(seleccion_texto)]

            st.session_state["nombre_linea"] = df_seleccionado.nombre_linea
            st.session_state["id_linea"] = df_seleccionado.id_linea

        else:
            st.warning("No se encontró ninguna coincidencia.")


st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)

alias_seleccionado = configurar_selector_dia()

st.write(
    "Si recién procesó nuevas líneas con Herramientas interactivas, limpie el caché desde el ícono superior derecho de la pantalla"
)
try:
    # --- Cargar configuraciones y conexiones en session_state ---
    if "configs" not in st.session_state:
        st.session_state.configs = utils.leer_configs_generales()

    configs = st.session_state.configs
    h3_legs_res = configs["resolucion_h3"]
    alias = configs["alias_db_data"]
    use_branches = configs["lineas_contienen_ramales"]
    metadata_lineas = cargar_tabla_sql("metadata_lineas", "insumos")[
        ["id_linea", "nombre_linea"]
    ]
    conn_insumos = iniciar_conexion_db(tipo="insumos")
except ValueError as e:
    st.error(
        f"Falta una base de datos requerida: {e}. \nSe requiere full acceso a Urbantrips para correr esta página"
    )
    st.stop()


# --- Inicializar variables en session_state ---
for var in [
    "id_linea",
    "nombre_linea",
    "day_type_kpi",
    "yr_mo_kpi",
    "secciones",
    "rango",
    "secciones_supply",
    "rango_supply",
]:
    if var not in st.session_state:
        st.session_state[var] = None


col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Periodo")

    kpi_lineas = levanto_tabla_sql_local("basic_kpi_by_line_hr")
    day_type_kpi = col1.selectbox("Tipo de dia  ", options=kpi_lineas.dia.unique())
    # add month and year
    yr_mo_kpi = col1.selectbox(
        "Periodo  ", options=kpi_lineas.yr_mo.unique(), key="year_month"
    )
    st.session_state["day_type_kpi"] = day_type_kpi
    st.session_state["yr_mo_kpi"] = yr_mo_kpi

with col2:
    st.subheader("Línea")
    seleccionar_linea("base_input", "base_select")
    id_linea = st.session_state["id_linea"]
    nombre_linea = st.session_state["nombre_linea"]

with st.expander("Factor de ocupación por horas"):
    nombre_linea_kpi = st.session_state["nombre_linea"]
    id_linea_kpi = st.session_state["id_linea"]
    kpi_stats_line_plot = kpi_lineas[
        (kpi_lineas.id_linea == st.session_state["id_linea"])
        & (kpi_lineas.dia == st.session_state["day_type_kpi"])
        & (kpi_lineas.yr_mo == st.session_state["yr_mo_kpi"])
    ]
    if len(kpi_stats_line_plot) > 0:

        # Grafico Factor de Oocupación
        f, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=kpi_stats_line_plot,
            x="hora",
            y="of",
            color="silver",
            ax=ax,
            label="Factor de ocupación",
        )

        sns.lineplot(
            data=kpi_stats_line_plot,
            x="hora",
            y="veh",
            ax=ax,
            color="Purple",
            label="Oferta - veh/hr",
        )
        sns.lineplot(
            data=kpi_stats_line_plot,
            x="hora",
            y="pax",
            ax=ax,
            color="Orange",
            label="Demanda - pax/hr",
        )

        ax.set_xlabel("Hora")
        ax.set_ylabel("Factor de Ocupación (%)")

        ax.set_title(
            f"Indicadores de oferta y demanda estadarizados\nLínea: {nombre_linea_kpi.replace('Línea ', '')} - Id linea: {id_linea_kpi} - {day_type_kpi}",
            fontdict={"size": 12},
        )

        # Add a footnote below and to the right side of the chart
        note = """
            *Los indicadores de Oferta y Demanda se estandarizaron para que
            coincidan con el eje de Factor de Ocupación
            """
        ax_note = ax.annotate(
            note,
            xy=(0, -0.22),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=7,
        )
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.left.set_position(("outward", 10))
        ax.spines.bottom.set_position(("outward", 10))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # Put a legend to the right of the current axis
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        st.pyplot(f)
    else:
        st.write("No hay datos para mostrar")

with st.expander("Demanda por segmento de recorrido"):
    st.subheader("Parámetros")

    col1, col2 = st.columns([1, 1])

    lineas = levanto_tabla_sql_local("ocupacion_por_linea_tramo")

    lineas = lineas[
        (lineas.id_linea == id_linea) & (lineas.yr_mo == st.session_state[f"yr_mo_kpi"])
    ]

    lineas.loc[lineas["hour_min"].notna(), "rango"] = (
        "de "
        + lineas.loc[lineas["hour_min"].notna(), "hour_min"].astype(int).astype(str)
        + " a "
        + lineas.loc[lineas["hour_max"].notna(), "hour_max"].astype(int).astype(str)
        + " hrs"
    )

    lineas.loc[lineas["hour_min"].isna(), "rango"] = "Todo el dia"

    n_sections = col1.selectbox("Secciones", options=lineas.n_sections.unique())

    rango = col2.selectbox("Rango horario", options=lineas.rango.unique())

    st.session_state["secciones"] = n_sections
    st.session_state["rango"] = rango
    lineas = lineas[
        (lineas.day_type == st.session_state["day_type_kpi"])
        & (lineas.n_sections == st.session_state["secciones"])
        & (lineas.rango == st.session_state["rango"])
    ]

    if len(lineas) > 0:
        if st.checkbox("Mostrar datos", value=False):
            st.write(lineas)

        f_lineas = plot_demand_by_section(
            lineas,
            st.session_state["id_linea"],
            st.session_state["nombre_linea"],
            st.session_state["day_type_kpi"],
            st.session_state["secciones"],
            st.session_state["rango"],
        )
        st.pyplot(f_lineas)
    else:
        st.write("No hay datos para mostrar")

with st.expander("Líneas de deseo por linea"):

    custom_query = """
    select m.*, co.x as lon_o, co.y as lat_o,  cd.x as lon_d, cd.y as lat_d
    from matrices_linea m
    left join matrices_linea_carto co
    on m.id_linea = co.id_linea
    and m.n_sections = co.n_sections
    and m.Origen = co.section_id
    left join matrices_linea_carto cd
    on m.id_linea = cd.id_linea
    and m.n_sections = cd.n_sections
    and m.Destino = cd.section_id 
    where lon_o is not NULL 
    and lat_o is not NULL ;
    """
    matriz = levanto_tabla_sql_local(
        tabla_sql="matrices_linea",
        query=custom_query,
    )

    matriz = create_linestring_od(matriz)

    matriz.loc[matriz["hour_min"].notna(), "rango"] = (
        "de "
        + matriz.loc[matriz["hour_min"].notna(), "hour_min"].astype(int).astype(str)
        + " a "
        + matriz.loc[matriz["hour_max"].notna(), "hour_max"].astype(int).astype(str)
        + " hrs"
    )

    matriz.loc[matriz["hour_min"].isna(), "rango"] = "Todo el dia"

    matriz = matriz[matriz.id_linea == st.session_state["id_linea"]]
    matriz = matriz[matriz.yr_mo == st.session_state["yr_mo_kpi"]]

    matriz = matriz[matriz.day_type == st.session_state["day_type_kpi"]]
    matriz = matriz[matriz.n_sections == st.session_state["secciones"]]
    matriz = matriz[matriz.rango == st.session_state["rango"]]

    if len(matriz) > 0:
        if st.checkbox("Mostrar datos ", value=False, key="mostrar_datos2"):
            st.write(matriz)

        k_jenks = st.slider("Cantidad de grupos", min_value=1, max_value=5, value=5)
        st.text(f"Hay un total de {matriz.legs.sum()} etapas")

        map = crear_mapa_folium(matriz, cmap="BuPu", var_fex="legs", k_jenks=k_jenks)

        st_map = st_folium(map, width=900, height=700)
    else:
        st.write("No hay datos para mostrar")

with st.expander("Matriz OD por linea"):

    matriz = levanto_tabla_sql_local("matrices_linea")
    # nl3 = traigo_nombre_lineas(matriz[["id_linea", "nombre_linea"]])

    matriz.loc[matriz["hour_min"].notna(), "rango"] = (
        "de "
        + matriz.loc[matriz["hour_min"].notna(), "hour_min"].astype(int).astype(str)
        + " a "
        + matriz.loc[matriz["hour_max"].notna(), "hour_max"].astype(int).astype(str)
        + " hrs"
    )

    matriz.loc[matriz["hour_min"].isna(), "rango"] = "Todo el dia"
    matriz = matriz[matriz.id_linea == st.session_state["id_linea"]]
    matriz = matriz[matriz.yr_mo == st.session_state[f"yr_mo_kpi"]]
    matriz = matriz[matriz.day_type == st.session_state["day_type_kpi"]]
    matriz = matriz[matriz.n_sections == st.session_state["secciones"]]
    matriz = matriz[matriz.rango == st.session_state["rango"]]

    if len(matriz) > 0:
        if st.checkbox("Mostrar datos ", value=False, key="mostrar_datos3"):
            st.write(matriz)
        if st.checkbox("Normalizar", value=True):
            values = "prop"
        else:
            values = "legs"

        od_heatmap = matriz.pivot_table(
            values=values, index="Origen", columns="Destino"
        )

        fig = px.imshow(
            od_heatmap,
            text_auto=True,
            color_continuous_scale="Blues",
        )

        fig.update_coloraxes(showscale=True)

        if len(od_heatmap) <= 20:
            fig.update_layout(width=800, height=800)
        elif (len(od_heatmap) > 20) & (len(od_heatmap) <= 40):
            fig.update_layout(width=1000, height=1000)
        elif len(od_heatmap) > 40:
            fig.update_layout(width=1200, height=1200)

        st.plotly_chart(fig)

    else:
        st.write("No hay datos para mostrar")

    zonas = levanto_tabla_sql("matrices_linea_carto")

    zonas = zonas.loc[
        (zonas.id_linea == id_linea)
        & (zonas.n_sections == st.session_state["secciones"]),
        :,
    ]

    col1, col2 = st.columns([1, 4])

    if col1.checkbox("Mostrar zonificacion"):

        if len(zonas) > 0:

            # Create a folium map centered on the data
            map_center = [
                zonas.geometry.centroid.y.mean(),
                zonas.geometry.centroid.x.mean(),
            ]

            fig = Figure(width=800, height=800)
            m = folium.Map(location=map_center, zoom_start=10, tiles="cartodbpositron")

            # Add GeoDataFrame to the map
            folium.GeoJson(zonas).add_to(m)

            for idx, row in zonas.iterrows():
                # Replace 'column_name' with the name of the column containing the detail
                detail = f"Sección {row['section_id']}"
                point = [
                    row["geometry"].representative_point().y,
                    row["geometry"].representative_point().x,
                ]
                marker = folium.CircleMarker(
                    location=point,
                    popup=detail,
                    color="black",
                    radius=2,
                    fill=True,
                    fill_color="black",
                    fill_opacity=1,
                )
                marker.add_to(m)

            # Display the map using folium_static
            with col2:
                folium_static(m)

with st.expander("Oferta por segmento de recorrido"):

    st.subheader("Parámetros")
    col1, col2 = st.columns([1, 1])

    lineas = levanto_tabla_sql_local("supply_stats_by_section_id")
    lineas = lineas[
        (lineas.id_linea == id_linea) & (lineas.yr_mo == st.session_state[f"yr_mo_kpi"])
    ]

    lineas.loc[lineas["hour_min"].notna(), "rango"] = (
        "de "
        + lineas.loc[lineas["hour_min"].notna(), "hour_min"].astype(int).astype(str)
        + " a "
        + lineas.loc[lineas["hour_max"].notna(), "hour_max"].astype(int).astype(str)
        + " hrs"
    )

    lineas.loc[lineas["hour_min"].isna(), "rango"] = "Todo el dia"

    n_sections = col1.selectbox(
        "Secciones ", options=lineas.n_sections.unique(), key="sections_supply"
    )

    rango = col2.selectbox(
        "Rango horario ", options=lineas.rango.unique(), key="range_supply"
    )

    st.session_state["secciones_supply"] = n_sections
    st.session_state["rango_supply"] = rango

    lineas = lineas[
        (lineas.day_type == st.session_state["day_type_kpi"])
        & (lineas.n_sections == st.session_state["secciones_supply"])
        & (lineas.rango == st.session_state["rango_supply"])
    ]

    if len(lineas) > 0:

        if st.checkbox(
            "Mostrar datos", value=False, key="display_supply_stats_by_section_id"
        ):
            st.write(lineas)

        f_lineas_speed = plot_speed_by_section(
            lineas,
            st.session_state["id_linea"],
            st.session_state["nombre_linea"],
            st.session_state["day_type_kpi"],
            st.session_state["secciones"],
            st.session_state["rango"],
        )
        st.pyplot(f_lineas_speed)

        f_lineas_frequency = plot_frequency_by_section(
            lineas,
            st.session_state["id_linea"],
            st.session_state["nombre_linea"],
            st.session_state["day_type_kpi"],
            st.session_state["secciones"],
            st.session_state["rango"],
        )
        st.pyplot(f_lineas_frequency)

    else:
        st.write("No hay datos para mostrar")
