import seaborn as sns
import pandas as pd
import contextily as cx
import os
from requests.exceptions import ConnectionError as r_ConnectionError
from PIL import UnidentifiedImageError
from urbantrips.kpi import kpi
import matplotlib.pyplot as plt

from urbantrips.viz.viz import (
    standarize_size,
    create_squared_polygon,
    get_branch_geoms_from_line,
)
from urbantrips.geo import geo
from urbantrips.utils.utils import (
    leer_configs_generales,
    traigo_db_path,
    iniciar_conexion_db,
    leer_alias,
    duracion,
    create_line_ids_sql_filter,
)


@duracion
def visualize_route_section_supply_data(
    line_ids=False,
    hour_range=False,
    day_type="weekday",
    n_sections=10,
    section_meters=None,
    factor=500,
    factor_min=1,
    save_gdf=False,
):
    """
    Visualize the average speed and frequency of buses per route section and direction

    Parameters
    ----------
    line_ids : int, list of ints or bool
        route id present in the ocupacion_por_linea_tramo table.
    hour_range : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24.
    day_type: str
        type of day. It can take `weekday`, `weekend` or a specific
        day in format 'YYYY-MM-DD'
    n_sections: int
        number of sections to split the route geom
    section_meters: int
        section lenght in meters to split the route geom. If specified,
        this will be used instead of n_sections.
    factor: int
        scaling factor to use for line width to plot section load
    factor_min: int
        minimum width of linea for low section loads to be displayed

    """
    sns.set_style("whitegrid")

    route_section_supply = get_route_section_supply_data(
        line_ids=line_ids,
        hour_range=hour_range,
        day_type=day_type,
        n_sections=n_sections,
        section_meters=section_meters,
    )

    # Create a speed viz for each route
    route_section_supply.groupby(["id_linea", "yr_mo"]).apply(
        viz_route_section_speed,
        factor=factor,
        factor_min=factor_min,
        return_gdfs=False,
        save_gdf=save_gdf,
    )

    # Create a frequency viz for each route
    route_section_supply.groupby(["id_linea", "yr_mo"]).apply(
        viz_route_section_frequency,
        factor=factor,
        factor_min=factor_min,
        return_gdfs=False,
        save_gdf=save_gdf,
    )


def viz_route_section_frequency(
    df, factor=500, factor_min=10, return_gdfs=False, save_gdf=False
):
    """
    Plots and saves a section frequency for a given route

    Parameters
    ----------
    df: pandas.DataFrame
        table for a given route in section supply data db table
    route geom: geopandas.GeoSeries
        route geoms with id_route as index
    factor: int
        scaling factor to use for line width to plot section speed
    factor_min: int
        minimum width of linea for low section speed to be displayed
    return_gdfs: bool
        if functions will return section load geodataframes per direction

    Returns
    -------
    gdf_d0 : geopandas.GeoDataFrame
        geodataframe with section load data and sections geoms.

    gdf_d1 : geopandas.GeoDataFrame
        geodataframe with section load data and sections geoms.
    """
    conn_insumos = iniciar_conexion_db(tipo="insumos")
    indicator_col = "frequency_interval"

    line_id = df.id_linea.unique().item()

    n_sections = df.n_sections.unique().item()
    mes = df.yr_mo.unique().item()
    day = df["day_type"].unique().item()

    # get line name from metadata
    s = f"select nombre_linea from metadata_lineas" + f" where id_linea = {line_id};"
    id_linea_str = pd.read_sql(s, conn_insumos)

    if len(id_linea_str) > 0:
        id_linea_str = id_linea_str.nombre_linea.item()
    else:
        id_linea_str = ""

    # Turn day type into printable string
    if day == "weekend":
        day_str = "Fin de semana"
    elif day == "weekday":
        day_str = "Dia habil"
    else:
        day_str = day

    section_ids = df.section_id.unique()
    print("Produciendo grafico de estimación de frecuencias por tramos", line_id)

    df["buff_factor"] = (factor + factor_min) / 2
    # standarize_size(series=df["frequency"], min_size=factor_min, max_size=factor)
    cols = [
        "id_linea",
        "yr_mo",
        "day_type",
        "n_sections",
        "sentido",
        "section_id",
        "hour_min",
        "hour_max",
        "n_vehicles",
        "avg_speed",
        "median_speed",
        "frequency",
        "frequency_interval",
        "buff_factor",
    ]

    df_d0 = df.loc[df.sentido == "ida", cols]
    df_d1 = df.loc[df.sentido == "vuelta", cols]
    # get data
    sections_geoms_q = f"""
    select * from routes_section_id_coords 
    where id_linea = {line_id}
    and n_sections = {n_sections}
    """
    conn_insumos = iniciar_conexion_db(tipo="insumos")
    sections_geoms = pd.read_sql(sections_geoms_q, conn_insumos)
    sections_geoms = geo.create_sections_geoms(sections_geoms, buffer_meters=False)

    # Arrows
    flecha_ida_wgs84 = sections_geoms.loc[
        sections_geoms.section_id == sections_geoms.section_id.min(), "geometry"
    ]
    flecha_ida_wgs84 = list(flecha_ida_wgs84.item().coords)
    flecha_ida_inicio_wgs84 = flecha_ida_wgs84[0]

    flecha_vuelta_wgs84 = sections_geoms.loc[
        sections_geoms.section_id == max(sections_geoms.section_id), "geometry"
    ]
    flecha_vuelta_wgs84 = list(flecha_vuelta_wgs84.item().coords)
    flecha_vuelta_fin_wgs84 = flecha_vuelta_wgs84[1]

    # Use a projected crs in meters
    epsg = geo.get_epsg_m()
    sections_geoms = sections_geoms.to_crs(epsg=epsg)

    gdf_d0 = sections_geoms.merge(
        df_d0, on=["id_linea", "n_sections", "section_id"], how="left"
    )

    gdf_d1 = sections_geoms.merge(
        df_d1, on=["id_linea", "n_sections", "section_id"], how="left"
    )

    # creando buffers en base a
    gdf_d0["geometry"] = gdf_d0.geometry.buffer(gdf_d0.buff_factor)
    gdf_d1["geometry"] = gdf_d1.geometry.buffer(gdf_d1.buff_factor)

    # creating plot
    f = plt.figure(tight_layout=True, figsize=(20, 10))
    ax1 = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2)

    font_dicc = {"fontsize": 18, "fontweight": "bold"}

    # create a squared box
    minx, miny, maxx, maxy = gdf_d0.total_bounds
    box = create_squared_polygon(minx, miny, maxx, maxy, epsg)
    box.plot(ax=ax1, color="#ffffff00")
    box.plot(ax=ax2, color="#ffffff00")

    # get branches' geoms
    branch_geoms = get_branch_geoms_from_line(id_linea=line_id)

    if branch_geoms is not None:
        branch_geoms = branch_geoms.to_crs(epsg=epsg)
        branch_geoms.plot(ax=ax1, color="Purple", alpha=0.4, linestyle="dashed")
        branch_geoms.plot(ax=ax2, color="Orange", alpha=0.4, linestyle="dashed")

    sections_geoms.plot(ax=ax1, color="black")
    sections_geoms.plot(ax=ax2, color="black")

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

    if not df.hour_min.isna().all():
        from_hr = df.hour_min.unique()[0]
        to_hr = df.hour_max.unique()[0]
        hr_str = f" {from_hr}-{to_hr} hrs"
        hour_range = [from_hr, to_hr]
    else:
        hr_str = ""
        hour_range = False

    title = "Estimación de frecuencia por segmentos del recorrido"
    title = (
        title
        + hr_str
        + " - "
        + day_str
        + "-"
        + mes
        + "-"
        + f" {id_linea_str} (id_linea: {line_id})"
    )
    f.suptitle(title)

    # Matching bar plot with route direction
    flecha_eo_xy = (0.4, 1.1)
    flecha_eo_text_xy = (0.05, 1.1)
    flecha_oe_xy = (0.6, 1.1)
    flecha_oe_text_xy = (0.95, 1.1)

    labels_eo = [""] * len(section_ids)
    labels_eo[0] = "INICIO"
    labels_eo[-1] = "FIN"
    labels_oe = [""] * len(section_ids)
    labels_oe[-1] = "INICIO"
    labels_oe[0] = "FIN"

    # For direction 0, get the last section of the route geom
    flecha_ida = sections_geoms.loc[
        sections_geoms.section_id == sections_geoms.section_id.max(), "geometry"
    ]
    flecha_ida = list(flecha_ida.item().coords)
    flecha_ida_inicio = flecha_ida[1]
    flecha_ida_fin = flecha_ida[0]

    # For direction 1, get the first section of the route geom
    flecha_vuelta = sections_geoms.loc[
        sections_geoms.section_id == sections_geoms.section_id.min(), "geometry"
    ]
    flecha_vuelta = list(flecha_vuelta.item().coords)
    # invert the direction of the arrow
    flecha_vuelta_inicio = flecha_vuelta[0]
    flecha_vuelta_fin = flecha_vuelta[1]

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

    prov = cx.providers.CartoDB.Positron
    try:
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string(), source=prov)
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string(), source=prov)
    except (UnidentifiedImageError, ValueError):
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string())
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string())
    except r_ConnectionError:
        pass

    alias = leer_alias()

    for frm in ["png", "pdf"]:
        archivo = f"{alias}_{mes}({day_str})_segmentos_id_linea_"
        archivo = archivo + f"{line_id}_frequency_{hr_str}_{n_sections}_sections.{frm}"
        db_path = os.path.join("resultados", frm, archivo)
        f.savefig(db_path, dpi=300)
    plt.close(f)

    if save_gdf:
        gdf_d0 = gdf_d0.to_crs(epsg=4326)
        gdf_d1 = gdf_d1.to_crs(epsg=4326)

        f_0 = f"segmentos_id_linea_{alias}_{mes}({day_str})_{line_id}_median_speed_{hr_str}_0.geojson"
        f_1 = f"segmentos_id_linea_{alias}_{mes}({day_str})_{line_id}_median_speed_{hr_str}_1.geojson"

        db_path_0 = os.path.join("resultados", "geojson", f_0)
        db_path_1 = os.path.join("resultados", "geojson", f_1)

        gdf_d0.to_file(db_path_0, driver="GeoJSON")
        gdf_d1.to_file(db_path_1, driver="GeoJSON")

    if return_gdfs:
        return gdf_d0, gdf_d1


def viz_route_section_speed(
    df, factor=500, factor_min=10, return_gdfs=False, save_gdf=False
):
    """
    Plots and saves a section median speed viz for a given route

    Parameters
    ----------
    df: pandas.DataFrame
        table for a given route in section supply data db table
    route geom: geopandas.GeoSeries
        route geoms with id_route as index
    factor: int
        scaling factor to use for line width to plot section speed
    factor_min: int
        minimum width of linea for low section speed to be displayed
    return_gdfs: bool
        if functions will return section load geodataframes per direction

    Returns
    -------
    gdf_d0 : geopandas.GeoDataFrame
        geodataframe with section load data and sections geoms.

    gdf_d1 : geopandas.GeoDataFrame
        geodataframe with section load data and sections geoms.
    """

    conn_insumos = iniciar_conexion_db(tipo="insumos")
    indicator_col = "speed_interval"

    line_id = df.id_linea.unique().item()
    n_sections = df.n_sections.unique().item()
    mes = df.yr_mo.unique().item()
    day = df["day_type"].unique().item()

    # get line name from metadata
    s = "select nombre_linea from metadata_lineas" + f" where id_linea = {line_id};"
    id_linea_str = pd.read_sql(s, conn_insumos)

    if len(id_linea_str) > 0:
        id_linea_str = id_linea_str.nombre_linea.item()
    else:
        id_linea_str = ""

    # Turn day type into printable string
    if day == "weekend":
        day_str = "Fin de semana"
    elif day == "weekday":
        day_str = "Dia habil"
    else:
        day_str = day

    section_ids = df.section_id.unique()
    print("Produciendo grafico de velocidad promedio por tramos", line_id)
    df["buff_factor"] = (factor + factor_min) / 2
    # standarize_size(series=df["frequency"], min_size=factor_min, max_size=factor)

    cols = [
        "id_linea",
        "yr_mo",
        "day_type",
        "n_sections",
        "sentido",
        "section_id",
        "hour_min",
        "hour_max",
        "n_vehicles",
        "avg_speed",
        "median_speed",
        "speed_interval",
        "frequency",
        "frequency_interval",
        "buff_factor",
    ]

    df_d0 = df.loc[df.sentido == "ida", cols]
    df_d1 = df.loc[df.sentido == "vuelta", cols]
    # get data
    sections_geoms_q = f"""
    select * from routes_section_id_coords 
    where id_linea = {line_id}
    and n_sections = {n_sections}
    """
    conn_insumos = iniciar_conexion_db(tipo="insumos")
    sections_geoms = pd.read_sql(sections_geoms_q, conn_insumos)
    sections_geoms = geo.create_sections_geoms(sections_geoms, buffer_meters=False)

    # Arrows
    flecha_ida_wgs84 = sections_geoms.loc[
        sections_geoms.section_id == sections_geoms.section_id.min(), "geometry"
    ]
    flecha_ida_wgs84 = list(flecha_ida_wgs84.item().coords)
    flecha_ida_inicio_wgs84 = flecha_ida_wgs84[0]

    flecha_vuelta_wgs84 = sections_geoms.loc[
        sections_geoms.section_id == max(sections_geoms.section_id), "geometry"
    ]
    flecha_vuelta_wgs84 = list(flecha_vuelta_wgs84.item().coords)
    flecha_vuelta_fin_wgs84 = flecha_vuelta_wgs84[1]

    # Use a projected crs in meters
    epsg = geo.get_epsg_m()
    sections_geoms = sections_geoms.to_crs(epsg=epsg)

    gdf_d0 = sections_geoms.merge(
        df_d0, on=["id_linea", "n_sections", "section_id"], how="left"
    )
    gdf_d0[indicator_col] = gdf_d0[indicator_col].fillna(0)

    gdf_d1 = sections_geoms.merge(
        df_d1, on=["id_linea", "n_sections", "section_id"], how="left"
    )
    gdf_d1[indicator_col] = gdf_d1[indicator_col].fillna(0)

    # save data for dashboard
    gdf_d0_dash = gdf_d0.to_crs(epsg=4326).copy()
    gdf_d1_dash = gdf_d1.to_crs(epsg=4326).copy()

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

    # create a squared box
    minx, miny, maxx, maxy = gdf_d0.total_bounds
    box = create_squared_polygon(minx, miny, maxx, maxy, epsg)
    box.plot(ax=ax1, color="#ffffff00")
    box.plot(ax=ax2, color="#ffffff00")

    # get branches' geoms
    branch_geoms = get_branch_geoms_from_line(id_linea=line_id)

    if branch_geoms is not None:
        branch_geoms = branch_geoms.to_crs(epsg=epsg)
        branch_geoms.plot(ax=ax1, color="Purple", alpha=0.4, linestyle="dashed")
        branch_geoms.plot(ax=ax2, color="Orange", alpha=0.4, linestyle="dashed")

    sections_geoms.plot(ax=ax1, color="black")
    sections_geoms.plot(ax=ax2, color="black")

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

    ax1.set_title("IDA", fontdict=font_dicc, y=1.0, pad=-20)
    ax2.set_title("VUELTA", fontdict=font_dicc, y=1.0, pad=-20)

    if not df.hour_min.isna().all():
        from_hr = df.hour_min.unique()[0]
        to_hr = df.hour_max.unique()[0]
        hr_str = f" {from_hr}-{to_hr} hrs"
        hour_range = [from_hr, to_hr]
    else:
        hr_str = ""
        hour_range = False

    title = "Velocidad mediana por segmentos del recorrido"
    title = (
        title
        + hr_str
        + " - "
        + day_str
        + "-"
        + mes
        + "-"
        + f" {id_linea_str} (id_linea: {line_id})"
    )
    f.suptitle(title)

    # Matching bar plot with route direction
    flecha_eo_xy = (0.4, 1.1)
    flecha_eo_text_xy = (0.05, 1.1)
    flecha_oe_xy = (0.6, 1.1)
    flecha_oe_text_xy = (0.95, 1.1)

    labels_eo = [""] * len(section_ids)
    labels_eo[0] = "INICIO"
    labels_eo[-1] = "FIN"
    labels_oe = [""] * len(section_ids)
    labels_oe[-1] = "INICIO"
    labels_oe[0] = "FIN"

    # check if route geom is drawn from west to east
    geom_dir_east = flecha_ida_inicio_wgs84[0] < flecha_vuelta_fin_wgs84[0]

    # Set arrows in barplots based on reout geom direction
    if geom_dir_east:

        flecha_ida_xy = flecha_eo_xy
        flecha_ida_text_xy = flecha_eo_text_xy
        labels_ida = labels_eo

        flecha_vuelta_xy = flecha_oe_xy
        flecha_vuelta_text_xy = flecha_oe_text_xy
        labels_vuelta = labels_oe

        # direction 0 east to west
        df_d0 = df_d0.sort_values("section_id", ascending=True)
        df_d1 = df_d1.sort_values("section_id", ascending=True)

    else:
        flecha_ida_xy = flecha_oe_xy
        flecha_ida_text_xy = flecha_oe_text_xy
        labels_ida = labels_oe

        flecha_vuelta_xy = flecha_eo_xy
        flecha_vuelta_text_xy = flecha_eo_text_xy
        labels_vuelta = labels_eo

        df_d0 = df_d0.sort_values("section_id", ascending=False)
        df_d1 = df_d1.sort_values("section_id", ascending=False)

    bar_variable = "avg_speed"
    sns.barplot(
        data=df_d0,
        x="section_id",
        y=bar_variable,
        ax=ax3,
        color="Grey",
        order=df_d0.section_id.values,
    )

    sns.barplot(
        data=df_d1,
        x="section_id",
        y=bar_variable,
        ax=ax4,
        color="Grey",
        order=df_d1.section_id.values,
    )

    # Axis
    ax3.set_xticklabels(labels_ida)
    ax4.set_xticklabels(labels_vuelta)

    y_axis_lable = "Velocidad mediana (kmh)"
    ax3.set_ylabel(y_axis_lable)
    ax3.set_xlabel("")

    ax4.get_yaxis().set_visible(False)

    ax4.set_ylabel("")
    ax4.set_xlabel("")
    max_y_barplot = max(df_d0[bar_variable].max(), df_d1[bar_variable].max())

    ax3.set_ylim(0, max_y_barplot)
    ax4.set_ylim(0, max_y_barplot)

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax4.spines.left.set_visible(False)
    ax4.spines.right.set_visible(False)
    ax4.spines.top.set_visible(False)

    ax3.grid(False)
    ax4.grid(False)
    # For direction 0, get the last section of the route geom
    flecha_ida = sections_geoms.loc[
        sections_geoms.section_id == sections_geoms.section_id.max(), "geometry"
    ]
    flecha_ida = list(flecha_ida.item().coords)
    flecha_ida_inicio = flecha_ida[1]
    flecha_ida_fin = flecha_ida[0]

    # For direction 1, get the first section of the route geom
    flecha_vuelta = sections_geoms.loc[
        sections_geoms.section_id == sections_geoms.section_id.min(), "geometry"
    ]
    flecha_vuelta = list(flecha_vuelta.item().coords)
    # invert the direction of the arrow
    flecha_vuelta_inicio = flecha_vuelta[0]
    flecha_vuelta_fin = flecha_vuelta[1]

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

    prov = cx.providers.CartoDB.Positron
    try:
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string(), source=prov)
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string(), source=prov)
    except (UnidentifiedImageError, ValueError):
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string())
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string())
    except r_ConnectionError:
        pass

    alias = leer_alias()

    for frm in ["png", "pdf"]:
        archivo = f"{alias}_{mes}({day_str})_segmentos_id_linea_"
        archivo = (
            archivo + f"{line_id}_median_speed_{hr_str}_{n_sections}_sections.{frm}"
        )
        db_path = os.path.join("resultados", frm, archivo)
        f.savefig(db_path, dpi=300)
    plt.close(f)
    # Save to dash db

    gdf_d0_dash["wkt"] = gdf_d0_dash.geometry.to_wkt()
    gdf_d0_dash["sentido"] = (
        gdf_d0_dash["sentido"].fillna(method="ffill").fillna(method="bfill")
    )
    gdf_d1_dash["wkt"] = gdf_d1_dash.geometry.to_wkt()
    gdf_d1_dash["sentido"] = (
        gdf_d1_dash["sentido"].fillna(method="ffill").fillna(method="bfill")
    )

    gdf_d_dash = pd.concat([gdf_d0_dash, gdf_d1_dash], ignore_index=True)
    gdf_d_dash["nombre_linea"] = id_linea_str

    cols = [
        "id_linea",
        "yr_mo",
        "nombre_linea",
        "day_type",
        "n_sections",
        "sentido",
        "section_id",
        "hour_min",
        "hour_max",
        "n_vehicles",
        "avg_speed",
        "median_speed",
        "frequency",
        "frequency_interval",
        "buff_factor",
        "wkt",
    ]

    gdf_d_dash = gdf_d_dash.reindex(columns=cols)

    # delete old data
    delete_df = sections_geoms.reindex(
        columns=["id_linea", "n_sections"]
    ).drop_duplicates()

    kpi.delete_old_route_section_load_data(
        route_geoms=delete_df,
        hour_range=hour_range,
        day_type=day,
        yr_mos=[mes],
        db_type="dash",
    )

    conn_dash = iniciar_conexion_db(tipo="dash")

    for var in ["yr_mo", "day_type", "hour_min", "hour_max"]:
        gdf_d_dash[var] = gdf_d_dash[var].fillna(method="ffill").fillna(method="bfill")

    gdf_d_dash.to_sql(
        "supply_stats_by_section_id", conn_dash, if_exists="append", index=False
    )
    conn_dash.close()

    if save_gdf:
        gdf_d0 = gdf_d0.to_crs(epsg=4326)
        gdf_d1 = gdf_d1.to_crs(epsg=4326)

        f_0 = f"segmentos_id_linea_{alias}_{mes}({day_str})_{line_id}_median_speed_{hr_str}_0.geojson"
        f_1 = f"segmentos_id_linea_{alias}_{mes}({day_str})_{line_id}_median_speed_{hr_str}_1.geojson"

        db_path_0 = os.path.join("resultados", "geojson", f_0)
        db_path_1 = os.path.join("resultados", "geojson", f_1)

        gdf_d0.to_file(db_path_0, driver="GeoJSON")
        gdf_d1.to_file(db_path_1, driver="GeoJSON")

    if return_gdfs:
        return gdf_d0, gdf_d1


def get_route_section_supply_data(
    line_ids=False,
    hour_range=False,
    day_type="weekday",
    n_sections=10,
    section_meters=None,
):
    """
    Get the supply stats per route section data

    Parameters
    ----------
    line_ids : int, list of ints or bool
        route id or list of route ids present in the legs dataset. Route
        section load will be computed for that subset of lines. If False, it
        will run with all routes.
    hour_range : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24. Route section
        load will be computed for legs happening within tat time range.
        If False it won't filter by hour.
    day_type: str
        type of day. It can take `weekday`, `weekend` or a specific
        day in format 'YYYY-MM-DD'
    n_sections: int
        number of sections to split the route geom
    section_meters: int
        section lenght in meters to split the route geom. If specified,
        this will be used instead of n_sections.

    Returns
    -------
    pandas.Data.Frame
        dataframe with load per section per route
    """

    conn_data = iniciar_conexion_db(tipo="data")

    q = load_route_section_supply_data_q(
        line_ids=line_ids,
        hour_range=hour_range,
        day_type=day_type,
        n_sections=n_sections,
        section_meters=section_meters,
    )

    # Read data from section load table
    section_load_data = pd.read_sql(q, conn_data)

    conn_data.close()

    if len(section_load_data) == 0:
        print("No hay datos de oferta para estos parametros.")
        print("Ejecurtar supply_kpi.compute_route_section_supply()")
        print(
            " id_linea:",
            line_ids,
            " rango_hrs:",
            hour_range,
            " n_sections:",
            n_sections,
            " section_meters:",
            section_meters,
            " day_type:",
            day_type,
        )

    return section_load_data


def load_route_section_supply_data_q(
    line_ids, hour_range, day_type, n_sections, section_meters
):
    """
    Creates a query that gets route section supply data from the db
    for a specific set of lineas, hours, section meters and day type

    Parameters
    ----------
    line_ids : int, list of ints or bool
        route id or list of route ids present in the legs dataset. Route
        section load will be computed for that subset of lines. If False, it
        will run with all routes.
    hour_range : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24. Route section
        load will be computed for legs happening within tat time range.
        If False it won't filter by hour.
    day_type: str
        type of day on which the section load is to be computed. It can take
        `weekday`, `weekend` or a specific day in format 'YYYY-MM-DD'
    n_sections: int
        number of sections to split the route geom
    section_meters: int
        section lenght in meters to split the route geom. If specified,
        this will be used instead of n_sections.
    Returns
    -------
    str
        query that gets data

    """

    line_ids_where = create_line_ids_sql_filter(line_ids)

    q_main_data = """
    select *
    from supply_stats_by_section_id
    """
    q_main_data = q_main_data + line_ids_where

    # hour range filter
    if hour_range:
        hora_min_filter = f"= {hour_range[0]}"
        hora_max_filter = f"= {hour_range[1]}"
    else:
        hora_min_filter = "is NULL"
        hora_max_filter = "is NULL"

    q_main_data = (
        q_main_data
        + f"""
        and hour_min {hora_min_filter}
        and hour_max {hora_max_filter}
        and day_type = '{day_type}'
        """
    )

    if section_meters:
        q_main_data = q_main_data + f" and section_meters = {section_meters}"

    else:
        q_main_data = q_main_data + f" and n_sections = {n_sections}"

    q_main_data = q_main_data + ";"
    return q_main_data
