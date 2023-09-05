import pandas as pd
import numpy as np
import os
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString, Polygon
import h3
import mapclassify
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from PIL import UnidentifiedImageError
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import contextily as ctx
from IPython.display import display
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.collections import QuadMesh
from pathlib import Path
from matplotlib import colors as mcolors
from matplotlib.text import Text
from mycolorpy import colorlist as mcp
from requests.exceptions import ConnectionError as r_ConnectionError

from urbantrips.kpi import kpi
from urbantrips.carto import carto
from urbantrips.geo import geo
from urbantrips.geo.geo import (
    normalizo_lat_lon, crear_linestring)
from urbantrips.utils.utils import (
    leer_configs_generales,
    traigo_db_path,
    iniciar_conexion_db,
    leer_alias,
    duracion)


def plotear_recorrido_lowess(id_linea, etapas, recorridos_lowess, alias):
    """
    Esta funcion toma un id_linea, un df de etapas, un gdf de recorridos_lowess
    y un alias y produce una viz de las etapas y el recorrido
    """
    e = etapas.loc[etapas.id_linea == id_linea, :]
    r = recorridos_lowess.loc[recorridos_lowess.id_linea == id_linea, :]

    if (len(e) > 0) & (len(r) > 0):
        try:
            fig, ax = plt.subplots(figsize=(3, 3), dpi=150)

            ax.scatter(e.longitud, e.latitud, color='orange', s=.3)
            r.plot(color='black', lw=.8, legend=False, ax=ax)

            ax.set_title(f'Linea {id_linea}', fontsize=6)
            ax.axis('off')

            db_path = os.path.join("resultados", "png",
                                   f"{alias}linea_{id_linea}.png")

            fig.savefig(db_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        except (AttributeError, ValueError):
            pass

    else:
        print(f"No se pudo producir un grafico para el id_linea {id_linea}")


@duracion
def visualize_route_section_load(id_linea=False, rango_hrs=False,
                                 day_type='weekday',
                                 n_sections=10, section_meters=None,
                                 indicador='cantidad_etapas', factor=1,
                                 factor_min=50,
                                 save_gdf=False):
    """
    Visualize the load per route section data per route

    Parameters
    ----------
    id_linea : int, list of ints or bool
        route id present in the ocupacion_por_linea_tramo table.
    rango_hrs : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24.
    day_type: str
        type of day. It can take `weekday`, `weekend` or a specific
        day in format 'YYYY-MM-DD'
    n_sections: int
        number of sections to split the route geom
    section_meters: int
        section lenght in meters to split the route geom. If specified,
        this will be used instead of n_sections.
    indicator: str
        Tipe of section load to display. 'cantidad_etapas' (amount of legs)
        or `prop_etapas` (proportion of legs)
    factor: int
        scaling factor to use for line width to plot section load
    factor_min: int
        minimum width of linea for low section loads to be displayed

    """
    sns.set_style("whitegrid")

    if id_linea:

        if type(id_linea) == int:
            id_linea = [id_linea]

    table = get_route_section_load(
        id_linea=id_linea,
        rango_hrs=rango_hrs,
        day_type=day_type,
        n_sections=n_sections,
        section_meters=section_meters)

    # Create a viz for each route
    table.groupby('id_linea').apply(
        viz_etapas_x_tramo_recorrido,
        indicator=indicador,
        factor=factor,
        factor_min=factor_min,
        return_gdfs=False,
        save_gdf=save_gdf,
    )


def get_route_section_load(id_linea=False, rango_hrs=False, day_type='weekday',
                           n_sections=10, section_meters=None,):
    """
    Get the load per route section data

    Parameters
    ----------
    id_linea : int, list of ints or bool
        route id present in the ocupacion_por_linea_tramo table.
    rango_hrs : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24.
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
    table : pandas.Data.Frame
        dataframe with load per section per route

    recorridos : geopandas.GeoDataFrame
        geodataframe with route geoms

    """

    conn_data = iniciar_conexion_db(tipo='data')

    # route id filter
    if id_linea:

        if type(id_linea) == int:
            id_linea = [id_linea]

        lineas_str = ",".join(map(str, id_linea))
    else:
        lineas_str = ''

    # create query to get data from db
    q = load_route_section_load_data_q(
        lineas_str, rango_hrs, n_sections, section_meters, day_type
    )

    # Read data from section load table
    table = pd.read_sql(q, conn_data)

    if len(table) == 0:
        print("No hay datos de carga por tramo para estos parametros.")
        print(" id_linea:", id_linea,
              " rango_hrs:", rango_hrs,
              " n_sections:", n_sections,
              " section_meters:", section_meters,
              " day_type:", day_type)

    return table


def load_route_section_load_data_q(
    lineas_str, rango_hrs, n_sections, section_meters, day_type
):
    """
    Creates a query that gets route section load data from the db
    for a specific set of lineas, hours, section meters and day type

    Parameters
    ----------
    lineas_str : str
        list of lines to query in a string format separated by comma
    rango_hrs : tuple or bool
        tuple holding hourly range (from,to) and from 0 to 24.
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
    str
        query that gets data

    """

    # hour range filter
    if rango_hrs:
        hora_min_filter = f"= {rango_hrs[0]}"
        hora_max_filter = f"= {rango_hrs[1]}"
    else:
        hora_min_filter = "is NULL"
        hora_max_filter = "is NULL"

    q = f"""
        select * from ocupacion_por_linea_tramo
        where hora_min {hora_min_filter}
        and hora_max {hora_max_filter}
        and day_type = '{day_type}'
        """

    if lineas_str != '':
        q = q + f" and id_linea in ({lineas_str})"

    if section_meters:
        q = q + f" and  section_meters = {section_meters}"

    else:
        q = (
            q +
            f" and n_sections = {n_sections} and section_meters is NULL"
        )
    q = q + ";"
    return q


def viz_etapas_x_tramo_recorrido(df,
                                 indicator='cantidad_etapas', factor=1,
                                 factor_min=50, return_gdfs=False,
                                 save_gdf=False):
    """
    Plots and saves a section load viz for a given route

    Parameters
    ----------
    df: pandas.DataFrame
        table for a given route in section load db table
    route geom: geopandas.GeoSeries
        route geoms with id_route as index
    indicator: str
        Tipe of section load to display. 'cantidad_etapas' (amount of legs)
        or `prop_etapas` (proportion of legs)
    factor: int
        scaling factor to use for line width to plot section load
    factor_min: int
        minimum width of linea for low section loads to be displayed
    return_gdfs: bool
        if functions will return section load geodataframes per direction

    Returns
    -------
    gdf_d0 : geopandas.GeoDataFrame
        geodataframe with section load data and sections geoms.

    gdf_d1 : geopandas.GeoDataFrame
        geodataframe with section load data and sections geoms.
    """
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    id_linea = df.id_linea.unique()[0]
    s = f"select nombre_linea from metadata_lineas" +\
        f" where id_linea = {id_linea};"
    id_linea_str = pd.read_sql(s, conn_insumos)

    if len(id_linea_str) > 0:
        id_linea_str = id_linea_str.nombre_linea.item()
    else:
        id_linea_str = ''

    day = df['day_type'].unique().item()

    if day == 'weekend':
        day_str = 'Fin de semana tipo'
    elif day == 'weekday':
        day_str = 'Dia de semana tipo'
    else:
        day_str = day

    section_ids = df.section_id.unique()

    print('Produciendo grafico de ocupacion por tramos', id_linea)

    # set a expansion factor for viz purposes
    df['buff_factor'] = df[indicator]*factor

    # Set a minimum for each section to be displated in map
    df['buff_factor'] = np.where(
        df['buff_factor'] <= factor_min, factor_min, df['buff_factor'])

    cols = ['id_linea', 'day_type', 'n_sections', 'sentido',
            'section_id', 'hora_min', 'hora_max', 'cantidad_etapas',
            'prop_etapas', 'buff_factor']

    df_d0 = df.loc[df.sentido == 'ida', cols]
    df_d1 = df.loc[df.sentido == 'vuelta', cols]

    # Create geoms for route in both directions
    df_geom = df.query("sentido == 'ida'")\
        .sort_values('section_id')\
        .reset_index(drop=True)

    geom = [LineString(
        [[df_geom.loc[i, 'x'], df_geom.loc[i, 'y']],
         [df_geom.loc[i+1, 'x'], df_geom.loc[i+1, 'y']]]
    ) for i in df_geom.index[:-1]]
    gdf = gpd.GeoDataFrame(pd.DataFrame(
        {'section_id': df_geom.section_id.iloc[:-1]}),
        geometry=geom, crs='epsg:4326')

    # Arrows
    flecha_ida_wgs84 = gdf.loc[gdf.section_id == 0.0, 'geometry']
    flecha_ida_wgs84 = list(flecha_ida_wgs84.item().coords)
    flecha_ida_inicio_wgs84 = flecha_ida_wgs84[0]
    flecha_ida_fin_wgs84 = flecha_ida_wgs84[1]

    flecha_vuelta_wgs84 = gdf.loc[gdf.section_id ==
                                  max(gdf.section_id), 'geometry']
    flecha_vuelta_wgs84 = list(flecha_vuelta_wgs84.item().coords)
    flecha_vuelta_inicio_wgs84 = flecha_vuelta_wgs84[0]
    flecha_vuelta_fin_wgs84 = flecha_vuelta_wgs84[1]

    # Use a projected crs in meters
    epsg = geo.get_epsg_m()
    gdf = gdf.to_crs(epsg=epsg)

    gdf_d0 = gdf\
        .merge(df_d0, on='section_id', how='left')\
        .fillna(0)

    gdf_d1 = gdf\
        .merge(df_d1, on='section_id', how='left')\
        .fillna(0)

    # save data for dashboard
    gdf_d0_dash = gdf_d0.to_crs(epsg=4326).copy()
    gdf_d1_dash = gdf_d1.to_crs(epsg=4326).copy()

    # creando buffers en base a
    gdf_d0['geometry'] = gdf_d0.geometry.buffer(gdf_d0.buff_factor)
    gdf_d1['geometry'] = gdf_d1.geometry.buffer(gdf_d1.buff_factor)

    # creating plot
    f = plt.figure(tight_layout=True, figsize=(20, 15))
    gs = f.add_gridspec(nrows=3, ncols=2)
    ax1 = f.add_subplot(gs[0:2, 0])
    ax2 = f.add_subplot(gs[0:2, 1])
    ax3 = f.add_subplot(gs[2, 0])
    ax4 = f.add_subplot(gs[2, 1])

    font_dicc = {'fontsize': 18,
                 'fontweight': 'bold'}

    # create a squared box
    minx, miny, maxx, maxy = gdf_d0.total_bounds
    box = create_squared_polygon(minx, miny, maxx, maxy, epsg)
    box.plot(ax=ax1, color='#ffffff00')
    box.plot(ax=ax2, color='#ffffff00')

    # get branches' geoms
    branch_geoms = get_branch_geoms_from_line(id_linea=id_linea)

    if branch_geoms is not None:
        branch_geoms = branch_geoms.to_crs(epsg=epsg)
        branch_geoms.plot(ax=ax1, color='Purple',
                          alpha=0.4, linestyle='dashed')
        branch_geoms.plot(ax=ax2, color='Orange',
                          alpha=0.4, linestyle='dashed')

    gdf.plot(ax=ax1, color='black')
    gdf.plot(ax=ax2, color='black')

    try:
        gdf_d0.plot(ax=ax1, column=indicator, cmap='BuPu',
                    scheme='fisherjenks', k=5, alpha=.6)
        gdf_d1.plot(ax=ax2, column=indicator, cmap='Oranges',
                    scheme='fisherjenks', k=5, alpha=.6)
    except ValueError:
        gdf_d0.plot(ax=ax1, column=indicator, cmap='BuPu', alpha=.6)
        gdf_d1.plot(ax=ax2, column=indicator, cmap='Oranges', alpha=.6)

    ax1.set_axis_off()
    ax2.set_axis_off()

    ax1.set_title('IDA', fontdict=font_dicc)
    ax2.set_title('VUELTA', fontdict=font_dicc)

    # Set title and plot axis
    if indicator == 'cantidad_etapas':
        title = 'Segmentos del recorrido - Cantidad de etapas'
        y_axis_lable = 'Cantidad de etapas por sentido'
    elif indicator == 'prop_etapas':
        title = 'Segmentos del recorrido - Porcentaje de etapas totales'
        y_axis_lable = 'Porcentaje del total de etapas'
    else:
        raise Exception(
            "Indicador debe ser 'cantidad_etapas' o 'prop_etapas'")

    if not df.hora_min.isna().all():
        from_hr = df.hora_min.unique()[0]
        to_hr = df.hora_max.unique()[0]
        hr_str = f' {from_hr}-{to_hr} hrs'
    else:
        hr_str = ''

    title = title + hr_str + ' - ' + day_str + \
        f" {id_linea_str} (id_linea: {id_linea})"
    f.suptitle(title, fontsize=18)

    # Matching bar plot with route direction
    flecha_eo_xy = (0.4, 1.1)
    flecha_eo_text_xy = (0.05, 1.1)
    flecha_oe_xy = (0.6, 1.1)
    flecha_oe_text_xy = (0.95, 1.1)

    labels_eo = [''] * len(section_ids)
    labels_eo[0] = 'INICIO'
    labels_eo[-1] = 'FIN'
    labels_oe = [''] * len(section_ids)
    labels_oe[-1] = 'INICIO'
    labels_oe[0] = 'FIN'

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
        df_d0 = df_d0.sort_values('section_id', ascending=True)
        df_d1 = df_d1.sort_values('section_id', ascending=True)

    else:
        flecha_ida_xy = flecha_oe_xy
        flecha_ida_text_xy = flecha_oe_text_xy
        labels_ida = labels_oe

        flecha_vuelta_xy = flecha_eo_xy
        flecha_vuelta_text_xy = flecha_eo_text_xy
        labels_vuelta = labels_eo

        df_d0 = df_d0.sort_values('section_id', ascending=False)
        df_d1 = df_d1.sort_values('section_id', ascending=False)

    sns.barplot(data=df_d0, x="section_id",
                y=indicator, ax=ax3, color='Purple',
                order=df_d0.section_id.values)

    sns.barplot(data=df_d1, x="section_id",
                y=indicator, ax=ax4, color='Orange',
                order=df_d1.section_id.values)

    # Axis
    ax3.set_xticklabels(labels_ida)
    ax4.set_xticklabels(labels_vuelta)

    ax3.set_ylabel(y_axis_lable)
    ax3.set_xlabel('')

    ax4.get_yaxis().set_visible(False)

    ax4.set_ylabel('')
    ax4.set_xlabel('')
    max_y_barplot = max(df_d0[indicator].max(), df_d1[indicator].max())
    ax3.set_ylim(0, max_y_barplot)
    ax4.set_ylim(0, max_y_barplot)

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax4.spines.left.set_visible(False)
    ax4.spines.right.set_visible(False)
    ax4.spines.top.set_visible(False)

    # For direction 0, get the last section of the route geom
    flecha_ida = gdf.loc[gdf.section_id == max(gdf.section_id), 'geometry']
    flecha_ida = list(flecha_ida.item().coords)
    flecha_ida_inicio = flecha_ida[1]
    flecha_ida_fin = flecha_ida[0]

    # For direction 1, get the first section of the route geom
    flecha_vuelta = gdf.loc[gdf.section_id == 0.0, 'geometry']
    flecha_vuelta = list(flecha_vuelta.item().coords)
    # invert the direction of the arrow
    flecha_vuelta_inicio = flecha_vuelta[0]
    flecha_vuelta_fin = flecha_vuelta[1]

    ax1.annotate('', xy=(flecha_ida_inicio[0],
                         flecha_ida_inicio[1]),
                 xytext=(flecha_ida_fin[0],
                         flecha_ida_fin[1]),
                 arrowprops=dict(facecolor='black',
                                 edgecolor='black'),
                 )

    ax2.annotate('', xy=(flecha_vuelta_inicio[0],
                         flecha_vuelta_inicio[1]),
                 xytext=(flecha_vuelta_fin[0],
                         flecha_vuelta_fin[1]),
                 arrowprops=dict(facecolor='black',
                                 edgecolor='black'),
                 )

    ax3.annotate('Sentido', xy=flecha_ida_xy, xytext=flecha_ida_text_xy,
                 size=16, va="center", ha="center",
                 xycoords='axes fraction',
                 arrowprops=dict(facecolor='Purple',
                                 shrink=0.05, edgecolor='Purple'),
                 )
    ax4.annotate('Sentido', xy=flecha_vuelta_xy, xytext=flecha_vuelta_text_xy,
                 size=16, va="center", ha="center",
                 xycoords='axes fraction',
                 arrowprops=dict(facecolor='Orange',
                                 shrink=0.05, edgecolor='Orange'),
                 )

    prov = cx.providers.Stamen.TonerLite
    try:
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string(), source=prov)
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string(), source=prov)
    except (UnidentifiedImageError):
        prov = cx.providers.CartoDB.Positron
        cx.add_basemap(ax1, crs=gdf_d0.crs.to_string(), source=prov)
        cx.add_basemap(ax2, crs=gdf_d1.crs.to_string(), source=prov)
    except (r_ConnectionError):
        pass

    alias = leer_alias()

    for frm in ['png', 'pdf']:
        archivo = f"{alias}_{day}_segmentos_id_linea_"
        archivo = archivo+f"{id_linea}_{indicator}_{hr_str}.{frm}"
        db_path = os.path.join("resultados", frm, archivo)
        f.savefig(db_path, dpi=300)
    plt.close(f)

    if save_gdf:
        gdf_d0 = gdf_d0.to_crs(epsg=4326)
        gdf_d1 = gdf_d1.to_crs(epsg=4326)

        f_0 = f'segmentos_id_linea_{id_linea}_{indicator}{hr_str}_0.geojson'
        f_1 = f'segmentos_id_linea_{id_linea}_{indicator}{hr_str}_1.geojson'

        db_path_0 = os.path.join("resultados", "geojson", f_0)
        db_path_1 = os.path.join("resultados", "geojson", f_1)

        gdf_d0.to_file(db_path_0, driver='GeoJSON')
        gdf_d1.to_file(db_path_1, driver='GeoJSON')

        conn_dash = iniciar_conexion_db(tipo='dash')

        gdf_d0_dash['wkt'] = gdf_d0_dash.geometry.to_wkt()
        gdf_d1_dash['wkt'] = gdf_d1_dash.geometry.to_wkt()

        gdf_d_dash = pd.concat([gdf_d0_dash, gdf_d1_dash], ignore_index=True)

        gdf_d_dash['nombre_linea'] = id_linea_str

        cols = ['id_linea',
                'nombre_linea',
                'day_type',
                'n_sections',
                'sentido',
                'section_id',
                'hora_min',
                'hora_max',
                'cantidad_etapas',
                'prop_etapas',
                'buff_factor',
                'wkt']

        gdf_d_dash = gdf_d_dash[cols]

        gdf_d_dash_ant = pd.read_sql_query(
            """
            SELECT *
            FROM ocupacion_por_linea_tramo
            """,
            conn_dash,
        )

        gdf_d_dash_ant = gdf_d_dash_ant[~(
            (gdf_d_dash_ant.id_linea.isin(
                gdf_d_dash.id_linea.unique().tolist())) &
            (gdf_d_dash_ant.day_type.isin(
                gdf_d_dash.day_type.unique().tolist())) &
            (gdf_d_dash_ant.n_sections.isin(
                gdf_d_dash.n_sections.unique().tolist())) &
            ((gdf_d_dash_ant.hora_min == from_hr)
             & (gdf_d_dash_ant.hora_max == to_hr))
        )]

        gdf_d_dash = pd.concat(
            [gdf_d_dash_ant, gdf_d_dash], ignore_index=True)

        gdf_d_dash.to_sql("ocupacion_por_linea_tramo", conn_dash,
                          if_exists="replace", index=False)

        conn_dash.close()

    if return_gdfs:
        return gdf_d0, gdf_d1


def plot_voronoi_zones(voi, hexs, hexs2, show_map, alias):
    fig = Figure(figsize=(13.5, 13.5), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plt.rcParams.update({"axes.facecolor": '#d4dadc',
                        'figure.facecolor': '#d4dadc'})
    voi = voi.to_crs(3857)
    voi.geometry.boundary.plot(edgecolor='grey', linewidth=.5, ax=ax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                    attribution=None, attribution_size=10)
    voi['coords'] = voi['geometry'].apply(
        lambda x: x.representative_point().coords[:])
    voi['coords'] = [coords[0] for coords in voi['coords']]
    voi.apply(lambda x: ax.annotate(
        text=x['Zona_voi'],
        xy=x.geometry.centroid.coords[0],
        ha='center',
        color='darkblue',
    ), axis=1)
    ax.set_title('Zonificación', fontsize=12)
    ax.axis('off')

    if show_map:

        display(fig)

        # Display figura temporal
        fig = Figure(figsize=(13.5, 13.5), dpi=70)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        hexs.to_crs(3857).plot(markersize=hexs['fex']/500, ax=ax)
        hexs2.to_crs(3857).boundary.plot(ax=ax, lw=.3)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                        attribution=None, attribution_size=10)
        ax.axis('off')

    # graba resultados
    file_path = os.path.join("resultados", "png", f"{alias}Zona_voi_map.png")
    fig.savefig(file_path, dpi=300)
    print('Zonificación guardada en', file_path)

    file_path = os.path.join("resultados", "pdf", f"{alias}Zona_voi_map.pdf")
    fig.savefig(file_path, dpi=300)
    voi = voi.to_crs(4326)

    file_path = os.path.join("resultados", f"{alias}Zona_voi.geojson")
    voi[['Zona_voi', 'geometry']].to_file(file_path)


def imprimir_matrices_od(viajes,
                         savefile='viajes',
                         title='Matriz OD',
                         var_fex="",
                         desc_dia='',
                         tipo_dia=''):

    alias = leer_alias()

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    zonas = pd.read_sql_query(
        """
        SELECT * from zonas
        """,
        conn_insumos,
    )

    conn_insumos.close()
    zonas[f'h3_r6'] = zonas['h3'].apply(h3.h3_to_parent, res=6)
    zonas[f'h3_r7'] = zonas['h3'].apply(h3.h3_to_parent, res=7)

    df, matriz_zonas = traigo_zonificacion(
        viajes, zonas, h3_o='h3_o', h3_d='h3_d')

    if len(var_fex) == 0:
        var_fex = 'var_fex'
        df[var_fex] = 1

    for i in matriz_zonas:
        var_zona = i[1]
        matriz_order = i[2]

        imprime_od(
            df,
            zona_origen=f"{var_zona}_o",
            zona_destino=f"{var_zona}_d",
            var_fex=var_fex,
            x_rotation=90,
            normalize=True,
            cmap="Reds",
            title='Matriz OD General',
            figsize_tuple='',
            matriz_order=matriz_order,
            savefile=f"{alias}{savefile}_{var_zona}",
            alias=alias,
            desc_dia=desc_dia,
            tipo_dia=tipo_dia,
            var_zona=var_zona,
            filtro1='Todos los viajes'
        )

        imprime_od(
            df[(df.cant_etapas > 1)],
            zona_origen=f"{var_zona}_o",
            zona_destino=f"{var_zona}_d",
            var_fex=var_fex,
            x_rotation=90,
            normalize=True,
            cmap="Reds",
            title='Matriz OD viajes con transferencia',
            figsize_tuple='',
            matriz_order=matriz_order,
            savefile=f"{alias}{savefile}_{var_zona}_transferencias",
            alias=alias,
            desc_dia=desc_dia,
            tipo_dia=tipo_dia,
            var_zona=var_zona,
            filtro1='Con transferencias'
        )

        imprime_od(
            df[(df.distance_osm_drive <= 5)],
            zona_origen=f"{var_zona}_o",
            zona_destino=f"{var_zona}_d",
            var_fex=var_fex,
            x_rotation=90,
            normalize=True,
            cmap="Reds",
            title='Matriz OD viajes cortos (<5kms)',
            figsize_tuple='',
            matriz_order=matriz_order,
            savefile=f"{alias}{savefile}_{var_zona}_corta_distancia",
            alias=alias,
            desc_dia=desc_dia,
            tipo_dia=tipo_dia,
            var_zona=var_zona,
            filtro1='Corta distancia (<5kms)'
        )

        # Imprime hora punta manana, mediodia, tarde

        df_tmp = df.groupby(['dia', 'hora'], as_index=False)[
            var_fex].sum().reset_index()
        df_tmp = df_tmp.groupby(['hora'])[var_fex].mean().reset_index()

        try:
            manana = df_tmp[(df_tmp.hora.astype(int) >= 6) & (
                df_tmp.hora.astype(int) < 12)][var_fex].idxmax()
        except ValueError:
            manana = None

        try:
            mediodia = df_tmp[(df_tmp.hora.astype(int) >= 12) & (
                df_tmp.hora.astype(int) < 16)][var_fex].idxmax()
        except ValueError:
            mediodia = None
        try:
            tarde = df_tmp[(df_tmp.hora.astype(int) >= 16) & (
                df_tmp.hora.astype(int) < 22)][var_fex].idxmax()
        except ValueError:
            tarde = None

        if manana != None:
            imprime_od(
                df[(df.hora.astype(int) >= manana-1) &
                    (df.hora.astype(int) <= manana+1)],
                zona_origen=f"{var_zona}_o",
                zona_destino=f"{var_zona}_d",
                var_fex=var_fex,
                x_rotation=90,
                normalize=True,
                cmap="Reds",
                title='Matriz OD viajes punta mañana',
                figsize_tuple='',
                matriz_order=matriz_order,
                savefile=f"{alias}{savefile}_{var_zona}_punta_manana",
                alias=alias,
                desc_dia=desc_dia,
                tipo_dia=tipo_dia,
                var_zona=var_zona,
                filtro1='Punta mañana'
            )

        if mediodia != None:
            imprime_od(
                df[(df.hora.astype(int) >= mediodia-1) &
                    (df.hora.astype(int) <= mediodia+1)],
                zona_origen=f"{var_zona}_o",
                zona_destino=f"{var_zona}_d",
                var_fex=var_fex,
                x_rotation=90,
                normalize=True,
                cmap="Reds",
                title='Matriz OD viajes punta mediodía',
                figsize_tuple='',
                matriz_order=matriz_order,
                savefile=f"{alias}{savefile}_{var_zona}_punta_mediodia",
                alias=alias,
                desc_dia=desc_dia,
                tipo_dia=tipo_dia,
                var_zona=var_zona,
                filtro1='Punta mediodía'

            )

        if tarde != None:
            imprime_od(
                df[(df.hora.astype(int) >= tarde-1) &
                    (df.hora.astype(int) <= tarde+1)],
                zona_origen=f"{var_zona}_o",
                zona_destino=f"{var_zona}_d",
                var_fex=var_fex,
                x_rotation=90,
                normalize=True,
                cmap="Reds",
                title='Matriz OD viajes punta tarde',
                figsize_tuple='',
                matriz_order=matriz_order,
                savefile=f"{alias}{savefile}_{var_zona}_punta_tarde",
                alias=alias,
                desc_dia=desc_dia,
                tipo_dia=tipo_dia,
                var_zona=var_zona,
                filtro1='Punta tarde'
            )


def imprime_lineas_deseo(df,
                         h3_o='',
                         h3_d='',
                         var_fex='',
                         title='Líneas de deseo',
                         savefile='lineas_deseo',
                         k_jenks=5,
                         filtro1='',
                         desc_dia='',
                         tipo_dia=''
                         ):
    """
    Esta funcion toma un df de viajes con destino validado
    nombres de columnas con el h3 de origen y destino
    un nombre con la columna del factor de expansion
    y nombres para el titulo del mapa y el archivo
    y produce un mapa con lineas de deseo para todas las
    geografias presentes en la tabla zonas
    """

    pd.options.mode.chained_assignment = None
    alias = leer_alias()

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    zonas = pd.read_sql_query(
        """
        SELECT * from zonas
        """,
        conn_insumos,
    )

    conn_insumos.close()

    zonas[f'h3_r6'] = zonas['h3'].apply(h3.h3_to_parent, res=6)
    zonas[f'h3_r7'] = zonas['h3'].apply(h3.h3_to_parent, res=7)

    zonas = gpd.GeoDataFrame(
        zonas,
        geometry=gpd.points_from_xy(zonas['longitud'], zonas['latitud']),
        crs=4326,
    )

    if len(h3_o) == 0:
        h3_o = 'h3_o_norm'
    if len(h3_d) == 0:
        h3_d = 'h3_d_norm'

    if len(var_fex) == 0:
        var_fex = 'fex'
        df[var_fex] = 1

    # Clasificar od en terminos de zonas
    df, matriz_zonas = traigo_zonificacion(df,
                                           zonas,
                                           h3_o=h3_o,
                                           h3_d=h3_d,
                                           res_agg=True)

    for m in matriz_zonas:
        var_zona = m[1]

        lineas_deseo(df,
                     zonas,
                     var_zona,
                     var_fex,
                     h3_o,
                     h3_d,
                     alpha=.4,
                     cmap='viridis_r',
                     porc_viajes=100,
                     title=title,
                     savefile=f"{alias}{savefile}_{var_zona}",
                     show_fig=False,
                     k_jenks=k_jenks,
                     alias=alias,
                     desc_dia=desc_dia,
                     tipo_dia=tipo_dia,
                     zona=var_zona,
                     filtro1='Todos los viajes'
                     )

        lineas_deseo(df[(df.cant_etapas > 1)],
                     zonas,
                     var_zona,
                     var_fex,
                     h3_o,
                     h3_d,
                     alpha=.4,
                     cmap='crest',
                     porc_viajes=90,
                     title=f'{title}\nViajes con transferencias',
                     savefile=f"{alias}{savefile}_{var_zona}_transferencias",
                     show_fig=False,
                     k_jenks=k_jenks,
                     alias=alias,
                     desc_dia=desc_dia,
                     tipo_dia=tipo_dia,
                     zona=var_zona,
                     filtro1='Con transferencias'
                     )

        lineas_deseo(df[(df.distance_osm_drive <= 5)],
                     zonas,
                     var_zona,
                     var_fex,
                     h3_o,
                     h3_d,
                     alpha=.4,
                     cmap='magma_r',
                     porc_viajes=90,
                     title=f'{title}\nViajes de corta distancia (<5kms)',
                     savefile=f"{alias}{savefile}_{var_zona}_corta_distancia",
                     show_fig=False,
                     k_jenks=k_jenks,
                     alias=alias,
                     desc_dia=desc_dia,
                     tipo_dia=tipo_dia,
                     zona=var_zona,
                     filtro1='Corta distancia (<5kms)'
                     )

        # Imprime hora punta manana, mediodia, tarde

        df_tmp = df\
            .groupby(['dia', 'hora'], as_index=False)\
            .factor_expansion_linea.sum()\
            .rename(columns={'factor_expansion_linea': 'cant'})\
            .reset_index()
        df_tmp = df_tmp.groupby(['hora']).cant.mean().reset_index()
        try:
            manana = df_tmp[(df_tmp.hora.astype(int) >= 6) & (
                df_tmp.hora.astype(int) < 12)].cant.idxmax()
        except ValueError:
            manana = None

        try:
            mediodia = df_tmp[(df_tmp.hora.astype(int) >= 12) & (
                df_tmp.hora.astype(int) < 16)].cant.idxmax()
        except ValueError:
            mediodia = None

        try:
            tarde = df_tmp[(df_tmp.hora.astype(int) >= 16) & (
                df_tmp.hora.astype(int) < 22)].cant.idxmax()
        except ValueError:
            tarde = None

        if manana != None:
            lineas_deseo(df[
                (df.hora.astype(int) >= manana-1) &
                (df.hora.astype(int) <= manana+1)],
                zonas,
                var_zona,
                var_fex,
                h3_o,
                h3_d,
                alpha=.4,
                cmap='magma_r',
                porc_viajes=90,
                title=f'{title}\nViajes en hora punta mañana',
                savefile=f"{alias}{savefile}_{var_zona}_punta_manana",
                show_fig=False,
                normalizo_latlon=False,
                k_jenks=k_jenks,
                alias=alias,
                desc_dia=desc_dia,
                tipo_dia=tipo_dia,
                zona=var_zona,
                filtro1='Punta Mañana')

        if mediodia != None:
            lineas_deseo(df[
                (df.hora.astype(int) >= mediodia-1) &
                (df.hora.astype(int) <= mediodia+1)],
                zonas,
                var_zona,
                var_fex,
                h3_o,
                h3_d,
                alpha=.4,
                cmap='magma_r',
                porc_viajes=90,
                title=f'{title}\nViajes en hora punta mediodia',
                savefile=f"{alias}{savefile}_{var_zona}_punta_mediodia",
                show_fig=False,
                normalizo_latlon=False,
                k_jenks=k_jenks,
                alias=alias,
                desc_dia=desc_dia,
                tipo_dia=tipo_dia,
                zona=var_zona,
                filtro1='Punta Mediodía')

        if tarde != None:
            lineas_deseo(df[
                (df.hora.astype(int) >= tarde-1) &
                (df.hora.astype(int) <= tarde+1)],
                zonas,
                var_zona,
                var_fex,
                h3_o,
                h3_d,
                alpha=.4,
                cmap='magma_r',
                porc_viajes=90,
                title=f'{title}\nViajes en hora punta tarde',
                savefile=f"{alias}{savefile}_{var_zona}_punta_tarde",
                show_fig=False,
                normalizo_latlon=False,
                k_jenks=k_jenks,
                alias=alias,
                desc_dia=desc_dia,
                tipo_dia=tipo_dia,
                zona=var_zona,
                filtro1='Punta Tarde')


def imprime_graficos_hora(viajes,
                          title='Cantidad de viajes en transporte público',
                          savefile='viajes',
                          var_fex='',
                          desc_dia='',
                          tipo_dia=''):

    pd.options.mode.chained_assignment = None
    configs = leer_configs_generales()
    db_path = traigo_db_path
    alias = leer_alias()

    df_aux = pd.DataFrame([(str(x).zfill(2))
                          for x in list(range(0, 24))], columns=['hora'])
    df_aux['dia'] = viajes.head(1).dia.values[0]
    df_aux['cant'] = 0
    df_aux['modo'] = viajes.modo.unique()[0]

    if not var_fex:
        viajes['cant'] = 1
    else:
        viajes['cant'] = viajes[var_fex]

    viajesxhora = pd.concat([viajes, df_aux], ignore_index=True)
    viajesxhora['hora'] = viajesxhora.hora.astype(str).str[:2].str.zfill(2)

    viajesxhora = viajesxhora.groupby(
        ['dia', 'hora']).cant.sum().reset_index()
    viajesxhora = viajesxhora.groupby(['hora']).cant.mean().reset_index()

    viajesxhora['cant'] = viajesxhora['cant'].round().astype(int)

    savefile_ = f'{savefile}_x_hora'

    viajesxhora_dash = viajesxhora.copy()
    viajesxhora_dash['modo'] = 'Todos'

    # Viajes por hora
    with sns.axes_style(
        {"axes.facecolor": "#cadce0",
         'figure.facecolor': '#cadce0',
         }):
        fig = Figure(figsize=(10, 3), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        viajesxhora.plot(ax=ax, legend=False, label=False)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Hora', fontsize=8)
        ax.set_ylabel('Viajes', fontsize=8)
        ax.set_xticks(list(range(0, 24)))
        ax.tick_params(labelsize=6)

        print("Nuevos archivos en resultados: ", f'{alias}{savefile_}')
        db_path = os.path.join("resultados", "png", f"{alias}{savefile_}.png")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

        db_path = os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

    # Viajes por hora y modo de transporte
    viajesxhora = pd.concat([viajes, df_aux], ignore_index=True)
    viajesxhora['hora'] = viajesxhora.hora.astype(str).str[:2].str.zfill(2)
    viajesxhora = viajesxhora.groupby(
        ['dia', 'hora', 'modo'], as_index=False).cant.sum()
    viajesxhora = viajesxhora.groupby(
        ['hora', 'modo'], as_index=False).cant.mean()

    viajesxhora.loc[viajesxhora.modo.str.contains(
        'Multi'), 'modo'] = 'Multietapa'
    viajesxhora = viajesxhora.groupby(['hora', 'modo'])[
        'cant'].sum().reset_index()

    viajesxhora['cant'] = viajesxhora['cant'].round().astype(int)

    # guarda distribución de viajes para dashboard
    viajesxhora_dash = pd.concat(
        [viajesxhora_dash, viajesxhora], ignore_index=True)

    viajesxhora_dash['tipo_dia'] = tipo_dia
    viajesxhora_dash['desc_dia'] = desc_dia

    viajesxhora_dash = viajesxhora_dash[[
        'tipo_dia', 'desc_dia', 'hora', 'cant', 'modo']]
    viajesxhora_dash.columns = ['tipo_dia', 'desc_dia', 'Hora', 'Viajes', 'Modo']

    conn_dash = iniciar_conexion_db(tipo='dash')

    query = f"""
        DELETE FROM viajes_hora
        WHERE desc_dia = "{desc_dia}"
        and tipo_dia = "{tipo_dia}"
    """

    conn_dash.execute(query)
    conn_dash.commit()
    
    modos = viajesxhora_dash.Modo.unique().tolist()
    hrs = [str(i).zfill(2) for i in range(0,24)]    
    for modo in modos:
        for hr in hrs:
            if len(viajesxhora_dash.loc[(viajesxhora_dash.Modo==modo)&(viajesxhora_dash.Hora==hr)])==0:
                
                viajesxhora_dash = pd.concat([
                                viajesxhora_dash,
                                pd.DataFrame([[tipo_dia, 
                               desc_dia, 
                               hr, 
                               0, 
                               modo]], 
                             columns = viajesxhora_dash.columns)
                            ])

    viajesxhora_dash.to_sql("viajes_hora", conn_dash,
                            if_exists="append", index=False)

    conn_dash.close()

    # Viajes por hora
    savefile_ = f'{savefile}_modo'
    with sns.axes_style(
        {"axes.facecolor": "#cadce0",
         'figure.facecolor': '#cadce0',
         }):

        fig = Figure(figsize=(10, 3), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        for i in viajesxhora.modo.unique():
            viajesxhora[viajesxhora.modo == i].reset_index().plot(
                ax=ax, y='cant', legend=True, label=i)

        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Hora', fontsize=8)
        ax.set_ylabel('Viajes', fontsize=8)
        ax.set_xticks(list(range(0, 24)))
        ax.tick_params(labelsize=6)
        print("Nuevos archivos en resultados: ", f'{alias}{savefile_}')
        db_path = os.path.join("resultados", "png", f"{alias}{savefile_}.png")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

        db_path = os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

    # Distribución de viajes
    savefile_ = f'{savefile}_dist'
    vi = viajes[(viajes.distance_osm_drive.notna())
                & (viajes.distance_osm_drive > 0)
                & (viajes.h3_o != viajes.h3_d)]
    vi['distance_osm_drive'] = vi['distance_osm_drive'].astype(int)

    vi_modo = vi\
        .groupby(['distance_osm_drive', 'modo'], as_index=False)\
        .factor_expansion_linea.sum()\
        .rename(columns={'factor_expansion_linea': 'cant'})

    vi = vi\
        .groupby('distance_osm_drive', as_index=False)\
        .factor_expansion_linea.sum()\
        .rename(columns={'factor_expansion_linea': 'cant'})

    vi = vi.loc[vi.cant > 0, ['distance_osm_drive', 'cant']
                ].sort_values('distance_osm_drive')

    vi['pc'] = round(vi.cant / vi.cant.sum() * 100, 5)
    vi['csum'] = vi.pc.cumsum()
    vi = vi[vi.csum <= 99.5]
    vi['Viajes (en miles)'] = round(vi.cant/1000)

    vi_modo['pc'] = round(vi_modo.cant / vi_modo.cant.sum() * 100, 5)
    vi_modo['csum'] = vi_modo.pc.cumsum()
    vi_modo = vi_modo[vi_modo.csum <= 99.5]

    # guarda distribución de viajes para dashboard

    vi_dash = vi.copy()
    vi_dash['modo'] = 'Todos'
    vi_dash = pd.concat([vi_dash, vi_modo], ignore_index=True)

    vi_dash['tipo_dia'] = tipo_dia
    vi_dash['desc_dia'] = desc_dia

    vi_dash = vi_dash[['desc_dia', 'tipo_dia',
                       'distance_osm_drive', 'cant', 'modo']]
    vi_dash.columns = ['desc_dia', 'tipo_dia', 'Distancia', 'Viajes', 'Modo']

    conn_dash = iniciar_conexion_db(tipo='dash')
    query = f"""
        DELETE FROM distribucion
        WHERE desc_dia = "{desc_dia}"
        and tipo_dia = "{tipo_dia}"
        """
    conn_dash.execute(query)
    conn_dash.commit()

    vi_dash.to_sql("distribucion", conn_dash, if_exists="append", index=False)
    conn_dash.close()

    ytitle = "Viajes"
    if vi.cant.mean() > 1000:
        vi['cant'] = round(vi['cant']/1000)
        ytitle = "Viajes (en miles)"

    sns.set_style("darkgrid", {"axes.facecolor": "#cadce0",
                  'figure.facecolor': '#cadce0', "grid.linestyle": ":"})

    fig = Figure(figsize=(8, 4), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    sns.histplot(x='distance_osm_drive', weights='cant',
                 data=vi, bins=len(vi), ax=ax)  # element='poly',
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Distancia (kms)", fontsize=10)
    ax.set_ylabel(ytitle, fontsize=10)
    ax.set_xticks(list(range(0, len(vi)+1, 5)))

    fig.tight_layout()

    print("Nuevos archivos en resultados: ", f'{alias}{savefile_}')
    db_path = os.path.join("resultados", "png", f"{alias}{savefile_}.png")
    fig.savefig(db_path, dpi=300, bbox_inches="tight")

    db_path = os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")
    fig.savefig(db_path, dpi=300, bbox_inches="tight")


def imprime_burbujas(df,
                     res=7,
                     h3_o='h3_o',
                     alpha=.4,
                     cmap='viridis_r',
                     var_fex='',
                     porc_viajes=90,
                     title='burbujas',
                     savefile='burbujas',
                     show_fig=False,
                     k_jenks=5):

    pd.options.mode.chained_assignment = None
    configs = leer_configs_generales()
    db_path = traigo_db_path
    alias = leer_alias()

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    zonas = pd.read_sql_query(
        """
        SELECT * from zonas
        """,
        conn_insumos,
    )

    conn_data.close()
    conn_insumos.close()

    zonas = gpd.GeoDataFrame(
        zonas,
        geometry=gpd.points_from_xy(zonas['longitud'], zonas['latitud']),
        crs=4326)

    if len(var_fex) == 0:
        var_fex = 'fex'
        df[var_fex] = 1

    df_agg = crea_df_burbujas(df,
                              zonas,
                              h3_o=h3_o,
                              var_fex=var_fex,
                              porc_viajes=porc_viajes,
                              res=res
                              )

    df_agg[var_fex] = df_agg[var_fex].round().astype(int)

    if len(df_agg) > 0:

        multip = df_agg[var_fex].head(1).values[0] / 500

        fig = Figure(figsize=(13.5, 13.5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        zonas[zonas['h3'].isin(df[h3_o].unique())].to_crs(
            3857).plot(ax=ax, alpha=0)
        try:
            df_agg.to_crs(3857).plot(ax=ax,
                                     alpha=alpha,
                                     cmap=cmap,
                                     markersize=df_agg[var_fex] / multip,
                                     column=var_fex,
                                     scheme='FisherJenks',
                                     k=k_jenks,
                                     legend=True,
                                     legend_kwds={
                                         'loc': 'upper right',
                                         'title': 'Viajes',
                                         'fontsize': 8,
                                            'title_fontsize': 10,
                                     }
                                     )
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                            attribution=None, attribution_size=10)

            ax.set_title(title, fontsize=12)

            leg = ax.get_legend()
            # leg._loc = 3

            for lbl in leg.get_texts():
                label_text = lbl.get_text()
                lower = label_text.split(',')[0]
                upper = label_text.split(',')[1]
                new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'
                lbl.set_text(new_text)

            ax.add_artist(
                ScaleBar(1, location='lower right', box_alpha=0, pad=1))
            ax.axis('off')

            if len(savefile) > 0:
                print("Nuevos archivos en resultados: ", f'{alias}{savefile}')
                db_path = os.path.join("resultados", "png",
                                       f"{alias}{savefile}.png")
                fig.savefig(db_path, dpi=300, bbox_inches="tight")

                db_path = os.path.join("resultados", "pdf",
                                       f"{alias}{savefile}.pdf")
                fig.savefig(db_path, dpi=300, bbox_inches="tight")

            if show_fig:
                display(fig)

        except (ValueError) as e:
            print(e)


def traigo_zonificacion(viajes,
                        zonas,
                        h3_o='h3_o',
                        h3_d='h3_d',
                        res_agg=False):
    """
    Esta funcion toma la tabla viajes
    la tabla zonas, los nombres de las columnas con el h3 de origen y destino
    y un parametro para usar h3 con resolucion mas agregada
    y clasifica los origenes y destinos de los viajes para cada zona
    """
    configs = leer_configs_generales()

    matriz_zonas = []
    vars_zona = []
    if 'Zona_voi' in zonas.columns:

        matriz_zonas = [['',
                         'Zona_voi',
                         [str(x) for x in list(
                             range(1, len(zonas.Zona_voi.unique())+1))]
                         ]]
        vars_zona = ['Zona_voi']

    if res_agg:
        zonas[f'h3_r6'] = zonas['h3'].apply(h3.h3_to_parent, res=6)
        zonas[f'h3_r7'] = zonas['h3'].apply(h3.h3_to_parent, res=7)

        matriz_zonas += [['', f'h3_r6', ''],
                         ['', f'h3_r7', '']]
        vars_zona += [f'h3_r6']
        vars_zona += [f'h3_r7']

    if configs["zonificaciones"]:
        for n in range(0, 5):

            try:
                file_zona = configs["zonificaciones"][f"geo{n+1}"]
                var_zona = configs["zonificaciones"][f"var{n+1}"]

                try:
                    matriz_order = configs["zonificaciones"][f"orden{n+1}"]
                except KeyError:
                    matriz_order = ""

                if matriz_order is None:
                    matriz_order = ""

                if var_zona in zonas.columns:
                    matriz_zonas += [[file_zona, var_zona, matriz_order]]
                    vars_zona += [var_zona]
            except KeyError:
                pass

    vars_o = [h3_o] + [f'{x}_o' for x in vars_zona]
    vars_d = [h3_d] + [f'{x}_d' for x in vars_zona]

    zonas_tmp = zonas[['h3']+vars_zona]
    zonas_tmp.columns = vars_o
    viajes = viajes.merge(
        zonas_tmp,
        on=h3_o
    )

    zonas_tmp = zonas[['h3']+vars_zona]
    zonas_tmp.columns = vars_d
    viajes = viajes.merge(
        zonas_tmp,
        on=h3_d
    )
    return viajes, matriz_zonas


def imprime_od(
    df,
    zona_origen,
    zona_destino,
    var_fex="",
    normalize=False,
    margins=False,
    matriz_order="",
    matriz_order_row="",
    matriz_order_col="",
    path_resultados=Path(),
    savefile="",
    title="Matriz OD",
    figsize_tuple='',
    fontsize=12,
    fmt="",
    cbar=False,
    x_rotation=0,
    y_rotation=0,
    cmap="Blues",
    total_color="navy",
    total_background_color="white",
    show_fig=False,
    alias='',
    desc_dia='',
    tipo_dia='',
    var_zona='',
    filtro1='',
):

    if len(fmt) == 0:
        if normalize:
            fmt = ".1%"
        else:
            fmt = ".1f"
    df = df[(df[zona_origen].notna()) & (df[zona_destino].notna())].copy()

    fill_value = mcolors.to_rgba(total_background_color)

    if len(matriz_order) > 0:
        matriz_order_row = matriz_order
        matriz_order_col = matriz_order

    if len(var_fex) == 0:
        var_fex = 'fex'
        df[var_fex] = 1

    df = df.groupby(['dia', zona_origen, zona_destino],
                    as_index=False)[var_fex].sum()
    df = df.groupby([zona_origen, zona_destino], as_index=False)[
        var_fex].mean()

    df[var_fex] = df[var_fex].round().astype(int)

    if len(df) > 0:

        vals = df.loc[~(df[zona_origen].isin(
            df[zona_destino].unique())), zona_origen].unique()
        for i in vals:
            df = pd.concat([df,
                            pd.DataFrame(
                                [[i, i, 0]],
                                columns=[
                                    zona_origen,
                                    zona_destino,
                                    var_fex
                                ])
                            ])
        vals = df.loc[~(df[zona_destino].isin(
            df[zona_origen].unique())), zona_destino].unique()
        for i in vals:
            df = pd.concat([df,
                            pd.DataFrame(
                                [[i, i, 0]],
                                columns=[
                                    zona_destino,
                                    zona_origen,
                                    var_fex])
                            ])

        od_heatmap = pd.crosstab(
            index=df[zona_origen],
            columns=df[zona_destino],
            values=df[var_fex],
            aggfunc="sum",
            normalize=normalize,
        )

        if len(figsize_tuple) == 0:
            figsize_tuple = (len(od_heatmap)+1, len(od_heatmap)+1)

        matriz_order = [i for i in matriz_order if i in od_heatmap.columns]
        matriz_order_row = [
            i for i in matriz_order_row if i in od_heatmap.columns]
        matriz_order_col = [
            i for i in matriz_order_col if i in od_heatmap.columns]

        if len(matriz_order_col) > 0:
            od_heatmap = od_heatmap[matriz_order_col]
        if len(matriz_order_row) > 0:

            od_heatmap = (
                od_heatmap.reset_index()
                .sort_values(
                    zona_origen,
                    key=lambda s: s.apply(matriz_order_row.index),
                    ignore_index=True,
                )
                .set_index(zona_origen)
            )

        for _ in od_heatmap.columns:
            od_heatmap.loc[od_heatmap[_] == 0, _] = None

        if margins:
            od_heatmap_sintot = od_heatmap.copy()
            od_heatmap["Total"] = od_heatmap.sum(axis=1)
            od_heatmap = pd.concat(
                [od_heatmap, pd.DataFrame([od_heatmap.sum()], index=["Total"])]
            )

            od_heatmap_tmp = od_heatmap.copy()
            od_heatmap_tmp["Total"] = 0
            od_heatmap_tmp.iloc[len(od_heatmap_tmp) - 1] = 0

            fig = Figure(figsize=figsize_tuple, dpi=150)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            sns.heatmap(
                od_heatmap_tmp,
                cmap=cmap,
                annot=True,
                fmt=fmt,
                annot_kws={"size": fontsize},
                square=True,
                linewidth=0.5,
                cbar=cbar,
                ax=ax,
            )

            # find your QuadMesh object and get array of colors
            facecolors_anterior = ax.findobj(QuadMesh)[0].get_facecolors()

        fig = Figure(figsize=figsize_tuple, dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        sns.heatmap(
            od_heatmap,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            square=True,
            linewidth=0.5,
            cbar=cbar,
            ax=ax,
            xticklabels=True,
            yticklabels=True,
        )

        ax.set_title(title, fontsize=fontsize)
        ax.set_ylabel("Origen", fontsize=fontsize)
        ax.set_xlabel("Destino", fontsize=fontsize)

        # move x and y ticks
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("right")

        ax.set_xticklabels(
            od_heatmap.columns.tolist(),
            rotation=x_rotation,
            ha="right",
        )

        if margins:

            # find your QuadMesh object and get array of colors
            quadmesh = ax.findobj(QuadMesh)[0]
            facecolors = quadmesh.get_facecolors()

            # replace background heatmap colors
            for i in range(0, len(facecolors)):
                if (((i + 1) % len(od_heatmap.columns)) != 0) & (
                    i < (len(facecolors) - len(od_heatmap.columns))
                ):
                    facecolors[i] = facecolors_anterior[i]
                else:
                    facecolors[i] = fill_value

            # set modified colors
            quadmesh.set_facecolors = facecolors

            # modify all text to black or white
            lst = []
            for _ in od_heatmap_sintot.columns:
                lst += od_heatmap[_].tolist()
            val_min = pd.DataFrame(lst, columns=["valor"]).drop_duplicates()
            val_min["val_type"] = pd.qcut(
                val_min.valor,
                q=4,
                labels=["1", "2", "3", "4"],
            )
            val_min = val_min[val_min.val_type == "4"].valor.min()

            col_totals = np.arange(
                len(od_heatmap.columns) - 1,
                (len(od_heatmap.columns)) * len(od_heatmap),
                len(od_heatmap.columns),
            ).tolist()
            ii = 0
            for i in ax.findobj(Text):

                if (ii in col_totals[:-1]) | (
                    (ii >= len(facecolors) - len(od_heatmap.columns))
                    & (ii < len(facecolors) - 1)
                ):
                    i.set_color(total_color)
                else:
                    try:
                        value_i = (
                            str(i)
                            .replace("Text(", "")
                            .replace(")", "")
                            .replace("'", "")
                            .split(",")[2]
                            .replace(" ", "")
                        )

                        if value_i == "Total":
                            i.set_color(total_color)

                        if ii <= len(facecolors) - 1:
                            value_i = float(value_i)
                            cond = (value_i >= val_min) | (
                                ii == len(facecolors) - 1)
                            if cond:
                                i.set_color("white")

                            if value_i == 0:
                                facecolors[ii] = fill_value
                                i.set_color("white")

                    except:
                        pass
                ii += 1

        fig.tight_layout()

        if len(savefile) > 0:

            savefile = savefile+'_matrizod'

            print("Nuevos archivos en resultados: ", savefile)

            db_path = os.path.join("resultados", "png", f"{savefile}.png")
            fig.savefig(db_path, dpi=300, bbox_inches="tight")

            db_path = os.path.join("resultados", "pdf", f"{savefile}.pdf")
            fig.savefig(db_path, dpi=300, bbox_inches="tight")

            db_path = os.path.join(
                "resultados", "matrices", f"{savefile}.xlsx")

            if normalize:
                dash_tot = df.copy()
                od1 = pd.crosstab(
                    index=df[zona_origen],
                    columns=df[zona_destino],
                    values=df[var_fex],
                    aggfunc="sum",
                    normalize=False,
                    margins=margins,
                )

                pd.concat(
                    [od1, pd.DataFrame([[], []]), od_heatmap],
                ).to_excel(db_path)

            else:
                od_heatmap.to_excel(path_resultados / (db_path))

        if show_fig:
            display(fig)

        # Guardo datos para el dashboard
        if 'h3_r' not in var_zona:

            conn_dash = iniciar_conexion_db(tipo='dash')

            df = df[[zona_origen, zona_destino, var_fex]].copy()
            df.columns = ['Origen', 'Destino', 'Viajes']

            df['desc_dia'] = desc_dia
            df['tipo_dia'] = tipo_dia
            df['var_zona'] = var_zona.replace('h3_r', 'H3 Resolucion ')
            df['filtro1'] = filtro1

            df_ant = pd.read_sql_query(
                """
                SELECT *
                FROM matrices
                """,
                conn_dash,
            )

            df_ant = df_ant[~(
                (df_ant.desc_dia == desc_dia) &
                (df_ant.tipo_dia == tipo_dia) &
                (df_ant.var_zona == var_zona
                 .replace('h3_r', 'H3 Resolucion ')) &
                (df_ant.filtro1 == filtro1)
            )]

            df = pd.concat([df_ant, df], ignore_index=True)

            if len(matriz_order_row) == 0:
                matriz_order_row = od_heatmap.reset_index()[
                    zona_origen].unique()
            if len(matriz_order_col) == 0:
                matriz_order_col = od_heatmap.columns

            n = 1
            cols = []
            for i in matriz_order_row:
                cols += [str(n).zfill(3)+'_'+str(i)]
                n += 1
            df['Origen'] = df.Origen.replace(matriz_order_row, cols)

            n = 1
            cols = []
            for i in matriz_order_col:
                cols += [str(n).zfill(3)+'_'+str(i)]
                n += 1
            df['Destino'] = df.Destino.replace(matriz_order_row, cols)

            df.to_sql("matrices", conn_dash, if_exists="replace", index=False)
            conn_dash.close()


def lineas_deseo(df,
                 zonas,
                 var_zona,
                 var_fex,
                 h3_o,
                 h3_d,
                 alpha=.4,
                 cmap='viridis_r',
                 porc_viajes=100,
                 title='Líneas de deseo',
                 savefile='lineas_deseo',
                 show_fig=True,
                 normalizo_latlon=True,
                 k_jenks=5,
                 alias='',
                 desc_dia='',
                 tipo_dia='',
                 zona='',
                 filtro1='',
                 ):

    hexs = zonas[(zonas.fex.notna()) & (zonas.fex != 0)]\
        .groupby(var_zona, as_index=False)\
        .size().drop(['size'], axis=1)

    hexs = hexs.merge(
        zonas[(zonas.fex.notna()) & (zonas.fex != 0)]
        .groupby(var_zona)
        .apply(lambda x: np.average(x['longitud'], weights=x['fex']))
        .reset_index()
        .rename(columns={0: 'longitud'}), how='left')

    hexs = hexs.merge(
        zonas[(zonas.fex.notna()) & (zonas.fex != 0)]
        .groupby(var_zona)
        .apply(lambda x: np.average(x['latitud'], weights=x['fex']))
        .reset_index()
        .rename(columns={0: 'latitud'}), how='left')

    tmp_o = f'{var_zona}_o'
    tmp_d = f'{var_zona}_d'

    if 'h3_' in tmp_o:
        tmp_h3_o = tmp_o
        tmp_h3_d = tmp_d
    else:
        tmp_h3_o = h3_o
        tmp_h3_d = h3_d

    # Normalizo con nueva zonificación (ESTO HACE QUE TODOS LOS ORIGENES
    # Y DESTINOS TENGAN UN MISMO SENTIDO)
    if (tmp_o != tmp_h3_o) & (tmp_d != tmp_h3_d):
        df_agg = df.groupby(['dia', tmp_h3_o, tmp_h3_d, tmp_o,
                            tmp_d], as_index=False).agg({var_fex: 'sum'})
    else:
        df_agg = df.groupby(['dia', tmp_h3_o, tmp_h3_d],
                            as_index=False).agg({var_fex: 'sum'})

    if normalizo_latlon:
        df_agg = normalizo_lat_lon(df_agg,
                                   h3_o=tmp_h3_o,
                                   h3_d=tmp_h3_d,
                                   origen=tmp_o,
                                   destino=tmp_d,
                                   )

        tmp_o = f'{var_zona}_o_norm'
        tmp_d = f'{var_zona}_d_norm'

    # Agrego a res de gráfico latlong
    df_agg = df_agg.groupby(['dia', tmp_o, tmp_d], as_index=False).agg(
        {var_fex: 'sum'})

    df_agg = df_agg.groupby([tmp_o, tmp_d], as_index=False).agg(
        {var_fex: 'mean'})

    df_agg[var_fex] = df_agg[var_fex].round().astype(int)

    df_agg = df_agg.merge(
        hexs.rename(columns={var_zona: tmp_o,
                             'latitud': 'lat_o',
                             'longitud': 'lon_o'})
    )
    df_agg = df_agg.merge(
        hexs.rename(columns={var_zona: tmp_d,
                             'latitud': 'lat_d',
                             'longitud': 'lon_d'})
    )

    df_agg = df_agg.sort_values(
        var_fex, ascending=False).reset_index(drop=True)
    df_agg['cumsum'] = round(
        df_agg[var_fex].cumsum() / df_agg[var_fex].sum() * 100)

    df_agg = df_agg[df_agg['cumsum'] <= porc_viajes]

    df_agg = df_agg[df_agg[tmp_o] != df_agg[tmp_d]]

    if len(df_agg) > 0:
        try:
            df_agg = crear_linestring(
                df_agg, 'lon_o', 'lat_o', 'lon_d', 'lat_d')

            multip = df_agg[var_fex].head(1).values[0] / 10

            fig = Figure(figsize=(13.5, 13.5), dpi=150)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            zonas[zonas['h3'].isin(df[h3_o].unique())].to_crs(
                3857).plot(ax=ax, alpha=0)

            # En caso de que no haya suficientes casos para 5 jenks
            try:
                df_agg.to_crs(3857).plot(ax=ax,
                                         alpha=alpha,
                                         cmap=cmap,
                                         lw=df_agg[var_fex]/multip,
                                         column=var_fex,
                                         scheme='FisherJenks',
                                         k=k_jenks,
                                         legend=True,
                                         legend_kwds={
                                             'loc': 'upper right',
                                             'title': 'Viajes',
                                             'fontsize': 8,
                                                'title_fontsize': 10,
                                         }
                                         )
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                                attribution=None, attribution_size=10)

                leg = ax.get_legend()
                # leg._loc = 3
                for lbl in leg.get_texts():
                    label_text = lbl.get_text()
                    lower = label_text.split(',')[0]
                    upper = label_text.split(',')[1]
                    new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'
                    lbl.set_text(new_text)

                title_ = f'{title}: {var_zona}s'

                ax.set_title(title_, fontsize=12)
                ax.add_artist(
                    ScaleBar(1, location='lower right', box_alpha=0, pad=1))
                ax.axis('off')

                fig.tight_layout()

                if len(savefile) > 0:

                    savefile = savefile+'_lineas_deseo'

                    print("Nuevos archivos en resultados: ",
                          savefile)
                    db_path = os.path.join(
                        "resultados", "png", f"{savefile}.png")
                    fig.savefig(db_path, dpi=300, bbox_inches="tight")

                    db_path = os.path.join(
                        "resultados", "pdf", f"{savefile}.pdf")
                    fig.savefig(db_path, dpi=300, bbox_inches="tight")

                    # Guarda geojson para el dashboard
                    # if not 'h3_r' in var_zona:
                    df_folium = df_agg.copy()
                    df_folium.columns = ['Origen', 'Destino', 'Viajes',
                                         'lon_o', 'lat_o', 'lon_d', 'lat_d',
                                         'cumsum', 'geometry']

                    df_folium = df_folium[[
                        'Origen', 'Destino', 'Viajes', 'lon_o', 'lat_o',
                        'lon_d', 'lat_d']]

                    df_folium['desc_dia'] = desc_dia
                    df_folium['tipo_dia'] = tipo_dia
                    df_folium['var_zona'] = var_zona.replace(
                        'h3_r', 'H3 Resolucion ')
                    df_folium['filtro1'] = filtro1

                    conn_dash = iniciar_conexion_db(tipo='dash')
                    var_zona_q = var_zona.replace('h3_r', 'H3 Resolucion ')

                    query = f"""
                    DELETE FROM lineas_deseo
                        WHERE
                        desc_dia = '{desc_dia}' and
                        tipo_dia = '{tipo_dia}' and
                        var_zona = '{var_zona_q}' and
                        filtro1 = '{filtro1}'
                    """

                    conn_dash.execute(query)
                    conn_dash.commit()

                    df_folium.to_sql("lineas_deseo", conn_dash,
                                     if_exists="append", index=False)
                    conn_dash.close()

                    crear_mapa_folium(df_agg,
                                      cmap,
                                      var_fex,
                                      savefile=f"{savefile}.html",
                                      k_jenks=k_jenks)

                if show_fig:
                    display(fig)
            except (ValueError) as e:
                print(e)

        except (ValueError) as e:
            pass


def crea_df_burbujas(df,
                     zonas,
                     h3_o='h3_o',
                     var_fex='',
                     porc_viajes=100,
                     res=7):

    zonas['h3_o_tmp'] = zonas['h3'].apply(h3.h3_to_parent, res=res)

    hexs = zonas[(zonas.fex.notna()) & (zonas.fex != 0)].groupby(
        'h3_o_tmp', as_index=False).size().drop(['size'], axis=1)

    hexs = hexs.merge(
        zonas[(zonas.fex.notna()) & (zonas.fex != 0)]
        .groupby('h3_o_tmp')
        .apply(lambda x: np.average(x['longitud'], weights=x['fex']))
        .reset_index().rename(columns={0: 'longitud'}), how='left')

    hexs = hexs.merge(
        zonas[(zonas.fex.notna()) & (zonas.fex != 0)]
        .groupby('h3_o_tmp')
        .apply(lambda x: np.average(x['latitud'], weights=x['fex']))
        .reset_index()
        .rename(columns={0: 'latitud'}), how='left')

    df['h3_o_tmp'] = df[h3_o].apply(h3.h3_to_parent, res=res)

    # Agrego a res de gráfico latlong
    df_agg = df.groupby(['dia', 'h3_o_tmp'],
                        as_index=False).agg({var_fex: 'sum'})
    df_agg = df_agg.groupby(
        ['h3_o_tmp'], as_index=False).agg({var_fex: 'mean'})

    df_agg = df_agg.merge(
        hexs.rename(columns={'latitud': 'lat_o',
                             'longitud': 'lon_o'})
    )

    df_agg = gpd.GeoDataFrame(
        df_agg,
        geometry=gpd.points_from_xy(df_agg['lon_o'], df_agg['lat_o']),
        crs=4326,)

    df_agg = df_agg.sort_values(
        var_fex, ascending=False).reset_index(drop=True)
    df_agg['cumsum'] = round(df_agg[var_fex].cumsum() /
                             df_agg[var_fex].sum() * 100)
    df_agg = df_agg[df_agg['cumsum'] < porc_viajes]

    return df_agg


def crear_mapa_folium(df_agg,
                      cmap,
                      var_fex,
                      savefile,
                      k_jenks=5):

    bins = [df_agg[var_fex].min()-1] + \
        mapclassify.FisherJenks(df_agg[var_fex], k=k_jenks).bins.tolist()
    range_bins = range(0, len(bins)-1)
    bins_labels = [
        f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins]
    df_agg['cuts'] = pd.cut(df_agg[var_fex], bins=bins, labels=bins_labels)

    from folium import Figure
    fig = Figure(width=800, height=800)
    m = folium.Map(location=[df_agg.lat_o.mean(
    ), df_agg.lon_o.mean()], zoom_start=9, tiles='cartodbpositron')

    title_html = """
    <h3 align="center" style="font-size:20px"><b>Your map title</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    line_w = 0.5

    colors = mcp.gen_color(cmap=cmap, n=k_jenks)

    n = 0
    for i in bins_labels:

        df_agg[df_agg.cuts == i].explore(
            m=m,
            color=colors[n],
            style_kwds={'fillOpacity': 0.3, 'weight': line_w},
            name=i,
            tooltip=False,
        )
        n += 1
        line_w += 3

    folium.LayerControl(name='xx').add_to(m)

    fig.add_child(m)

    db_path = os.path.join("resultados", "html", savefile)
    m.save(db_path)


def save_zones():
    """
    Esta función guarda las geografías de las zonas para el dashboard
    """
    print('Creando zonificación para dashboard')

    configs = leer_configs_generales()

    try:
        zonificaciones = configs['zonificaciones']
    except KeyError:
        zonificaciones = []

    geo_files = [['zona_voi.geojson', 'Zona_voi']]

    if zonificaciones:
        for n in range(0, 5):

            try:
                file_zona = zonificaciones[f"geo{n+1}"]
                var_zona = zonificaciones[f"var{n+1}"]
                geo_files += [[file_zona, var_zona]]

            except KeyError:
                pass

    zonas = pd.DataFrame([])
    for i in geo_files:
        file = os.path.join("data", "data_ciudad", f'{i[0]}')
        if os.path.isfile(file):
            df = gpd.read_file(file)
            df = df[[i[1], 'geometry']]
            df.columns = ['Zona', 'geometry']
            df['tipo_zona'] = i[1]
            zonas = pd.concat([zonas, df])

    zonas = zonas.dissolve(by=['tipo_zona', 'Zona'], as_index=False)
    zonas['wkt'] = zonas.geometry.to_wkt()
    zonas = zonas.drop(['geometry'], axis=1)

    conn_dash = iniciar_conexion_db(tipo='dash')
    zonas.to_sql("zonas", conn_dash, if_exists="replace", index=False)
    conn_dash.close()


def particion_modal(viajes_dia, etapas_dia, tipo_dia, desc_dia):

    particion_viajes = viajes_dia.groupby(
        'modo', as_index=False).factor_expansion_linea.sum().round()
    particion_viajes['modal'] = (particion_viajes['factor_expansion_linea'] /
                                 viajes_dia.factor_expansion_linea.sum() * 100
                                 ).round()
    particion_viajes = particion_viajes.sort_values(
        'modal', ascending=False).drop(['factor_expansion_linea'], axis=1)
    particion_viajes['tipo'] = 'viajes'
    particion_viajes['tipo_dia'] = tipo_dia
    particion_viajes['desc_dia'] = desc_dia
    particion_etapas = etapas_dia.groupby(
        'modo', as_index=False).factor_expansion_linea.sum().round()

    particion_etapas['modal'] = (particion_etapas['factor_expansion_linea'] /
                                 etapas_dia.factor_expansion_linea.sum() * 100
                                 ).round()
    particion_etapas = particion_etapas.sort_values(
        'modal', ascending=False).drop(['factor_expansion_linea'], axis=1)
    particion_etapas['tipo'] = 'etapas'
    particion_etapas['desc_dia'] = desc_dia
    particion_etapas['tipo_dia'] = tipo_dia
    particion = pd.concat(
        [particion_viajes, particion_etapas], ignore_index=True)

    conn_dash = iniciar_conexion_db(tipo='dash')

    query = f'DELETE FROM particion_modal WHERE desc_dia = "{desc_dia}" & tipo_dia = "{tipo_dia}"'
    conn_dash.execute(query)
    conn_dash.commit()
    particion['modo'] = particion.modo.str.capitalize()
    particion.to_sql("particion_modal", conn_dash,
                     if_exists="append", index=False)
    conn_dash.close()


def plot_dispatched_services_wrapper():
    conn_data = iniciar_conexion_db(tipo='data')

    q = """
    select *
    from services_by_line_hour
    where dia = 'weekday';
    """
    service_data = pd.read_sql(q, conn_data)

    if len(service_data) > 0:
        service_data.groupby(['id_linea']).apply(
            plot_dispatched_services_by_line_day)

    conn_data.close()


def plot_dispatched_services_by_line_day(df):
    """
    Reads services' data and plots how many services
    by line, type of day (weekday weekend), and hour.
    Saves it in results dir

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with dispatched services by hour from
        services_by_line_hour table with

    Returns
    -------
    None

    """
    line_id = df.id_linea.unique().item()
    day = df.dia.unique().item()

    if day == 'weekend':
        day_str = 'Fin de semana tipo'
    elif day == 'weekday':
        day_str = 'Dia de semana tipo'
    else:
        day_str = day

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    s = f"select nombre_linea from metadata_lineas" +\
        f" where id_linea = {line_id};"
    id_linea_str = pd.read_sql(s, conn_insumos)
    conn_insumos.close()

    if len(id_linea_str) > 0:
        id_linea_str = id_linea_str.nombre_linea.item()
        id_linea_str = id_linea_str + ' -'
    else:
        id_linea_str = ''

    print("Creando plot de servicios despachados por linea", "id linea:", line_id)

    f, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=df,
        x="hora",
        y="servicios",
        hue="id_linea",
        ax=ax)

    ax.get_legend().remove()
    ax.set_xlabel("Hora")
    ax.set_ylabel("Cantidad de servicios despachados")

    f.suptitle(f"Cantidad de servicios despachados por hora y día",
               fontdict={'size': 18,
                         'weight': 'bold'})
    ax.set_title(f"{id_linea_str} id linea: {line_id} - Dia: {day_str}",
                 fontdict={"fontsize": 11})

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.left.set_position(('outward', 10))
    ax.spines.bottom.set_position(('outward', 10))

    ax.grid(axis='y')

    for frm in ['png', 'pdf']:
        archivo = f'servicios_despachados_id_linea_{line_id}_{day}.{frm}'
        db_path = os.path.join("resultados", frm, archivo)
        f.savefig(db_path, dpi=300)
        plt.close()


def plot_basic_kpi_wrapper():
    sns.set_style("whitegrid")

    conn_data = iniciar_conexion_db(tipo='data')

    q = """
    select *
    from basic_kpi_by_line_hr
    where dia = 'weekday';
    """
    kpi_data = pd.read_sql(q, conn_data)

    if len(kpi_data) > 0:
        kpi_data.groupby(['id_linea']).apply(
            plot_basic_kpi, standarize_supply_demand=False)

    conn_data.close()


def plot_basic_kpi(kpi_by_line_hr, standarize_supply_demand=False,
                   *args, **kwargs):
    line_id = kpi_by_line_hr.id_linea.unique().item()
    day = kpi_by_line_hr.dia.unique().item()
    alias = leer_alias()

    if day == 'weekend':
        day_str = 'Fin de semana tipo'
    elif day == 'weekday':
        day_str = 'Dia de semana tipo'
    else:
        day_str = day

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    s = f"select nombre_linea from metadata_lineas" +\
        f" where id_linea = {line_id};"

    id_linea_str = pd.read_sql(s, conn_insumos)
    conn_insumos.close()

    if len(id_linea_str) > 0:
        id_linea_str = id_linea_str.nombre_linea.item()
        id_linea_str = id_linea_str + ' -'
    else:
        id_linea_str = ''

    # Create empty df with 0 - 23 hrs
    kpi_stats_line_plot = pd.DataFrame(
        {'id_linea': [line_id] * 24, 'hora': range(0, 24)})

    kpi_stats_line_plot = kpi_stats_line_plot\
        .merge(kpi_by_line_hr.query(f"id_linea == {line_id}"),
               on=['id_linea', 'hora'],
               how='left')

    if standarize_supply_demand:
        supply_factor = kpi_stats_line_plot.of.max()\
            / kpi_stats_line_plot.veh.max()
        demand_factor = kpi_stats_line_plot.of.max()\
            / kpi_stats_line_plot.pax.max()
        kpi_stats_line_plot.veh = kpi_stats_line_plot.veh * supply_factor
        kpi_stats_line_plot.pax = kpi_stats_line_plot.pax * demand_factor
        note = """
            Los indicadores de Oferta y Demanda se estandarizaron para que
            coincidan con máximo del eje de Factor de Ocupación
        """
        ylabel_str = "Factor de Ocupación (%)"
    else:
        kpi_stats_line_plot.veh = kpi_stats_line_plot.veh / \
            kpi_stats_line_plot.veh.sum() * 100
        kpi_stats_line_plot.pax = kpi_stats_line_plot.pax / \
            kpi_stats_line_plot.pax.sum() * 100
        note = """
        Oferta y Demanda expresan la distribución porcentual por
        hora de la sumatoria de veh-hr y de los pax-hr 
        respectivamente 
        """
        ylabel_str = "%"
    missing_data = (kpi_stats_line_plot.pax.isna().all()) |\
        (kpi_stats_line_plot.dmt.isna().all()) |\
        (kpi_stats_line_plot.of.isna().all())

    if missing_data:
        print("No es posible crear plot de KPI basicos por linea", "id linea:", line_id)

    else:
        print("Creando plot de KPI basicos por linea", "id linea:", line_id)

        f, ax = plt.subplots(figsize=(8, 6))

        sns.barplot(data=kpi_stats_line_plot, x='hora', y='of',
                    color='silver', ax=ax, label='Factor de ocupación')

        sns.lineplot(data=kpi_stats_line_plot, x="hora", y="veh", ax=ax,
                     color='Purple', label='Oferta')
        sns.lineplot(data=kpi_stats_line_plot, x="hora", y="pax", ax=ax,
                     color='Orange', label='Demanda')

        ax.set_xlabel("Hora")
        ax.set_ylabel(ylabel_str)

        f.suptitle(f"Indicadores de oferta y demanda estadarizados",
                   fontdict={'size': 18,
                             'weight': 'bold'})

        ax.set_title(f"{id_linea_str} id linea: {line_id} - Dia: {day_str}",
                     fontdict={"fontsize": 11})
        # Add a footnote below and to the right side of the chart

        ax_note = ax.annotate(note,
                              xy=(0, -.18),
                              xycoords='axes fraction',
                              ha='left',
                              va="center",
                              fontsize=10)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.left.set_position(('outward', 10))
        ax.spines.bottom.set_position(('outward', 10))

        for frm in ['png', 'pdf']:
            archivo = f'{alias}_kpi_basicos_id_linea_{line_id}_{day}.{frm}'
            db_path = os.path.join("resultados", frm, archivo)
            f.savefig(db_path, dpi=300, bbox_extra_artists=(
                ax_note,), bbox_inches='tight')
            plt.close()

        # add to dash
        kpi_stats_line_plot['nombre_linea'] = id_linea_str
        kpi_stats_line_plot['dia'] = day
        kpi_stats_line_plot = kpi_stats_line_plot\
            .reindex(columns=[
                'dia',
                'id_linea',
                'nombre_linea',
                'hora',
                'veh',
                'pax',
                'dmt',
                'of',
                'speed_kmh']
            )

        conn_dash = iniciar_conexion_db(tipo='dash')

        query = f"""
            DELETE FROM basic_kpi_by_line_hr
            WHERE dia = "{day}"
            and id_linea = "{line_id}"
            """
        conn_dash.execute(query)
        conn_dash.commit()
        
        kpi_stats_line_plot.to_sql(
            "basic_kpi_by_line_hr",
            conn_dash,
            if_exists="append",
            index=False,
        )
        conn_dash.close()


def get_branch_geoms_from_line(id_linea):
    """
    Takes a line id and returns a geoSeries with
    all branches' geoms
    """
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    branch_geoms_query = f"""
        select * from branches_geoms bg 
        where id_ramal in (
            select id_ramal from metadata_ramales mr 
            where id_linea = {id_linea}
        )
        ;
    """
    branch_geoms = pd.read_sql(branch_geoms_query, conn_insumos)
    branch_geoms = gpd.GeoSeries.from_wkt(
        branch_geoms.wkt.values,
        index=branch_geoms.id_ramal.values,
        crs='EPSG:4326')

    conn_insumos.close()

    if len(branch_geoms) == 0:
        branch_geoms = None

    return branch_geoms


def create_squared_polygon(min_x, min_y, max_x, max_y, epsg):

    width = max(max_x - min_x, max_y - min_y)
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    square_bbox_min_x = center_x - width / 2
    square_bbox_min_y = center_y - width / 2
    square_bbox_max_x = center_x + width / 2
    square_bbox_max_y = center_y + width / 2

    square_bbox_coords = [
        (square_bbox_min_x, square_bbox_min_y),
        (square_bbox_max_x, square_bbox_min_y),
        (square_bbox_max_x, square_bbox_max_y),
        (square_bbox_min_x, square_bbox_max_y)
    ]

    p = Polygon(square_bbox_coords)
    s = gpd.GeoSeries([p], crs=f'EPSG:{epsg}')
    return s


def format_num(num, lpad=10):
    fnum = '{:,}'.format(num).replace(
        ".", "*").replace(",", ".").replace("*", ",")
    if lpad > 0:
        fnum = fnum.rjust(lpad, ' ')
    return fnum


def indicadores_dash():
    alias = leer_alias()

    configs = leer_configs_generales()

    conn_data = iniciar_conexion_db(tipo='data')

    indicadores = pd.read_sql_query(
        """
        SELECT *
        FROM indicadores
        """,
        conn_data,
    )

    indicadores['dia'] = pd.to_datetime(indicadores.dia)
    indicadores['dow'] = indicadores.dia.dt.dayofweek
    indicadores['mo'] = indicadores.dia.dt.month
    indicadores['yr'] = indicadores.dia.dt.year

    indicadores['desc_dia'] = indicadores['yr'].astype(str).str.zfill(
        4) + '/' + indicadores['mo'].astype(str).str.zfill(2)
    indicadores['tipo_dia'] = 'Hábil'
    indicadores.loc[indicadores.dow >= 5, 'tipo_dia'] = 'Fin de semana'

    indicadores = indicadores.groupby(['desc_dia', 'tipo_dia', 'detalle'], as_index=False).agg({
        'indicador': 'mean', 'porcentaje': 'mean'})
    indicadores.loc[indicadores.detalle == 'Cantidad de etapas con destinos validados',
                    'detalle'] = 'Transacciones válidas \n(Etapas con destinos validados)'
    indicadores.loc[indicadores.detalle ==
                    'Cantidad total de viajes expandidos', 'detalle'] = 'Viajes'
    indicadores.loc[indicadores.detalle ==
                    'Cantidad total de etapas', 'detalle'] = 'Etapas'
    indicadores.loc[indicadores.detalle ==
                    'Cantidad total de usuarios', 'detalle'] = 'Usuarios'
    indicadores.loc[indicadores.detalle ==
                    'Cantidad de viajes cortos (<5kms)', 'detalle'] = 'Viajes cortos (<5kms)'
    indicadores.loc[indicadores.detalle == 'Cantidad de viajes con transferencia',
                    'detalle'] = 'Viajes con transferencia'

    conn_data.close()

    indicadores.loc[indicadores.detalle.isin(['Cantidad de transacciones totales',
                                              'Cantidad de tarjetas únicas',
                                              'Cantidad de transacciones limpias',
                                              ]), 'orden'] = 1

    indicadores.loc[indicadores.detalle.str.contains(
        'Transacciones válidas'), 'orden'] = 1

    indicadores.loc[indicadores.detalle.isin(['Viajes',
                                              'Etapas',
                                              'Usuarios',
                                              'Viajes cortos (<5kms)',
                                              'Viajes con transferencia',
                                              'Distancia de los viajes (promedio en kms)',
                                              'Distancia de los viajes (mediana en kms)'
                                              ]), 'orden'] = 2

    indicadores.loc[indicadores.detalle.isin(['Viajes autobus',
                                              'Viajes Multietapa',
                                              'Viajes Multimodal',
                                              'Viajes metro',
                                              'Viajes tren'
                                              ]), 'orden'] = 3

    indicadores['Valor'] = indicadores.indicador.apply(format_num)
    indicadores['porcentaje'] = indicadores.porcentaje.apply(format_num)

    indicadores = indicadores[indicadores.orden.notna()]

    indicadores.loc[~(indicadores.detalle.str.contains('Distancia')), 'Valor'] = indicadores.loc[~(
        indicadores.detalle.str.contains('Distancia')), 'Valor'].str.split(',').str[0]

    indicadores = indicadores.drop(['indicador'], axis=1)
    indicadores = indicadores.rename(columns={'detalle': 'Indicador'})

    indicadores.loc[indicadores.Indicador.str.contains('Transacciones válidas'),
                    'Valor'] += ' ('+indicadores.loc[
        indicadores.Indicador.str.contains('Transacciones válidas'),
        'porcentaje'].str.replace(' ', '')+'%)'

    indicadores.loc[indicadores.orden == 3,
                    'Valor'] += ' ('+indicadores.loc[indicadores.orden == 3,
                                                     'porcentaje'].str.replace(' ', '')+'%)'

    indicadores.loc[indicadores.Indicador == 'Viajes cortos (<5kms)',
                    'Valor'] += ' ('+indicadores.loc[
        indicadores.Indicador == 'Viajes cortos (<5kms)', 'porcentaje'].str.replace(' ', '')+'%)'
    indicadores.loc[indicadores.Indicador == 'Viajes con transferencia',
                    'Valor'] += ' ('+indicadores.loc[
        indicadores.Indicador == 'Viajes con transferencia', 'porcentaje'].str.replace(' ', '')+'%)'

    indicadores.loc[indicadores.orden == 1,
                    'Titulo'] = 'Información del dataset original'
    indicadores.loc[indicadores.orden == 2, 'Titulo'] = 'Información procesada'
    indicadores.loc[indicadores.orden == 3, 'Titulo'] = 'Partición modal'

    conn_dash = iniciar_conexion_db(tipo='dash')
    indicadores.to_sql("indicadores", conn_dash,
                       if_exists="replace", index=False)
    conn_dash.close()
    
@duracion
def create_visualizations():
    """
    Esta funcion corre las diferentes funciones de visualizaciones
    """

    pd.options.mode.chained_assignment = None

    # Leer informacion de viajes y distancias
    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    viajes = pd.read_sql_query(
        """
        SELECT *
        FROM viajes
        where od_validado==1
        """,
        conn_data,
    )

    etapas = pd.read_sql_query(
        """
        SELECT *
        FROM etapas
        where od_validado==1
        """,
        conn_data,
    )

    distancias = pd.read_sql_query(
        """
        SELECT *
        FROM distancias
        """,
        conn_insumos,
    )

    conn_insumos.close()
    conn_data.close()

    # Agrego campo de distancias de los viajes
    viajes = viajes.merge(distancias,
                          how='left',
                          on=['h3_o', 'h3_d'])

    # Imputar anio, mes y tipo de dia
    viajes['yr'] = pd.to_datetime(viajes.dia).dt.year
    viajes['mo'] = pd.to_datetime(viajes.dia).dt.month
    viajes['dow'] = pd.to_datetime(viajes.dia).dt.day_of_week
    viajes.loc[viajes.dow >= 5, 'tipo_dia'] = 'Fin de semana'
    viajes.loc[viajes.dow < 5, 'tipo_dia'] = 'Día hábil'

    # Imputar anio, mes y tipo de dia
    etapas['yr'] = pd.to_datetime(etapas.dia).dt.year
    etapas['mo'] = pd.to_datetime(etapas.dia).dt.month
    etapas['dow'] = pd.to_datetime(etapas.dia).dt.day_of_week
    etapas.loc[etapas.dow >= 5, 'tipo_dia'] = 'Fin de semana'
    etapas.loc[etapas.dow < 5, 'tipo_dia'] = 'Día hábil'

    v_iter = viajes\
        .groupby(['yr', 'mo', 'tipo_dia'], as_index=False)\
        .factor_expansion_linea.sum()\
        .iterrows()

    for _, i in v_iter:

        desc_dia = f'{str(i.mo).zfill(2)}/{i.yr} ({i.tipo_dia})'
        desc_dia_file = f'{i.yr}-{str(i.mo).zfill(2)}({i.tipo_dia})'

        viajes_dia = viajes[(viajes.yr == i.yr) & (
            viajes.mo == i.mo) & (viajes.tipo_dia == i.tipo_dia)]

        etapas_dia = etapas[(etapas.yr == i.yr) & (
            etapas.mo == i.mo) & (etapas.tipo_dia == i.tipo_dia)]

        # partición modal
        particion_modal(viajes_dia, etapas_dia, tipo_dia=i.tipo_dia, desc_dia=f'{str(i.mo).zfill(2)}/{i.yr}')

        print('Imprimiendo tabla de matrices OD')
        # Impirmir tablas con matrices OD
        imprimir_matrices_od(viajes=viajes_dia,
                             var_fex='factor_expansion_linea',
                             title=f'Matriz OD {desc_dia}',
                             savefile=f'{desc_dia_file}',
                             desc_dia=f'{str(i.mo).zfill(2)}/{i.yr}',
                             tipo_dia=i.tipo_dia,
                             )

        print('Imprimiendo mapas de líneas de deseo')
        # Imprimir lineas de deseo
        imprime_lineas_deseo(df=viajes_dia,
                             h3_o='',
                             h3_d='',
                             var_fex='factor_expansion_linea',
                             title=f'Líneas de deseo {desc_dia}',
                             savefile=f'{desc_dia_file}',
                             desc_dia=f'{str(i.mo).zfill(2)}/{i.yr}',
                             tipo_dia=i.tipo_dia)

        print('Imprimiendo gráficos')
        titulo = f'Cantidad de viajes en transporte público {desc_dia}'
        imprime_graficos_hora(viajes_dia,
                              title=titulo,
                              savefile=f'{desc_dia_file}_viajes',
                              var_fex='factor_expansion_linea',
                              desc_dia=f'{str(i.mo).zfill(2)}/{i.yr}',
                              tipo_dia=i.tipo_dia)

        print('Imprimiendo mapas de burbujas')
        viajes_n = viajes_dia[(viajes_dia.id_viaje > 1)]
        imprime_burbujas(viajes_n,
                         res=7,
                         h3_o='h3_o',
                         alpha=.4,
                         cmap='rocket_r',
                         var_fex='factor_expansion_linea',
                         porc_viajes=100,
                         title=f'Destinos de los viajes {desc_dia}',
                         savefile=f'{desc_dia_file}_burb_destinos',
                         show_fig=False,
                         k_jenks=5)

        viajes_n = viajes_dia[(viajes_dia.id_viaje == 1)]
        imprime_burbujas(viajes_n,
                         res=7,
                         h3_o='h3_o',
                         alpha=.4,
                         cmap='flare',
                         var_fex='factor_expansion_linea',
                         porc_viajes=100,
                         title=f'Hogares {desc_dia}',
                         savefile=f'{desc_dia_file}_burb_hogares',
                         show_fig=False,
                         k_jenks=5)

    save_zones()

    print('Indicadores para dash')
    indicadores_dash()

    # plor dispatched services
    plot_dispatched_services_wrapper()

    # plot basic kpi if exists
    plot_basic_kpi_wrapper()

