import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from PIL import Image
import requests
import mapclassify
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from mycolorpy import colorlist as mcp
import os

import yaml
import sqlite3
from shapely import wkt
from folium import Figure
from shapely.geometry import LineString

def create_linestring(df, 
                      lat_o='lat_o', 
                      lon_o='lon_o', 
                      lat_d='lat_d', 
                      lon_d='lon_d'):

    # Create LineString objects from the coordinates
    geometry = [LineString([(row['lon_o'], row['lat_o']),
                           (row['lon_d'], row['lat_d'])])
                for _, row in df.iterrows()]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    return gdf

def leer_configs_generales():
    """
    Esta funcion lee los configs generales
    """
    path = os.path.join("configs", "configuraciones_generales.yaml")

    try:
        with open(path, 'r', encoding="utf8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        print(f'Error al leer el archivo de configuracion: {error}')

    return config

def leer_alias(tipo='data'):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve el alias seteado en el archivo de congifuracion
    """
    configs = leer_configs_generales()
    # Setear el tipo de key en base al tipo de datos
    if tipo == 'data':
        key = 'alias_db_data'
    elif tipo == 'insumos':
        key = 'alias_db_insumos'
    elif tipo == 'dash':
        key = 'alias_db_data'
    else:
        raise ValueError('tipo invalido: %s' % tipo)
    # Leer el alias
    try:
        alias = configs[key] + '_'
    except KeyError:
        alias = ''
    return alias

def traigo_db_path(tipo='data'):
    """
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve el path a una base de datos con esa informacion
    """
    if tipo not in ('data', 'insumos', 'dash'):
        raise ValueError('tipo invalido: %s' % tipo)

    alias = leer_alias(tipo)
    db_path = os.path.join("data", "db", f"{alias}{tipo}.sqlite")

    return db_path

def iniciar_conexion_db(tipo='data'):
    """"
    Esta funcion toma un tipo de datos (data o insumos)
    y devuelve una conexion sqlite a la db
    """
    db_path = traigo_db_path(tipo)
    assert os.path.isfile(db_path), f'No existe la base de datos para el dashboard en {db_path}'
    conn = sqlite3.connect(db_path, timeout=10)
    return conn

@st.cache_data
def levanto_tabla_sql(tabla_sql, 
                      has_linestring=False,
                      has_wkt=False):

    conn_dash = iniciar_conexion_db(tipo='dash')

    tabla = pd.read_sql_query(
        f"""
        SELECT *
        FROM {tabla_sql}
        """,
        conn_dash,
    )

    conn_dash.close()
    
    if has_linestring:
        tabla = create_linestring(tabla)
        
    if has_wkt:
        tabla["geometry"] = tabla.wkt.apply(wkt.loads)
        tabla = gpd.GeoDataFrame(tabla, 
                                   crs=4326)
        tabla = tabla.drop(['wkt'], axis=1)

    if 'dia' in tabla.columns:
        tabla.loc[tabla.dia=='weekday', 'dia'] = 'Día hábil'
        tabla.loc[tabla.dia=='weekend', 'dia'] = 'Fin de semana'
    if 'day_type' in tabla.columns:
        tabla.loc[tabla.day_type=='weekday', 'day_type'] = 'Día hábil'
        tabla.loc[tabla.day_type=='weekend', 'day_type'] = 'Fin de semana'

    if 'nombre_linea' in tabla.columns:
        tabla['nombre_linea'] = tabla['nombre_linea'].str.replace(' -', '')
    if 'Modo' in tabla.columns:
        tabla['Modo'] = tabla['Modo'].str.capitalize()

    return tabla

@st.cache_data
def get_logo():
    file_logo = os.path.join(
        "docs", "urbantrips_logo.jpg")
    if not os.path.isfile(file_logo):
        # URL of the image file on Github
        url = 'https://github.com/EL-BID/UrbanTrips/blob/18be313301c979dae5fd27ac5b83f89c76e2dd5f/docs/urbantrips_logo.jpg'

        # Send a request to get the content of the image file
        response = requests.get(url)

        # Save the content to a local file
        with open(file_logo, 'wb') as f:
            f.write(response.content)
    image = Image.open(file_logo)
    return image

def plot_lineas(lineas, id_linea, nombre_linea, day_type, n_sections, rango):

    gdf = lineas[(lineas.id_linea == id_linea)&
                    (lineas.day_type == day_type)&
                    (lineas.n_sections == n_sections)].copy()

    gdf_d0 = lineas[(lineas.id_linea == id_linea)&
                    (lineas.day_type == day_type)&
                    (lineas.n_sections == n_sections)&
                    (lineas.sentido=='ida')].copy()

    gdf_d1 = lineas[(lineas.id_linea == id_linea)&
                    (lineas.day_type == day_type)&
                    (lineas.n_sections == n_sections)&
                    (lineas.sentido=='vuelta')].copy()
    
    indicator = 'prop_etapas'
    
    gdf_d0[indicator] = (gdf_d0['cantidad_etapas'] / gdf_d0['cantidad_etapas'].sum() * 100).round(2)
    gdf_d1[indicator] = (gdf_d1['cantidad_etapas'] / gdf_d1['cantidad_etapas'].sum() * 100).round(2)

    
    # creating plot

    f = plt.figure(tight_layout=True, figsize=(18, 10), dpi=8)
    gs = f.add_gridspec(nrows=3, ncols=2)
    ax1 = f.add_subplot(gs[0:2, 0])
    ax2 = f.add_subplot(gs[0:2, 1])
    ax3 = f.add_subplot(gs[2, 0])
    ax4 = f.add_subplot(gs[2, 1])

    font_dicc = {'fontsize': 18,
                 'fontweight': 'bold'}

    gdf_d0.plot(ax=ax1, color='purple', alpha=.7, linewidth=gdf_d0[indicator])
    gdf_d1.plot(ax=ax2, color='orange', alpha=.7, linewidth=gdf_d1[indicator])

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

    if nombre_linea == '':
        title = f"Id línea {id_linea} - {day_type}\n{rango}"
    else:
        title = f"Línea: {nombre_linea.replace('Línea ', '')} - Id línea: {id_linea} - {day_type}\n{rango}"

    
    f.suptitle(title, fontsize=20)

    # Matching bar plot with route direction
    flecha_eo_xy = (0.4, 1.1)
    flecha_eo_text_xy = (0.05, 1.1)
    flecha_oe_xy = (0.6, 1.1)
    flecha_oe_text_xy = (0.95, 1.1)

    labels_eo = [''] * len(gdf_d0)
    labels_eo[0] = 'INICIO'
    labels_eo[-1] = 'FIN'
    labels_oe = [''] * len(gdf_d0)
    labels_oe[-1] = 'INICIO'
    labels_oe[0] = 'FIN'
    
    
    # Arrows
    flecha_ida_wgs84 = gdf_d0.loc[gdf_d0.section_id == 0.0, 'geometry']
    flecha_ida_wgs84 = list(flecha_ida_wgs84.item().coords)
    flecha_ida_inicio_wgs84 = flecha_ida_wgs84[0]
    flecha_ida_fin_wgs84 = flecha_ida_wgs84[1]

    flecha_vuelta_wgs84 = gdf_d1.loc[gdf_d1.section_id ==
                                  max(gdf_d1.section_id), 'geometry']
    flecha_vuelta_wgs84 = list(flecha_vuelta_wgs84.item().coords)
    flecha_vuelta_inicio_wgs84 = flecha_vuelta_wgs84[0]
    flecha_vuelta_fin_wgs84 = flecha_vuelta_wgs84[1]
    

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
        gdf_d0 = gdf_d0.sort_values('section_id', ascending=True)
        gdf_d1 = gdf_d1.sort_values('section_id', ascending=True)

    else:
        flecha_ida_xy = flecha_oe_xy
        flecha_ida_text_xy = flecha_oe_text_xy
        labels_ida = labels_oe

        flecha_vuelta_xy = flecha_eo_xy
        flecha_vuelta_text_xy = flecha_eo_text_xy
        labels_vuelta = labels_eo

        gdf_d0 = gdf_d0.sort_values('section_id', ascending=False)
        gdf_d1 = gdf_d1.sort_values('section_id', ascending=False)

    sns.barplot(data=gdf_d0, x="section_id",
                y=indicator, ax=ax3, color='Purple',
                order=gdf_d0.section_id.values)

    sns.barplot(data=gdf_d1, x="section_id",
                y=indicator, ax=ax4, color='Orange',
                order=gdf_d1.section_id.values)

    # Axis
    ax3.set_xticklabels(labels_ida)
    ax4.set_xticklabels(labels_vuelta)

    ax3.set_ylabel(y_axis_lable)
    ax3.set_xlabel('')

    ax4.get_yaxis().set_visible(False)

    ax4.set_ylabel('')
    ax4.set_xlabel('')
    max_y_barplot = max(gdf_d0[indicator].max(), gdf_d1[indicator].max())
    ax3.set_ylim(0, max_y_barplot)
    ax4.set_ylim(0, max_y_barplot)

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax4.spines.left.set_visible(False)
    ax4.spines.right.set_visible(False)
    ax4.spines.top.set_visible(False)

    # For direction 0, get the last section of the route geom
    flecha_ida = gdf_d0.loc[gdf_d0.section_id == max(gdf_d0.section_id), 'geometry']
    flecha_ida = list(flecha_ida.item().coords)
    flecha_ida_inicio = flecha_ida[1]
    flecha_ida_fin = flecha_ida[0]

    # For direction 1, get the first section of the route geom
    flecha_vuelta = gdf_d1.loc[gdf_d1.section_id == 0.0, 'geometry']
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

    cx.add_basemap(ax1, crs=gdf_d0.crs.to_string(), source=prov, attribution_size=7)
    cx.add_basemap(ax2, crs=gdf_d1.crs.to_string(), source=prov, attribution_size=7)

    plt.close(f)
    return f

@st.cache_data
def traigo_nombre_lineas(df):
    return df[df.nombre_linea.notna()].sort_values('nombre_linea').nombre_linea.unique()
    

st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)


with st.expander ('Cargas por horas'):
    col1, col2 = st.columns([1, 4])
    
    kpi_lineas = levanto_tabla_sql('basic_kpi_by_line_hr')
    nl1 = traigo_nombre_lineas(kpi_lineas)
    
    if len(kpi_lineas) > 0:
        if len(nl1) > 0:
            nombre_linea_kpi = col1.selectbox('Línea  ', options=nl1)
            id_linea_kpi = kpi_lineas[kpi_lineas.nombre_linea==nombre_linea_kpi].id_linea.values[0]
        else:
            nombre_linea_kpi = ''
            id_linea_kpi = col1.selectbox('Línea ', options=kpi_lineas.id_linea.unique())

    day_type_kpi = col1.selectbox('Tipo de dia  ', options=kpi_lineas.dia.unique())

    kpi_stats_line_plot = kpi_lineas[(kpi_lineas.id_linea == id_linea_kpi)&(kpi_lineas.dia == day_type_kpi)]

    # if col2.checkbox('Ver datos: cargas por hora'):
    #     col2.write(kpi_stats_line_plot)

    if len(kpi_stats_line_plot) > 0:

        # Grafico Factor de Oocupación
        f, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=kpi_stats_line_plot, x='hora', y='of',
                    color='silver', ax=ax, label='Factor de ocupación')
        
        sns.lineplot(data=kpi_stats_line_plot, x="hora", y="veh", ax=ax,
                     color='Purple', label='Oferta - veh/hr')
        sns.lineplot(data=kpi_stats_line_plot, x="hora", y="pax", ax=ax,
                     color='Orange', label='Demanda - pax/hr')
        
        ax.set_xlabel("Hora")
        ax.set_ylabel("Factor de Ocupación (%)")
        
        ax.set_title(f"Indicadores de oferta y demanda estadarizados\nLínea: {nombre_linea_kpi.replace('Línea ', '')} - Id linea: {id_linea_kpi} - {day_type_kpi}", 
                     fontdict={'size': 12})
        
        # Add a footnote below and to the right side of the chart
        note = """
            *Los indicadores de Oferta y Demanda se estandarizaron para que
            coincidan con el eje de Factor de Ocupación
            """
        ax_note = ax.annotate(note,
                              xy=(0, -.22),
                              xycoords='axes fraction',
                              ha='left',
                              va="center",
                              fontsize=7)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.left.set_position(('outward', 10))
        ax.spines.bottom.set_position(('outward', 10))
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # Put a legend to the right of the current axis
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        col2.pyplot(f)
    else:
        st.write('No hay datos para mostrar')
    
with st.expander ('Cargas por tramos'):

    col1, col2 = st.columns([1, 4])
    
    lineas = levanto_tabla_sql('ocupacion_por_linea_tramo', has_wkt=True)
    nl2 = traigo_nombre_lineas(lineas[['id_linea', 'nombre_linea']])
    
    lineas['rango'] = 'de '+lineas['hora_min'].astype(str)+' a '+ lineas['hora_max'].astype(str) + ' hs'
    if len(lineas) > 0:
        if len(nl2) > 0:
            nombre_linea = col1.selectbox('Línea  ', options=nl2)
            id_linea = lineas[lineas.nombre_linea==nombre_linea].id_linea.values[0]
        else:
            nombre_linea = ''
            id_linea = col1.selectbox('Línea ', options=lineas.id_linea.unique())
            
        day_type = col1.selectbox('Tipo de dia ', options=lineas.day_type.unique())
        n_sections = col1.selectbox('Secciones ', options=lineas.n_sections.unique())    
        rango = col1.selectbox('Rango horario ', options=lineas.rango.unique())    
        
        lineas = lineas[(lineas.id_linea==id_linea)&(lineas.day_type==day_type)&(lineas.n_sections==n_sections)&(lineas.rango==rango)]

        # if col2.checkbox('Ver datos: cargas por tramos'):
        #     col2.write(lineas)

        if len(lineas) > 0:        
            f_lineas = plot_lineas(lineas, id_linea, nombre_linea, day_type, n_sections, rango)
            col2.pyplot(f_lineas)
        else:
            st.write('No hay datos para mostrar')
    
