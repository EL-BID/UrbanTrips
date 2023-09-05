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
    if 'modo' in tabla.columns:
        tabla['modo'] = tabla['modo'].str.capitalize()


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

def crear_mapa_folium(df_agg,
                      cmap,
                      var_fex,
                      savefile='',
                      k_jenks=5):

    bins = [df_agg[var_fex].min()-1] + \
        mapclassify.FisherJenks(df_agg[var_fex], k=k_jenks).bins.tolist()
    range_bins = range(0, len(bins)-1)
    bins_labels = [
        f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins]
    df_agg['cuts'] = pd.cut(df_agg[var_fex], bins=bins, labels=bins_labels)

    
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
            style_kwds={'fillOpacity': 0.1, 'weight': line_w},
            name=i,
            tooltip=False,
        )
        n += 1
        line_w += 3

    folium.LayerControl(name='xx').add_to(m)

    fig.add_child(m)

    return fig



st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)
   

with st.expander ('Partición modal'):
    
    col1, col2, col3 = st.columns([1, 3, 3])
    particion_modal = levanto_tabla_sql('particion_modal') 
    desc_dia_m = col1.selectbox('Periodo', options=particion_modal.desc_dia.unique(), key='desc_dia_m')
    tipo_dia_m = col1.selectbox('Tipo de día', options=particion_modal.tipo_dia.unique(), key='tipo_dia_m')
    
    # Etapas
    particion_modal_etapas = particion_modal[(particion_modal.desc_dia==desc_dia_m)&(particion_modal.tipo_dia==tipo_dia_m)&(particion_modal.tipo=='etapas')]    
    if col2.checkbox('Ver datos: etapas'):
        col2.write(particion_modal_etapas)
    fig2 = px.bar(particion_modal_etapas, x='modo', y='modal')
    fig2.update_layout(title_text='Partición modal de Etapas')
    fig2.update_xaxes(title_text='Modo')
    fig2.update_yaxes(title_text='Partición modal (%)')
    fig2.update_traces(marker_color='brown')
    col2.plotly_chart(fig2)
    
    # Viajes
    particion_modal_viajes = particion_modal[(particion_modal.desc_dia==desc_dia_m)&(particion_modal.tipo_dia==tipo_dia_m)&(particion_modal.tipo=='viajes')]        
    if col3.checkbox('Ver datos: viajes'):
        col3.write(particion_modal_viajes)   
    fig = px.bar(particion_modal_viajes, x='modo', y='modal')
    fig.update_layout(title_text='Partición modal de Viajes')
    fig.update_xaxes(title_text='Modo')
    fig.update_yaxes(title_text='Partición modal (%)')
    fig.update_traces(marker_color='navy')
    col3.plotly_chart(fig)



with st.expander ('Distancias de viajes'):
    
    col1, col2 = st.columns([1, 4])
    
    hist_values = levanto_tabla_sql('distribucion') 
    hist_values.columns = ['desc_dia', 'tipo_dia', 'Distancia (kms)', 'Viajes', 'Modo']
    hist_values = hist_values[hist_values['Distancia (kms)']<=60]
    hist_values = hist_values.sort_values(['Modo', 'Distancia (kms)'])
    
    if col2.checkbox('Ver datos: distribución de viajes'):
        col2.write(hist_values)
    
    desc_dia_d = col1.selectbox('Periodo', options=hist_values.desc_dia.unique(), key='desc_dia_d')
    tipo_dia_d = col1.selectbox('Tipo de dia', options=hist_values.tipo_dia.unique(), key='tipo_dia_d')
    modo_d = col1.selectbox('Modo', options=hist_values.Modo.unique())
    
    hist_values = hist_values[(hist_values.desc_dia==desc_dia_d)&(hist_values.tipo_dia==tipo_dia_d)&(hist_values.Modo==modo_d)]
        
    fig = px.histogram(hist_values, x='Distancia (kms)', y='Viajes', nbins=len(hist_values))
    fig.update_xaxes(type='category')
    fig.update_yaxes(title_text='Viajes')
    
    fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tickangle=0,
                tickfont=dict(size=9)
            ),
            yaxis=dict(
                tickfont=dict(size=9)
            )
        )
    
    col2.plotly_chart(fig)
    
    
with st.expander ('Viajes por hora'):
    
    col1, col2 = st.columns([1, 4])
    
    viajes_hora = levanto_tabla_sql('viajes_hora') 
        
    desc_dia_h = col1.selectbox('Periodo', options=viajes_hora.desc_dia.unique(), key='desc_dia_h')
    tipo_dia_h = col1.selectbox('Tipo de dia', options=viajes_hora.tipo_dia.unique(), key='tipo_dia_h')
    modo_h = col1.selectbox('Modo', options=['Todos', 'Por modos'], key='modo_h')
    
    if modo_h=='Todos':
        viajes_hora = viajes_hora[(viajes_hora.desc_dia==desc_dia_h)&(viajes_hora.tipo_dia==tipo_dia_h)&(viajes_hora.Modo=='Todos')]
    else:
        viajes_hora = viajes_hora[(viajes_hora.desc_dia==desc_dia_h)&(viajes_hora.tipo_dia==tipo_dia_h)&(viajes_hora.Modo!='Todos')]

    viajes_hora = viajes_hora.sort_values('Hora')
    if col2.checkbox('Ver datos: viajes por hora'):
        col2.write(viajes_hora)

        
    fig_horas = px.line(viajes_hora, x="Hora", y="Viajes", color='Modo', symbol="Modo")
    
    fig_horas.update_xaxes(type='category')
    # fig_horas.update_layout()
    
    col2.plotly_chart(fig_horas)
    
    


with st.expander ('Líneas de deseo'):
    
    col1, col2 = st.columns([1, 4])

    lineas_deseo = levanto_tabla_sql('lineas_deseo', 
                                  has_linestring=True) 

    desc_dia = col1.selectbox('Periodo', options=lineas_deseo.desc_dia.unique())
    tipo_dia = col1.selectbox('Tipo de dia', options=lineas_deseo.tipo_dia.unique())
    var_zona = col1.selectbox('Zonificación', options=lineas_deseo.var_zona.unique())
    filtro1 = col1.selectbox('Filtro', options=lineas_deseo.filtro1.unique())


    df_agg = lineas_deseo[(
                       (lineas_deseo.desc_dia==desc_dia)&
                       (lineas_deseo.tipo_dia==tipo_dia)&
                       (lineas_deseo.var_zona==var_zona)&
                        (lineas_deseo.filtro1==filtro1)
                      )].copy()


    if len(df_agg)>0:

        map = crear_mapa_folium(df_agg,
                                          cmap='BuPu',
                                          var_fex='Viajes',
                                          k_jenks=5)

        with col2:
            st_map = st_folium(map, width=900, height=700)
    else:

        col2.markdown("""
        <style>
        .big-font {
            font-size:40px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        col2.markdown('<p class="big-font">            ¡¡ No hay datos para mostrar !!</p>', unsafe_allow_html=True)

    
with st.expander ('Matrices OD'):
    col1, col2 = st.columns([1, 4])
    
    matriz = levanto_tabla_sql('matrices') 
    
    if len(matriz) > 0:             
        
        if col1.checkbox('Normalizar', value=True):
            normalize=True
        else:
            normalize=False

        
        desc_dia_ = col1.selectbox('Periodo ', options=matriz.desc_dia.unique())
        tipo_dia_ = col1.selectbox('Tipo de dia ', options=matriz.tipo_dia.unique())
        var_zona_ = col1.selectbox('Zonificación ', options=matriz.var_zona.unique())
        filtro1_ = col1.selectbox('Filtro ', options=matriz.filtro1.unique())

        matriz = matriz[(  (matriz.desc_dia==desc_dia_)&
                           (matriz.tipo_dia==tipo_dia_)&
                           (matriz.var_zona==var_zona_)&
                            (matriz.filtro1==filtro1_)
                          )].copy()

        od_heatmap = pd.crosstab(
                index=matriz['Origen'],
                columns=matriz['Destino'],
                values=matriz['Viajes'],
                aggfunc="sum",
                normalize=normalize,
            )
        od_heatmap = (od_heatmap * 100).round(1)
        
        
        od_heatmap = od_heatmap.reset_index()
        od_heatmap['Origen'] = od_heatmap['Origen'].str[4:]
        od_heatmap = od_heatmap.set_index('Origen')
        od_heatmap.columns = [i[4:] for i in od_heatmap.columns]
        
        fig = px.imshow(od_heatmap, text_auto=True, color_continuous_scale='Blues',)
        
        fig.update_coloraxes(showscale=False)

        
        if len(od_heatmap)<=20:
            fig.update_layout(width=800, height=800)
        elif (len(od_heatmap)>20)&(len(od_heatmap)<=40):
            fig.update_layout(width=1000, height=1000)
        elif len(od_heatmap)>40:
            fig.update_layout(width=1200, height=1200)
            
        col2.plotly_chart(fig)

    else:
        st.write('No hay datos para mostrar')

    zonas = levanto_tabla_sql('zonas', has_wkt=True)
    zonas = zonas[zonas.tipo_zona == var_zona_]
    
    col1, col2 = st.columns([1, 4])
    
    if col1.checkbox('Mostrar zonificacion'):
        
        # Create a folium map centered on the data
        map_center = [zonas.geometry.centroid.y.mean(), zonas.geometry.centroid.x.mean()]

        fig = Figure(width=800, height=800)
        m = folium.Map(location=map_center, zoom_start=10, tiles='cartodbpositron')

        # Add GeoDataFrame to the map
        folium.GeoJson(zonas).add_to(m)

        for idx, row in zonas.iterrows():
            detail = row['Zona']  # Replace 'column_name' with the name of the column containing the detail
            point = [row['geometry'].representative_point().y, row['geometry'].representative_point().x]
            marker = folium.Marker(location=point, popup=detail)
            marker.add_to(m)

        # Display the map using folium_static
        with col2:
            folium_static(m)