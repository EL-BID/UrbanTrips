import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mapclassify
import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import requests
from PIL import Image
from shapely import wkt
import yaml
import sqlite3
from shapely import wkt
from folium import Figure
from shapely.geometry import LineString

from dash_utils import levanto_tabla_sql, get_logo, traigo_indicadores


st.set_page_config(layout="wide")

st.sidebar.success('Seleccione página')

logo = get_logo()
st.image(logo)


st.markdown('<div style="text-align: justify;">urbantrips es una biblioteca de código abierto que toma información de un sistema de pago con tarjeta inteligente de transporte público y, a través de un procesamiento de la información que infiere destinos de los viajes y construye las cadenas de viaje para cada usuario, produce matrices de origen-destino y otros indicadores (KPI) para rutas de autobús. El principal objetivo de la librería es producir insumos útiles para la gestión del transporte público a partir de requerimientos mínimos de información y pre-procesamiento. Con sólo una tabla geolocalizada de transacciones económicas proveniente de un sistema de pago electrónico, se podrán generar resultados, que serán más precisos cuanto más información adicional se incorpore al proceso a través de los archivos opcionales. El proceso elabora las matrices, los indicadores y construye una serie de gráficos y mapas de transporte.</div>', unsafe_allow_html=True)
st.text('')

col1, col2, col3 = st.columns([1, 3, 3])


indicadores = levanto_tabla_sql('indicadores')

if len(indicadores) > 0:
    desc_dia_i = col1.selectbox(
        'Periodo', options=indicadores.desc_dia.unique(), key='desc_dia_i')
    tipo_dia_i = col1.selectbox(
        'Tipo de dia', options=indicadores.tipo_dia.unique(), key='tipo_dia_i')
    
    
    indicadores = indicadores[(indicadores.desc_dia == desc_dia_i) & (
        indicadores.tipo_dia == tipo_dia_i)]
    
    df = indicadores.loc[indicadores.orden == 1, ['Indicador', 'Valor']].copy()
    titulo = indicadores.loc[indicadores.orden == 1].Titulo.unique()[0]
    
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    
    # Inject CSS with Markdown
    col2.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    col2.text(titulo)
    col2.table(df)
    
    
    df = indicadores.loc[indicadores.orden == 2, ['Indicador', 'Valor']].copy()
    titulo = indicadores.loc[indicadores.orden == 2].Titulo.unique()[0]
    
    col3.text(titulo)
    
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    
    # Inject CSS with Markdown
    col3.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    
    col3.table(df)
    
    df = indicadores.loc[indicadores.orden == 3, ['Indicador', 'Valor']].copy()
    titulo = indicadores.loc[indicadores.orden == 3].Titulo.unique()[0]
    
    col2.text(titulo)
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    
    # Inject CSS with Markdown
    col2.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    col2.table(df)
else:

    # Usar HTML para personalizar el estilo del texto
    texto_html = """
        <style>
        .big-font {
            font-size:30px !important;
            font-weight:bold;
        }
        </style>
        <div class='big-font'>
            No hay datos de indicadores            
        </div>
        """   
    col2.markdown(texto_html, unsafe_allow_html=True)
    texto_html = """
        <style>
        .big-font {
            font-size:30px !important;
            font-weight:bold;
        }
        </style>
        <div class='big-font'>
            Verifique que los procesos se corrieron correctamente            
        </div>
        """   
    col2.markdown(texto_html, unsafe_allow_html=True)

with st.expander('Indicadores'):
    col1, col2, col3 = st.columns([2, 2, 2])

    general, modal, distancias = traigo_indicadores('all')

    st.session_state.general_ = general.loc[general.mes==desc_dia_i, ['Tipo', 'Indicador', 'Valor']].set_index('Tipo')
    st.session_state.modal_ = modal.loc[modal.mes==desc_dia_i, ['Tipo', 'Indicador', 'Valor']].set_index('Tipo')
    st.session_state.distancias_ = distancias.loc[distancias.mes==desc_dia_i, ['Tipo', 'Indicador', 'Valor']].set_index('Tipo')


    if len(st.session_state.etapas_) > 0:
        col1.write(f'Periodo: {desc_dia_i}')
        col1.table(st.session_state.general_)
        col2.write('')
        col2.table(st.session_state.modal_)
        col3.write('')
        col3.table(st.session_state.distancias_)