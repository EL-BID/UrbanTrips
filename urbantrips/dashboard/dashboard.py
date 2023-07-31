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

import yaml
import sqlite3


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
    conn = sqlite3.connect(db_path, timeout=10)
    return conn

@st.cache_data
def levanto_tabla_sql(tabla_sql):

    conn_dash = iniciar_conexion_db(tipo='dash')

    tabla = pd.read_sql_query(
        f"""
        SELECT *
        FROM {tabla_sql}
        """,
        conn_dash,
    )

    conn_dash.close()
    
    return tabla



@st.cache_data
def get_logo():
    if not os.path.isdir(os.path.join("docs")):
        os.mkdir(os.path.join("docs"))
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
    

    
st.set_page_config(layout="wide")
    
st.sidebar.success('Seleccione página')

logo = get_logo()
st.image(logo)


st.markdown('<div style="text-align: justify;">urbantrips es una biblioteca de código abierto que toma información de un sistema de pago con tarjeta inteligente de transporte público y, a través de un procesamiento de la información que infiere destinos de los viajes y construye las cadenas de viaje para cada usuario, produce matrices de origen-destino y otros indicadores (KPI) para rutas de autobús. El principal objetivo de la librería es producir insumos útiles para la gestión del transporte público a partir de requerimientos mínimos de información y pre-procesamiento. Con sólo una tabla geolocalizada de transacciones económicas proveniente de un sistema de pago electrónico, se podrán generar resultados, que serán más precisos cuanto más información adicional se incorpore al proceso a través de los archivos opcionales. El proceso elabora las matrices, los indicadores y construye una serie de gráficos y mapas de transporte.</div>', unsafe_allow_html=True)
st.text('')

col1, col2, col3 = st.columns([1, 3, 3])


indicadores = levanto_tabla_sql('indicadores')


desc_dia_i = col1.selectbox('Periodo', options=indicadores.desc_dia.unique(), key='desc_dia_i')
tipo_dia_i = col1.selectbox('Tipo de dia', options=indicadores.tipo_dia.unique(), key='tipo_dia_i')

    
indicadores = indicadores[(indicadores.desc_dia==desc_dia_i)&(indicadores.tipo_dia==tipo_dia_i)]

df = indicadores.loc[indicadores.orden==1, ['Indicador', 'Valor']].copy()
titulo = indicadores.loc[indicadores.orden==1].Titulo.unique()[0]

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


df = indicadores.loc[indicadores.orden==2, ['Indicador', 'Valor']].copy()
titulo = indicadores.loc[indicadores.orden==2].Titulo.unique()[0]

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

df = indicadores.loc[indicadores.orden==3, ['Indicador', 'Valor']].copy()
titulo = indicadores.loc[indicadores.orden==3].Titulo.unique()[0]

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

