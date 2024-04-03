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
from shapely.geometry import LineString, Point


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
    assert os.path.isfile(
        db_path), f'No existe la base de datos para el dashboard en {db_path}'
    conn = sqlite3.connect(db_path, timeout=10)
    return conn

# Calculate weighted mean, handling division by zero or empty inputs


def weighted_mean(series, weights):
    try:
        result = (series * weights).sum() / weights.sum()
    except ZeroDivisionError:
        result = np.nan
    return result


def normalize_vars(tabla):
    if 'dia' in tabla.columns:
        tabla.loc[tabla.dia == 'weekday', 'dia'] = 'Día hábil'
        tabla.loc[tabla.dia == 'weekend', 'dia'] = 'Fin de semana'
    if 'day_type' in tabla.columns:
        tabla.loc[tabla.day_type == 'weekday', 'day_type'] = 'Día hábil'
        tabla.loc[tabla.day_type == 'weekend', 'day_type'] = 'Fin de semana'

    if 'nombre_linea' in tabla.columns:
        tabla['nombre_linea'] = tabla['nombre_linea'].str.replace(' -', '')
    if 'Modo' in tabla.columns:
        tabla['Modo'] = tabla['Modo'].str.capitalize()
    if 'modo' in tabla.columns:
        tabla['modo'] = tabla['modo'].str.capitalize()
    return tabla


@st.cache_data
def levanto_tabla_sql(tabla_sql, custom_query=False,
                      tabla_tipo='dash'):

    conn = iniciar_conexion_db(tipo=tabla_tipo)

    try:
        if not custom_query:
            tabla = pd.read_sql_query(
                f"""
                SELECT *
                FROM {tabla_sql}
                """,
                conn,
            )
        else:
            tabla = pd.read_sql_query(
                custom_query,
                conn,
            )
    except:
        print(f'{tabla_sql} no existe')
        tabla = pd.DataFrame([])

    conn.close()

    if len(tabla) > 0:
        if 'wkt' in tabla.columns:
            tabla["geometry"] = tabla.wkt.apply(wkt.loads)
            tabla = gpd.GeoDataFrame(tabla,
                                     crs=4326)
            tabla = tabla.drop(['wkt'], axis=1)

    tabla = normalize_vars(tabla)

    return tabla


@st.cache_data
def get_logo():
    file_logo = os.path.join(
        "docs", "urbantrips_logo.jpg")
    if not os.path.isfile(file_logo):
        # URL of the image file on Github
        url = 'https://raw.githubusercontent.com/EL-BID/UrbanTrips/main/docs/urbantrips_logo.jpg'

        # Send a request to get the content of the image file
        response = requests.get(url)

        # Save the content to a local file
        with open(file_logo, 'wb') as f:
            f.write(response.content)
    image = Image.open(file_logo)
    return image


@st.cache_data
def create_linestring_od(df,
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
