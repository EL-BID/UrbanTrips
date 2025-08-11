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

from dash_utils import (
    levanto_tabla_sql,
    get_logo,
    traigo_indicadores,
    configurar_selector_dia,
    formatear_columnas_numericas
)





st.set_page_config(layout="wide")

st.sidebar.success("Seleccione página")

logo = get_logo()
st.image(logo)


st.markdown(
    '<div style="text-align: justify;">urbantrips es una biblioteca de código abierto que toma información de un sistema de pago con tarjeta inteligente de transporte público y, a través de un procesamiento de la información que infiere destinos de los viajes y construye las cadenas de viaje para cada usuario, produce matrices de origen-destino y otros indicadores (KPI) para rutas de autobús. El principal objetivo de la librería es producir insumos útiles para la gestión del transporte público a partir de requerimientos mínimos de información y pre-procesamiento. Con sólo una tabla geolocalizada de transacciones económicas proveniente de un sistema de pago electrónico, se podrán generar resultados, que serán más precisos cuanto más información adicional se incorpore al proceso a través de los archivos opcionales. El proceso elabora las matrices, los indicadores y construye una serie de gráficos y mapas de transporte.</div>',
    unsafe_allow_html=True,
)
st.text("")

alias_seleccionado = configurar_selector_dia()


col1, col2, col3 = st.columns([1, 3, 3])

indicadores = levanto_tabla_sql("indicadores", "data")
indicadores = formatear_columnas_numericas(indicadores, ['porcentaje'], False)

if len(indicadores) > 0:
    desc_dia_i = col1.selectbox(
        "Dia", options=indicadores.dia.unique(), key="desc_dia_i"
    )

    indicadores = indicadores[(indicadores.dia == desc_dia_i)]

    trx = indicadores.loc[
        indicadores.tabla == "transacciones", ["detalle", "indicador", "porcentaje"]
    ].copy()
    
    col2.write("Transacciones")
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col2.write(trx)

    
    trx = indicadores.loc[
        indicadores.tabla == "etapas", ["detalle", "indicador", "porcentaje"]
    ].copy()
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col2.write("Etapas")
    col2.write(trx)
    trx = indicadores.loc[
        indicadores.tabla == "viajes", ["detalle", "indicador", "porcentaje"]
    ].copy()
    col2.write("Viajes")
    
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col2.write(trx)
    trx = indicadores.loc[
        indicadores.tabla.isin(["tarjetas", "usuarios"]),
        ["detalle", "indicador", "porcentaje"],
    ].copy()
    col2.write("Tarjetas")
    
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col2.write(trx)
    trx = indicadores.loc[
        indicadores.tabla == "etapas_expandidas", ["detalle", "indicador", "porcentaje"]
    ].copy()
    
    col2.write("Etapas expandidas")
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col2.write(trx)
    trx = indicadores.loc[
        indicadores.tabla == "viajes expandidos", ["detalle", "indicador", "porcentaje"]
    ].copy()
    
    col2.write("Viajes expandidos")
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col2.write(trx)
    trx = indicadores.loc[
        indicadores.tabla == "usuarios expandidos",
        ["detalle", "indicador", "porcentaje"],
    ].copy()
    
    col3.write("Usuarios")
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col3.write(trx)
    trx = indicadores.loc[
        indicadores.tabla == "modos viajes", ["detalle", "indicador", "porcentaje"]
    ].copy()
    
    col3.write("Partición modal Viajes")
    trx = formatear_columnas_numericas(trx, ['indicador'], True)
    col3.write(trx)

    
    trx = indicadores.loc[
        (indicadores.tabla == "avg") &
        (indicadores.detalle.str.contains('promedio')), ["detalle", "indicador", "porcentaje"]
    ].copy()
    
    col3.write("Promedios")
    trx = formatear_columnas_numericas(trx, ['indicador'], False)
    col3.write(trx)

    trx = indicadores.loc[
        (indicadores.tabla == "avg") &
        (indicadores.detalle.str.contains('mediana')), ["detalle", "indicador", "porcentaje"]
    ].copy()
    
    col3.write("Medianas")
    trx = formatear_columnas_numericas(trx, ['indicador'], False)
    col3.write(trx)

