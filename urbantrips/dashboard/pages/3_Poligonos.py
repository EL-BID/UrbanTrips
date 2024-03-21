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

from dash_utils import levanto_tabla_sql, get_logo, weighted_mean



@st.cache_data
def creo_lineas_deseo_linestrings(etapas_agrupadas_zon, viajes_agrupados_zon):
# etapas_agrupadas_zon = etapas_agrupadas.copy()
# viajes_agrupados_zon = viajes_agrupados.copy()
# if True:

    gpd_etapas_agrupadas_all = pd.DataFrame([])
    gpd_viajes_agrupados_all = pd.DataFrame([])

    grouped = etapas_agrupadas_zon.groupby(['id_polygon', 'zona'], as_index=False).factor_expansion_tarjeta.count()

    # Loop over unique combinations of col1 and col2
    for index, row in grouped.iterrows():
        id_polygon = row.id_polygon
        zona = row.zona
            
        # preparo etapas para armar linestrings
        gpd_etapas_agrupadas = etapas_agrupadas_zon[(etapas_agrupadas_zon.id_polygon==id_polygon)&(etapas_agrupadas_zon.zona==zona)].copy()
        gpd_etapas_agrupadas = gpd_etapas_agrupadas.reset_index(drop=True) .reset_index().rename(columns={'index':'linea_deseo'})

        # gpd_etapas_agrupadas['transferencia'] = 0
        # gpd_etapas_agrupadas.loc[(gpd_etapas_agrupadas.h3_transfer1_norm!='')|(gpd_etapas_agrupadas.h3_transfer2_norm!=''), 'transferencia'] = 1
        
        tmp = pd.DataFrame([])
        n=1
        for i in ['h3_inicio_norm', 'h3_transfer1_norm', 'h3_transfer2_norm', 'h3_fin_norm']:
            df = gpd_etapas_agrupadas[['linea_deseo',
                                       'id_polygon',
                                       'zona',
                                       i, 
                                       'poly_od', 
                                       'poly_transfer',
                                       'transferencia', 
                                       'factor_expansion_tarjeta', 
                                       f'lat{n}', 
                                       f'lon{n}']].rename(columns={i:'h3', f'lat{n}':'lat', f'lon{n}':'lon', 'factor_expansion_tarjeta':'viajes'})
            df['id_etapa'] = n
            
            tmp = pd.concat([tmp, df], ignore_index=True)
            
            n+=1
        
        gpd_etapas_agrupadas = tmp.loc[tmp.h3!='', ['linea_deseo', 
                                                    'id_polygon',
                                                    'zona',
                                                    'id_etapa', 
                                                    'h3', 
                                                    'poly_od', 
                                                    'poly_transfer',
                                                    'transferencia', 
                                                    'viajes', 
                                                    'lat', 
                                                    'lon']].sort_values(['linea_deseo', 'id_etapa']).reset_index(drop=True)    

        gpd_etapas_agrupadas_all = pd.concat([gpd_etapas_agrupadas_all, gpd_etapas_agrupadas], ignore_index=True)
    
        # preparo viajes para armar linestring            
        gpd_viajes_agrupados = viajes_agrupados_zon[(viajes_agrupados_zon.id_polygon==id_polygon)&(viajes_agrupados_zon.zona==zona)].copy()
        gpd_viajes_agrupados = gpd_viajes_agrupados.reset_index(drop=True).reset_index().rename(columns={'index':'linea_deseo'})
        
        tmp = pd.DataFrame([])
        n=1
        for i in ['h3_inicio_norm', 'h3_fin_norm']:
            if n == 2:
                n = 4
                
            df = gpd_viajes_agrupados[['linea_deseo', 
                                       'id_polygon',
                                       'zona',
                                           i,           
                                           'poly_od', 
                                           'poly_transfer',
                                           'transferencia', 
                                           'factor_expansion_tarjeta', 
                                           f'lat{n}', 
                                           f'lon{n}']].rename(columns={i:'h3', f'lat{n}':'lat', f'lon{n}':'lon', 'factor_expansion_tarjeta':'viajes'})
            df['id_etapa'] = n
            
            tmp = pd.concat([tmp, df], ignore_index=True)
            
            n+=1
        
        gpd_viajes_agrupados = tmp.loc[tmp.h3!='', ['linea_deseo', 
                                                    'id_polygon',
                                                    'zona',
                                                    'id_etapa', 
                                                    'h3', 
                                                    'poly_od', 
                                                    'poly_transfer',
                                                    'transferencia', 
                                                    'viajes', 
                                                    'lat', 
                                                    'lon']].sort_values(['linea_deseo', 'id_etapa']).reset_index(drop=True)    

        gpd_viajes_agrupados_all = pd.concat([gpd_viajes_agrupados_all, gpd_viajes_agrupados], ignore_index=True)


    lineas_deseo_etapas = crear_linestring(gpd_etapas_agrupadas_all, 
                                           order_by=['id_polygon', 'zona', 'linea_deseo', 'id_etapa'], 
                                           group_by=['id_polygon', 'zona', 'linea_deseo', 'poly_od', 'poly_transfer', 'transferencia'])
    
    lineas_deseo_viajes = crear_linestring(gpd_viajes_agrupados_all, 
                                           order_by=['id_polygon', 'zona', 'linea_deseo', 'id_etapa'], 
                                           group_by=['id_polygon', 'zona', 'linea_deseo', 'poly_od', 'poly_transfer', 'transferencia'])


    
    return lineas_deseo_etapas, lineas_deseo_viajes

def crear_linestring(df, order_by=['id_polygon', 'linea_deseo', 'id_etapa'], group_by=['id_polygon', 'linea_deseo', 'transferencia', 'poly_od', 'poly_transfer']):

    # Ensure the DataFrame is sorted by `linea_deseo` and `id_etapa`
    df = df.sort_values(by=order_by)
    
    # Create Point geometries using latitude and longitude
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    
    # Convert DataFrame to GeoDataFrame
    gdf_points = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Create LineStrings
    lines = gdf_points.groupby(group_by).apply(lambda x: LineString(x['geometry'].tolist()))
    
    # Convert the Series to a GeoDataFrame
    gdf_lines = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
    
    gdf_lines = gdf_lines.reset_index()


    df = df.groupby(group_by, as_index=False).viajes.max()
    df = df.merge(gdf_lines)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    return  df

@st.cache_data
def agrupar_viajes(etapas_agrupadas_zon, viajes_agrupados_zon):
    
    viajes_agrupados_zon['transferencia'] = 99
    etapas_agrupadas_zon['transferencia'] = 99

    viajes_agrupados_zon_wmean = viajes_agrupados_zon.groupby(['id_polygon',
                                                               'zona', 
                                                               'h3_inicio_norm',                                                                        
                                                               'h3_fin_norm', 
                                                               'poly_od',
                                                               'poly_transfer', 
                                                               'transferencia'],
                                               as_index=False).apply(lambda x: pd.Series({
                                                        'lat1': weighted_mean(x['lat1'], x['factor_expansion_tarjeta']),
                                                        'lon1': weighted_mean(x['lon1'], x['factor_expansion_tarjeta']),                                                         
                                                        'lat4': weighted_mean(x['lat4'], x['factor_expansion_tarjeta']),                        
                                                        'lon4': weighted_mean(x['lon4'], x['factor_expansion_tarjeta']),
                                                                    }))    
    
    viajes_agrupados_zon = viajes_agrupados_zon.groupby(['id_polygon',
                                                         'zona', 
                                                         'h3_inicio_norm', 
                                                         'h3_fin_norm', 
                                                         'poly_od',
                                                         'poly_transfer',
                                                         'transferencia'], 
                                                        as_index=False).factor_expansion_tarjeta.sum().sort_values('factor_expansion_tarjeta',                 
                                                                                                            ascending=False).round().reset_index(drop=True)

    etapas_agrupadas_zon_wmean = etapas_agrupadas_zon.groupby(['id_polygon',
                                                               'zona', 
                                                               'h3_inicio_norm',
                                                               'h3_transfer1_norm',                                                                        
                                                               'h3_transfer2_norm',                                                                        
                                                               'h3_fin_norm', 
                                                               'poly_od', 
                                                               'poly_transfer',
                                                               'transferencia'],
                                               as_index=False).apply(lambda x: pd.Series({
                                                        'lat1': weighted_mean(x['lat1'], x['factor_expansion_tarjeta']),
                                                        'lon1': weighted_mean(x['lon1'], x['factor_expansion_tarjeta']),
                                                        'lat2': weighted_mean(x['lat2'], x['factor_expansion_tarjeta']),
                                                        'lon2': weighted_mean(x['lon2'], x['factor_expansion_tarjeta']),
                                                        'lat3': weighted_mean(x['lat3'], x['factor_expansion_tarjeta']),
                                                        'lon3': weighted_mean(x['lon3'], x['factor_expansion_tarjeta']),
                                                        'lat4': weighted_mean(x['lat4'], x['factor_expansion_tarjeta']),                        
                                                        'lon4': weighted_mean(x['lon4'], x['factor_expansion_tarjeta']),
                                                                    }))    
    
    etapas_agrupadas_zon = etapas_agrupadas_zon.groupby(['id_polygon',
                                                         'zona',
                                                         'h3_inicio_norm',  
                                                         'h3_transfer1_norm',                                                                  
                                                         'h3_transfer2_norm',                                                                  
                                                         'h3_fin_norm', 
                                                         'poly_od', 
                                                         'poly_transfer',
                                                         'transferencia'], as_index=False).factor_expansion_tarjeta.sum().sort_values('factor_expansion_tarjeta',                 
                                                                                                            ascending=False).round().reset_index(drop=True)

    viajes_agrupados_zon = viajes_agrupados_zon.merge(viajes_agrupados_zon_wmean)
    etapas_agrupadas_zon = etapas_agrupadas_zon.merge(etapas_agrupadas_zon_wmean)
    
    return etapas_agrupadas_zon, viajes_agrupados_zon
    
def crear_mapa_poligonos(df_viajes,
                         df_etapas,
                         zonif,
                         var_fex,
                         cmap_viajes = 'Blues',
                         cmap_etapas = 'Greens',
                         map_title = '',                      
                         savefile='',
                         k_jenks=5):

    m = ''
    if (len(df_viajes) > 0)|(len(df_etapas)>0):

        fig = Figure(width=800, height=800)

        m = folium.Map(location=[poly.geometry.representative_point().y.mean(), 
                                 poly.geometry.representative_point().x.mean()], zoom_start=12, tiles='cartodbpositron')

    
        title_html = """
        <h3 align="center" style="font-size:20px"><b>Your map title</b></h3>
        """
        title_html = title_html.replace('Your map title', map_title)
        m.get_root().html.add_child(folium.Element(title_html))
    
        line_w = 0.5
    
        colors_viajes = mcp.gen_color(cmap=cmap_viajes, n=k_jenks)
        colors_etapas = mcp.gen_color(cmap=cmap_etapas, n=k_jenks)
    
        # Etapas
        if len(df_etapas) > 0:
            try:
                bins = [df_etapas[var_fex].min()-1] + \
                    mapclassify.FisherJenks(df_etapas[var_fex], k=k_jenks).bins.tolist()
            except ValueError:
                bins = [df_etapas[var_fex].min()-1] + \
                    mapclassify.FisherJenks(df_etapas[var_fex], k=k_jenks-2).bins.tolist()

            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} etapas' for n in range_bins]
            df_etapas['cuts'] = pd.cut(df_etapas[var_fex], bins=bins, labels=bins_labels)
        
            n = 0
            for i in bins_labels:
        
                df_etapas[df_etapas.cuts == i].explore(
                    m=m,
                    color=colors_etapas[n],
                    style_kwds={'fillOpacity': 0.1, 'weight': line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 3
    
    
        #Viajes
        if len(df_viajes) > 0:
            try:
                bins = [df_viajes[var_fex].min()-1] + \
                    mapclassify.FisherJenks(df_viajes[var_fex], k=k_jenks).bins.tolist()
            except ValueError:
                bins = [df_viajes[var_fex].min()-1] + \
                    mapclassify.FisherJenks(df_viajes[var_fex], k=k_jenks-2).bins.tolist()
                
            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins]
            df_viajes['cuts'] = pd.cut(df_viajes[var_fex], bins=bins, labels=bins_labels)
        
            n = 0
            for i in bins_labels:
        
                df_viajes[df_viajes.cuts == i].explore(
                    m=m,
                    color=colors_viajes[n],
                    style_kwds={'fillOpacity': 0.1, 'weight': line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 3
    
    
    
        # Agrego polígono
        if len(poly) > 0:
            geojson = poly.to_json()    
            # Add the GeoJSON to the map as a GeoJson Layer
            folium.GeoJson(
                geojson,
                name=poly.id.values[0],
                style_function=lambda feature: {
                    'fillColor': 'navy',
                    'color': 'navy',
                    'weight': 2,
                    'fillOpacity': .5,
                    
                }
            ).add_to(m)
    
    
        folium.LayerControl(name='xx').add_to(m)
    
        # fig.add_child(m)


    return m


st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)


# with st.expander('Polígonos'):

col1, col2 = st.columns([1, 4])


poligonos = levanto_tabla_sql('poligonos')

if len(poligonos) > 0:

    viajes_agrupados = levanto_tabla_sql('viajes_agrupados')
    etapas_agrupadas = levanto_tabla_sql('etapas_agrupadas')
    
    etapas_agrupadas_all, viajes_agrupados_all  = agrupar_viajes(etapas_agrupadas, viajes_agrupados)
    lineas_deseo_etapas_all, lineas_deseo_viajes_all = creo_lineas_deseo_linestrings(etapas_agrupadas_all, viajes_agrupados_all)
    
    lineas_deseo_etapas, lineas_deseo_viajes = creo_lineas_deseo_linestrings(etapas_agrupadas, viajes_agrupados)
    
    
    desc_poly = col1.selectbox(
        'Polígono', options=etapas_agrupadas.id_polygon.unique())
    desc_zona = col1.selectbox(
        'Zonificación', options=etapas_agrupadas.zona.unique())
    
    desc_viajes = col1.checkbox(
        'Viajes', value=True)
    
    desc_etapas = col1.checkbox(
        'etapas', value=True)
    
    desc_viajes_cTransf = col1.checkbox(
        'Viajes con transferencia', value=True)
    
    desc_viajes_sTransf = col1.checkbox(
        'Viajes sin transferencia', value=True)
    
    
    etapas = ''
    viajes = ''
    if (desc_viajes_cTransf)&(desc_viajes_sTransf):
        if desc_etapas:
            etapas = lineas_deseo_etapas_all[(lineas_deseo_etapas_all.id_polygon == desc_poly)&(lineas_deseo_etapas_all.zona == desc_zona)].drop(['transferencia'], axis=1)
        if desc_viajes:
            viajes = lineas_deseo_viajes_all[(lineas_deseo_viajes_all.id_polygon == desc_poly)&(lineas_deseo_viajes_all.zona == desc_zona)].drop(['transferencia'], axis=1)
    
    if (not desc_viajes_cTransf)&(desc_viajes_sTransf): # Solo viajes sin transferencia
        if desc_etapas:
            etapas = lineas_deseo_etapas[(lineas_deseo_etapas.id_polygon == desc_poly)&(lineas_deseo_etapas.zona == desc_zona)&(lineas_deseo_etapas.transferencia==0)]
        if desc_viajes:
            viajes = lineas_deseo_viajes[(lineas_deseo_viajes.id_polygon == desc_poly)&(lineas_deseo_viajes.zona == desc_zona)&(lineas_deseo_viajes.transferencia==0)]
    
    if (desc_viajes_cTransf)&(not desc_viajes_sTransf): # Solo viajes sin transferencia
        if desc_etapas:
            etapas = lineas_deseo_etapas[(lineas_deseo_etapas.id_polygon == desc_poly)&(lineas_deseo_etapas.zona == desc_zona)&(lineas_deseo_etapas.transferencia==1)]
        if desc_viajes:
            viajes = lineas_deseo_viajes[(lineas_deseo_viajes.id_polygon == desc_poly)&(lineas_deseo_viajes.zona == desc_zona)&(lineas_deseo_viajes.transferencia==1)]
    
    
    if col2.checkbox('Ver datos: Viajes'):
            col2.write(viajes)
    
    if col2.checkbox('Ver datos: Etapas'):
            col2.write(etapas)
    
    
    
    
    poly = poligonos[poligonos.id==desc_poly]
    
    if (len(etapas) > 0)|(len(viajes) > 0):
    
        map = crear_mapa_poligonos(df_viajes = viajes,
                         df_etapas = etapas,
                         zonif = poly,
                         var_fex = 'viajes',
                         cmap_viajes = 'Blues',
                         cmap_etapas = 'Greens',
                         map_title = desc_poly,                      
                         savefile='',
                         k_jenks=5)
    
        with col2:
            st_map = st_folium(map, width=1200, height=1000)
    else:
    
        col2.markdown("""
        <style>
        .big-font {
            font-size:40px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
        col2.markdown(
            '<p class="big-font">            ¡¡ No hay datos para mostrar !!</p>', unsafe_allow_html=True)

else:
    
    col2.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col2.markdown(
        '<p class="big-font">            ¡¡ No hay datos para mostrar !!</p>', unsafe_allow_html=True)

