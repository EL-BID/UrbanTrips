import pandas as pd
import streamlit as st
import h3
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import json
from folium import plugins
from shapely import wkt
from dash_utils import (
    iniciar_conexion_db, get_logo
)
from streamlit_folium import folium_static


def levanto_tabla_sql(tabla_sql, tabla_tipo="dash", query=''):

    conn = iniciar_conexion_db(tipo=tabla_tipo)

    try:
        if len(query) == 0:
            query = f"""
            SELECT *
            FROM {tabla_sql}
            """

        tabla = pd.read_sql_query( query, conn )
    except:
        print(f"{tabla_sql} no existe")
        tabla = pd.DataFrame([])

    conn.close()

    if len(tabla) > 0:
        if "wkt" in tabla.columns:
            tabla["geometry"] = tabla.wkt.apply(wkt.loads)
            tabla = gpd.GeoDataFrame(tabla, crs=4326)
            tabla = tabla.drop(["wkt"], axis=1)

    return tabla
    
@st.cache_data
def traigo_mes_dia():
    mes_dia = levanto_tabla_sql('etapas_agregadas', 'dash', 'SELECT DISTINCT mes, tipo_dia FROM etapas_agregadas;')
    mes = mes_dia.mes.values.tolist()
    tipo_dia = mes_dia.tipo_dia.values.tolist()
    return mes, tipo_dia

# Convert geometry to H3 indices
def get_h3_indices_in_geometry(geometry, resolution):
    geojson = mapping(geometry)
    h3_indices = list(h3.polyfill(geojson, resolution, geo_json_conformant=True))
    return h3_indices

# Convert H3 indices to GeoDataFrame
def h3_indices_to_gdf(h3_indices):
    hex_geometries = [Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in h3_indices]
    return gpd.GeoDataFrame({'h3_index': h3_indices}, geometry=hex_geometries, crs='EPSG:4326')

# Initialize session state for zones
if 'zona_1' not in st.session_state:
    st.session_state['zona_1'] = []
if 'zona_2' not in st.session_state:
    st.session_state['zona_2'] = []

def main():

    
    st.set_page_config(layout="wide")
    logo = get_logo()
    st.image(logo)

    mes_lst, tipo_dia_lst = traigo_mes_dia()
    
    with st.expander('Selecciono zonas', expanded=True):
        col1, col2 = st.columns([1, 4])
        
        # # Sidebar controls
        # resolution = col1.slider("Selecciona la Resolución H3", min_value=0, max_value=15, value=8, step=1)
        resolution = 8
        
        # Initialize Folium map
        m = folium.Map(location=[-34.593, -58.451], zoom_start=10)
        draw = plugins.Draw(
            export=False,
            draw_options={'polygon': True, 'rectangle': True},
            edit_options={'edit': True, 'remove': True}
        )
        draw.add_to(m)
        
        # Display map with drawing tools
        with col2:
            output = st_folium(m, width=700, height=700, key='map')

        # Handle user drawing
        if output.get('last_active_drawing'):
            geometry_data = output['last_active_drawing']['geometry']
            geometry = shape(geometry_data)
            h3_indices = get_h3_indices_in_geometry(geometry, resolution)
            
            # Save hexagons to session state based on button clicks
            if col1.button("Guardar en Zona 1"):
                st.session_state['zona_1'].extend(h3_indices)
            if col1.button("Guardar en Zona 2"):
                st.session_state['zona_2'].extend(h3_indices)

        zona1 = st.session_state['zona_1']
        zona2 = st.session_state['zona_2']



        # Convertir la lista de índices H3 a una cadena en formato de lista de Python
        zona1_str = json.dumps(zona1)        
        col2.code(zona1_str, language='python')
        
        zona2_str = json.dumps(zona2)        
        col2.code(zona2_str, language='python')

    with st.expander('Resultados', expanded=True):
        col1, col2, col3, col4 = st.columns([1, 2, 2, 3])
        zona1 = st.session_state['zona_1']
        zona2 = st.session_state['zona_2']

        # mes_lst = ['Todos'] + etapas_all.mes.unique().tolist()                
        desc_mes = col1.selectbox(
                'Mes', options=mes_lst)

        desc_tipo_dia = col1.selectbox(
                        'Tipo dia', options=tipo_dia_lst)

        
        if len(zona1) > 0 and len(zona2) > 0:
            h3_values = ", ".join(f"'{item}'" for item in zona1 + zona2)
            ## Etapas
            query = f"SELECT * FROM etapas_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values}) OR h3_d IN ({h3_values}));"
            etapas = levanto_tabla_sql('etapas_agregadas', tabla_tipo='dash', query=query)
            
            etapas['Zona_1'] = ''
            etapas['Zona_2'] = ''
            etapas.loc[etapas.h3_o.isin(zona1), 'Zona_1'] = 'Zona 1'
            etapas.loc[etapas.h3_o.isin(zona2), 'Zona_1'] = 'Zona 2'
            etapas.loc[etapas.h3_d.isin(zona1), 'Zona_2'] = 'Zona 1'
            etapas.loc[etapas.h3_d.isin(zona2), 'Zona_2'] = 'Zona 2'
            etapas = etapas[(etapas.Zona_1 != '') & (etapas.Zona_2 != '') & (etapas.Zona_1 != etapas.Zona_2) & (etapas.Zona_1 != etapas.Zona_2)]
            

            zonasod_e = etapas.groupby(['Zona_1', 'Zona_2'], as_index=False).factor_expansion_linea.sum().round().rename(columns={'factor_expansion_linea':'Viajes'})
            zonasod_e['Viajes'] = zonasod_e['Viajes'].astype(int)


            modos_e = etapas.groupby(['modo', 'nombre_linea'], as_index=False).factor_expansion_linea.sum().round().rename(columns={'factor_expansion_linea':'Viajes', 
                                                                                                                                  'nombre_linea': 'Línea', 
                                                                                                                                  'modo': 'Modo'})
            modos_e['Viajes'] = modos_e['Viajes'].astype(int)
            col2.write('Etapas')
            col2.markdown(zonasod_e.to_html(index=False), unsafe_allow_html=True)
            col2.markdown(modos_e.to_html(index=False), unsafe_allow_html=True)
            

            ## Viajes
            query = f"SELECT * FROM viajes_agregados WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values}) OR h3_d IN ({h3_values}));"
            viajes = levanto_tabla_sql('viajes_agregados', tabla_tipo='dash', query=query)
            
            viajes['Zona_1'] = ''
            viajes['Zona_2'] = ''
            viajes.loc[viajes.h3_o.isin(zona1), 'Zona_1'] = 'Zona 1'
            viajes.loc[viajes.h3_o.isin(zona2), 'Zona_1'] = 'Zona 2'
            viajes.loc[viajes.h3_d.isin(zona1), 'Zona_2'] = 'Zona 1'
            viajes.loc[viajes.h3_d.isin(zona2), 'Zona_2'] = 'Zona 2'
            viajes = viajes[(viajes.Zona_1 != '') & (viajes.Zona_2 != '') & (viajes.Zona_1 != viajes.Zona_2) & (viajes.Zona_1 != viajes.Zona_2)]
            

            zonasod_e = viajes.groupby(['Zona_1', 'Zona_2'], as_index=False).factor_expansion_linea.sum().round().rename(columns={'factor_expansion_linea':'Viajes'})
            zonasod_e['Viajes'] = zonasod_e['Viajes'].astype(int)

            modos_v = viajes.groupby(['modo'], as_index=False).factor_expansion_linea.sum().round().rename(columns={'factor_expansion_linea':'Viajes', 
                                                                                                                   'modo': 'Modo'})

            modos_v['Viajes'] = modos_v['Viajes'].astype(int)
            col3.write('Viajes')
            col3.markdown(zonasod_e.to_html(index=False), unsafe_allow_html=True)
            col3.markdown(modos_v.to_html(index=False), unsafe_allow_html=True)

            ## Mapa

            # Create unified geometry for each zone
            def zona_to_geometry(h3_list):
                polygons = [Polygon(h3.h3_to_geo_boundary(h3_index, geo_json=True)) for h3_index in h3_list]
                return unary_union(polygons)

            geometry_zona1 = zona_to_geometry(st.session_state['zona_1'])
            geometry_zona2 = zona_to_geometry(st.session_state['zona_2'])
            gdf = gpd.GeoDataFrame({
                'zona': ['Zona 1', 'Zona 2'],
                'geometry': [geometry_zona1, geometry_zona2]
            }, crs="EPSG:4326")

            # Plot the zones on a new Folium map
            m2 = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=10)
            folium.GeoJson(gdf, name="GeoData").add_to(m2)

            with col4:
                st_folium(m2, width=700, height=700)

if __name__ == '__main__':
    main()
