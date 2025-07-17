import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
from folium import GeoJson, GeoJsonTooltip
import mapclassify
import plotly.express as px
from folium import Figure
from shapely import wkt
from dash_utils import (
    levanto_tabla_sql, levanto_tabla_sql_local, get_logo,
    create_data_folium, traigo_indicadores,
    extract_hex_colors_from_cmap,
    iniciar_conexion_db, normalize_vars,
    bring_latlon, traigo_lista_zonas, get_h3_indices_in_geometry,
    traigo_tablas_con_filtros,
    configurar_selector_dia
)
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import mapping
from datetime import datetime


import folium
import pandas as pd
import numpy as np
from branca.colormap import linear
from folium import Map, LayerControl, CircleMarker, PolyLine, GeoJson, Popup, FeatureGroup
import mapclassify



# 🎨 Mapeo de colormaps válidos
COLORMAPS = {
    'Blues': linear.Blues_09,
    'Greens': linear.Greens_09,
    'YlOrRd': linear.YlOrRd_09,
    'PuRd': linear.PuRd_09,
    'YlGn': linear.YlGn_09,
    'viridis': linear.viridis,
    'viridis_r': linear.viridis.scale(1, 0),  # Invertir manualmente
    'inferno': linear.inferno,
    'inferno_r': linear.inferno.scale(1, 0),  # Invertir manualmente
    'magma': linear.magma,
    'magma_r': linear.magma.scale(1, 0),  # Invertir manualmente
    'plasma': linear.plasma,
    'plasma_r': linear.plasma.scale(1, 0)  # Invertir manualmente
}


def obtener_clases_fisherjenks(df: pd.DataFrame, var_fex: str, max_clases: int = 5, min_clases: int = 1):
    """
    Determina un número óptimo de clases usando Fisher-Jenks, ajustándose si hay pocos valores únicos.

    Parámetros:
        - df (pd.DataFrame): DataFrame con la columna a clasificar.
        - var_fex (str): Nombre de la columna de valores.
        - max_clases (int): Número máximo de clases.
        - min_clases (int): Número mínimo de clases.

    Retorna:
        - list: Lista de bins clasificados.
    """
    unique_values = df[var_fex].nunique()
    k = min(max_clases, max(min_clases, unique_values))

    while k >= min_clases:
        try:
            bins = [df[var_fex].min() - 1] + mapclassify.FisherJenks(df[var_fex], k=k).bins.tolist()
            return bins
        except ValueError:
            k -= 1

    # Si no se puede clasificar, devolver un bin único
    return [df[var_fex].min() - 1, df[var_fex].max()]



def simplificar_geometrias(df: pd.DataFrame, tolerance: float = 0.001):
    """Simplifica las geometrías para mejorar el rendimiento."""
    if 'geometry' in df.columns:
        df['geometry'] = df['geometry'].simplify(tolerance, preserve_topology=True)
    return df


def crear_mapa_lineas_deseo(df_viajes: pd.DataFrame,
                            df_etapas: pd.DataFrame,
                            zonif: pd.DataFrame,
                            origenes: pd.DataFrame,
                            destinos: pd.DataFrame,
                            transferencias: pd.DataFrame,
                            var_fex: str,
                            cmap_viajes: str = 'viridis_r',
                            cmap_etapas: str = 'magma_r',
                            cmap_puntos: str = 'YlOrRd',
                            map_title: str = '',
                            savefile: str = '',
                            k_jenks: int = 5,
                            latlon: list = None) -> folium.Map:
    """
    Crea un mapa interactivo con capas diferenciadas para viajes, etapas, orígenes, destinos y transferencias.
    Incluye optimizaciones de rendimiento y utiliza Fisher-Jenks para la clasificación.

    Retorna:
        - folium.Map: Mapa interactivo.
    """

    if len(df_viajes)>0:
        df_viajes = df_viajes[df_viajes['geometry'].notna()]
    if len(df_etapas)>0:
        df_etapas = df_etapas[df_etapas['geometry'].notna()]

    # if latlon is None:
    #     latlon = [-34.6037, -58.3816]  # Default a Buenos Aires
    latlon = [-34.6037, -58.3816]
    # 🗺️ Crear el mapa
    m = folium.Map(location=latlon, zoom_start=9, tiles='cartodbpositron')

    # 🔄 Preprocesamiento de Datos
    print(datetime.now(), 'simplificar geometrias', len(df_viajes), len(df_etapas))
    df_etapas, df_viajes, origenes, destinos, transferencias = [
        simplificar_geometrias(df) for df in [df_etapas, df_viajes, origenes, destinos, transferencias]
    ]

    print(datetime.now(), 'fin simplificar geometrias', len(df_viajes), len(df_etapas))
    
    # 🔗 Agregar capas de líneas
    def agregar_capa_lineas(df, nombre, var_fex, cmap, weight_base=.5):
        if len(df) == 0:
            return

        bins = obtener_clases_fisherjenks(df, var_fex)
        range_bins = range(0, len(bins)-1)
        bins_labels = [f'{int(bins[n])} a {int(bins[n+1])}' for n in range_bins]

        colors = extract_hex_colors_from_cmap(
                    cmap='viridis_r', n=k_jenks)

        weight_op = .8
        for i, label in enumerate(bins_labels):
            capa = FeatureGroup(name=f"{nombre} - {label}")
            subset = df[(df[var_fex] >= bins[i]) & (df[var_fex] < bins[i + 1])]
            for _, row in subset.iterrows():
                PolyLine(
                    locations=[(point[1], point[0]) for point in row.geometry.coords],
                    color=colors[i],
                    weight=weight_base,  # Aumentar el grosor de las líneas
                    opacity=weight_op,
                    popup=Popup(f"{nombre}: {row[var_fex]}")
                ).add_to(capa)
            capa.add_to(m)
            weight_base += 3
            weight_op += .1
            # style_kwds={'fillOpacity': 0.1, 'weight': line_w}

    # 🟢 Agregar capas de puntos
    def agregar_capa_puntos(df, nombre, var_fex, cmap):
        if len(df) == 0:
            return

        capa = FeatureGroup(name=nombre)
        colormap = COLORMAPS.get(cmap, linear.viridis).scale(df[var_fex].min(), df[var_fex].max())
        colormap.caption = f"Escala {nombre}"
        colormap.add_to(m)

        for _, row in df.iterrows():
            CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=10 + (row[var_fex] / df[var_fex].max()) * 8,  # Aumentar el tamaño de las burbujas
                color=colormap(row[var_fex]),
                fill=True,
                fill_opacity=0.8,
                popup=Popup(f"{nombre}: {row[var_fex]}")
            ).add_to(capa)
        capa.add_to(m)

    agregar_capa_lineas(df_etapas, "Etapas", var_fex, cmap_etapas, weight_base=.5)
    agregar_capa_lineas(df_viajes, "Viajes", var_fex, cmap_viajes, weight_base=.5)
    agregar_capa_puntos(origenes, "Orígenes", var_fex, cmap_puntos)
    agregar_capa_puntos(destinos, "Destinos", var_fex, cmap_puntos)
    agregar_capa_puntos(transferencias, "Transferencias", var_fex, cmap_puntos)

    if len(zonif) > 0:
        GeoJson(
            data=zonif.__geo_interface__,
            name="Zonificación",
            style_function=lambda x: {'fillColor': 'blue', 'color': 'navy', 'weight': 2, 'fillOpacity': 0},
            tooltip=GeoJsonTooltip(
                    fields=['id'],
                    aliases=['ID:'],
                    labels=True,
                    sticky=True
                )
        ).add_to(m)

    folium.LayerControl().add_to(m)

    if savefile:
        m.save(savefile)

    return m



# Función para detectar cambios
def hay_cambios_en_filtros(current, last):
    return current != last

st.set_page_config(layout="wide")

alias_seleccionado = configurar_selector_dia()

logo = get_logo()
st.image(logo)

with st.expander('Líneas de Deseo', expanded=True):

    col1, col2, col3 = st.columns([1, 7, 1])

    variables = [
            'last_filters', 
            'last_options', 
            'data_cargada', 
            'lista_etapas', 
            'matrices_all', 
            'etapas_all', 
            'etapas', 
            'viajes', 
            'matriz', 
            'origenes', 
            'destinos',
            'general', 
            'modal', 
            'distancia_seleccionada', 
            'mes', 
            'tipo_dia', 
            'zona', 
            'transferencia', 
            'modo_agregado', 
            'rango_hora_seleccionado', 
            'distancia', 
            'socio_indicadores_all', 
            'alias_seleccionado',
        ]
        
    variables_bool = ['etapas_seleccionada',
                    'viajes_seleccionado',
                    'origenes_seleccionado',
                    'destinos_seleccionado', 
                    'transferencias_seleccionado']
    # Inicializar todas las variables con None si no existen en session_state
    for var in variables:
        if var not in st.session_state:
            st.session_state[var] = ''
    
    for var in variables_bool:
        if var not in st.session_state:
            st.session_state[var] = False
    
    st.session_state.lista_etapas = levanto_tabla_sql('socio_indicadores', 'dash', 'SELECT DISTINCT mes FROM socio_indicadores;')
    st.session_state.lista_etapas = ['Todos'] + st.session_state.lista_etapas.mes.unique().tolist()        
   
    if len(st.session_state.lista_etapas) > 0:
        zonificaciones = levanto_tabla_sql('zonificaciones')

        equivalencia_zonas = levanto_tabla_sql('equivalencia_zonas', 'dash')
            
        socio_indicadores = levanto_tabla_sql('socio_indicadores')
        
        lista_tipo_dia = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT tipo_dia FROM agg_etapas;')
        lista_zonas = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT zona FROM agg_etapas;').sort_values('zona')
        lista_modos_agregados = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT modo_agregado FROM agg_etapas;')
        lista_rango_hora = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT rango_hora FROM agg_etapas;')
        lista_distancia_db = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT distancia FROM agg_etapas;')        
        lista_zonas = traigo_lista_zonas('etapas')
        
        # Inicializar valores de `st.session_state` solo si no existen
        if 'last_filters' not in st.session_state:
            st.session_state.last_filters = {
                'mes': 'Todos',
                'tipo_dia': None,
                'zona': None,
                'transferencia': 'Todos',
                'modo_agregado': 'Todos',
                'rango_hora_seleccionado': 'Todos',
                'distancia_seleccionada': 'Todas',
                'filtro_seleccion1':'Todos',
                'filtro_seleccion2':'Todos',
                'zona_filtro_seleccion1': None,
                'zona_filtro_seleccion2': None,
                'alias_seleccionado': alias_seleccionado
            }
            
        if 'data_cargada' not in st.session_state:
            st.session_state.data_cargada = False

        valores_zonas = lista_zonas.zona.unique().tolist()
        lista_distancia = ['Todas'] + lista_distancia_db[lista_distancia_db.distancia != '99'].distancia.unique().tolist()
        lista_transfer = ['Todos', 'Con transferencia', 'Sin transferencia']
        lista_modos = ['Todos'] + lista_modos_agregados[lista_modos_agregados.modo_agregado != '99'].modo_agregado.unique().tolist()        
        lista_rango_hora = ['Todos'] + lista_rango_hora[lista_rango_hora.rango_hora != '99'].rango_hora.unique().tolist()
        
        
        # Opciones de los filtros en Streamlit        
        mes_seleccionado = col1.selectbox('Mes', options=st.session_state.lista_etapas, index=1)        
        tipo_dia_seleccionado = col1.selectbox('Tipo día', options=lista_tipo_dia.tipo_dia.unique())        
        zona_seleccionada = col1.selectbox('Zonificación', options=valores_zonas)        
        transfer_seleccionado = col1.selectbox('Transferencias', options=lista_transfer)                
        modo_seleccionado = col1.selectbox('Modos', options=[text for text in lista_modos])                
        rango_hora_seleccionado = col1.selectbox('Rango hora', options=[text for text in lista_rango_hora])                
        distancia_seleccionada = col1.selectbox('Distancia', options=lista_distancia)

        vi_et_seleccion = col1.selectbox('Datos de', options=['Etapas', 'Viajes', 'Ninguno'], index=1)        
        st.session_state.viajes_seleccionado = vi_et_seleccion == 'Viajes'
        st.session_state.etapas_seleccionada = vi_et_seleccion == 'Etapas'

        col3.write('Agregar Filtros')
        index_zona = valores_zonas.index(zona_seleccionada)
        
        zona_filtro_seleccion1 = col3.selectbox('Zona Filtro 1', options=valores_zonas, key='zon1', index=index_zona)
        lista_zonas_all = ['Todos'] + lista_zonas[lista_zonas.zona == zona_filtro_seleccion1].Nombre.unique().tolist()
        filtro_seleccion1 = col3.selectbox('Filtro 1', options=lista_zonas_all, key='filtro1')
        zona_filtro_seleccion2 = col3.selectbox('Zona Filtro 2', options=valores_zonas, key='zon2', index=index_zona)
        lista_zonas_all = ['Todos'] + lista_zonas[lista_zonas.zona == zona_filtro_seleccion2].Nombre.unique().tolist()
        filtro_seleccion2 = col3.selectbox('Filtro 2', options=lista_zonas_all, key='filtro2')

        tipo_filtro = col3.selectbox('Tipo de Filtro', options=['OD y Transferencias', 'Solo OD'])
        
        col3.write('Mostrar:')
        st.session_state.origenes_seleccionado = col3.checkbox(
            ':blue[Origenes]', value=False)

        st.session_state.destinos_seleccionado = col3.checkbox(
            ':orange[Destinos]', value=False)

        st.session_state.transferencias_seleccionado = col3.checkbox(
            ':red[Transferencias]', value=False)
        
        # zonificacion_seleccion = col3.checkbox(
        #     'Mostrar zonificación', value=True)
        
        zona_mostrar = col3.selectbox('Mostrar zonificación', options=valores_zonas+['Ninguna'], index=index_zona)
        # if zonificacion_seleccion:
        if zona_mostrar != 'Ninguna':
            zonif = zonificaciones[zonificaciones.zona == zona_mostrar]
        else:
            zonif = ''

        mtabla = col2.checkbox('Mostrar tabla', value=False)
        
        # Construye el diccionario de filtros actual
        current_filters = {
            'mes': None if mes_seleccionado == 'Todos' else mes_seleccionado,
            'tipo_dia': tipo_dia_seleccionado,
            'zona': None if zona_seleccionada == 'Todos' else zona_seleccionada,
            'transferencia': None if transfer_seleccionado == 'Todos' else (1 if transfer_seleccionado == 'Con transferencia' else 0),
            'modo_agregado': None if modo_seleccionado == 'Todos' else modo_seleccionado,
            'rango_hora': None if rango_hora_seleccionado == 'Todos' else rango_hora_seleccionado,
            'distancia': None if distancia_seleccionada == 'Todas' else distancia_seleccionada,
            'filtro_seleccion1': None if filtro_seleccion1 == 'Todos' else filtro_seleccion1,            
            'filtro_seleccion2': None if filtro_seleccion2 == 'Todos' else filtro_seleccion2,   
            'zona_filtro_seleccion1': zona_filtro_seleccion1,
            'zona_filtro_seleccion2': zona_filtro_seleccion2,
            'tipo_filtro': tipo_filtro,
            'alias_seleccionado': alias_seleccionado, 
        }

        current_options = { 'etapas_seleccionada': st.session_state.etapas_seleccionada,
                            'viajes_seleccionado': st.session_state.viajes_seleccionado,
                            'origenes_seleccionado': st.session_state.origenes_seleccionado, 
                            'destinos_seleccionado': st.session_state.destinos_seleccionado,
                            'vi_et_seleccion': vi_et_seleccion,
                            'transferencias_seleccionado': st.session_state.transferencias_seleccionado,
                            'zona_mostrar': zona_mostrar, 
                            'mtabla': mtabla}
        
        # Solo cargar datos si hay cambios en los filtros
        if hay_cambios_en_filtros(current_filters, st.session_state.last_filters):
            
            query = ""
            conditions = " AND ".join(f"{key} = '{value}'" for key, value in current_filters.items() 
                                      if (value is not None)
                                          &(key != 'filtro_seleccion1')
                                          &(key != 'filtro_seleccion2')
                                          &(key != 'zona_filtro_seleccion1')
                                          &(key != 'zona_filtro_seleccion2')
                                          &(key != 'zona_filtro_seleccion1')
                                          &(key != 'tipo_filtro')
                                          &(key != 'alias_seleccionado')
                                     )
            if conditions:
                query += f" WHERE {conditions}"

            conditions_etapas1 = ''
            conditions_matrices1 = ''
            st.session_state['zona_1'] = []

            if filtro_seleccion1 != 'Todos':
                
                if tipo_filtro == 'OD y Transferencias':
                    conditions_etapas1 = f" AND (inicio_norm = '{filtro_seleccion1}' OR transfer1_norm = '{filtro_seleccion1}' OR transfer2_norm = '{filtro_seleccion1}' OR fin_norm = '{filtro_seleccion1}')"
                else:
                    conditions_etapas1 = f" AND (inicio_norm = '{filtro_seleccion1}' OR fin_norm = '{filtro_seleccion1}')"
                
                conditions_matrices1 = f" AND (inicio = '{filtro_seleccion1}' OR fin = '{filtro_seleccion1}')"

                # Obtener la geometría filtrada
                geometry = zonificaciones[
                    (zonificaciones.zona == zona_filtro_seleccion1) & 
                    (zonificaciones.id == filtro_seleccion1)
                ].geometry.values[0]
                
                # Inicializar una lista para almacenar los índices H3
                h3_indices_total = []
                
                # Verificar el tipo de geometría
                if isinstance(geometry, Polygon):
                    # Si es un Polygon, procesarlo directamente
                    h3_indices = get_h3_indices_in_geometry(geometry, 8)
                    h3_indices_total.extend(h3_indices)
                elif isinstance(geometry, MultiPolygon):
                    # Si es un MultiPolygon, iterar sobre cada Polygon
                    for poly in geometry.geoms:
                        h3_indices = get_h3_indices_in_geometry(poly, 8)
                        h3_indices_total.extend(h3_indices)
                else:
                    st.error("La geometría proporcionada no es un Polygon ni un MultiPolygon.")
                
                # Extender los índices H3 en el estado de la sesión
                st.session_state['zona_1'] = []
                st.session_state['zona_1'].extend(h3_indices_total)

            
            conditions_etapas2 = ''
            conditions_matrices2 = ''
            st.session_state['zona_2'] = []
            
            if filtro_seleccion2 != 'Todos':
                if tipo_filtro == 'OD y Transferencias':
                    conditions_etapas2 = f" AND (inicio_norm = '{filtro_seleccion2}' OR transfer1_norm = '{filtro_seleccion2}' OR transfer2_norm = '{filtro_seleccion2}' OR fin_norm = '{filtro_seleccion2}')"
                else:
                    conditions_etapas2 = f" AND (inicio_norm = '{filtro_seleccion2}' OR fin_norm = '{filtro_seleccion2}')"
                
                conditions_matrices2 = f" AND (inicio = '{filtro_seleccion2}' OR fin = '{filtro_seleccion2}')"
                geometry = zonificaciones[
                    (zonificaciones.zona == zona_filtro_seleccion2) & 
                    (zonificaciones.id == filtro_seleccion2)
                ].geometry.values[0]
                
                # Inicializar una lista para almacenar los índices H3
                h3_indices_total = []
                
                # Verificar el tipo de geometría
                if isinstance(geometry, Polygon):
                    # Si es un Polygon, procesarlo directamente
                    h3_indices = get_h3_indices_in_geometry(geometry, 8)
                    h3_indices_total.extend(h3_indices)
                elif isinstance(geometry, MultiPolygon):
                    # Si es un MultiPolygon, iterar sobre cada Polygon
                    for poly in geometry.geoms:
                        h3_indices = get_h3_indices_in_geometry(poly, 8)
                        h3_indices_total.extend(h3_indices)
                else:
                    st.error("La geometría proporcionada no es un Polygon ni un MultiPolygon.")
                
                # Extender los índices H3 en el estado de la sesión
                st.session_state['zona_2'] = []
                st.session_state['zona_2'].extend(h3_indices_total)

            query_etapas = query + conditions_etapas1 + conditions_etapas2
            query_matrices = query + conditions_matrices1 + conditions_matrices2

            if ((filtro_seleccion1 != 'Todos')|(filtro_seleccion2 != 'Todos'))&((zona_seleccionada!=zona_filtro_seleccion1)|(zona_seleccionada!=zona_filtro_seleccion2)):
                #Cuando la zonificación de los filtros es diferente a la zonificación del mapa

                if (filtro_seleccion1 == 'Todos')|(filtro_seleccion2 == 'Todos'):
                    col2.write('')
                    col2.write('')
                    col2.write('Si la zona del filtro 1 o del filtro 2 son diferentes a la zonificación elegida, Filtro 1 o Filtro 2 no se puede ser igual a "Todos"')
                    col2.write('')
                    col2.write('')
                    st.session_state.etapas_all = pd.DataFrame([])
                    st.session_state.matrices_all = pd.DataFrame([])
                else:
                    
                    agg_etapas, agg_matrices = traigo_tablas_con_filtros(mes_seleccionado, 
                                                                         tipo_dia_seleccionado, 
                                                                         zona_seleccionada, 
                                                                         zona_filtro_seleccion1, 
                                                                         filtro_seleccion1, 
                                                                         zona_filtro_seleccion2, 
                                                                         filtro_seleccion2, 
                                                                         tipo_filtro,
                                                                         equivalencia_zonas,
                                                                         zonificaciones,
                                                                         )
                    st.session_state.etapas_all = agg_etapas.copy()
                    st.session_state.matrices_all = agg_matrices.copy()

            else:

                st.session_state.etapas_all = levanto_tabla_sql_local('agg_etapas', tabla_tipo='dash', query=f"SELECT * FROM agg_etapas{query_etapas}")
                print(query_matrices)
                st.session_state.matrices_all = levanto_tabla_sql_local('agg_matrices', tabla_tipo='dash', query=f"SELECT * FROM agg_matrices{query_matrices}")    

            if len(st.session_state.matrices_all)!=0:

                if mes_seleccionado != 'Todos':            
                    st.session_state.socio_indicadores_all = socio_indicadores[(socio_indicadores.mes==mes_seleccionado)&(socio_indicadores.tipo_dia==tipo_dia_seleccionado)].copy()
        
                else:
                    st.session_state.socio_indicadores_all = socio_indicadores[(socio_indicadores.tipo_dia==tipo_dia_seleccionado)].copy()

                st.session_state.socio_indicadores_all = st.session_state.socio_indicadores_all.groupby(["tabla", "tipo_dia", "Genero", "Tarifa", "Modo"], as_index=False)[[
                                    "Distancia", "Tiempo de viaje", "Velocidad", "Etapas promedio", "Viajes promedio", "Tiempo entre viajes", "factor_expansion_linea"
                                    ]] .mean().round(2)
    
                if transfer_seleccionado == 'Todos':
                    st.session_state.desc_transfers = True
                else:
                    st.session_state.desc_transfers = False
       
                if modo_seleccionado == 'Todos':
                    st.session_state.desc_modos = True
                else:
                    st.session_state.desc_modos = False
        
                if rango_hora_seleccionado == 'Todos':
                    st.session_state.desc_horas = True
                else:
                    st.session_state.desc_horas = False
        
                if distancia_seleccionada == 'Todas':
                    st.session_state.desc_distancia = True
                else:
                    st.session_state.desc_distancia = False
        
                st.session_state.agg_cols_etapas = ['zona',
                                                   'inicio_norm',
                                                   'transfer1_norm',
                                                   'transfer2_norm',
                                                   'fin_norm',
                                                   'transferencia',
                                                   'modo_agregado',
                                                   'rango_hora',
                                                   'distancia']
                st.session_state.agg_cols_viajes = ['zona',
                                                   'inicio_norm',
                                                   'fin_norm',
                                                   'transferencia',
                                                   'modo_agregado',
                                                   'rango_hora',
                                                   'distancia']
                
        if len(st.session_state.etapas_all)==0:
            col2.write('No hay datos para mostrar')
        else:

            if not st.session_state.data_cargada or \
                        hay_cambios_en_filtros(current_options, st.session_state.last_options) or \
                        hay_cambios_en_filtros(current_filters, st.session_state.last_filters):
                
                # Actualiza los filtros en `session_state` para detectar cambios futuros
                st.session_state.last_filters = current_filters.copy()    
                st.session_state.last_options = current_options.copy()
                st.session_state.data_cargada = True    

                st.session_state.etapas,   \
                st.session_state.viajes,   \
                st.session_state.matriz,   \
                st.session_state.origenes, \
                st.session_state.destinos, \
                st.session_state.transferencias = create_data_folium(st.session_state.etapas_all.copy(),
                                                                st.session_state.matrices_all.copy(),
                                                                agg_transferencias=st.session_state.desc_transfers,
                                                                agg_modo=st.session_state.desc_modos,
                                                                agg_hora=st.session_state.desc_horas,
                                                                agg_distancia=st.session_state.desc_distancia,
                                                                agg_cols_etapas=st.session_state.agg_cols_etapas,
                                                                agg_cols_viajes=st.session_state.agg_cols_viajes,
                                                                etapas_seleccionada=st.session_state.etapas_seleccionada,
                                                                viajes_seleccionado=st.session_state.viajes_seleccionado,
                                                                origenes_seleccionado=st.session_state.origenes_seleccionado,
                                                                destinos_seleccionado=st.session_state.destinos_seleccionado,
                                                                transferencias_seleccionado=st.session_state.transferencias_seleccionado)
                
                if ((len(st.session_state.etapas) > 0)           \
                    | (len(st.session_state.viajes) > 0)         \
                    | (len(st.session_state.origenes) > 0)       \
                    | (len(st.session_state.destinos) > 0)       \
                    | (len(st.session_state.transferencias) > 0))\
                    | (len(zona_mostrar)>0):

                    latlon = bring_latlon()

                    st.session_state.map = crear_mapa_lineas_deseo(df_viajes=st.session_state.viajes,
                                                      df_etapas=st.session_state.etapas,
                                                      zonif=zonif,
                                                      origenes=st.session_state.origenes,
                                                      destinos=st.session_state.destinos,
                                                      transferencias=st.session_state.transferencias,
                                                      var_fex='factor_expansion_linea',
                                                      cmap_viajes='viridis_r',
                                                      cmap_etapas='magma_r',
                                                      map_title='Líneas de Deseo',
                                                      savefile='',
                                                      k_jenks=5,
                                                      latlon=latlon)
                   
                    if st.session_state.map:
                        with col2:
                            folium_static(st.session_state.map, width=1000, height=800)
                            # output = st_folium(st.session_state.map, width=1000, height=800, key='m', returned_objects=["center"])
                        if mtabla:
                            if len(st.session_state.etapas)>0:
                                col2.write('Etapas')
                                # col2.dataframe(st.session_state.etapas[['inicio_norm', 
                                #                                         'transfer1_norm', 
                                #                                         'transfer2_norm', 
                                #                                         'fin_norm', 
                                #                                         'factor_expansion_linea']].rename(columns={'factor_expansion_linea':'Etapas'}))  #
                                col2.write(st.session_state.etapas)
                            if len(st.session_state.viajes)>0:
                                col2.write('Viajes')
                                # col2.dataframe(st.session_state.viajes[['inicio_norm', 'fin_norm', 'factor_expansion_linea']].rename(columns={'factor_expansion_linea':'Viajes'})) #
                                col2.write(st.session_state.viajes)

                    else:
                        col2.text("No hay datos suficientes para mostrar el mapa.")
                else:
                    col2.text("No hay datos suficientes para mostrar el mapa.")


with st.expander('Matrices'):

    col1, col2 = st.columns([1, 4])

    if len(st.session_state.matriz) > 0:

        tipo_matriz = col1.selectbox(
                'Variable', options=['Viajes', 'Distancia promedio (kms)', 'Tiempo promedio (min)', 'Velocidad promedio (km/h)'])

        normalize = False
        if tipo_matriz == 'Viajes':
            var_matriz = 'factor_expansion_linea'
            normalize = col1.checkbox('Normalizar', value=True)

        mmatriz = col1.checkbox('Mostrar tabla', value=False, key='mmatriz')
        col1.write(f'Mes: {mes_seleccionado}')
        col1.write(f'Tipo día: {tipo_dia_seleccionado}')
        col1.write(f'Transferencias: {transfer_seleccionado}')
        col1.write(f'Modos: {modo_seleccionado}')
        col1.write(f'Rango hora: {rango_hora_seleccionado}')
        col1.write(f'Distancias: {distancia_seleccionada}')        
    
        if tipo_matriz == 'Distancia promedio (kms)':
            var_matriz = 'distance_osm_drive'
        if tipo_matriz == 'Tiempo promedio (min)':
            var_matriz = 'travel_time_min'
        if tipo_matriz == 'Velocidad promedio (km/h)':
            var_matriz = 'travel_speed'

        od_heatmap = pd.crosstab(
            index=st.session_state.matriz['Origen'],
            columns=st.session_state.matriz['Destino'],
            values=st.session_state.matriz[var_matriz],
            aggfunc="sum",
            normalize=normalize,
        )
        
        if normalize:
            od_heatmap = (od_heatmap * 100).round(2)
        else:
            od_heatmap = od_heatmap.round(0)
        
        od_heatmap = od_heatmap.reset_index()
        od_heatmap['Origen'] = od_heatmap['Origen'].str[4:]
        od_heatmap = od_heatmap.set_index('Origen')
        od_heatmap.columns = [i[4:] for i in od_heatmap.columns]

        fig = px.imshow(od_heatmap, text_auto=True,
                        color_continuous_scale='Blues',)

        fig.update_coloraxes(showscale=False)

        if len(od_heatmap) <= 20:
            fig.update_layout(width=1000, height=1000)
        elif (len(od_heatmap) > 20) & (len(od_heatmap) <= 40):
            fig.update_layout(width=1000, height=1000)
        elif len(od_heatmap) > 40:
            fig.update_layout(width=1000, height=1000)

        col2.plotly_chart(fig)
        if mmatriz:
            col2.write(st.session_state.matriz)

    else:
        col2.text('No hay datos para mostrar')



with st.expander('Zonas', expanded=False):
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
    zona1 = st.session_state['zona_1']
    zona2 = st.session_state['zona_2']

    if len(zona1) > 0:
        query1 = f"SELECT * FROM etapas_agregadas WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND ({zona_filtro_seleccion1}_o = '{filtro_seleccion1}');"     
        etapas1 = levanto_tabla_sql_local('etapas_agregadas', tabla_tipo='dash', query=query1)

        if len(etapas1) > 0:
            etapas1['Zona_1'] = 'Zona 1'

            ## Viajes
            query1 = f"SELECT * FROM viajes_agregados WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND {zona_filtro_seleccion1}_o = '{filtro_seleccion1}';"
            viajes1 = levanto_tabla_sql_local('viajes_agregados', tabla_tipo='dash', query=query1)
            viajes1['Zona_1'] = 'Zona 1'

            modos_e1 = etapas1.groupby(['modo', 'nombre_linea'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Etapas', 
                                                                                                                      'nombre_linea': 'Línea', 'modo': 'Modo'})

            modos_v1 = viajes1.groupby(['modo'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Viajes', 
                                                                                                                      'modo': 'Modo'})

            # Calculate the total and append as a new row
            total_row1e = pd.DataFrame({
                'Modo': ['Total'],
                'Línea': ['-'],
                'Etapas': [modos_e1['Etapas'].sum()]
            })                        
            modos_e1 = pd.concat([modos_e1, total_row1e], ignore_index=True)

            
            # Calculate the total and append as a new row
            total_row1 = pd.DataFrame({
                'Modo': ['Total'],                
                'Viajes': [modos_v1['Viajes'].sum()]
            })                        
            modos_v1 = pd.concat([modos_v1, total_row1], ignore_index=True)
            
            col2.markdown(
                f"""
                <h3 style='font-size:22px;'>{filtro_seleccion1}</h3>
                """, 
                unsafe_allow_html=True
            )

            col2.write('Etapas')        
            modos_e1['Etapas'] = modos_e1['Etapas'].round()            
            col2.dataframe(modos_e1.set_index('Modo'), height=400, width=400)
            
            col3.markdown(
                f"""
                <h3 style='font-size:22px;'></h3>
                """, 
                unsafe_allow_html=True
            )
            col3.write('Viajes')
            modos_v1['Viajes'] = modos_v1['Viajes'].round()
            col3.dataframe(modos_v1.set_index('Modo'), height=400, width=300)
    
    if len(zona2) > 0:

        query2 = f"SELECT * FROM etapas_agregadas WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND ({zona_filtro_seleccion2}_o = '{filtro_seleccion2}');"     
        etapas2 = levanto_tabla_sql_local('etapas_agregadas', tabla_tipo='dash', query=query2)

        if len(etapas2) > 0:

            ## Etapas
            if len(etapas2) > 0:
                etapas2['Zona_2'] = 'Zona 2'
                
                ## Viajes                
                query2 = f"SELECT * FROM viajes_agregados WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND {zona_filtro_seleccion2}_o = '{filtro_seleccion2}';"
                viajes2 = levanto_tabla_sql_local('viajes_agregados', tabla_tipo='dash', query=query2)
                viajes2['Zona_2'] = 'Zona 2'
    
                modos_e2 = etapas2.groupby(['modo', 'nombre_linea'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Etapas', 
                                                                                                                          'nombre_linea': 'Línea', 'modo': 'Modo'})
    
                modos_v2 = viajes2.groupby(['modo'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Viajes', 
                                                                                                                          'modo': 'Modo'})
                # Calculate the total and append as a new row
                total_row2e = pd.DataFrame({
                    'Modo': ['Total'],
                    'Línea': ['-'],
                    'Etapas': [modos_e2['Etapas'].sum()]
                })                        
                modos_e2 = pd.concat([modos_e2, total_row2e], ignore_index=True)
                
                # Calculate the total and append as a new row
                total_row2 = pd.DataFrame({
                    'Modo': ['Total'],                
                    'Viajes': [modos_v2['Viajes'].sum()]
                })                        
                modos_v2 = pd.concat([modos_v2, total_row2], ignore_index=True)
 
                col4.markdown(
                        f"""
                        <h3 style='font-size:22px;'>{filtro_seleccion2}</h3>
                        """, 
                        unsafe_allow_html=True
                    )
                col4.write('Etapas')  
                modos_e2['Etapas'] = modos_e2['Etapas'].round()
                col4.dataframe(modos_e2.set_index('Modo'), height=400, width=400)
    
                modos_v2['Viajes'] = modos_v2['Viajes'].round()
                col5.markdown(
                    f"""
                    <h3 style='font-size:22px;'></h3>
                    """, 
                    unsafe_allow_html=True
                )
    
                col5.write('Viajes')
                col5.dataframe(modos_v2.set_index('Modo'), height=400, width=300)

with st.expander('Viajes entre zonas', expanded=True):
    col1, col2, col3 = st.columns([1, 2, 4]) 

    transferencias_modos = pd.DataFrame([])
    modos_e = pd.DataFrame([])
    modos_v = pd.DataFrame([])
    transferencias = pd.DataFrame([])
    zonasod_e = pd.DataFrame([])
    zonasod_v = pd.DataFrame([])
    
    if len(zona1) > 0 and len(zona2) > 0:

        col1.write(f'Mes: {mes_seleccionado}')
        col1.write(f'Tipo día: {tipo_dia_seleccionado}')
        col1.write(f'Zona 1: {filtro_seleccion1}')
        col1.write(f'Zona 2: {filtro_seleccion2}')

        ## Etapas
        h3_values = [filtro_seleccion1, filtro_seleccion2]
        h3_values = ', '.join(f"'{valor}'" for valor in h3_values)
        if zona_seleccionada == zona_filtro_seleccion1 == zona_filtro_seleccion2:            
            query = f"SELECT * FROM etapas_agregadas WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND ({zona_seleccionada}_o IN ({h3_values}) OR {zona_seleccionada}_d IN ({h3_values}));"
        else:
            query = f"""SELECT * FROM etapas_agregadas WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND
                        ((({zona_filtro_seleccion1}_o IN ({h3_values}) OR {zona_filtro_seleccion1}_d IN ({h3_values}))) AND 
                         (({zona_filtro_seleccion2}_o IN ({h3_values}) OR {zona_filtro_seleccion2}_d IN ({h3_values}))));"""

        etapas = levanto_tabla_sql_local('etapas_agregadas', tabla_tipo='dash', query=query)

        if len(etapas) > 0:
        
            etapas['Zona_1'] = ''
            etapas['Zona_2'] = ''
            etapas.loc[etapas.h3_o.isin(zona1), 'Zona_1'] = 'Zona 1'
            etapas.loc[etapas.h3_o.isin(zona2), 'Zona_1'] = 'Zona 2'
            etapas.loc[etapas.h3_d.isin(zona1), 'Zona_2'] = 'Zona 1'
            etapas.loc[etapas.h3_d.isin(zona2), 'Zona_2'] = 'Zona 2'
            etapas = etapas[(etapas.Zona_1 != '') & (etapas.Zona_2 != '') & (etapas.Zona_1 != etapas.Zona_2)]
            
            etapas = etapas.fillna('')
            
            zonasod_e = etapas.groupby(['Zona_1', 'Zona_2'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Etapas'}) #.round()
            zonasod_e['Etapas'] = zonasod_e['Etapas'].apply(lambda x: f"{int(x):,}")

            zonasod_e['Zonas'] = zonasod_e['Zona_1'] + ' - ' + zonasod_e['Zona_2']
            zonasod_e = zonasod_e[['Zonas', 'Etapas']]
            
            modos_e = etapas.groupby(['modo', 'nombre_linea'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Viajes', 
                                                                                                                                  'nombre_linea': 'Líneas', 
                                                                                                                                  'modo': 'Modo'}) #.round()

            
        ## Viajes
        h3_values = [filtro_seleccion1, filtro_seleccion2]
        h3_values = ', '.join(f"'{valor}'" for valor in h3_values)
        if zona_seleccionada == zona_filtro_seleccion1 == zona_filtro_seleccion2:
            query = f"SELECT * FROM viajes_agregados WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND ({zona_seleccionada}_o IN ({h3_values}) OR {zona_seleccionada}_d IN ({h3_values}));"
        else:
            query = f"""SELECT * FROM viajes_agregados WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND 
                        (({zona_filtro_seleccion1}_o IN ({h3_values}) OR {zona_filtro_seleccion1}_d IN ({h3_values})) 
                           AND ({zona_filtro_seleccion2}_o IN ({h3_values}) OR {zona_filtro_seleccion2}_d IN ({h3_values})));"""
            
        viajes = levanto_tabla_sql_local('viajes_agregados', tabla_tipo='dash', query=query)
        
        if len(viajes) > 0:
        
            viajes['Zona_1'] = ''
            viajes['Zona_2'] = ''
            viajes.loc[viajes.h3_o.isin(zona1), 'Zona_1'] = 'Zona 1'
            viajes.loc[viajes.h3_o.isin(zona2), 'Zona_1'] = 'Zona 2'
            viajes.loc[viajes.h3_d.isin(zona1), 'Zona_2'] = 'Zona 1'
            viajes.loc[viajes.h3_d.isin(zona2), 'Zona_2'] = 'Zona 2'
            viajes = viajes[(viajes.Zona_1 != '') & (viajes.Zona_2 != '') & (viajes.Zona_1 != viajes.Zona_2)]

            zonasod_v = viajes.groupby(['Zona_1', 'Zona_2'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Viajes'})
           
            zonasod_v['Zonas'] = zonasod_v['Zona_1'] + ' - ' + zonasod_v['Zona_2']
            zonasod_v = zonasod_v[['Zonas', 'Viajes']]
            zonasod_v['Viajes'] = zonasod_v['Viajes'].apply(lambda x: f"{int(x):,}")

            modos_v = viajes.groupby(['modo'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Viajes', 
                                                                                                                   'modo': 'Modo'})

            if len(modos_v)>0:
                # Calculate the total and append as a new row
                total_row = pd.DataFrame({
                    'Modo': ['Total'],
                    'Viajes': [modos_v['Viajes'].sum()]
                })                        
                modos_v = pd.concat([modos_v, total_row], ignore_index=True)
                modos_v['Viajes'] = modos_v['Viajes'].apply(lambda x: f"{int(x):,}")

      # Transferencias
        h3_values = [filtro_seleccion1, filtro_seleccion2]
        h3_values = ', '.join(f"'{valor}'" for valor in h3_values)
        if zona_seleccionada == zona_filtro_seleccion1 == zona_filtro_seleccion2:            
            query = f"SELECT * FROM transferencias_agregadas WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND ({zona_seleccionada}_o IN ({h3_values}) OR {zona_seleccionada}_d IN ({h3_values}));"
        else:
            query = f"""SELECT * FROM transferencias_agregadas WHERE mes = '{mes_seleccionado}' AND tipo_dia = '{tipo_dia_seleccionado}' AND 
            (({zona_filtro_seleccion1}_o IN ({h3_values}) OR {zona_filtro_seleccion1}_d IN ({h3_values})) AND
             ({zona_filtro_seleccion2}_o IN ({h3_values}) OR {zona_filtro_seleccion2}_d IN ({h3_values})));"""

        
        transferencias = levanto_tabla_sql_local('transferencias_agregadas', tabla_tipo='dash', query=query)

        
        if len(transferencias) > 0:
    
            transferencias['Zona_1'] = ''
            transferencias['Zona_2'] = ''
            transferencias.loc[transferencias.h3_o.isin(zona1), 'Zona_1'] = 'Zona 1'
            transferencias.loc[transferencias.h3_o.isin(zona2), 'Zona_1'] = 'Zona 2'
            transferencias.loc[transferencias.h3_d.isin(zona1), 'Zona_2'] = 'Zona 1'
            transferencias.loc[transferencias.h3_d.isin(zona2), 'Zona_2'] = 'Zona 2'
            transferencias = transferencias[(transferencias.Zona_1 != '') & (transferencias.Zona_2 != '') & (transferencias.Zona_1 != transferencias.Zona_2)]
            
            transferencias = transferencias.fillna('')
            
            transferencias = transferencias.groupby(['modo', 
                                                     'seq_lineas'], as_index=False).factor_expansion_linea.sum().rename(columns={'factor_expansion_linea':'Viajes', 
                                                                                                                                 'modo':'Modo', 'seq_lineas':'Líneas'}).sort_values('Viajes', ascending=False)

            # Calculate the total and append as a new row
            if len(transferencias)>0:
                total_rowe = pd.DataFrame({
                    'Modo': ['Total'],
                    'Líneas': ['-'],
                    'Viajes': [transferencias['Viajes'].sum()]
                })                        
                transferencias = pd.concat([transferencias, total_rowe], ignore_index=True)
                transferencias['Viajes'] = transferencias['Viajes'].apply(lambda x: f"{int(x):,}")

    
        # Muestro resultados en el dashboard

        col2.write('Etapas')
        if len(zonasod_e) > 0:            
            col2.dataframe(zonasod_e.set_index('Zonas'), height=100, width=300)    
        else:
            col2.write('No hay datos para mostrar')

        col2.write('Viajes')                 
        if len(zonasod_v):               
            col2.dataframe(zonasod_v.set_index('Zonas'), height=100, width=300)    
        else:
            col2.write('No hay datos para mostrar')
        
        col2.write('Modal')                
        if len(modos_v)>0:
            col2.dataframe(modos_v.set_index('Modo'), height=300, width=300)    
        else:
            col2.write('No hay datos para mostrar')
        
        col3.write('Viajes por líneas')        
        if len(transferencias)>0:
            col3.dataframe(transferencias.set_index('Modo'), height=700, width=800)        
        else:
            col3.write('No hay datos para mostrar')

        

