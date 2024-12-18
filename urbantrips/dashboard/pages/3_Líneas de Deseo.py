import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
import mapclassify
import plotly.express as px
from folium import Figure
from shapely import wkt
from dash_utils import (
    levanto_tabla_sql, get_logo,
    create_data_folium, traigo_indicadores,
    extract_hex_colors_from_cmap,
    iniciar_conexion_db, normalize_vars,
    bring_latlon, traigo_zonas_values, get_h3_indices_in_geometry
)

def crear_mapa_lineas_deseo(df_viajes,
                            df_etapas,
                            zonif,
                            origenes,
                            destinos,
                            transferencias,
                            var_fex,
                            cmap_viajes='Blues',
                            cmap_etapas='Greens',
                            map_title='',
                            savefile='',
                            k_jenks=5,
                            latlon=''):

    m = ''
    # if (len(df_viajes) > 0) | (len(df_etapas) > 0) | (len(origenes) > 0) | (len(destinos) > 0) | (len(transferencias) > 0) | (desc_zonif):
    if True:
        if len(latlon) == 0:
            if len(df_etapas) > 0:
                y_val = df_etapas.sample(100, replace=True).geometry.representative_point().y.mean()
                x_val = df_etapas.sample(100, replace=True).geometry.representative_point().x.mean()
            elif len(df_viajes) > 0:
                y_val = df_viajes.sample(100, replace=True).geometry.representative_point().y.mean()
                x_val = df_viajes.sample(100, replace=True).geometry.representative_point().x.mean()
            elif len(origenes) > 0:
                y_val = origenes.sample(100, replace=True).geometry.representative_point().y.mean()
                x_val = origenes.sample(100, replace=True).geometry.representative_point().x.mean()
            elif len(destinos) > 0:
                y_val = destinos.sample(100, replace=True).geometry.representative_point().y.mean()
                x_val = destinos.sample(100, replace=True).geometry.representative_point().x.mean()
            elif len(transferencias) > 0:
                y_val = transferencias.sample(100, replace=True).geometry.representative_point().y.mean()
                x_val = transferencias.sample(100, replace=True).geometry.representative_point().x.mean()

            latlon = [y_val, x_val] 

        fig = Figure(width=800, height=600)
        m = folium.Map(location=latlon,
                       zoom_start=10, tiles='cartodbpositron')

        colors_viajes = extract_hex_colors_from_cmap(
            cmap='viridis_r', n=k_jenks)
        colors_etapas = extract_hex_colors_from_cmap(cmap='magma_r', n=k_jenks)

        # Etapas
        line_w = 0.5
        if len(df_etapas) > 0:
            try:
                bins = [df_etapas[var_fex].min()-1] + \
                    mapclassify.FisherJenks(
                        df_etapas[var_fex], k=k_jenks).bins.tolist()
            except ValueError:
                bins = [df_etapas[var_fex].min()-1] + \
                    mapclassify.FisherJenks(
                        df_etapas[var_fex], k=k_jenks-3).bins.tolist()
            except ValueError:
                bins = [df_etapas[var_fex].min()-1] + \
                    mapclassify.FisherJenks(
                        df_etapas[var_fex], k=1).bins.tolist()

            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} etapas' for n in range_bins]
            df_etapas['cuts'] = pd.cut(
                df_etapas[var_fex], bins=bins, labels=bins_labels)

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

        # Viajes
        line_w = 0.5
        if len(df_viajes) > 0:

            try:
                # Intentar clasificar con k clases
                bins = [df_viajes[var_fex].min() - 1] + \
                       mapclassify.FisherJenks(df_viajes[var_fex], k=k_jenks).bins.tolist()
            except ValueError:
                # Si falla, reducir k dinámicamente
                while k_jenks > 1:
                    try:
                        bins = [df_viajes[var_fex].min() - 1] + \
                               mapclassify.FisherJenks(df_viajes[var_fex], k=k_jenks - 1).bins.tolist()
                        break
                    except ValueError:
                        k_jenks -= 1
                else:
                    # Si no se puede crear ni una categoría, asignar un único bin
                    bins = [df_viajes[var_fex].min() - 1, df_viajes[var_fex].max()]
            
            # Eliminar duplicados en bins
            bins = sorted(set(bins))
            
            # Crear etiquetas únicas para los bins
            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins
            ]
            
            # Garantizar que las etiquetas sean únicas
            bins_labels = [f"{label} ({i})" for i, label in enumerate(bins_labels)]
            
            # Aplicar pd.cut con ordered=False para evitar el error
            df_viajes['cuts'] = pd.cut(
                df_viajes[var_fex], bins=bins, labels=bins_labels, ordered=False
            )

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

        if len(origenes) > 0:
            try:
                # Intentar clasificar con k inicial
                bins = [origenes['factor_expansion_linea'].min() - 1] + \
                       mapclassify.FisherJenks(origenes['factor_expansion_linea'], k=5).bins.tolist()
            except ValueError:
                # Reducir k dinámicamente en caso de error
                k = 5
                while k > 1:
                    try:
                        bins = [origenes['factor_expansion_linea'].min() - 1] + \
                               mapclassify.FisherJenks(origenes['factor_expansion_linea'], k=k).bins.tolist()
                        break
                    except ValueError:
                        k -= 1
                else:
                    # Si no se pueden generar bins, usar un único bin
                    bins = [origenes['factor_expansion_linea'].min() - 1, origenes['factor_expansion_linea'].max()]
            
            print(bins)


            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} origenes' for n in range_bins]

            origenes['cuts'] = pd.cut(
                origenes['factor_expansion_linea'], bins=bins, labels=bins_labels)

            n = 0
            line_w = 10
            for i in bins_labels:

                origenes[origenes.cuts == i].explore(
                    m=m,
                    color="#0173b299",
                    style_kwds={'fillOpacity': 0.1, 'weight': line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 5

        if len(destinos) > 0:
            try:
                bins = [destinos['factor_expansion_linea'].min()-1] + \
                    mapclassify.FisherJenks(
                        destinos['factor_expansion_linea'], k=5).bins.tolist()
            except ValueError:
                bins = [destinos['factor_expansion_linea'].min()-1] + \
                    mapclassify.FisherJenks(
                        destinos['factor_expansion_linea'], k=5-3).bins.tolist()

            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} destinos' for n in range_bins]

            destinos['cuts'] = pd.cut(
                destinos['factor_expansion_linea'], bins=bins, labels=bins_labels)

            n = 0
            line_w = 10
            for i in bins_labels:

                destinos[destinos.cuts == i].explore(
                    m=m,
                    color="#de8f0599",
                    style_kwds={'fillOpacity': 0.1, 'weight': line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 5

        if len(transferencias) > 0:
            try:
                bins = [transferencias['factor_expansion_linea'].min()-1] + \
                    mapclassify.FisherJenks(
                        transferencias['factor_expansion_linea'], k=5).bins.tolist()
            except ValueError:
                bins = [transferencias['factor_expansion_linea'].min()-1] + \
                    mapclassify.FisherJenks(
                        transferencias['factor_expansion_linea'], k=5-3).bins.tolist()

            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} transferencias' for n in range_bins]

            transferencias['cuts'] = pd.cut(
                transferencias['factor_expansion_linea'], bins=bins, labels=bins_labels)

            n = 0
            line_w = 10
            for i in bins_labels:

                transferencias[transferencias.cuts == i].explore(
                    m=m,
                    color="#FF0000",
                    style_kwds={'fillOpacity': 0.1, 'weight': line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 5
        
        # Agrego zonificación
        if len(zonif) > 0:
            geojson = zonif.to_json()
            # Add the GeoJSON to the map as a GeoJson Layer
            folium.GeoJson(
                geojson,
                name='Zonificación',
                style_function=lambda feature: {
                    'fillColor': 'navy',
                    'color': 'navy',
                    'weight': .5,
                    'fillOpacity': .2,

                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['id'], labels=False, sticky=False)
            ).add_to(m)

        folium.LayerControl(name='mapa').add_to(m)

    return m

def levanto_tabla_sql_local(tabla_sql, tabla_tipo="dash", query=''):

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

    tabla = normalize_vars(tabla)

    return tabla
    


# Función para detectar cambios
def hay_cambios_en_filtros(current, last):
    return current != last

st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)

with st.expander('Líneas de Deseo', expanded=True):

    col1, col2 = st.columns([1, 4])

    variables = [
            'last_filters', 'last_options', 'data_cargada', 'etapas_lst', 'matrices_all', 'etapas_all',
            'matrices_', 'etapas_', 'etapas', 'viajes', 'matriz', 'origenes', 'destinos',
            'general', 'modal', 'distancias', 'mes', 'tipo_dia', 'zona', 'transferencia', 
            'modo_agregado', 'rango_hora', 'distancia', 'socio_indicadores_', 'general_', 'modal_', 'distancias_'
        ]
        
    # Inicializar todas las variables con None si no existen en session_state
    for var in variables:
        if var not in st.session_state:
            st.session_state[var] = ''
    
    etapas_lst_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT mes FROM agg_etapas;')
   
    if len(etapas_lst_) > 0:

        zonificaciones = levanto_tabla_sql('zonificaciones')
        socio_indicadores = levanto_tabla_sql('socio_indicadores')
        desc_tipo_dia_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT tipo_dia FROM agg_etapas;')
        desc_zona_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT zona FROM agg_etapas;').sort_values('zona')
        modos_list_all_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT modo_agregado FROM agg_etapas;')
        rango_hora_all_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT rango_hora FROM agg_etapas;')
        distancia_all_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT distancia FROM agg_etapas;')
        zonas_values = traigo_zonas_values('etapas')


        # st.session_state.etapas_all = st.session_state.etapas_all[st.session_state.etapas_all.factor_expansion_linea > 0].copy()
        # general, modal, distancias = traigo_indicadores('all')
        
        # Inicializar valores de `st.session_state` solo si no existen
        if 'last_filters' not in st.session_state:
            st.session_state.last_filters = {
                'mes': 'Todos',
                'tipo_dia': None,
                'zona': None,
                'transferencia': 'Todos',
                'modo_agregado': 'Todos',
                'rango_hora': 'Todos',
                'distancia': 'Todas',
                'desc_zonas_values':'Todos'
            }
            
        if 'data_cargada' not in st.session_state:
            st.session_state.data_cargada = False
        
        # Opciones de los filtros en Streamlit
        st.session_state.etapas_lst = ['Todos'] + etapas_lst_.mes.unique().tolist()        
        desc_mes = col1.selectbox('Mes', options=st.session_state.etapas_lst, index=1)
        
        desc_tipo_dia = col1.selectbox('Tipo día', options=desc_tipo_dia_.tipo_dia.unique())
        desc_zona = col1.selectbox('Zonificación', options=desc_zona_.zona.unique())
        transf_list_all = ['Todos', 'Con transferencia', 'Sin transferencia']
        transf_list = col1.selectbox('Transferencias', options=transf_list_all)
        
        modos_list_all = ['Todos'] + modos_list_all_[modos_list_all_.modo_agregado != '99'].modo_agregado.unique().tolist()
        # modos_list = col1.selectbox('Modos', options=[text.capitalize() for text in modos_list_all])
        modos_list = col1.selectbox('Modos', options=[text for text in modos_list_all])
        
        rango_hora_all = ['Todos'] + rango_hora_all_[rango_hora_all_.rango_hora != '99'].rango_hora.unique().tolist()
        rango_hora = col1.selectbox('Rango hora', options=[text for text in rango_hora_all])
        
        distancia_all = ['Todas'] + distancia_all_[distancia_all_.distancia != '99'].distancia.unique().tolist()
        distancia = col1.selectbox('Distancia', options=distancia_all)

        desc_et_vi = col1.selectbox('Datos de', options=['Etapas', 'Viajes', 'Ninguno'], index=1)
        if desc_et_vi == 'Viajes':
            desc_viajes = True
            desc_etapas = False
        elif desc_et_vi == 'Etapas':
            desc_viajes = False
            desc_etapas = True
        else:
            desc_viajes = False
            desc_etapas = False

        zonas_values_all = ['Todos'] + zonas_values[zonas_values.zona == desc_zona].Nombre.unique().tolist()
        desc_zonas_values1 = col1.selectbox('Filtro 1', options=zonas_values_all, key='filtro1')
        desc_zonas_values2 = col1.selectbox('Filtro 2', options=zonas_values_all, key='filtro2')

        

        
        desc_origenes = col1.checkbox(
            ':blue[Origenes]', value=False)

        desc_destinos = col1.checkbox(
            ':orange[Destinos]', value=False)

        desc_transferencias = col1.checkbox(
            ':red[Transferencias]', value=False)
        
        desc_zonif = col1.checkbox(
            'Mostrar zonificación', value=True)
        if desc_zonif:
            zonif = zonificaciones[zonificaciones.zona == desc_zona]
        else:
            zonif = ''

        mtabla = col2.checkbox('Mostrar tabla', value=False)
        
        # Construye el diccionario de filtros actual
        current_filters = {
            'mes': None if desc_mes == 'Todos' else desc_mes,
            'tipo_dia': desc_tipo_dia,
            'zona': None if desc_zona == 'Todos' else desc_zona,
            'transferencia': None if transf_list == 'Todos' else (1 if transf_list == 'Con transferencia' else 0),
            'modo_agregado': None if modos_list == 'Todos' else modos_list,
            'rango_hora': None if rango_hora == 'Todos' else rango_hora,
            'distancia': None if distancia == 'Todas' else distancia,
            'desc_zonas_values1': None if desc_zonas_values1 == 'Todos' else desc_zonas_values1,            
            'desc_zonas_values2': None if desc_zonas_values2 == 'Todos' else desc_zonas_values2,            
        }
        
        current_options = { 'desc_etapas': desc_etapas,
                            'desc_viajes': desc_viajes,
                            'desc_origenes': desc_origenes, 
                            'desc_destinos': desc_destinos,
                            'desc_et_vi': desc_et_vi,
                            'desc_transferencias': desc_transferencias,
                            'desc_zonif': desc_zonif, 
                            'mtabla': mtabla}
        

        
        # Solo cargar datos si hay cambios en los filtros
        if hay_cambios_en_filtros(current_filters, st.session_state.last_filters):
            
            query = ""
            conditions = " AND ".join(f"{key} = '{value}'" for key, value in current_filters.items() if (value is not None)&(key != 'desc_zonas_values1')&(key != 'desc_zonas_values2'))
            if conditions:
                query += f" WHERE {conditions}"

            conditions_etapas1 = ''
            conditions_matrices1 = ''
            st.session_state['zona_1'] = []
            
            if desc_zonas_values1 != 'Todos':
                
                conditions_etapas1 = f" AND (inicio_norm = '{desc_zonas_values1}' OR transfer1_norm = '{desc_zonas_values1}' OR transfer2_norm = '{desc_zonas_values1}' OR fin_norm = '{desc_zonas_values1}')"
                conditions_matrices1 = f" AND (inicio = '{desc_zonas_values1}' OR fin = '{desc_zonas_values1}')"
                
                geometry = zonificaciones[(zonificaciones.zona == desc_zona)&(zonificaciones.id==desc_zonas_values1)].geometry.values[0]
                h3_indices = get_h3_indices_in_geometry(geometry, 8)
                st.session_state['zona_1'].extend(h3_indices)
            
            conditions_etapas2 = ''
            conditions_matrices2 = ''
            st.session_state['zona_2'] = []
            if desc_zonas_values2 != 'Todos':
                conditions_etapas2 = f" AND (inicio_norm = '{desc_zonas_values2}' OR transfer1_norm = '{desc_zonas_values2}' OR transfer2_norm = '{desc_zonas_values2}' OR fin_norm = '{desc_zonas_values2}')"
                conditions_matrices2 = f" AND (inicio = '{desc_zonas_values2}' OR fin = '{desc_zonas_values2}')"
                geometry = zonificaciones[(zonificaciones.zona == desc_zona)&(zonificaciones.id==desc_zonas_values2)].geometry.values[0]
                h3_indices = get_h3_indices_in_geometry(geometry, 8)
                st.session_state['zona_2'].extend(h3_indices)

            query_etapas = query + conditions_etapas1 + conditions_etapas2
            query_matrices = query + conditions_matrices1 + conditions_matrices2

            st.session_state.etapas_ = levanto_tabla_sql_local('agg_etapas', tabla_tipo='dash', query=f"SELECT * FROM agg_etapas{query_etapas}")
            st.session_state.matrices_ = levanto_tabla_sql_local('agg_matrices', tabla_tipo='dash', query=f"SELECT * FROM agg_matrices{query_matrices}")    

            if len(st.session_state.matrices_)==0:
                col2.write('No hay datos para mostrar')
            else:

                if desc_mes != 'Todos':            
                    st.session_state.socio_indicadores_ = socio_indicadores[(socio_indicadores.mes==desc_mes)&(socio_indicadores.tipo_dia==desc_tipo_dia)].copy()
        
                else:
                    st.session_state.socio_indicadores_ = socio_indicadores[(socio_indicadores.tipo_dia==desc_tipo_dia)].copy()
                    
                st.session_state.etapas_ = st.session_state.etapas_.groupby(["tipo_dia", 
                                                                             "zona", 
                                                                             "inicio_norm", 
                                                                             "transfer1_norm", 
                                                                             "transfer2_norm", 
                                                                             "fin_norm", 
                                                                             "transferencia", 
                                                                             "modo_agregado", 
                                                                             "rango_hora", 
                                                                             "genero", 
                                                                             "tarifa", 
                                                                             "distancia"], as_index=False)[[
                                                                                                "distance_osm_drive", 
                                                                                                "distance_osm_drive_etapas", 
                                                                                                "travel_time_min", 
                                                                                                "travel_speed", 
                                                                                                "lat1_norm", 
                                                                                                "lon1_norm", 
                                                                                                "lat2_norm", 
                                                                                                "lon2_norm", "lat3_norm", "lon3_norm", "lat4_norm", "lon4_norm", "factor_expansion_linea"
                                                                                                        ]] .mean().round(2)
    
    
                st.session_state.matrices_ = st.session_state.matrices_.groupby(["id_polygon", 
                                                                               "tipo_dia", 
                                                                               "zona", 
                                                                               "inicio", 
                                                                               "fin", 
                                                                               "transferencia", 
                                                                               "modo_agregado", 
                                                                               "rango_hora", 
                                                                               "genero", 
                                                                               "tarifa", 
                                                                               "distancia", 
                                                                               "orden_origen", 
                                                                               "orden_destino", 
                                                                               "Origen", 
                                                                               "Destino", ], as_index=False)[[
                                                                                            "lat1", 
                                                                                            "lon1", 
                                                                                            "lat4", "lon4", "distance_osm_drive", "travel_time_min", "travel_speed", "factor_expansion_linea"]] .mean().round(2)
    
                st.session_state.socio_indicadores_ = st.session_state.socio_indicadores_.groupby(["tabla", "tipo_dia", "Genero", "Tarifa", "Modo"], as_index=False)[[
                                    "Distancia", "Tiempo de viaje", "Velocidad", "Etapas promedio", "Viajes promedio", "Tiempo entre viajes", "factor_expansion_linea"
                                    ]] .mean().round(2)
    
                
                if transf_list == 'Todos':
                    st.session_state.desc_transfers = True
                else:
                    st.session_state.desc_transfers = False
       
                if modos_list == 'Todos':
                    st.session_state.desc_modos = True
                else:
                    st.session_state.desc_modos = False
        
                if rango_hora == 'Todos':
                    st.session_state.desc_horas = True
                else:
                    st.session_state.desc_horas = False
        
                if distancia == 'Todas':
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
                
        if len(st.session_state.etapas_)==0:
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
                st.session_state.transferencias = create_data_folium(st.session_state.etapas_,
                                                                st.session_state.matrices_,
                                                                agg_transferencias=st.session_state.desc_transfers,
                                                                agg_modo=st.session_state.desc_modos,
                                                                agg_hora=st.session_state.desc_horas,
                                                                agg_distancia=st.session_state.desc_distancia,
                                                                agg_cols_etapas=st.session_state.agg_cols_etapas,
                                                                agg_cols_viajes=st.session_state.agg_cols_viajes,
                                                                desc_etapas=desc_etapas,
                                                                desc_viajes=desc_viajes,
                                                                desc_origenes=desc_origenes,
                                                                desc_destinos=desc_destinos,
                                                                desc_transferencias=desc_transferencias)

                
                if ((len(st.session_state.etapas) > 0)           \
                    | (len(st.session_state.viajes) > 0)         \
                    | (len(st.session_state.origenes) > 0)       \
                    | (len(st.session_state.destinos) > 0)       \
                    | (len(st.session_state.transferencias) > 0))\
                    | (desc_zonif):

                    latlon = bring_latlon()
                    
                    st.session_state.map = crear_mapa_lineas_deseo(df_viajes=st.session_state.viajes,
                                                      df_etapas=st.session_state.etapas,
                                                      zonif=zonif,
                                                      origenes=st.session_state.origenes,
                                                      destinos=st.session_state.destinos,
                                                      transferencias=st.session_state.transferencias,
                                                      var_fex='factor_expansion_linea',
                                                      cmap_viajes='Blues',
                                                      cmap_etapas='Greens',
                                                      map_title='Líneas de Deseo',
                                                      savefile='',
                                                      k_jenks=5,
                                                      latlon=latlon)
                   
                    if st.session_state.map:
                        with col2:
                            folium_static(st.session_state.map, width=1000, height=800)
                            # output = st_folium(st.session_state.map, width=1000, height=800, key='m', returned_objects=["center"])
                        if mtabla:
                            col2.dataframe(st.session_state.etapas_[['inicio_norm', 'transfer1_norm', 'transfer2_norm', 'fin_norm', 'factor_expansion_linea']]) #

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

        col1.write(f'Mes: {desc_mes}')
        col1.write(f'Tipo día: {desc_tipo_dia}')
        col1.write(f'Transferencias: {transf_list}')
        col1.write(f'Modos: {modos_list}')
        col1.write(f'Rango hora: {rango_hora}')
        col1.write(f'Distancias: {distancia}')        
    
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
    else:
        col2.text('No hay datos para mostrar')



with st.expander('Zonas', expanded=False):
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
    zona1 = st.session_state['zona_1']
    zona2 = st.session_state['zona_2']
   

    if len(zona1) > 0:
        query1 = f"SELECT * FROM etapas_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND ({desc_zona}_o = '{desc_zonas_values1}');"     
        etapas1 = levanto_tabla_sql_local('etapas_agregadas', tabla_tipo='dash', query=query1)


        if len(etapas1) > 0:
            etapas1['Zona_1'] = 'Zona 1'

            ## Viajes
            query1 = f"SELECT * FROM viajes_agregados WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND {desc_zona}_o = '{desc_zonas_values1}';"
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
                <h3 style='font-size:22px;'>{desc_zonas_values1}</h3>
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

        query2 = f"SELECT * FROM etapas_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND ({desc_zona}_o = '{desc_zonas_values2}');"     
        etapas2 = levanto_tabla_sql_local('etapas_agregadas', tabla_tipo='dash', query=query2)

        if len(etapas2) > 0:

            ## Etapas
            if len(etapas2) > 0:
                etapas2['Zona_2'] = 'Zona 2'
                
                ## Viajes                
                query2 = f"SELECT * FROM viajes_agregados WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND {desc_zona}_o = '{desc_zonas_values2}';"
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
                        <h3 style='font-size:22px;'>{desc_zonas_values2}</h3>
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

        col1.write(f'Mes: {desc_mes}')
        col1.write(f'Tipo día: {desc_tipo_dia}')
        col1.write(f'Zona 1: {desc_zonas_values1}')
        col1.write(f'Zona 2: {desc_zonas_values2}')

        ## Etapas
        h3_values = [desc_zonas_values1, desc_zonas_values2]
        h3_values = ', '.join(f"'{valor}'" for valor in h3_values)
        query = f"SELECT * FROM etapas_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND ({desc_zona}_o IN ({h3_values}) OR {desc_zona}_d IN ({h3_values}));"
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
        # ({desc_zona}_o = '{desc_zonas_values2}')
        h3_values = [desc_zonas_values1, desc_zonas_values2]
        h3_values = ', '.join(f"'{valor}'" for valor in h3_values)
        query = f"SELECT * FROM viajes_agregados WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND ({desc_zona}_o IN ({h3_values}) OR {desc_zona}_d IN ({h3_values}));"
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
        h3_values = [desc_zonas_values1, desc_zonas_values2]
        h3_values = ', '.join(f"'{valor}'" for valor in h3_values)
        query = f"SELECT * FROM transferencias_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND ({desc_zona}_o IN ({h3_values}) OR {desc_zona}_d IN ({h3_values}));"
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

        

