import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import mapclassify
import plotly.express as px
from folium import Figure
from shapely import wkt
from dash_utils import (
    levanto_tabla_sql, get_logo,
    create_data_folium, traigo_indicadores,
    extract_hex_colors_from_cmap,
    iniciar_conexion_db, normalize_vars

)

from streamlit_folium import folium_static
def crear_mapa_lineas_deseo(df_viajes,
                            df_etapas,
                            zonif,
                            origenes,
                            destinos,
                            var_fex,
                            cmap_viajes='Blues',
                            cmap_etapas='Greens',
                            map_title='',
                            savefile='',
                            k_jenks=5,
                            ):

    m = ''
    if (len(df_viajes) > 0) | (len(df_etapas) > 0) | (len(origenes) > 0) | (len(destinos) > 0):
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

        fig = Figure(width=800, height=600)
        m = folium.Map(location=[y_val, x_val],
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
                bins = [df_viajes[var_fex].min()-1] + \
                    mapclassify.FisherJenks(
                        df_viajes[var_fex], k=k_jenks).bins.tolist()
            except ValueError:
                bins = [df_viajes[var_fex].min()-1] + \
                    mapclassify.FisherJenks(
                        df_viajes[var_fex], k=k_jenks-2).bins.tolist()

            range_bins = range(0, len(bins)-1)
            bins_labels = [
                f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins]
            df_viajes['cuts'] = pd.cut(
                df_viajes[var_fex], bins=bins, labels=bins_labels)

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
                bins = [origenes['factor_expansion_linea'].min()-1] + \
                    mapclassify.FisherJenks(
                        origenes['factor_expansion_linea'], k=5).bins.tolist()
            except ValueError:
                bins = [origenes['factor_expansion_linea'].min()-1] + \
                    mapclassify.FisherJenks(
                        origenes['factor_expansion_linea'], k=5-3).bins.tolist()

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

        folium.LayerControl(name='xxx').add_to(m)

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
    
def traigo_socio_indicadores(socio_indicadores):    

    df = socio_indicadores[socio_indicadores.tabla=='viajes-genero-tarifa'].copy()
    totals = pd.crosstab(values=df.factor_expansion_linea, columns=df.Genero, index=df.Tarifa, aggfunc='sum', margins=True, margins_name='Total', normalize=False).fillna(0).round().astype(int).apply(lambda col: col.map(lambda x: f'{x:,.0f}'.replace(',', '.')))
    totals_porc = (pd.crosstab(values=df.factor_expansion_linea, columns=df.Genero, index=df.Tarifa, aggfunc='sum', margins=True, margins_name='Total', normalize=True) * 100).round(2)
  
    modos = socio_indicadores[socio_indicadores.tabla=='etapas-genero-modo'].copy()
    modos_genero_abs = pd.crosstab(values=modos.factor_expansion_linea, index=[modos.Genero], columns=modos.Modo, aggfunc='sum', normalize=False, margins=True, margins_name='Total').fillna(0).astype(int).apply(lambda col: col.map(lambda x: f'{x:,.0f}'.replace(',', '.')))
    modos_genero_porc = (pd.crosstab(values=modos.factor_expansion_linea, index=modos.Genero, columns=modos.Modo, aggfunc='sum', normalize=True, margins=True, margins_name='Total') * 100).round(2)
    
    modos = socio_indicadores[socio_indicadores.tabla=='etapas-tarifa-modo'].copy()
    modos_tarifa_abs = pd.crosstab(values=modos.factor_expansion_linea, index=[modos.Tarifa], columns=modos.Modo, aggfunc='sum', normalize=False, margins=True, margins_name='Total').fillna(0).astype(int).apply(lambda col: col.map(lambda x: f'{x:,.0f}'.replace(',', '.')))
    modos_tarifa_porc = (pd.crosstab(values=modos.factor_expansion_linea, index=modos.Tarifa, columns=modos.Modo, aggfunc='sum', normalize=True, margins=True, margins_name='Total') * 100).round(2)

    avg_distances = pd.crosstab(values=df.Distancia, columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(2)
    avg_times = pd.crosstab(values=df['Tiempo de viaje'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(2)
    avg_velocity = pd.crosstab(values=df['Velocidad'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(2)
    avg_etapas = pd.crosstab(values=df['Etapas promedio'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).round(2).fillna('')
    user = socio_indicadores[socio_indicadores.tabla=='usuario-genero-tarifa'].copy()
    avg_viajes = pd.crosstab(values=user['Viajes promedio'], index=[user.Tarifa], columns=user.Genero, margins=True, margins_name='Total', aggfunc=lambda x: (x * user.loc[x.index, 'factor_expansion_linea']).sum() / user.loc[x.index, 'factor_expansion_linea'].sum(),).round(2).fillna('') 

    avg_tiempo_entre_viajes = pd.crosstab(values=df['Tiempo entre viajes'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(2)
    
    return totals, totals_porc, avg_distances, avg_times, avg_velocity, modos_genero_abs, modos_genero_porc, modos_tarifa_abs, modos_tarifa_porc, avg_viajes, avg_etapas, avg_tiempo_entre_viajes

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

          
    # st.session_state.etapas = traigo()
    
    if len(etapas_lst_) > 0:

        zonificaciones = levanto_tabla_sql('zonificaciones')
        socio_indicadores = levanto_tabla_sql('socio_indicadores')
        desc_tipo_dia_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT tipo_dia FROM agg_etapas;')
        desc_zona_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT zona FROM agg_etapas;')
        modos_list_all_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT modo_agregado FROM agg_etapas;')
        rango_hora_all_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT rango_hora FROM agg_etapas;')
        distancia_all_ = levanto_tabla_sql('agg_etapas', 'dash', 'SELECT DISTINCT distancia FROM agg_etapas;')
        

        # st.session_state.etapas_all = st.session_state.etapas_all[st.session_state.etapas_all.factor_expansion_linea > 0].copy()
        general, modal, distancias = traigo_indicadores('all')
        
        # Inicializar valores de `st.session_state` solo si no existen
        if 'last_filters' not in st.session_state:
            st.session_state.last_filters = {
                'mes': 'Todos',
                'tipo_dia': None,
                'zona': None,
                'transferencia': 'Todos',
                'modo_agregado': 'Todos',
                'rango_hora': 'Todos',
                'distancia': 'Todas'
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
        # rango_hora = col1.selectbox('Rango hora', options=[text.capitalize() for text in rango_hora_all])
        rango_hora = col1.selectbox('Rango hora', options=[text for text in rango_hora_all])
        
        distancia_all = ['Todas'] + distancia_all_[distancia_all_.distancia != '99'].distancia.unique().tolist()
        distancia = col1.selectbox('Distancia', options=distancia_all)

        desc_etapas = col1.checkbox(
            'Etapas', value=False)

        desc_viajes = col1.checkbox(
            'Viajes', value=True)

        desc_origenes = col1.checkbox(
            ':blue[Origenes]', value=False)

        desc_destinos = col1.checkbox(
            ':orange[Destinos]', value=False)

        desc_zonif = col1.checkbox(
            'Mostrar zonificación', value=True)
        if desc_zonif:
            zonif = zonificaciones[zonificaciones.zona == desc_zona]
        else:
            zonif = ''
        
        # Construye el diccionario de filtros actual
        current_filters = {
            'mes': None if desc_mes == 'Todos' else desc_mes,
            'tipo_dia': desc_tipo_dia,
            'zona': None if desc_zona == 'Todos' else desc_zona,
            'transferencia': None if transf_list == 'Todos' else (1 if transf_list == 'Con transferencia' else 0),
            'modo_agregado': None if modos_list == 'Todos' else modos_list,
            'rango_hora': None if rango_hora == 'Todos' else rango_hora,
            'distancia': None if distancia == 'Todas' else distancia,
        }
        current_options = { 'desc_etapas': desc_etapas,
                            'desc_viajes': desc_viajes,
                            'desc_origenes': desc_origenes, 
                            'desc_destinos': desc_destinos,
                            'desc_zonif': desc_zonif, }
        

        
        # Solo cargar datos si hay cambios en los filtros
        if hay_cambios_en_filtros(current_filters, st.session_state.last_filters):
            
            query = ""
            conditions = " AND ".join(f"{key} = '{value}'" for key, value in current_filters.items() if value is not None)
            if conditions:
                query += f" WHERE {conditions}"

            st.session_state.matrices_ = levanto_tabla_sql_local('agg_matrices', tabla_tipo='dash', query=f"SELECT * FROM agg_matrices{query}")    
            st.session_state.etapas_ = levanto_tabla_sql_local('agg_etapas', tabla_tipo='dash', query=f"SELECT * FROM agg_etapas{query}")
            


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
    
                
                st.session_state.general_ = general.loc[general.mes==desc_mes, ['Tipo', 'Indicador', 'Valor']].set_index('Tipo')
                st.session_state.modal_ = modal.loc[modal.mes==desc_mes, ['Tipo', 'Indicador', 'Valor']].set_index('Tipo')
                st.session_state.distancias_ = distancias.loc[distancias.mes==desc_mes, ['Tipo', 'Indicador', 'Valor']].set_index('Tipo')
        
                if transf_list == 'Todos':
                    st.session_state.desc_transfers = True
                else:
                    st.session_state.desc_transfers = False
                    # if transf_list == 'Con transferencia':
                    #     st.session_state.etapas_ = st.session_state.etapas_[(st.session_state.etapas_.transferencia == 1)]
                    #     st.session_state.matrices_ = st.session_state.matrices_[(st.session_state.matrices_.transferencia == 1)]
                    # elif transf_list == 'Sin transferencia':
                    #     st.session_state.etapas_ = st.session_state.etapas_[(st.session_state.etapas_.transferencia == 0)]
                    #     st.session_state.matrices_ = st.session_state.matrices_[(st.session_state.matrices_.transferencia == 0)]
                    # else:
                    #     st.session_state.etapas_ = pd.DataFrame([])
                    #     st.session_state.matrices_ = pd.DataFrame([])
        
                if modos_list == 'Todos':
                    st.session_state.desc_modos = True
                else:
                    st.session_state.desc_modos = False
                    # st.session_state.etapas_ = st.session_state.etapas_[
                    #     (st.session_state.etapas_.modo_agregado.str.lower() == modos_list.lower())]
                    # st.session_state.matrices_ = st.session_state.matrices_[
                    #     (st.session_state.matrices_.modo_agregado.str.lower() == modos_list.lower())]
        
                if rango_hora == 'Todos':
                    st.session_state.desc_horas = True
                else:
                    st.session_state.desc_horas = False
                    # st.session_state.etapas_ = st.session_state.etapas_[(st.session_state.etapas_.rango_hora == rango_hora)]
                    # st.session_state.matrices_ = st.session_state.matrices_[(st.session_state.matrices_.rango_hora == rango_hora)]
        
                if distancia == 'Todas':
                    st.session_state.desc_distancia = True
                else:
                    st.session_state.desc_distancia = False
                    # st.session_state.etapas_ = st.session_state.etapas_[(st.session_state.etapas_.distancia == distancia)]
                    # st.session_state.matrices_ = st.session_state.matrices_[(st.session_state.matrices_.distancia == distancia)]
        
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
                
                st.session_state.etapas, st.session_state.viajes, st.session_state.matriz, st.session_state.origenes, st.session_state.destinos = create_data_folium(st.session_state.etapas_,
                                                                                                                                                                    st.session_state.matrices_,
                                                                                                                                                                    agg_transferencias=st.session_state.desc_transfers,
                                                                                                                                                                    agg_modo=st.session_state.desc_modos,
                                                                                                                                                                    agg_hora=st.session_state.desc_horas,
                                                                                                                                                                    agg_distancia=st.session_state.desc_distancia,
                                                                                                                                                                    agg_cols_etapas=st.session_state.agg_cols_etapas,
                                                                                                                                                                    agg_cols_viajes=st.session_state.agg_cols_viajes)
    
                st.session_state.etapas = st.session_state.etapas[st.session_state.etapas.inicio_norm != st.session_state.etapas.fin_norm].copy()
                st.session_state.viajes = st.session_state.viajes[st.session_state.viajes.inicio_norm != st.session_state.viajes.fin_norm].copy()
        
                if not desc_etapas:
                    st.session_state.etapas = pd.DataFrame([])
        
                if not desc_viajes:
                    st.session_state.viajes = pd.DataFrame([])
        
                if not desc_origenes:
                    st.session_state.origenes = pd.DataFrame([])
        
                if not desc_destinos:
                    st.session_state.destinos = pd.DataFrame([])
            
                if (len(st.session_state.etapas) > 0) | (len(st.session_state.viajes) > 0) | (len(st.session_state.origenes) > 0) | (len(st.session_state.destinos) > 0):
                    
                    m = crear_mapa_lineas_deseo(df_viajes=st.session_state.viajes,
                                                  df_etapas=st.session_state.etapas,
                                                  zonif=zonif,
                                                  origenes=st.session_state.origenes,
                                                  destinos=st.session_state.destinos,
                                                  var_fex='factor_expansion_linea',
                                                  cmap_viajes='Blues',
                                                  cmap_etapas='Greens',
                                                  map_title='Líneas de Deseo',
                                                  savefile='',
                                                  k_jenks=5)
                    if m:
                        st.session_state.map = m
                    
                    if st.session_state.map:
                        with col2:
                            folium_static(st.session_state.map, width=1000, height=800)
                    else:
                        col2.text("No hay datos suficientes para mostrar el mapa.")
    
    


with st.expander('Indicadores'):
    col1, col2, col3 = st.columns([2, 2, 2])

    if len(st.session_state.etapas_) > 0:
        col1.table(st.session_state.general_)
        col2.table(st.session_state.modal_)
        col3.table(st.session_state.distancias_)

with st.expander('Matrices'):

    col1, col2 = st.columns([1, 4])
    if len(st.session_state.matriz) > 0:

        # col2.table(st.session_state.matriz)
    
        tipo_matriz = col1.selectbox(
                'Variable', options=['Viajes', 'Distancia promedio (kms)', 'Tiempo promedio (min)', 'Velocidad promedio (km/h)'])
            
        normalize = False
        if tipo_matriz == 'Viajes':
            var_matriz = 'factor_expansion_linea'
            normalize = col1.checkbox('Normalizar', value=True)
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

with st.expander('Género y tarifas'):
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    totals, totals_porc, avg_distances, avg_times, avg_velocity, modos_genero_abs, modos_genero_porc, modos_tarifa_abs, modos_tarifa_porc, avg_viajes, avg_etapas, avg_tiempo_entre_viajes = traigo_socio_indicadores(st.session_state.socio_indicadores_)

    col2.markdown("<h4 style='font-size:16px;'>Total de viajes por género y tarifa</h4>", unsafe_allow_html=True)
    col2.table(totals)
    col3.markdown("<h4 style='font-size:16px;'>Porcentaje de viajes por género y tarifa</h4>", unsafe_allow_html=True)
    col3.table(totals_porc.round(2).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Cantidad promedio de viajes por género y tarifa</h4>", unsafe_allow_html=True)
    col2.table(avg_viajes.round(2).astype(str))
    col3.markdown("<h4 style='font-size:16px;'>Cantidad promedio de etapas por género y tarifa</h4>", unsafe_allow_html=True)
    col3.table(avg_etapas.round(2).astype(str))

    
    col2.markdown("<h4 style='font-size:16px;'>Total de etapas por género y modo</h4>", unsafe_allow_html=True)
    col2.table(modos_genero_abs)
    col3.markdown("<h4 style='font-size:16px;'>Porcentaje de etapas por género y modo</h4>", unsafe_allow_html=True)
    col3.table(modos_genero_porc.round(2).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Total de etapas por tarifa y modo</h4>", unsafe_allow_html=True)
    col2.table(modos_tarifa_abs)
    col3.markdown("<h4 style='font-size:16px;'>Porcentaje de etapas por tarifa y modo</h4>", unsafe_allow_html=True)
    col3.table(modos_tarifa_porc.round(2).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Distancias promedio (kms)</h4>", unsafe_allow_html=True)
    col2.table(avg_distances.round(2).astype(str))

    col3.markdown("<h4 style='font-size:16px;'>Tiempos promedio (minutos)</h4>", unsafe_allow_html=True)
    col3.table(avg_times.round(2).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Velocidades promedio (kms/hora)</h4>", unsafe_allow_html=True)
    col2.table(avg_velocity.round(2).astype(str))

    col3.markdown("<h4 style='font-size:16px;'>Tiempos promedio entre viajes (minutos)</h4>", unsafe_allow_html=True)
    col3.table(avg_tiempo_entre_viajes.round(2).astype(str))


