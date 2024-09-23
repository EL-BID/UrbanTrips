import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import mapclassify
import plotly.express as px
from folium import Figure
from dash_utils import (
    levanto_tabla_sql, get_logo,
    create_data_folium, traigo_indicadores,
    extract_hex_colors_from_cmap
)


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
            y_val = etapas.geometry.representative_point().y.mean()
            x_val = etapas.geometry.representative_point().x.mean()
        elif len(df_viajes) > 0:
            y_val = viajes.geometry.representative_point().y.mean()
            x_val = viajes.geometry.representative_point().x.mean()
        elif len(origenes) > 0:
            y_val = origenes.geometry.representative_point().y.mean()
            x_val = origenes.geometry.representative_point().x.mean()
        elif len(destinos) > 0:
            y_val = destinos.geometry.representative_point().y.mean()
            x_val = destinos.geometry.representative_point().x.mean()

        fig = Figure(width=2000, height=2000)
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

def traigo_socio_indicadores(socio_indicadores):    

    df = socio_indicadores[socio_indicadores.tabla=='viajes-genero-tarifa'].copy()
    totals = pd.crosstab(values=df.factor_expansion_linea, columns=df.Genero, index=df.Tarifa, aggfunc='sum', margins=True, margins_name='Total', normalize=False).fillna(0).round().astype(int).apply(lambda col: col.map(lambda x: f'{x:,.0f}'.replace(',', '.')))
    totals_porc = (pd.crosstab(values=df.factor_expansion_linea, columns=df.Genero, index=df.Tarifa, aggfunc='sum', margins=True, margins_name='Total', normalize=True) * 100).round(1)
  
    modos = socio_indicadores[socio_indicadores.tabla=='etapas-genero-modo'].copy()
    modos_genero_abs = pd.crosstab(values=modos.factor_expansion_linea, index=[modos.Genero], columns=modos.Modo, aggfunc='sum', normalize=False, margins=True, margins_name='Total').fillna(0).astype(int).apply(lambda col: col.map(lambda x: f'{x:,.0f}'.replace(',', '.')))
    modos_genero_porc = (pd.crosstab(values=modos.factor_expansion_linea, index=modos.Genero, columns=modos.Modo, aggfunc='sum', normalize=True, margins=True, margins_name='Total') * 100).round(1)
    
    modos = socio_indicadores[socio_indicadores.tabla=='etapas-tarifa-modo'].copy()
    modos_tarifa_abs = pd.crosstab(values=modos.factor_expansion_linea, index=[modos.Tarifa], columns=modos.Modo, aggfunc='sum', normalize=False, margins=True, margins_name='Total').fillna(0).astype(int).apply(lambda col: col.map(lambda x: f'{x:,.0f}'.replace(',', '.')))
    modos_tarifa_porc = (pd.crosstab(values=modos.factor_expansion_linea, index=modos.Tarifa, columns=modos.Modo, aggfunc='sum', normalize=True, margins=True, margins_name='Total') * 100).round(1)

    avg_distances = pd.crosstab(values=df.Distancia, columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(1)
    avg_times = pd.crosstab(values=df['Tiempo de viaje'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(1)
    avg_velocity = pd.crosstab(values=df['Velocidad'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(1)
    avg_etapas = pd.crosstab(values=df['Etapas promedio'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).round(1).fillna('')
    user = socio_indicadores[socio_indicadores.tabla=='usuario-genero-tarifa'].copy()
    avg_viajes = pd.crosstab(values=user['Viajes promedio'], index=[user.Tarifa], columns=user.Genero, margins=True, margins_name='Total', aggfunc=lambda x: (x * user.loc[x.index, 'factor_expansion_linea']).sum() / user.loc[x.index, 'factor_expansion_linea'].sum(),).round(1).fillna('') 

    avg_tiempo_entre_viajes = pd.crosstab(values=df['Tiempo entre viajes'], columns=df.Genero, index=df.Tarifa, margins=True, margins_name='Total',aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(1)
    
    return totals, totals_porc, avg_distances, avg_times, avg_velocity, modos_genero_abs, modos_genero_porc, modos_tarifa_abs, modos_tarifa_porc, avg_viajes, avg_etapas, avg_tiempo_entre_viajes


st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)


with st.expander('Líneas de Deseo', expanded=True):

    col1, col2 = st.columns([1, 4])

    etapas_all = levanto_tabla_sql('agg_etapas')
    matrices_all = levanto_tabla_sql('agg_matrices')
    zonificaciones = levanto_tabla_sql('zonificaciones')
    socio_indicadores = levanto_tabla_sql('socio_indicadores')

    if len(etapas_all) > 0:
        etapas_all = etapas_all[etapas_all.factor_expansion_linea > 0].copy()
        general, modal, distancias = traigo_indicadores('all')

        
        etapas_lst = ['Todos'] + etapas_all.mes.unique().tolist()        
        desc_mes = col1.selectbox(
            'Mes', options=etapas_lst)
        
        desc_tipo_dia = col1.selectbox(
            'Tipo día', options=etapas_all.tipo_dia.unique())

        desc_zona = col1.selectbox(
            'Zonificación', options=etapas_all.zona.unique())
        zonif = zonificaciones[zonificaciones.zona == desc_zona]

        desc_etapas = col1.checkbox(
            'Etapas', value=True)

        desc_viajes = col1.checkbox(
            'Viajes', value=False)

        desc_origenes = col1.checkbox(
            ':blue[Origenes]', value=False)

        desc_destinos = col1.checkbox(
            ':orange[Destinos]', value=False)

        transf_list_all = ['Todos', 'Con transferencia', 'Sin transferencia']
        transf_list = col1.selectbox(
            'Transferencias', options=transf_list_all)

        modos_list_all = ['Todos']+etapas_all[etapas_all.modo_agregado !=
                                              '99'].modo_agregado.unique().tolist()
        modos_list = [text.capitalize() for text in modos_list_all]
        modos_list = col1.selectbox(
            'Modos', options=modos_list_all)

        rango_hora_all = ['Todos']+etapas_all[etapas_all.rango_hora !=
                                              '99'].rango_hora.unique().tolist()
        rango_hora = [text.capitalize() for text in rango_hora_all]
        rango_hora = col1.selectbox(
            'Rango hora', options=rango_hora_all)

        distancia_all = ['Todas']+etapas_all[etapas_all.distancia !=
                                             '99'].distancia.unique().tolist()
        distancia = col1.selectbox(
            'Distancia', options=distancia_all)

        if desc_mes != 'Todos':            
            etapas_ = etapas_all[(etapas_all.zona == desc_zona)&(etapas_all.mes==desc_mes)&(etapas_all.tipo_dia==desc_tipo_dia)].copy()
            matrices_ = matrices_all[(matrices_all.zona == desc_zona)&(matrices_all.mes==desc_mes)&(matrices_all.tipo_dia==desc_tipo_dia)].copy()
            socio_indicadores_ = socio_indicadores[(socio_indicadores.mes==desc_mes)&(socio_indicadores.tipo_dia==desc_tipo_dia)].copy()

        else:
            etapas_ = etapas_all[(etapas_all.zona == desc_zona)&(etapas_all.tipo_dia==desc_tipo_dia)].copy()
            matrices_ = matrices_all[(matrices_all.zona == desc_zona)&(matrices_all.tipo_dia==desc_tipo_dia)].copy()
            socio_indicadores_ = socio_indicadores[(socio_indicadores.tipo_dia==desc_tipo_dia)].copy()

            etapas_['mes'] = 'Todos'
            matrices_['mes'] = 'Todos'
            socio_indicadores_['mes'] = 'Todos'

        
        
        general_ = general[['Tipo', 'Indicador', 'Valor']].set_index('Tipo')
        modal_ = modal[['Tipo', 'Indicador', 'Valor']].set_index('Tipo')
        distancias_ = distancias[[
            'Tipo', 'Indicador', 'Valor']].set_index('Tipo')

        if transf_list == 'Todos':
            desc_transfers = True
        else:
            desc_transfers = False
            if transf_list == 'Con transferencia':
                etapas_ = etapas_[(etapas_.transferencia == 1)]
                matrices_ = matrices_[(matrices_.transferencia == 1)]
            elif transf_list == 'Sin transferencia':
                etapas_ = etapas_[(etapas_.transferencia == 0)]
                matrices_ = matrices_[(matrices_.transferencia == 0)]
            else:
                etapas_ = pd.DataFrame([])
                matrices_ = pd.DataFrame([])

        if modos_list == 'Todos':
            desc_modos = True
        else:
            desc_modos = False
            etapas_ = etapas_[
                (etapas_.modo_agregado.str.lower() == modos_list.lower())]
            matrices_ = matrices_[
                (matrices_.modo_agregado.str.lower() == modos_list.lower())]

        if rango_hora == 'Todos':
            desc_horas = True
        else:
            desc_horas = False
            etapas_ = etapas_[(etapas_.rango_hora == rango_hora)]
            matrices_ = matrices_[(matrices_.rango_hora == rango_hora)]

        if distancia == 'Todas':
            desc_distancia = True
        else:
            desc_distancia = False
            etapas_ = etapas_[(etapas_.distancia == distancia)]
            matrices_ = matrices_[(matrices_.distancia == distancia)]

        agg_cols_etapas = ['zona',
                           'inicio_norm',
                           'transfer1_norm',
                           'transfer2_norm',
                           'fin_norm',
                           'transferencia',
                           'modo_agregado',
                           'rango_hora',
                           'distancia']
        agg_cols_viajes = ['zona',
                           'inicio_norm',
                           'fin_norm',
                           'transferencia',
                           'modo_agregado',
                           'rango_hora',
                           'distancia']

        etapas, viajes, matriz, origenes, destinos = create_data_folium(etapas_,
                                                                        matrices_,
                                                                        agg_transferencias=desc_transfers,
                                                                        agg_modo=desc_modos,
                                                                        agg_hora=desc_horas,
                                                                        agg_distancia=desc_distancia,
                                                                        agg_cols_etapas=agg_cols_etapas,
                                                                        agg_cols_viajes=agg_cols_viajes)

        etapas = etapas[etapas.inicio_norm != etapas.fin_norm].copy()
        viajes = viajes[viajes.inicio_norm != viajes.fin_norm].copy()

        if not desc_etapas:
            etapas = pd.DataFrame([])

        if not desc_viajes:
            viajes = pd.DataFrame([])

        if not desc_origenes:
            origenes = pd.DataFrame([])

        if not desc_destinos:
            destinos = pd.DataFrame([])

        desc_zonif = col1.checkbox(
            'Mostrar zonificación', value=False)
        if not desc_zonif:
            zonif = ''

        if not desc_origenes:
            origenes = ''
        if not desc_destinos:
            destinos = ''

        if (len(etapas) > 0) | (len(viajes) > 0) | (len(origenes) > 0) | (len(destinos) > 0):

            map = crear_mapa_lineas_deseo(df_viajes=viajes,
                                          df_etapas=etapas,
                                          zonif=zonif,
                                          origenes=origenes,
                                          destinos=destinos,
                                          var_fex='factor_expansion_linea',
                                          cmap_viajes='Blues',
                                          cmap_etapas='Greens',
                                          map_title='Líneas de Deseo',
                                          savefile='',
                                          k_jenks=5)

            with col2:
                # st_map = st_folium(map, width=1200, height=1000) #
                folium_static(map, width=1200, height=600)

        else:
            matriz = pd.DataFrame([])
            # Usar HTML para personalizar el estilo del texto
            texto_html = """
                <style>
                .big-font {
                    font-size:30px !important;
                    font-weight:bold;
                }
                </style>
                <div class='big-font'>
                    No hay datos para mostrar            
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

    else:
        matriz = pd.DataFrame([])
        # Usar HTML para personalizar el estilo del texto
        texto_html = """
            <style>
            .big-font {
                font-size:30px !important;
                font-weight:bold;
            }
            </style>
            <div class='big-font'>
                No hay datos para mostrar            
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

    if len(etapas_all) > 0:
        col1.table(general_)
        col2.table(modal_)
        col3.table(distancias_)

with st.expander('Matrices'):

    col1, col2 = st.columns([1, 4])

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

    if len(matriz) > 0:
        od_heatmap = pd.crosstab(
            index=matriz['Origen'],
            columns=matriz['Destino'],
            values=matriz[var_matriz],
            aggfunc="sum",
            normalize=normalize,
        )
        
        if normalize:
            od_heatmap = (od_heatmap * 100).round(1)
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
            fig.update_layout(width=800, height=800)
        elif (len(od_heatmap) > 20) & (len(od_heatmap) <= 40):
            fig.update_layout(width=1100, height=1100)
        elif len(od_heatmap) > 40:
            fig.update_layout(width=1400, height=1400)

        col2.plotly_chart(fig)

with st.expander('Género y tarifas'):
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    totals, totals_porc, avg_distances, avg_times, avg_velocity, modos_genero_abs, modos_genero_porc, modos_tarifa_abs, modos_tarifa_porc, avg_viajes, avg_etapas, avg_tiempo_entre_viajes = traigo_socio_indicadores(socio_indicadores_)

    col2.markdown("<h4 style='font-size:16px;'>Total de viajes por género y tarifa</h4>", unsafe_allow_html=True)
    col2.table(totals)
    col3.markdown("<h4 style='font-size:16px;'>Porcentaje de viajes por género y tarifa</h4>", unsafe_allow_html=True)
    col3.table(totals_porc.round(1).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Cantidad promedio de viajes por género y tarifa</h4>", unsafe_allow_html=True)
    col2.table(avg_viajes.round(1).astype(str))
    col3.markdown("<h4 style='font-size:16px;'>Cantidad promedio de transferencias por género y tarifa</h4>", unsafe_allow_html=True)
    col3.table(avg_etapas.round(1).astype(str))

    
    col2.markdown("<h4 style='font-size:16px;'>Etapas por género y modo</h4>", unsafe_allow_html=True)
    col2.table(modos_genero_abs)
    col3.markdown("<h4 style='font-size:16px;'>Porcentaje de etapas por género y modo</h4>", unsafe_allow_html=True)
    col3.table(modos_genero_porc.round(1).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Etapas por tarifa y modo</h4>", unsafe_allow_html=True)
    col2.table(modos_tarifa_abs)
    col3.markdown("<h4 style='font-size:16px;'>Porcentaje de etapas por tarifa y modo</h4>", unsafe_allow_html=True)
    col3.table(modos_tarifa_porc.round(1).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Distancias promedio (kms)</h4>", unsafe_allow_html=True)
    col2.table(avg_distances.round(1).astype(str))

    col3.markdown("<h4 style='font-size:16px;'>Tiempos promedio (minutos)</h4>", unsafe_allow_html=True)
    col3.table(avg_times.round(1).astype(str))

    col2.markdown("<h4 style='font-size:16px;'>Velocidades promedio (kms/hora)</h4>", unsafe_allow_html=True)
    col2.table(avg_velocity.round(1).astype(str))

    col3.markdown("<h4 style='font-size:16px;'>Tiempos promedio entre viajes (minutos)</h4>", unsafe_allow_html=True)
    col3.table(avg_tiempo_entre_viajes.round(1).astype(str))


