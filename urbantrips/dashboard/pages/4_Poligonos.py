import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import mapclassify
import plotly.express as px
from folium import Figure
from dash_utils import (
    levanto_tabla_sql, get_logo,
    create_data_folium, traigo_indicadores,
    extract_hex_colors_from_cmap
)


def crear_mapa_poligonos(df_viajes,
                         df_etapas,
                         poly,
                         zonif,
                         origenes,
                         destinos,
                         var_fex,
                         cmap_viajes='Blues',
                         cmap_etapas='Greens',
                         map_title='',
                         savefile='',
                         k_jenks=5,
                         show_poly=False):

    m = ''
    if (len(df_viajes) > 0) | (len(df_etapas) > 0) | (len(origenes) > 0) | (len(destinos) > 0):

        fig = Figure(width=800, height=600)
        m = folium.Map(location=[poly.geometry.representative_point().y.mean(
        ), poly.geometry.representative_point().x.mean()], zoom_start=10, tiles='cartodbpositron')

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
                    # color=colors_origenes[n],
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
                    # color=colors_destinos[n],
                    color="#de8f0599",
                    style_kwds={'fillOpacity': 0.1, 'weight': line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 5

        # Agrego polígono
        if (len(poly) > 0) & (show_poly):
            geojson = poly.to_json()
            # Add the GeoJSON to the map as a GeoJson Layer
            folium.GeoJson(
                geojson,
                name=poly.id.values[0],
                style_function=lambda feature: {
                    'fillColor': 'grey',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': .5,

                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['id'], labels=False, sticky=False)
            ).add_to(m)

        # Agrego polígono
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

        folium.LayerControl(name='xx').add_to(m)

    return m


st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)


poligonos = levanto_tabla_sql('poligonos')
etapas_all = levanto_tabla_sql('poly_etapas')
matrices_all = levanto_tabla_sql('poly_matrices')
zonificaciones = levanto_tabla_sql('zonificaciones')
general_ = ''
with st.expander('Polígonos', expanded=True):

    col1, col2 = st.columns([1, 4])

    if (len(poligonos) > 0) & (len(etapas_all) > 0):

        general, modal, distancias = traigo_indicadores('poligonos')

        # etapas_lst = ['Todos'] + etapas_all.mes.unique().tolist()        
        etapas_lst = etapas_all.mes.unique().tolist()    
        desc_mes = col1.selectbox(
            'Mes', options=etapas_lst)
        
        desc_tipo_dia = col1.selectbox(
            'Tipo día', options=etapas_all.tipo_dia.unique())

        desc_poly = col1.selectbox(
            'Polígono', options=etapas_all.id_polygon.unique())
        desc_zona = col1.selectbox(
            'Zonificación', options=etapas_all.zona.unique())
        zonif = zonificaciones[zonificaciones.zona == desc_zona]

        desc_etapas = col1.checkbox(
            label='Etapas', value=True)

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

        rango_hora_all = [
            'Todos']+etapas_all[etapas_all.rango_hora != '99'].rango_hora.unique().tolist()
        rango_hora = [text.capitalize() for text in rango_hora_all]
        rango_hora = col1.selectbox(
            'Rango hora', options=rango_hora_all)

        distancia_all = [
            'Todas']+etapas_all[etapas_all.distancia != '99'].distancia.unique().tolist()
        distancia = col1.selectbox(
            'Distancia', options=distancia_all)

        etapas_ = etapas_all[(etapas_all.mes==desc_mes)&(etapas_all.tipo_dia==desc_tipo_dia)&
                            (etapas_all.id_polygon == desc_poly) &
                             (etapas_all.zona == desc_zona)].copy()
        matrices_ = matrices_all[(matrices_all.mes==desc_mes)&(matrices_all.tipo_dia==desc_tipo_dia)&
                                (matrices_all.id_polygon == desc_poly) &
                                (matrices_all.zona == desc_zona)].copy()

        general_ = general.loc[ (general.mes==desc_mes)&(general.tipo_dia==desc_tipo_dia)&
                               ( general.id_polygon == desc_poly), [
            'Tipo', 'Indicador', 'Valor']].set_index('Tipo')
        modal_ = modal.loc[ (modal.mes==desc_mes)&(modal.tipo_dia==desc_tipo_dia)&
                            (modal.id_polygon == desc_poly), [
            'Tipo', 'Indicador', 'Valor']].set_index('Tipo')
        distancias_ = distancias.loc[(distancias.mes==desc_mes)&(distancias.tipo_dia==desc_tipo_dia)&
                                    (distancias.id_polygon == desc_poly), [
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

        agg_cols_etapas = ['id_polygon',
                           'zona',
                           'inicio_norm',
                           'transfer1_norm',
                           'transfer2_norm',
                           'fin_norm',
                           'poly_inicio_norm',
                           'poly_transfer1_norm',
                           'poly_transfer2_norm',
                           'poly_fin_norm',
                           'transferencia',
                           'modo_agregado',
                           'rango_hora',
                           'distancia']
        agg_cols_viajes = ['id_polygon',
                           'zona',
                           'inicio_norm',
                           'fin_norm',
                           'poly_inicio_norm',
                           'poly_fin_norm',
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

        show_poly = col1.checkbox(
            'Mostrar polígono', value=True)

        poly = poligonos[(poligonos.id == desc_poly)]

        if not desc_origenes:
            origenes = ''
        if not desc_destinos:
            destinos = ''

        if (len(etapas) > 0) | (len(viajes) > 0) | (len(origenes) > 0) | (len(destinos) > 0):

            map = crear_mapa_poligonos(df_viajes=viajes,
                                       df_etapas=etapas,
                                       poly=poly,
                                       zonif=zonif,
                                       origenes=origenes,
                                       destinos=destinos,
                                       var_fex='factor_expansion_linea',
                                       cmap_viajes='Blues',
                                       cmap_etapas='Greens',
                                       map_title=desc_poly,
                                       savefile='',
                                       k_jenks=5,
                                       show_poly=show_poly)

            with col2:
                # st_map = st_folium(map, width=1200, height=1000) #
                st_folium(map, width=1500, height=1000)

        else:
            matriz = pd.DataFrame([])
            col2.text('No hay datos para mostrar')

    else:
        matriz = pd.DataFrame([])

        col2.text('No hay datos para mostrar')


with st.expander('Indicadores'):
    col1, col2, col3 = st.columns([2, 2, 2])
    if len(general_) > 0:
        col1.table(general_)
        col2.table(modal_)
        col3.table(distancias_)
    else:
        col2.text('No hay datos para mostrar')

with st.expander('Matrices'):

    col1, col2 = st.columns([1, 4])

    if len(matriz) > 0:

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
                values=matriz['factor_expansion_linea'],
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
    
            if len(matriz) <= 20:
                fig.update_layout(width=800, height=800)
            elif (len(matriz) > 20) & (len(od_heatmap) <= 40):
                fig.update_layout(width=1000, height=1000)
            elif len(matriz) > 40:
                fig.update_layout(width=1400, height=800)
    
            col2.plotly_chart(fig)
    else:
        col2.text('No hay datos para mostrar')
