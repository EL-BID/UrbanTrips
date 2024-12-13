import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import mapclassify
import plotly.express as px
from folium import Figure
from dash_utils import (levanto_tabla_sql, get_logo,
                        create_linestring_od, extract_hex_colors_from_cmap)

def traigo_socio_indicadores(socio_indicadores):    
    totals = None
    totals_porc = 0
    avg_distances = 0
    avg_times = 0
    avg_velocity = 0
    modos_genero_abs = 0
    modos_genero_porc = 0
    modos_tarifa_abs = 0
    modos_tarifa_porc = 0
    avg_viajes = 0
    avg_etapas = 0
    avg_tiempo_entre_viajes = 0
    
    if len(socio_indicadores) > 0:

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
        avg_viajes = pd.crosstab(values=user['Viajes promedio'], 
                                 index=[user.Tarifa], 
                                 columns=user.Genero, 
                                 margins=True, 
                                 margins_name='Total', 
                                     aggfunc=lambda x: (x * user.loc[x.index, 'factor_expansion_linea']).sum() / user.loc[x.index, 'factor_expansion_linea'].sum(),).round(2).fillna('') 
    
        avg_tiempo_entre_viajes = pd.crosstab(values=df['Tiempo entre viajes'], 
                                              columns=df.Genero, 
                                              index=df.Tarifa, 
                                              margins=True, 
                                              margins_name='Total',
                                              aggfunc=lambda x: (x * df.loc[x.index, 'factor_expansion_linea']).sum() / df.loc[x.index, 'factor_expansion_linea'].sum(), ).fillna(0).round(2)
    
    return totals, totals_porc, avg_distances, avg_times, avg_velocity, modos_genero_abs, modos_genero_porc, modos_tarifa_abs, modos_tarifa_porc, avg_viajes, avg_etapas, avg_tiempo_entre_viajes
    
def crear_mapa_folium(df_agg,
                      cmap,
                      var_fex,
                      savefile='',
                      k_jenks=5):

    bins = [df_agg[var_fex].min()-1] + \
        mapclassify.FisherJenks(df_agg[var_fex], k=k_jenks).bins.tolist()
    range_bins = range(0, len(bins)-1)
    bins_labels = [
        f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins]
    df_agg['cuts'] = pd.cut(df_agg[var_fex], bins=bins, labels=bins_labels)

    fig = Figure(width=800, height=800)
    m = folium.Map(location=[df_agg.lat_o.mean(
    ), df_agg.lon_o.mean()], zoom_start=9, tiles='cartodbpositron')

    title_html = """
    <h3 align="center" style="font-size:20px"><b>Your map title</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    line_w = 0.5

    colors = extract_hex_colors_from_cmap(cmap=cmap, n=k_jenks)

    n = 0
    for i in bins_labels:

        df_agg[df_agg.cuts == i].explore(
            m=m,
            color=colors[n],
            style_kwds={'fillOpacity': 0.1, 'weight': line_w},
            name=i,
            tooltip=False,
        )
        n += 1
        line_w += 3

    folium.LayerControl(name='xx').add_to(m)

    fig.add_child(m)

    return fig


st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)


with st.expander('Partición modal'):

    col1, col2, col3 = st.columns([1, 3, 3])
    particion_modal = levanto_tabla_sql('particion_modal')

    desc_dia_m = col1.selectbox(
        'Periodo', options=particion_modal.desc_dia.unique(), key='desc_dia_m')
    tipo_dia_m = col1.selectbox(
        'Tipo de día', options=particion_modal.tipo_dia.unique(), key='tipo_dia_m')

    # Etapas
    particion_modal_etapas = particion_modal[(particion_modal.desc_dia == desc_dia_m) & (
        particion_modal.tipo_dia == tipo_dia_m) & (particion_modal.tipo == 'etapas')]
    if col2.checkbox('Ver datos: etapas'):
        col2.write(particion_modal_etapas)
    fig2 = px.bar(particion_modal_etapas, x='modo', y='modal')
    fig2.update_layout(title_text='Partición modal de Etapas')
    fig2.update_xaxes(title_text='Modo')
    fig2.update_yaxes(title_text='Partición modal (%)')
    fig2.update_traces(marker_color='brown')
    col2.plotly_chart(fig2)

    # Viajes
    particion_modal_viajes = particion_modal[(particion_modal.desc_dia == desc_dia_m) & (
        particion_modal.tipo_dia == tipo_dia_m) & (particion_modal.tipo == 'viajes')]
    if col3.checkbox('Ver datos: viajes'):
        col3.write(particion_modal_viajes)
    fig = px.bar(particion_modal_viajes, x='modo', y='modal')
    fig.update_layout(title_text='Partición modal de Viajes')
    fig.update_xaxes(title_text='Modo')
    fig.update_yaxes(title_text='Partición modal (%)')
    fig.update_traces(marker_color='navy')
    col3.plotly_chart(fig)


with st.expander('Distancias de viajes'):

    col1, col2 = st.columns([1, 4])

    hist_values = levanto_tabla_sql('distribucion')

    if len(hist_values) > 0:
        hist_values.columns = ['desc_dia', 'tipo_dia',
                               'Distancia (kms)', 'Viajes', 'Modo']
        hist_values = hist_values[hist_values['Distancia (kms)'] <= 60]
        hist_values = hist_values.sort_values(['Modo', 'Distancia (kms)'])

        if col2.checkbox('Ver datos: distribución de viajes'):
            col2.write(hist_values)

        desc_dia_d = col1.selectbox(
            'Periodo', options=hist_values.desc_dia.unique(), key='desc_dia_d')
        tipo_dia_d = col1.selectbox(
            'Tipo de dia', options=hist_values.tipo_dia.unique(), key='tipo_dia_d')

        dist = hist_values.Modo.unique().tolist()
        dist.remove('Todos')
        dist = ['Todos'] + dist
        modo_d = col1.selectbox('Modo', options=dist)

        hist_values = hist_values[(hist_values.desc_dia == desc_dia_d) & (
            hist_values.tipo_dia == tipo_dia_d) & (hist_values.Modo == modo_d)]

        fig = px.histogram(hist_values, x='Distancia (kms)',
                           y='Viajes', nbins=len(hist_values))
        fig.update_xaxes(type='category')
        fig.update_yaxes(title_text='Viajes')

        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tickangle=0,
                tickfont=dict(size=9)
            ),
            yaxis=dict(
                tickfont=dict(size=9)
            )
        )

        col2.plotly_chart(fig)
    else:
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


with st.expander('Viajes por hora'):

    col1, col2 = st.columns([1, 4])

    viajes_hora = levanto_tabla_sql('viajes_hora')

    desc_dia_h = col1.selectbox(
        'Periodo', options=viajes_hora.desc_dia.unique(), key='desc_dia_h')
    tipo_dia_h = col1.selectbox(
        'Tipo de dia', options=viajes_hora.tipo_dia.unique(), key='tipo_dia_h')
    modo_h = col1.selectbox(
        'Modo', options=['Todos', 'Por modos'], key='modo_h')

    if modo_h == 'Todos':
        viajes_hora = viajes_hora[(viajes_hora.desc_dia == desc_dia_h) & (
            viajes_hora.tipo_dia == tipo_dia_h) & (viajes_hora.Modo == 'Todos')]
    else:
        viajes_hora = viajes_hora[(viajes_hora.desc_dia == desc_dia_h) & (
            viajes_hora.tipo_dia == tipo_dia_h) & (viajes_hora.Modo != 'Todos')]

    viajes_hora = viajes_hora.sort_values('Hora')
    if col2.checkbox('Ver datos: viajes por hora'):
        col2.write(viajes_hora)

    fig_horas = px.line(viajes_hora, x="Hora", y="Viajes",
                        color='Modo', symbol="Modo")

    fig_horas.update_xaxes(type='category')
    # fig_horas.update_layout()

    col2.plotly_chart(fig_horas)


with st.expander('Género y tarifas'):
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    totals, totals_porc, avg_distances, avg_times, avg_velocity, modos_genero_abs, modos_genero_porc, modos_tarifa_abs, modos_tarifa_porc, avg_viajes, avg_etapas, avg_tiempo_entre_viajes = traigo_socio_indicadores(st.session_state.socio_indicadores_)


    if totals is not None:
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
    else:
        col2.write('No hay datos para mostrar')
