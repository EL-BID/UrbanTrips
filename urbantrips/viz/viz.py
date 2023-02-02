import pandas as pd
import numpy as np
import os
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
import h3
import mapclassify
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from PIL import UnidentifiedImageError
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import contextily as ctx
from IPython.display import display
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.collections import QuadMesh
from pathlib import Path
from matplotlib import colors as mcolors
from matplotlib.text import Text
from mycolorpy import colorlist as mcp
from requests.exceptions import ConnectionError as r_ConnectionError

from urbantrips.geo.geo import (
    normalizo_lat_lon, crear_linestring)
from urbantrips.utils.utils import (
    leer_configs_generales,
    traigo_db_path,
    iniciar_conexion_db,
    leer_alias)


def plotear_recorrido_lowess(id_linea, etapas, recorridos_lowess, alias):
    """
    Esta funcion toma un id_linea, un df de etapas, un gdf de recorridos_lowess
    y un alias y produce una viz de las etapas y el recorrido
    """
    e = etapas.loc[etapas.id_linea == id_linea, :]
    r = recorridos_lowess.loc[recorridos_lowess.id_linea == id_linea, :]

    if (len(e) > 0) & (len(r) > 0):
        try:
            fig, ax = plt.subplots(figsize=(3, 3), dpi=150)

            ax.scatter(e.longitud, e.latitud, color='orange', s=.3)
            r.plot(color='black', lw=.8, legend=False, ax=ax)

            ax.set_title(f'Linea {id_linea}', fontsize=6)
            ax.axis('off')

            db_path = os.path.join("resultados", "png",
                                   f"{alias}linea_{id_linea}.png")

            fig.savefig(db_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        except (AttributeError, ValueError):
            pass

    else:
        print(f"No se pudo producir un grafico para el id_linea {id_linea}")


def visualize_route_section_load(id_linea=False, rango_hrs=False,
                                 indicador='cantidad_etapas', factor=500,
                                 factor_minimo=50):
    """
    Esta funcion toma un id linea y un rango horario
    le de la tabla de ocupacion por linea tramo
    y visualiza la carga por tramo para las lineas
    """

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    if type(id_linea) == int:
        id_linea = [id_linea]

    # si se especifica la linea
    if id_linea:
        lineas_str = ','.join(map(str, id_linea))

    q_rec = f"select * from recorridos"

    q_tabla = """
        select *
        from ocupacion_por_linea_tramo
        """

    if id_linea:
        q_rec = q_rec + f" where id_linea in ({lineas_str})"
        q_tabla = q_tabla + f" where id_linea in ({lineas_str})"
        if rango_hrs:
            q_tabla = q_tabla + \
                f" and hora_min = {rango_hrs[0]} and hora_max = {rango_hrs[1]}"
        else:
            q_tabla = q_tabla + " and hora_min is NULL and hora_max is NULL"

    else:
        q_tabla = q_tabla + " where hora_min is NULL and hora_max is NULL"

    # Leer datos de carga por tramo por linea
    tabla = pd.read_sql(q_tabla, conn_data)
    # Leer recorridos
    recorridos = pd.read_sql(q_rec, conn_insumos)
    recorridos['wkt'] = recorridos.wkt.apply(wkt.loads)

    # Visualizar para cada linea y rango horario
    tabla.groupby('id_linea').apply(
        viz_etapas_x_tramo_recorrido, recorridos, rango_hrs,
        indicador, factor, factor_minimo)


def viz_etapas_x_tramo_recorrido(tabla, recorridos, rango_hrs,
                                 indicador='cantidad_etapas', factor=500,
                                 factor_minimo=50):
    """
    Esta funcion toma un id linea y produce una visualizacion
    con la cantidad de etapas por tramo de recorrido para ambos
    sentidos
    """
    id_linea = tabla.id_linea.unique()[0]
    rec = recorridos.loc[recorridos.id_linea == id_linea, 'wkt'].item()

    print('Produciendo grafico de ocupacion por tramos', id_linea)

    tabla_ida = tabla.loc[tabla.sentido == 'ida', [
        'tramos', 'cantidad_etapas', 'prop_etapas']]
    tabla_vuelta = tabla.loc[tabla.sentido == 'vuelta', [
        'tramos', 'cantidad_etapas', 'prop_etapas']]

    rosa = '#E71E79'
    celeste = '#19C2C2'

    reemplazo_vuelta = {
        0.0: 1.0,
        0.1: 0.9,
        0.2: 0.8,
        0.3: 0.7,
        0.4: 0.6,
        0.5: 0.5,
        0.6: 0.4,
        0.7: 0.3,
        0.8: 0.2,
        0.9: 0.1,
        1.0: 0.0,
    }
    print("Produciendo geometrias de recorrido con datos...")
    # geoms lineas
    tramos_ida_geom = []
    tramos_vuelta_geom = []

    tramos = pd.Series(np.linspace(0, 1, 11)).round(1)

    for i in range(10):
        l_ida = LineString([rec.interpolate(tramos[i], normalized=True),
                            rec.interpolate(tramos[i+1], normalized=True)
                            ])
        tramos_ida_geom.append(l_ida)

        l_vuelta = LineString([rec.interpolate(reemplazo_vuelta[tramos[i]],
                                               normalized=True),
                               rec.interpolate(
                                   reemplazo_vuelta[tramos[i+1]],
            normalized=True)
        ])
        tramos_vuelta_geom.append(l_vuelta)

    # produciendo los geodataframe de ambos sentidos
    gdf_ida = gpd.GeoDataFrame(pd.DataFrame(
        {'tramos': tramos[:-1]}), geometry=tramos_ida_geom, crs='epsg:4326')

    # proyectar en posgar 2007
    epsg = 9265
    gdf_ida = gdf_ida.to_crs(epsg=epsg)
    gdf_ida = gdf_ida.merge(tabla_ida, on='tramos', how='left')

    gdf_ida = gdf_ida.fillna(0)
    gdf_ida['factor'] = gdf_ida['prop_etapas']*factor

    # fijando un valor minimo para que haya una linea
    gdf_ida['factor'] = np.where(
        gdf_ida['factor'] <= factor_minimo, factor_minimo, gdf_ida['factor'])

    gdf_vuelta = gpd.GeoDataFrame(pd.DataFrame(
        {'tramos': tramos[:-1]}), geometry=tramos_vuelta_geom, crs='epsg:4326')

    # proyectar en posgar 2007
    gdf_vuelta = gdf_vuelta.to_crs(epsg=epsg)

    gdf_vuelta = gdf_vuelta.merge(tabla_vuelta, on='tramos', how='left')
    gdf_vuelta = gdf_vuelta.fillna(0)
    gdf_vuelta['factor'] = gdf_vuelta['prop_etapas']*factor

    # fijando un valor minimo para que haya una linea
    gdf_vuelta['factor'] = np.where(
        gdf_vuelta['factor'] <= factor_minimo, factor_minimo,
        gdf_vuelta['factor'],)

    # creando las flechas
    flecha_ida = gdf_ida.loc[gdf_ida.tramos == 0.0, 'geometry']
    flecha_ida = list(flecha_ida.item().coords)
    flecha_ida_inicio = flecha_ida[0]
    flecha_ida_fin = flecha_ida[1]

    flecha_vuelta = gdf_vuelta.loc[gdf_vuelta.tramos == 0.0, 'geometry']
    flecha_vuelta = list(flecha_vuelta.item().coords)
    flecha_vuelta_inicio = flecha_vuelta[0]
    flecha_vuelta_fin = flecha_vuelta[1]

    # ordenando las columnas para que coincidan con al recorrido
    flecha_oe_xy = (0.4, 1.1)
    flecha_oe_text_xy = (0.05, 1.1)
    flecha_eo_xy = (0.6, 1.1)
    flecha_eo_text_xy = (0.95, 1.1)
    labels_eo = ['Fin', '', '', '', '', 'Medio', '', '', '', 'Inicio']
    labels_oe = ['Inicio', '', '', '', '', 'Medio', '', '', '', 'Fin']

    # creando buffers en base a
    gdf_ida['geometry'] = gdf_ida.geometry.buffer(gdf_ida.factor)
    gdf_vuelta['geometry'] = gdf_vuelta.geometry.buffer(gdf_vuelta.factor)

    # si sentido este a oeste:
    if flecha_ida_inicio[0] > flecha_ida_fin[0]:
        gdf_ida = gdf_ida.sort_values('tramos', ascending=False)
        flecha_ida_xy = flecha_eo_xy
        flecha_ida_text_xy = flecha_eo_text_xy
        labels_ida = labels_eo
    else:
        flecha_ida_xy = flecha_oe_xy
        flecha_ida_text_xy = flecha_oe_text_xy
        labels_ida = labels_oe

    if flecha_vuelta_inicio[0] > flecha_vuelta_fin[0]:
        gdf_vuelta = gdf_vuelta.sort_values('tramos', ascending=False)

        flecha_vuelta_xy = flecha_eo_xy
        flecha_vuelta_text_xy = flecha_eo_text_xy
        labels_vuelta = labels_eo

    else:
        flecha_vuelta_xy = flecha_oe_xy
        flecha_vuelta_text_xy = flecha_oe_text_xy
        labels_vuelta = labels_oe

    # Setear titulo y eje de acuerdo a indicador

    if indicador == 'cantidad_etapas':
        titulo = 'Segmentos del recorrido - Cantidad de etapas'
        eje_y = 'Cantidad de etapas por sentido'
    elif indicador == 'prop_etapas':
        titulo = 'Segmentos del recorrido - Porcentaje de etapas totales'
        eje_y = 'Porcentaje del total de etapas'
    else:
        raise Exception(
            "Indicador debe ser 'cantidad_etapas' o 'prop_etapas'")
    if rango_hrs:
        rango_str = f' {rango_hrs[0]}-{rango_hrs[1]} hrs'
    else:
        rango_str = ''

    titulo = titulo + rango_str + ' - ' + str(id_linea)

    print("Creando gráfico")
    f = plt.figure(tight_layout=True, figsize=(20, 15))
    gs = f.add_gridspec(nrows=3, ncols=2)
    ax1 = f.add_subplot(gs[0:2, 0])
    ax2 = f.add_subplot(gs[0:2, 1])
    ax3 = f.add_subplot(gs[2, 0])
    ax4 = f.add_subplot(gs[2, 1])

    font_dicc = {'fontsize': 18,
                 'fontweight': 'bold'}

    prov = cx.providers.Stamen.TonerLite

    gdf_ida.plot(ax=ax1, color=celeste, alpha=.8)

    gdf_vuelta.plot(ax=ax2, color=rosa, alpha=.8)

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.set_title('IDA', fontdict=font_dicc)
    ax2.set_title('VUELTA', fontdict=font_dicc)

    sns.barplot(data=gdf_ida, x="tramos",
                y=indicador, ax=ax3, color=celeste,
                order=gdf_ida.tramos.values)

    sns.barplot(data=gdf_vuelta, x="tramos",
                y=indicador, ax=ax4, color=rosa,
                order=gdf_vuelta.tramos.values)

    ax3.set_xticklabels(labels_ida)
    ax4.set_xticklabels(labels_vuelta)

    ax3.set_ylabel(eje_y)
    ax3.set_xlabel('')

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)

    ax4.spines.left.set_visible(False)
    ax4.spines.right.set_visible(False)
    ax4.spines.top.set_visible(False)

    ax4.get_yaxis().set_visible(False)

    ax4.set_ylabel('')
    ax4.set_xlabel('')

    margen_flecha = factor * 2

    ax1.annotate('', xy=(flecha_ida_fin[0] + margen_flecha,
                         flecha_ida_fin[1] + margen_flecha),
                 xytext=(flecha_ida_inicio[0] + margen_flecha,
                         flecha_ida_inicio[1] + margen_flecha),
                 va="center", ha="center",
                 arrowprops=dict(facecolor=celeste,
                                 shrink=0.05, edgecolor=celeste),
                 )

    ax2.annotate('', xy=(flecha_vuelta_fin[0] + margen_flecha,
                         flecha_vuelta_fin[1] + margen_flecha),
                 xytext=(flecha_vuelta_inicio[0] + margen_flecha,
                         flecha_vuelta_inicio[1] + margen_flecha),
                 va="center", ha="center",
                 arrowprops=dict(facecolor=rosa, shrink=0.05, edgecolor=rosa),
                 )

    ax3.annotate('Sentido', xy=flecha_ida_xy, xytext=flecha_ida_text_xy,
                 size=16, va="center", ha="center",
                 xycoords='axes fraction',
                 arrowprops=dict(facecolor=celeste,
                                 shrink=0.05, edgecolor=celeste),
                 )
    ax4.annotate('Sentido', xy=flecha_vuelta_xy, xytext=flecha_vuelta_text_xy,
                 size=16, va="center", ha="center",
                 xycoords='axes fraction',
                 arrowprops=dict(facecolor=rosa, shrink=0.05, edgecolor=rosa),
                 )

    f.suptitle(titulo, fontsize=20)
    try:
        cx.add_basemap(ax1, crs=gdf_ida.crs.to_string(), source=prov)
        cx.add_basemap(ax2, crs=gdf_vuelta.crs.to_string(), source=prov)
    except (UnidentifiedImageError):
        prov = cx.providers.CartoDB.Positron
        cx.add_basemap(ax1, crs=gdf_ida.crs.to_string(), source=prov)
        cx.add_basemap(ax2, crs=gdf_vuelta.crs.to_string(), source=prov)
    except (r_ConnectionError):
        pass
    for frm in ['png', 'pdf']:
        archivo = f'segmentos_id_linea_{id_linea}_{indicador}{rango_str}.{frm}'
        db_path = os.path.join("resultados", frm, archivo)
        f.savefig(db_path, dpi=300)
    plt.close(f)


def plot_voronoi_zones(voi, hexs, hexs2, show_map, alias):
    fig = Figure(figsize=(13.5, 13.5), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plt.rcParams.update({"axes.facecolor": '#d4dadc',
                        'figure.facecolor': '#d4dadc'})
    voi = voi.to_crs(3857)
    voi.geometry.boundary.plot(edgecolor='grey', linewidth=.5, ax=ax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                    attribution=None, attribution_size=10)
    voi['coords'] = voi['geometry'].apply(
        lambda x: x.representative_point().coords[:])
    voi['coords'] = [coords[0] for coords in voi['coords']]
    voi.apply(lambda x: ax.annotate(
        text=x['Zona_voi'],
        xy=x.geometry.centroid.coords[0],
        ha='center',
        color='darkblue',
    ), axis=1)
    ax.set_title('Zonificación', fontsize=12)
    ax.axis('off')

    if show_map:

        display(fig)

        # Display figura temporal
        fig = Figure(figsize=(13.5, 13.5), dpi=70)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        hexs.to_crs(3857).plot(markersize=hexs['fex']/500, ax=ax)
        hexs2.to_crs(3857).boundary.plot(ax=ax, lw=.3)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                        attribution=None, attribution_size=10)
        ax.axis('off')

    # graba resultados
    file_path = os.path.join("resultados", "png", f"{alias}Zona_voi_map.png")
    fig.savefig(file_path, dpi=300)
    print('Zonificación guardada en', file_path)

    file_path = os.path.join("resultados", "pdf", f"{alias}Zona_voi_map.pdf")
    fig.savefig(file_path, dpi=300)
    voi = voi.to_crs(4326)

    file_path = os.path.join("resultados", "pdf", f"{alias}Zona_voi.geojson")
    voi[['Zona_voi', 'geometry']].to_file(file_path)


def create_visualizations():
    """
    Esta funcion corre las diferentes funciones de visualizaciones
    """
    pd.options.mode.chained_assignment = None

    # Leer informacion de viajes y distancias
    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    viajes = pd.read_sql_query(
        """
        SELECT *
        FROM viajes
        """,
        conn_data,
    )

    factores_expansion = pd.read_sql_query(
        """
        SELECT *
        FROM factores_expansion
        """,
        conn_data,
    )

    distancias = pd.read_sql_query(
        """
        SELECT *
        FROM distancias
        """,
        conn_insumos,
    )

    conn_insumos.close()
    conn_data.close()

    # Agrego factor de expansión a viajes
    viajes = viajes.merge(factores_expansion[['dia',
                                              'id_tarjeta',
                                              'factor_expansion']],
                          on=['dia',
                              'id_tarjeta'])

    # Agrego campo de distancias de los viajes
    viajes = viajes.merge(distancias,
                          how='left',
                          on=['h3_o', 'h3_d'])

    # Imputar anio, mes y tipo de dia
    viajes['yr'] = pd.to_datetime(viajes.dia).dt.year
    viajes['mo'] = pd.to_datetime(viajes.dia).dt.month
    viajes['dow'] = pd.to_datetime(viajes.dia).dt.day_of_week
    viajes.loc[viajes.dow >= 5, 'tipo_dia'] = 'Fin de semana'
    viajes.loc[viajes.dow < 5, 'tipo_dia'] = 'Día hábil'
    v_iter = viajes.groupby(['yr', 'mo', 'tipo_dia'],
                            as_index=False).factor_expansion.sum().iterrows()
    for _, i in v_iter:

        desc_dia = f'{str(i.mo).zfill(2)}/{i.yr} ({i.tipo_dia})'
        desc_dia_file = f'{i.yr}-{str(i.mo).zfill(2)}({i.tipo_dia})'

        viajes_dia = viajes[(viajes.yr == i.yr) & (
            viajes.mo == i.mo) & (viajes.tipo_dia == i.tipo_dia)]

        print('Imprimiendo tabla de matrices OD')
        # Impirmir tablas con matrices OD
        imprimir_matrices_od(viajes=viajes_dia,
                             var_fex='factor_expansion',
                             title=f'Matriz OD {desc_dia}',
                             savefile=f'{desc_dia_file}'
                             )

        print('Imprimiendo mapas de líneas de deseo')
        # Imprimir lineas de deseo
        imprime_lineas_deseo(df=viajes_dia,
                             h3_o='',
                             h3_d='',
                             var_fex='factor_expansion',
                             title=f'Líneas de deseo {desc_dia}',
                             savefile=f'{desc_dia_file}')

        print('Imprimiendo gráficos')
        titulo = f'Cantidad de viajes en transporte público {desc_dia}'
        imprime_graficos_hora(viajes_dia,
                              title=titulo,
                              savefile=f'{desc_dia_file}_viajes',
                              var_fex='factor_expansion')

        print('Imprimiendo mapas de burbujas')
        viajes_n = viajes_dia[(viajes_dia.id_viaje > 1)]
        imprime_burbujas(viajes_n,
                         res=7,
                         h3_o='h3_o',
                         alpha=.4,
                         cmap='rocket_r',
                         var_fex='factor_expansion',
                         porc_viajes=100,
                         title=f'Destinos de los viajes {desc_dia}',
                         savefile=f'{desc_dia_file}_burb_destinos',
                         show_fig=False)

        viajes_n = viajes_dia[(viajes_dia.id_viaje == 1)]
        imprime_burbujas(viajes_n,
                         res=7,
                         h3_o='h3_o',
                         alpha=.4,
                         cmap='flare',
                         var_fex='factor_expansion',
                         porc_viajes=100,
                         title=f'Hogares {desc_dia}',
                         savefile=f'{desc_dia_file}_burb_hogares',
                         show_fig=False)


def imprimir_matrices_od(viajes,
                         savefile='viajes',
                         title='Matriz OD',
                         var_fex=""):

    alias = leer_alias()

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    zonas = pd.read_sql_query(
        """
        SELECT * from zonas
        """,
        conn_insumos,
    )

    conn_insumos.close()
    zonas[f'h3_r6'] = zonas['h3'].apply(h3.h3_to_parent, res=6)
    zonas[f'h3_r7'] = zonas['h3'].apply(h3.h3_to_parent, res=7)

    df, matriz_zonas = traigo_zonificacion(
        viajes, zonas, h3_o='h3_o', h3_d='h3_d')

    if len(var_fex) == 0:
        var_fex = 'var_fex'
        df[var_fex] = 1

    for i in matriz_zonas:
        var_zona = i[1]
        matriz_order = i[2]

        imprime_od(
            df,
            zona_origen=f"{var_zona}_o",
            zona_destino=f"{var_zona}_d",
            var_fex=var_fex,
            x_rotation=90,
            normalize=True,
            cmap="Reds",
            title='Matriz OD General',
            figsize_tuple='',
            matriz_order=matriz_order,
            savefile=f"{alias}{savefile}_{var_zona}",

        )

        imprime_od(
            df[(df.cant_etapas > 1)],
            zona_origen=f"{var_zona}_o",
            zona_destino=f"{var_zona}_d",
            var_fex=var_fex,
            x_rotation=90,
            normalize=True,
            cmap="Reds",
            title='Matriz OD viajes con transferencia',
            figsize_tuple='',
            matriz_order=matriz_order,
            savefile=f"{alias}{savefile}_{var_zona}_transferencias",
        )

        imprime_od(
            df[(df.distance_osm_drive <= 5)],
            zona_origen=f"{var_zona}_o",
            zona_destino=f"{var_zona}_d",
            var_fex=var_fex,
            x_rotation=90,
            normalize=True,
            cmap="Reds",
            title='Matriz OD viajes cortos (<5kms)',
            figsize_tuple='',
            matriz_order=matriz_order,
            savefile=f"{alias}{savefile}_{var_zona}_corta_distancia",
        )

        # Imprime hora punta manana, mediodia, tarde

        df_tmp = df.groupby(['dia', 'hora'], as_index=False)[
            var_fex].sum().reset_index()
        df_tmp = df_tmp.groupby(['hora'])[var_fex].mean().reset_index()

        try:
            manana = df_tmp[(df_tmp.hora.astype(int) >= 6) & (
                df_tmp.hora.astype(int) < 12)][var_fex].idxmax()
        except ValueError:
            manana = np.nan

        try:
            mediodia = df_tmp[(df_tmp.hora.astype(int) >= 12) & (
                df_tmp.hora.astype(int) < 16)][var_fex].idxmax()
        except ValueError:
            mediodia = np.nan
        try:
            tarde = df_tmp[(df_tmp.hora.astype(int) >= 16) & (
                df_tmp.hora.astype(int) < 22)][var_fex].idxmax()
        except ValueError:
            tarde = np.nan

        if manana != np.nan:
            imprime_od(
                df[(df.hora.astype(int) >= manana-1) &
                    (df.hora.astype(int) <= manana+1)],
                zona_origen=f"{var_zona}_o",
                zona_destino=f"{var_zona}_d",
                var_fex=var_fex,
                x_rotation=90,
                normalize=True,
                cmap="Reds",
                title='Matriz OD viajes punta mañana',
                figsize_tuple='',
                matriz_order=matriz_order,
                savefile=f"{alias}{savefile}_{var_zona}_punta_manana",
            )

        if mediodia != np.nan:
            imprime_od(
                df[(df.hora.astype(int) >= mediodia-1) &
                    (df.hora.astype(int) <= mediodia+1)],
                zona_origen=f"{var_zona}_o",
                zona_destino=f"{var_zona}_d",
                var_fex=var_fex,
                x_rotation=90,
                normalize=True,
                cmap="Reds",
                title='Matriz OD viajes punta mediodía',
                figsize_tuple='',
                matriz_order=matriz_order,
                savefile=f"{alias}{savefile}_{var_zona}_punta_mediodia",
            )

        if tarde != np.nan:
            imprime_od(
                df[(df.hora.astype(int) >= tarde-1) &
                    (df.hora.astype(int) <= tarde+1)],
                zona_origen=f"{var_zona}_o",
                zona_destino=f"{var_zona}_d",
                var_fex=var_fex,
                x_rotation=90,
                normalize=True,
                cmap="Reds",
                title='Matriz OD viajes punta tarde',
                figsize_tuple='',
                matriz_order=matriz_order,
                savefile=f"{alias}{savefile}_{var_zona}_punta_tarde",
            )


def imprime_lineas_deseo(df,
                         h3_o='',
                         h3_d='',
                         var_fex='',
                         title='Líneas de deseo',
                         savefile='lineas_deseo',
                         ):
    """
    Esta funcion toma un df de viajes con destino validado
    nombres de columnas con el h3 de origen y destino
    un nombre con la columna del factor de expansion
    y nombres para el titulo del mapa y el archivo
    y produce un mapa con lineas de deseo para todas las
    geografias presentes en la tabla zonas
    """

    pd.options.mode.chained_assignment = None
    alias = leer_alias()

    conn_insumos = iniciar_conexion_db(tipo='insumos')

    zonas = pd.read_sql_query(
        """
        SELECT * from zonas
        """,
        conn_insumos,
    )

    conn_insumos.close()

    zonas[f'h3_r6'] = zonas['h3'].apply(h3.h3_to_parent, res=6)
    zonas[f'h3_r7'] = zonas['h3'].apply(h3.h3_to_parent, res=7)

    zonas = gpd.GeoDataFrame(
        zonas,
        geometry=gpd.points_from_xy(zonas['longitud'], zonas['latitud']),
        crs=4326,
    )

    if len(h3_o) == 0:
        h3_o = 'h3_o_norm'
    if len(h3_d) == 0:
        h3_d = 'h3_d_norm'

    if len(var_fex) == 0:
        var_fex = 'fex'
        df[var_fex] = 1

    # Clasificar od en terminos de zonas
    df, matriz_zonas = traigo_zonificacion(df,
                                           zonas,
                                           h3_o=h3_o,
                                           h3_d=h3_d,
                                           res_agg=True)

    for m in matriz_zonas:
        var_zona = m[1]

        lineas_deseo(df,
                     zonas,
                     var_zona,
                     var_fex,
                     h3_o,
                     h3_d,
                     alpha=.4,
                     cmap='viridis_r',
                     porc_viajes=100,
                     title=title,
                     savefile=f"{alias}{savefile}_{var_zona}",
                     show_fig=False
                     )

        lineas_deseo(df[(df.cant_etapas > 1)],
                     zonas,
                     var_zona,
                     var_fex,
                     h3_o,
                     h3_d,
                     alpha=.4,
                     cmap='crest',
                     porc_viajes=90,
                     title=f'{title}\nViajes con transferencias',
                     savefile=f"{alias}{savefile}_{var_zona}_transferencias",
                     show_fig=False
                     )

        lineas_deseo(df[(df.distance_osm_drive <= 5)],
                     zonas,
                     var_zona,
                     var_fex,
                     h3_o,
                     h3_d,
                     alpha=.4,
                     cmap='magma_r',
                     porc_viajes=90,
                     title=f'{title}\nViajes de corta distancia (<5kms)',
                     savefile=f"{alias}{savefile}_{var_zona}_corta_distancia",
                     show_fig=False
                     )

        # Imprime hora punta manana, mediodia, tarde

        df_tmp = df\
            .groupby(['dia', 'hora'], as_index=False)\
            .factor_expansion.sum()\
            .rename(columns={'factor_expansion': 'cant'})\
            .reset_index()
        df_tmp = df_tmp.groupby(['hora']).cant.mean().reset_index()
        try:
            manana = df_tmp[(df_tmp.hora.astype(int) >= 6) & (
                df_tmp.hora.astype(int) < 12)].cant.idxmax()
        except ValueError:
            manana = np.nan

        try:
            mediodia = df_tmp[(df_tmp.hora.astype(int) >= 12) & (
                df_tmp.hora.astype(int) < 16)].cant.idxmax()
        except ValueError:
            mediodia = np.nan

        try:
            tarde = df_tmp[(df_tmp.hora.astype(int) >= 16) & (
                df_tmp.hora.astype(int) < 22)].cant.idxmax()
        except ValueError:
            tarde = np.nan

        if manana != np.nan:
            lineas_deseo(df[
                (df.hora.astype(int) >= manana-1) &
                (df.hora.astype(int) <= manana+1)],
                zonas,
                var_zona,
                var_fex,
                h3_o,
                h3_d,
                alpha=.4,
                cmap='magma_r',
                porc_viajes=90,
                title=f'{title}\nViajes en hora punta mañana',
                savefile=f"{alias}{savefile}_{var_zona}_punta_manana",
                show_fig=False,
                normalizo_latlon=False)

        if mediodia != np.nan:
            lineas_deseo(df[
                (df.hora.astype(int) >= mediodia-1) &
                (df.hora.astype(int) <= mediodia+1)],
                zonas,
                var_zona,
                var_fex,
                h3_o,
                h3_d,
                alpha=.4,
                cmap='magma_r',
                porc_viajes=90,
                title=f'{title}\nViajes en hora punta mediodia',
                savefile=f"{alias}{savefile}_{var_zona}_punta_mediodia",
                show_fig=False,
                normalizo_latlon=False)

        if tarde != np.nan:
            lineas_deseo(df[
                (df.hora.astype(int) >= tarde-1) &
                (df.hora.astype(int) <= tarde+1)],
                zonas,
                var_zona,
                var_fex,
                h3_o,
                h3_d,
                alpha=.4,
                cmap='magma_r',
                porc_viajes=90,
                title=f'{title}\nViajes en hora punta tarde',
                savefile=f"{alias}{savefile}_{var_zona}_punta_tarde",
                show_fig=False,
                normalizo_latlon=False)


def imprime_graficos_hora(viajes,
                          title='Cantidad de viajes en transporte público',
                          savefile='viajes',
                          var_fex=''):

    pd.options.mode.chained_assignment = None
    configs = leer_configs_generales()
    db_path = traigo_db_path
    alias = leer_alias()

    df_aux = pd.DataFrame([(str(x).zfill(2))
                          for x in list(range(0, 24))], columns=['hora'])
    df_aux['dia'] = viajes.head(1).dia.values[0]
    df_aux['cant'] = 0
    df_aux['modo'] = viajes.modo.unique()[0]

    if not var_fex:
        viajes['cant'] = 1
    else:
        viajes['cant'] = viajes[var_fex]

    viajesxhora = pd.concat([viajes, df_aux], ignore_index=True)
    viajesxhora['hora'] = viajesxhora.hora.astype(str).str[:2].str.zfill(2)

    viajesxhora = viajesxhora.groupby(
        ['dia', 'hora']).cant.sum().reset_index()
    viajesxhora = viajesxhora.groupby(['hora']).cant.mean().reset_index()

    viajesxhora['cant'] = viajesxhora['cant'].round().astype(int)

    savefile_ = f'{savefile}_x_hora'

    # Viajes por hora
    with sns.axes_style(
        {"axes.facecolor": "#cadce0",
         'figure.facecolor': '#cadce0',
         }):
        fig = Figure(figsize=(10, 3), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        viajesxhora.plot(ax=ax, legend=False, label=False)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Hora', fontsize=8)
        ax.set_ylabel('Viajes', fontsize=8)
        ax.set_xticks(list(range(0, 24)))
        ax.tick_params(labelsize=6)

        print("Nuevos archivos en resultados: ", f'{alias}{savefile_}')
        db_path = os.path.join("resultados", "png", f"{alias}{savefile_}.png")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

        db_path = os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

    # Viajes por hora y modo de transporte
    viajesxhora = pd.concat([viajes, df_aux], ignore_index=True)
    viajesxhora['hora'] = viajesxhora.hora.astype(str).str[:2].str.zfill(2)
    viajesxhora = viajesxhora.groupby(
        ['dia', 'hora', 'modo'], as_index=False).cant.sum()
    viajesxhora = viajesxhora.groupby(
        ['hora', 'modo'], as_index=False).cant.mean()

    viajesxhora.loc[viajesxhora.modo.str.contains(
        'Multi'), 'modo'] = 'Multietapa'
    viajesxhora = viajesxhora.groupby(['hora', 'modo'])[
        'cant'].sum().reset_index()

    viajesxhora['cant'] = viajesxhora['cant'].round().astype(int)

    # Viajes por hora
    savefile_ = f'{savefile}_modo'
    with sns.axes_style(
        {"axes.facecolor": "#cadce0",
         'figure.facecolor': '#cadce0',
         }):

        fig = Figure(figsize=(10, 3), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        for i in viajesxhora.modo.unique():
            viajesxhora[viajesxhora.modo == i].reset_index().plot(
                ax=ax, y='cant', legend=True, label=i)

        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Hora', fontsize=8)
        ax.set_ylabel('Viajes', fontsize=8)
        ax.set_xticks(list(range(0, 24)))
        ax.tick_params(labelsize=6)
        print("Nuevos archivos en resultados: ", f'{alias}{savefile_}')
        db_path = os.path.join("resultados", "png", f"{alias}{savefile_}.png")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

        db_path = os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")
        fig.savefig(db_path, dpi=300, bbox_inches="tight")

    # Distribución de viajes
    savefile_ = f'{savefile}_dist'
    vi = viajes[(viajes.distance_osm_drive.notna())
                & (viajes.distance_osm_drive > 0)
                & (viajes.h3_o != viajes.h3_d)]
    vi['distance_osm_drive'] = vi['distance_osm_drive'].astype(int)

    vi = vi\
        .groupby('distance_osm_drive', as_index=False)\
        .factor_expansion.sum()\
        .rename(columns={'factor_expansion': 'cant'})

    ytitle = "Viajes"
    if vi.cant.mean() > 1000:
        vi['cant'] = round(vi['cant']/1000)
        ytitle = "Viajes (en miles)"

    vi = vi.loc[vi.cant > 0, ['distance_osm_drive', 'cant']
                ].sort_values('distance_osm_drive')

    sns.set_style("darkgrid", {"axes.facecolor": "#cadce0",
                  'figure.facecolor': '#cadce0', "grid.linestyle": ":"})

    fig = Figure(figsize=(8, 4), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    sns.histplot(x='distance_osm_drive', weights='cant',
                 data=vi, bins=len(vi), element='poly', ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Distancia (kms)", fontsize=10)
    ax.set_ylabel(ytitle, fontsize=10)
    ax.set_xticks(list(range(0, len(vi)+1, 5)))

    fig.tight_layout()

    print("Nuevos archivos en resultados: ", f'{alias}{savefile_}')
    db_path = os.path.join("resultados", "png", f"{alias}{savefile_}.png")
    fig.savefig(db_path, dpi=300, bbox_inches="tight")

    db_path = os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")
    fig.savefig(db_path, dpi=300, bbox_inches="tight")


def imprime_burbujas(df,
                     res=7,
                     h3_o='h3_o',
                     alpha=.4,
                     cmap='viridis_r',
                     var_fex='',
                     porc_viajes=90,
                     title='burbujas',
                     savefile='burbujas',
                     show_fig=False):

    pd.options.mode.chained_assignment = None
    configs = leer_configs_generales()
    db_path = traigo_db_path
    alias = leer_alias()

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    zonas = pd.read_sql_query(
        """
        SELECT * from zonas
        """,
        conn_insumos,
    )

    conn_data.close()
    conn_insumos.close()

    zonas = gpd.GeoDataFrame(
        zonas,
        geometry=gpd.points_from_xy(zonas['longitud'], zonas['latitud']),
        crs=4326)

    if len(var_fex) == 0:
        var_fex = 'fex'
        df[var_fex] = 1

    df_agg = crea_df_burbujas(df,
                              zonas,
                              h3_o=h3_o,
                              var_fex=var_fex,
                              porc_viajes=porc_viajes,
                              res=res
                              )

    df_agg[var_fex] = df_agg[var_fex].round().astype(int)

    if len(df_agg) > 0:

        multip = df_agg[var_fex].head(1).values[0] / 500

        fig = Figure(figsize=(13.5, 13.5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        zonas[zonas['h3'].isin(df[h3_o].unique())].to_crs(
            3857).plot(ax=ax, alpha=0)
        try:
            df_agg.to_crs(3857).plot(ax=ax,
                                     alpha=alpha,
                                     cmap=cmap,
                                     markersize=df_agg[var_fex] / multip,
                                     column=var_fex,
                                     scheme='FisherJenks',
                                     k=5,
                                     legend=True,
                                     legend_kwds={
                                         'loc': 'upper right',
                                         'title': 'Viajes',
                                         'fontsize': 8,
                                            'title_fontsize': 10,
                                     }
                                     )
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                            attribution=None, attribution_size=10)

            ax.set_title(title, fontsize=12)

            leg = ax.get_legend()
            # leg._loc = 3

            for lbl in leg.get_texts():
                label_text = lbl.get_text()
                lower = label_text.split(',')[0]
                upper = label_text.split(',')[1]
                new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'
                lbl.set_text(new_text)

            ax.add_artist(
                ScaleBar(1, location='lower right', box_alpha=0, pad=1))
            ax.axis('off')

            if len(savefile) > 0:
                print("Nuevos archivos en resultados: ", f'{alias}{savefile}')
                db_path = os.path.join("resultados", "png",
                                       f"{alias}{savefile}.png")
                fig.savefig(db_path, dpi=300, bbox_inches="tight")

                db_path = os.path.join("resultados", "pdf",
                                       f"{alias}{savefile}.pdf")
                fig.savefig(db_path, dpi=300, bbox_inches="tight")

            if show_fig:
                display(fig)

        except (ValueError) as e:
            print(e)


def traigo_zonificacion(viajes,
                        zonas,
                        h3_o='h3_o',
                        h3_d='h3_d',
                        res_agg=False):
    """
    Esta funcion toma la tabla viajes
    la tabla zonas, los nombres de las columnas con el h3 de origen y destino
    y un parametro para usar h3 con resolucion mas agregada
    y clasifica los origenes y destinos de los viajes para cada zona
    """
    configs = leer_configs_generales()

    matriz_zonas = []
    vars_zona = []
    if 'Zona_voi' in zonas.columns:

        matriz_zonas = [['',
                         'Zona_voi',
                         [str(x) for x in list(
                             range(1, len(zonas.Zona_voi.unique())+1))]
                         ]]
        vars_zona = ['Zona_voi']

    if res_agg:
        zonas[f'h3_r6'] = zonas['h3'].apply(h3.h3_to_parent, res=6)
        zonas[f'h3_r7'] = zonas['h3'].apply(h3.h3_to_parent, res=7)

        matriz_zonas += [['', f'h3_r6', ''],
                         ['', f'h3_r7', '']]
        vars_zona += [f'h3_r6']
        vars_zona += [f'h3_r7']

    if configs["zonificaciones"]:
        for n in range(0, 5):

            try:
                file_zona = configs["zonificaciones"][f"geo{n+1}"]
                var_zona = configs["zonificaciones"][f"var{n+1}"]

                try:
                    matriz_order = configs["zonificaciones"][f"orden{n+1}"]
                except KeyError:
                    matriz_order = ""

                if var_zona in zonas.columns:
                    matriz_zonas += [[file_zona, var_zona, matriz_order]]
                    vars_zona += [var_zona]
            except KeyError:
                pass

    vars_o = [h3_o] + [f'{x}_o' for x in vars_zona]
    vars_d = [h3_d] + [f'{x}_d' for x in vars_zona]

    zonas_tmp = zonas[['h3']+vars_zona]
    zonas_tmp.columns = vars_o
    viajes = viajes.merge(
        zonas_tmp,
        on=h3_o
    )

    zonas_tmp = zonas[['h3']+vars_zona]
    zonas_tmp.columns = vars_d
    viajes = viajes.merge(
        zonas_tmp,
        on=h3_d
    )
    return viajes, matriz_zonas


def imprime_od(
    df,
    zona_origen,
    zona_destino,
    var_fex="",
    normalize=False,
    margins=False,
    matriz_order="",
    matriz_order_row="",
    matriz_order_col="",
    path_resultados=Path(),
    savefile="",
    title="Matriz OD",
    figsize_tuple='',
    fontsize=12,
    fmt="",
    cbar=False,
    x_rotation=0,
    y_rotation=0,
    cmap="Blues",
    total_color="navy",
    total_background_color="white",
    show_fig=False,
):

    if len(fmt) == 0:
        if normalize:
            fmt = ".1%"
        else:
            fmt = ".1f"
    df = df[(df[zona_origen].notna()) & (df[zona_destino].notna())].copy()

    fill_value = mcolors.to_rgba(total_background_color)

    if len(matriz_order) > 0:
        matriz_order_row = matriz_order
        matriz_order_col = matriz_order

    if len(var_fex) == 0:
        var_fex = 'fex'
        df[var_fex] = 1

    df = df.groupby(['dia', zona_origen, zona_destino],
                    as_index=False)[var_fex].sum()
    df = df.groupby([zona_origen, zona_destino], as_index=False)[
        var_fex].mean()

    df[var_fex] = df[var_fex].round().astype(int)

    if len(df) > 0:

        vals = df.loc[~(df[zona_origen].isin(
            df[zona_destino].unique())), zona_origen].unique()
        for i in vals:
            df = pd.concat([df,
                            pd.DataFrame(
                                [[i, i, 0]],
                                columns=[
                                    zona_origen,
                                    zona_destino,
                                    var_fex
                                ])
                            ])
        vals = df.loc[~(df[zona_destino].isin(
            df[zona_origen].unique())), zona_destino].unique()
        for i in vals:
            df = pd.concat([df,
                            pd.DataFrame(
                                [[i, i, 0]],
                                columns=[
                                    zona_destino,
                                    zona_origen,
                                    var_fex])
                            ])

        od_heatmap = pd.crosstab(
            index=df[zona_origen],
            columns=df[zona_destino],
            values=df[var_fex],
            aggfunc="sum",
            normalize=normalize,
        )

        if len(figsize_tuple) == 0:
            figsize_tuple = (len(od_heatmap)+1, len(od_heatmap)+1)

        matriz_order = [i for i in matriz_order if i in od_heatmap.columns]
        matriz_order_row = [
            i for i in matriz_order_row if i in od_heatmap.columns]
        matriz_order_col = [
            i for i in matriz_order_col if i in od_heatmap.columns]

        if len(matriz_order_col) > 0:
            od_heatmap = od_heatmap[matriz_order_col]
        if len(matriz_order_row) > 0:

            od_heatmap = (
                od_heatmap.reset_index()
                .sort_values(
                    zona_origen,
                    key=lambda s: s.apply(matriz_order_row.index),
                    ignore_index=True,
                )
                .set_index(zona_origen)
            )

        for _ in od_heatmap.columns:
            od_heatmap.loc[od_heatmap[_] == 0, _] = np.nan

        if margins:
            od_heatmap_sintot = od_heatmap.copy()
            od_heatmap["Total"] = od_heatmap.sum(axis=1)
            od_heatmap = pd.concat(
                [od_heatmap, pd.DataFrame([od_heatmap.sum()], index=["Total"])]
            )

            od_heatmap_tmp = od_heatmap.copy()
            od_heatmap_tmp["Total"] = 0
            od_heatmap_tmp.iloc[len(od_heatmap_tmp) - 1] = 0

            fig = Figure(figsize=figsize_tuple, dpi=150)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            sns.heatmap(
                od_heatmap_tmp,
                cmap=cmap,
                annot=True,
                fmt=fmt,
                annot_kws={"size": fontsize},
                square=True,
                linewidth=0.5,
                cbar=cbar,
                ax=ax,
            )

            # find your QuadMesh object and get array of colors
            facecolors_anterior = ax.findobj(QuadMesh)[0].get_facecolors()

        fig = Figure(figsize=figsize_tuple, dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        sns.heatmap(
            od_heatmap,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            square=True,
            linewidth=0.5,
            cbar=cbar,
            ax=ax,
            xticklabels=True,
            yticklabels=True,
        )

        ax.set_title(title, fontsize=fontsize)
        ax.set_ylabel("Origen", fontsize=fontsize)
        ax.set_xlabel("Destino", fontsize=fontsize)

        # move x and y ticks
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("right")

        ax.set_xticklabels(
            od_heatmap.columns.tolist(),
            rotation=x_rotation,
            ha="right",
        )

        if margins:

            # find your QuadMesh object and get array of colors
            quadmesh = ax.findobj(QuadMesh)[0]
            facecolors = quadmesh.get_facecolors()

            # replace background heatmap colors
            for i in range(0, len(facecolors)):
                if (((i + 1) % len(od_heatmap.columns)) != 0) & (
                    i < (len(facecolors) - len(od_heatmap.columns))
                ):
                    facecolors[i] = facecolors_anterior[i]
                else:
                    facecolors[i] = fill_value

            # set modified colors
            quadmesh.set_facecolors = facecolors

            # modify all text to black or white
            lst = []
            for _ in od_heatmap_sintot.columns:
                lst += od_heatmap[_].tolist()
            val_min = pd.DataFrame(lst, columns=["valor"]).drop_duplicates()
            val_min["val_type"] = pd.qcut(
                val_min.valor,
                q=4,
                labels=["1", "2", "3", "4"],
            )
            val_min = val_min[val_min.val_type == "4"].valor.min()

            col_totals = np.arange(
                len(od_heatmap.columns) - 1,
                (len(od_heatmap.columns)) * len(od_heatmap),
                len(od_heatmap.columns),
            ).tolist()
            ii = 0
            for i in ax.findobj(Text):

                if (ii in col_totals[:-1]) | (
                    (ii >= len(facecolors) - len(od_heatmap.columns))
                    & (ii < len(facecolors) - 1)
                ):
                    i.set_color(total_color)
                else:
                    try:
                        value_i = (
                            str(i)
                            .replace("Text(", "")
                            .replace(")", "")
                            .replace("'", "")
                            .split(",")[2]
                            .replace(" ", "")
                        )

                        if value_i == "Total":
                            i.set_color(total_color)

                        if ii <= len(facecolors) - 1:
                            value_i = float(value_i)
                            cond = (value_i >= val_min) | (
                                ii == len(facecolors) - 1)
                            if cond:
                                i.set_color("white")

                            if value_i == 0:
                                facecolors[ii] = fill_value
                                i.set_color("white")

                    except:
                        pass
                ii += 1

        fig.tight_layout()

        if len(savefile) > 0:

            savefile = savefile+'_matrizod'

            print("Nuevos archivos en resultados: ", savefile)

            db_path = os.path.join("resultados", "png", f"{savefile}.png")
            fig.savefig(db_path, dpi=300, bbox_inches="tight")

            db_path = os.path.join("resultados", "pdf", f"{savefile}.pdf")
            fig.savefig(db_path, dpi=300, bbox_inches="tight")

            db_path = os.path.join(
                "resultados", "matrices", f"{savefile}.xlsx")

            if normalize:

                od1 = pd.crosstab(
                    index=df[zona_origen],
                    columns=df[zona_destino],
                    values=df[var_fex],
                    aggfunc="sum",
                    normalize=False,
                    margins=margins,
                )

                pd.concat(
                    [od1, pd.DataFrame([[], []]), od_heatmap],
                ).to_excel(db_path)
            else:
                od_heatmap.to_excel(path_resultados / (db_path))

        if show_fig:
            display(fig)


def lineas_deseo(df,
                 zonas,
                 var_zona,
                 var_fex,
                 h3_o,
                 h3_d,
                 alpha=.4,
                 cmap='viridis_r',
                 porc_viajes=100,
                 title='Líneas de deseo',
                 savefile='lineas_deseo',
                 show_fig=True,
                 normalizo_latlon=True):

    hexs = zonas.groupby(
        var_zona, as_index=False).size().drop(['size'], axis=1)
    hexs = hexs.merge(
        zonas.groupby(var_zona
                      ).apply(lambda x: np.average(x['longitud'],
                                                   weights=x['fex'])
                              ).reset_index(
        ).rename(columns={0: 'longitud'}),
        how='left')
    hexs = hexs.merge(
        zonas.groupby(var_zona
                      ).apply(lambda x: np.average(x['latitud'],
                                                   weights=x['fex'])
                              ).reset_index(
        ).rename(columns={0: 'latitud'}),
        how='left')

    tmp_o = f'{var_zona}_o'
    tmp_d = f'{var_zona}_d'

    if 'h3_' in tmp_o:
        tmp_h3_o = tmp_o
        tmp_h3_d = tmp_d
    else:
        tmp_h3_o = h3_o
        tmp_h3_d = h3_d

    # Normalizo con nueva zonificación
    if (tmp_o != tmp_h3_o) & (tmp_d != tmp_h3_d):
        df_agg = df.groupby(['dia', tmp_h3_o, tmp_h3_d, tmp_o,
                            tmp_d], as_index=False).agg({var_fex: 'sum'})
    else:
        df_agg = df.groupby(['dia', tmp_h3_o, tmp_h3_d],
                            as_index=False).agg({var_fex: 'sum'})

    if normalizo_latlon:
        df_agg = normalizo_lat_lon(df_agg,
                                   h3_o=tmp_h3_o,
                                   h3_d=tmp_h3_d,
                                   origen=tmp_o,
                                   destino=tmp_d,
                                   )

        tmp_o = f'{var_zona}_o_norm'
        tmp_d = f'{var_zona}_d_norm'

    # Agrego a res de gráfico latlong
    df_agg = df_agg.groupby(['dia', tmp_o, tmp_d], as_index=False).agg(
        {var_fex: 'sum'})

    df_agg = df_agg.groupby([tmp_o, tmp_d], as_index=False).agg(
        {var_fex: 'mean'})

    df_agg[var_fex] = df_agg[var_fex].round().astype(int)

    df_agg = df_agg.merge(
        hexs.rename(columns={var_zona: tmp_o,
                             'latitud': 'lat_o',
                             'longitud': 'lon_o'})
    )
    df_agg = df_agg.merge(
        hexs.rename(columns={var_zona: tmp_d,
                             'latitud': 'lat_d',
                             'longitud': 'lon_d'})
    )

    df_agg = df_agg.sort_values(
        var_fex, ascending=False).reset_index(drop=True)
    df_agg['cumsum'] = round(
        df_agg[var_fex].cumsum() / df_agg[var_fex].sum() * 100)

    df_agg = df_agg[df_agg['cumsum'] <= porc_viajes]

    df_agg = df_agg[df_agg[tmp_o] != df_agg[tmp_d]]

    if len(df_agg) > 0:
        try:
            df_agg = crear_linestring(
                df_agg, 'lon_o', 'lat_o', 'lon_d', 'lat_d')

            multip = df_agg[var_fex].head(1).values[0] / 10

            fig = Figure(figsize=(13.5, 13.5), dpi=150)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            zonas[zonas['h3'].isin(df[h3_o].unique())].to_crs(
                3857).plot(ax=ax, alpha=0)

            # En caso de que no haya suficientes casos para 5 jenks
            try:
                df_agg.to_crs(3857).plot(ax=ax,
                                         alpha=alpha,
                                         cmap=cmap,
                                         lw=df_agg[var_fex]/multip,
                                         column=var_fex,
                                         scheme='FisherJenks',
                                         k=5,
                                         legend=True,
                                         legend_kwds={
                                             'loc': 'upper right',
                                             'title': 'Viajes',
                                             'fontsize': 8,
                                                'title_fontsize': 10,
                                         }
                                         )
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                                attribution=None, attribution_size=10)

                leg = ax.get_legend()
                # leg._loc = 3
                for lbl in leg.get_texts():
                    label_text = lbl.get_text()
                    lower = label_text.split(',')[0]
                    upper = label_text.split(',')[1]
                    new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'
                    lbl.set_text(new_text)

                title_ = f'{title}: {var_zona}s'

                ax.set_title(title_, fontsize=12)
                ax.add_artist(
                    ScaleBar(1, location='lower right', box_alpha=0, pad=1))
                ax.axis('off')

                fig.tight_layout()

                if len(savefile) > 0:

                    savefile = savefile+'_lineas_deseo'

                    print("Nuevos archivos en resultados: ",
                          savefile)
                    db_path = os.path.join(
                        "resultados", "png", f"{savefile}.png")
                    fig.savefig(db_path, dpi=300, bbox_inches="tight")

                    db_path = os.path.join(
                        "resultados", "pdf", f"{savefile}.pdf")
                    fig.savefig(db_path, dpi=300, bbox_inches="tight")

                    crear_mapa_folium(df_agg,
                                      cmap,
                                      var_fex,
                                      savefile=f"{savefile}.html")

                if show_fig:
                    display(fig)
            except (ValueError) as e:
                print(e)

        except (ValueError) as e:
            pass


def crea_df_burbujas(df,
                     zonas,
                     h3_o='h3_o',
                     var_fex='',
                     porc_viajes=100,
                     res=7):

    zonas['h3_o_tmp'] = zonas['h3'].apply(h3.h3_to_parent, res=res)

    hexs = zonas.groupby(
        'h3_o_tmp', as_index=False).size().drop(['size'], axis=1)
    hexs = hexs.merge(
        zonas.groupby('h3_o_tmp'
                      ).apply(lambda x: np.average(x['longitud'],
                                                   weights=x['fex'])
                              ).reset_index(
        ).rename(columns={0: 'longitud'}),
        how='left')
    hexs = hexs.merge(
        zonas.groupby('h3_o_tmp'
                      ).apply(lambda x: np.average(x['latitud'],
                                                   weights=x['fex'])
                              ).reset_index(
        ).rename(columns={0: 'latitud'}),
        how='left')

    df['h3_o_tmp'] = df[h3_o].apply(h3.h3_to_parent, res=res)

    # Agrego a res de gráfico latlong
    df_agg = df.groupby(['dia', 'h3_o_tmp'],
                        as_index=False).agg({var_fex: 'sum'})
    df_agg = df_agg.groupby(
        ['h3_o_tmp'], as_index=False).agg({var_fex: 'mean'})

    df_agg = df_agg.merge(
        hexs.rename(columns={'latitud': 'lat_o',
                             'longitud': 'lon_o'})
    )

    df_agg = gpd.GeoDataFrame(
        df_agg,
        geometry=gpd.points_from_xy(df_agg['lon_o'], df_agg['lat_o']),
        crs=4326,)

    df_agg = df_agg.sort_values(
        var_fex, ascending=False).reset_index(drop=True)
    df_agg['cumsum'] = round(df_agg[var_fex].cumsum() /
                             df_agg[var_fex].sum() * 100)
    df_agg = df_agg[df_agg['cumsum'] < porc_viajes]

    return df_agg


def crear_mapa_folium(df_agg,
                      cmap,
                      var_fex,
                      savefile):

    bins = [df_agg[var_fex].min()-1] + \
        mapclassify.FisherJenks(df_agg[var_fex], k=5).bins.tolist()
    range_bins = range(0, len(bins)-1)
    bins_labels = [
        f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins]
    df_agg['cuts'] = pd.cut(df_agg[var_fex], bins=bins, labels=bins_labels)

    from folium import Figure
    fig = Figure(width=800, height=800)
    m = folium.Map(location=[df_agg.lat_o.mean(
    ), df_agg.lon_o.mean()], zoom_start=9, tiles='cartodbpositron')

    title_html = """
    <h3 align="center" style="font-size:20px"><b>Your map title</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    line_w = 0.5

    colors = mcp.gen_color(cmap=cmap, n=5)

    n = 0
    for i in bins_labels:

        df_agg[df_agg.cuts == i].explore(
            m=m,
            color=colors[n],
            style_kwds={'fillOpacity': 0.3, 'weight': line_w},
            name=i,
            tooltip=False,
        )
        n += 1
        line_w += 3

    folium.LayerControl(name='xx').add_to(m)

    fig.add_child(m)

    db_path = os.path.join("resultados", "html", savefile)
    m.save(db_path)
