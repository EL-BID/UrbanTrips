import pandas as pd
import geopandas as gpd
import numpy as np
from urbantrips.utils.utils import iniciar_conexion_db, levanto_tabla_sql
from urbantrips.geo.geo import normalizo_lat_lon
from urbantrips.utils.utils import traigo_tabla_zonas, calculate_weighted_means
from urbantrips.geo.geo import h3_to_lat_lon, h3toparent, h3_to_geodataframe, point_to_h3, create_h3_gdf
from urbantrips.utils.check_configs import check_config
from shapely.geometry import Point


def load_and_process_data():
    """
    Load and process data from databases, returning the etapas and viajes DataFrames.

    Returns:
        etapas (DataFrame): Processed DataFrame containing stage data.
        viajes (DataFrame): Processed DataFrame containing journey data.
    """

    # Establish connections to different databases for input data and operational data
    conn_insumos = iniciar_conexion_db(tipo='insumos')
    conn_data = iniciar_conexion_db(tipo='data')

    # Load distance data from 'distancias' table in 'insumos' database
    q_distancias = """
    SELECT
        h3_o, h3_d, distance_osm_drive, distance_osm_walk, distance_h3
    FROM
        distancias
    """
    distancias = pd.read_sql_query(q_distancias, conn_insumos)

    # Load stage data from 'etapas' table in 'data' database and merge with distance data
    etapas = pd.read_sql_query("SELECT * FROM etapas where od_validado==1", conn_data)
    etapas = etapas.merge(distancias, how='left')

    # Load journey data from 'viajes' table in 'data' database and merge with distance data
    viajes = pd.read_sql_query("SELECT * FROM viajes where od_validado==1", conn_data)
    viajes = viajes.merge(distancias, how='left')

    # Load travel times from gps and stations
    travel_times = pd.read_sql_query("SELECT * FROM travel_times_gps", conn_data)
    travel_times_stations = pd.read_sql_query("SELECT * FROM travel_times_stations", conn_data)

    travel_times = pd.concat([travel_times, travel_times_stations], ignore_index=True)
    
    if len(travel_times) > 0:
        etapas = etapas.merge(travel_times, how='left', on=['dia', 'id'])
    
        viajes = viajes.merge(etapas.groupby(['dia', 'id_tarjeta', 'id_viaje'], as_index=False).travel_time_min.sum().round(2), 
                              how='left', 
                              on=['dia', 'id_tarjeta', 'id_viaje'])
    
        viajes['travel_speed'] = (
                viajes['distance_osm_drive'] /
                (viajes['travel_time_min']/60)
            ).round(1)
                
        viajes.loc[(viajes.travel_speed==np.inf)|(viajes.travel_speed>=50), 'travel_speed'] = np.nan
    else:
        etapas['travel_time_min'] = np.nan
        etapas['travel_speed'] = np.nan
        viajes['travel_time_min'] = np.nan
        viajes['travel_speed'] = np.nan

    viajes['transferencia'] = 0
    viajes.loc[viajes.cant_etapas > 1, 'transferencia'] = 1
    viajes['rango_hora'] = '0-12'
    viajes.loc[(viajes.hora >= 13) & (
        viajes.hora <= 16), 'rango_hora'] = '13-16'
    viajes.loc[(viajes.hora >= 17) & (
        viajes.hora <= 24), 'rango_hora'] = '17-24'

    viajes['distancia'] = 'Viajes cortos (<=5kms)'
    viajes.loc[(viajes.distance_osm_drive > 5),
               'distancia'] = 'Viajes largos (>5kms)'

    viajes['tipo_dia_'] = pd.to_datetime(viajes.dia).dt.weekday.astype(str).copy()
    viajes['tipo_dia'] = 'Hábil'
    viajes.loc[viajes.tipo_dia_.astype(int)>=5, 'tipo_dia'] = 'Fin de Semana'
    viajes = viajes.drop(['tipo_dia_'], axis=1)
    
    viajes['mes'] = viajes.dia.str[:7]

    viajes['Fecha'] = viajes['dia'] + ' ' +viajes['tiempo']
    viajes['Fecha'] = pd.to_datetime(viajes['Fecha'])
    viajes['Fecha_next'] = viajes.groupby(['dia', 'id_tarjeta'])['Fecha'].shift(-1)
    viajes['diff_time'] = viajes['Fecha_next'] - viajes['Fecha']
    viajes['diff_time'] = (viajes.diff_time.dt.seconds / 60).round()

    
    etapas['tipo_dia_'] = pd.to_datetime(etapas.dia).dt.weekday.astype(str).copy()
    etapas['tipo_dia'] = 'Hábil'
    etapas.loc[etapas.tipo_dia_.astype(int)>=5, 'tipo_dia'] = 'Fin de Semana'
    etapas = etapas.drop(['tipo_dia_'], axis=1)
    etapas['mes'] = etapas.dia.str[:7]

    return etapas, viajes


def format_values(row):
    if row['type_val'] == 'int':
        return f"{int(row['Valor']):,}".replace(',', '.')
    elif row['type_val'] == 'float':
        return f"{row['Valor']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    elif row['type_val'] == 'percentage':
        return f"{row['Valor']:.2f}%".replace('.', ',')
    else:
        return str(row['Valor'])


def format_dataframe(df):
    df['Valor_str'] = df.apply(format_values, axis=1)
    return df


def construyo_indicadores(viajes):

    if 'id_polygon' not in viajes.columns:
        viajes['id_polygon'] = 'NONE'


    ind1 = viajes.groupby(['id_polygon'], as_index=False).factor_expansion_linea.sum(
    ).round(0).rename(columns={'factor_expansion_linea': 'Valor'})
    ind1['Indicador'] = 'Cantidad de Viajes'
    ind1['Valor'] = ind1.Valor.astype(int)
    ind1['Tipo'] = 'General'
    ind1['type_val'] = 'int'

    ind2 = viajes[viajes.transferencia == 1].groupby(['id_polygon'], as_index=False).factor_expansion_linea.sum(
    ).round(0).rename(columns={'factor_expansion_linea': 'Valor'})
    ind2['Indicador'] = 'Cantidad de Viajes con Transferencia'
    ind2 = ind2.merge(ind1[['id_polygon', 'Valor']].rename(
        columns={'Valor': 'Tot'}), how='left')
    ind2['Valor'] = (ind2['Valor'] / ind2['Tot'] * 100).round(2)
    ind2['Tipo'] = 'General'
    ind2['type_val'] = 'percentage'

    ind3 = viajes.groupby(['id_polygon', 'rango_hora'], as_index=False).factor_expansion_linea.sum(
    ).round(0).rename(columns={'factor_expansion_linea': 'Valor'})
    ind3['Indicador'] = 'Cantidad de Según Rango Horas'
    ind3['Tot'] = ind3.groupby(['id_polygon']).Valor.transform('sum')
    ind3['Valor'] = (ind3['Valor'] / ind3['Tot'] * 100).round(2)
    ind3['Indicador'] = 'Cantidad de Viajes de '+ind3['rango_hora']+'hs'
    ind3['Tipo'] = 'General'
    ind3['type_val'] = 'percentage'

    ind4 = viajes.groupby(['id_polygon', 'modo'], as_index=False).factor_expansion_linea.sum(
    ).round(0).rename(columns={'factor_expansion_linea': 'Valor'})
    ind4['Indicador'] = 'Partición Modal'
    ind4['Tot'] = ind4.groupby(['id_polygon']).Valor.transform('sum')
    ind4['Valor'] = (ind4['Valor'] / ind4['Tot'] * 100).round(2)
    ind4 = ind4.sort_values(['id_polygon', 'Valor'], ascending=False)
    ind4['Indicador'] = ind4['modo']
    ind4['Tipo'] = 'Modal'
    ind4['type_val'] = 'percentage'

    ind9 = viajes.groupby(['id_polygon', 'distancia'], as_index=False).factor_expansion_linea.sum(
    ).round(0).rename(columns={'factor_expansion_linea': 'Valor'})
    ind9['Indicador'] = 'Partición Modal'
    ind9['Tot'] = ind9.groupby(['id_polygon']).Valor.transform('sum')
    ind9['Valor'] = (ind9['Valor'] / ind9['Tot'] * 100).round(2)
    ind9 = ind9.sort_values(['id_polygon', 'Valor'], ascending=False)
    ind9['Indicador'] = 'Cantidad de '+ind9['distancia']
    ind9['Tipo'] = 'General'
    ind9['type_val'] = 'percentage'

    ind5 = viajes.groupby(['id_polygon', 'id_tarjeta'], as_index=False).factor_expansion_linea.first().groupby(
        ['id_polygon'], as_index=False).factor_expansion_linea.sum().round(0).rename(columns={'factor_expansion_linea': 'Valor'})
    ind5['Indicador'] = 'Cantidad de Usuarios'
    ind5['Tipo'] = 'General'
    ind5['type_val'] = 'int'

    ind6 = calculate_weighted_means(viajes,
                                    aggregate_cols=['id_polygon'],
                                    weighted_mean_cols=['distance_osm_drive'],
                                    weight_col='factor_expansion_linea').round(2).rename(columns={'distance_osm_drive': 'Valor'})
    ind6['Tipo'] = 'Distancias'
    ind6['Indicador'] = 'Distancia Promedio (kms)'
    ind6['type_val'] = 'float'

    ind7 = calculate_weighted_means(viajes,
                                    aggregate_cols=['id_polygon', 'modo'],
                                    weighted_mean_cols=['distance_osm_drive'],
                                    weight_col='factor_expansion_linea').round(2).rename(columns={'distance_osm_drive': 'Valor'})
    ind7['Tipo'] = 'Distancias'
    ind7['Indicador'] = 'Distancia Promedio (' + ind7.modo + ') (kms)'
    ind7['type_val'] = 'float'

    ind8 = calculate_weighted_means(viajes,
                                    aggregate_cols=['id_polygon', 'distancia'],
                                    weighted_mean_cols=['distance_osm_drive'],
                                    weight_col='factor_expansion_linea').round(2).rename(columns={'distance_osm_drive': 'Valor'})
    ind8['Tipo'] = 'Distancias'
    ind8['Indicador'] = 'Distancia Promedio ' + ind8.distancia
    ind8['type_val'] = 'float'

    indicadores = pd.concat(
        [ind1, ind5, ind2, ind3, ind6, ind9, ind7, ind8, ind4])
    indicadores = format_dataframe(indicadores)
    indicadores = indicadores[['id_polygon', 'Tipo', 'Indicador', 'Valor_str']].rename(
        columns={'Valor_str': 'Valor'})

    indicadores = indicadores.sort_values(['id_polygon', 'Tipo'])

    return indicadores


def select_h3_from_polygon(poly, res=8, spacing=.0001, viz=False):
    """
    Fill a polygon with points spaced at the given distance apart.
    Create hexagons that correspond to the polygon
    """

    if 'id' not in poly.columns:
        poly = poly.reset_index().rename(columns={'index': 'id'})

    points_result = pd.DataFrame([])
    poly = poly.reset_index(drop=True).to_crs(4326)
    for i, row in poly.iterrows():

        polygon = poly.geometry[i]

        # Get the bounding box coordinates
        minx, miny, maxx, maxy = polygon.buffer(.008).bounds

        # Create a meshgrid of x and y values based on the spacing
        x_coords = list(np.arange(minx, maxx, spacing))
        y_coords = list(np.arange(miny, maxy, spacing))

        points = []
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                # if polygon.contains(point):
                points.append(point)

        points = gpd.GeoDataFrame(geometry=points, crs=4326)
        points['polygon_number'] = row.id
        points_result = pd.concat([points_result, points])

    points_result = gpd.sjoin(points_result, poly)

    points_result['h3'] = points_result.apply(
        point_to_h3, axis=1, resolution=res)

    points_result = points_result.groupby(['polygon_number', 'h3'], as_index=False).size(
    ).drop(['size'], axis=1).rename(columns={'h3_index': 'h3'})

    gdf_hexs = h3_to_geodataframe(points_result.h3).rename(
        columns={'h3_index': 'h3'})
    gdf_hexs = gdf_hexs.merge(points_result, on='h3')[['polygon_number', 'h3', 'geometry']].sort_values(
        ['polygon_number', 'h3']).reset_index(drop=True)

    if viz:
        ax = poly.boundary.plot(linewidth=1.5, figsize=(15, 15))
        # gdf_points.plot(ax=ax, alpha=.2)
        gdf_hexs.plot(ax=ax, alpha=.6)

    return gdf_hexs.rename(columns={'polygon_number': 'id'})


def select_cases_from_polygons(etapas, viajes, polygons, res=8):
    '''
    Dado un dataframe de polígonos, selecciona los casos de etapas y viajes que se encuentran dentro del polígono
    '''
    print('Selecciona casos dentro de polígonos')
    etapas_selec = pd.DataFrame([])
    viajes_selec = pd.DataFrame([])
    gdf_hexs_all = pd.DataFrame([])

    for i, row in polygons.iterrows():

        poly = polygons[polygons.id == row.id].copy()

        gdf_hexs = select_h3_from_polygon(poly,
                                          res=res,
                                          viz=False)

        gdf_hexs = gdf_hexs[['id', 'h3']].rename(
            columns={'h3': 'h3_o', 'id': 'id_polygon'})

        seleccionar = etapas.merge(gdf_hexs, on='h3_o')[
            ['dia', 'id_tarjeta', 'id_viaje', 'id_polygon']].drop_duplicates()

        tmp = etapas.merge(seleccionar)

        etapas_selec = pd.concat([etapas_selec,
                                  tmp], ignore_index=True)

        tmp = viajes.merge(seleccionar)
        viajes_selec = pd.concat([viajes_selec,
                                  tmp], ignore_index=True)

        gdf_hexs['polygon_lon'] = poly.representative_point().x.values[0]
        gdf_hexs['polygon_lat'] = poly.representative_point().y.values[0]

        gdf_hexs_all = pd.concat([gdf_hexs_all, gdf_hexs],
                                 ignore_index=True)

    return etapas_selec, viajes_selec, polygons, gdf_hexs_all


def agrupar_viajes(etapas_agrupadas,
                   aggregate_cols,
                   weighted_mean_cols,
                   weight_col,
                   zero_to_nan,
                   agg_transferencias=False,
                   agg_modo=False,
                   agg_hora=False,
                   agg_distancia=False,
                   agg_genero=False,
                   agg_tarifa=False):

    etapas_agrupadas_zon = etapas_agrupadas.copy()

    if agg_transferencias:
        etapas_agrupadas_zon['transferencia'] = 99
    if agg_modo:
        etapas_agrupadas_zon['modo_agregado'] = 99
    if agg_hora:
        etapas_agrupadas_zon['rango_hora'] = 99
    if agg_distancia:
        etapas_agrupadas_zon['distancia'] = 99
    if agg_genero:
        etapas_agrupadas_zon['genero'] = 99
    if agg_tarifa:
        etapas_agrupadas_zon['tarifa'] = 99

    etapas_agrupadas_zon = calculate_weighted_means(etapas_agrupadas_zon,
                                                    aggregate_cols=aggregate_cols,
                                                    weighted_mean_cols=weighted_mean_cols,
                                                    weight_col=weight_col,
                                                    zero_to_nan=zero_to_nan)

    return etapas_agrupadas_zon


def construyo_matrices(etapas_desagrupadas,
                       aggregate_cols,
                       zonificaciones,
                       agg_transferencias=False,
                       agg_modo=False,
                       agg_hora=False,
                       agg_distancia=False,
                       agg_genero=False,
                       agg_tarifa=False
                       ):

    matriz = etapas_desagrupadas.copy()

    if agg_transferencias:
        matriz['transferencia'] = 99
    if agg_modo:
        matriz['modo_agregado'] = 99
    if agg_hora:
        matriz['rango_hora'] = 99
    if agg_distancia:
        matriz['distancia'] = 99
    if agg_genero:
        matriz['genero'] = 99
    if agg_tarifa:
        matriz['tarifa'] = 99

    matriz = calculate_weighted_means(matriz,
                                      aggregate_cols=aggregate_cols,
                                      weighted_mean_cols=[
                                          'lat1', 'lon1', 'lat4', 'lon4', 'distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                      weight_col='factor_expansion_linea',
                                      zero_to_nan=[
                                          'lat1', 'lon1', 'lat4', 'lon4'],
                                      )

    zonificaciones['orden'] = zonificaciones['orden'].fillna(0)
    matriz = matriz.merge(
        zonificaciones[['zona', 'id', 'orden']].rename(
            columns={'id': 'inicio', 'orden': 'orden_origen'}),
        on=['zona', 'inicio'])

    matriz = matriz.merge(
        zonificaciones[['zona', 'id', 'orden']].rename(
            columns={'id': 'fin', 'orden': 'orden_destino'}),
        on=['zona', 'fin'])

    matriz['Origen'] = matriz.orden_origen.astype(
        int).astype(str).str.zfill(3)+'_'+matriz.inicio
    matriz['Destino'] = matriz.orden_destino.astype(
        int).astype(str).str.zfill(3)+'_'+matriz.fin

    return matriz


def creo_h3_equivalencias(polygons_h3, polygon, res, zonificaciones):

    poly_sel = h3_to_geodataframe(polygons_h3, 'h3_o')
    poly_sel_all = pd.DataFrame([])

    if 'res_' in res:
        # for i in res:
        if True:
            resol = int(res.replace('res_', ''))
            i = f'res_{resol}'
            poly_sel = poly_sel[['h3_o', 'geometry']].copy()
            poly_sel[f'zona_{i}'] = poly_sel['h3_o'].apply(
                h3toparent, res=resol)
            poly_2 = h3_to_geodataframe(poly_sel, f'zona_{i}')
            poly_ovl = gpd.overlay(
                poly_sel[['h3_o', 'geometry']], poly_2, how='intersection', keep_geom_type=False)
            poly_ovl = poly_ovl.dissolve(by=f'zona_{i}', as_index=False)
            poly_ovl = gpd.overlay(
                poly_ovl, polygon[['geometry']], how='intersection', keep_geom_type=False)
            poly_ovl[f'lat_res_{resol}'] = poly_ovl.geometry.to_crs(
                4326).representative_point().y
            poly_ovl[f'lon_res_{resol}'] = poly_ovl.geometry.to_crs(
                4326).representative_point().x
            poly_ovl = poly_ovl.drop(['geometry', 'h3_o'], axis=1)
            poly_sel = poly_sel.merge(poly_ovl, on=f'zona_{i}', how='left')

            if len(poly_sel_all) == 0:
                poly_sel_all = poly_sel.copy()
            else:
                poly_sel = poly_sel.drop(['geometry'], axis=1)
                poly_sel_all = poly_sel_all.merge(poly_sel, on='h3_o')

    else:

        poly_sel = h3_to_geodataframe(polygons_h3, 'h3_o')
        for zonas in zonificaciones.zona.unique():
            zona = zonificaciones[zonificaciones.zona == zonas]
            poly_ovl = gpd.overlay(
                poly_sel[['h3_o', 'geometry']], zona, how='intersection', keep_geom_type=False)
            poly_ovl_agg = poly_ovl.dissolve(by='id', as_index=False)
            poly_ovl_agg = gpd.overlay(
                poly_ovl_agg, polygon[['geometry']], how='intersection', keep_geom_type=False)

            poly_ovl_agg[f'lat_{zonas}'] = poly_ovl.geometry.to_crs(
                4326).representative_point().y
            poly_ovl_agg[f'lon_{zonas}'] = poly_ovl.geometry.to_crs(
                4326).representative_point().x
            poly_ovl_agg[f'zona_{zonas}'] = poly_ovl_agg.id

            poly_ovl_agg['geometry'] = poly_ovl_agg.geometry.representative_point()

            poly_ovl_agg[f'lat_{zonas}'] = poly_ovl_agg.geometry.y
            poly_ovl_agg[f'lon_{zonas}'] = poly_ovl_agg.geometry.x
            poly_ovl_agg[f'zona_{zonas}'] = poly_ovl_agg.id

            poly_ovl = poly_ovl.merge(poly_ovl_agg[['id', f'zona_{zonas}', f'lat_{zonas}', f'lon_{zonas}']],
                                      on=f'id',
                                      how='left')

            if len(poly_sel_all) == 0:
                poly_sel_all = poly_ovl.copy()
            else:

                poly_sel_all = poly_sel_all.merge(
                    poly_ovl[['h3_o', f'zona_{zonas}', f'lat_{zonas}', f'lon_{zonas}']], on='h3_o', how='left')

    return poly_sel_all


def determinar_modo_agregado(grupo):
    modos_unicos = grupo['modo'].unique()

    if len(modos_unicos) == 1:  # Solo un modo en todo el viaje
        if len(grupo) > 1:  # Más de una etapa
            return f"multietapa ({modos_unicos[0]})"
        else:
            return modos_unicos[0]
    else:
        return "multimodal"


def normalizo_zona(df, zonificaciones):
    if len(zonificaciones) > 0:
        cols = df.columns

        zonificaciones['latlon'] = zonificaciones.geometry.representative_point().y.astype(
            str) + ', '+zonificaciones.geometry.representative_point().x.astype(str)
        zonificaciones['aux'] = 1

        zonificacion_tmp1 = zonificaciones[['id', 'aux', 'geometry']].rename(columns={
                                                                             'id': 'tmp_o'})
        zonificacion_tmp1['geometry'] = zonificacion_tmp1['geometry'].representative_point(
        )
        zonificacion_tmp1['h3_o'] = zonificacion_tmp1.apply(
            point_to_h3, axis=1, resolution=8)
        zonificacion_tmp1['lat_o'] = zonificacion_tmp1.geometry.y
        zonificacion_tmp1['lon_o'] = zonificacion_tmp1.geometry.x
        zonificacion_tmp1 = zonificacion_tmp1.drop(['geometry'], axis=1)

        zonificacion_tmp2 = zonificaciones[['id', 'aux', 'geometry']].rename(columns={
                                                                             'id': 'tmp_d'})
        zonificacion_tmp2['geometry'] = zonificacion_tmp2['geometry'].representative_point(
        )
        zonificacion_tmp2['h3_d'] = zonificacion_tmp2.apply(
            point_to_h3, axis=1, resolution=8)
        zonificacion_tmp1['lat_d'] = zonificacion_tmp2.geometry.y
        zonificacion_tmp1['lon_d'] = zonificacion_tmp2.geometry.x
        zonificacion_tmp2 = zonificacion_tmp2.drop(['geometry'], axis=1)

        zonificacion_tmp = zonificacion_tmp1.merge(zonificacion_tmp2, on='aux')
        zonificacion_tmp = normalizo_lat_lon(
            zonificacion_tmp, h3_o='h3_o', h3_d='h3_d')
        zonificacion_tmp = zonificacion_tmp[[
            'tmp_o', 'tmp_d', 'h3_o', 'h3_d', 'h3_o_norm', 'h3_d_norm']]
        zonificacion_tmp1 = zonificacion_tmp[zonificacion_tmp.h3_o ==
                                             zonificacion_tmp.h3_o_norm].copy()
        zonificacion_tmp1['tmp_o_norm'] = zonificacion_tmp1['tmp_o']
        zonificacion_tmp1['tmp_d_norm'] = zonificacion_tmp1['tmp_d']
        zonificacion_tmp2 = zonificacion_tmp[zonificacion_tmp.h3_o !=
                                             zonificacion_tmp.h3_o_norm].copy()
        zonificacion_tmp2['tmp_o_norm'] = zonificacion_tmp2['tmp_d']
        zonificacion_tmp2['tmp_d_norm'] = zonificacion_tmp2['tmp_o']
        zonificacion_tmp = pd.concat(
            [zonificacion_tmp1, zonificacion_tmp2], ignore_index=True)
        zonificacion_tmp = zonificacion_tmp[['tmp_o', 'tmp_d', 'tmp_o_norm', 'tmp_d_norm']].rename(columns={'tmp_o': 'inicio_norm',
                                                                                                            'tmp_d': 'fin_norm'})

        df = df.merge(zonificacion_tmp, how='left',
                      on=['inicio_norm', 'fin_norm'])
        tmp1 = df[df.inicio_norm == df.tmp_o_norm]
        tmp2 = df[df.inicio_norm != df.tmp_o_norm]
        tmp2 = tmp2.rename(columns={'inicio_norm': 'fin_norm',
                                    'fin_norm': 'inicio_norm',
                                    'poly_inicio_norm': 'poly_fin_norm',
                                    'poly_fin_norm': 'poly_inicio_norm',
                                    'lat1_norm': 'lat4_norm',
                                    'lon1_norm': 'lon4_norm',
                                    'lat4_norm': 'lat1_norm',
                                    'lon4_norm': 'lon1_norm',
                                    })
        tmp2_a = tmp2.loc[tmp2.transfer2_norm == '']
        tmp2_b = tmp2.loc[tmp2.transfer2_norm != '']
        tmp2_b = tmp2_b.rename(columns={'transfer1_norm': 'transfer2_norm',
                                        'transfer2_norm': 'transfer1_norm',
                                        'poly_transfer1_norm': 'poly_transfer2_norm',
                                        'poly_transfer2_norm': 'poly_transfer1_norm',
                                        'lat2_norm': 'lat3_norm',
                                        'lon2_norm': 'lon3_norm',
                                        'lat3_norm': 'lat2_norm',
                                        'lon3_norm': 'lon2_norm', })

        tmp1 = tmp1[cols]
        tmp2_a = tmp2_a[cols]
        tmp2_b = tmp2_b[cols]

        df = pd.concat([tmp1, tmp2_a, tmp2_b], ignore_index=True)
    return df


def crea_socio_indicadores(etapas, viajes):
    print('Creo indicadores de género y tarifa')

    socio_indicadores = pd.DataFrame([])
    
    viajesx = calculate_weighted_means(viajes,
                                aggregate_cols=['mes', 'dia', 'tipo_dia', 'genero', 'tarifa'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed', 'cant_etapas', 'diff_time'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True)
    
    viajesx = calculate_weighted_means(viajesx,
                                aggregate_cols=['mes', 'tipo_dia', 'genero', 'tarifa'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed', 'cant_etapas', 'diff_time'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=False).round(3)

    
    etapasx = calculate_weighted_means(etapas,
                                aggregate_cols=['mes', 'dia', 'tipo_dia', 'genero', 'tarifa', 'modo'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True)
    
    etapasx = calculate_weighted_means(etapasx,
                                aggregate_cols=['mes', 'tipo_dia', 'genero', 'tarifa', 'modo'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=False).round(3)
    
    # calcular tabla de indicadores
    etapasxx = calculate_weighted_means(etapasx,
                                aggregate_cols=['mes', 'tipo_dia', 'genero', 'modo'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True).round(3)
    
    etapasxx['tabla'] = 'etapas-genero-modo'
    socio_indicadores = pd.concat([socio_indicadores, etapasxx], ignore_index=True)
    
    etapasxx = calculate_weighted_means(etapasx,
                                aggregate_cols=['mes', 'tipo_dia', 'tarifa', 'modo'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True).round(3)
    
    etapasxx['tabla'] = 'etapas-tarifa-modo'
    socio_indicadores = pd.concat([socio_indicadores, etapasxx], ignore_index=True)
    
    
    viajesxx = calculate_weighted_means(viajesx,
                                aggregate_cols=['mes', 'tipo_dia', 'genero', 'tarifa'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed', 'cant_etapas', 'diff_time'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True).round(3)
    
    viajesxx['tabla'] = 'viajes-genero-tarifa'
    socio_indicadores = pd.concat([socio_indicadores, viajesxx], ignore_index=True)
    
    # Calculo viajes promedio por día por género y tarifa
    userx = viajes.copy()
    userx['tarifa'] = userx['tarifa'].str.replace('-', '')
    userx = userx.groupby(['dia', 'id_tarjeta'])['tarifa'].apply(lambda x: '-'.join(x.unique())).reset_index()
    userx.loc[userx.tarifa.str[-1] == '-', 'tarifa'] = userx.loc[userx.tarifa.str[-1] == '-', :].tarifa.str[:-1]
    userx.loc[userx.tarifa.str[:1] == '-', 'tarifa'] = userx.loc[userx.tarifa.str[:1] == '-', :].tarifa.str[1:]
    userx = userx.rename(columns={'tarifa': 'tarifa_agg'})
    userx.loc[userx.tarifa_agg=='', 'tarifa_agg'] = '-'
    userx = viajes.merge(userx, how='left')
    userx = userx.groupby(['mes', 'dia', 'tipo_dia', 'id_tarjeta', 'genero', 'tarifa_agg'], as_index=False).agg({'factor_expansion_tarjeta':'count', 'factor_expansion_linea':'mean'}).rename(columns={'factor_expansion_tarjeta':'cant_viajes'}).rename(columns={'tarifa_agg':'tarifa'})
    
    userx = calculate_weighted_means(userx,
                                    aggregate_cols=['dia', 'mes', 'tipo_dia', 'genero', 'tarifa'],
                                    weighted_mean_cols=['cant_viajes'],
                                    weight_col='factor_expansion_linea',                                
                                    var_fex_summed=True).round(3)
    
    userx = calculate_weighted_means(userx,
                                    aggregate_cols=['mes', 'tipo_dia', 'genero', 'tarifa'],
                                    weighted_mean_cols=['cant_viajes'],
                                    weight_col='factor_expansion_linea',                                
                                    var_fex_summed=False).round(3)
    
    userx['tabla'] = 'usuario-genero-tarifa'
    socio_indicadores = pd.concat([socio_indicadores, userx], ignore_index=True)
    
    
    # Preparo socioindicadores final
    socio_indicadores = socio_indicadores[['tabla', 'mes', 'tipo_dia', 'genero', 'tarifa', 'modo', 'distance_osm_drive', 'travel_time_min', 'travel_speed', 'cant_etapas', 'cant_viajes', 'diff_time', 'factor_expansion_linea']]
    socio_indicadores.columns = ['tabla', 'mes', 'tipo_dia', 'Genero', 'Tarifa', 'Modo', 'Distancia', 'Tiempo de viaje', 'Velocidad', 'Etapas promedio', 'Viajes promedio', 'Tiempo entre viajes', 'factor_expansion_linea']


    return socio_indicadores


def preparo_lineas_deseo(etapas_selec, viajes_selec, polygons_h3='', poligonos='', res=6):

    print('Preparo líneas de deseo')
    zonificaciones = levanto_tabla_sql('zonificaciones')

    if len(polygons_h3) == 0:
        id_polygon = 'NONE'
        polygons_h3 = pd.DataFrame([['NONE']], columns=['id_polygon'])
        poligonos = pd.DataFrame([['NONE', 'NONE']], columns=['id', 'tipo'])
        etapas_selec['id_polygon'] = 'NONE'
        viajes_selec['id_polygon'] = 'NONE'

    etapas_selec = etapas_selec[etapas_selec.distance_osm_drive.notna()].copy()
    etapas_selec = etapas_selec.rename(
        columns={'distance_osm_drive': 'distance_osm_drive_etapas'})
    distancias_all = etapas_selec.groupby(
        ['id_polygon', 'dia', 'id_tarjeta', 'id_viaje'], as_index=False)[['distance_osm_drive_etapas']].sum()
    distancias_all = distancias_all.merge(viajes_selec.loc[viajes_selec.od_validado == 1, [
                                          'id_polygon', 'dia', 'id_tarjeta', 'id_viaje', 'distance_osm_drive', 'travel_time_min', 'travel_speed']])
    distancias_all['distancia'] = 'Viajes cortos (<=5kms)'
    distancias_all.loc[(distancias_all.distance_osm_drive > 5),
                       'distancia'] = 'Viajes largos (>5kms)'

    # Agrupamos por 'dia', 'id_tarjeta' e 'id_viaje' y aplicamos la función para determinar la partición modal
    viajes_modo_agg = etapas_selec.groupby(['dia',
                                            'id_tarjeta',
                                            'id_viaje']).apply(
        determinar_modo_agregado).reset_index(name='modo_agregado').sort_values(['dia',
                                                                                 'id_tarjeta',
                                                                                 'id_viaje']).reset_index(drop=True)

    etapas_selec = etapas_selec.merge(viajes_modo_agg, how='left')

    # Traigo zonas
    zonas_data, zonas_cols = traigo_tabla_zonas()

    if type(res) == int:
        res = [res]
    for i in res:
        res = [f'res_{i}']
        h3_vals = pd.concat([etapas_selec.loc[etapas_selec.h3_o.notna(),
                                              ['h3_o']].rename(columns={'h3_o': 'h3'}),
                             etapas_selec.loc[etapas_selec.h3_d.notna(),
                                              ['h3_d']].rename(columns={'h3_d': 'h3'})]).drop_duplicates()
        h3_vals['h3_res'] = h3_vals['h3'].apply(h3toparent, res=i)
        h3_zona = create_h3_gdf(h3_vals.h3_res.tolist()).rename(
            columns={'hexagon_id': 'id'}).drop_duplicates()
        h3_zona['zona'] = res[0]
        zonificaciones = pd.concat(
            [zonificaciones, h3_zona], ignore_index=True)

    resol = res[0]
    zonas = res + zonas_cols

    etapas_selec['rango_hora'] = '0-12'
    etapas_selec.loc[(etapas_selec.hora >= 13) & (
        etapas_selec.hora <= 16), 'rango_hora'] = '13-16'
    etapas_selec.loc[(etapas_selec.hora >= 17) & (
        etapas_selec.hora <= 24), 'rango_hora'] = '17-24'

    etapas_agrupadas_all = pd.DataFrame([])
    gpd_viajes_agrupados_all = pd.DataFrame([])

    for id_polygon in polygons_h3.id_polygon.unique():

        poly_h3 = polygons_h3[polygons_h3.id_polygon == id_polygon]
        poly = poligonos[poligonos.id == id_polygon]
        tipo_poly = poly.tipo.values[0]
        print('')
        print(f'Polígono {id_polygon} - Tipo: {tipo_poly}')

        # Preparo Etapas con inicio, transferencias y fin del viaje
        etapas_all = etapas_selec.loc[(etapas_selec.id_polygon == id_polygon), ['dia',
                                                                                'id_tarjeta',
                                                                                'id_viaje',
                                                                                'id_etapa',
                                                                                'h3_o',
                                                                                'h3_d',
                                                                                'modo_agregado',
                                                                                'rango_hora',
                                                                                'genero',
                                                                                'tarifa',
                                                                                'factor_expansion_linea']]
        etapas_all['etapa_max'] = etapas_all.groupby(
            ['dia', 'id_tarjeta', 'id_viaje']).id_etapa.transform('max')

        # Borro los casos que tienen 3 transferencias o más
        if len(etapas_all[etapas_all.etapa_max > 3]) > 0:
            nborrar = len(etapas_all[etapas_all.etapa_max > 3][['id_tarjeta',
                                                                'id_viaje']].value_counts()) / len(etapas_all[['id_tarjeta',
                                                                                                               'id_viaje']].value_counts()) * 100
            print(
                f'Se van a borrar los viajes que tienen más de 3 etapas, representan el {round(nborrar,2)}% de los viajes para el polígono {id_polygon}')
            etapas_all = etapas_all[etapas_all.etapa_max <= 3].copy()

        etapas_all['ultimo_viaje'] = 0
        etapas_all.loc[etapas_all.etapa_max ==
                       etapas_all.id_etapa, 'ultimo_viaje'] = 1

        ultimo_viaje = etapas_all[etapas_all.ultimo_viaje == 1]

        etapas_all['h3'] = etapas_all['h3_o']
        etapas_all = etapas_all[['dia',
                                 'id_tarjeta',
                                 'id_viaje',
                                 'id_etapa',
                                 'h3',
                                 'modo_agregado',
                                 'rango_hora',
                                 'genero', 
                                 'tarifa',
                                 'factor_expansion_linea']]
        etapas_all['ultimo_viaje'] = 0

        ultimo_viaje['h3'] = ultimo_viaje['h3_d']
        ultimo_viaje['id_etapa'] += 1
        ultimo_viaje = ultimo_viaje[['dia',
                                     'id_tarjeta',
                                     'id_viaje',
                                     'id_etapa',
                                     'h3',
                                     'modo_agregado',
                                     'rango_hora',
                                     'genero', 
                                     'tarifa',
                                     'factor_expansion_linea',
                                     'ultimo_viaje']]

        etapas_all = pd.concat([etapas_all, ultimo_viaje]).sort_values(
            ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa']).reset_index(drop=True)

        etapas_all['tipo_viaje'] = 'Transfer_' + \
            (etapas_all['id_etapa']-1).astype(str)
        etapas_all.loc[etapas_all.ultimo_viaje == 1, 'tipo_viaje'] = 'Fin'
        etapas_all.loc[etapas_all.id_etapa == 1, 'tipo_viaje'] = 'Inicio'

        etapas_all['polygon'] = ''
        if id_polygon != 'NONE':
            etapas_all.loc[etapas_all.h3.isin(
                poly_h3.h3_o.unique()), 'polygon'] = id_polygon

        etapas_all = etapas_all.drop(['ultimo_viaje'], axis=1)

        # Guardo las coordenadas de los H3
        h3_coords = etapas_all.groupby(
            'h3', as_index=False).id_viaje.count().drop(['id_viaje'], axis=1)
        h3_coords[['lat', 'lon']] = h3_coords.h3.apply(h3_to_lat_lon)

        # Preparo cada etapa de viaje para poder hacer la agrupación y tener inicio, transferencias y destino en un mismo registro
        inicio = etapas_all.loc[etapas_all.tipo_viaje == 'Inicio', ['dia',
                                                                    'id_tarjeta',
                                                                    'id_viaje',
                                                                    'h3',
                                                                    'modo_agregado',
                                                                    'rango_hora',
                                                                    'genero',
                                                                    'tarifa',
                                                                    'factor_expansion_linea',
                                                                    'polygon']].rename(columns={'h3': 'h3_inicio',
                                                                                                'polygon': 'poly_inicio'})
        
        
        fin = etapas_all.loc[etapas_all.tipo_viaje == 'Fin', ['dia',
                                                              'id_tarjeta',
                                                              'id_viaje',
                                                              'h3',
                                                              'polygon']].rename(columns={'h3': 'h3_fin',
                                                                                          'polygon': 'poly_fin'})
        transfer1 = etapas_all.loc[etapas_all.tipo_viaje == 'Transfer_1', ['dia',
                                                                           'id_tarjeta',
                                                                           'id_viaje',
                                                                           'h3',
                                                                           'polygon']].rename(columns={'h3': 'h3_transfer1', 'polygon': 'poly_transfer1'})
        transfer2 = etapas_all.loc[etapas_all.tipo_viaje == 'Transfer_2', ['dia',
                                                                           'id_tarjeta',
                                                                           'id_viaje',
                                                                           'h3',
                                                                           'polygon']].rename(columns={'h3': 'h3_transfer2',
                                                                                                       'polygon': 'poly_transfer2'})
        

        
        etapas_agrupadas = inicio.merge(transfer1, how='left').merge(
            transfer2, how='left').merge(fin, how='left').fillna('')

        etapas_agrupadas = etapas_agrupadas[['dia',
                                             'id_tarjeta',
                                             'id_viaje',
                                             'h3_inicio',
                                             'h3_transfer1',
                                             'h3_transfer2',
                                             'h3_fin',
                                             'poly_inicio',
                                             'poly_transfer1',
                                             'poly_transfer2',
                                             'poly_fin',
                                             'modo_agregado',
                                             'rango_hora',
                                             'genero', 
                                             'tarifa',
                                             'factor_expansion_linea']]


        
        for zona in zonas:

            if id_polygon != 'NONE':
                # print(id_polygon, zona)
                h3_equivalencias = creo_h3_equivalencias(polygons_h3[polygons_h3.id_polygon == id_polygon].copy(),
                                                         poligonos[poligonos.id ==
                                                                   id_polygon],
                                                         zona,
                                                         zonificaciones[zonificaciones.zona == zona].copy())

            # Preparo para agrupar por líneas de deseo y cambiar de resolución si es necesario
            etapas_agrupadas_zon = etapas_agrupadas.copy()

            etapas_agrupadas_zon['id_polygon'] = id_polygon
            etapas_agrupadas_zon['zona'] = zona

            etapas_agrupadas_zon['inicio_norm'] = etapas_agrupadas_zon['h3_inicio']
            etapas_agrupadas_zon['transfer1_norm'] = etapas_agrupadas_zon['h3_transfer1']
            etapas_agrupadas_zon['transfer2_norm'] = etapas_agrupadas_zon['h3_transfer2']
            etapas_agrupadas_zon['fin_norm'] = etapas_agrupadas_zon['h3_fin']
            etapas_agrupadas_zon['poly_inicio_norm'] = etapas_agrupadas_zon['poly_inicio']
            etapas_agrupadas_zon['poly_transfer1_norm'] = etapas_agrupadas_zon['poly_transfer1']
            etapas_agrupadas_zon['poly_transfer2_norm'] = etapas_agrupadas_zon['poly_transfer2']
            etapas_agrupadas_zon['poly_fin_norm'] = etapas_agrupadas_zon['poly_fin']

            n = 1
            for i in ['inicio_norm', 'transfer1_norm', 'transfer2_norm', 'fin_norm']:

                etapas_agrupadas_zon = etapas_agrupadas_zon.merge(
                    h3_coords.rename(columns={'h3': i}), how='left', on=i)
                etapas_agrupadas_zon[f'lon{n}'] = etapas_agrupadas_zon['lon']
                etapas_agrupadas_zon[f'lat{n}'] = etapas_agrupadas_zon['lat']
                etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                    ['lon', 'lat'], axis=1)

                # Selecciono el centroide del polígono en vez del centroide de cada hexágono

                if tipo_poly == 'poligono':
                    etapas_agrupadas_zon.loc[etapas_agrupadas_zon[i].isin(
                        poly_h3.h3_o.unique()), f'lat{n}'] = poly_h3.polygon_lat.mean()
                    etapas_agrupadas_zon.loc[etapas_agrupadas_zon[i].isin(
                        poly_h3.h3_o.unique()), f'lon{n}'] = poly_h3.polygon_lon.mean()

                if f'{i}_ant' not in etapas_agrupadas_zon.columns:
                    etapas_agrupadas_zon[f'{i}_ant'] = etapas_agrupadas_zon[i]

                if 'res_' in zona:
                    resol = int(zona.replace('res_', ''))
                    etapas_agrupadas_zon[i] = etapas_agrupadas_zon[i].apply(
                        h3toparent, res=resol)

                else:
                    zonas_data_ = zonas_data.groupby(
                        ['h3', 'fex', 'latitud', 'longitud'], as_index=False)[zona].first()
                    etapas_agrupadas_zon = etapas_agrupadas_zon.merge(
                        zonas_data_[['h3', zona]].rename(columns={'h3': i, zona: 'zona_tmp'}), how='left')

                    etapas_agrupadas_zon[i] = etapas_agrupadas_zon['zona_tmp']
                    etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                        ['zona_tmp'], axis=1)
                    if len(etapas_agrupadas_zon[(etapas_agrupadas_zon.inicio_norm.isna()) | (etapas_agrupadas_zon.fin_norm.isna())]) > 0:
                        cant_etapas = len(etapas_agrupadas_zon[(etapas_agrupadas_zon.inicio_norm.isna()) | (
                            etapas_agrupadas_zon.fin_norm.isna())])
                        print(
                            f'Hay {cant_etapas} registros a los que no se les pudo asignar {zona}')

                    etapas_agrupadas_zon = etapas_agrupadas_zon[~(
                        (etapas_agrupadas_zon.inicio_norm.isna()) | (etapas_agrupadas_zon.fin_norm.isna()))]

                # Si es cuenca modifico las latitudes longitudes donde coincide el polígono de cuenca con el h3
                if (tipo_poly == 'cuenca'):
                    # reemplazo latitudes y longitudes de cuenca para normalizar
                    poly_var = i.replace('h3_', '').replace('_norm', '')
                    h3_equivalencias_agg = h3_equivalencias.groupby(
                        [f'zona_{zona}', f'lat_{zona}', f'lon_{zona}'], as_index=False).h3_o.count().drop(['h3_o'], axis=1)

                    etapas_agrupadas_zon = etapas_agrupadas_zon.merge(h3_equivalencias_agg[[f'zona_{zona}',
                                                                                            f'lat_{zona}',
                                                                                            f'lon_{zona}']].rename(columns={f'zona_{zona}': i}),
                                                                      how='left', on=i)

                    etapas_agrupadas_zon.loc[(etapas_agrupadas_zon[f'lat_{zona}'].notna())
                                             & (etapas_agrupadas_zon[f'poly_{poly_var}'] != ''),
                                             f'lat{n}'] = etapas_agrupadas_zon.loc[
                        (etapas_agrupadas_zon[f'lat_{zona}'].notna())
                        & (etapas_agrupadas_zon[f'poly_{poly_var}'] != ''),
                        f'lat_{zona}']

                    etapas_agrupadas_zon.loc[(etapas_agrupadas_zon[f'lon_{zona}'].notna())
                                             & (etapas_agrupadas_zon[f'poly_{poly_var}'] != ''),
                                             f'lon{n}'] = etapas_agrupadas_zon.loc[
                        (etapas_agrupadas_zon[f'lon_{zona}'].notna())
                        & (etapas_agrupadas_zon[f'poly_{poly_var}'] != ''),
                        f'lon_{zona}']

                    etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
                        [f'lon_{zona}', f'lat_{zona}'], axis=1)

                etapas_agrupadas_zon[i] = etapas_agrupadas_zon[i].fillna('')
                n += 1

            etapas_agrupadas_zon['inicio'] = etapas_agrupadas_zon['inicio_norm']
            etapas_agrupadas_zon['transfer1'] = etapas_agrupadas_zon['transfer1_norm']
            etapas_agrupadas_zon['transfer2'] = etapas_agrupadas_zon['transfer2_norm']
            etapas_agrupadas_zon['fin'] = etapas_agrupadas_zon['fin_norm']
            etapas_agrupadas_zon['poly_inicio'] = etapas_agrupadas_zon['poly_inicio_norm']
            etapas_agrupadas_zon['poly_transfer1'] = etapas_agrupadas_zon['poly_transfer1_norm']
            etapas_agrupadas_zon['poly_transfer2'] = etapas_agrupadas_zon['poly_transfer2_norm']
            etapas_agrupadas_zon['poly_fin'] = etapas_agrupadas_zon['poly_fin_norm']
            etapas_agrupadas_zon['lat1_norm'] = etapas_agrupadas_zon['lat1']
            etapas_agrupadas_zon['lat2_norm'] = etapas_agrupadas_zon['lat2']
            etapas_agrupadas_zon['lat3_norm'] = etapas_agrupadas_zon['lat3']
            etapas_agrupadas_zon['lat4_norm'] = etapas_agrupadas_zon['lat4']
            etapas_agrupadas_zon['lon1_norm'] = etapas_agrupadas_zon['lon1']
            etapas_agrupadas_zon['lon2_norm'] = etapas_agrupadas_zon['lon2']
            etapas_agrupadas_zon['lon3_norm'] = etapas_agrupadas_zon['lon3']
            etapas_agrupadas_zon['lon4_norm'] = etapas_agrupadas_zon['lon4']

            etapas_agrupadas_zon = normalizo_zona(etapas_agrupadas_zon,
                                                  zonificaciones[zonificaciones.zona == zona].copy())

            etapas_agrupadas_all = pd.concat(
                [etapas_agrupadas_all, etapas_agrupadas_zon], ignore_index=True)

    etapas_agrupadas_all['transferencia'] = 0
    etapas_agrupadas_all.loc[(etapas_agrupadas_all.transfer1_norm != '') | (
        etapas_agrupadas_all.transfer2_norm != ''), 'transferencia'] = 1

    etapas_agrupadas_all = etapas_agrupadas_all.merge(
        distancias_all, how='left')

    etapas_agrupadas_all['tipo_dia_'] = pd.to_datetime(etapas_agrupadas_all.dia).dt.weekday.astype(str).copy()
    etapas_agrupadas_all['tipo_dia'] = 'Hábil'
    etapas_agrupadas_all.loc[etapas_agrupadas_all.tipo_dia_.astype(int)>=5, 'tipo_dia'] = 'Fin de Semana'
    etapas_agrupadas_all = etapas_agrupadas_all.drop(['tipo_dia_'], axis=1)
    etapas_agrupadas_all['mes'] = etapas_agrupadas_all.dia.str[:7]

    etapas_agrupadas_all = etapas_agrupadas_all[['id_polygon', 'zona', 'dia', 'mes', 'tipo_dia', 'id_tarjeta', 'id_viaje',
                                                 'h3_inicio', 'h3_transfer1', 'h3_transfer2', 'h3_fin',
                                                 'inicio', 'transfer1', 'transfer2', 'fin',
                                                 'poly_inicio', 'poly_transfer1', 'poly_transfer2', 'poly_fin',
                                                 'inicio_norm', 'transfer1_norm', 'transfer2_norm', 'fin_norm',
                                                 'poly_inicio_norm', 'poly_transfer1_norm', 'poly_transfer2_norm', 'poly_fin_norm',
                                                 'lon1', 'lat1', 'lon2', 'lat2', 'lon3', 'lat3', 'lon4', 'lat4',
                                                 'lon1_norm', 'lat1_norm', 'lon2_norm', 'lat2_norm', 'lon3_norm', 'lat3_norm', 'lon4_norm', 'lat4_norm',
                                                 'transferencia', 'modo_agregado', 'rango_hora', 'genero', 'tarifa', 'distancia', 
                                                 'distance_osm_drive', 'distance_osm_drive_etapas', 
                                                  'travel_time_min', 'travel_speed',
                                                 'factor_expansion_linea']]

    etapas_sin_agrupar = etapas_agrupadas_all.copy()

    aggregate_cols = ['id_polygon', 'dia', 'mes', 'tipo_dia', 'zona', 'inicio', 'fin', 'poly_inicio',
                      'poly_fin', 'transferencia', 'modo_agregado', 'rango_hora', 'genero', 'tarifa', 'distancia']
    viajes_matrices = construyo_matrices(etapas_sin_agrupar,
                                         aggregate_cols,
                                         zonificaciones,
                                         False,
                                         False,
                                         False)

    # Agrupación de viajes
    aggregate_cols = ['id_polygon',
                      'dia',
                      'mes', 
                      'tipo_dia',
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
                      'genero', 
                      'tarifa',
                      'distancia']

    weighted_mean_cols = ['distance_osm_drive',
                          'distance_osm_drive_etapas',
                          'travel_time_min', 
                          'travel_speed',
                          'lat1_norm',
                          'lon1_norm',
                          'lat2_norm',
                          'lon2_norm',
                          'lat3_norm',
                          'lon3_norm',
                          'lat4_norm',
                          'lon4_norm']

    weight_col = 'factor_expansion_linea'

    zero_to_nan = ['lat1_norm',
                   'lon1_norm',
                   'lat2_norm',
                   'lon2_norm',
                   'lat3_norm',
                   'lon3_norm',
                   'lat4_norm',
                   'lon4_norm']

    etapas_agrupadas_all = agrupar_viajes(etapas_agrupadas_all,
                                          aggregate_cols,
                                          weighted_mean_cols,
                                          weight_col,
                                          zero_to_nan,
                                          agg_transferencias=False,
                                          agg_modo=False,
                                          agg_hora=False,
                                          agg_distancia=False)

    zonificaciones['lat'] = zonificaciones.geometry.representative_point().y
    zonificaciones['lon'] = zonificaciones.geometry.representative_point().x

    n = 1
    poly_lst = ['poly_inicio', 'poly_transfer1', 'poly_transfer2', 'poly_fin']
    for i in ['inicio', 'transfer1', 'transfer2', 'fin']:
        etapas_agrupadas_all = etapas_agrupadas_all.merge(zonificaciones[['zona', 'id', 'lat', 'lon']].rename(columns={'id': f'{i}_norm', 'lat': f'lat{n}_norm_tmp', 'lon': f'lon{n}_norm_tmp'}),
                                                          how='left',
                                                          on=['zona', f'{i}_norm'])
        etapas_agrupadas_all.loc[etapas_agrupadas_all[f'{poly_lst[n-1]}_norm'] == '',
                                 f'lat{n}_norm'] = etapas_agrupadas_all.loc[etapas_agrupadas_all[f'{poly_lst[n-1]}_norm'] == '', f'lat{n}_norm_tmp']
        etapas_agrupadas_all.loc[etapas_agrupadas_all[f'{poly_lst[n-1]}_norm'] == '',
                                 f'lon{n}_norm'] = etapas_agrupadas_all.loc[etapas_agrupadas_all[f'{poly_lst[n-1]}_norm'] == '', f'lon{n}_norm_tmp']

        etapas_agrupadas_all = etapas_agrupadas_all.drop(
            [f'lat{n}_norm_tmp', f'lon{n}_norm_tmp'], axis=1)

        if (n == 1) | (n == 4):
            viajes_matrices = viajes_matrices.merge(zonificaciones[['zona', 'id', 'lat', 'lon']].rename(columns={'id': f'{i}', 'lat': f'lat{n}_tmp', 'lon': f'lon{n}_tmp'}),
                                                    how='left',
                                                    on=['zona', f'{i}'])
            viajes_matrices.loc[viajes_matrices[f'{poly_lst[n-1]}'] == '',
                                f'lat{n}'] = viajes_matrices.loc[viajes_matrices[f'{poly_lst[n-1]}'] == '', f'lat{n}_tmp']
            viajes_matrices.loc[viajes_matrices[f'{poly_lst[n-1]}'] == '',
                                f'lon{n}'] = viajes_matrices.loc[viajes_matrices[f'{poly_lst[n-1]}'] == '', f'lon{n}_tmp']
            viajes_matrices = viajes_matrices.drop(
                [f'lat{n}_tmp', f'lon{n}_tmp'], axis=1)

        n += 1

    if id_polygon == 'NONE':
        etapas_agrupadas_all = etapas_agrupadas_all.drop(['id_polygon',
                                                          'poly_inicio_norm',
                                                          'poly_transfer1_norm',
                                                          'poly_transfer2_norm',
                                                          'poly_fin_norm'], axis=1)

        viajes_matrices = viajes_matrices.drop(
            ['poly_inicio', 'poly_fin'], axis=1)

    # # Agrupar a nivel de mes y corregir factor de expansión
    sum_viajes = etapas_agrupadas_all.groupby(['dia', 'mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.sum().groupby(['mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.mean().round()
        
    aggregate_cols = ['mes', 'tipo_dia', 'zona', 'inicio_norm', 'transfer1_norm','transfer2_norm', 'fin_norm', 'transferencia', 'modo_agregado', 'rango_hora', 'genero', 'tarifa', 'distancia', ]
    weighted_mean_cols=['distance_osm_drive', 'distance_osm_drive_etapas', 'travel_time_min', 'travel_speed', 'lat1_norm', 'lon1_norm', 'lat2_norm', 'lon2_norm', 'lat3_norm', 'lon3_norm', 'lat4_norm', 'lon4_norm']
    
    etapas_agrupadas_all = calculate_weighted_means(etapas_agrupadas_all,
                                aggregate_cols=aggregate_cols,
                                weighted_mean_cols=weighted_mean_cols,
                                weight_col='factor_expansion_linea',
                                zero_to_nan=zero_to_nan,
                                var_fex_summed=False)

    sum_viajes['factor_expansion_linea'] = 1 - (sum_viajes['factor_expansion_linea'] / etapas_agrupadas_all.groupby(['mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.sum().factor_expansion_linea )
    sum_viajes = sum_viajes.rename(columns={'factor_expansion_linea':'factor_correccion'})
    
    etapas_agrupadas_all = etapas_agrupadas_all.merge(sum_viajes)
    etapas_agrupadas_all['factor_expansion_linea2'] = etapas_agrupadas_all['factor_expansion_linea'] * etapas_agrupadas_all['factor_correccion']
    etapas_agrupadas_all['factor_expansion_linea2'] = etapas_agrupadas_all['factor_expansion_linea'] - etapas_agrupadas_all['factor_expansion_linea2']
    etapas_agrupadas_all = etapas_agrupadas_all.drop(['factor_correccion', 'factor_expansion_linea'], axis=1)
    etapas_agrupadas_all = etapas_agrupadas_all.rename(columns={'factor_expansion_linea2':'factor_expansion_linea'})

    # # Agrupar a nivel de mes y corregir factor de expansión
    sum_viajes = viajes_matrices.groupby(['dia', 'mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.sum().groupby(['mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.mean()
    
    aggregate_cols = ['id_polygon', 'mes', 'tipo_dia', 'zona', 'inicio', 'fin', 'transferencia', 'modo_agregado', 'rango_hora', 'genero', 'tarifa', 'distancia', 'orden_origen', 'orden_destino', 'Origen', 'Destino']
    weighted_mean_cols = ['lat1', 'lon1', 'lat4', 'lon4', 'distance_osm_drive', 'travel_time_min', 'travel_speed',]
    zero_to_nan = ['lat1', 'lon1', 'lat4', 'lon4']

    viajes_matrices = calculate_weighted_means(viajes_matrices,
                                aggregate_cols=aggregate_cols,
                                weighted_mean_cols=weighted_mean_cols,
                                weight_col='factor_expansion_linea',
                                zero_to_nan=zero_to_nan,
                                var_fex_summed=False)

    sum_viajes['factor_expansion_linea'] = 1 - (sum_viajes['factor_expansion_linea'] / viajes_matrices.groupby(['mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.sum().factor_expansion_linea )
    sum_viajes = sum_viajes.rename(columns={'factor_expansion_linea':'factor_correccion'})

    viajes_matrices = viajes_matrices.merge(sum_viajes)
    viajes_matrices['factor_expansion_linea2'] = viajes_matrices['factor_expansion_linea'] * viajes_matrices['factor_correccion']
    viajes_matrices['factor_expansion_linea2'] = viajes_matrices['factor_expansion_linea'] - viajes_matrices['factor_expansion_linea2']
    viajes_matrices = viajes_matrices.drop(['factor_correccion', 'factor_expansion_linea'], axis=1)
    viajes_matrices = viajes_matrices.rename(columns={'factor_expansion_linea2':'factor_expansion_linea'})

    return etapas_agrupadas_all, etapas_sin_agrupar, viajes_matrices, zonificaciones


def proceso_poligonos():

    print('Procesa polígonos')

    check_config()

    zonificaciones = levanto_tabla_sql('zonificaciones')

    poligonos = levanto_tabla_sql('poligonos')

    if len(poligonos) > 0:
        print('identifica viajes en polígonos')
        # Read trips and jorneys
        etapas, viajes = load_and_process_data()
        # # Select cases based fron polygon
        etapas_selec, viajes_selec, polygons, polygons_h3 = select_cases_from_polygons(
            etapas[etapas.od_validado == 1], viajes[viajes.od_validado == 1], poligonos, res=8)

        etapas_agrupadas, etapas_sin_agrupar, viajes_matrices, zonificaciones = preparo_lineas_deseo(
            etapas_selec, viajes_selec, polygons_h3, poligonos=poligonos, res=[6])

        indicadores = construyo_indicadores(viajes_selec)

        conn_dash = iniciar_conexion_db(tipo='dash')
        etapas_agrupadas = etapas_agrupadas.fillna(0)
        etapas_agrupadas.to_sql("poly_etapas",
                                conn_dash, if_exists="replace", index=False,)

        viajes_matrices.to_sql("poly_matrices",
                               conn_dash, if_exists="replace", index=False,)

        indicadores.to_sql("poly_indicadores",
                           conn_dash, if_exists="replace", index=False,)

        conn_dash.close()


def proceso_lineas_deseo():

    print('Procesa etapas')

    check_config()

    zonificaciones = levanto_tabla_sql('zonificaciones')
    zonificaciones['lat'] = zonificaciones.geometry.representative_point().y
    zonificaciones['lon'] = zonificaciones.geometry.representative_point().x

    etapas, viajes = load_and_process_data()

    etapas_agrupadas, etapas_sin_agrupar, viajes_matrices, zonificaciones = preparo_lineas_deseo(
        etapas, viajes, res=[6])

    indicadores = construyo_indicadores(viajes)

    socio_indicadores = crea_socio_indicadores(etapas, viajes)

    conn_dash = iniciar_conexion_db(tipo='dash')

    etapas_agrupadas = etapas_agrupadas.fillna(0)
    etapas_agrupadas.to_sql("agg_etapas",
                            conn_dash, if_exists="replace", index=False,)

    viajes_matrices.to_sql("agg_matrices",
                           conn_dash, if_exists="replace", index=False,)

    indicadores.to_sql("agg_indicadores",
                       conn_dash, if_exists="replace", index=False,)

    socio_indicadores.to_sql("socio_indicadores",
                       conn_dash, if_exists="replace", index=False,)


    conn_dash.close()
