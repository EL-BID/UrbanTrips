import pandas as pd
import geopandas as gpd
import numpy as np
import h3
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
import mapclassify
import os
from shapely.geometry import Point, Polygon, LineString

from urbantrips.utils.utils import iniciar_conexion_db, leer_alias, leer_configs_generales, levanto_tabla_sql
from urbantrips.geo.geo import normalizo_lat_lon
from urbantrips.utils.utils import traigo_tabla_zonas
from urbantrips.geo.geo import h3_to_lat_lon, h3toparent, h3_to_geodataframe, point_to_h3, weighted_mean

import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from folium import Figure
from mycolorpy import colorlist as mcp


def load_and_process_data():
    """
    Load and process data from databases, returning the etapas and viajes DataFrames.
    
    Returns:
        etapas (DataFrame): Processed DataFrame containing stage data.
        viajes (DataFrame): Processed DataFrame containing journey data.
    """
    # Read configuration or alias for database connections
    alias = leer_alias()
    
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
    etapas = pd.read_sql_query("SELECT * FROM etapas", conn_data)
    etapas = etapas.merge(distancias, how='left')
    
    # Load journey data from 'viajes' table in 'data' database and merge with distance data
    viajes = pd.read_sql_query("SELECT * FROM viajes", conn_data)
    viajes = viajes.merge(distancias, how='left')
    
    return etapas, viajes

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


def select_h3_from_polygon(poly, res=8, spacing=.0001, viz=False):
    """
    Fill a polygon with points spaced at the given distance apart.
    Create hexagons that correspond to the polygon
    """
    
    if 'id' not in poly.columns: 
        poly = poly.reset_index().rename(columns={'index':'id'})
        
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

    points_result['h3'] = points_result.apply(point_to_h3, axis=1, resolution=res)
    
    points_result = points_result.groupby(['polygon_number', 'h3'], as_index=False).size().drop(['size'], axis=1).rename(columns={'h3_index':'h3'})

    gdf_hexs = h3_to_geodataframe(points_result.h3).rename(columns={'h3_index':'h3'}) 
    gdf_hexs = gdf_hexs.merge(points_result, on='h3')[['polygon_number', 'h3', 'geometry']].sort_values(['polygon_number', 'h3']).reset_index(drop=True)
    
    if viz:
        ax = poly.boundary.plot(linewidth=1.5, figsize=(15,15))
        # gdf_points.plot(ax=ax, alpha=.2)
        gdf_hexs.plot(ax=ax, alpha=.6)

    
    return gdf_hexs.rename(columns={'polygon_number':'id'})

def select_cases_from_polygons(etapas, viajes, polygons, res=8):
    '''
    Dado un dataframe de polígonos, selecciona los casos de etapas y viajes que se encuentran dentro del polígono
    '''
    
    etapas_selec = pd.DataFrame([])
    viajes_selec = pd.DataFrame([])
    gdf_hexs_all = pd.DataFrame([])

    for i, row in polygons.iterrows():

        poly = polygons[polygons.id == row.id].copy()
    
        gdf_hexs = select_h3_from_polygon(poly, 
                                   res=res,                            
                                   viz=False)
        
        gdf_hexs = gdf_hexs[['id', 'h3']].rename(columns={'h3':'h3_o', 'id':'id_polygon'})
    
        seleccionar = etapas.merge(gdf_hexs, on='h3_o')[['dia', 'id_tarjeta', 'id_viaje', 'id_polygon']].drop_duplicates()
        
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

def preparo_lineas_deseo(etapas_selec, polygons_h3, res=8, normalize_polygon=False):

    # Traigo zonas
    zonas_data, zonas_cols = traigo_tabla_zonas()
    res = 3
    if type(res) == int:
        res = [res]    
    for i in res:
        res = [f'res_{i}']
    zonas = res + zonas_cols
    
    gpd_etapas_agrupadas_all = pd.DataFrame([])
    gpd_viajes_agrupados_all = pd.DataFrame([])

    for id_polygon in polygons_h3.id_polygon.unique():
        
    
        poly_h3 = polygons_h3[polygons_h3.id_polygon==id_polygon]
    
        ## Preparo Etapas con inicio, transferencias y fin del viaje    
        etapas_all = etapas_selec.loc[(etapas_selec.id_polygon == id_polygon), ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa', 'h3_o', 'h3_d', 'factor_expansion_tarjeta']]    
        etapas_all['etapa_max'] = etapas_all.groupby(['dia', 'id_tarjeta', 'id_viaje']).id_etapa.transform('max')

        # Borro los casos que tienen 3 transferencias o más
        if len(etapas_all[etapas_all.etapa_max > 3]) > 0: 
            nborrar = len(etapas_all[etapas_all.etapa_max > 3][['id_tarjeta', 'id_viaje']].value_counts()) / len(etapas_all[['id_tarjeta', 'id_viaje']].value_counts()) * 100
            print(f'Se van a borrar los viajes que tienen más de 3 etapas, representan el {round(nborrar,2)}% de los viajes para el polígono {id_polygon}')
            etapas_all = etapas_all[etapas_all.etapa_max <= 3].copy()
            
        etapas_all['ultimo_viaje'] = 0
        etapas_all.loc[etapas_all.etapa_max==etapas_all.id_etapa, 'ultimo_viaje'] = 1
        
        ultimo_viaje = etapas_all[etapas_all.ultimo_viaje==1]
        
        etapas_all['h3'] = etapas_all['h3_o']
        etapas_all = etapas_all[['dia', 'id_tarjeta', 'id_viaje', 'id_etapa', 'h3','factor_expansion_tarjeta']]
        etapas_all['ultimo_viaje'] = 0
        
        ultimo_viaje['h3'] = ultimo_viaje['h3_d']
        ultimo_viaje['id_etapa'] += 1
        ultimo_viaje = ultimo_viaje[['dia', 'id_tarjeta', 'id_viaje', 'id_etapa', 'h3','factor_expansion_tarjeta', 'ultimo_viaje']]
                
        etapas_all = pd.concat([etapas_all, ultimo_viaje]).sort_values(['dia', 'id_tarjeta', 'id_viaje', 'id_etapa']).reset_index(drop=True)
        
        etapas_all['tipo_viaje'] = 'Transfer_' + (etapas_all['id_etapa']-1).astype(str)
        etapas_all.loc[etapas_all.ultimo_viaje==1, 'tipo_viaje'] = 'Fin'
        etapas_all.loc[etapas_all.id_etapa==1, 'tipo_viaje'] = 'Inicio'
    
        etapas_all['polygon'] = ''
        etapas_all.loc[etapas_all.h3.isin(poly_h3.h3_o.unique()), 'polygon'] = id_polygon
    
        etapas_all = etapas_all.drop(['ultimo_viaje'], axis=1)
        
        # Preparo cada etapa de viaje para poder hacer la agrupación y tener inicio, transferencias y destino en un mismo registro
        inicio = etapas_all.loc[etapas_all.tipo_viaje == 'Inicio', ['dia', 'id_tarjeta', 'id_viaje', 'h3', 'factor_expansion_tarjeta', 'polygon']].rename(columns={'h3':'h3_inicio', 'polygon': 'poly_inicio'})
        fin = etapas_all.loc[etapas_all.tipo_viaje == 'Fin', ['dia', 'id_tarjeta', 'id_viaje', 'h3', 'polygon']].rename(columns={'h3':'h3_fin', 'polygon': 'poly_fin'})
        transfer1 = etapas_all.loc[etapas_all.tipo_viaje == 'Transfer_1', ['dia', 'id_tarjeta', 'id_viaje', 'h3', 'polygon']].rename(columns={'h3':'h3_transfer1', 'polygon': 'poly_transfer1'})
        transfer2 = etapas_all.loc[etapas_all.tipo_viaje == 'Transfer_2', ['dia', 'id_tarjeta', 'id_viaje', 'h3', 'polygon']].rename(columns={'h3':'h3_transfer2', 'polygon': 'poly_transfer2'})
        etapas_agrupadas = inicio.merge(transfer1, how='left').merge(transfer2, how='left').merge(fin, how='left').fillna('')

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
                                             'factor_expansion_tarjeta']]

    
        # normalizo inicio y destino y transferencia 1 con transferencia 2
        t1 = normalizo_lat_lon(etapas_agrupadas.copy(),
                                   h3_o='h3_inicio',
                                   h3_d='h3_fin'
                                   )    
        t2 = normalizo_lat_lon(etapas_agrupadas[(etapas_agrupadas.h3_transfer1!='')&(etapas_agrupadas.h3_transfer2!='')].copy(),
                                   h3_o='h3_transfer1',
                                   h3_d='h3_transfer2'
                                   )
        
        t1 = t1[['dia', 'id_tarjeta', 'id_viaje', 'h3_inicio_norm', 'h3_fin_norm']]
        t2 = t2[['dia', 'id_tarjeta', 'id_viaje', 'h3_transfer1_norm', 'h3_transfer2_norm']]
        t1 = t1.merge(t2, how='left')
        
        t1 = t1[['dia', 'id_tarjeta', 'id_viaje', 'h3_inicio_norm', 'h3_transfer1_norm', 'h3_transfer2_norm', 'h3_fin_norm']]
        
        etapas_agrupadas = etapas_agrupadas.merge(t1, how='left').fillna('')

        # Cambio de ubicación los casos que se normalizaron
        etapas_agrupadas['poly_inicio_tmp'] = etapas_agrupadas['poly_inicio']
        etapas_agrupadas['poly_fin_tmp'] = etapas_agrupadas['poly_fin']
        etapas_agrupadas['poly_transfer1_tmp'] = etapas_agrupadas['poly_transfer1']
        etapas_agrupadas['poly_transfer2_tmp'] = etapas_agrupadas['poly_transfer2']
                
        etapas_agrupadas.loc[etapas_agrupadas.h3_inicio!=etapas_agrupadas.h3_inicio_norm, 
                                            'poly_inicio'] = etapas_agrupadas.loc[etapas_agrupadas.h3_inicio!=etapas_agrupadas.h3_inicio_norm, 'poly_fin_tmp']
        etapas_agrupadas.loc[etapas_agrupadas.h3_fin!=etapas_agrupadas.h3_fin_norm, 
                                            'poly_fin'] = etapas_agrupadas.loc[etapas_agrupadas.h3_fin!=etapas_agrupadas.h3_fin_norm, 'poly_inicio_tmp']
        
        etapas_agrupadas.loc[(etapas_agrupadas.h3_transfer1!=etapas_agrupadas.h3_transfer1_norm)&((etapas_agrupadas.h3_transfer1!='')&(etapas_agrupadas.h3_transfer2!='')), 
                                            'poly_transfer1'] = etapas_agrupadas.loc[(etapas_agrupadas.h3_transfer1!=etapas_agrupadas.h3_transfer1_norm)&((etapas_agrupadas.h3_transfer1!='')&(etapas_agrupadas.h3_transfer2!='')), 'poly_transfer2_tmp']
        etapas_agrupadas.loc[(etapas_agrupadas.h3_transfer2!=etapas_agrupadas.h3_transfer2_norm)&((etapas_agrupadas.h3_transfer1!='')&(etapas_agrupadas.h3_transfer2!='')), 
                                            'poly_transfer2'] = etapas_agrupadas.loc[(etapas_agrupadas.h3_transfer2!=etapas_agrupadas.h3_transfer2_norm)&((etapas_agrupadas.h3_transfer1!='')&(etapas_agrupadas.h3_transfer2!='')), 'poly_transfer1_tmp']
        etapas_agrupadas = etapas_agrupadas.drop(['poly_inicio_tmp', 'poly_fin_tmp', 'poly_transfer1_tmp', 'poly_transfer2_tmp'], axis=1)

        etapas_agrupadas.loc[(etapas_agrupadas.h3_transfer1_norm=='')&(etapas_agrupadas.h3_transfer2_norm==''), 'h3_transfer1_norm'] = etapas_agrupadas.loc[(etapas_agrupadas.h3_transfer1_norm=='')&(etapas_agrupadas.h3_transfer2_norm==''), 'h3_transfer1']
        
        etapas_agrupadas = etapas_agrupadas.groupby(['h3_inicio_norm',                                                      
                                                     'h3_transfer1_norm', 
                                                     'h3_transfer2_norm', 
                                                     'h3_fin_norm',                                                     
                                                     'poly_inicio', 
                                                     'poly_transfer1', 
                                                     'poly_transfer2', 
                                                     'poly_fin'],
                                                   as_index=False).factor_expansion_tarjeta.sum()

        etapas_agrupadas['poly_od'] = 0
        etapas_agrupadas.loc[(etapas_agrupadas.poly_inicio!='')|(etapas_agrupadas.poly_fin!=''), 'poly_od'] = 1        
        etapas_agrupadas['poly_transfer'] = 0
        etapas_agrupadas.loc[(etapas_agrupadas.poly_transfer1!='')|(etapas_agrupadas.poly_transfer2!=''), 'poly_transfer'] = 1

        for zona in zonas:

            # Preparo para agrupar por líneas de deseo y cambiar de resolución si es necesario
            etapas_agrupadas_zon = etapas_agrupadas.copy()
            cols_coord = []
            n = 1
            for i in ['h3_inicio_norm', 'h3_transfer1_norm', 'h3_transfer2_norm', 'h3_fin_norm']:
                etapas_agrupadas_zon[[f'lat{n}', f'lon{n}']] = etapas_agrupadas_zon[i].apply(h3_to_lat_lon) 
            
                # Selecciono el centroide del polígono en vez del centroide de cada hexágono            
                if normalize_polygon:    
                    etapas_agrupadas_zon.loc[etapas_agrupadas_zon[i].isin(poly_h3.h3_o.unique()), f'lat{n}'] = poly_h3.polygon_lat.mean()
                    etapas_agrupadas_zon.loc[etapas_agrupadas_zon[i].isin(poly_h3.h3_o.unique()), f'lon{n}'] = poly_h3.polygon_lon.mean()    

                    etapas_agrupadas_zon[f'coord{n}'] = 0
                    etapas_agrupadas_zon.loc[t1[i].isin(poly_h3.h3_o.unique()), f'coord{n}'] = 1
                    cols_coord += [f'coord{n}']
            
                n+=1

                if 'res_' in zona:    
                    etapas_agrupadas_zon[i] = etapas_agrupadas_zon[i].apply(h3toparent, res=int(zona.replace('res_', '')))
                else:
                    etapas_agrupadas_zon = etapas_agrupadas_zon.merge(zonas_data[['h3', zona]].rename(columns={'h3':i, zona:'zona_tmp'}), how='left')
                    etapas_agrupadas_zon[i] = etapas_agrupadas_zon['zona_tmp']
                    etapas_agrupadas_zon = etapas_agrupadas_zon.drop(['zona_tmp'], axis=1)
                    if len(etapas_agrupadas_zon[(etapas_agrupadas_zon.h3_inicio_norm.isna())|(etapas_agrupadas_zon.h3_fin_norm.isna())])>0:
                        print(f'Hay {len(etapas_agrupadas_zon[(etapas_agrupadas_zon.h3_inicio_norm.isna())|(etapas_agrupadas_zon.h3_fin_norm.isna())])} registros a los que no se les pudo asignar {zona}')
                    etapas_agrupadas_zon = etapas_agrupadas_zon[~((etapas_agrupadas_zon.h3_inicio_norm.isna())|(etapas_agrupadas_zon.h3_fin_norm.isna()))]
                etapas_agrupadas_zon[i] = etapas_agrupadas_zon[i].fillna('')

            etapas_agrupadas_zon['transferencia'] = 0
            etapas_agrupadas_zon.loc[(etapas_agrupadas_zon.h3_transfer1_norm!='')|(etapas_agrupadas_zon.h3_transfer2_norm!=''), 'transferencia'] = 1
            
            viajes_agrupados_zon = etapas_agrupadas_zon[['h3_inicio_norm', 
                                                         'h3_fin_norm', 
                                                         'factor_expansion_tarjeta', 
                                                         'poly_od', 
                                                         'poly_transfer', 
                                                         'transferencia', 
                                                         'poly_inicio', 
                                                         'poly_transfer1', 
                                                         'poly_transfer2', 
                                                         'poly_fin',
                                                         'lat1', 
                                                         'lon1',
                                                         'coord1', 
                                                         'lat4',
                                                         'lon4', 
                                                         'coord4']]

            viajes_agrupados_zon_wmean = viajes_agrupados_zon.groupby(['h3_inicio_norm',                                                                        
                                                                       'h3_fin_norm', 
                                                                       'poly_od',
                                                                       'poly_transfer', 
                                                                       'transferencia',
                                                                        'poly_inicio', 
                                                                        'poly_transfer1', 
                                                                        'poly_transfer2', 
                                                                        'poly_fin',
                                                                      ],
                                                       as_index=False).apply(lambda x: pd.Series({
                                                                'lat1': weighted_mean(x['lat1'], x['factor_expansion_tarjeta']),
                                                                'lon1': weighted_mean(x['lon1'], x['factor_expansion_tarjeta']),                                                         
                                                                'lat4': weighted_mean(x['lat4'], x['factor_expansion_tarjeta']),                        
                                                                'lon4': weighted_mean(x['lon4'], x['factor_expansion_tarjeta']),
                                                                            }))    
            
            viajes_agrupados_zon = viajes_agrupados_zon.groupby(['h3_inicio_norm', 
                                                                 'h3_fin_norm', 
                                                                 'poly_od',
                                                                 'poly_transfer',
                                                                 'transferencia',
                                                                 'poly_inicio', 
                                                                 'poly_transfer1', 
                                                                 'poly_transfer2', 
                                                                 'poly_fin',
                                                                ], 
                                                                as_index=False).factor_expansion_tarjeta.sum().sort_values('factor_expansion_tarjeta',                 
                                                                                                                    ascending=False).round().reset_index(drop=True)

            etapas_agrupadas_zon_wmean = etapas_agrupadas_zon.groupby(['h3_inicio_norm',
                                                                       'h3_transfer1_norm',                                                                        
                                                                       'h3_transfer2_norm',                                                                        
                                                                       'h3_fin_norm', 
                                                                       'poly_od', 
                                                                       'poly_transfer',
                                                                       'transferencia', 
                                                                       'poly_inicio', 
                                                                       'poly_transfer1', 
                                                                       'poly_transfer2', 
                                                                       'poly_fin',                                                                      
                                                                      ],
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
            
            etapas_agrupadas_zon = etapas_agrupadas_zon.groupby(['h3_inicio_norm',  
                                                                 'h3_transfer1_norm',                                                                  
                                                                 'h3_transfer2_norm',                                                                  
                                                                 'h3_fin_norm', 
                                                                 'poly_od', 
                                                                 'poly_transfer',                                                                 
                                                                 'transferencia', 
                                                                 'poly_inicio', 
                                                                 'poly_transfer1', 
                                                                 'poly_transfer2', 
                                                                 'poly_fin',
                                                                ], as_index=False).factor_expansion_tarjeta.sum().sort_values('factor_expansion_tarjeta',                 
                                                                                                                    ascending=False).round().reset_index(drop=True)

            viajes_agrupados_zon = viajes_agrupados_zon.merge(viajes_agrupados_zon_wmean)

            etapas_agrupadas_zon = etapas_agrupadas_zon.merge(etapas_agrupadas_zon_wmean)


            etapas_agrupadas_zon['id_polygon'] = id_polygon            
            etapas_agrupadas_zon['zona'] = zona
            viajes_agrupados_zon['id_polygon'] = id_polygon            
            viajes_agrupados_zon['zona'] = zona

            viajes_agrupados_zon = viajes_agrupados_zon[['id_polygon', 
                                                         'zona', 
                                                         'h3_inicio_norm', 
                                                         'h3_fin_norm', 
                                                         'poly_od', 
                                                         'poly_transfer', 
                                                         'transferencia', 
                                                         'poly_inicio', 
                                                         'poly_transfer1', 
                                                         'poly_transfer2', 
                                                         'poly_fin',
                                                         'factor_expansion_tarjeta', 
                                                         'lat1', 'lon1',
                                                         'lat4', 'lon4',]]
            etapas_agrupadas_zon = etapas_agrupadas_zon[['id_polygon', 
                                                         'zona', 
                                                         'h3_inicio_norm', 
                                                         'h3_transfer1_norm', 
                                                         'h3_transfer2_norm',
                                                         'h3_fin_norm', 
                                                         'poly_od', 
                                                         'poly_transfer', 
                                                         'transferencia',
                                                         'poly_inicio', 
                                                         'poly_transfer1', 
                                                         'poly_transfer2', 
                                                         'poly_fin',                                                         
                                                         'factor_expansion_tarjeta', 
                                                        'lat1', 'lon1',
                                                        'lat2', 'lon2',
                                                        'lat3', 'lon3',
                                                        'lat4', 'lon4',]]

            gpd_etapas_agrupadas_all = pd.concat([gpd_etapas_agrupadas_all, etapas_agrupadas_zon], ignore_index=True)
            gpd_viajes_agrupados_all = pd.concat([gpd_viajes_agrupados_all, viajes_agrupados_zon], ignore_index=True)

    return gpd_etapas_agrupadas_all, gpd_viajes_agrupados_all
    
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

        gpd_etapas_agrupadas['transferencia'] = 0
        gpd_etapas_agrupadas.loc[(gpd_etapas_agrupadas.h3_transfer1_norm!='')|(gpd_etapas_agrupadas.h3_transfer2_norm!=''), 'transferencia'] = 1
        
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

def proceso_poligonos():
    print('Procesa polígonos')
    zonificaciones = levanto_tabla_sql('zonificaciones')
    
    poligonos = levanto_tabla_sql('poligonos')
    
    if len(poligonos) > 0:
        # Read trips and jorneys
        etapas, viajes = load_and_process_data()
        # Select cases based fron polygon
        etapas_selec, viajes_selec, polygons, polygons_h3 = select_cases_from_polygons(etapas[etapas.od_validado==1], viajes[viajes.od_validado==1], poligonos, res=8)
        etapas_agrupadas, viajes_agrupados = preparo_lineas_deseo(etapas_selec, polygons_h3, res=[6], normalize_polygon=True)
        
        conn_dash = iniciar_conexion_db(tipo='dash')
        etapas_agrupadas.to_sql("etapas_agrupadas",
                     conn_dash, if_exists="replace", index=False,)
        viajes_agrupados.to_sql("viajes_agrupados",
                     conn_dash, if_exists="replace", index=False,)
        conn_dash.close()