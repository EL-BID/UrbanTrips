import os
import pandas as pd
import numpy as np
from pathlib import Path
from urbantrips.utils import utils
from urbantrips.utils.utils import levanto_tabla_sql, guardar_tabla_sql, duracion
import unidecode
from datetime import datetime

def cal_velocidad_comercial(servicios):
    # Conversión de columnas a datetime
    servicios['min_datetime'] = pd.to_datetime(servicios['min_datetime'])
    servicios['max_datetime'] = pd.to_datetime(servicios['max_datetime'])

    # Cálculo de duración del servicio en minutos
    servicios['diff_minutes'] = (servicios['max_datetime'] - servicios['min_datetime']).dt.total_seconds() / 60

    # Cálculo de velocidad comercial
    servicios['velocidad_comercial'] = servicios['distance_km'] / (servicios['diff_minutes'] / 60)

    # Extraer hora de finalización del servicio
    servicios['hour'] = servicios['max_datetime'].dt.hour

    # Velocidad comercial por línea y ramal en hora pico AM
    filtro_pico_am = (servicios['diff_minutes'] < 180) & (servicios['hour'].between(6, 10))
    vel_comercial_linea_ramal_pico = (
        servicios[filtro_pico_am]
        .groupby(['dia', 'id_linea', 'id_ramal'], as_index=False)['velocidad_comercial']
        .mean().round(1)
    )

    # Distancia media recorrida por vehículo en ramal
    km_recorridos_ramal = (
        servicios.groupby(['dia', 'id_linea', 'id_ramal', 'interno'], as_index=False)['distance_km']
        .sum()
        .groupby(['dia', 'id_linea', 'id_ramal'], as_index=False)['distance_km']
        .mean().rename(columns={'distance_km': 'distancia_media_veh'}).round(1)
    )

    vel_comercial_linea_ramal_pico = vel_comercial_linea_ramal_pico.merge(km_recorridos_ramal, how='left')

    # Velocidad comercial total por línea (todo el día)
    vel_comercial_linea_all = (
        servicios.groupby(['dia', 'id_linea'], as_index=False)['velocidad_comercial']
        .mean().round(1)
    )

    # Velocidad comercial AM
    vel_comercial_linea_am = (
        servicios[filtro_pico_am]
        .groupby(['dia', 'id_linea'], as_index=False)['velocidad_comercial']
        .mean().round(1)
        .rename(columns={'velocidad_comercial': 'velocidad_comercial_am'})
    )

    # Velocidad comercial PM (15 a 19 hs)
    filtro_pico_pm = (servicios['diff_minutes'] < 180) & (servicios['hour'].between(15, 19))
    vel_comercial_linea_pm = (
        servicios[filtro_pico_pm]
        .groupby(['dia', 'id_linea'], as_index=False)['velocidad_comercial']
        .mean().round(1)
        .rename(columns={'velocidad_comercial': 'velocidad_comercial_pm'})
    )

    # Consolidar velocidades comerciales
    vel_comercial_linea = (
        vel_comercial_linea_all
        .merge(vel_comercial_linea_am, how='left')
        .merge(vel_comercial_linea_pm, how='left')
    )

    # Distancia media recorrida por vehículo (total)
    km_recorridos_linea = (
        servicios.groupby(['dia', 'id_linea', 'interno'], as_index=False)['distance_km']
        .sum()
        .groupby(['dia', 'id_linea'], as_index=False)['distance_km']
        .mean().rename(columns={'distance_km': 'distancia_media_veh'}).round(1)
    )

    vel_comercial_linea = vel_comercial_linea.merge(km_recorridos_linea, how='left')

    return vel_comercial_linea


def levanto_data(alias_data, alias_insumos, etapas=[], viajes=[]):

    print('Preparo Datos')

    gps = levanto_tabla_sql('gps', 'data', alias_db=alias_data)

    trx = levanto_tabla_sql('transacciones', 'data', alias_db=alias_data)

    lineas = levanto_tabla_sql(
        'metadata_lineas',
        'insumos',
        alias_db=alias_insumos,
        query='SELECT DISTINCT id_linea, nombre_linea, empresa FROM metadata_lineas ORDER BY id_linea'
    )

    kpis = levanto_tabla_sql('kpi_by_day_line', tabla_tipo='data', alias_db=alias_data)

    servicios = levanto_tabla_sql(
        'services',
        tabla_tipo='data',
        alias_db=alias_data,
        query='SELECT * FROM services WHERE valid = 1'
    )

    # Procesamiento de GPS y cálculo de flota
    gps['fecha'] = pd.to_datetime(gps['fecha'], unit='s')
    gps['dia'] = gps['fecha'].dt.strftime('%Y-%m-%d')

    flota = (
        gps.groupby(['dia', 'id_linea'], as_index=False)
        .size()
        .rename(columns={'size': 'flota'})
    )

    # Cálculo de velocidad comercial
    vel_comercial_linea = cal_velocidad_comercial(servicios)

    # Procesamiento de transacciones

    kpis_varios = flota.merge(vel_comercial_linea, how='left').merge(kpis, how='left')

    return trx, etapas, gps, servicios, kpis_varios, lineas

@duracion
def agrego_lineas(cols, trx, etapas, gps, servicios, kpis_varios, lineas):

    print('Agrego líneas')
    
    # Agregado de transacciones
    resumen_tarifas = (
        etapas
        .groupby(cols + ['modo']+['tarifa_agregada'])['factor_expansion_linea']
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    
    
    resumen_genero = (
        etapas
        .groupby(cols + ['modo']+['genero_agregado'])['factor_expansion_linea']
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    
    tot = (
        etapas
        .groupby(cols + ['modo'])['factor_expansion_linea']
        .sum()
        .reset_index()
        .rename(columns={'factor_expansion_linea':'transacciones'})
    )
    
    # Agregado de etapas con medias ponderadas
    etapas_agg = utils.calculate_weighted_means(
        etapas,
        aggregate_cols=cols + ['modo'],
        weighted_mean_cols=['distancia', 'travel_time_min', 'travel_speed'],
        zero_to_nan=['distancia', 'travel_time_min', 'travel_speed'],
        weight_col='factor_expansion_linea',
        var_fex_summed=False
    ).round(2).rename(columns={    
        'distancia': 'distancia_media_pax'
    })
    
    tot = tot.merge(resumen_genero, 
                               how='left', 
                               on=cols + ['modo']
                              ).merge(resumen_tarifas, how='left', on=cols + ['modo'])
    
    
    # # Redondear solo columnas numéricas
    for col in tot.select_dtypes(include='float').columns:
        try:
            tot[col] = pd.to_numeric(tot[col], errors='coerce').round().astype('Int64')
        except Exception as e:
            print(f"Error en columna {col}: {e}")
    
    etapas_agg = tot.merge(etapas_agg, how='left', on=cols + ['modo'])

    # Agregado de cantidad de internos en transacciones
    internos_agg = (
        trx.groupby(cols + ['interno'], as_index=False).size()
           .groupby(cols, as_index=False).size()
           .rename(columns={'size': 'cant_internos_en_trx'})
    )

    # Agregado de cantidad de internos con GPS
    gps_agg = (
        gps.groupby(cols + ['interno'], as_index=False).size()
           .groupby(cols, as_index=False).size()
           .rename(columns={'size': 'cant_internos_en_gps'})
    )

    # Agregado de servicios válidos
    serv_agg = (
        servicios[servicios.valid == 1]
        .groupby(cols, as_index=False)
        .agg({
            'interno': 'count',
            'distance_km': 'sum',
            'min_ts': 'sum'
        }).rename(columns={
            'interno': 'cant_servicios',
            'distance_km': 'serv_distance_km',
            'min_ts': 'serv_min_ts'
        })
    )

    # Merge de todos los datasets
    all = (
        etapas_agg        
        .merge(internos_agg, how='left')
        .merge(gps_agg, how='left')
        .merge(kpis_varios, how='left')
        .merge(lineas, how='left')
        .merge(serv_agg, how='left')
    )

    # # Cálculo de porcentajes
    # all['tarifa_social_porc'] = (all['tarifa_social'] / all['transacciones'] * 100).round(1)
    # all['tarifa_educacion_jubilacion_porc'] = (all['educacion_jubilacion'] / all['transacciones'] * 100).round(1)

    # Cálculo de mes
    all['mes'] = all['dia'].str[:7]

    # Reordenamiento y selección de columnas finales

    # cols_genero_tarifa = etapas.genero_agregado.unique().tolist()+etapas.tarifa_agregada.unique().tolist() 


    # Redondeo de valores
    all['transacciones'] = all['transacciones'].round(0)
    all['tot_pax'] = all['tot_pax'].round(0).fillna(0)
    all['flota'] = all['flota'].round(0)
    all['serv_min_ts'] = all['serv_min_ts'].round(2)
    all = all.round({col: 2 for col in all.select_dtypes(include='float').columns})

    for i in ['Femenino', 'Masculino', 'No informado', 'educacion_jubilacion', 'tarifa_social', 'sin_descuento']:
        if i not in all.columns:
            all[i] = 0
    
    all = all[['dia', 'mes', 'id_linea',
       'nombre_linea', 'empresa', 'modo', 'transacciones', 'Femenino', 'Masculino',
       'No informado', 'educacion_jubilacion', 'sin_descuento',
       'tarifa_social', 'travel_time_min',
       'travel_speed', 'cant_internos_en_gps', 'cant_internos_en_trx', 'flota', 'velocidad_comercial',
       'velocidad_comercial_am', 'velocidad_comercial_pm',
       'distancia_media_veh', 'tot_km', 'distancia_media_pax', 'dmt_mean',
       'dmt_median', 'pvd', 'kvd', 'ipk', 'fo_mean', 'fo_median', 'factor_expansion_linea']]
    
    return all

@duracion
def calculo_kpi_lineas(alias_data='', alias_dash='', alias_insumos='', etapas=[], viajes=[]):
    print('calculo kpi lineas')
    trx, etapas, gps, servicios, kpis_varios, lineas = levanto_data(alias_data, alias_insumos, etapas=etapas, viajes=viajes)
    kpis = agrego_lineas(['dia', 'id_linea'], trx, etapas, gps, servicios, kpis_varios, lineas)    
    guardar_tabla_sql(kpis, 
                      table_name='kpis_lineas', 
                      tabla_tipo='general', 
                      alias_db=alias_dash, 
                      filtros={"mes": kpis.dia.unique().tolist()},
                      modo="append")
    return kpis