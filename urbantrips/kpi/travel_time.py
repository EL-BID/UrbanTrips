query_etapas = f"""
select e.id,e.id_tarjeta,e.dia,e.id_viaje,e.id_etapa,e.tiempo,e.hora,e.modo,e.id_linea,e.id_ramal,e.interno,e.h3_o,e.h3_d 
from etapas e
inner join ({query_viajes}) v
on v.dia = e.dia
and v.id_tarjeta = e.id_tarjeta
and v.id_viaje = e.id_viaje 
"""
e = pd.read_sql(query_etapas, conn_data)


# etapas por modo
res_parent = 7
e_res = h3.h3_get_resolution(e.h3_d.iloc[0])

e_bus = e.loc[e.modo == 'autobus', :].copy()
e_bus['h3_parent'] = e_bus.h3_d.map(
    lambda h: h3.h3_to_parent(h, res=res_parent))
e_bus['etapa_timestamp'] = pd.to_datetime(
    e_bus['dia'] + ' ' + e_bus['tiempo']).map(lambda s: s.timestamp())

query_etapas_para_gps = f"""
select distinct e.dia,e.id_linea,e.id_ramal,e.interno 
from etapas e
inner join ({query_viajes}) v
on v.dia = e.dia
and v.id_tarjeta = e.id_tarjeta
and v.id_viaje = e.id_viaje 
where modo = 'autobus'"""

query_gps = f"""
select g.id, g.latitud, g.longitud,g.id_original,g.dia,g.id_linea,g.id_ramal,g.interno,g.fecha as gps_fecha,g.h3 
from gps  g
inner join ({query_etapas_para_gps}) as e
where g.dia = e.dia
and g.id_linea = e.id_linea
and g.id_ramal = e.id_ramal
and g.interno = e.interno
;
"""

gps = pd.read_sql(query_gps, conn_data)
gps['h3_parent'] = gps.h3.map(lambda h: h3.h3_to_parent(h, res=res_parent))
gps['h3_e_res'] = gps.h3.map(lambda h: h3.h3_to_parent(h, res=e_res))

#  GTM -3 corregir
gps['datetime'] = (gps.gps_fecha + (3*60*60)).map(datetime.fromtimestamp)


# Paso 3.1 to each leg add the vehicle positions near it

legs_with_gps = e_bus.merge(gps,
                            on=['dia', 'id_linea', 'id_ramal',
                                'interno', 'h3_parent'],
                            suffixes=('_e', '_gps'),
                            how='inner')

legs_with_gps['time_diff'] = (
    legs_with_gps.gps_fecha - legs_with_gps['etapa_timestamp'])/60
legs_with_gps["distance_h3"] = legs_with_gps.apply(
    geo.h3dist,
    axis=1,
    distancia_entre_hex=1,
    h3_o='h3_d',
    h3_d='h3_e_res'
)
# only keep for possible destinations vehicle positions after the origin
legs_with_gps = legs_with_gps.loc[legs_with_gps.time_diff > 0]

# paso 3.2 get the df index that minimices time difference
min_timediff_indexes = legs_with_gps\
    .reindex(columns=['id_e', 'time_diff'])\
    .groupby('id_e').idxmin()
min_timediff_indexes.columns = ['index']
min_timediff_indexes = min_timediff_indexes.reset_index(drop=True)

# add index to legs data to perform the join using leg it and index with min time diff
legs_with_gps = legs_with_gps.reset_index()
legs_with_travel_time = legs_with_gps.merge(
    min_timediff_indexes, on='index', how='inner')

# Paso 3.3 remove ourliers using iqr
quartiles = legs_with_travel_time\
    .reindex(columns=['id_linea', 'time_diff'])\
    .groupby('id_linea', as_index=False)\
    .apply(lambda s: s.quantile([0.25, 0.75]))
quartiles.index = quartiles.index.droplevel(0)
quartiles = quartiles.reset_index().rename(columns={'index': 'q'})
quartiles = quartiles.pivot(
    index='id_linea', columns='q', values='time_diff').reset_index()
quartiles['iqr'] = quartiles[0.75] - quartiles[0.25]
quartiles['top_iqr'] = quartiles[0.75] + (1.5 * quartiles.iqr)

distancia_entre_hex = h3.edge_length(h3.h3_get_resolution(
    legs_with_travel_time.h3_o.iloc[0]), unit='km') * 2

legs_with_travel_time['distancia'] = legs_with_travel_time.apply(
    geo.h3dist,
    axis=1,
    distancia_entre_hex=1,
    h3_o='h3_o',
    h3_d='h3_d'
)
legs_with_travel_time['travel_time_hr'] = legs_with_travel_time['time_diff'] / 60

legs_with_travel_time['vc'] = legs_with_travel_time['distancia'] / \
    legs_with_travel_time['travel_time_hr']

# remove outliers
original_pre_outliers = len(legs_with_travel_time)
print(original_pre_outliers)
legs_with_travel_time = legs_with_travel_time.merge(
    quartiles.reindex(columns=['id_linea', 'top_iqr']))

legs_with_travel_time = legs_with_travel_time.loc[legs_with_travel_time.time_diff <= legs_with_travel_time.top_iqr,
                                                  ['id_e', 'tiempo', 'id_linea', 'interno', 'h3_o', 'h3_d', 'time_diff',
                                                   'distancia', 'travel_time_hr', 'vc']]

legs_with_travel_time = legs_with_travel_time.rename(
    columns={'time_diff': 'travel_time'})
print(len(legs_with_travel_time)/original_pre_outliers)
legs_with_travel_time = legs_with_travel_time.loc[legs_with_travel_time.vc < 70, :]
print(len(legs_with_travel_time)/original_pre_outliers)

tiempos_viaje_bus = legs_with_travel_time.reindex(
    columns=['id_e', 'travel_time']).copy()
