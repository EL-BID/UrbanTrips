import pandas as pd
import geopandas as gpd
import numpy as np
import os
from urbantrips.utils.utils import iniciar_conexion_db, levanto_tabla_sql, guardar_tabla_sql
from urbantrips.geo.geo import normalizo_lat_lon
from urbantrips.utils.utils import traigo_tabla_zonas, calculate_weighted_means
from urbantrips.geo.geo import h3_to_lat_lon, h3toparent, h3_to_geodataframe, point_to_h3, create_h3_gdf
from urbantrips.utils.check_configs import check_config
from shapely.geometry import Point
from urbantrips.utils.utils import leer_alias, leer_configs_generales, duracion
from urbantrips.carto import carto
from urbantrips.kpi.kpi_lineas import calculo_kpi_lineas
import unidecode

def clasificar_tarifa_agregada_social(serie_tarifa_agregada):
    """
    Clasifica categorías de tarifa_agregada social en:
    - 'educacion_jubilacion': incluye escolar, jubilado, pensionado, etc.
    - 'tarifa_social': otras categorías con descuento.
    - 'sin_descuento': valores nulos o vacíos.
    """
    def normalizar(texto):
        if pd.isna(texto):
            return ''
        return unidecode.unidecode(str(texto).strip().lower())
    
    def clasificar(valor):
        val = normalizar(valor)

        if val in {'', '-'}:
            return 'sin_descuento'
        
        if any(palabra in val for palabra in ['jubilad', 'pensionad', 'escolar']):
            return 'educacion_jubilacion'
        
        return 'tarifa_social'
    
    return serie_tarifa_agregada.apply(clasificar)

def clasificar_genero_agregado(serie_genero_agregado):
    def normalizar(val):
        if pd.isna(val):
            return ''
        return str(val).strip().lower()

    def mapear(val):
        val = normalizar(val)
        if val in ['m', 'masculino', 'varón', 'varon', 'hombre']:
            return 'Masculino'
        elif val in ['f', 'femenino', 'mujer']:
            return 'Femenino'
        else:
            return 'No informado'

    return serie_genero_agregado.apply(mapear)

def load_and_process_data(alias_data: str = "", alias_insumos: str = ""):
    """
    Devuelve los DataFrames `etapas` y `viajes` procesados
    sin alterar la lógica de negocio pero con pasos internos
    más rápidos y ordenados.
    """

    print('Levanto Data Etapas y Viajes')
    
    # ── import internos ─────────────────────────────────────────────
    import pandas as pd, numpy as np, gc
    from datetime import datetime
    from urbantrips.utils.utils import iniciar_conexion_db

    # ── conexiones ──────────────────────────────────────────────────
    conn_ins = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)
    conn_dat = iniciar_conexion_db(tipo="data",    alias_db=alias_data)

    print('etapas')
    # ── 1. leer etapas y viajes (filtrados) ─────────────────────────
    etapas  = pd.read_sql_query("SELECT * FROM etapas  WHERE od_validado = 1",
                                conn_dat)
    print('viajes')
    viajes  = pd.read_sql_query("SELECT * FROM viajes  WHERE od_validado = 1",
                                conn_dat)

    etapas['tarifa_agregada'] = clasificar_tarifa_agregada_social(etapas['tarifa'])    
    etapas['genero_agregado'] = clasificar_genero_agregado(etapas['genero'])
    viajes['tarifa_agregada'] = clasificar_tarifa_agregada_social(viajes['tarifa'])    
    viajes['genero_agregado'] = clasificar_genero_agregado(viajes['genero'])
    
    # ── 2. traer SOLO distancias necesarias ─────────────────────────
    keys = (pd.concat([etapas[['h3_o', 'h3_d']],
                       viajes[['h3_o', 'h3_d']]], ignore_index=True)
              .drop_duplicates())
    keys.to_sql("tmp_dist_keys", conn_ins, if_exists="replace", index=False)

    q_dist = """
    SELECT d.h3_o, d.h3_d, d.distance_osm_drive,
           d.distance_osm_walk, d.distance_h3
    FROM   distancias AS d
    JOIN   tmp_dist_keys AS k
      ON   d.h3_o = k.h3_o AND d.h3_d = k.h3_d
    """
    print('distancias')
    distancias = pd.read_sql_query(q_dist, conn_ins)
    conn_ins.execute("DROP TABLE IF EXISTS tmp_dist_keys")

    # merges
    
    etapas  = etapas.merge(distancias, how="left", on=["h3_o", "h3_d"])    
    viajes  = viajes.merge(distancias, how="left", on=["h3_o", "h3_d"])

    # ── 3. travel-times (deduplicado desde SQL) ─────────────────────
    q_tt = """
    SELECT DISTINCT dia, id, travel_time_min
    FROM   travel_times_legs
    """
    ttimes = pd.read_sql_query(q_tt, conn_dat)
    conn_dat.close(); conn_ins.close()

    # ── 4. incorporar travel_time_min y velocidades ─────────────────
    print('travel times')
    if len(ttimes):
        etapas = etapas.merge(ttimes, on=["dia", "id"], how="left")
        etapas["travel_time_min"] = etapas["travel_time_min"].fillna(0)

        etapas["travel_speed"] = np.where(
            etapas["travel_time_min"] > 0,
            (etapas["distance_osm_drive"] /
             (etapas["travel_time_min"] / 60)).round(1),
            np.nan
        )

        # --- agregación al nivel de viaje (sum + min en una pasada) --
        keys_vj = ["dia", "id_tarjeta", "id_viaje"]
        agg_vj = (etapas.groupby(keys_vj, as_index=False)
                         .agg(tt_sum=("travel_time_min", "sum"),
                              tt_min=("travel_time_min", "min")))

        agg_vj["travel_time_min"] = np.where(
            agg_vj["tt_min"] == 0, 0, agg_vj["tt_sum"]).round(2)
        agg_vj = agg_vj[keys_vj + ["travel_time_min"]]

        viajes = viajes.merge(agg_vj, on=keys_vj, how="left")


        viajes["travel_speed"] = np.where(
            viajes["travel_time_min"] > 0,
            (viajes["distance_osm_drive"] /
             (viajes["travel_time_min"] / 60)).round(1),
            np.nan
        )
        viajes.loc[viajes["travel_speed"] >= 50, "travel_speed"] = np.nan
    else:
        etapas[["travel_time_min", "travel_speed"]] = np.nan
        viajes[["travel_time_min", "travel_speed"]] = np.nan
    

    # ── 5. flags y rangos horarios (vectorizado) ────────────────────
    viajes["transferencia"] = (viajes["cant_etapas"] > 1).astype(int)

    cond_rh = [viajes["hora"].between(13, 16), viajes["hora"].between(17, 24)]
    viajes["rango_hora"] = np.select(cond_rh, ["13-16", "17-24"], default="0-12")

    viajes["distancia"] = np.where(
        viajes["distance_osm_drive"] > 5,
        "Viajes largos (>5kms)", "Viajes cortos (<=5kms)")

    viajes["tipo_dia"] = np.where(
        pd.to_datetime(viajes["dia"]).dt.dayofweek >= 5,
        "Fin de Semana", "Hábil")

    viajes["mes"] = pd.to_datetime(viajes["dia"]).dt.to_period("M").astype(str)

    viajes["Fecha"]      = pd.to_datetime(viajes["dia"] + " " + viajes["tiempo"])
    viajes["Fecha_next"] = viajes.groupby(["dia", "id_tarjeta"])["Fecha"].shift(-1)
    viajes["diff_time"]  = ((viajes["Fecha_next"] - viajes["Fecha"])
                            .dt.seconds / 60).round()
    
    # mismas transformaciones para etapas
    etapas["tipo_dia"] = np.where(
        pd.to_datetime(etapas["dia"]).dt.dayofweek >= 5,
        "Fin de Semana", "Hábil")
    etapas["mes"] = pd.to_datetime(etapas["dia"]).dt.to_period("M").astype(str)

    cond_rh_e = [etapas["hora"].between(13, 16), etapas["hora"].between(17, 24)]
    etapas["rango_hora"] = np.select(cond_rh_e, ["13-16", "17-24"], default="0-12")

    etapas = etapas.merge(
        viajes[["dia", "id_tarjeta", "id_viaje", "distancia", "transferencia"]],
        how="left"
    )

    # ── 6. partición modal (vectorizada) ────────────────────────────
    print('modal')
    keys_mod = ["dia", "id_tarjeta", "id_viaje"]
    tmp = (etapas.groupby(keys_mod, sort=False)
                 .agg(n_unique=("modo", "nunique"),
                      etapas_tot=("modo", "size"),
                      modo_ref=("modo", "first")))
    tmp["modo_agregado"] = np.where(
        tmp["n_unique"] == 1,
        np.where(tmp["etapas_tot"] > 1,
                 "multietapa (" + tmp["modo_ref"] + ")",
                 tmp["modo_ref"]),
        "multimodal")
    etapas = etapas.merge(tmp["modo_agregado"].reset_index(), on=keys_mod)
    viajes = viajes.merge(
        etapas.groupby(keys_mod + ["modo_agregado"], as_index=False)
              .size().drop(columns="size"), how="left")

    # ── 7. eliminar registros sin distancias ────────────────────────
    etapas = etapas[etapas["distance_osm_drive"].notna()]
    viajes = viajes[viajes["distance_osm_drive"].notna()]

    # ── 8. rellenar nulos finales ───────────────────────────────────
    for df in (etapas, viajes):
        df["travel_time_min"].fillna(0, inplace=True)
        df["travel_speed"].fillna(0,  inplace=True)

    # ── 9. columnas finales ─────────────────────────────────────────
    etapas = etapas[[
        "id","dia","mes","tipo_dia","id_tarjeta","id_viaje","id_etapa","tiempo",
        "hora","modo","id_linea","id_ramal","interno","h3_o","h3_d","latitud",
        "longitud","od_validado","factor_expansion_original",
        "factor_expansion_linea","factor_expansion_tarjeta",
        "distance_osm_drive","travel_time_min","travel_speed",
        "modo_agregado","rango_hora","distancia","transferencia",
        "genero_agregado","tarifa_agregada"
    ]]

    viajes = viajes[[
        "dia","mes","tipo_dia","id_tarjeta","id_viaje","Fecha","tiempo","hora",
        "cant_etapas","modo","autobus","tren","metro","tranvia","brt","cable",
        "lancha","otros","h3_o","h3_d","od_validado",
        "factor_expansion_linea","factor_expansion_tarjeta",
        "distance_osm_drive","travel_time_min","travel_speed",
        "diff_time","modo_agregado","rango_hora","distancia","transferencia",
        "genero_agregado","tarifa_agregada"
    ]]

    gc.collect()
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


def construyo_indicadores(viajes, poligonos=False):

    if poligonos:
        nombre_tabla = 'poly_indicadores'
    else:
        nombre_tabla = 'agg_indicadores'

    if 'id_polygon' not in viajes.columns:
        viajes['id_polygon'] = 'NONE'


    ind1 = viajes.groupby(['id_polygon', 'dia', 'mes', 'tipo_dia'], as_index=False).factor_expansion_linea.sum(
                                ).round(0).rename(columns={'factor_expansion_linea': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia'], as_index=False).Valor.mean().round()
    ind1['Indicador'] = 'Cantidad de Viajes'
    ind1['Valor'] = ind1.Valor.astype(int)
    ind1['Tipo'] = 'General'
    ind1['type_val'] = 'int'

    ind2 = viajes[viajes.transferencia == 1].groupby(['id_polygon', 'dia', 'mes', 'tipo_dia'], as_index=False).factor_expansion_linea.sum(
                ).round(0).rename(columns={'factor_expansion_linea': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia'], as_index=False).Valor.mean().round()
    ind2['Indicador'] = 'Cantidad de Viajes con Transferencia'
    ind2 = ind2.merge(ind1[['id_polygon', 'mes', 'tipo_dia', 'Valor']].rename(
        columns={'Valor': 'Tot'}), how='left')
    ind2['Valor'] = (ind2['Valor'] / ind2['Tot'] * 100).round(2)
    ind2['Tipo'] = 'General'
    ind2['type_val'] = 'percentage'

    ind3 = viajes.groupby(['id_polygon', 'dia', 'mes', 'tipo_dia', 'rango_hora'], as_index=False).factor_expansion_linea.sum(
                                ).round(0).rename(columns={'factor_expansion_linea': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia', 'rango_hora'], as_index=False).Valor.mean().round()
    ind3['Indicador'] = 'Cantidad de Según Rango Horas'
    ind3['Tot'] = ind3.groupby(['id_polygon', 'mes', 'tipo_dia']).Valor.transform('sum')
    ind3['Valor'] = (ind3['Valor'] / ind3['Tot'] * 100).round(2)
    ind3['Indicador'] = 'Cantidad de Viajes de '+ind3['rango_hora']+'hs'
    ind3['Tipo'] = 'General'
    ind3['type_val'] = 'percentage'

    ind4 = viajes.groupby(['id_polygon', 'dia', 'mes', 'tipo_dia', 'modo'], as_index=False).factor_expansion_linea.sum(
                ).round(0).rename(columns={'factor_expansion_linea': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia', 'modo'], as_index=False).Valor.mean().round()
    ind4['Indicador'] = 'Partición Modal'
    ind4['Tot'] = ind4.groupby(['id_polygon', 'mes', 'tipo_dia']).Valor.transform('sum')
    ind4['Valor'] = (ind4['Valor'] / ind4['Tot'] * 100).round(2)
    ind4 = ind4.sort_values(['id_polygon', 'Valor'], ascending=False)
    ind4['Indicador'] = ind4['modo']
    ind4['Tipo'] = 'Modal'
    ind4['type_val'] = 'percentage'

    ind9 = viajes.groupby(['id_polygon', 'dia', 'mes', 'tipo_dia', 'distancia'], as_index=False).factor_expansion_linea.sum(
                        ).round(0).rename(columns={'factor_expansion_linea': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia', 'distancia'], as_index=False).Valor.mean().round()
    ind9['Indicador'] = 'Partición Modal'
    ind9['Tot'] = ind9.groupby(['id_polygon', 'mes', 'tipo_dia']).Valor.transform('sum')
    ind9['Valor'] = (ind9['Valor'] / ind9['Tot'] * 100).round(2)
    ind9 = ind9.sort_values(['id_polygon', 'Valor'], ascending=False)
    ind9['Indicador'] = 'Cantidad de '+ind9['distancia']
    ind9['Tipo'] = 'General'
    ind9['type_val'] = 'percentage'

    ind5 = viajes.groupby(['id_polygon', 
                           'dia', 
                           'mes', 
                           'tipo_dia', 
                           'id_tarjeta'], 
                                  as_index=False).factor_expansion_linea.first().groupby(['id_polygon', 
                                                                                          'dia', 
                                                                                          'mes', 
                                                                                          'tipo_dia'], 
                                                         as_index=False).factor_expansion_linea.sum().groupby(['id_polygon', 
                                                                                                               'mes', 
                                                                                                               'tipo_dia'], 
                                                                                            as_index=False).factor_expansion_linea.mean().round().rename(columns={'factor_expansion_linea': 'Valor'})
    ind5['Indicador'] = 'Cantidad de Usuarios'
    ind5['Tipo'] = 'General'
    ind5['type_val'] = 'int'

    ind6 = calculate_weighted_means(viajes,
                                    aggregate_cols=['id_polygon', 'dia', 'mes', 'tipo_dia'],
                                    weighted_mean_cols=['distance_osm_drive'],
                                    weight_col='factor_expansion_linea').rename(columns={'distance_osm_drive': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia'], as_index=False).Valor.mean().round(2)
    ind6['Tipo'] = 'Distancias'
    ind6['Indicador'] = 'Distancia Promedio (kms)'
    ind6['type_val'] = 'float'

    ind7 = calculate_weighted_means(viajes,
                                    aggregate_cols=['id_polygon', 'dia', 'mes', 'tipo_dia', 'modo'],
                                    weighted_mean_cols=['distance_osm_drive'],
                                    weight_col='factor_expansion_linea').rename(columns={'distance_osm_drive': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia', 'modo'], as_index=False).Valor.mean().round(2)
    ind7['Tipo'] = 'Distancias'
    ind7['Indicador'] = 'Distancia Promedio (' + ind7.modo + ') (kms)'
    ind7['type_val'] = 'float'

    ind8 = calculate_weighted_means(viajes,
                                    aggregate_cols=['id_polygon', 'dia', 'mes', 'tipo_dia', 'distancia'],
                                    weighted_mean_cols=['distance_osm_drive'],
                                    weight_col='factor_expansion_linea').rename(columns={'distance_osm_drive': 'Valor'}).groupby(['id_polygon', 'mes', 'tipo_dia', 'distancia'], as_index=False).Valor.mean().round(2)
    ind8['Tipo'] = 'Distancias'
    ind8['Indicador'] = 'Distancia Promedio ' + ind8.distancia
    ind8['type_val'] = 'float'

    indicadores = pd.concat(
        [ind1, ind5, ind2, ind3, ind6, ind9, ind7, ind8, ind4])

    tupla_mes = tuple(indicadores.mes.unique().tolist() + ['Todos']) 
    if len(tupla_mes) == 1:
        query = f"""
            SELECT *
            FROM {nombre_tabla}
            WHERE mes != '{tupla_mes[0]}'
        """
    else:
        query = f"""
            SELECT *
            FROM {nombre_tabla}
            WHERE mes NOT IN {tupla_mes}
        """

    indicadores_ant = levanto_tabla_sql(nombre_tabla, 'dash', query=query)

    indicadores = pd.concat([indicadores[['id_polygon', 'mes', 'tipo_dia', 'Tipo', 'Indicador', 'type_val', 'Valor']], 
                                         indicadores_ant], ignore_index=True)

    indicadores_todos = indicadores.groupby(['id_polygon', 'Tipo', 'Indicador', 'type_val'], as_index=False).Valor.mean().round(2)
    indicadores_todos['mes'] = 'Todos'
    indicadores = pd.concat([indicadores, indicadores_todos])
    
    indicadores = format_dataframe(indicadores)
    indicadores = indicadores[['id_polygon', 'mes', 'tipo_dia', 'Tipo', 'Indicador', 'Valor_str']].rename(
        columns={'Valor_str': 'Valor'})

    indicadores = indicadores.sort_values(['id_polygon', 'mes', 'tipo_dia', 'Tipo', 'Indicador'])

    if poligonos:
        guardar_tabla_sql(indicadores, 
                          'poly_indicadores', 
                          'dash', 
                          {'mes': indicadores.mes.unique().tolist()})
    else:
        guardar_tabla_sql(indicadores, 
                          'agg_indicadores', 
                          'dash', 
                          {'mes': indicadores.mes.unique().tolist()})






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
            ['dia', 'id_tarjeta', 'id_viaje', 'id_polygon']] 
        seleccionar = seleccionar.groupby(['dia', 'id_tarjeta', 'id_viaje', 'id_polygon'], as_index=False).size()
        seleccionar['coincidencias'] = 'False'
        seleccionar.loc[seleccionar['size']>1, 'coincidencias'] = 'True'
        seleccionar = seleccionar.drop(['size'], axis=1)
        
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
                   agg_genero_agregado=False,
                   agg_tarifa_agregada=False):

    etapas_agrupadas_zon = etapas_agrupadas.copy()

    if agg_transferencias:
        etapas_agrupadas_zon['transferencia'] = 99
    if agg_modo:
        etapas_agrupadas_zon['modo_agregado'] = 99
    if agg_hora:
        etapas_agrupadas_zon['rango_hora'] = 99
    if agg_distancia:
        etapas_agrupadas_zon['distancia'] = 99
    if agg_genero_agregado:
        etapas_agrupadas_zon['genero_agregado'] = 99
    if agg_tarifa_agregada:
        etapas_agrupadas_zon['tarifa_agregada'] = 99

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
                       agg_genero_agregado=False,
                       agg_tarifa_agregada=False
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
    if agg_genero_agregado:
        matriz['genero_agregado'] = 99
    if agg_tarifa_agregada:
        matriz['tarifa_agregada'] = 99

    matriz = calculate_weighted_means(matriz,
                                      aggregate_cols=aggregate_cols,
                                      weighted_mean_cols=[
                                          'lat1', 
                                          'lon1', 
                                          'lat4', 
                                          'lon4', 
                                          'distance_osm_drive', 
                                          'travel_time_min', 
                                          'travel_speed'],
                                      weight_col='factor_expansion_linea',
                                      zero_to_nan=[
                                          'lat1', 
                                          'lon1', 
                                          'lat4', 
                                          'lon4', 
                                          'travel_time_min', 
                                          'travel_speed'],
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

def agg_matriz(df,
               aggregate_cols=['id_polygon', 
                               'zona', 
                               'Origen', 
                               'Destino',
                               'transferencia', 
                               'modo_agregado', 
                               'rango_hora', 
                               'distancia'],
               weight_col=['distance_osm_drive', 
                           'travel_time_min', 
                           'travel_speed'],
               weight_var='factor_expansion_linea',               
               agg_transferencias=False,
               agg_modo=False,
               agg_hora=False,
               agg_distancia=False):

    if len(df) > 0:
        if agg_transferencias:
            df['transferencia'] = 99
        if agg_modo:
            df['modo_agregado'] = 99
        if agg_hora:
            df['rango_hora'] = 99
        if agg_distancia:
            df['distancia'] = 99
        
        df1 = df.groupby(aggregate_cols, as_index=False)[weight_var].sum()

        df2 = calculate_weighted_means(df,
                              aggregate_cols=aggregate_cols,
                              weighted_mean_cols=weight_col,
                              weight_col=weight_var
                              )
        df = df1.merge(df2)


    return df

def imprimo_matrices_od():
    print('Imprimo matrices OD')
    alias = leer_alias()
    
    matrices_all = levanto_tabla_sql('agg_matrices')
    
    agg_transferencias=True
    agg_modo=True
    agg_hora=True
    agg_distancia=True
    
    matrices_all.loc[matrices_all.travel_time_min==0, 'travel_time_min'] = np.nan
    matrices_all.loc[matrices_all.travel_speed==0, 'travel_speed'] = np.nan
    matrices = matrices_all.groupby(["id_polygon", 
                                     "tipo_dia", 
                                     "zona", 
                                     "inicio", 
                                     "fin", 
                                     "transferencia", 
                                     "modo_agregado", 
                                     "rango_hora", 
                                     "genero_agregado", 
                                     "tarifa_agregada", 
                                     "distancia", 
                                     "orden_origen", 
                                     "orden_destino", 
                                     "Origen", 
                                     "Destino", ], as_index=False)[[
                                                "lat1", 
                                                "lon1", 
                                                "lat4", 
                                                "lon4", 
                                                "distance_osm_drive", 
                                                "travel_time_min", 
                                                "travel_speed", 
                                                "factor_expansion_linea"
                                                                    ]] .mean().round(2)
    
    matriz_ = agg_matriz(matrices,
                        aggregate_cols=['id_polygon', 
                                        'zona', 
                                        'Origen', 
                                        'Destino',
                                        'transferencia', 
                                        'modo_agregado', 
                                        'rango_hora', 
                                        'distancia'],
                        weight_col=['distance_osm_drive', 
                                    'travel_time_min', 
                                    'travel_speed'],
                        weight_var='factor_expansion_linea',
                        agg_transferencias=agg_transferencias,
                        agg_modo=agg_modo,
                        agg_hora=agg_hora,
                        agg_distancia=agg_distancia)
    
    for var_zona in matriz_.zona.unique():
    
        savefile=f"{alias}matriz_{var_zona}"
                
        matriz = matriz_[matriz_.zona==var_zona]
        var_matriz = 'factor_expansion_linea'
        normalize=False
       
        od_heatmap = pd.crosstab(
                    index=matriz['Origen'],
                    columns=matriz['Destino'],
                    values=matriz[var_matriz],
                    aggfunc="sum",
                    normalize=False,
                )
        
        od_heatmap = od_heatmap.reset_index()
        od_heatmap['Origen'] = od_heatmap['Origen'].str[4:]
        od_heatmap = od_heatmap.set_index('Origen')
        od_heatmap.columns = [i[4:] for i in od_heatmap.columns]
                
        db_path = os.path.join("resultados", "matrices", f"{savefile}.xlsx")
        od_heatmap.reset_index().fillna('').to_excel(db_path, index=False)

        od_heatmap = pd.crosstab(
            index=matriz['Origen'],
            columns=matriz['Destino'],
            values=matriz[var_matriz],
            aggfunc="sum",
            normalize=True,
        )
        
        od_heatmap = od_heatmap.reset_index()
        od_heatmap['Origen'] = od_heatmap['Origen'].str[4:]
        od_heatmap = od_heatmap.set_index('Origen')
        od_heatmap.columns = [i[4:] for i in od_heatmap.columns]
                
        db_path2 = os.path.join("resultados", "matrices", f"{savefile}_normalizada.xlsx")
        od_heatmap.reset_index().fillna('').to_excel(db_path2, index=False)

        print(db_path, '---', db_path2)

def crea_socio_indicadores(etapas, viajes):
    print('Creo indicadores de género y tarifa_agregada')

    socio_indicadores = pd.DataFrame([])
    viajes.loc[viajes.travel_time_min==0, 'travel_time_min'] = np.nan
    viajes.loc[viajes.travel_speed==0, 'travel_speed'] = np.nan
    etapas.loc[etapas.travel_time_min==0, 'travel_time_min'] = np.nan
    etapas.loc[etapas.travel_speed==0, 'travel_speed'] = np.nan

    viajesx = calculate_weighted_means(viajes,
                                aggregate_cols=['mes', 
                                                'dia', 
                                                'tipo_dia', 
                                                'genero_agregado', 
                                                'tarifa_agregada'],
                                weighted_mean_cols=['distance_osm_drive', 
                                                    'travel_time_min', 
                                                    'travel_speed', 
                                                    'cant_etapas', 
                                                    'diff_time'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True)
    
    viajesx = calculate_weighted_means(viajesx,
                                aggregate_cols=['mes', 
                                                'tipo_dia', 
                                                'genero_agregado', 
                                                'tarifa_agregada'],
                                weighted_mean_cols=['distance_osm_drive', 
                                                    'travel_time_min', 
                                                    'travel_speed', 
                                                    'cant_etapas', 
                                                    'diff_time'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=False).round(3)

    
    etapasx = calculate_weighted_means(etapas,
                                aggregate_cols=['mes', 
                                                'dia', 
                                                'tipo_dia', 
                                                'genero_agregado', 
                                                'tarifa_agregada', 
                                                'modo'],
                                weighted_mean_cols=['distance_osm_drive', 
                                                    'travel_time_min', 
                                                    'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True)
    
    etapasx = calculate_weighted_means(etapasx,
                                aggregate_cols=['mes', 
                                                'tipo_dia', 
                                                'genero_agregado', 
                                                'tarifa_agregada', 
                                                'modo'],
                                weighted_mean_cols=['distance_osm_drive', 
                                                    'travel_time_min', 
                                                    'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=False).round(3)
    
    # calcular tabla de indicadores
    etapasxx = calculate_weighted_means(etapasx,
                                aggregate_cols=['mes', 
                                                'tipo_dia', 
                                                'genero_agregado', 
                                                'modo'],
                                weighted_mean_cols=['distance_osm_drive', 
                                                    'travel_time_min', 
                                                    'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True).round(3)
    
    etapasxx['tabla'] = 'etapas-genero_agregado-modo'
    socio_indicadores = pd.concat([socio_indicadores, etapasxx], ignore_index=True)
    
    etapasxx = calculate_weighted_means(etapasx,
                                aggregate_cols=['mes', 'tipo_dia', 'tarifa_agregada', 'modo'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True).round(3)
    
    etapasxx['tabla'] = 'etapas-tarifa_agregada-modo'
    socio_indicadores = pd.concat([socio_indicadores, etapasxx], ignore_index=True)
    
    
    viajesxx = calculate_weighted_means(viajesx,
                                aggregate_cols=['mes', 'tipo_dia', 'genero_agregado', 'tarifa_agregada'],
                                weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed', 'cant_etapas', 'diff_time'],
                                weight_col='factor_expansion_linea',                                
                                var_fex_summed=True).round(3)
    
    viajesxx['tabla'] = 'viajes-genero_agregado-tarifa_agregada'
    socio_indicadores = pd.concat([socio_indicadores, viajesxx], ignore_index=True)
    
    # Calculo viajes promedio por día por género y tarifa_agregada
    userx = viajes.copy()
    userx['tarifa_agregada'] = userx['tarifa_agregada'].str.replace('-', '')
    userx = userx.groupby(['dia', 'id_tarjeta'])['tarifa_agregada'].apply(lambda x: '-'.join(x.unique())).reset_index()
    userx.loc[userx.tarifa_agregada.str[-1] == '-', 'tarifa_agregada'] = userx.loc[userx.tarifa_agregada.str[-1] == '-', :].tarifa_agregada.str[:-1]
    userx.loc[userx.tarifa_agregada.str[:1] == '-', 'tarifa_agregada'] = userx.loc[userx.tarifa_agregada.str[:1] == '-', :].tarifa_agregada.str[1:]
    userx = userx.rename(columns={'tarifa_agregada': 'tarifa_agregada_agg'})
    userx.loc[userx.tarifa_agregada_agg=='', 'tarifa_agregada_agg'] = '-'
    userx = viajes.merge(userx, how='left')
    userx = userx.groupby(['mes', 'dia', 'tipo_dia', 'id_tarjeta', 'genero_agregado', 'tarifa_agregada_agg'], as_index=False).agg({'factor_expansion_tarjeta':'count', 'factor_expansion_linea':'mean'}).rename(columns={'factor_expansion_tarjeta':'cant_viajes'}).rename(columns={'tarifa_agregada_agg':'tarifa_agregada'})
    
    userx = calculate_weighted_means(userx,
                                    aggregate_cols=['dia', 'mes', 'tipo_dia', 'genero_agregado', 'tarifa_agregada'],
                                    weighted_mean_cols=['cant_viajes'],
                                    weight_col='factor_expansion_linea',                                
                                    var_fex_summed=True).round(3)
    
    userx = calculate_weighted_means(userx,
                                    aggregate_cols=['mes', 'tipo_dia', 'genero_agregado', 'tarifa_agregada'],
                                    weighted_mean_cols=['cant_viajes'],
                                    weight_col='factor_expansion_linea',                                
                                    var_fex_summed=False).round(3)
    
    userx['tabla'] = 'usuario-genero_agregado-tarifa_agregada'
    socio_indicadores = pd.concat([socio_indicadores, userx], ignore_index=True)
    
    
    # Preparo socioindicadores final
    socio_indicadores = socio_indicadores[['tabla', 'mes', 'tipo_dia', 'genero_agregado', 'tarifa_agregada', 'modo', 'distance_osm_drive', 'travel_time_min', 'travel_speed', 'cant_etapas', 'cant_viajes', 'diff_time', 'factor_expansion_linea']]
    socio_indicadores.columns = ['tabla', 'mes', 'tipo_dia', 'genero_agregado', 'tarifa_agregada', 'Modo', 'Distancia', 'Tiempo de viaje', 'Velocidad', 'Etapas promedio', 'Viajes promedio', 'Tiempo entre viajes', 'factor_expansion_linea']
    
    socio_indicadores['genero_agregado'] = socio_indicadores['genero_agregado'].fillna('')
    socio_indicadores['tarifa_agregada'] = socio_indicadores['tarifa_agregada'].fillna('')
    socio_indicadores['Modo'] = socio_indicadores['Modo'].fillna('')

    socio_indicadores = socio_indicadores.sort_values(['tabla', 'mes', 'tipo_dia'])

    guardar_tabla_sql(socio_indicadores, 
                      'socio_indicadores', 
                      'dash', 
                      {'mes': socio_indicadores.mes.unique().tolist()})

    

def preparo_etapas_agregadas(etapas, viajes):

    e_agg = etapas.groupby(['dia', 'mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo', 'id_linea'], as_index=False).factor_expansion_linea.sum()
    e_agg = e_agg.groupby(['mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo', 'id_linea'], as_index=False).factor_expansion_linea.mean()
    e_agg = e_agg[e_agg.h3_o!=e_agg.h3_d]
    lineas = levanto_tabla_sql('metadata_lineas', 'insumos')
    e_agg = e_agg.merge(lineas[['id_linea', 'nombre_linea']])

    v_agg = viajes.groupby(['dia', 'mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo'], as_index=False).factor_expansion_linea.sum()
    v_agg = v_agg.groupby(['mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo'], as_index=False).factor_expansion_linea.mean()
    v_agg = v_agg[v_agg.h3_o!=v_agg.h3_d]
    
    etapas['etapas_max'] = etapas.groupby(['dia', 'id_tarjeta', 'id_viaje']).id_etapa.transform('max')
    
    transfers = etapas.loc[:, ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa', 'etapas_max', 'id_linea', 'h3_o', 'h3_d', 'factor_expansion_linea']] #(etapas.etapas_max>1)
    transfers = transfers.merge(lineas[['id_linea', 'nombre_linea']], how='left')
    transfers = transfers.pivot(index=['dia', 'id_tarjeta', 'id_viaje'], columns='id_etapa', values='nombre_linea').reset_index().fillna('')
    transfers['seq_lineas'] = ''
    for i in range(1, etapas.etapas_max.max()+1):
        transfers['seq_lineas'] += transfers[i] + ' -- '
        transfers['seq_lineas'] = transfers['seq_lineas'].str.replace(' --  -- ', '')

    transfers.loc[transfers.seq_lineas.str[-4:] == ' -- ', 'seq_lineas'] = transfers.loc[transfers.seq_lineas.str[-4:] == ' -- ', 'seq_lineas'].str[:-4]
    transfers = viajes.merge(transfers[['dia', 'id_tarjeta', 'id_viaje', 'seq_lineas']])
    transfers = transfers.groupby(['dia', 'mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo', 'seq_lineas'], as_index=False).factor_expansion_linea.sum()
    transfers = transfers.groupby(['mes', 'tipo_dia', 'h3_o', 'h3_d', 'modo', 'seq_lineas'], as_index=False).factor_expansion_linea.mean()

    zonas = levanto_tabla_sql('zonas', 'insumos')
    if len(zonas) > 0:
        zonas_cols = zonas.columns.tolist()
        zonas_cols = [item for item in zonas_cols if item not in ['fex', 'latitud', 'longitud']]
        zonas = zonas[zonas_cols]
        
        zonas_cols_o = [f'{item}_o' for item in zonas_cols]
        zonas_cols_d = [f'{item}_d' for item in zonas_cols]
        
        zonas.columns = zonas_cols_o
        e_agg = e_agg.merge(zonas, how='left')
        v_agg = v_agg.merge(zonas, how='left')
        transfers = transfers.merge(zonas, how='left')
        
        zonas.columns = zonas_cols_d
        e_agg = e_agg.merge(zonas, how='left')
        v_agg = v_agg.merge(zonas, how='left')
        transfers = transfers.merge(zonas, how='left')

    # guardar_tabla_sql(e_agg, 
    #               'etapas_agregadas', 
    #               'dash', 
    #               {'mes': e_agg.mes.unique().tolist()})

    # guardar_tabla_sql(v_agg, 
    #               'viajes_agregados', 
    #               'dash', 
    #               {'mes': v_agg.mes.unique().tolist()})

    # guardar_tabla_sql(transfers, 
    #           'transferencias_agregadas', 
    #           'dash', 
    #           {'mes': v_agg.mes.unique().tolist()})
    conn_dash = iniciar_conexion_db(tipo='dash')
    e_agg.to_sql("etapas_agregadas",
             conn_dash, if_exists="replace", index=False,)
    v_agg.to_sql("viajes_agregados",
             conn_dash, if_exists="replace", index=False,)
    transfers.to_sql("transferencias_agregadas",
             conn_dash, if_exists="replace", index=False,)

    conn_dash.close()


def preparo_lineas_deseo(etapas_selec, viajes_selec, polygons_h3='', poligonos='', res=6):
# etapas_selec = etapas.copy()
# viajes_selec = viajes.copy()
# polygons_h3=''
# poligonos=''
# res=[6, 8]
# if True:

    print('Preparo líneas de deseo')
    zonificaciones = levanto_tabla_sql('zonificaciones')

    if len(polygons_h3) == 0:
        id_polygon = 'NONE'
        polygons_h3 = pd.DataFrame([['NONE']], columns=['id_polygon'])
        poligonos = pd.DataFrame([['NONE', 'NONE']], columns=['id', 'tipo'])
        etapas_selec['id_polygon'] = 'NONE'
        viajes_selec['id_polygon'] = 'NONE'
        etapas_selec['coincidencias'] = 'NONE'
        viajes_selec['coincidencias'] = 'NONE'

    # Traigo zonas
    zonas_data, zonas_cols = traigo_tabla_zonas()

    if type(res) == int:
        res = [res]
        
    res_vars = []
    for i in res:
        res_vars += [f'res_{i}']
        if not f'res_{i}' in zonificaciones.zona.unique().tolist():
            h3_vals = pd.concat([etapas_selec.loc[etapas_selec.h3_o.notna(),
                                                  ['h3_o']].rename(columns={'h3_o': 'h3'}),
                                 etapas_selec.loc[etapas_selec.h3_d.notna(),
                                                  ['h3_d']].rename(columns={'h3_d': 'h3'})]).drop_duplicates()
            h3_vals['h3_res'] = h3_vals['h3'].apply(h3toparent, res=i)
            
            h3_zona = create_h3_gdf(h3_vals.h3_res.tolist()).rename(
                columns={'hexagon_id': 'id'}).drop_duplicates()
            h3_zona['zona'] = f'res_{i}'        
            zonificaciones = pd.concat(
                [zonificaciones, h3_zona], ignore_index=True)

    zonas_cols = [x for x in zonas_cols if 'res' not in x]
    zonas = zonas_cols + res_vars
    print('Zonas', zonas)

    for id_polygon in polygons_h3.id_polygon.unique():

        poly_h3 = polygons_h3[polygons_h3.id_polygon == id_polygon]
        poly = poligonos[poligonos.id == id_polygon]
        tipo_poly = poly.tipo.values[0]

        # Preparo Etapas con inicio, transferencias y fin del viaje
        etapas_all = etapas_selec.loc[(etapas_selec.id_polygon == id_polygon), ['dia',
                                                                                'id_tarjeta',
                                                                                'id_viaje',
                                                                                'id_etapa',
                                                                                'h3_o',
                                                                                'h3_d',
                                                                                'modo_agregado',
                                                                                'rango_hora',
                                                                                'genero_agregado',
                                                                                'tarifa_agregada',
                                                                                'transferencia', 
                                                                                'distancia',
                                                                                'distance_osm_drive', 
                                                                                'travel_time_min', 
                                                                                'travel_speed',
                                                                                'coincidencias',
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
                                 'genero_agregado', 
                                 'tarifa_agregada',
                                 'transferencia', 
                                 'distancia',
                                 'distance_osm_drive', 
                                 'travel_time_min', 
                                 'travel_speed',
                                 'coincidencias',
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
                                     'genero_agregado', 
                                     'tarifa_agregada',
                                     'transferencia', 
                                     'distancia',
                                     'distance_osm_drive', 
                                     'travel_time_min', 
                                     'travel_speed',
                                     'coincidencias',
                                     'factor_expansion_linea',
                                     'ultimo_viaje']]

        etapas_all = pd.concat([etapas_all, ultimo_viaje]).sort_values(
            ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa']).reset_index(drop=True)

        etapas_all['tipo_viaje'] = 'Transfer_' + (etapas_all['id_etapa']-1).astype(str)
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
        # h3_coords[['lat', 'lon']] = h3_coords.h3.apply(h3_to_lat_lon)
        h3_coords[['lat', 'lon']] = h3_coords.h3.apply(lambda x: pd.Series(h3_to_lat_lon(x)))


        # Preparo cada etapa de viaje para poder hacer la agrupación y tener inicio, transferencias y destino en un mismo registro
        inicio = etapas_all.loc[etapas_all.tipo_viaje == 'Inicio', ['dia',
                                                                    'id_tarjeta',
                                                                    'id_viaje',
                                                                    'h3',
                                                                    'modo_agregado',
                                                                    'rango_hora',
                                                                    'genero_agregado',
                                                                    'tarifa_agregada',
                                                                    'transferencia', 
                                                                    'distancia',
                                                                    'distance_osm_drive', 
                                                                    'travel_time_min', 
                                                                    'travel_speed',
                                                                    'coincidencias',
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
                                             'genero_agregado', 
                                             'tarifa_agregada',
                                             'transferencia', 
                                             'distancia',
                                             'distance_osm_drive', 
                                             'travel_time_min', 
                                             'travel_speed',
                                             'coincidencias',
                                             'factor_expansion_linea']]
        
        for zona in zonas:
            print('')
            print(f'Polígono {id_polygon} - Tipo: {tipo_poly} - Zona: {zona}')

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
        
            etapas_agrupadas_zon['tipo_dia_'] = pd.to_datetime(etapas_agrupadas_zon.dia).dt.weekday.astype(str).copy()
            etapas_agrupadas_zon['tipo_dia'] = 'Hábil'
            etapas_agrupadas_zon.loc[etapas_agrupadas_zon.tipo_dia_.astype(int)>=5, 'tipo_dia'] = 'Fin de Semana'
            etapas_agrupadas_zon = etapas_agrupadas_zon.drop(['tipo_dia_'], axis=1)
            etapas_agrupadas_zon['mes'] = etapas_agrupadas_zon.dia.str[:7]

            etapas_agrupadas_zon = etapas_agrupadas_zon[['id_polygon', 'zona', 'dia', 'mes', 'tipo_dia', 'id_tarjeta', 'id_viaje',
                                                         'h3_inicio', 'h3_transfer1', 'h3_transfer2', 'h3_fin',
                                                         'inicio', 'transfer1', 'transfer2', 'fin',
                                                         'poly_inicio', 'poly_transfer1', 'poly_transfer2', 'poly_fin',
                                                         'inicio_norm', 'transfer1_norm', 'transfer2_norm', 'fin_norm',
                                                         'poly_inicio_norm', 'poly_transfer1_norm', 'poly_transfer2_norm', 'poly_fin_norm',
                                                         'lon1', 'lat1', 'lon2', 'lat2', 'lon3', 'lat3', 'lon4', 'lat4',
                                                         'lon1_norm', 'lat1_norm', 'lon2_norm', 'lat2_norm', 'lon3_norm', 'lat3_norm', 'lon4_norm', 'lat4_norm',
                                                         'transferencia', 'modo_agregado', 'rango_hora', 'genero_agregado', 'tarifa_agregada', 'coincidencias', 'distancia', 
                                                         'distance_osm_drive', 
                                                          'travel_time_min', 'travel_speed',
                                                         'factor_expansion_linea']]
        
            aggregate_cols = ['id_polygon', 'dia', 'mes', 'tipo_dia', 'zona', 'inicio', 'fin', 'poly_inicio',
                              'poly_fin', 'transferencia', 'modo_agregado', 'rango_hora', 'genero_agregado', 'tarifa_agregada', 'coincidencias', 'distancia']
            
            viajes_matrices = construyo_matrices(etapas_agrupadas_zon,
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
                              'genero_agregado', 
                              'tarifa_agregada',
                              'coincidencias',
                              'distancia']
        
            weighted_mean_cols = ['distance_osm_drive',                                  
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
                           'lon4_norm',
                           'travel_time_min',
                           'travel_speed']
        
            etapas_agrupadas_zon = agrupar_viajes(etapas_agrupadas_zon,
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
                etapas_agrupadas_zon = etapas_agrupadas_zon.merge(zonificaciones[['zona', 'id', 'lat', 'lon']].rename(columns={'id': f'{i}_norm', 'lat': f'lat{n}_norm_tmp', 'lon': f'lon{n}_norm_tmp'}),
                                                                  how='left',
                                                                  on=['zona', f'{i}_norm'])
                etapas_agrupadas_zon.loc[etapas_agrupadas_zon[f'{poly_lst[n-1]}_norm'] == '',
                                         f'lat{n}_norm'] = etapas_agrupadas_zon.loc[etapas_agrupadas_zon[f'{poly_lst[n-1]}_norm'] == '', f'lat{n}_norm_tmp']
                etapas_agrupadas_zon.loc[etapas_agrupadas_zon[f'{poly_lst[n-1]}_norm'] == '',
                                         f'lon{n}_norm'] = etapas_agrupadas_zon.loc[etapas_agrupadas_zon[f'{poly_lst[n-1]}_norm'] == '', f'lon{n}_norm_tmp']
        
                etapas_agrupadas_zon = etapas_agrupadas_zon.drop(
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
        
        
        
            # # Agrupar a nivel de mes y corregir factor de expansión
            sum_viajes = etapas_agrupadas_zon.groupby(['dia', 
                                                       'mes', 
                                                       'tipo_dia', 
                                                       'zona'], as_index=False).factor_expansion_linea.sum().groupby(['mes', 
                                                                                                                      'tipo_dia', 
                                                                                                                      'zona'], as_index=False).factor_expansion_linea.mean().round()
                
            aggregate_cols = ['mes', 
                              'tipo_dia', 
                              'id_polygon', 
                              'poly_inicio_norm',
                              'poly_transfer1_norm',
                              'poly_transfer2_norm',
                              'poly_fin_norm',
                              'zona', 
                              'inicio_norm', 
                              'transfer1_norm',
                              'transfer2_norm', 
                              'fin_norm', 
                              'transferencia', 
                              'modo_agregado', 
                              'rango_hora', 
                              'genero_agregado', 
                              'tarifa_agregada', 
                              'coincidencias',
                              'distancia', ]
            weighted_mean_cols=['distance_osm_drive',                                 
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
            
            etapas_agrupadas_zon = calculate_weighted_means(etapas_agrupadas_zon,
                                        aggregate_cols=aggregate_cols,
                                        weighted_mean_cols=weighted_mean_cols,
                                        weight_col='factor_expansion_linea',
                                        zero_to_nan=zero_to_nan,
                                        var_fex_summed=False)
        
        
            sum_viajes['factor_expansion_linea'] = 1 - (sum_viajes['factor_expansion_linea'] / etapas_agrupadas_zon.groupby(['mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.sum().factor_expansion_linea )
            sum_viajes = sum_viajes.rename(columns={'factor_expansion_linea':'factor_correccion'})
            
            etapas_agrupadas_zon = etapas_agrupadas_zon.merge(sum_viajes)
            etapas_agrupadas_zon['factor_expansion_linea2'] = etapas_agrupadas_zon['factor_expansion_linea'] * etapas_agrupadas_zon['factor_correccion']
            etapas_agrupadas_zon['factor_expansion_linea2'] = etapas_agrupadas_zon['factor_expansion_linea'] - etapas_agrupadas_zon['factor_expansion_linea2']
            etapas_agrupadas_zon = etapas_agrupadas_zon.drop(['factor_correccion', 'factor_expansion_linea'], axis=1)
            etapas_agrupadas_zon = etapas_agrupadas_zon.rename(columns={'factor_expansion_linea2':'factor_expansion_linea'})
        
            # # Agrupar a nivel de mes y corregir factor de expansión
            sum_viajes = viajes_matrices.groupby(['dia', 'mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.sum().groupby(['mes', 'tipo_dia', 'zona'], as_index=False).factor_expansion_linea.mean()
            
            aggregate_cols = ['id_polygon', 
                              'poly_inicio', 
                              'poly_fin',
                              'mes', 
                              'tipo_dia', 
                              'zona', 
                              'inicio', 
                              'fin', 
                              'transferencia', 
                              'modo_agregado', 
                              'rango_hora', 
                              'genero_agregado', 
                              'tarifa_agregada', 
                              'coincidencias',
                              'distancia', 
                              'orden_origen', 
                              'orden_destino', 
                              'Origen', 
                              'Destino']
            weighted_mean_cols = ['lat1', 
                                  'lon1', 
                                  'lat4', 
                                  'lon4', 
                                  'distance_osm_drive', 
                                  'travel_time_min', 
                                  'travel_speed',]
            zero_to_nan = ['lat1', 
                           'lon1', 
                           'lat4', 
                           'lon4',
                           'travel_speed',
                           'travel_time_min']
        
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
                
            if len(poligonos[poligonos.tipo=='cuenca'])>0:
            
                etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_inicio_norm.isin(
                            poligonos[poligonos.tipo=='cuenca'].id.unique()), 
                                'inicio_norm'] = etapas_agrupadas_zon.loc[
                                            etapas_agrupadas_zon.poly_inicio_norm.isin(
                                                    poligonos[poligonos.tipo=='cuenca'].id.unique()), 'inicio_norm']+' (cuenca)'
                etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_transfer1_norm.isin(
                                    poligonos[poligonos.tipo=='cuenca'].id.unique()), 
                                        'transfer1_norm'] = etapas_agrupadas_zon.loc[
                                            etapas_agrupadas_zon.poly_transfer1_norm.isin(poligonos[poligonos.tipo=='cuenca'].id.unique()), 'transfer1_norm']+' (cuenca)'
                etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_transfer2_norm.isin(
                                    poligonos[poligonos.tipo=='cuenca'].id.unique()), 
                                        'transfer2_norm'] = etapas_agrupadas_zon.loc[etapas_agrupadas_zon.poly_transfer2_norm.isin(poligonos[poligonos.tipo=='cuenca'].id.unique()), 'transfer2_norm']+' (cuenca)'
                etapas_agrupadas_zon.loc[
                        etapas_agrupadas_zon.poly_fin_norm.isin(
                            poligonos[poligonos.tipo=='cuenca'].id.unique()), 
                                'fin_norm'] = etapas_agrupadas_zon.loc[etapas_agrupadas_zon.poly_fin_norm.isin(poligonos[poligonos.tipo=='cuenca'].id.unique()), 'fin_norm']+' (cuenca)'
                viajes_matrices.loc[
                        viajes_matrices.poly_inicio.isin(
                                poligonos[poligonos.tipo=='cuenca'].id.unique()), 
                                    'Origen'] = viajes_matrices.loc[viajes_matrices.poly_inicio.isin(poligonos[poligonos.tipo=='cuenca'].id.unique()), 'Origen']+' (cuenca)'
                viajes_matrices.loc[
                        viajes_matrices.poly_fin.isin(
                            poligonos[poligonos.tipo=='cuenca'].id.unique()), 
                                'Destino'] = viajes_matrices.loc[viajes_matrices.poly_fin.isin(poligonos[poligonos.tipo=='cuenca'].id.unique()), 'Destino']+' (cuenca)'
                viajes_matrices.loc[
                            viajes_matrices.poly_inicio.isin(
                                poligonos[poligonos.tipo=='cuenca'].id.unique()), 
                                    'inicio'] = viajes_matrices.loc[viajes_matrices.poly_inicio.isin(poligonos[poligonos.tipo=='cuenca'].id.unique()), 'inicio']+' (cuenca)'
                viajes_matrices.loc[
                            viajes_matrices.poly_fin.isin(
                                    poligonos[poligonos.tipo==
                                        'cuenca'].id.unique()), 'fin'] = viajes_matrices.loc[viajes_matrices.poly_fin.isin(poligonos[poligonos.tipo=='cuenca'].id.unique()), 'fin']+' (cuenca)'
                

            etapas_agrupadas_zon = etapas_agrupadas_zon.fillna(0)

            if id_polygon == 'NONE':

                etapas_agrupadas_zon = etapas_agrupadas_zon.drop(['id_polygon',
                                                                  'poly_inicio_norm',
                                                                  'poly_transfer1_norm',
                                                                  'poly_transfer2_norm',
                                                                  'poly_fin_norm'], axis=1)
        
                viajes_matrices = viajes_matrices.drop(
                    ['poly_inicio', 
                     'poly_fin'], axis=1)


                print(zona, etapas_agrupadas_zon.factor_expansion_linea.sum())
                
                guardar_tabla_sql(etapas_agrupadas_zon, 
                                  'agg_etapas', 
                                  'dash', 
                                  {'mes': etapas_agrupadas_zon.mes.unique().tolist(),                                      
                                  'zona': etapas_agrupadas_zon.zona.unique().tolist()})
                
                guardar_tabla_sql(viajes_matrices, 
                                  'agg_matrices', 
                                  'dash', 
                                  {'mes': viajes_matrices.mes.unique().tolist(),                                      
                                  'zona': viajes_matrices.zona.unique().tolist()})
            else:
                guardar_tabla_sql(etapas_agrupadas_zon, 
                                  'poly_etapas', 
                                  'dash', 
                                  {'mes': etapas_agrupadas_zon.mes.unique().tolist(),
                                  'zona': etapas_agrupadas_zon.zona.unique().tolist(),
                                  'id_polygon': etapas_agrupadas_zon.id_polygon.unique().tolist()})
        
                guardar_tabla_sql(viajes_matrices, 
                                  'poly_matrices', 
                                  'dash', 
                                  {'mes': viajes_matrices.mes.unique().tolist(),
                                  'zona': viajes_matrices.zona.unique().tolist(),
                                  'id_polygon': viajes_matrices.id_polygon.unique().tolist()})

def guarda_particion_modal(etapas):
    df_dummies = pd.get_dummies(etapas.modo)
    etapas = pd.concat([etapas, df_dummies], axis=1)
    cols_dummies = df_dummies.columns.tolist()
    
    etapas_modos = etapas.groupby(['mes', 'tipo_dia', 'genero_agregado', 'id_tarjeta', 'id_viaje'], as_index=False).factor_expansion_linea.mean().merge(
                                                                                etapas.groupby(['dia', 'id_tarjeta', 'id_viaje'], as_index=False)[cols_dummies].sum(), how='left')
    
    cols = ['mes', 'tipo_dia', 'genero_agregado', ]+cols_dummies
    etapas_modos = etapas_modos.groupby(cols, as_index=False).factor_expansion_linea.sum().copy()
    for i in cols_dummies:
        etapas_modos = etapas_modos.rename(columns={i:i.capitalize()})
    guardar_tabla_sql(etapas_modos, 'datos_particion_modal', filtros={'mes': etapas_modos.mes.unique().tolist()})


def agrego_lineas(cols, trx, etapas, gps, servicios, kpis, lineas):
    trx_agg = trx.groupby(cols+['modo'], as_index=False).factor_expansion.sum().rename(columns={'factor_expansion':'transacciones'})
    lineas_agg = lineas[['id_linea', 'nombre_linea', 'empresa']].drop_duplicates()
    etapas_agg = calculate_weighted_means(etapas,
                                        aggregate_cols=cols+['modo'],
                                        weighted_mean_cols=['distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                        zero_to_nan=['distance_osm_drive', 'travel_time_min', 'travel_speed'],
                                        weight_col='factor_expansion_linea', var_fex_summed=False).round(2).rename(columns={'modo':'modo_new'},
                                        ).rename(columns={'distance_osm_drive':'distancia_media'})
    internos_agg = trx.groupby(cols+['interno'], as_index=False).size().groupby(cols, as_index=False).size().rename(columns={'size':'cant_internos_en_trx'})
    
    gps_agg = gps.groupby(cols+['interno'], as_index=False).size().groupby(cols, 
                                                                      as_index=False).size().rename(columns={'size':'cant_internos_en_gps'})
    
    serv_agg = servicios[servicios.valid==1].groupby(cols, as_index=False).agg({'interno':'count', 
                                                                                        'distance_km':'sum', 
                                                                                        'min_ts': 'sum', }).rename(columns={'interno':'cant_servicios', 
                                                                                                                            'distance_km':'serv_distance_km', 
                                                                                                                            'min_ts':'serv_min_ts'})
    
    
    all = trx_agg.merge(etapas_agg, how='left').          \
                    merge(internos_agg, how='left').      \
                    merge(gps_agg, how='left'). \
                    merge(kpis, how='left').              \
                    merge(lineas_agg, how='left').        \
                    merge(serv_agg, how='left').round(2)
    
    all = all[cols+['nombre_linea', 
                    'empresa', 
                    'modo', 
                    'transacciones', 
                    'distancia_media',
                    'travel_time_min', 'travel_speed', 
                    'cant_internos_en_trx', 'cant_internos_en_gps', 'tot_veh', 'tot_km', 'tot_pax',
                   'dmt_mean', 'dmt_median', 'pvd', 'kvd', 'ipk', 'fo_mean', 'fo_median']]
    all['transacciones'] = all['transacciones'].round(0)
    all['tot_pax'] = all['tot_pax'].round(0)
    return all

def resumen_x_linea(etapas, viajes):
    gps = levanto_tabla_sql('gps', 'data')
    gps['fecha'] = pd.to_datetime(gps['fecha'], unit='s')
    lineas = levanto_tabla_sql('metadata_lineas', 'insumos')
    kpis = levanto_tabla_sql('kpi_by_day_line', tabla_tipo='data')
    servicios = levanto_tabla_sql('services', tabla_tipo='data')
    lineas = lineas[['id_linea', 'nombre_linea', 'empresa']].sort_values(['id_linea'])
    
    trx = levanto_tabla_sql('transacciones', 'data')
    if 'tarifa_agregada' in trx.columns:
        trx['tarifa_agregada'] = trx['tarifa_agregada'].fillna('')
    if 'genero_agregado' in trx.columns:
        trx['genero_agregado'] = trx['genero_agregado'].fillna('')

    #Agrego líneas
    all = agrego_lineas(['dia', 'id_linea'], trx, etapas, gps, servicios, kpis, lineas)

    all['mes'] = all['dia'].str[:7]
    
    all = all.groupby(['mes', 'id_linea', 'nombre_linea', 'empresa', 'modo'], as_index=False)[['transacciones',
                                                                         'distancia_media', 
                                                                         'travel_time_min', 
                                                                         'travel_speed',
                                                                         'cant_internos_en_trx', 
                                                                         'cant_internos_en_gps', 
                                                                         'tot_veh', 
                                                                         'tot_km',
                                                                         'tot_pax', 
                                                                         'dmt_mean', 
                                                                         'dmt_median', 
                                                                         'pvd', 
                                                                         'kvd', 
                                                                         'ipk', 
                                                                         'fo_mean',
                                                                         'fo_median']].mean()

    
    guardar_tabla_sql(all, 
                      'resumen_lineas', 
                      'dash', 
                      {'mes': all.mes.unique().tolist()})

    #Agrego líneas y Ramal
    all = agrego_lineas(['dia', 'id_linea', 'id_ramal'], trx, etapas, gps, servicios, kpis, lineas)

    all['mes'] = all['dia'].str[:7]
    
    all = all.groupby(['mes', 
                       'id_linea', 
                       'id_ramal', 
                       'nombre_linea', 
                       'empresa', 
                       'modo'], as_index=False)[['transacciones',
                                                                         'distancia_media', 
                                                                         'travel_time_min', 
                                                                         'travel_speed',
                                                                         'cant_internos_en_trx', 
                                                                         'cant_internos_en_gps', 
                                                                         'tot_veh', 
                                                                         'tot_km',
                                                                         'tot_pax', 
                                                                         'dmt_mean', 
                                                                         'dmt_median', 
                                                                         'pvd', 
                                                                         'kvd', 
                                                                         'ipk', 
                                                                         'fo_mean',
                                                                         'fo_median']].mean()

    
    guardar_tabla_sql(all, 
                      'resumen_lineas_ramal', 
                      'dash', 
                      {'mes': all.mes.unique().tolist()})

@duracion
def proceso_poligonos(etapas=[], viajes=[], zonificaciones=[]):

    print('Procesa polígonos')
    poligonos = levanto_tabla_sql('poligonos')

    if len(poligonos) > 0:

        configs = leer_configs_generales()
        res = configs['resolucion_h3']
        
        print('identifica viajes en polígonos')
        # Select cases based fron polygon
        etapas_selec, viajes_selec, polygons, polygons_h3 = select_cases_from_polygons(
            etapas[etapas.od_validado == 1], viajes[viajes.od_validado == 1], poligonos, res=res)

        preparo_lineas_deseo(etapas_selec, viajes_selec, polygons_h3, poligonos=poligonos, res=[6, 7])

        construyo_indicadores(viajes_selec, poligonos=True)



@duracion
def proceso_lineas_deseo(etapas=[], viajes=[], zonificaciones=[]):

    preparo_etapas_agregadas(etapas.copy(), viajes.copy())

    preparo_lineas_deseo(etapas, viajes, res=[6, 7]) #, 8

    resumen_x_linea(etapas, viajes)

    construyo_indicadores(viajes, poligonos=False)

    crea_socio_indicadores(etapas, viajes)
    
    print('Guardo datos para dashboard')

    guarda_particion_modal(etapas)
    
    # imprimo_matrices_od()

@duracion
def preparo_indicadores_dash(check_configs=True, lineas_deseo=True, poligonos=True, kpis=True):
    if check_configs:
        check_config()
        carto.guardo_zonificaciones()

    zonificaciones = levanto_tabla_sql('zonificaciones')

    etapas, viajes = load_and_process_data()

    if lineas_deseo:
        print('Proceso lineas de deseo')
        proceso_lineas_deseo(etapas=etapas, 
                             viajes=viajes, 
                             zonificaciones=zonificaciones)
    if poligonos:
        print('Proceso Polígonos')
        proceso_poligonos(etapas=etapas, viajes=viajes, zonificaciones=zonificaciones)

    if kpis:
        print('Proceso kpis')
        kpis = calculo_kpi_lineas(etapas=etapas, viajes=viajes)

# def ensure_sqlite_indexes(alias_data, alias_insumos):
#     """
#     Crea (si no existen) los índices sobre las claves (h3_o, h3_d)
#     en las tres tablas que intervienen en la unión.  Ejecutar una sola
#     vez por base; el costo es mínimo gracias a IF NOT EXISTS.

#     Parameters
#     ----------
#     conn_data     : sqlite3.Connection   # conexión a la base que contiene etapas y viajes
#     conn_insumos  : sqlite3.Connection   # conexión a la base que contiene distancias
#     """
#     conn_insumos = iniciar_conexion_db(tipo='insumos', alias_db=alias_insumos)
#     conn_data = iniciar_conexion_db(tipo='data', alias_db=alias_data)
#     # ── índices en la base insumos ─────────────────────────────────
#     with conn_insumos:          # autocommit
#         conn_insumos.execute(
#             "CREATE INDEX IF NOT EXISTS idx_dist_h3 "
#             "ON distancias (h3_o, h3_d);"
#         )

#     # ── índices en la base data ───────────────────────────────────
#     with conn_data:
#         conn_data.execute(
#             "CREATE INDEX IF NOT EXISTS idx_etp_h3 "
#             "ON etapas (h3_o, h3_d);"
#         )
#         conn_data.execute(
#             "CREATE INDEX IF NOT EXISTS idx_vje_h3 "
#             "ON viajes (h3_o, h3_d);"
#         )
        
#     conn_data.close()
#     conn_insumos.close()

# ensure_sqlite_indexes(alias_data="amba_2024_9_18", alias_insumos="amba_2024_9")


# import pandas as pd
# from pandas.testing import assert_frame_equal

# def compare_dataframes(
#     df_old: pd.DataFrame,
#     df_new: pd.DataFrame,
#     keys: list,
#     cols: list | None = None,
#     rtol: float = 1e-05,
#     atol: float = 1e-08,
# ) -> bool:
#     """
#     Compara dos DataFrames y devuelve True si son iguales en las columnas indicadas.
#     - keys: lista de columnas que identifican unívocamente cada fila.
#     - cols: columnas a comparar (None = todas las comunes).
#     - rtol, atol: tolerancias numéricas para floats (usa check_exact si rtol=atol=0).
#     """
#     if cols is None:
#         cols = sorted(set(df_old.columns).intersection(df_new.columns))

#     left  = df_old.sort_values(keys)[cols].reset_index(drop=True)
#     right = df_new.sort_values(keys)[cols].reset_index(drop=True)

#     try:
#         exact = (rtol == 0 and atol == 0)
#         assert_frame_equal(
#             left,
#             right,
#             check_dtype=False,
#             check_exact=exact,
#             rtol=rtol,
#             atol=atol,
#         )
#         return True
#     except AssertionError as err:
#         print("Diferencias encontradas:\n", err)
#         return False
# # columnas que identifican cada fila de Etapas
# clave_etapas = ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa']

# # compara todas las columnas comunes con tolerancia numérica cero
# iguales = compare_dataframes(
#     df_old = etapas,      # dataframe original
#     df_new = etapas_g,    # dataframe generado tras la optimización
#     keys   = clave_etapas,
#     cols   = None,        # (None ⇒ compara todas las columnas comunes)
#     rtol   = 0,           # compara floats exactamente
#     atol   = 0
# )

# print("¿Son idénticos?:", iguales)
