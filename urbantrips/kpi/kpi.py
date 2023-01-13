import pandas as pd
import numpy as np
import h3
import weightedstats as ws
from math import floor
from shapely import wkt
from shapely.geometry import Point

from urbantrips.geo.geo import h3_from_row
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    crear_tablas_indicadores_operativos)


@duracion
def compute_route_section_load(id_linea=False, rango_hrs=False):
    """
    Esta funcion calcula para todas las lineas las cargas por tramo.
    Para aquellas lineas con recorridos reales en un geojson, utiliza estos.
    En caso contrario utiliza los recorridos simplificados
    """

    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    # BORRAR DATA ANTERIOR
    # si se especifica la linea
    if id_linea:

        if type(id_linea) == int:
            id_linea = [id_linea]

        lineas_str = ','.join(map(str, id_linea))
        # y se especifica un rango se borra ese rango
        if rango_hrs:
            q_delete = f"""
                delete from ocupacion_por_linea_tramo
                where id_linea in ({lineas_str})
                and hora_min = {rango_hrs[0]}
                and hora_max = {rango_hrs[1]}
                """
        # si no se borra las que no tengan rango
        else:
            q_delete = f"""
            delete from ocupacion_por_linea_tramo
            where id_linea in ({lineas_str})
            and hora_min is NULL
            and hora_max is NULL
            ;
            """
    else:
        q_delete = """
        delete from ocupacion_por_linea_tramo
        where hora_min is NULL
        and hora_max is NULL
        """

    cur = conn_data.cursor()
    cur.execute(q_delete)
    conn_data.commit()

    # Leer datos de etapas y recorridos
    q_rec = f"select * from recorridos"

    q_etapas = """
        select e.*,d.h3_d, f.factor_expansion
        from etapas e
        left join destinos d
        on d.id = e.id
        left join factores_expansion f
        on e.dia = f.dia
        and e.id_tarjeta = f.id_tarjeta
    """
    # Si se especifica linea traer solo esos datos
    if id_linea:
        q_rec = q_rec + f" where id_linea in ({lineas_str})"
        q_etapas = q_etapas + f" and id_linea in ({lineas_str})"

    etapas = pd.read_sql(q_etapas, conn_data)
    recorridos = pd.read_sql(q_rec, conn_insumos)
    recorridos['wkt'] = recorridos.wkt.apply(wkt.loads)

    if rango_hrs:
        filtro = (etapas.hora >= rango_hrs[0]) & (etapas.hora <= rango_hrs[1])
        etapas = etapas.loc[filtro, :]

    print("Calculando carga por tramo para lineas...")
    etapas_por_tramos = etapas.groupby('id_linea').apply(
        crear_tabla_carga_x_tramo, recorridos, rango_hrs)

    etapas_por_tramos = etapas_por_tramos.reset_index(drop=True)

    print("Subiendo datos a db...")
    etapas_por_tramos.to_sql(
        "ocupacion_por_linea_tramo", conn_data, if_exists="append",
        index=False,)


def crear_tabla_carga_x_tramo(df, recorridos,
                              rango_hrs=False, *args, **kwargs):
    """
    Esta funcon toma una tabla de etapas con od de una id_linea,
    un gdf de recorridos y un rango horario y
    devuelve para esa id_linea una tabla con la carga por tramo
    """
    id_linea = df.id_linea.unique()[0]

    print(f"Calculando carga id_linea {id_linea}")

    if (recorridos.id_linea == id_linea).any():

        recorrido = recorridos.loc[recorridos.id_linea ==
                                   id_linea, 'wkt'].item()

        # crear puntos de origen y destino
        df['o'] = df.h3_o.map(lambda h: Point(h3.h3_to_geo(h)[::-1]))
        df['d'] = df.h3_d.map(lambda h: Point(h3.h3_to_geo(h)[::-1]))

        # proyectar cada punto sobre el recorrido de la linea
        df['o_proj'] = df.apply(proyectar, axis=1, args=('o', recorrido))
        df['d_proj'] = df.apply(proyectar, axis=1, args=('d', recorrido))

        # imputar un sentido a cada etapa
        df = df.reindex(columns=['o_proj', 'd_proj', 'factor_expansion'])
        df['sentido'] = ['ida' if row.o_proj <=
                         row.d_proj else 'vuelta' for _, row in df.iterrows()]

        # calcular totales por sentido
        totales_x_sentido = df\
            .groupby('sentido', as_index=False)\
            .agg(cant_etapas_sentido=('factor_expansion', 'sum'))

        # desglosa cada par od en todos los tramos que atraviesa
        etapas_desglosada_tramos = pd.concat(
            [construir_df_tramos_etapa(row) for _, row in df.iterrows()])

        etapas_desglosada_tramos.tramos = etapas_desglosada_tramos.tramos.map(
            lambda x: round(x, 1))

        etapas_por_tramos = etapas_desglosada_tramos\
            .groupby(['sentido', 'tramos'], as_index=False)\
            .agg(size=('factor_expansion', 'sum'))

        # si no tiene informacion para todos los tramos para cada sentido
        if len(etapas_por_tramos) < 20:
            tramos_sentido_full_set = pd.DataFrame(
                {'sentido': ['ida', 'vuelta'] * 10,
                 'tramos': np.repeat(np.arange(0.0, 1, 0.1), 2),
                 'size': [0] * 20
                 }
            )
            tramos_sentido_full_set.tramos = (
                tramos_sentido_full_set.tramos.map(
                    lambda x: round(x, 1)))

            etapas_por_tramos_full = tramos_sentido_full_set.merge(
                etapas_por_tramos, how='left', on=['sentido', 'tramos'])
            etapas_por_tramos_full['cantidad_etapas'] = etapas_por_tramos_full\
                .size_y.combine_first(etapas_por_tramos_full.size_x)
            etapas_por_tramos_full = etapas_por_tramos_full.reindex(
                columns=['sentido', 'tramos', 'cantidad_etapas'])

        else:
            etapas_por_tramos_full = etapas_por_tramos.rename(
                columns={'size': 'cantidad_etapas'})

        # sumar totales sentido y computar prop etapas
        etapas_por_tramos_full = etapas_por_tramos_full.merge(
            totales_x_sentido, how='left', on='sentido')

        etapas_por_tramos_full['prop_etapas'] = (
            etapas_por_tramos_full['cantidad_etapas'] /
            etapas_por_tramos_full.cant_etapas_sentido)
        etapas_por_tramos_full.prop_etapas = (
            etapas_por_tramos_full.prop_etapas.fillna(
                0))

        etapas_por_tramos_full = etapas_por_tramos_full.drop(
            'cant_etapas_sentido', axis=1)
        etapas_por_tramos_full['id_linea'] = id_linea

        # sumar rango horario
        if rango_hrs:
            etapas_por_tramos_full['hora_min'] = rango_hrs[0]
            etapas_por_tramos_full['hora_max'] = rango_hrs[1]
        else:
            etapas_por_tramos_full['hora_min'] = None
            etapas_por_tramos_full['hora_max'] = None

        # formatear de acuerdo a schema db
        etapas_por_tramos_full = etapas_por_tramos_full.reindex(
            columns=['id_linea', 'sentido', 'tramos', 'hora_min', 'hora_max',
                     'cantidad_etapas', 'prop_etapas'])
        return etapas_por_tramos_full
    else:
        print("No existe recorrido para id_linea:", id_linea)


def primer_decimal(num):
    return floor(num * 10) / 10


def proyectar(row, col, recorrido):
    """
    Esta funcion toma una fila de etapas, un origen o destinio
    y un recorrido
    y proyecta las coordenadas sobre el recorrido devolviendo
    el primer decimal del recorrido normalizado
    """
    return primer_decimal(recorrido.project(row[col], normalized=True))


def construir_df_tramos_etapa(row):
    """
    Esta funcion toma una fila de un df de con el lrs del O y el D
    y calcula para cada par todos los tramos de la linea que recorre
    """

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

    lim_inf = row.o_proj
    lim_sup = row.d_proj
    sentido = row.sentido
    f_exp = row.factor_expansion

    if row.sentido == 'vuelta':
        lim_inf = reemplazo_vuelta[lim_inf]
        lim_sup = reemplazo_vuelta[lim_sup]

    tramos_atraviesa_etapa = np.arange(lim_inf, lim_sup + 0.1, 0.1)
    tope = tramos_atraviesa_etapa <= 0.9
    tramos_atraviesa_etapa = tramos_atraviesa_etapa[tope]

    df = pd.DataFrame(
        {
            'sentido': [sentido] * len(tramos_atraviesa_etapa),
            'tramos': tramos_atraviesa_etapa,
            'factor_expansion': [f_exp] * len(tramos_atraviesa_etapa),
        }
    )
    return df


@duracion
def compute_kpi():
    """
    Esta funcion toma los datos de oferta de la tabla gps
    los datos de demanda de la tabla trx
    y produce una serie de indicadores operativos por
    dia y linea y por dia, linea, interno
    """
    # crear tablas
    crear_tablas_indicadores_operativos()

    print('Produciendo indicadores operativos...')
    conn_data = iniciar_conexion_db(tipo='data')
    conn_insumos = iniciar_conexion_db(tipo='insumos')

    cur = conn_data.cursor()
    q = """
        SELECT tbl_name FROM sqlite_master
        WHERE type='table'
        AND tbl_name='gps';
    """
    listOfTables = cur.execute(q).fetchall()

    if listOfTables == []:
        print("No existe tabla GPS en la base")
        print("No se pudeden computar indicadores de oferta")
        return None

    res = 11
    distancia_entre_hex = h3.edge_length(resolution=res, unit="km")
    distancia_entre_hex = distancia_entre_hex * 2

    print('Leyendo datos de oferta')
    q = """
    select * from gps
    order by dia, id_linea, interno, fecha
    """
    gps = pd.read_sql(q, conn_data)

    # Georeferenciar con h3
    gps['h3'] = gps.apply(h3_from_row, axis=1,
                          args=(res, 'latitud', 'longitud'))

    # Producir un lag con respecto al siguiente posicionamiento gps
    gps["h3_lag"] = gps.reindex(columns=['dia', 'id_linea', 'interno', 'h3'])\
        .groupby(['dia', 'id_linea', 'interno'])\
        .shift(-1)

    # Calcular distancia h3
    gps = gps.dropna(subset=['h3', 'h3_lag'])
    gps_dict = gps.to_dict('records')
    gps['dist_km'] = list(map(distancia_h3, gps_dict))
    gps['dist_km'] = gps['dist_km'] * distancia_entre_hex

    print('Leyendo datos de demanda')
    q = """
        SELECT e.dia,e.id_linea,e.interno,e.id_tarjeta,e.h3_o,
        d.h3_d, f.factor_expansion
        from etapas e
        LEFT JOIN destinos d
        ON e.id = d.id
        LEFT JOIN factores_expansion f
        ON e.id_tarjeta = f.id_tarjeta
        AND e.dia = f.dia
    """
    etapas = pd.read_sql(q, conn_data)
    distancias = pd.read_sql_query(
        """
        SELECT *
        FROM distancias
        """,
        conn_insumos,
    )
    # usar distancias h3 cuando no hay osm
    distancias.distance_osm_drive = (
        distancias.distance_osm_drive.combine_first(distancias.distance_h3)
    )

    # obtener etapas y sus distancias recorridas
    etapas = etapas.merge(distancias, how='left', on=['h3_o', 'h3_d'])

    print('Calculando indicadores de oferta por interno')
    # Calcular kilometros vehiculo dia kvd
    oferta_interno = gps\
        .groupby(['id_linea', 'dia', 'interno'], as_index=False)\
        .agg(
            kvd=('dist_km', 'sum')
        )
    # Eliminar los vehiculos que tengan 0 kms recorridos
    oferta_interno = oferta_interno.loc[oferta_interno.kvd > 0]

    print('Calculando indicadores de demanda por interno')

    # calcular pax veh dia (pvd) y distancia media recorrida (dmt)
    demanda_interno = etapas\
        .groupby(['id_linea', 'dia', 'interno'], as_index=False)\
        .apply(indicadores_demanda_interno)

    print('Calculando indicadores operativos por dia e interno')
    indicadores_interno = oferta_interno.merge(
        demanda_interno, how='left', on=[
            'id_linea', 'dia', 'interno']
    )
    internos_sin_demanda = indicadores_interno.pvd.isna().sum()
    internos_sin_demanda = round(
        internos_sin_demanda/len(indicadores_interno) * 100,
        2
    )
    print(f'Hay {internos_sin_demanda} por ciento de internos sin demanda')

    print('Calculando IPK y FO')
    # calcular indice pasajero kilometros (ipk) y factor de ocupacion (fo)
    indicadores_interno['ipk'] = indicadores_interno.pvd / \
        indicadores_interno.kvd

    # Calcular espacios-km ofertados (EKO) y los espacios-km demandados (EKD).
    eko = (indicadores_interno.kvd * 60)
    ekd = (indicadores_interno.pvd * indicadores_interno.dmt_mean)
    indicadores_interno['fo'] = ekd/eko

    print('Subiendo indicadores por interno a la db')
    cols = [
        'id_linea', 'dia', 'interno', 'kvd', 'pvd',
        'dmt_mean', 'dmt_median', 'ipk', 'fo'
    ]
    indicadores_interno = indicadores_interno.reindex(columns=cols)
    indicadores_interno.to_sql(
        'indicadores_operativos_interno', conn_data,
        if_exists="append", index=False,
    )

    print('Calculando indicadores de demanda por linea y dia')

    demanda_linea = etapas\
        .groupby(['id_linea', 'dia'], as_index=False)\
        .apply(indicadores_demanda_linea)

    print('Calculando indicadores de oferta por linea y dia')

    oferta_linea = oferta_interno.groupby(['id_linea', 'dia'], as_index=False)\
        .agg(
        tot_veh=('interno', 'count'),
        tot_km=('kvd', 'sum'),
    )

    indicadores_linea = oferta_linea.merge(
        demanda_linea, how='left', on=['id_linea', 'dia'])
    indicadores_linea['pvd'] = indicadores_linea.tot_pax / \
        indicadores_linea.tot_veh
    indicadores_linea['kvd'] = indicadores_linea.tot_km / \
        indicadores_linea.tot_veh
    indicadores_linea['ipk'] = indicadores_linea.tot_pax / \
        indicadores_linea.tot_km

    # Calcular espacios-km ofertados (EKO) y los espacios-km demandados (EKD).
    eko = (indicadores_linea.tot_km * 60)
    ekd = (indicadores_linea.tot_pax *
           indicadores_linea.dmt_mean)

    indicadores_linea['fo'] = ekd / eko

    print('Subiendo indicadores por linea a la db')

    cols = [
        'id_linea', 'dia', 'tot_veh', 'tot_km', 'tot_pax', 'dmt_mean',
        'dmt_median', 'pvd', 'kvd', 'ipk', 'fo'
    ]
    indicadores_linea = indicadores_linea.reindex(columns=cols)
    indicadores_linea.to_sql(
        'indicadores_operativos_linea', conn_data,
        if_exists="append", index=False,
    )


def distancia_h3(row, *args, **kwargs):
    try:
        out = h3.h3_distance(row['h3'], row['h3_lag'])
    except ValueError as e:
        out = None
    return out


def indicadores_demanda_interno(df):
    d = {}
    d['pvd'] = df['factor_expansion'].sum()
    d['dmt_mean'] = np.average(
        a=df.distance_osm_drive, weights=df.factor_expansion)
    d['dmt_median'] = ws.weighted_median(
        data=df.distance_osm_drive.tolist(),
        weights=df.factor_expansion.tolist())
    return pd.Series(d, index=['pvd', 'dmt_mean', 'dmt_median'])


def indicadores_demanda_linea(df):
    d = {}
    d['tot_pax'] = df['factor_expansion'].sum()
    d['dmt_mean'] = np.average(
        a=df.distance_osm_drive, weights=df.factor_expansion)
    d['dmt_median'] = ws.weighted_median(
        data=df.distance_osm_drive.tolist(),
        weights=df.factor_expansion.tolist())
    return pd.Series(d, index=['tot_pax', 'dmt_mean', 'dmt_median'])
