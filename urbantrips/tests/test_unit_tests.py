import pandas as pd
import os
import pytest
import os

from urbantrips.kpi import kpi
from urbantrips.datamodel import legs, misc, trips, transactions
from urbantrips.destinations import destinations as dest
from urbantrips.geo import geo
from urbantrips.utils import utils
from urbantrips.carto import carto, routes
from urbantrips.viz import viz
from urbantrips.carto.routes import process_routes_metadata


@pytest.fixture
def df_latlng():
    df = pd.DataFrame(
        {
            "latitud": [-34.6158037, 39.441915],
            "longitud": [-58.5033381, -0.3771238],
        }
    )
    return df


@pytest.fixture
def path_test_data():
    path = os.path.join(os.getcwd(), "urbantrips", "tests", "data")
    return path


@pytest.fixture
def matriz_validacion_test_amba(path_test_data):
    path = os.path.join(path_test_data, "matriz_validacion_amba_test.csv")
    df = pd.read_csv(path, dtype={"id_linea": int})
    return df


@pytest.fixture
def df_etapas(path_test_data):
    path = os.path.join(path_test_data, "subset_etapas.csv")
    df = pd.read_csv(path, dtype={"id_tarjeta": str})
    return df


@pytest.fixture
def df_trx(path_test_data):
    path = os.path.join(path_test_data, "subset_transacciones.csv")
    df = pd.read_csv(path, dtype={"id_tarjeta": str})
    return df


@pytest.fixture
def df_test_id_viaje():
    dia_1 = pd.DataFrame(
        {
            "id": range(1, 8),
            "fecha_dt": [
                "2022-08-11 12:00",
                "2022-08-11 12:30",
                "2022-08-11 14:30",
                "2022-08-11 18:30",
                "2022-08-11 19:30",
                "2022-08-11 09:30",
                "2022-08-11 10:30",
            ],
            "id_tarjeta": [1] * 5 + [2, 2],
        }
    )

    dia_2 = pd.DataFrame(
        {
            "id": range(10, 17),
            "fecha_dt": [
                "2022-08-12 12:00",
                "2022-08-12 12:30",
                "2022-08-12 14:30",
                "2022-08-12 18:30",
                "2022-08-12 19:30",
                "2022-08-12 09:30",
                "2022-08-12 9:31",
            ],
            "id_tarjeta": [1] * 5 + [2, 2],
        }
    )

    df = pd.concat([dia_1, dia_2])

    df.fecha_dt = pd.to_datetime(df.fecha_dt)
    df["dia"] = df.fecha_dt.dt.strftime("%Y-%m-%d")

    df["hora_shift"] = (
        df.reindex(columns=["dia", "id_tarjeta", "fecha_dt"])
        .groupby(["dia", "id_tarjeta"])
        .shift(1)
    )
    df["delta"] = df.fecha_dt - df.hora_shift
    df["delta"] = df["delta"].fillna(pd.Timedelta(seconds=0))
    df["delta"] = df.delta.dt.total_seconds()
    df["delta"] = df["delta"].map(int)
    df["hora"] = df.fecha_dt.dt.strftime("%H:%M:%S")

    return df


def test_destinos_potenciales(df_etapas):
    def check(d):
        primer_origen = d.h3_o.iloc[[0]]
        origenes_sig = d.h3_o.iloc[1:]
        destinos = pd.concat([origenes_sig, primer_origen]).values
        comparacion = d.h3_d.values == destinos
        return all(comparacion)

    destinos_potenciales = dest.imputar_destino_potencial(df_etapas)
    assert destinos_potenciales.groupby("id_tarjeta").apply(check).all()


def test_asignar_id_viaje_etapa(df_trx):
    df = legs.asignar_id_viaje_etapa_orden_trx(df_trx)

    # Caso simple 4 colectivos 4 viajes de 1 etapa
    df_4_simples = df.loc[df.id_tarjeta == "37030208", :]

    # Caso multimodal 2 viajes, 3 etapas por viaje, 3 modos
    multim = df.loc[df.id_tarjeta == "3839538659", :]

    # Checkout y trx en misma hora para viajes 2 y 3
    chkout = df.loc[df.id_tarjeta == "37035823", :]
    chkout = chkout.loc[chkout.id_viaje.isin([2, 3])]

    assert len(df_4_simples) == 4
    assert (df_4_simples.id_viaje == [1, 2, 3, 4]).all()
    assert df_4_simples.id_etapa.unique()[0] == 1

    assert (multim.id_viaje == [1] * 3 + [2] * 3).all()
    assert (multim.id_etapa == [1, 2, 3] * 2).all()

    assert (chkout.id_viaje == [2, 2, 3]).all()
    assert (chkout.id_etapa == [1, 2, 1]).all()


def test_h3_from_row(df_latlng):

    lat = "latitud"
    lng = "longitud"
    res = 8
    row = df_latlng.iloc[0]
    out = geo.h3_from_row(row, res, lat, lng)

    assert out == "88c2e312b9fffff"


def test_referenciar_h3(df_latlng):
    out = geo.referenciar_h3(df=df_latlng, res=8, nombre_h3="h3")
    assert out.h3.iloc[1] == "8839540a87fffff"


def test_crear_viaje_id_acumulada_tarjeta_1(df_test_id_viaje):
    dia = df_test_id_viaje.dia == "2022-08-11"
    tarj = df_test_id_viaje.id_tarjeta == 1
    mask = (dia) & (tarj)
    dia_tarjeta = df_test_id_viaje.loc[mask]

    viajes_id_120 = legs.crear_viaje_id_acumulada(
        dia_tarjeta,
        ventana_viajes=120 * 60,
    )
    viajes_id_150 = legs.crear_viaje_id_acumulada(
        dia_tarjeta,
        ventana_viajes=150 * 60,
    )
    viajes_id_30 = legs.crear_viaje_id_acumulada(
        dia_tarjeta,
        ventana_viajes=30 * 60,
    )
    viajes_id_29 = legs.crear_viaje_id_acumulada(
        dia_tarjeta,
        ventana_viajes=29 * 60,
    )

    assert viajes_id_120 == [1, 1, 2, 3, 3]
    assert viajes_id_150 == [1, 1, 1, 2, 2]
    assert viajes_id_30 == [1, 1, 2, 3, 4]
    assert viajes_id_29 == [1, 2, 3, 4, 5]


def test_crear_viaje_id_acumulada(df_test_id_viaje):
    trx = df_test_id_viaje.copy()
    trx = trx.rename(columns={"fecha_dt": "fecha"})
    trx = legs.asignar_id_viaje_etapa_fecha_completa(
        trx,
        ventana_viajes=120 * 60,
    )

    assert (trx.id_viaje == [1, 1, 2, 3, 3, 1, 1, 1, 1, 2, 3, 3, 1, 1]).all()
    assert (trx.id_etapa == [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2]).all()


def test_cambiar_id_tarjeta_trx_simul_delta(df_test_id_viaje):
    trx = df_test_id_viaje.copy()
    duplicado_extra = pd.DataFrame(
        {
            "id": 17,
            "fecha_dt": "2022-08-12 09:33:00",
            "id_tarjeta": 2,
            "dia": "2022-08-12",
            "hora_shift": "2022-08-12 09:30:00",
            "delta": 3 * 60,
            "hora": "09:33:00",
        },
        index=[0],
    )
    trx = pd.concat([trx, duplicado_extra]).reset_index(drop=True).copy()
    trx.id_tarjeta = trx.id_tarjeta.map(str)
    trx["id_linea"] = 5
    trx["interno"] = 10
    trx = trx.rename(columns={"fecha_dt": "fecha"})
    trx = trx.reset_index(drop=True)

    trx_5, tarjetas_duplicadas_5 = legs.cambiar_id_tarjeta_trx_simul_fecha(
        trx.copy(), ventana_duplicado=5
    )

    assert len(tarjetas_duplicadas_5) == 2
    assert (tarjetas_duplicadas_5.id_tarjeta_original == ["2", "2"]).all()
    assert (tarjetas_duplicadas_5.id_tarjeta_nuevo == ["2_1", "2_2"]).all()

    assert (trx_5.loc[trx_5['id'].isin([15, 16, 17]),
            'id_tarjeta'] == ['2_0', '2_1', '2_2']).all()

    trx_1, tarjetas_duplicadas_1 = legs.cambiar_id_tarjeta_trx_simul_fecha(
        trx.copy(), ventana_duplicado=1
    )

    assert len(tarjetas_duplicadas_1) == 1
    assert (tarjetas_duplicadas_1.id_tarjeta_original == ["2"]).all()
    assert (tarjetas_duplicadas_1.id_tarjeta_nuevo == ["2_1"]).all()
    assert (trx_1.loc[trx_1['id'].isin([15, 16, 17]),
            'id_tarjeta'] == ['2_0', '2_1', '2_0']).all()


def create_test_trx(geolocalizar_trx_config,
                    nombre_archivo_trx,
                    nombres_variables_trx,
                    formato_fecha,
                    col_hora,
                    tipo_trx_invalidas,
                    nombre_archivo_gps,
                    nombres_variables_gps):

    for tipo in ['data', 'insumos']:
        filePath = utils.traigo_db_path(tipo=tipo)
        if os.path.exists(filePath):
            os.remove(filePath)

    # crear_directorios:
    utils.create_directories()

    # crear base de datos:
    utils.create_db()

    transactions.create_transactions(geolocalizar_trx_config,
                                     nombre_archivo_trx,
                                     nombres_variables_trx,
                                     formato_fecha,
                                     col_hora,
                                     tipo_trx_invalidas,
                                     nombre_archivo_gps,
                                     nombres_variables_gps)


def test_amba_integration(matriz_validacion_test_amba):
    configs = utils.leer_configs_generales()
    utils.create_db()
    routes.process_routes_metadata()
    geolocalizar_trx_config = configs["geolocalizar_trx"]
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]
    nombre_archivo_gps = configs["nombre_archivo_gps"]
    nombres_variables_gps = configs["nombres_variables_gps"]

    create_test_trx(geolocalizar_trx_config,
                    nombre_archivo_trx,
                    nombres_variables_trx,
                    formato_fecha,
                    col_hora,
                    tipo_trx_invalidas,
                    nombre_archivo_gps,
                    nombres_variables_gps)

    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }

    conn_data = utils.iniciar_conexion_db(tipo='data')
    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')

    routes.process_routes_metadata()

    trx = pd.read_sql("select * from transacciones", conn_data)

    # testear formateo de columnas
    cols = ['id', 'id_original', 'id_tarjeta', 'fecha', 'dia', 'tiempo',
            'hora', 'modo', 'id_linea', 'id_ramal', 'interno', 'orden_trx',
            'latitud', 'longitud', 'factor_expansion']
    assert all(trx.columns.isin(cols))

    # testear que no haya faltantes
    for c in ["id_tarjeta", "fecha", "id_linea", "latitud", "longitud"]:
        assert trx[c].notna().all()

    # testear que no haya tarjetas con trx unica
    assert (trx.groupby('id_tarjeta').size() > 1).all()

    # longitud del string igual para todas las tarjetas
    assert trx.id_tarjeta.str.len().std() == 0

    legs.create_legs_from_transactions(trx_order_params)

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    routes.process_routes_geoms()

    # imputar destinos
    dest.infer_destinations()

    q = """
    select *
    from etapas e
    """
    etapas = pd.read_sql(q, conn_data)

    # chequear id viajes
    etapa = etapas.loc[etapas.id_tarjeta == '0037030208', :]
    assert (etapa.id_viaje == range(1, 5)).all()
    # chequear id etapas
    assert (etapa.id_etapa == 1).all()

    # chequear h3_o
    assert (etapa.loc[etapa.id_viaje == 3, 'h3_o']
            == '88c2e311dbfffff').iloc[0]

    # chequear h3_d
    assert (etapa.loc[etapa.id_viaje == 2, 'h3_d']
            == '88c2e311dbfffff').iloc[0]

    # chequear validacion por si y por no
    assert etapa.loc[etapa.id_viaje == 2, 'od_validado'].iloc[0] == 1


def test_amba_destinos_min_distancia(matriz_validacion_test_amba):

    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]
    nombre_archivo_gps = configs["nombre_archivo_gps"]
    nombres_variables_gps = configs["nombres_variables_gps"]

    create_test_trx(geolocalizar_trx_config,
                    nombre_archivo_trx,
                    nombres_variables_trx,
                    formato_fecha,
                    col_hora,
                    tipo_trx_invalidas,
                    nombre_archivo_gps,
                    nombres_variables_gps)

    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }

    conn_data = utils.iniciar_conexion_db(tipo='data')
    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')

    trx = pd.read_sql("select * from transacciones", conn_data)

    # testear formateo de columnas
    cols = ['id', 'id_original', 'id_tarjeta', 'fecha', 'dia', 'tiempo',
            'hora', 'modo', 'id_linea', 'id_ramal', 'interno', 'orden_trx',
            'latitud', 'longitud', 'factor_expansion']
    assert all(trx.columns.isin(cols))

    # testear que no haya faltantes
    for c in ["id_tarjeta", "fecha", "id_linea", "latitud", "longitud"]:
        assert trx[c].notna().all()

    # testear que no haya tarjetas con trx unica
    assert (trx.groupby('id_tarjeta').size() > 1).all()

    # longitud del string igual para todas las tarjetas
    assert trx.id_tarjeta.str.len().std() == 0

    legs.create_legs_from_transactions(trx_order_params)

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    # imputar destinos minimizando distancia
    q = """
    select e.*
    from etapas e
    order by dia,id_tarjeta,id_viaje,id_etapa,hora,tiempo
    """
    etapas_sin_d = pd.read_sql_query(q, conn_data)
    etapas_sin_d = etapas_sin_d.drop(['h3_d', 'od_validado'], axis=1)

    # add id_linea_agg
    routes.process_routes_metadata()

    metadata_lineas = pd.read_sql_query(
        """
        SELECT *
        FROM metadata_lineas
        """,
        conn_insumos,
    )

    etapas_sin_d = etapas_sin_d.merge(metadata_lineas[['id_linea',
                                                       'id_linea_agg']],
                                      how='left',
                                      on='id_linea')

    etapas_destinos_potencial = dest.imputar_destino_potencial(etapas_sin_d)
    destinos = dest.imputar_destino_min_distancia(etapas_destinos_potencial)

    etapas_sin_d = etapas_sin_d.drop('h3_d', axis=1)
    etapas = etapas_sin_d.merge(destinos, on=['id'], how='left')

    etapas = etapas\
        .sort_values(
            ['dia', 'id_tarjeta', 'id_viaje', 'id_etapa', 'hora', 'tiempo'])\
        .reset_index(drop=True)

    etapas['od_validado'] = etapas['od_validado'].fillna(0).astype(int)
    etapas['h3_d'] = etapas['h3_d'].fillna('')

    etapa = etapas.loc[etapas.id_tarjeta == '3839538659', :]

    etapa = etapa.reindex(columns=['id_viaje', 'id_etapa',
                                   'h3_o', 'h3_d', 'od_validado'])
    # casos para armar tests con nuevos destinos
    # tarjeta 3839538659. la vuelta en tren que termine en la estacion de tren
    assert (etapa.loc[(etapa.id_viaje == 2) & (etapa.id_etapa == 2), 'h3_d']
            == '88c2e38b23fffff').iloc[0]

    # tarjeta 0037035823. vuelve a la casa en otra linea y el destinio no es
    #  en la primera trx del dia. sino en la de la parada de la linea
    etapa = etapas.loc[etapas.id_tarjeta == '0037035823', :]

    assert (etapa.loc[(etapa.id_viaje == 3) & (etapa.id_etapa == 1), 'h3_d']
            == '88c2e3a1a7fffff').iloc[0]

    # tarjeta 1939538599 se toma el subte fin del dia y su primer viaje fue
    # en villa fiorito
    etapa = etapas.loc[etapas.id_tarjeta == '1939538599', :]
    assert (etapa.loc[(etapa.id_viaje == 3) & (
        etapa.id_etapa == 2), 'h3_d'].iloc[0] == '')


def test_viz_lowes():

    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]
    nombre_archivo_gps = configs["nombre_archivo_gps"]
    nombres_variables_gps = configs["nombres_variables_gps"]

    create_test_trx(geolocalizar_trx_config,
                    nombre_archivo_trx,
                    nombres_variables_trx,
                    formato_fecha,
                    col_hora,
                    tipo_trx_invalidas,
                    nombre_archivo_gps,
                    nombres_variables_gps)

    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }
    legs.create_legs_from_transactions(trx_order_params)

    conn_data = utils.iniciar_conexion_db(tipo='data')
    q = "select * from etapas"
    etapas = pd.read_sql(q, conn_data)

    recorridos_lowess = etapas.groupby(
        'id_linea').apply(geo.lowess_linea).reset_index()

    assert recorridos_lowess.geometry.type.unique()[0] == 'LineString'
    alias = ''
    id_linea = 16
    viz.plotear_recorrido_lowess(
        id_linea=id_linea, etapas=etapas, recorridos_lowess=recorridos_lowess,
        alias=alias,)
    file_path = os.path.join(
        "resultados", "png", f"{alias}linea_{id_linea}.png")
    assert os.path.isfile(file_path)


def test_section_load_viz(matriz_validacion_test_amba):

    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]
    nombre_archivo_gps = configs["nombre_archivo_gps"]
    nombres_variables_gps = configs["nombres_variables_gps"]

    create_test_trx(geolocalizar_trx_config,
                    nombre_archivo_trx,
                    nombres_variables_trx,
                    formato_fecha,
                    col_hora,
                    tipo_trx_invalidas,
                    nombre_archivo_gps,
                    nombres_variables_gps)

    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }
    resolucion_h3 = configs["resolucion_h3"]
    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    ring_size = geo.get_h3_buffer_ring_size(
        resolucion_h3, tolerancia_parada_destino
    )

    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')
    routes.process_routes_metadata()
    routes.process_routes_geoms()

    legs.create_legs_from_transactions(trx_order_params)

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    carto.update_stations_catchment_area(ring_size=ring_size)

    dest.infer_destinations()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

    carto.create_zones_table()
    # Infer route geometries based on legs data
    routes.infer_routes_geoms(plotear_lineas=False)

    # Build final routes from official an inferred sources
    routes.build_routes_from_official_inferred()

    kpi.compute_route_section_load(id_linea=32, rango_hrs=False)
    viz.visualize_route_section_load(id_linea=32, rango_hrs=False)


def test_viz(matriz_validacion_test_amba):

    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]
    nombre_archivo_gps = configs["nombre_archivo_gps"]
    nombres_variables_gps = configs["nombres_variables_gps"]

    create_test_trx(geolocalizar_trx_config,
                    nombre_archivo_trx,
                    nombres_variables_trx,
                    formato_fecha,
                    col_hora,
                    tipo_trx_invalidas,
                    nombre_archivo_gps,
                    nombres_variables_gps)

    routes.process_routes_metadata()

    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }
    resolucion_h3 = configs["resolucion_h3"]
    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    ring_size = geo.get_h3_buffer_ring_size(
        resolucion_h3, tolerancia_parada_destino
    )

    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')
    conn_data = utils.iniciar_conexion_db(tipo='data')

    legs.create_legs_from_transactions(trx_order_params)

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    carto.update_stations_catchment_area(ring_size=ring_size)

    dest.infer_destinations()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

    carto.create_zones_table()

    carto.create_voronoi_zones()

    viz.create_visualizations()

    viajes = pd.read_sql("select * from viajes", conn_data)
    viajes['distance_osm_drive'] = 0
    viajes['h3_o_norm'] = viajes.h3_o
    viajes['h3_d_norm'] = viajes.h3_d
    viajes['factor_expansion'] = 1

    viz.imprime_burbujas(viajes,
                         res=7,
                         h3_o='h3_o',
                         alpha=.4,
                         cmap='flare',
                         var_fex='',
                         porc_viajes=100,
                         title=f'Hogares',
                         savefile=f'testing_burb_hogares',
                         show_fig=True,
                         k_jenks=1)

    viz.imprime_lineas_deseo(df=viajes,
                             h3_o='',
                             h3_d='',
                             var_fex='',
                             title=f'Lineas de deseo',
                             savefile='Lineas de deseo',
                             k_jenks=1)

    viz.imprimir_matrices_od(viajes,
                             savefile='viajes',
                             title='Matriz OD',
                             var_fex="")

    zonas = pd.read_sql("select * from zonas;", conn_insumos)
    df, matriz_zonas = viz.traigo_zonificacion(
        viajes, zonas, h3_o='h3_o', h3_d='h3_d')

    for i in matriz_zonas:
        var_zona = i[1]
        matriz_order = i[2]

        viz.imprime_od(
            df,
            zona_origen=f"{var_zona}_o",
            zona_destino=f"{var_zona}_d",
            var_fex='',
            x_rotation=90,
            normalize=True,
            cmap="Reds",
            title='Matriz OD General',
            figsize_tuple='',
            matriz_order=matriz_order,
            savefile=f"{var_zona}",
            margins=True,
        )


def test_gps(matriz_validacion_test_amba):

    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = True
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = "%d/%m/%Y %H:%M:%S"
    col_hora = False
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = "transacciones_amba_test_geocode.csv"
    nombre_archivo_gps = "gps_amba_test.csv"
    nombres_variables_gps = {
        'id_gps': 'id_gps',
        'id_linea_gps': 'id_linea_gps',
        'id_ramal_gps': 'id_ramal_gps',
        'interno_gps': 'interno_gps',
        'fecha_gps': 'fecha_gps',
        'latitud_gps': 'latitud_gps',
        'longitud_gps': 'longitud_gps',
    }
    trx_order_params = {
        "criterio": "fecha_completa",
        "ventana_viajes": 120,
        "ventana_duplicado": 5,
    }
    resolucion_h3 = configs["resolucion_h3"]
    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    ring_size = geo.get_h3_buffer_ring_size(
        resolucion_h3, tolerancia_parada_destino
    )

    create_test_trx(geolocalizar_trx_config,
                    nombre_archivo_trx,
                    nombres_variables_trx,
                    formato_fecha,
                    col_hora,
                    tipo_trx_invalidas,
                    nombre_archivo_gps,
                    nombres_variables_gps)

    routes.process_routes_metadata()

    legs.create_legs_from_transactions(trx_order_params)

    # confirm latlong for card_id 37030208
    conn_insumos = utils.iniciar_conexion_db(tipo='insumos')
    conn_data = utils.iniciar_conexion_db(tipo='data')

    trx = pd.read_sql("select * from transacciones", conn_data)
    gps = pd.read_sql("select * from gps", conn_data)

    assert len(trx) == 2

    gps_latlong = gps.loc[gps.id_original == 2, ['latitud', 'longitud']]
    trx_latlong = trx.loc[trx.id_original ==
                          '2189303', ['latitud', 'longitud']]

    assert gps_latlong.latitud.item() == trx_latlong.latitud.item()
    assert gps_latlong.longitud.item() == trx_latlong.longitud.item()

    gps_latlong = gps.loc[gps.id_original == 13, ['latitud', 'longitud']]
    trx_latlong = trx.loc[trx.id_original ==
                          '2189304', ['latitud', 'longitud']]

    assert gps_latlong.latitud.item() == trx_latlong.latitud.item()
    assert gps_latlong.longitud.item() == trx_latlong.longitud.item()

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    carto.update_stations_catchment_area(ring_size=ring_size)

    dest.infer_destinations()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

    carto.create_distances_table(use_parallel=True)

    distancias = pd.read_sql("select * from distancias;", conn_insumos)
    mean_distances = distancias.distance_osm_drive.mean()
    assert len(distancias) > 0

    kpi.compute_kpi()

    q = """
        SELECT e.dia,e.id_tarjeta,e.factor_expansion_linea as factor_expansion
        from etapas e
    """
    fe = pd.read_sql(q, conn_data)
    tot_pax = fe.factor_expansion.sum()

    kpi_df = pd.read_sql(
        "select * from kpi_by_day_line;", conn_data)

    assert round(kpi_df.tot_km.iloc[0]) == 16
    assert kpi_df.tot_veh.iloc[0] == 2
    assert kpi_df.dmt_mean.iloc[0] == mean_distances
    assert kpi_df.tot_pax.iloc[0] == tot_pax

    carto.create_zones_table()

    # Persist datamodel into csv tables
    misc.persist_datamodel_tables()
