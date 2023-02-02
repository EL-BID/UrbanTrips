from urbantrips.datamodel import legs, misc, legs, trips
from urbantrips.datamodel import transactions
from numpy import dtype
import pandas as pd
import os
import sys
import pytest
import itertools
import os

from urbantrips.destinations import destinations as dest
from urbantrips.datamodel import legs
from urbantrips.geo import geo
from urbantrips.utils import utils
from urbantrips.carto import carto
from urbantrips.viz import viz


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


def create_test_trx():
    for tipo in ['data', 'insumos']:
        filePath = utils.traigo_db_path(tipo=tipo)
        if os.path.exists(filePath):
            os.remove(filePath)

    print("Abriendo archivos de configuracion")
    configs = utils.leer_configs_generales()

    # crear_directorios:
    utils.crear_directorios()

    # crear base de datos:
    utils.crear_base()

    transactions.create_transactions()
    return configs


def test_amba_integration(matriz_validacion_test_amba):
    configs = create_test_trx()

    criterio_orden_transacciones = {
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

    legs.create_legs_from_transactions(criterio_orden_transacciones)

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    # imputar destinos
    dest.infer_destinations()

    q = """
    select e.*,d.h3_d,d.od_validado
    from etapas e, destinos d
    where e.id = d.id
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

    configs = create_test_trx()

    criterio_orden_transacciones = {
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

    legs.create_legs_from_transactions(criterio_orden_transacciones)

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    # imputar destinos minimizando distancia
    q = """
    select e.*
    from etapas e
    left join  (select *, 1 as en_destinos from destinos) d
    on e.id = d.id
    where d.id is null
    order by dia,id_tarjeta,id_viaje,id_etapa,hora,tiempo
    """
    etapas_sin_d = pd.read_sql_query(q, conn_data)

    etapas_destinos_potencial = dest.imputar_destino_potencial(etapas_sin_d)
    destinos = dest.imputar_destino_min_distancia(etapas_destinos_potencial)
    destinos.to_sql("destinos", conn_data, if_exists="append", index=False)

    q = """
    select e.*,d.h3_d,d.od_validado
    from etapas e, destinos d
    where e.id = d.id
    """
    etapas = pd.read_sql(q, conn_data)

    etapa = etapas.loc[etapas.id_tarjeta == '3839538659', :]

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
        etapa.id_etapa == 2), 'h3_d'].isna().iloc[0])


def test_viz_lowes():

    configs = create_test_trx()
    criterio_orden_transacciones = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }
    legs.create_legs_from_transactions(criterio_orden_transacciones)

    conn_data = utils.iniciar_conexion_db(tipo='data')
    q = "select * from etapas"
    etapas = pd.read_sql(q, conn_data)

    recorridos_lowess = etapas.groupby(
        'id_linea').apply(carto.lowess_linea).reset_index()

    assert recorridos_lowess.geometry.type.unique()[0] == 'LineString'
    alias = ''
    id_linea = 16
    viz.plotear_recorrido_lowess(
        id_linea=id_linea, etapas=etapas, recorridos_lowess=recorridos_lowess, alias=alias)
    file_path = os.path.join(
        "resultados", "png", f"{alias}linea_{id_linea}.png")
    assert os.path.isfile(file_path)


def test_carto(matriz_validacion_test_amba):

    configs = create_test_trx()

    criterio_orden_transacciones = {
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

    legs.create_legs_from_transactions(criterio_orden_transacciones)

    # actualizar matriz de validacion
    matriz_validacion_test_amba.to_sql(
        "matriz_validacion", conn_insumos, if_exists="replace", index=False)

    carto.update_stations_catchment_area(ring_size=ring_size)

    dest.infer_destinations()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

    # Create TAZs
    carto.create_zones_table()

    # Create voronoi TAZs
    carto.create_voronoi_zones()

    zonas = pd.read_sql("select * from zonas", conn_insumos)
    assert len(zonas) > 0

    conn_data = utils.iniciar_conexion_db(tipo='data')

    conn_data.execute(
        """delete from viajes where id_tarjeta <> '0037030208';"""
    )
    conn_data.execute(
        """delete from etapas where id_tarjeta <> '0037030208';"""
    )
    conn_data.commit()
    conn_data.close()

    carto.create_distances_table(use_parallel=True)
    distancias = pd.read_sql("select * from distancias;", conn_insumos)

    assert len(distancias) > 0

    viz.create_visualizations()
