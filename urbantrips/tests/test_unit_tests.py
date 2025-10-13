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

    df_trx["tiempo"] = None
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
        ventana_viajes=120,
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

    assert (
        trx_5.loc[trx_5["id"].isin([15, 16, 17]), "id_tarjeta"] == ["2_0", "2_1", "2_2"]
    ).all()

    trx_1, tarjetas_duplicadas_1 = legs.cambiar_id_tarjeta_trx_simul_fecha(
        trx.copy(), ventana_duplicado=1
    )

    assert len(tarjetas_duplicadas_1) == 1
    assert (tarjetas_duplicadas_1.id_tarjeta_original == ["2"]).all()
    assert (tarjetas_duplicadas_1.id_tarjeta_nuevo == ["2_1"]).all()
    assert (
        trx_1.loc[trx_1["id"].isin([15, 16, 17]), "id_tarjeta"] == ["2_0", "2_1", "2_0"]
    ).all()
