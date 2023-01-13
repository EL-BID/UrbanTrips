from numpy import dtype
import pandas as pd
import os
import sys
import pytest
import itertools
import os


# sys.path.insert(0, os.path.join(os.getcwd(), "od_matrix"))
# from utils import utils as utils  # noqa
# from utils import estructura as est  # noqa
# from utils import destinos as dest  # noqa
# from utils import limpieza as lmp  # noqa
from destinations import destinations as dest


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
    path = os.path.join(os.getcwd(), "od_matrix", "tests", "data")
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
