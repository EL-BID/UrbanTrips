"""Equivalencia de los atajos de memoria del ingest de gps a escala mes.

Una corrida de un mes en UN solo archivo (el caso del cliente: 104,5 M pings gps en
`ut_amba_202607_gps.csv`) no entra en un DataFrame. `process_and_upload_gps_table`
pasó a leer el csv por chunks y a resolver en SQL lo que necesita ver el archivo
entero. Estos tests fijan que las piezas que se reescribieron devuelven EXACTAMENTE lo
mismo que la versión que tenía todo en memoria; la parte SQL (dedup, id interno,
odómetro) se verifica en `integration/adapters/test_gps_prep_sql.py`.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from urbantrips.datamodel.transactions import (
    _veh_exp_acumular,
    _veh_exp_final,
    convertir_fechas,
    crear_id_interno,
    get_veh_expansion_from_gps,
)


def _gps_sintetico(n=4000, seed=0):
    """Frame gps con longitudes nulas, en cero y normales, para que aparezcan las
    tres situaciones que distingue `all_gps_broken`."""
    rng = np.random.default_rng(seed)
    lon = rng.choice([-58.4, -58.5, 0.0, np.nan], size=n, p=[0.4, 0.3, 0.2, 0.1])
    gps = pd.DataFrame(
        {
            "id_linea": rng.integers(1, 6, size=n),
            "dia": rng.choice(["2026-07-01", "2026-07-02", "2026-07-03"], size=n),
            "interno": rng.integers(1, 12, size=n),
            "latitud": -34.6,
            "longitud": lon,
        }
    )
    # vehículos enteros rotos: uno todo en cero y otro todo nulo
    gps.loc[(gps.id_linea == 1) & (gps.interno == 1), "longitud"] = 0.0
    gps.loc[(gps.id_linea == 2) & (gps.interno == 2), "longitud"] = np.nan
    # y una clave nula, que el groupby tiene que descartar
    gps.loc[gps.index[:5], "interno"] = np.nan
    return gps


def _acumular_por_chunks(gps, n_chunks):
    acumulado = None
    for pedazo in np.array_split(gps, n_chunks):
        acumulado = _veh_exp_acumular(acumulado, pedazo)
    return _veh_exp_final(acumulado)


@pytest.mark.parametrize("n_chunks", [1, 2, 7, 50])
def test_factores_de_expansion_por_chunks_dan_lo_mismo_que_el_archivo_entero(n_chunks):
    gps = _gps_sintetico()
    esperado = get_veh_expansion_from_gps(gps).reset_index(drop=True)
    obtenido = _acumular_por_chunks(gps, n_chunks).reset_index(drop=True)
    pd.testing.assert_frame_equal(esperado, obtenido, check_dtype=False)


def test_factores_de_expansion_detectan_los_vehiculos_rotos():
    """Contra la definición, no contra la otra implementación: un vehículo está roto
    si TODAS sus longitudes son 0 o TODAS son nulas."""
    gps = pd.DataFrame(
        {
            "id_linea": [1, 1, 1, 1, 1, 1],
            "dia": ["2026-07-01"] * 6,
            "interno": [10, 10, 20, 20, 30, 30],
            "longitud": [0.0, 0.0, np.nan, np.nan, -58.4, 0.0],
        }
    )
    veh_exp = _acumular_por_chunks(gps, 3)
    assert veh_exp["unique_vehicles"].tolist() == [3]
    assert veh_exp["broken_gps_veh"].tolist() == [2]  # el 10 y el 20, no el 30
    # 3/(3-2) = 3, pero el factor está topeado en 2
    assert veh_exp["veh_exp"].tolist() == [2.0]


def test_factores_de_expansion_sin_filas():
    vacio = _veh_exp_final(None)
    assert list(vacio.columns) == [
        "id_linea",
        "dia",
        "unique_vehicles",
        "broken_gps_veh",
        "veh_exp",
    ]
    assert len(vacio) == 0


def test_dia_tiene_los_mismos_valores_que_strftime():
    fechas = pd.date_range("2026-07-01 05:00", periods=5000, freq="17min")
    df = pd.DataFrame({"fecha": fechas.strftime("%d/%m/%Y %H:%M:%S")})
    esperado = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M:%S").dt.strftime(
        "%Y-%m-%d"
    )
    salida = convertir_fechas(df.copy(), "%d/%m/%Y %H:%M:%S")
    assert salida["dia"].tolist() == esperado.tolist()


def test_dia_comparte_un_solo_string_por_dia():
    """Lo que baja la RAM: 1 objeto str por día, no uno por fila (6,2 GB → 0,7 GB
    sobre el mes del cliente)."""
    fechas = pd.date_range("2026-07-01 05:00", periods=5000, freq="17min")
    df = pd.DataFrame({"fecha": fechas.strftime("%d/%m/%Y %H:%M:%S")})
    salida = convertir_fechas(df, "%d/%m/%Y %H:%M:%S")
    n_objetos = len({id(x) for x in salida["dia"].to_numpy()})
    assert n_objetos == salida["dia"].nunique()
    assert n_objetos < len(salida)


def test_dia_ignora_las_filas_con_fecha_invalida():
    df = pd.DataFrame(
        {"fecha": ["01/07/2026 05:00:00", "no es fecha", "02/07/2026 06:00:00"]}
    )
    salida = convertir_fechas(df, "%d/%m/%Y %H:%M:%S")
    assert salida["dia"].tolist() == ["2026-07-01", "2026-07-02"]


def test_crear_id_interno_devuelve_un_array_int64():
    """`list(range(...))` materializaba un int de Python por fila (~3,9 GB sobre las
    104,5 M filas gps del mes)."""
    ctx = MagicMock()
    ctx.data.get_max_id.return_value = 100
    ids = crear_id_interno(ctx, n_rows=5, tipo_tabla="gps")
    assert isinstance(ids, np.ndarray)
    assert ids.dtype == np.int64
    assert ids.tolist() == [100, 101, 102, 103, 104]
