"""Tramos GPS con velocidad imposible (`corregir_distancias_implausibles`).

La distancia entre dos puntos GPS consecutivos se mide por la red vial. Cuando
esa distancia implica una velocidad mayor al tope hay dos casos: un salto del GPS
(también imposible en línea recta, el tramo pasa a 0) o un artefacto de la red
(posible en línea recta, se usa la distancia en línea recta).
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from urbantrips.datamodel import transactions
from urbantrips.datamodel.transactions import corregir_distancias_implausibles
from urbantrips.geo.geo import distancia_recta_km
from urbantrips.utils.utils import leer_tuning_num


def _corregir(distancia, recta, delta, tope=100.0):
    corregida, salto, artefacto = corregir_distancias_implausibles(
        distancia, recta, delta, tope_kmh=tope
    )
    return corregida.tolist(), salto.tolist(), artefacto.tolist()


def test_tramo_normal_no_se_toca():
    # 1 km en 4 minutos: 15 km/h
    corregida, salto, artefacto = _corregir([1.0], [0.9], [240])
    assert corregida == [1.0]
    assert salto == [False] and artefacto == [False]


def test_tramo_rapido_real_de_autopista_no_se_toca():
    # 6.2 km por red en 240 s son 93 km/h: rapido pero debajo del tope
    corregida, salto, artefacto = _corregir([6.2], [6.0], [240])
    assert corregida == [6.2]
    assert salto == [False] and artefacto == [False]


def test_salto_del_gps_pasa_a_cero():
    # 20 km en 60 s (1200 km/h), tambien imposible en linea recta
    corregida, salto, artefacto = _corregir([20.0], [18.0], [60])
    assert corregida == [0.0]
    assert salto == [True] and artefacto == [False]


def test_artefacto_de_red_usa_la_linea_recta():
    # 250 m por red en 5 s (180 km/h) pero 30 m en linea recta (21.6 km/h)
    corregida, salto, artefacto = _corregir([0.25], [0.03], [5])
    assert corregida == [0.03]
    assert salto == [False] and artefacto == [True]


def test_autopista_con_red_sobrestimada_usa_la_linea_recta():
    # 7.35 km por red en 240 s (110 km/h) y 6.06 km en linea recta (91 km/h)
    corregida, salto, artefacto = _corregir([7.35], [6.06], [240])
    assert corregida == [6.06]
    assert artefacto == [True]


def test_intervalo_invalido_o_sin_recta_no_se_toca():
    corregida, salto, artefacto = _corregir(
        [5.0, 5.0, 5.0], [4.0, np.nan, 4.0], [np.nan, 10, -3]
    )
    assert corregida == [5.0, 5.0, 5.0]
    assert not any(salto) and not any(artefacto)


def test_mismo_segundo_con_salto_pasa_a_cero():
    # dos puntos en el mismo segundo a 4 km en linea recta: se evalua con 1 s
    corregida, salto, artefacto = _corregir([5.0], [4.0], [0])
    assert corregida == [0.0]
    assert salto == [True] and artefacto == [False]


def test_mismo_segundo_con_artefacto_de_red_usa_la_linea_recta():
    # 550 m por red y 10 m en linea recta (36 km/h en 1 s)
    corregida, salto, artefacto = _corregir([0.55], [0.01], [0])
    assert corregida == [0.01]
    assert salto == [False] and artefacto == [True]


def test_mismo_segundo_con_movimiento_posible_no_se_toca():
    # 20 m por red y 15 m en linea recta: menos de 100 km/h en 1 s
    # el primer punto de cada vehiculo llega asi: intervalo 0, sin recta, 0 km
    corregida, salto, artefacto = _corregir([0.02, 0.0], [0.015, np.nan], [0, 0])
    assert corregida == [0.02, 0.0]
    assert not any(salto) and not any(artefacto)


def test_el_tope_es_configurable():
    # 6.2 km en 240 s son 93 km/h: con tope 80 pasa a corregirse
    corregida, salto, artefacto = _corregir([6.2], [5.0], [240], tope=80.0)
    assert corregida == [5.0]
    assert artefacto == [True]


def test_tope_por_defecto_sale_de_tuning():
    with patch(
        "urbantrips.utils.utils.leer_configs_tuning",
        return_value={"gps_velocidad_maxima_kmh": 80},
    ):
        assert transactions._velocidad_maxima_gps_kmh() == 80.0
    with patch(
        "urbantrips.utils.utils.leer_configs_tuning",
        return_value={"gps_velocidad_maxima_kmh": None},
    ):
        assert transactions._velocidad_maxima_gps_kmh() == 100.0


@pytest.mark.parametrize("valor", [-50, float("nan"), float("inf"), "rapido", True])
def test_tuning_invalido_usa_el_default_y_avisa(valor, caplog):
    with patch(
        "urbantrips.utils.utils.leer_configs_tuning",
        return_value={"gps_velocidad_maxima_kmh": valor},
    ), caplog.at_level("WARNING"):
        assert leer_tuning_num("gps_velocidad_maxima_kmh", 100.0) == 100.0
    assert "gps_velocidad_maxima_kmh" in caplog.text


@pytest.mark.parametrize("valor", [None, 0, ""])
def test_tuning_vacio_usa_el_default_sin_avisar(valor, caplog):
    with patch(
        "urbantrips.utils.utils.leer_configs_tuning",
        return_value={"gps_velocidad_maxima_kmh": valor},
    ), caplog.at_level("WARNING"):
        assert leer_tuning_num("gps_velocidad_maxima_kmh", 100.0) == 100.0
    assert caplog.text == ""


def test_tuning_valido_se_convierte_al_tipo():
    with patch(
        "urbantrips.utils.utils.leer_configs_tuning",
        return_value={"filas": "5e6", "tope": "80"},
    ):
        valor = leer_tuning_num("filas", 1, int)
        assert valor == 5_000_000 and isinstance(valor, int)
        assert leer_tuning_num("tope", 100.0) == 80.0


def test_distancia_recta_km():
    # un grado de latitud son unos 111.2 km
    d = distancia_recta_km([-34.0], [-58.0], [-35.0], [-58.0])
    assert d[0] == pytest.approx(111.2, abs=0.1)
    assert np.isnan(distancia_recta_km([np.nan], [-58.0], [-35.0], [-58.0])[0])


def test_compute_distance_km_gps_corrige_por_vehiculo():
    """De punta a punta, con la distancia por red simulada.

    Un vehículo con cuatro puntos: el segundo tramo es un artefacto de la red y
    el tercero un salto. Otro vehículo arranca lejos del primero: su primer punto
    no se compara con el último del otro vehículo.
    """
    lat0, lon0 = -32.90, -68.85
    paso = 0.0045  # unos 500 m de latitud
    gps = pd.DataFrame(
        {
            "dia": ["2025-10-25"] * 6,
            "id_linea": [1] * 6,
            "id_ramal": [1] * 6,
            "interno": [1, 1, 1, 1, 2, 2],
            "fecha": [0, 120, 125, 185, 60, 180],
            "latitud": [lat0, lat0 + paso, lat0 + paso + 0.0003, lat0 + 0.5,
                        lat0 - 0.3, lat0 - 0.3 + paso],
            "longitud": [lon0] * 6,
        }
    )
    # distancia por red de cada punto contra el anterior del mismo vehiculo
    red = {0: 0.0, 120: 0.55, 125: 0.30, 185: 60.0, 60: 0.0, 180: 0.52}

    def _od(od_df, distance_col, **kwargs):
        out = od_df.copy()
        out[distance_col] = out["fecha"].map(red)
        return out

    with patch.object(
        transactions, "leer_configs_generales",
        return_value={"lineas_contienen_ramales": False},
    ), patch.object(transactions, "compute_od_distances", side_effect=_od), patch.object(
        transactions, "_velocidad_maxima_gps_kmh", return_value=100.0
    ):
        out = transactions.compute_distance_km_gps(gps, ctx=None)

    out = out.sort_values(["interno", "fecha"])
    d = out["distance_km"].tolist()
    assert d[0] == 0.0
    assert d[1] == 0.55                          # 16.5 km/h: normal
    assert d[2] == pytest.approx(0.033, abs=0.002)  # 216 km/h por red, 24 en recta
    assert d[3] == 0.0                           # salto de 55 km en 60 s
    assert d[4] == 0.0                           # primer punto del otro vehiculo
    assert d[5] == 0.52                          # 15.6 km/h: normal
    assert "_recta_km" not in out.columns
