"""Precisión de los indicadores de kpis_lineas (UT-019, observación de Mendoza).

Los indicadores se redondeaban a 1 decimal al guardarse y después se promediaban
sobre esos valores ya redondeados. Ahora se guardan con DECIMALES_KPI decimales.
"""

import pandas as pd
import pytest

from urbantrips.kpi.kpi_lineas import DECIMALES_KPI, cal_velocidad_comercial


def _servicios():
    # Dos servicios de la misma línea y día: 10 km en 37 min y 12 km en 41 min.
    return pd.DataFrame(
        {
            "dia": ["2026-10-01", "2026-10-01"],
            "id_linea": [1, 1],
            "interno": [101, 102],
            "min_datetime": ["2026-10-01 07:00:00", "2026-10-01 07:10:00"],
            "max_datetime": ["2026-10-01 07:37:00", "2026-10-01 07:51:00"],
            "distance_route": [10.0, 12.0],
            "distance_route_gps": [10.0, 12.0],
        }
    )


def test_decimales_kpi_permite_distinguir_valores_chicos():
    # un FO de 0,14 no puede quedar en 0,1
    assert DECIMALES_KPI >= 3
    assert round(0.143, DECIMALES_KPI) == pytest.approx(0.143)


def test_velocidad_comercial_no_se_redondea_a_un_decimal():
    vel = cal_velocidad_comercial(_servicios())
    esperado = ((10 / (37 / 60)) + (12 / (41 / 60))) / 2  # 16,887... km/h
    valor = vel.loc[0, "velocidad_comercial_route"]
    assert valor == pytest.approx(round(esperado, DECIMALES_KPI))
    # con el redondeo anterior a 1 decimal daba 16,9
    assert valor != pytest.approx(round(esperado, 1))


def test_distancia_media_por_vehiculo_conserva_decimales():
    s = _servicios()
    s["distance_route"] = [10.123, 12.0]
    vel = cal_velocidad_comercial(s)
    assert vel.loc[0, "distancia_media_veh_route"] == pytest.approx(11.062, abs=1e-3)
