import pandas as pd

import numpy as np
import pytest

from urbantrips.datamodel.services import (
    classify_line_gps_points_into_services,
    compute_new_services_stats,
    gps_service_distances_to_km,
)


def test_gps_service_distances_to_km_converts_odometer_metres_to_km():
    """El nivel servicio debe dejar distance_route_gps en KM, no en metros.

    distance_servicio_mts (odómetro) viene en metros y debe dividirse por 1000
    para quedar en la misma unidad que distance_route (km). Sin esto, todo lo
    derivado (tot_km_route_gps, ipk_route_gps, velocidad_comercial_route_gps...)
    quedaba inflado 1000x. Guarda contra la regresión del bug de unidades.
    """
    gps_points = pd.DataFrame({
        "distance_km": [1.5, 3.0, 0.0],            # ping-based, ya en km
        "distance_servicio_mts": [1500.0, 3000.0, None],  # odómetro en METROS
        "interno": ["1", "1", "2"],
    })

    out = gps_service_distances_to_km(gps_points)

    # distance_route = distance_km (sin cambios de unidad)
    assert out["distance_route"].tolist() == [1.5, 3.0, 0.0]
    # distance_route_gps = metros / 1000 -> km; NULL -> NaN (odómetro ausente)
    assert out["distance_route_gps"].iloc[0] == 1.5
    assert out["distance_route_gps"].iloc[1] == 3.0
    assert np.isnan(out["distance_route_gps"].iloc[2])
    # la columna cruda en metros ya no debe quedar
    assert "distance_servicio_mts" not in out.columns
    # ambas familias en la misma escala (ratio ~1, no ~1000)
    m = out["distance_route"] > 0
    ratio = (out.loc[m, "distance_route_gps"] / out.loc[m, "distance_route"]).dropna()
    assert (ratio.between(0.5, 2.0)).all()


def test_classify_services_from_gps_markers_vectorized_per_vehicle():
    gps = pd.DataFrame(
        {
            "id": [3, 1, 2, 4, 5],
            "dia": ["2022-08-11"] * 5,
            "id_linea": [10] * 5,
            "id_ramal": [100] * 5,
            "interno": [1, 1, 1, 2, 2],
            "fecha": [30, 10, 20, 10, 20],
            "service_type": [None, "start_service", None, None, "start_service"],
            "distance_route": [0.2, 0.3, 0.0, 0.4, 0.5],
        }
    )

    result = classify_line_gps_points_into_services(
        gps,
        line_stops_gdf=None,
        trust_service_type_gps=True,
    )

    result = result.sort_values(["interno", "fecha"]).reset_index(drop=True)

    assert result["original_service_id"].tolist() == [1, 1, 1, 0, 1]
    assert result["new_service_id"].tolist() == [1, 1, 1, 0, 1]
    assert result["service_id"].tolist() == [0, 0, 0, 0, 1]
    assert result["idling"].tolist() == [False, True, False, False, False]


def test_compute_new_services_stats_handles_zero_original_distance():
    services = pd.DataFrame(
        {
            "id_linea": [10],
            "id_ramal": [100],
            "dia": ["2022-08-11"],
            "interno": [1],
            "original_service_id": [1],
            "service_id": [0],
            "valid": [True],
            "total_points": [6],
            "prop_idling": [0.0],
            "distance_route": [0.0],
        }
    )

    result = compute_new_services_stats(services)

    assert result.loc[0, "distancia_recorrida_original"] == 0
    assert result.loc[0, "prop_distancia_recuperada"] is None


def test_compute_new_services_stats_uses_unrounded_distance_for_recovered_ratio():
    services = pd.DataFrame(
        {
            "id_linea": [10, 10],
            "id_ramal": [100, 100],
            "dia": ["2022-08-11", "2022-08-11"],
            "interno": [1, 1],
            "original_service_id": [1, 1],
            "service_id": [0, 1],
            "valid": [True, False],
            "total_points": [6, 6],
            "prop_idling": [0.0, 0.0],
            "distance_route": [0.2, 0.2],
        }
    )

    result = compute_new_services_stats(services)

    assert result.loc[0, "distancia_recorrida_original"] == 0
    assert result.loc[0, "prop_distancia_recuperada"] == 0.5


# --- Validez de servicios por tiempo detenido --------------------------------
# Un servicio se descartaba si mas de la mitad de sus PUNTOS avanzaba menos de
# 100 m. Con equipos GPS que emiten rafagas de puntos al detenerse (Mendoza: uno
# cada ~4 s parado, uno cada ~70 s en movimiento) se descartaban servicios
# normales junto con sus km. La regla mide ahora TIEMPO detenido.


def _traza(tramos, id_linea=10, id_ramal=100, interno=1, dia="2025-10-27"):
    """Arma los puntos GPS de un servicio a partir de (segundos, km) por tramo."""
    t = 1_761_500_000
    filas = [{"fecha": t, "distance_route": 0.0}]
    for seg, km in tramos:
        t += seg
        filas.append({"fecha": t, "distance_route": km})
    df = pd.DataFrame(filas)
    df["id"] = range(len(df))
    df["id_linea"], df["id_ramal"], df["interno"], df["dia"] = id_linea, id_ramal, interno, dia
    df["distance_route_gps"] = np.nan
    df["original_service_id"], df["service_id"] = 1, 0
    return df


def _servicio(tramos):
    from urbantrips.datamodel.services import (
        create_line_services_table,
        marcar_puntos_detenidos,
    )

    puntos = _traza(tramos)
    puntos["idling"] = marcar_puntos_detenidos(
        puntos, ["dia", "id_ramal", "interno"], velocidad_detenido_kmh=2.0
    )
    return puntos, create_line_services_table(puntos).iloc[0]


def test_punto_detenido_se_define_por_velocidad_no_por_distancia():
    # 80 m en 10 s son 28,8 km/h: en movimiento, aunque sean menos de 100 m.
    # 80 m en 240 s son 1,2 km/h: detenido.
    puntos, _ = _servicio([(10, 0.08), (240, 0.08), (240, 1.2)])
    assert puntos["idling"].tolist() == [True, False, True, False]


def test_marcar_puntos_detenidos_respeta_el_orden_original_de_las_filas():
    from urbantrips.datamodel.services import marcar_puntos_detenidos

    puntos = _traza([(10, 0.08), (240, 0.08), (240, 1.2)])
    desordenado = puntos.iloc[[3, 1, 0, 2]]
    detenido = marcar_puntos_detenidos(
        desordenado, ["dia", "id_ramal", "interno"], velocidad_detenido_kmh=2.0
    )
    assert detenido.tolist() == [False, False, True, True]


def test_servicio_con_gps_regular_en_movimiento_es_valido():
    # un punto cada 240 s avanzando 1,2 km (18 km/h), como el GPS de AMBA
    _, s = _servicio([(240, 1.2)] * 20)
    assert bool(s["valid"])
    assert s["prop_idling"] == 0
    assert s["motivo_no_valido"] is None
    assert s["horas_servicio"] == 20 * 240 / 3600


def test_servicio_con_rafagas_al_detenerse_es_valido():
    """Caso Mendoza: muchos puntos parado, pocos en movimiento, poco tiempo parado."""
    tramos = []
    for _ in range(15):
        tramos += [(70, 0.35)]          # 70 s en movimiento a 18 km/h
        tramos += [(4, 0.0)] * 6        # 24 s detenido, un punto cada 4 s
    puntos, s = _servicio(tramos)

    # con la regla anterior (proporcion de puntos) quedaba descartado
    assert (puntos["distance_route"] < 0.1).mean() > 0.5
    # por tiempo: 24 s detenido de cada 94 s
    assert s["prop_idling"] == round(24 / 94, 2)
    assert bool(s["valid"])
    assert s["distance_route"] == pytest.approx(15 * 0.35)


def test_servicio_realmente_detenido_no_es_valido():
    # mas de la mitad del tiempo parado, con GPS regular
    _, s = _servicio([(240, 1.2)] * 4 + [(240, 0.0)] * 10)
    assert s["prop_idling"] == round(10 / 14, 2)
    assert not bool(s["valid"])
    assert s["motivo_no_valido"] == "detenido"


def test_servicio_con_pocos_puntos_no_es_valido():
    _, s = _servicio([(240, 1.2)] * 4)      # 5 puntos
    assert not bool(s["valid"])
    assert s["motivo_no_valido"] == "pocos_puntos"


def test_primer_punto_del_servicio_no_suma_tiempo_del_servicio_anterior():
    from urbantrips.datamodel.services import create_line_services_table

    # dos servicios del mismo vehiculo separados por 2 horas de espera
    a = _traza([(240, 1.2)] * 8)
    b = _traza([(240, 1.2)] * 8)
    b["fecha"] += 8 * 240 + 7200
    b["service_id"] = 1
    puntos = pd.concat([a, b], ignore_index=True)
    puntos["idling"] = puntos["distance_route"] == 0   # el primer punto de cada uno

    servicios = create_line_services_table(puntos)

    assert servicios["horas_servicio"].tolist() == [8 * 240 / 3600] * 2
    assert servicios["prop_idling"].tolist() == [0.0, 0.0]
    assert servicios["valid"].all()


def test_columnas_nuevas_de_services_quedan_al_final():
    _, s = _servicio([(240, 1.2)] * 8)
    assert list(s.index) == [
        "id_linea", "id_ramal", "dia", "interno", "original_service_id",
        "service_id", "total_points", "distance_route", "distance_route_gps",
        "min_ts", "max_ts", "min_datetime", "max_datetime", "prop_idling", "valid",
        "horas_servicio", "motivo_no_valido",
    ]
