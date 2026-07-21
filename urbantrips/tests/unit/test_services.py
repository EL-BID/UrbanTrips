import pandas as pd

import numpy as np

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
