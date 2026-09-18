"""El gps de cada vehículo tiene que ordenarse antes de mirar el ping siguiente.

`compute_section_supply_stats` calcula, para cada ping, el ping siguiente del
mismo vehículo con `groupby(...).shift(-1)`, y de ahí saca el sentido, el delta
de tiempo y la velocidad. Pero `shift` toma el orden en que vienen las filas, y
la consulta que las trae no tiene `ORDER BY`.

En `abmza`, 58 de 72 vehículos venían con su gps desordenado, así que se
apareaban pings no consecutivos. Peor: agregarle un filtro de días a la consulta
cambiaba el plan de DuckDB, cambiaba el orden de 41.850 de 42.231 filas y con él
los resultados. Las otras dos funciones que hacen lo mismo sobre gps
(`compute_speed_by_day_veh_hour` y el ingest de transacciones) ya ordenaban.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from urbantrips.kpi.supply_kpi import compute_section_supply_stats


def _recorrido():
    """Un recorrido recto de oeste a este sobre el AMBA."""
    return gpd.GeoDataFrame(
        {"id_linea": [1], "n_sections": [10], "section_meters": [500]},
        geometry=[LineString([(-58.50, -34.60), (-58.45, -34.60)])],
        crs="EPSG:4326",
    )


def _gps_ordenado():
    """Un vehículo recorriendo la línea de punta a punta, ping a ping."""
    n = 12
    base = 1_700_000_000
    return pd.DataFrame(
        {
            "id_linea": [1] * n,
            "id_ramal": [1] * n,
            "interno": [7] * n,
            "dia": ["2026-03-09"] * n,
            "yr_mo": ["2026-03"] * n,
            # de oeste a este, un ping cada 10 minutos
            "longitud": [-58.50 + i * 0.004 for i in range(n)],
            "latitud": [-34.60] * n,
            "fecha": [base + i * 600 for i in range(n)],
            "distance_km": [0.4] * n,
        }
    )


def test_el_resultado_no_depende_del_orden_en_que_llega_el_gps():
    ordenado = _gps_ordenado()
    # el orden que devolvía DuckDB no es el cronológico
    desordenado = ordenado.sample(frac=1, random_state=0).reset_index(drop=True)

    esperado = compute_section_supply_stats(ordenado.copy(), _recorrido())
    obtenido = compute_section_supply_stats(desordenado.copy(), _recorrido())

    llave = ["dia", "sentido", "section_id"]
    esperado = esperado.sort_values(llave).reset_index(drop=True)
    obtenido = obtenido.sort_values(llave).reset_index(drop=True)

    pd.testing.assert_frame_equal(esperado, obtenido)


def test_el_vehiculo_que_avanza_sobre_el_recorrido_es_ida():
    stats = compute_section_supply_stats(
        _gps_ordenado().sample(frac=1, random_state=1).reset_index(drop=True),
        _recorrido(),
    )

    # Va de oeste a este, o sea en el sentido del recorrido: sin ordenar, los
    # pings apareados al azar daban "vuelta" en cualquier lado.
    assert set(stats.sentido.unique()) == {"ida"}
