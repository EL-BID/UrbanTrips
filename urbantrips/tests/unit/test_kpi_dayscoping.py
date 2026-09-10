"""Acotar por día lo que leen los KPI por sección, y proyectar una vez por celda.

Las cuatro herramientas interactivas del dashboard barrían todos los días de la
base: con un mes cargado eso es leer millones de etapas por línea. Ahora el
usuario elige los días y el filtro baja al SQL.

En el mismo camino, `add_od_lrs_to_legs_from_route` proyectaba origen y destino
fila por fila. La proyección depende solo de la celda h3 y del recorrido, y las
celdas se repiten muchísimo: en la línea más cargada del AMBA, 1.982.279 etapas
de un mes usan 828 orígenes y 852 destinos distintos.
"""

import itertools

import pandas as pd
import pytest
from shapely.geometry import LineString

from urbantrips.carto.routes import get_route_section_id
from urbantrips.geo import geo
from urbantrips.kpi.kpi import add_od_lrs_to_legs_from_route
from urbantrips.utils.utils import create_days_sql_filter


# ---------------------------------------------------------------------------
# filtro de días
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dias", [None, [], ()])
def test_sin_dias_no_filtra(dias):
    assert create_days_sql_filter(dias) == ""


def test_ordena_y_deduplica():
    clausula = create_days_sql_filter(["2026-03-10", "2026-03-09", "2026-03-09"])

    assert clausula == " AND dia IN ('2026-03-09', '2026-03-10')"


def test_respeta_columna_y_prefijo():
    clausula = create_days_sql_filter(
        ["2026-03-09"], col="e.dia", prefix=" WHERE "
    )

    assert clausula == " WHERE e.dia IN ('2026-03-09')"


def test_rechaza_lo_que_no_sea_una_fecha():
    # El filtro se interpola en el SQL: cualquier cosa que no sea una fecha se
    # rechaza antes de llegar a la base.
    with pytest.raises(Exception, match="YYYY-MM-DD"):
        create_days_sql_filter(["2026-03-09", "'; DROP TABLE etapas; --"])


# ---------------------------------------------------------------------------
# proyección sobre el recorrido: una vez por celda, mismo resultado
# ---------------------------------------------------------------------------


def _proyeccion_fila_por_fila(legs_df, route_geom):
    """La implementación anterior, como referencia de equivalencia."""
    legs_df = legs_df.copy()
    legs_df["o"] = legs_df["h3_o"].map(geo.create_point_from_h3)
    legs_df["d"] = legs_df["h3_d"].map(geo.create_point_from_h3)
    legs_df["o_proj"] = list(
        map(get_route_section_id, legs_df["o"], itertools.repeat(route_geom))
    )
    legs_df["d_proj"] = list(
        map(get_route_section_id, legs_df["d"], itertools.repeat(route_geom))
    )
    return legs_df


@pytest.fixture
def etapas_y_recorrido():
    import h3

    # Un recorrido sobre el AMBA y celdas h3 tomadas a lo largo de él.
    coords = [(-58.50 + i * 0.01, -34.60 + i * 0.004) for i in range(12)]
    route_geom = LineString(coords)

    celdas = [h3.latlng_to_cell(lat, lon, 8) for lon, lat in coords]
    # Cada celda aparece muchas veces, que es el caso real.
    legs = pd.DataFrame(
        {
            "h3_o": [celdas[i % len(celdas)] for i in range(300)],
            "h3_d": [celdas[(i * 7) % len(celdas)] for i in range(300)],
        }
    )
    return legs, route_geom


def test_proyeccion_identica_a_la_version_fila_por_fila(etapas_y_recorrido):
    legs, route_geom = etapas_y_recorrido

    esperado = _proyeccion_fila_por_fila(legs, route_geom)
    obtenido = add_od_lrs_to_legs_from_route(legs.copy(), route_geom)

    pd.testing.assert_series_equal(esperado.o_proj, obtenido.o_proj)
    pd.testing.assert_series_equal(esperado.d_proj, obtenido.d_proj)


def test_proyecta_una_sola_vez_por_celda(monkeypatch, etapas_y_recorrido):
    legs, route_geom = etapas_y_recorrido
    celdas_distintas = len(set(legs.h3_o) | set(legs.h3_d))

    llamadas = []
    original = geo.create_point_from_h3

    def espia(celda):
        llamadas.append(celda)
        return original(celda)

    monkeypatch.setattr(
        "urbantrips.kpi.kpi.geo.create_point_from_h3", espia
    )
    add_od_lrs_to_legs_from_route(legs, route_geom)

    assert len(llamadas) == celdas_distintas
    assert celdas_distintas < len(legs)
