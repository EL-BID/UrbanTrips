"""Reglas para dividir un recorrido en secciones.

El caso que motivó estos tests: el dashboard reventaba con "Bin edges must be
unique" al pedir la matriz OD de una línea sin recorrido oficial. El recorrido
inferido medía 10.675 km, dividido por los metros por sección daba 5.306 tramos,
y como los identificadores de sección se redondean a 3 decimales los cortes se
repetían y `pd.cut` fallaba.
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from urbantrips.carto.carto import create_route_section_ids
from urbantrips.carto.route_sections import (
    N_SECTIONS_DEFAULT,
    SectionParamsError,
    describe_route_sections,
    resolve_route_sections,
    resolve_section_params,
)

# Faja 5 de Gauss-Kruger (POSGAR 2007), la que usa AMBA.
EPSG_M = 9265


def _recorrido(id_linea, largo_m):
    """Un recorrido recto del largo pedido, en un CRS en metros."""
    return gpd.GeoDataFrame(
        {"id_linea": [id_linea]},
        geometry=[LineString([(5_500_000, 6_100_000), (5_500_000 + largo_m, 6_100_000)])],
        crs=f"EPSG:{EPSG_M}",
    )


# ---------------------------------------------------------------------------
# el guard técnico: nunca más bordes repetidos
# ---------------------------------------------------------------------------


def test_create_route_section_ids_corta_antes_de_generar_cortes_repetidos():
    # 5306 es la cantidad exacta que reventaba en el dashboard del cliente.
    with pytest.raises(ValueError, match="5306"):
        create_route_section_ids(5306)


def test_create_route_section_ids_acepta_el_maximo_y_no_repite_cortes():
    section_ids = create_route_section_ids(1000)

    assert section_ids.is_unique
    assert len(section_ids) == 1001


def test_create_route_section_ids_rechaza_cero_secciones():
    with pytest.raises(ValueError):
        create_route_section_ids(0)


# ---------------------------------------------------------------------------
# uno u otro, nunca los dos
# ---------------------------------------------------------------------------


def test_sin_parametros_usa_el_default_de_secciones():
    assert resolve_section_params() == (N_SECTIONS_DEFAULT, None)


def test_con_los_dos_parametros_corta():
    # Antes section_meters ganaba en silencio y el n_sections que el usuario
    # había escrito se descartaba sin aviso.
    with pytest.raises(SectionParamsError, match="no por las dos cosas"):
        resolve_section_params(n_sections=10, section_meters=1000)


@pytest.mark.parametrize("n_sections", [4, 21, 100])
def test_cantidad_de_secciones_fuera_de_rango(n_sections):
    with pytest.raises(SectionParamsError, match="entre 5 y 20"):
        resolve_section_params(n_sections=n_sections)


@pytest.mark.parametrize("section_meters", [100, 499, 10001])
def test_metros_por_seccion_fuera_de_rango(section_meters):
    with pytest.raises(SectionParamsError, match="entre 500 y 10.000"):
        resolve_section_params(section_meters=section_meters)


# ---------------------------------------------------------------------------
# el parámetro que falta se deriva del largo
# ---------------------------------------------------------------------------


def test_por_cantidad_de_secciones_deriva_los_metros():
    resuelto = resolve_route_sections(_recorrido(1, 20_000), n_sections=10, epsg_m=EPSG_M)

    assert resuelto.n_sections.item() == 10
    assert resuelto.section_meters.item() == 2000


def test_por_metros_deriva_la_cantidad_de_secciones():
    resuelto = resolve_route_sections(
        _recorrido(1, 20_000), section_meters=2000, epsg_m=EPSG_M
    )

    assert resuelto.n_sections.item() == 10
    assert resuelto.section_meters.item() == 2000


def test_devuelve_los_recorridos_en_4326():
    resuelto = resolve_route_sections(_recorrido(1, 20_000), epsg_m=EPSG_M)

    assert resuelto.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# el recorrido roto se detecta solo
# ---------------------------------------------------------------------------


def test_recorrido_inferido_roto_no_se_puede_dividir():
    # El caso del cliente: linea sin recorrido oficial, recorrido inferido de
    # 10.675 km. Antes esto llegaba a pd.cut con 5306 cortes repetidos.
    roto = _recorrido(350, 10_675_000)

    with pytest.raises(SectionParamsError, match="no corresponde a una línea real"):
        resolve_route_sections(roto, section_meters=2000, epsg_m=EPSG_M)


def test_recorrido_roto_tampoco_pasa_por_cantidad_de_secciones():
    roto = _recorrido(350, 10_675_000)

    with pytest.raises(SectionParamsError, match="no corresponde a una línea real"):
        resolve_route_sections(roto, n_sections=10, epsg_m=EPSG_M)


def test_el_mensaje_nombra_la_linea_y_su_largo():
    roto = _recorrido(350, 10_675_000)

    with pytest.raises(SectionParamsError) as exc:
        resolve_route_sections(roto, epsg_m=EPSG_M)

    assert "Línea 350" in str(exc.value)
    assert "10.675,0 km" in str(exc.value)


def test_recorrido_degenerado_no_se_puede_dividir():
    with pytest.raises(SectionParamsError, match="demasiado corto"):
        resolve_route_sections(_recorrido(1, 100), epsg_m=EPSG_M)


# ---------------------------------------------------------------------------
# recorridos legítimos largos: advertencia, no bloqueo
# ---------------------------------------------------------------------------


def test_recorrido_largo_pero_real_procesa_con_advertencia():
    # 113 km es el recorrido oficial más largo de AMBA: con 10 secciones cada
    # tramo mide 11 km, más de lo recomendado, pero la línea existe y los
    # resultados sirven.
    largo = _recorrido(7, 113_000)

    detalle = describe_route_sections(largo, n_sections=10, epsg_m=EPSG_M)

    assert detalle.valido.item()
    assert "por encima de los 10.000 m recomendados" in detalle.motivo.item()

    resuelto = resolve_route_sections(largo, n_sections=10, epsg_m=EPSG_M)
    assert resuelto.n_sections.item() == 10


def test_advertencia_cuando_salen_demasiadas_secciones():
    detalle = describe_route_sections(
        _recorrido(7, 69_000), section_meters=1000, epsg_m=EPSG_M
    )

    assert detalle.valido.item()
    assert detalle.n_sections.item() == 69
    assert "por encima de los 20 recomendados" in detalle.motivo.item()


# ---------------------------------------------------------------------------
# describe: lo que el dashboard muestra antes de procesar
# ---------------------------------------------------------------------------


def test_describe_no_levanta_excepcion_y_marca_cada_linea():
    recorridos = gpd.GeoDataFrame(
        {"id_linea": [1, 350]},
        geometry=[
            LineString([(5_500_000, 6_100_000), (5_520_000, 6_100_000)]),
            LineString([(5_500_000, 6_100_000), (16_175_000, 6_100_000)]),
        ],
        crs=f"EPSG:{EPSG_M}",
    )

    detalle = describe_route_sections(recorridos, epsg_m=EPSG_M)

    assert detalle.set_index("id_linea").valido.to_dict() == {1: True, 350: False}
    assert detalle.loc[detalle.id_linea == 1, "motivo"].item() == ""
