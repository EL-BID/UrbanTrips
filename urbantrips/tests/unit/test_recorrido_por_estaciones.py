"""Recorrido de una línea ferroviaria: unir las estaciones, no suavizar.

En tren y subte la gente sube sólo en las estaciones, así que los orígenes de
las etapas no son una nube difusa sino un puñado de coordenadas exactas: la
línea 427 (`FFCC SARMIENTO`) tiene 8,1 millones de etapas en 16 coordenadas.
Suavizar eso con una regresión es lo que hacía fallar a `lowess_linea`.

El orden no viene en el dato — son pasajeros sueltos subiendo en distintas
estaciones — pero estas líneas son abiertas, así que el orden es el camino más
corto que pasa por todas una vez.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

geo = pytest.importorskip("urbantrips.geo.geo")
routes = pytest.importorskip("urbantrips.carto.routes")

EPSG_M = 9265


@pytest.fixture(autouse=True)
def _epsg_fijo(monkeypatch):
    monkeypatch.setattr(geo, "get_epsg_m", lambda: EPSG_M)


def _estaciones(coords, etapas_por_estacion=50, id_linea=427):
    """Etapas repetidas en cada estación, como llegan de la base."""
    filas = []
    for lon, lat in coords:
        filas += [{"id_linea": id_linea, "longitud": lon, "latitud": lat}] * (
            etapas_por_estacion
        )
    return pd.DataFrame(filas)


def _corredor_este_oeste(n=16):
    """El caso Sarmiento: estaciones alineadas de este a oeste."""
    return [(-58.38 - i * 0.025, -34.62) for i in range(n)]


def _largo_km(recorrido):
    return recorrido.to_crs(EPSG_M).geometry.length.item() / 1000


# ---------------------------------------------------------------------------
# el orden se recupera aunque el dato llegue mezclado
# ---------------------------------------------------------------------------


def test_arma_el_corredor_aunque_las_estaciones_lleguen_desordenadas():
    coords = _corredor_este_oeste()
    mezcladas = list(np.random.default_rng(0).permutation(coords))

    recorrido = geo.recorrido_por_estaciones(_estaciones(mezcladas))

    assert recorrido is not None
    linea = recorrido.geometry.item()
    assert len(linea.coords) == len(coords)

    # las estaciones quedan ordenadas a lo largo del corredor
    lons = [lon for lon, _ in linea.coords]
    assert lons == sorted(lons) or lons == sorted(lons, reverse=True)


def test_el_largo_es_el_del_corredor_y_no_un_zigzag():
    # 16 estaciones cada 0,025 grados de longitud sobre el paralelo -34,62
    recorrido = geo.recorrido_por_estaciones(_estaciones(_corredor_este_oeste()))

    puntos = gpd.GeoSeries(
        gpd.points_from_xy(*zip(*_corredor_este_oeste())), crs=4326
    ).to_crs(EPSG_M)
    extension = np.hypot(np.ptp(puntos.x.values), np.ptp(puntos.y.values)) / 1000

    # sin zigzag, el camino mide lo que la línea de punta a punta
    assert _largo_km(recorrido) == pytest.approx(extension, rel=0.01)


def test_funciona_igual_con_un_corredor_norte_sur():
    """`lowess_linea` ajusta la longitud en función de la latitud y se degenera
    en un corredor este-oeste; unir estaciones no depende de la orientación."""
    coords = [(-58.45, -34.55 - i * 0.02) for i in range(12)]

    recorrido = geo.recorrido_por_estaciones(_estaciones(coords))

    lats = [lat for _, lat in recorrido.geometry.item().coords]
    assert lats == sorted(lats) or lats == sorted(lats, reverse=True)


def test_devuelve_el_recorrido_en_4326():
    recorrido = geo.recorrido_por_estaciones(_estaciones(_corredor_este_oeste()))

    assert recorrido.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# cuándo NO aplica
# ---------------------------------------------------------------------------


def test_con_menos_de_tres_estaciones_no_hay_recorrido():
    # el caso de la línea 1161 (FFCC ROCA): 120 etapas en un solo punto
    assert geo.recorrido_por_estaciones(_estaciones([(-58.4, -34.6)])) is None


def test_ignora_los_puntos_con_muy_pocas_etapas():
    coords = _corredor_este_oeste(6)
    df = pd.concat([
        _estaciones(coords),
        # ruido: un punto lejano con una sola etapa
        _estaciones([(-59.5, -35.4)], etapas_por_estacion=1),
    ])

    recorrido = geo.recorrido_por_estaciones(df)

    assert len(recorrido.geometry.item().coords) == len(coords)


def test_una_nube_difusa_no_es_una_linea_de_estaciones():
    """El tranvía de Mendoza: 11.347 coordenadas distintas y sólo el 3,8 % de
    las etapas en las 40 más usadas. El modo la hacía candidata, el dato no."""
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "id_linea": [100] * n * 5,
        "longitud": np.repeat(-58.5 + rng.random(n) * 0.2, 5),
        "latitud": np.repeat(-34.6 + rng.random(n) * 0.2, 5),
    })

    assert geo.recorrido_por_estaciones(df) is None


# ---------------------------------------------------------------------------
# la elección del método
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", ["tren", "metro", "tranvia"])
def test_los_modos_de_estaciones_usan_el_metodo_de_estaciones(modo, monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        geo, "recorrido_por_estaciones", lambda df: llamadas.append("estaciones")
    )
    monkeypatch.setattr(geo, "lowess_linea", lambda df: llamadas.append("suavizado"))

    routes.inferir_recorrido(_estaciones(_corredor_este_oeste()), modo)

    assert llamadas[0] == "estaciones"


@pytest.mark.parametrize("modo", ["autobus", "lancha", None])
def test_el_resto_de_los_modos_se_suaviza(modo, monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        geo, "recorrido_por_estaciones", lambda df: llamadas.append("estaciones")
    )
    monkeypatch.setattr(geo, "lowess_linea", lambda df: llamadas.append("suavizado"))

    routes.inferir_recorrido(_estaciones(_corredor_este_oeste()), modo)

    assert llamadas == ["suavizado"]


def test_si_el_modo_es_candidato_pero_el_dato_no_cierra_se_suaviza(monkeypatch):
    llamadas = []
    monkeypatch.setattr(geo, "recorrido_por_estaciones", lambda df: None)
    monkeypatch.setattr(geo, "lowess_linea", lambda df: llamadas.append("suavizado"))

    routes.inferir_recorrido(_estaciones(_corredor_este_oeste()), "tranvia")

    assert llamadas == ["suavizado"]


# ---------------------------------------------------------------------------
# avisos cuando el trazado no parece un corredor único
# ---------------------------------------------------------------------------


def test_avisa_cuando_las_estaciones_forman_una_y(caplog):
    """El caso `FFCC BELGRANO SUR`: 2 ramales, 53 km sobre 34 de extensión.

    El ramal sale del MEDIO del tronco, así que el camino tiene que recorrer un
    brazo, volver sobre sus pasos y seguir por el otro. Un ramal que sale de una
    punta forma una L, que es un corredor legítimo y no dispara el aviso.
    """
    tronco = [(-58.38 - i * 0.02, -34.62) for i in range(6)]
    brazo = [(-58.43, -34.62 - i * 0.02) for i in range(1, 6)]

    with caplog.at_level("WARNING"):
        geo.recorrido_por_estaciones(_estaciones(tronco + brazo))

    assert any("ramales" in r.message for r in caplog.records)


def test_avisa_cuando_hay_un_salto_enorme_entre_estaciones(caplog):
    """El caso 431 (`FFCC MITRE`): 7 estaciones repartidas en 82 km."""
    coords = [(-58.38 - i * 0.01, -34.62) for i in range(4)]
    coords += [(-59.30 - i * 0.01, -34.62) for i in range(3)]

    with caplog.at_level("WARNING"):
        geo.recorrido_por_estaciones(_estaciones(coords))

    assert any("salto" in r.message for r in caplog.records)


def test_un_corredor_limpio_no_dispara_avisos(caplog):
    with caplog.at_level("WARNING"):
        geo.recorrido_por_estaciones(_estaciones(_corredor_este_oeste()))

    assert [r.message for r in caplog.records] == []
