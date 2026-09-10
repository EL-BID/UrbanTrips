"""Reglas para dividir el recorrido de una línea en secciones.

El recorrido se parte en tramos consecutivos para medir demanda, oferta y
matriz OD por sección. El tamaño de esos tramos se define de **una sola** de
estas dos formas, nunca las dos a la vez:

- ``n_sections``: el recorrido se divide en esa cantidad de tramos iguales.
- ``section_meters``: cada tramo mide esos metros y la cantidad sale del largo.

El que no se pasa se deriva del largo del recorrido. Si no se pasa ninguno se
usa ``N_SECTIONS_DEFAULT``.

Los rangos no son caprichosos: menos de 5 tramos no distingue nada dentro del
recorrido y más de 20 (o tramos de menos de 500 m) parte la demanda en pedazos
tan chicos que el resultado es ruido.

Además se valida el largo del propio recorrido. Las líneas sin recorrido
oficial caen al inferido, que puede medir miles de kilómetros y no representar
una línea real. Ese chequeo es el que evita que un recorrido roto llegue a
``pd.cut`` con miles de bordes repetidos (el error "Bin edges must be unique"),
y el dashboard lo usa para ofrecer recalcular el recorrido de esa línea.
"""

import logging

import pandas as pd

from urbantrips.geo import geo

logger = logging.getLogger(__name__)


# Rango razonable para el parámetro que elige el usuario.
N_SECTIONS_MIN = 5
N_SECTIONS_MAX = 20
SECTION_METERS_MIN = 500
SECTION_METERS_MAX = 10000

# Valores que ofrece el selector del dashboard: de a 100 m hasta los 5 km y de
# a 1 km de ahí para arriba. Arriba de los 5 km el paso chico no cambia la
# lectura (entre tramos de 5.100 y de 5.200 m no hay diferencia que mirar) y
# sólo llena el control de posiciones inútiles.
SECTION_METERS_STEP_BREAK = 5000
SECTION_METERS_OPTIONS = list(
    range(SECTION_METERS_MIN, SECTION_METERS_STEP_BREAK, 100)
) + list(range(SECTION_METERS_STEP_BREAK, SECTION_METERS_MAX + 1, 1000))

# Con qué se divide si no se pide nada.
N_SECTIONS_DEFAULT = 10

# Largo plausible de un recorrido. El más largo con recorrido oficial en AMBA
# mide 113 km; por encima de este techo la geometría es basura (los recorridos
# inferidos rotos miden ~10.600 km).
ROUTE_LENGTH_MIN_M = 500
ROUTE_LENGTH_MAX_M = 200_000


class SectionParamsError(Exception):
    """Los parámetros de secciones no sirven para el recorrido de esa línea."""


def formatear_numero(value, decimals=0):
    """Formatea un número al estilo local: 10.675,1 en vez de 10,675.1."""
    texto = f"{value:,.{decimals}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def formatear_largo(metros):
    """Largo de recorrido legible: metros si es corto, km si no."""
    if metros >= 10_000:
        return f"{formatear_numero(metros / 1000, 1)} km"
    return f"{formatear_numero(metros)} m"


def recorrido_plausible(route_geoms, epsg_m=None):
    """
    Por cada recorrido, si su largo es plausible para una línea real.

    Es el mismo chequeo que usa `describe_route_sections`, pero sin necesitar
    parámetros de sección: sirve para decidir si una geometría guardada se puede
    seguir usando o hay que volver a calcularla.

    Parameters
    ----------
    route_geoms : geopandas.GeoDataFrame
        recorridos con columna geometry, en cualquier CRS.
    epsg_m : int or None
        CRS en metros para medir. Si no se pasa se lee de la configuración.

    Returns
    -------
    pandas.Series of bool, con el índice de route_geoms.
    """
    if len(route_geoms) == 0:
        return pd.Series(dtype=bool)

    if epsg_m is None:
        epsg_m = geo.get_epsg_m()

    largos = route_geoms.to_crs(epsg=epsg_m).geometry.length
    return largos.between(ROUTE_LENGTH_MIN_M, ROUTE_LENGTH_MAX_M)


def resolve_section_params(n_sections=None, section_meters=None):
    """
    Aplica la regla de "uno u otro" sobre los parámetros pedidos.

    Devuelve la tupla ``(n_sections, section_meters)`` donde exactamente uno de
    los dos es un entero y el otro es None: el que quedó en None se deriva
    después, por línea, a partir del largo del recorrido.

    Sin ninguno de los dos se usa ``N_SECTIONS_DEFAULT``. Con los dos se corta:
    hasta ahora ``section_meters`` ganaba en silencio y el ``n_sections`` que el
    usuario había escrito se descartaba sin aviso.
    """
    if n_sections is not None and section_meters is not None:
        raise SectionParamsError(
            "El recorrido se divide por cantidad de secciones o por metros por "
            "sección, no por las dos cosas. Se recibieron n_sections="
            f"{n_sections} y section_meters={section_meters}: elegí una sola."
        )

    if n_sections is None and section_meters is None:
        return N_SECTIONS_DEFAULT, None

    if n_sections is not None:
        n_sections = int(n_sections)
        if not (N_SECTIONS_MIN <= n_sections <= N_SECTIONS_MAX):
            raise SectionParamsError(
                f"Se pidieron {n_sections} secciones, pero el recorrido se "
                f"divide en entre {N_SECTIONS_MIN} y {N_SECTIONS_MAX} tramos. "
                "Con menos no se distingue nada dentro del recorrido y con más "
                "la demanda queda partida en pedazos demasiado chicos."
            )
        return n_sections, None

    section_meters = int(section_meters)
    if not (SECTION_METERS_MIN <= section_meters <= SECTION_METERS_MAX):
        raise SectionParamsError(
            f"Se pidieron secciones de {formatear_numero(section_meters)} m, pero cada "
            f"tramo debe medir entre {formatear_numero(SECTION_METERS_MIN)} y "
            f"{formatear_numero(SECTION_METERS_MAX)} m. Con tramos más chicos la demanda "
            "queda partida en pedazos demasiado chicos y con tramos más "
            "grandes no se distingue nada dentro del recorrido."
        )
    return None, section_meters


def describe_route_sections(
    route_geoms, n_sections=None, section_meters=None, epsg_m=None
):
    """
    Resuelve, para cada recorrido, en cuántas secciones queda dividido y cuánto
    mide cada una, sin levantar excepciones.

    Es la versión que usa el dashboard para mostrarle al usuario qué va a pasar
    antes de procesar. La versión que corta es `resolve_route_sections`.

    Parameters
    ----------
    route_geoms : geopandas.GeoDataFrame
        recorridos con columnas id_linea y geometry, en cualquier CRS.
    n_sections : int or None
    section_meters : int or None
    epsg_m : int or None
        CRS en metros para medir. Si no se pasa se lee de la configuración.

    Returns
    -------
    pandas.DataFrame
        una fila por recorrido con id_linea, largo_m, n_sections,
        section_meters, valido (bool) y motivo (str, vacío si es válido).
        `motivo` explica por qué no se puede procesar, o queda como advertencia
        cuando el valor derivado se sale del rango razonable pero la línea sirve
        igual (recorridos legítimos muy largos, por ejemplo).
    """
    n_sections, section_meters = resolve_section_params(n_sections, section_meters)

    if epsg_m is None:
        epsg_m = geo.get_epsg_m()

    projected = route_geoms.to_crs(epsg=epsg_m)
    largos = projected.geometry.length

    filas = []
    for (_, row), largo in zip(route_geoms.iterrows(), largos):
        filas.append(_describe_una(row.id_linea, largo, n_sections, section_meters))

    return pd.DataFrame(
        filas,
        columns=[
            "id_linea",
            "largo_m",
            "n_sections",
            "section_meters",
            "valido",
            "motivo",
        ],
    )


def _describe_una(id_linea, largo_m, n_sections, section_meters):
    """Resuelve y valida una sola línea. Ver `describe_route_sections`."""
    base = {
        "id_linea": id_linea,
        "largo_m": largo_m,
        "n_sections": None,
        "section_meters": None,
        "valido": False,
        "motivo": "",
    }

    if largo_m > ROUTE_LENGTH_MAX_M:
        base["motivo"] = (
            f"El recorrido mide {formatear_largo(largo_m)}, un valor que no "
            "corresponde a una línea real, así que no se puede analizar por "
            "secciones."
        )
        return base

    if largo_m < ROUTE_LENGTH_MIN_M:
        base["motivo"] = (
            f"El recorrido mide {formatear_largo(largo_m)}, demasiado corto para "
            "dividirlo en secciones."
        )
        return base

    if n_sections is not None:
        derivado = int(largo_m / n_sections)
        base["n_sections"] = n_sections
        base["section_meters"] = derivado
        base["valido"] = True
        if derivado > SECTION_METERS_MAX:
            base["motivo"] = (
                f"Recorrido de {formatear_largo(largo_m)}: con {n_sections} "
                f"secciones cada tramo mide {formatear_numero(derivado)} m, por encima de "
                f"los {formatear_numero(SECTION_METERS_MAX)} m recomendados."
            )
        elif derivado < SECTION_METERS_MIN:
            base["motivo"] = (
                f"Recorrido de {formatear_largo(largo_m)}: con {n_sections} "
                f"secciones cada tramo mide {formatear_numero(derivado)} m, por debajo de "
                f"los {formatear_numero(SECTION_METERS_MIN)} m recomendados."
            )
        return base

    derivado = int(largo_m / section_meters)
    if derivado < 1:
        base["motivo"] = (
            f"El recorrido mide {formatear_largo(largo_m)}, menos que los "
            f"{formatear_numero(section_meters)} m pedidos por sección."
        )
        return base

    base["n_sections"] = derivado
    base["section_meters"] = section_meters
    base["valido"] = True
    if derivado > N_SECTIONS_MAX:
        base["motivo"] = (
            f"Recorrido de {formatear_largo(largo_m)}: con secciones de "
            f"{formatear_numero(section_meters)} m salen {formatear_numero(derivado)} tramos, por "
            f"encima de los {N_SECTIONS_MAX} recomendados."
        )
    elif derivado < N_SECTIONS_MIN:
        base["motivo"] = (
            f"Recorrido de {formatear_largo(largo_m)}: con secciones de "
            f"{formatear_numero(section_meters)} m salen {formatear_numero(derivado)} tramos, por "
            f"debajo de los {N_SECTIONS_MIN} recomendados."
        )
    return base


def resolve_route_sections(
    route_geoms, n_sections=None, section_meters=None, epsg_m=None
):
    """
    Devuelve los recorridos con las columnas n_sections y section_meters ya
    resueltas, cortando si alguno no se puede dividir.

    Parameters
    ----------
    route_geoms : geopandas.GeoDataFrame
        recorridos con columnas id_linea y geometry.
    n_sections : int or None
    section_meters : int or None
        exactamente uno de los dos; ver `resolve_section_params`.
    epsg_m : int or None

    Returns
    -------
    geopandas.GeoDataFrame
        los mismos recorridos en epsg 4326, con n_sections y section_meters.

    Raises
    ------
    SectionParamsError
        si los parámetros no cumplen la regla de "uno u otro" o si algún
        recorrido no se puede dividir en secciones.
    """
    if len(route_geoms) == 0:
        return route_geoms

    detalle = describe_route_sections(
        route_geoms, n_sections, section_meters, epsg_m=epsg_m
    )

    invalidas = detalle.loc[~detalle.valido, :]
    if len(invalidas) > 0:
        mensajes = [
            f"Línea {row.id_linea}: {row.motivo}" for _, row in invalidas.iterrows()
        ]
        raise SectionParamsError("\n".join(mensajes))

    for _, row in detalle.loc[detalle.motivo != "", :].iterrows():
        logger.warning("Línea %s: %s", row.id_linea, row.motivo)

    route_geoms = route_geoms.to_crs(epsg=4326)
    route_geoms = route_geoms.copy()
    route_geoms["n_sections"] = detalle.n_sections.astype(int).values
    route_geoms["section_meters"] = detalle.section_meters.astype(int).values

    return route_geoms
