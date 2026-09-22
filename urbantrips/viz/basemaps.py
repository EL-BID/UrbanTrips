# urbantrips/viz/basemaps.py
"""Basemap único del proyecto, para folium y para contextily.

Por qué existe este módulo
--------------------------
CARTO empezó a exigir API key para sus basemaps. Los tiles de Positron siguen
respondiendo 200 con un PNG válido, pero vienen con "API KEY REQUIRED" estampado
en diagonal sobre el mapa. Como la imagen es válida, nada se entera del problema:
`xyzservices` sigue reportando `requires_token() == False`, contextily descarga
sin error y los `try/except (UnidentifiedImageError, ValueError)` de los
llamadores nunca se disparan. El mapa simplemente sale marcado.

Se reemplaza por Esri World Light Gray, que no pide credenciales y conserva el
criterio por el que se había elegido Positron: gris neutro, para que los datos
que se dibujan encima resalten. Un basemap cargado (OpenStreetMap, que es a lo
que caían los fallbacks) compite visualmente con los datos.

Dos capas, no una
-----------------
Esri sirve el fondo y los nombres de localidades por separado: `World_Light_Gray_Base`
y `World_Light_Gray_Reference`. Los helpers de acá agregan las dos, porque el
canvas solo no trae ninguna etiqueta y deja los mapas sin referencias.

Zoom
----
Esri llega hasta z16 y Positron llegaba a z20. Sin tratamiento, al acercarse más
allá de 16 el fondo desaparecía. En folium se declara `max_zoom=20` con
`max_native_zoom=16`, así Leaflet escala los tiles de z16 en vez de dejar el mapa
en blanco. En contextily el zoom lo resuelve la librería dentro del rango del
proveedor.

Todo el proyecto pasa por acá: si CARTO vuelve atrás, si se consigue una API key
o si hay que cambiar de proveedor otra vez, se toca este archivo y nada más.
"""

from __future__ import annotations

import logging

import xyzservices

logger = logging.getLogger(__name__)

_ESRI_BASE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
    "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_LABELS = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
    "World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
)

ATTR = "Esri — World Light Gray"

#: Fondo gris (sin etiquetas). Es el `source` para contextily.
CANVAS = xyzservices.TileProvider(
    name="Esri.WorldGrayCanvas",
    url=_ESRI_BASE,
    attribution=ATTR,
    max_zoom=16,
)

#: Nombres de localidades. Va SIEMPRE encima de CANVAS, nunca sola.
LABELS = xyzservices.TileProvider(
    name="Esri.WorldGrayReference",
    url=_ESRI_LABELS,
    attribution="",
    max_zoom=16,
)

# Para folium / geopandas.explore, que reciben la URL como string.
FOLIUM_TILES = _ESRI_BASE
FOLIUM_LABELS = _ESRI_LABELS
FOLIUM_ATTR = ATTR

_MAX_NATIVE_ZOOM = 16
_MAX_ZOOM = 20


# ── folium ────────────────────────────────────────────────────────────────────

def add_labels(m) -> None:
    """Agrega la capa de nombres sobre un mapa folium ya creado.

    `control=False` la deja fuera del LayerControl: es parte del fondo, no una
    capa de datos que el usuario tenga que poder apagar.
    """
    import folium

    folium.TileLayer(
        tiles=FOLIUM_LABELS,
        attr=FOLIUM_ATTR,
        name="Etiquetas",
        overlay=True,
        control=False,
        max_zoom=_MAX_ZOOM,
        max_native_zoom=_MAX_NATIVE_ZOOM,
    ).add_to(m)


def folium_map(*args, **kwargs):
    """`folium.Map` con el basemap del proyecto (fondo + etiquetas).

    Reemplaza a `folium.Map(...)` en los llamadores. Acepta y descarta un `tiles=`
    para que migrar no obligue a tocar cada lista de argumentos.
    """
    import folium

    kwargs.pop("tiles", None)
    kwargs.pop("attr", None)

    m = folium.Map(*args, tiles=None, **kwargs)
    folium.TileLayer(
        tiles=FOLIUM_TILES,
        attr=FOLIUM_ATTR,
        name="Mapa base",
        control=False,
        max_zoom=_MAX_ZOOM,
        max_native_zoom=_MAX_NATIVE_ZOOM,
    ).add_to(m)
    add_labels(m)
    return m


# ── contextily (mapas estáticos de matplotlib) ────────────────────────────────

def add_basemap(ax, crs=None, source=None, labels=True, **kwargs) -> None:
    """`contextily.add_basemap` con el basemap del proyecto.

    Misma firma que la de contextily, así que los llamadores solo cambian el
    prefijo. `source` se acepta para no romper las llamadas que ya lo pasaban:
    si viene el proveedor viejo de CARTO —o nada— se usa CANVAS igual.

    Las etiquetas se agregan en una segunda pasada y su error se traga: si esa
    capa falla (zoom fuera de rango, corte de red), el mapa sale sin nombres,
    que es mejor que sin mapa. El fondo sí propaga el error, porque los
    llamadores tienen sus propios try/except para quedarse sin basemap.
    """
    import contextily as cx

    if source is None or getattr(source, "name", "").startswith("CartoDB"):
        source = CANVAS

    if crs is not None:
        kwargs["crs"] = crs

    cx.add_basemap(ax, source=source, **kwargs)

    if not labels:
        return

    # La atribución ya la puso el fondo: la capa de nombres no la repite.
    label_kwargs = dict(kwargs)
    label_kwargs["attribution"] = ""
    try:
        cx.add_basemap(ax, source=LABELS, **label_kwargs)
    except Exception as e:  # noqa: BLE001 - ver docstring
        logger.debug("[basemaps] no se pudieron agregar las etiquetas: %s", e)
