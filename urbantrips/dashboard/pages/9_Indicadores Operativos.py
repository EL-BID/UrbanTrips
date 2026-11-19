# -*- coding: utf-8 -*-
"""
Dashboard de KPIs – Indicadores Operativos
=========================================
Panel Streamlit que muestra:

1. **Indicadores**: un panel único con tres selectores —Día, Modo y Línea— que
   cubren desde el total del sistema hasta una línea en un día concreto.
   Con "Día = Todos" muestra el promedio diario y, si hay más de un día, el
   acumulado del período. Los ratios se agregan ponderados por su denominador
   (ver `RATIO_AGG`), no promediados a mano.
2. **Base completa** con descarga CSV.

Antes esto eran dos secciones ("KPIs por línea" y "Totales del sistema") que
daban números distintos para el mismo indicador: la primera leía la fila
`dia = "Promedios"` de la tabla (media simple, sin ponderar los ratios) y la
segunda los recalculaba. Ahora hay un solo criterio y esa fila no se usa.

Los indicadores operativos se muestran para las **tres familias de distancia**
del refactor (`_od`, `_route`, `_route_gps`), siempre las tres y en ese orden;
la que no tenga datos en la corrida aparece igual, vacía y con el motivo.

Mejora de presentación (jul‑2025)
---------------------------------
* Alineación consistente de métricas: siempre se generan **6 columnas fijas**;
  si un grupo tiene menos indicadores, las celdas restantes quedan vacías, de
  modo que la primera, segunda, tercera métrica, etc. aparecen siempre en la
  misma posición.
* Tamaño de fuente reducido en `st.metric`.
"""

import re

import streamlit as st
import pandas as pd
from dash_utils import levanto_tabla_sql, get_logo, configurar_selector_corrida

# La lógica de agregación vive en el paquete, no acá: esta página no es
# importable (ejecuta `st.*` a nivel de módulo) y por lo tanto no se podía
# cubrir con tests. Ver urbantrips/kpi/agregacion_indicadores.py.
from urbantrips.kpi.agregacion_indicadores import (
    COBERTURA_POR_ETAPAS,
    GENDER_COLS,
    NOMBRE_DIAS,
    TARIFA_COLS,
    TIPOS_DIA,
    agregar_sistema,
    etiqueta_dia,
    nota_capacidad_modos,
    nota_cobertura,
    preparar,
)
from urbantrips.utils.utils import leer_configs_generales

# -----------------------------------------------------------------------------
# Configuración global y estilo
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Indicadores Operativos por Línea", layout="wide")

st.markdown(
    """
    <style>
    /* Streamlit renombró el testid de st.metric: hasta ~1.25 era
       "metric-container", desde 1.26 son "stMetric"/"stMetricLabel"/
       "stMetricValue". El CSS original apuntaba solo al viejo, así que en
       1.40 no se aplicaba nada y las métricas salían en tamaño default
       (~2.25rem el valor), que no entra en 6 columnas sin hacer zoom out.
       Se mantienen ambas familias de selectores por compatibilidad. */

    [data-testid="stMetric"],
    div[data-testid="metric-container"] {
        padding: 0.3rem 0.5rem;
        background: rgba(128, 128, 128, 0.07);
        border-radius: 6px;
        overflow: visible;
    }

    /* Etiqueta: chica y en varios renglones. `min-height` de dos líneas para
       que los valores queden alineados entre columnas aunque unas etiquetas
       ocupen una línea y otras dos. */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    div[data-testid="metric-container"] > div:first-child {
        font-size: 0.72rem !important;
        line-height: 1.2 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
        opacity: 0.85;
    }
    [data-testid="stMetricLabel"] {
        min-height: 2.4em;
        display: flex;
        align-items: flex-start;
    }

    /* Valor: sin cortar en dos líneas, que es lo que descoloca la grilla. */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    div[data-testid="metric-container"] > div:nth-child(2) {
        font-size: 1.15rem !important;
        line-height: 1.3 !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }

    /* Columnas y separadores más apretados: con 3 filas de métricas por
       expander, los márgenes default obligan a scrollear. */
    [data-testid="stHorizontalBlock"] { gap: 0.5rem; }
    hr { margin: 0.5rem 0 !important; }

    /* Textos explicativos: son para leer, no rótulos, así que van más grandes
       que la etiqueta de las métricas (0.72rem) y con interlínea holgada. */
    [data-testid="stCaptionContainer"] p {
        font-size: 0.85rem !important;
        line-height: 1.45 !important;
        opacity: 0.9;
    }

    /* h5 se usa solo para la empresa de la línea seleccionada: va pegado al
       título de arriba, no flotando en el medio. */
    h5 {
        margin-top: -0.4rem !important;
        padding-top: 0 !important;
        margin-bottom: 0.7rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    st.image(get_logo())
except Exception:
    pass

try:
    alias_sel = configurar_selector_corrida()
except Exception:
    alias_sel = "default"

# -----------------------------------------------------------------------------
# Carga de datos
# -----------------------------------------------------------------------------


COLS_FILTRO_METADATA = ["grupo", "empresa"]


def _limpiar_texto(serie: pd.Series) -> pd.Series:
    """Texto sin espacios de borde; vacío, "nan" o "none" quedan nulos."""
    limpio = serie.astype("string").str.strip()
    return limpio.mask(limpio.fillna("").str.lower().isin(["", "nan", "none"]))


@st.cache_data(show_spinner=False)
def load_grupos() -> pd.DataFrame:
    """`grupo` de cada línea, desde `metadata_lineas`. Se lee en vivo (no
    viene en `kpis_lineas`) para que cambiar el agrupamiento no obligue a
    recalcular los indicadores. Vacío si la base no tiene la columna."""
    try:
        meta = levanto_tabla_sql("metadata_lineas", "insumos", tabla_opcional=True)
    except Exception:
        return pd.DataFrame()
    if meta is None or meta.empty or "grupo" not in meta.columns:
        return pd.DataFrame()
    return meta[["id_linea", "grupo"]].drop_duplicates("id_linea")


@st.cache_data(show_spinner=False)
def load_kpis() -> pd.DataFrame:
    """`kpis_lineas` con el `grupo` de cada línea y las columnas de los
    filtros de metadata ya limpias, para filtrar con una comparación directa."""
    df = levanto_tabla_sql("kpis_lineas", "general")
    grupos = load_grupos()
    if not df.empty and not grupos.empty:
        # id_linea puede llegar como texto, entero o float según la tabla
        # ("12", 12, 12.0): se cruza como número
        por_id = pd.Series(
            grupos["grupo"].to_numpy(),
            index=pd.to_numeric(grupos["id_linea"], errors="coerce"),
        )
        por_id = por_id[por_id.index.notna()]
        df["grupo"] = pd.to_numeric(df["id_linea"], errors="coerce").map(por_id)
    for col in COLS_FILTRO_METADATA:
        if col in df.columns:
            df[col] = _limpiar_texto(df[col])
    return df


@st.cache_data(show_spinner=False)
def load_kpis_ramales() -> pd.DataFrame:
    """`kpis_ramales` existe solo si la corrida tuvo lineas_contienen_ramales=True."""
    try:
        df = levanto_tabla_sql("kpis_ramales", "general", tabla_opcional=True)
    except Exception:
        return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    for col in COLS_FILTRO_METADATA:
        if col in df.columns:
            df[col] = _limpiar_texto(df[col])
    return df


def valores_filtro(df: pd.DataFrame, col: str) -> list:
    """Valores no vacíos de `col` en `df` (ya limpia, ver `load_kpis`), en
    orden natural: "Grupo 200" antes que "Grupo 1800"."""
    if df.empty or col not in df.columns:
        return []
    return sorted(
        df[col].dropna().unique(),
        key=lambda v: [int(t) if t.isdigit() else t.casefold()
                       for t in re.split(r"(\d+)", v)],
    )


def filtro_metadata(etiqueta: str, col: str, base: pd.DataFrame):
    """Selector de `col` (grupo, empresa) sobre las líneas de `base`. Devuelve
    (valor elegido, `base` filtrada). Se muestra siempre: si la metadata no
    trae la columna, queda deshabilitado en "Todos" y el "?" dice por qué."""
    disp = valores_filtro(base, col)
    if not disp:
        st.selectbox(
            etiqueta, ["Todos"], index=0, disabled=True, key=f"filtro_{col}_vacio",
            help=f"La metadata de líneas no trae `{col}` para esta selección "
                 "(columna opcional del csv de líneas).",
        )
        return "Todos", base
    sel = st.selectbox(etiqueta, ["Todos"] + disp, index=0, key=f"filtro_{col}")
    return sel, base if sel == "Todos" else base[base[col] == sel]


kpis_df = load_kpis()
kpis_ram_df = load_kpis_ramales()

kpis_df = preparar(kpis_df)
kpis_ram_df = preparar(kpis_ram_df)
kpis_ram_dias = (
    kpis_ram_df[kpis_ram_df["dia"] != "Promedios"].copy()
    if not kpis_ram_df.empty
    else pd.DataFrame()
)

# `kpis_lineas` trae una fila agregada por línea con dia = "Promedios" (la crea
# calculo_kpi_lineas). Es una fila de resumen, no un día: sumarla junto a los
# días duplica los totales — en una corrida de un solo día los duplica exacto.
# La sección 1 la usa como opción del selector; las secciones que agregan usan
# `kpis_dias`.
kpis_dias = kpis_df[kpis_df["dia"] != "Promedios"].copy()

# -----------------------------------------------------------------------------
# Etiquetas y grupos
# -----------------------------------------------------------------------------

LABELS = {
    "vehiculos_operativos": "Vehículos operativos",
    "cant_internos_en_trx": "Vehículos con transacciones",
    "cobertura_gps": "Cobertura GPS",
    "transacciones": "Transacciones",
    "Masculino": "Masculino",
    "Femenino": "Femenino",
    "No informado": "No informado",
    "sin_descuento": "Sin descuento",
    "tarifa_social": "Tarifa social",
    "educacion_jubilacion": "Estudiantes/Jubilados",
    "travel_time_min": "Tiempo promedio viaje (min)",
    "kmh_od": "Velocidad viaje Pax (km/h)",
    "distancia_media_pax": "Dist. media Pax etapa (km)",
    "pvd": "Pasajeros / vehículo",
    # familia _od (distancia origen-destino sobre H3)
    "dmt_mean_od": "Distancia media Pax (km)",
    "dmt_median_od": "Distancia mediana Pax (km)",
    # familia _route (recorrido reconstruido ping a ping)
    "velocidad_comercial_route": "Vel. comercial (km/h)",
    "distancia_media_veh_route": "Dist. media/veh (km)",
    "fo_mean_route": "Factor Ocupación (media)",
    "fo_median_route": "Factor Ocupación (mediana)",
    "tot_km_route": "Km recorridos",
    "tot_km_route_estimado": "Km estimados (por cobertura)",
    "kvd_route": "Km / vehículo",
    "dmt_mean_route": "Distancia media Pax (km)",
    "dmt_median_route": "Distancia mediana Pax (km)",
    "ipk_route": "IPK",
    # familia _route_gps (odómetro / distancia de servicio del GPS)
    "velocidad_comercial_route_gps": "Vel. comercial (km/h)",
    "distancia_media_veh_route_gps": "Dist. media/veh (km)",
    "fo_mean_route_gps": "Factor Ocupación (media)",
    "fo_median_route_gps": "Factor Ocupación (mediana)",
    "tot_km_route_gps": "Km recorridos",
    "dmt_mean_route_gps": "Distancia media Pax (km)",
    "dmt_median_route_gps": "Distancia mediana Pax (km)",
    "ipk_route_gps": "IPK",
}

# Ayuda breve de cada indicador: se muestra con el "?" junto al nombre. Las
# familias _route y _route_gps comparten texto (ver `ayuda_de`).
AYUDAS = {
    "vehiculos_operativos": "Vehículos (línea + interno) con al menos un servicio "
                            "GPS válido. Las validadoras de una formación cuentan "
                            "como un vehículo.",
    "cant_internos_en_trx": "Vehículos (línea + interno) con al menos una "
                            "validación. Las validadoras de una formación cuentan "
                            "una vez; en tren, metro y tranvía no cuentan los "
                            "equipos de estación (sin GPS).",
    "cobertura_gps": "Parte de las etapas que ocurre dentro de un servicio GPS "
                     "válido de su vehículo. Con cobertura baja (umbral en "
                     "tuning.yaml) los indicadores de oferta quedan vacíos.",
    "transacciones": "Validaciones de la línea (etapas expandidas al total de "
                     "transacciones).",
    "travel_time_min": "Tiempo medio de la etapa, desde la validación hasta el "
                       "destino estimado.",
    "kmh_od": "Distancia origen-destino ÷ tiempo de la etapa. Se descartan las "
              "etapas de 80 km/h o más.",
    "pvd": "Pasajeros que viajaron dentro de un servicio GPS válido ÷ vehículos "
           "operativos (con servicio válido). No usa todas las transacciones: "
           "las etapas fuera de un servicio válido quedan afuera del numerador, "
           "igual que su vehículo del denominador.",
    "dmt_mean_od": "Promedio de la distancia del pasajero en camino mínimo por la "
                   "red vial, entre origen y destino.",
    "dmt_median_od": "Mediana de la distancia del pasajero en camino mínimo por "
                     "la red vial, entre origen y destino.",
    "velocidad_comercial": "Km recorridos ÷ duración, en los servicios GPS "
                           "válidos.",
    "dmt_mean": "Promedio de la distancia que recorre el pasajero sobre el "
                "recorrido del vehículo, entre subida y bajada.",
    "dmt_median": "Mediana de la distancia que recorre el pasajero sobre el "
                  "recorrido del vehículo, entre subida y bajada.",
    "tot_km": "Km recorridos por los servicios GPS válidos.",
    "tot_km_route_estimado": "Km recorridos ÷ cobertura GPS: estimación de los km "
                             "totales si la operación sin GPS se parece a la "
                             "medida.",
    "distancia_media_veh": "Km recorridos por cada vehículo, promediados entre "
                           "vehículos.",
    "kvd": "Km recorridos ÷ vehículos operativos.",
    "ipk": "Pasajeros de los servicios GPS válidos ÷ km de esos servicios.",
    "fo_mean": "Parte de los lugares ofrecidos que va ocupada: pasajeros × "
               "distancia media ÷ (km × capacidad del vehículo del modo).",
    "fo_median": "Como el factor de ocupación medio, con la distancia mediana del "
                 "pasajero.",
}
for _c in GENDER_COLS:
    AYUDAS[_c] = "Transacciones según el género informado en la tarjeta, y su % del total."
for _c in TARIFA_COLS:
    AYUDAS[_c] = "Transacciones según el tipo de tarifa de la tarjeta, y su % del total."
_SUFIJOS_FAMILIA = ("_route_gps", "_route")


def ayuda_de(col: str) -> str | None:
    """Ayuda de `col`: la propia o la de su nombre sin sufijo de familia. En la
    familia del odómetro se aclara de dónde sale la distancia."""
    if col in AYUDAS:
        return AYUDAS[col]
    for sufijo in _SUFIJOS_FAMILIA:
        if col.endswith(sufijo) and col[: -len(sufijo)] in AYUDAS:
            texto = AYUDAS[col[: -len(sufijo)]]
            if sufijo == "_route_gps":
                texto += " Distancias del odómetro del GPS."
            return texto
    return None


GENERAL_COLS = [
    "vehiculos_operativos",
    "cant_internos_en_trx",
    "cobertura_gps",
    "transacciones",
]

# -----------------------------------------------------------------------------
# Las tres familias de distancia del refactor
# -----------------------------------------------------------------------------
# El refactor calcula tres distancias distintas y, con cada una, su propia
# familia de indicadores derivados:
#
#   _od         distancia origen-destino sobre H3. Es la distancia del *viaje
#               del pasajero*, no la recorrida por el vehículo: no tiene
#               velocidad comercial, ni IPK, ni km de oferta.
#   _route      recorrido del vehículo reconstruido ping a ping desde el GPS.
#   _route_gps  distancia de servicio que informa el propio GPS (odómetro).
#
# La tercera **depende del insumo**: en `ut_amba_20260514` (sistau)
# `distance_route_gps` viene en 0,0 para los 142.494 servicios válidos, mientras
# que en `marzo_mes_completo_2026_fix` trae 86.490.142 km. Por eso las familias
# se detectan de los datos (`familias_disponibles`) en vez de asumirse.
# Indicadores que NO dependen de qué distancia se use: van primero, fuera de
# los bloques por familia, para no sugerir que pertenecen a alguno.
#   travel_time_min  tiempo de viaje del pasajero, en minutos
#   pvd              pasajeros por vehículo = pasajeros dentro de servicios
#                    válidos / vehículos con servicio válido (kpi/cobertura.py)
SIN_DISTANCIA = ["travel_time_min", "pvd"]

FAMILIAS = {
    "od": {
        "label": "OD (origen–destino)",
        # Acá NO va el factor de ocupación. En kpi.py se calcula
        #     fo_*_od = (tot_pax · dmt_*_od) / (tot_km_route · capacidad)
        # con el numerador en distancia OD y el denominador en km de recorrido
        # real: un híbrido que subestima la ocupación (el pasajero viaja más que
        # el camino mínimo). Desde 2026-11 tampoco se guarda en kpis_lineas:
        # queda solo en kpi_by_day_line. El FO está en las dos familias de
        # recorrido, donde numerador y denominador miden lo mismo.
        #
        # `distancia_media_pax` también se omite: es la misma media ponderada de
        # `distance_od` que `dmt_mean_od` por otro camino (369 de 402 líneas
        # idénticas, el resto difiere ±0,1 km por redondeo).
        "nota": "Distancia media Pax: camino mínimo sobre la red vial entre el "
                "origen y el destino de la etapa (no el recorrido del vehículo).",
        "ayuda": "Distancia en camino mínimo sobre la red vial entre el origen y "
                 "el destino de la etapa, ubicados en hexágonos H3. Mide el "
                 "desplazamiento del "
                 "pasajero, no el recorrido del vehículo: por eso no tiene "
                 "velocidad comercial, ni IPK, ni km de oferta, ni factor de "
                 "ocupación.",
        "filas": [["kmh_od", "dmt_mean_od", "dmt_median_od"]],
        "probe": ["dmt_mean_od"],
    },
    "route": {
        "label": "Recorrido (ping a ping)",
        "ayuda": "Recorrido real del vehículo, reconstruido sumando los tramos "
                 "entre pings sucesivos del GPS. Es la base de los km de oferta, "
                 "de la velocidad comercial y del factor de ocupación.",
        # La primera fila repite el orden de la familia OD —velocidad, distancia
        # media, distancia mediana— para que las tres familias se lean en
        # paralelo, columna contra columna. En la segunda van los que solo
        # existen cuando hay recorrido real.
        "filas": [
            ["velocidad_comercial_route", "dmt_mean_route", "dmt_median_route"],
            ["tot_km_route", "tot_km_route_estimado", "distancia_media_veh_route",
             "ipk_route", "fo_mean_route", "fo_median_route"],
        ],
        "probe": ["velocidad_comercial_route", "tot_km_route"],
    },
    "route_gps": {
        "label": "Recorrido (odómetro GPS)",
        "ayuda": "Distancia de servicio informada por el propio equipo GPS, en "
                 "vez de reconstruida. Mide lo mismo que la anterior por otra "
                 "vía, así que la comparación entre ambas sirve de control. No "
                 "todos los insumos la traen.",
        "filas": [
            ["velocidad_comercial_route_gps", "dmt_mean_route_gps",
             "dmt_median_route_gps"],
            ["tot_km_route_gps", "distancia_media_veh_route_gps", "ipk_route_gps",
             "fo_mean_route_gps", "fo_median_route_gps"],
        ],
        "probe": ["velocidad_comercial_route_gps", "distancia_media_veh_route_gps"],
    },
}

# `pvd` es un cociente, pero con decimales sugiere una precisión que el dato no
# tiene: se muestra entero (el valor guardado no se redondea). Por debajo de 10
# (ramal o línea con muy poca demanda) lleva un decimal, para que 0,4 no se lea
# como 0 ni 1,4 como 1.
DECIMAL_BAJO_10_COLS = {"pvd"}
INT_DISPLAY_COLS = (
    set(GENERAL_COLS + GENDER_COLS + TARIFA_COLS
        + ["tot_km_route", "tot_km_route_gps", "tot_km_route_estimado", "pvd"])
    - {"cobertura_gps"}
)
PCT_DISPLAY_COLS = {"cobertura_gps"}
TOTAL_SLOTS = 6  # columnas fijas por fila


def familias_disponibles(d: pd.DataFrame) -> list:
    """Familias de distancia que tienen datos reales en esta corrida.

    `distance_route_gps` viene en 0 cuando el insumo no trae odómetro, así que
    no alcanza con que la columna exista: tiene que tener algún valor != 0.
    """
    out = []
    for key, spec in FAMILIAS.items():
        tiene = any(
            col in d.columns and d[col].fillna(0).abs().sum() > 0
            for col in spec["probe"]
        )
        if tiene:
            out.append(key)
    return out

# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------


def fmt(val, col):
    if pd.isna(val):
        return "–"
    if col in PCT_DISPLAY_COLS:
        return f"{val * 100:.1f} %"
    if col in DECIMAL_BAJO_10_COLS and abs(val) < 10:
        return f"{val:,.1f}"
    if col in INT_DISPLAY_COLS or (
        isinstance(val, (int, float)) and float(val).is_integer()
    ):
        return f"{int(round(val)):,}"
    return f"{val:,.2f}"


def bloques_demo_con_datos(df: pd.DataFrame) -> list[list[str]]:
    """Filas de género y/o tarifa que realmente tienen dato en la selección.

    Sin dato de género todo cae en "No informado" (Masculino y Femenino en 0);
    sin dato de tarifa todo cae en "sin_descuento". Mostrar eso como si fuera
    una distribución real engaña, así que ese bloque se omite.
    """

    def _suma(cols):
        vals = [
            pd.to_numeric(df[c].iloc[0], errors="coerce")
            for c in cols
            if c in df.columns
        ]
        return pd.Series(vals, dtype="float64").sum(min_count=1)

    bloques = []
    if _suma(["Masculino", "Femenino"]) > 0:
        bloques.append(GENDER_COLS)
    if _suma(["tarifa_social", "educacion_jubilacion"]) > 0:
        bloques.append(TARIFA_COLS)
    return bloques


def metric_row(df: pd.DataFrame, cols: list[str], *, pct=False, labels=None):
    """Muestra métricas en una fila de TOTAL_SLOTS columnas fijas. `labels`
    reemplaza la etiqueta de LABELS para las columnas que trae."""
    labels = {**LABELS, **(labels or {})}
    st_cols = st.columns(TOTAL_SLOTS)
    total = df["transacciones"].iloc[0] if "transacciones" in df.columns else None
    # Rellenar la lista a TOTAL_SLOTS con None
    padded_cols = cols + [None] * (TOTAL_SLOTS - len(cols))
    for idx, col in enumerate(padded_cols[:TOTAL_SLOTS]):
        if col is None or col not in df.columns:
            st_cols[idx].markdown(" ")  # espacio en blanco
            continue
        val = df[col].iloc[0]
        text = fmt(val, col)
        if pct and total and total > 0:
            text += f" ({val / total * 100:.1f} %)"
        st_cols[idx].metric(labels.get(col, col), text, help=ayuda_de(col))



# Las tres familias se muestran siempre, una debajo de otra. Las que no tienen
# datos en la corrida aparecen igual, con guiones, para que se vea que existen
# y que están vacías — en vez de desaparecer sin explicación.
_fams_con_datos = familias_disponibles(kpis_dias)


def bloque_familias(fila: pd.DataFrame) -> None:
    """Indicadores operativos: primero los que no dependen de la distancia,
    después las tres familias de distancia (siempre las tres, con su
    explicación). La que no tenga datos aparece igual, con guiones."""
    st.markdown("**Independientes de la distancia**")
    st.caption(
        "No cambian según qué distancia se use, así que van fuera de los "
        "bloques de abajo."
    )
    metric_row(fila, SIN_DISTANCIA)
    st.markdown("")

    for key, spec in FAMILIAS.items():
        cols_fam = [c for f in spec["filas"] for c in f]
        vacia = key not in _fams_con_datos

        datos = fila
        if vacia:
            # sin datos las columnas llegan en 0, no en NaN, y un "0" se lee
            # como un valor medido. Se fuerzan a NaN para que salga el guion.
            datos = fila.copy()
            for c in cols_fam:
                if c in datos.columns:
                    datos[c] = float("nan")

        st.markdown(f"**{spec['label']}**" + ("  ·  _sin datos_" if vacia else ""))
        st.caption(
            spec["ayuda"]
            + (
                "  \n:orange[Esta corrida no tiene esta distancia: el insumo GPS "
                "no trae odómetro, `distance_route_gps` llega en 0.]"
                if vacia
                else ""
            )
        )
        for f in spec["filas"]:
            metric_row(datos, f)
        if spec.get("nota") and not vacia:
            st.caption(spec["nota"])


# -----------------------------------------------------------------------------
# Panel único: Día × Modo × Línea
# -----------------------------------------------------------------------------
# Un solo criterio de agregación (`agregar_sistema`) para todos los casos. No se
# usa la fila `dia = "Promedios"` de `calculo_kpi_lineas` (media simple sin
# ponderar): "Día = Todos" calcula el promedio ponderado.
#
#   Día=Todos  + Línea=Todas  -> promedio diario del sistema (+ acumulado)
#   Día=fecha  + Línea=Todas  -> total de ese día
#   Día=Todos  + Línea=X      -> promedio diario de esa línea
#   Día=fecha  + Línea=X      -> esa línea ese día (una sola fila)

with st.expander("Indicadores", expanded=True):
    col_filt, col_out = st.columns([2, 10])

    with col_filt:
        # El tipo de día acota qué fechas ofrece el selector de abajo y, con
        # "Día = Todos", sobre qué días se promedia.
        _tipos = [t for t in TIPOS_DIA
                  if t == "Todos" or (kpis_dias["tipo_dia"] == t).any()]
        tipo_dia_sel = st.selectbox("Tipo de día", _tipos, index=0)

        _kd = (
            kpis_dias if tipo_dia_sel == "Todos"
            else kpis_dias[kpis_dias["tipo_dia"] == tipo_dia_sel]
        )
        dias_disp = sorted(_kd["dia"].dropna().unique())
        # se muestra "2026-03-14 (sábado)" pero el valor sigue siendo la fecha
        dia_sel = st.selectbox(
            "Día",
            ["Todos"] + dias_disp,
            index=0,
            format_func=lambda d: d if d == "Todos" else etiqueta_dia(d),
        )

        modo_options = ["Todos"] + sorted(kpis_dias["modo"].dropna().unique())
        modo_sel = st.selectbox("Modo", modo_options, index=0)

        # las líneas ofrecidas dependen del modo (y del grupo y la empresa),
        # para no listar 400 cuando ya se filtró
        _base_lineas = (
            kpis_dias if modo_sel == "Todos" else kpis_dias[kpis_dias["modo"] == modo_sel]
        )

        # Grupo y empresa salen de la metadata de líneas; cada uno acota las
        # opciones del siguiente.
        grupo_sel, _base_lineas = filtro_metadata("Grupo", "grupo", _base_lineas)
        empresa_sel, _base_lineas = filtro_metadata("Empresa", "empresa", _base_lineas)
        # "Nombre (id)" y ordenado por nombre: se busca por cómo se llama la
        # línea, no por su número interno
        lineas = _base_lineas[["id_linea", "nombre_linea"]].drop_duplicates()
        lineas["label"] = lineas.apply(
            lambda x: f"{x.nombre_linea} ({x.id_linea})", axis=1
        )
        lineas = lineas.sort_values(
            "nombre_linea", key=lambda s: s.astype(str).str.casefold()
        )
        linea_sel = st.selectbox("Línea", ["Todas"] + lineas["label"].tolist(), index=0)

        # El ramal solo se ofrece con una línea elegida (hay más de mil en el
        # sistema) y solo si la corrida generó `kpis_ramales`, que depende de
        # lineas_contienen_ramales = True en el yaml.
        ramal_sel = "Todos"
        _ramales = pd.DataFrame()
        if linea_sel != "Todas" and not kpis_ram_dias.empty:
            _id_l = str(lineas.set_index("label").loc[linea_sel, "id_linea"])
            _con_nombre = "nombre_ramal" in kpis_ram_dias.columns
            _ramales = (
                kpis_ram_dias[kpis_ram_dias["id_linea"].astype(str) == _id_l]
                [["id_ramal"] + (["nombre_ramal"] if _con_nombre else [])]
                .drop_duplicates("id_ramal")
            )
            if not _ramales.empty:
                # mismo criterio que el selector de línea: "Nombre (id)",
                # ordenado por nombre. Sin nombre queda solo el id.
                _ramales["label"] = _ramales.apply(
                    lambda x: (
                        f"{x.nombre_ramal} ({x.id_ramal})"
                        if _con_nombre and str(x.nombre_ramal).strip()
                        else f"{x.id_ramal}"
                    ),
                    axis=1,
                )
                _ramales = _ramales.sort_values(
                    "label", key=lambda s: s.astype(str).str.casefold()
                )
                ramal_sel = st.selectbox(
                    "Ramal", ["Todos"] + _ramales["label"].tolist(), index=0
                )
        elif linea_sel == "Todas" and not kpis_ram_dias.empty:
            st.caption("Elegí una línea para abrir por ramal.")

    with col_out:
        # con un ramal elegido la base pasa a ser `kpis_ramales`, que es la
        # misma cuenta un nivel más abajo (tabla aparte, ver kpi_lineas.py)
        por_ramal = ramal_sel != "Todos"
        df_sel = kpis_ram_dias if por_ramal else _base_lineas
        if tipo_dia_sel != "Todos" and "tipo_dia" in df_sel.columns:
            df_sel = df_sel[df_sel["tipo_dia"] == tipo_dia_sel]
        if dia_sel != "Todos":
            df_sel = df_sel[df_sel["dia"] == dia_sel]
        if linea_sel != "Todas":
            _id = lineas.set_index("label").loc[linea_sel, "id_linea"]
            df_sel = df_sel[df_sel["id_linea"].astype(str) == str(_id)]
        if por_ramal:
            _id_r = _ramales.set_index("label").loc[ramal_sel, "id_ramal"]
            df_sel = df_sel[df_sel["id_ramal"].astype(str) == str(_id_r)]

        if df_sel.empty:
            st.warning("Sin datos para los filtros seleccionados.")
        else:
            promedio_diario = dia_sel == "Todos"
            n_dias = df_sel["dia"].nunique()
            n_lineas = df_sel["id_linea"].nunique()
            agg = agregar_sistema(df_sel, promedio_diario=promedio_diario)

            # título que describe exactamente qué se está viendo
            if promedio_diario:
                _unidad = NOMBRE_DIAS[tipo_dia_sel][0 if n_dias == 1 else 1]
                _cuando = f"Promedio diario · {n_dias} {_unidad}"
            else:
                _cuando = f"Día {dia_sel}"
            if linea_sel != "Todas":
                _que = linea_sel + (f"  ·  ramal {ramal_sel}" if por_ramal else "")
            else:
                _partes = [modo_sel if modo_sel != "Todos" else "todos los modos"]
                if grupo_sel != "Todos":
                    _partes.append(grupo_sel)
                if empresa_sel != "Todos":
                    _partes.append(empresa_sel)
                _partes.append(f"{n_lineas} líneas")
                _que = " · ".join(_partes)
            st.markdown(f"### {_cuando} — {_que}")

            # con una línea elegida se muestra su empresa, que viene de la
            # metadata de líneas. Puede haber más de una si la línea cambió de
            # operador entre los días seleccionados.
            if linea_sel != "Todas":
                _emps = valores_filtro(df_sel, "empresa")
                if _emps:
                    _etq = "Empresa" if len(_emps) == 1 else "Empresas"
                    # encabezado y no caption: identifica de quién son los
                    # números, así que se lee al mismo nivel que el título
                    st.markdown(f"##### {_etq}: {' · '.join(_emps)}")

            # con varios días los conteos de vehículos son el promedio de los
            # conteos diarios (por eso pueden tener decimales); con uno solo
            # son el conteo de ese día
            _varios_dias = promedio_diario and n_dias > 1
            _sufijo = " (promedio)" if _varios_dias else ""
            metric_row(agg, GENERAL_COLS, labels={
                c: LABELS[c] + _sufijo
                for c in ("vehiculos_operativos", "cant_internos_en_trx")
            })
            if _varios_dias:
                st.caption(
                    "Los conteos son el **promedio de los totales de cada día** "
                    f"({n_dias} días), no la suma del período. Elegí un día "
                    "concreto para ver su total, o mirá el acumulado más abajo."
                )

            # Qué líneas entran en lo que se muestra: con los filtros de modo,
            # grupo y empresa no siempre es obvio. Una fila por línea, y la
            # descarga de las filas diarias de esa selección.
            # Una fila por línea aunque haya cambiado de empresa (o de grupo)
            # entre los días: esas columnas juntan los valores distintos.
            _cols_lista = [c for c in ["id_linea", "nombre_linea", "modo"]
                           if c in df_sel.columns]
            _juntar = [c for c in ["grupo", "empresa"] if c in df_sel.columns]
            _lista = (
                df_sel.groupby(_cols_lista, dropna=False, as_index=False)
                .agg(
                    **{c: (c, lambda s: " · ".join(valores_filtro(s.to_frame(), s.name)))
                       for c in _juntar},
                    dias=("dia", "nunique"),
                    transacciones=("transacciones", "sum"),
                )
                .sort_values("nombre_linea", key=lambda x: x.astype(str).str.casefold())
            )
            _lista["transacciones"] = _lista["transacciones"].round().astype("Int64")
            with st.expander(f"Líneas en la selección ({len(_lista)})", expanded=False):
                st.dataframe(
                    _lista.rename(columns={
                        "id_linea": "Id", "nombre_linea": "Línea", "modo": "Modo",
                        "grupo": "Grupo", "empresa": "Empresa", "dias": "Días",
                        "transacciones": "Transacciones (suma de los días)",
                    }),
                    width="stretch", hide_index=True,
                )
                st.download_button(
                    label="Descargar indicadores diarios de esta selección (CSV)",
                    data=df_sel.drop(columns=[COBERTURA_POR_ETAPAS], errors="ignore")
                    .to_csv(index=False).encode("utf-8"),
                    file_name="kpis_lineas_seleccion.csv",
                    mime="text/csv",
                )
            st.divider()

            _bloques_demo = bloques_demo_con_datos(agg)
            if _bloques_demo:
                st.markdown("#### Género y tipo de tarifa")
                for _cols_demo in _bloques_demo:
                    metric_row(agg, _cols_demo, pct=True)
                if _varios_dias:
                    st.caption(
                        f"Promedio diario de transacciones ({n_dias} días); el "
                        "porcentaje es sobre las transacciones del mismo promedio."
                    )
                elif not promedio_diario:
                    st.caption(f"Transacciones del día {dia_sel}.")
                st.divider()

            # Acumulado del período: solo tiene sentido con varios días y solo
            # para lo que se puede sumar. Los conteos de vehículos quedan fuera
            # (ver PERIOD_SUM_COLS) porque sumarlos daría vehículo-día.
            if _varios_dias:
                acum = agregar_sistema(df_sel, promedio_diario=False)
                st.markdown(f"#### Acumulado del período · {n_dias} días")
                # los km de las dos familias de recorrido, para poder comparar
                metric_row(
                    acum, ["transacciones", "tot_km_route", "tot_km_route_gps"]
                )
                for _cols_demo in bloques_demo_con_datos(acum):
                    metric_row(acum, _cols_demo, pct=True)
                st.caption(
                    "Suma de los días seleccionados. Solo se acumulan las "
                    "variables aditivas: transacciones, km y su apertura por "
                    "género y tarifa. Los vehículos no se suman entre días "
                    "porque el mismo interno reaparece cada día."
                )
                st.divider()

            # con varios días los indicadores operativos son el promedio del
            # período (ratios ponderados), no los de un día
            st.markdown("#### Operativos" + (" (promedio)" if _varios_dias else ""))
            bloque_familias(agg)
            st.caption(
                "Los indicadores operativos son **ratios**: no se suman, se agregan "
                "ponderando cada uno por su propio denominador. Los de **oferta** "
                "(velocidad comercial, IPK, factor de ocupación, km por vehículo) "
                "se ponderan por oferta —km de esa misma familia de distancia, o "
                "vehículos—, nunca por demanda; los del **pasajero** (distancia y "
                "tiempo de viaje) por transacciones. La velocidad comercial usa "
                "media armónica sobre los km, que equivale a km totales / horas "
                "totales.  \n"
                "El **factor de ocupación** aparece solo en las dos familias de "
                "recorrido: su denominador son los km que efectivamente recorrió "
                "el vehículo, así que no tiene una versión OD."
            )
            st.caption(
                nota_capacidad_modos(
                    df_sel["modo"] if "modo" in df_sel.columns else [],
                    leer_configs_generales(autogenerado=False).get("capacidad_modos"),
                )
            )

            cob = agg["cobertura_gps"].iloc[0]
            if modo_sel == "Todos" and linea_sel == "Todas":
                st.caption(
                    "⚠️ El agregado mezcla modos: subte y tren no reportan GPS, así "
                    "que deprimen la cobertura y distorsionan los indicadores de "
                    "oferta. Conviene mirarlos modo por modo."
                )
            elif pd.isna(cob):
                st.caption(
                    "⚠️ No hay GPS en esta selección: los indicadores derivados de "
                    "servicios (km, velocidad comercial, IPK) no están disponibles. "
                    "Los internos con transacciones sí son un dato real."
                )
            elif cob < 0.8:
                st.caption(
                    f"⚠️ Cobertura GPS baja ({cob * 100:.1f} %): los indicadores de "
                    "oferta describen solo la parte de la operación que tiene GPS."
                )

            _nota_cobertura = nota_cobertura(df_sel)
            if _nota_cobertura:
                st.caption(_nota_cobertura)


# -----------------------------------------------------------------------------
# 3. Base completa
# -----------------------------------------------------------------------------

with st.expander("Base completa", expanded=False):
    # Solo días reales: las filas "Promedios" son un resumen por línea y, si se
    # suman junto con los días en una planilla, inflan los totales del período.
    # la marca interna de cómo se midió la cobertura no es un dato de la base
    kpis_export = kpis_df.drop(columns=[COBERTURA_POR_ETAPAS], errors="ignore")
    kpis_tabla = kpis_export[kpis_export["dia"] != "Promedios"]
    st.dataframe(kpis_tabla, width="stretch", hide_index=True)
    st.caption(
        "La tabla y el CSV completo incluyen sólo los días procesados. "
        "El resumen por línea se descarga por separado y no debe sumarse a los días."
    )
    col_desc_dias, col_desc_resumen = st.columns(2)
    with col_desc_dias:
        st.download_button(
            label="Descargar CSV completo",
            data=kpis_tabla.to_csv(index=False).encode("utf-8"),
            file_name="kpis_lineas_completo.csv",
            mime="text/csv",
        )
    kpis_resumen = kpis_export[kpis_export["dia"] == "Promedios"]
    if not kpis_resumen.empty:
        with col_desc_resumen:
            st.download_button(
                label="Descargar resumen por línea",
                data=kpis_resumen.to_csv(index=False).encode("utf-8"),
                file_name="kpis_lineas_resumen.csv",
                mime="text/csv",
            )
