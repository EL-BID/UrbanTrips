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

import streamlit as st
import pandas as pd
from dash_utils import levanto_tabla_sql, get_logo, configurar_selector_corrida

# La lógica de agregación vive en el paquete, no acá: esta página no es
# importable (ejecuta `st.*` a nivel de módulo) y por lo tanto no se podía
# cubrir con tests. Ver urbantrips/kpi/agregacion_indicadores.py.
from urbantrips.kpi.agregacion_indicadores import (
    DEMO_COLS,
    NOMBRE_DIAS,
    TIPOS_DIA,
    agregar_sistema,
    etiqueta_dia,
    preparar,
)

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


@st.cache_data(show_spinner=False)
def load_kpis() -> pd.DataFrame:
    df = levanto_tabla_sql("kpis_lineas", "general")
    return df


@st.cache_data(show_spinner=False)
def load_kpis_ramales() -> pd.DataFrame:
    """`kpis_ramales` existe solo si la corrida tuvo lineas_contienen_ramales=True."""
    try:
        df = levanto_tabla_sql("kpis_ramales", "general")
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


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
    "vehiculos_operativos": "Vehículos operativos (GPS)",
    "cant_internos_en_trx": "Internos con transacciones",
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
    "fo_mean_od": "Factor Ocupación (media)",
    "fo_median_od": "Factor Ocupación (mediana)",
    "dmt_mean_od": "Distancia media Pax (km)",
    "dmt_median_od": "Distancia mediana Pax (km)",
    # familia _route (recorrido reconstruido ping a ping)
    "velocidad_comercial_route": "Vel. comercial (km/h)",
    "distancia_media_veh_route": "Dist. media/veh (km)",
    "fo_mean_route": "Factor Ocupación (media)",
    "fo_median_route": "Factor Ocupación (mediana)",
    "tot_km_route": "Km recorridos",
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
#   pvd              pasajeros por vehículo = tot_pax / tot_veh
SIN_DISTANCIA = ["travel_time_min", "pvd"]

FAMILIAS = {
    "od": {
        "label": "OD (origen–destino)",
        # Acá NO va el factor de ocupación. En kpi.py se calcula como
        #     fo_*_od = (tot_pax · dmt_*_od) / (tot_km_route · 60)
        # o sea con el numerador en distancia OD pero el denominador en km de
        # recorrido real: la oferta solo existe en kilómetros recorridos. Es un
        # híbrido, no un indicador de esta familia, y mostrarlo acá hacía
        # pensar que la ocupación se mide sobre la distancia OD. Queda en las
        # dos familias de recorrido, donde numerador y denominador coinciden.
        #
        # `distancia_media_pax` también se omite: es la misma media ponderada de
        # `distance_od` que `dmt_mean_od` por otro camino (369 de 402 líneas
        # idénticas, el resto difiere ±0,1 km por redondeo).
        "ayuda": "Distancia en camino mínimo entre el origen y el destino de la "
                 "etapa, sobre la grilla H3. Mide el desplazamiento del "
                 "pasajero, no el recorrido del vehículo: por eso no tiene "
                 "velocidad comercial, ni IPK, ni km de oferta, ni factor de "
                 "ocupación.",
        "filas": [["kmh_od", "dmt_mean_od", "dmt_median_od"]],
        "probe": ["dmt_mean_od", "fo_mean_od"],
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
            ["tot_km_route", "distancia_media_veh_route", "ipk_route",
             "fo_mean_route", "fo_median_route"],
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

INT_DISPLAY_COLS = (
    set(GENERAL_COLS + DEMO_COLS + ["tot_km_route", "tot_km_route_gps"])
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
    if col in INT_DISPLAY_COLS or (
        isinstance(val, (int, float)) and float(val).is_integer()
    ):
        return f"{int(round(val)):,}"
    return f"{val:,.2f}"


def metric_row(df: pd.DataFrame, cols: list[str], *, pct=False):
    """Muestra métricas en una fila de TOTAL_SLOTS columnas fijas."""
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
        st_cols[idx].metric(LABELS.get(col, col), text)



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


# -----------------------------------------------------------------------------
# Panel único: Día × Modo × Línea
# -----------------------------------------------------------------------------
# Antes había dos secciones —"KPIs por línea" y "Totales del sistema"— que
# mostraban los mismos indicadores con criterios distintos: la de línea leía la
# fila `dia = "Promedios"` que escribe `calculo_kpi_lineas` (media simple de las
# filas línea-día, sin ponderar los ratios) y la de sistema los recalculaba
# ponderados. Un mismo indicador daba dos números según dónde se lo mirara.
#
# Ahora hay un solo panel y un solo criterio de agregación (`agregar_sistema`).
# La fila "Promedios" de la tabla ya no se usa para nada: "Día = Todos" produce
# el promedio bien calculado. Los tres selectores cubren todos los casos:
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

        # las líneas ofrecidas dependen del modo, para no listar 400 cuando ya
        # se filtró a uno solo
        _base_lineas = (
            kpis_dias if modo_sel == "Todos" else kpis_dias[kpis_dias["modo"] == modo_sel]
        )
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
            elif modo_sel != "Todos":
                _que = f"{modo_sel} · {n_lineas} líneas"
            else:
                _que = f"todos los modos · {n_lineas} líneas"
            st.markdown(f"### {_cuando} — {_que}")

            # con una línea elegida se muestra su empresa, que viene de la
            # metadata de líneas. Puede haber más de una si la línea cambió de
            # operador entre los días seleccionados.
            if linea_sel != "Todas" and "empresa" in df_sel.columns:
                _emps = sorted(
                    {e.strip() for e in df_sel["empresa"].dropna().astype(str)
                     if e.strip() and e.strip().lower() != "nan"}
                )
                if _emps:
                    _etq = "Empresa" if len(_emps) == 1 else "Empresas"
                    # encabezado y no caption: identifica de quién son los
                    # números, así que se lee al mismo nivel que el título
                    st.markdown(f"##### {_etq}: {' · '.join(_emps)}")

            metric_row(agg, GENERAL_COLS)
            if promedio_diario and n_dias > 1:
                st.caption(
                    "Los conteos son el **promedio de los totales de cada día** "
                    f"({n_dias} días), no la suma del período. Elegí un día "
                    "concreto para ver su total, o mirá el acumulado más abajo."
                )
            st.divider()

            st.markdown("#### Género y tipo de tarifa")
            metric_row(agg, DEMO_COLS, pct=True)
            st.divider()

            # Acumulado del período: solo tiene sentido con varios días y solo
            # para lo que se puede sumar. Los conteos de vehículos quedan fuera
            # (ver PERIOD_SUM_COLS) porque sumarlos daría vehículo-día.
            if promedio_diario and n_dias > 1:
                acum = agregar_sistema(df_sel, promedio_diario=False)
                st.markdown(f"#### Acumulado del período · {n_dias} días")
                # los km de las dos familias de recorrido, para poder comparar
                metric_row(
                    acum, ["transacciones", "tot_km_route", "tot_km_route_gps"]
                )
                metric_row(acum, DEMO_COLS, pct=True)
                st.caption(
                    "Suma de los días seleccionados. Solo se acumulan las "
                    "variables aditivas: transacciones, km y su apertura por "
                    "género y tarifa. Los vehículos no se suman entre días "
                    "porque el mismo interno reaparece cada día."
                )
                st.divider()

            st.markdown("#### Operativos")
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
                    "oferta describen solo la fracción de la flota que reporta GPS."
                )


# -----------------------------------------------------------------------------
# 3. Base completa
# -----------------------------------------------------------------------------

with st.expander("Base completa", expanded=False):
    st.dataframe(kpis_df, width="stretch", hide_index=True)
    st.download_button(
        label="Descargar CSV completo",
        data=kpis_df.to_csv(index=False).encode("utf-8"),
        file_name="kpis_lineas_completo.csv",
        mime="text/csv",
    )
