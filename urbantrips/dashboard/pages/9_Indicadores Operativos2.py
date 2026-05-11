# -*- coding: utf-8 -*-
"""
Dashboard de Indicadores Operativos – UrbanTrips
================================================

Lee de las tablas de KPI generadas por urbantrips.kpi:
- kpi_by_day_line: KPIs línea-día (pax, km, IPK, FO, DMT con tres distancias)
- basic_kpi_by_line_day / basic_kpi_by_line_hr: KPIs operativos básicos
- kpi_by_day_line_service: KPIs a nivel servicio
- services_by_line_hour: cantidad de servicios despachados

Las tres distancias disponibles son:
- od: shortest path por red entre origen y destino del pasajero
- route: distancia recorrida sobre la traza GPS calculada por UrbanTrips
- route_gps: distancia recorrida según odómetro de la validadora
"""

import html as _html
import streamlit as st
import pandas as pd
import plotly.express as px
from dash_utils import (
    levanto_tabla_sql,
    levanto_tabla_sql_local,
    get_logo,
    configurar_selector_dia,
)

# -----------------------------------------------------------------------------
# Configuración global
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Indicadores Operativos", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] > div:first-child {font-size:0.75rem;}
    div[data-testid="metric-container"] > div:nth-child(2) {font-size:1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    st.image(get_logo())
except Exception:
    pass

try:
    alias_sel = configurar_selector_dia()
except Exception:
    alias_sel = "default"

# -----------------------------------------------------------------------------
# Diccionarios de etiquetas y tooltips
# -----------------------------------------------------------------------------

LABELS = {
    # Identificadores
    "id_linea": "ID línea",
    "nombre_linea": "Línea",
    "id_ramal": "Ramal",
    "interno": "Interno",
    "service_id": "ID servicio",
    "modo": "Modo",
    "empresa": "Empresa",
    "dia": "Día",
    "yr_mo": "Año-Mes",
    "hora": "Hora",
    "hora_inicio": "Hora inicio",
    "hora_fin": "Hora fin",

    # Oferta
    "tot_veh": "Vehículos (tot_veh)",
    "tot_km": "Km recorrido (tot_km)",
    "tot_km_gps": "Km odómetro (tot_km_gps)",
    "veh": "Vehículos (veh)",
    "servicios": "Servicios (servicios)",

    # Demanda
    "tot_pax": "Pasajeros (tot_pax)",
    "pax": "Pasajeros (pax)",
    "factor_expansion_linea": "Factor exp. (factor_expansion_linea)",

    # DMT
    "dmt_mean_od": "DMT media – red (dmt_mean_od)",
    "dmt_mean_route": "DMT media – recorrido (dmt_mean_route)",
    "dmt_mean_route_gps": "DMT media – odómetro (dmt_mean_route_gps)",
    "dmt_median_od": "DMT mediana – red (dmt_median_od)",
    "dmt_median_route": "DMT mediana – recorrido (dmt_median_route)",
    "dmt_median_route_gps": "DMT mediana – odómetro (dmt_median_route_gps)",
    "dmt_route": "DMT recorrido (dmt_route)",
    "dmt_route_gps": "DMT odómetro (dmt_route_gps)",

    # Productividad
    "pvd": "PVD (pvd)",
    "kvd": "KVD recorrido (kvd)",
    "kvd_gps": "KVD odómetro (kvd_gps)",
    "ipk_route": "IPK recorrido (ipk_route)",
    "ipk_route_gps": "IPK odómetro (ipk_route_gps)",

    # EKD / EKO / FO
    "ekd_mean_od": "EKD media – red (ekd_mean_od)",
    "ekd_mean_route": "EKD media – recorrido (ekd_mean_route)",
    "ekd_mean_route_gps": "EKD media – odómetro (ekd_mean_route_gps)",
    "eko": "EKO (eko)",
    "eko_gps": "EKO odómetro (eko_gps)",
    "fo_mean_od": "FO medio – red (fo_mean_od)",
    "fo_mean_route": "FO medio – recorrido (fo_mean_route)",
    "fo_mean_route_gps": "FO medio – odómetro (fo_mean_route_gps)",
    "fo_median_od": "FO mediana – red (fo_median_od)",
    "fo_median_route": "FO mediana – recorrido (fo_median_route)",
    "fo_median_route_gps": "FO mediana – odómetro (fo_median_route_gps)",

    # Ocupación a nivel vehículo
    "eq_pax": "Pax-equiv. (eq_pax)",
    "eq_pax_gps": "Pax-equiv. odómetro (eq_pax_gps)",
    "of": "Ocupación (of)",

    # Velocidades / tiempos
    "kmh_route": "Velocidad recorrido (kmh_route)",
    "kmh_route_gps": "Velocidad odómetro (kmh_route_gps)",
    "kmh_od": "Velocidad red (kmh_od)",
    "travel_time_min": "Tiempo viaje min (travel_time_min)",
}

HELP_TEXTS = {
    # Identificadores
    "id_linea": "Identificador único de la línea de transporte.",
    "nombre_linea": "Nombre de la línea de transporte.",
    "id_ramal": "Identificador del ramal dentro de la línea (si la línea tiene ramales).",
    "interno": "Identificador del vehículo (coche) que prestó el servicio.",
    "service_id": "Identificador del servicio (recorrido completo de un vehículo).",
    "modo": "Modo de transporte (colectivo, tren, subte, etc.).",
    "empresa": "Empresa operadora de la línea.",
    "dia": "Fecha (YYYY-MM-DD) o tipo de día agregado (weekday/weekend).",
    "yr_mo": "Año-mes (YYYY-MM) para análisis de evolución temporal.",
    "hora": "Hora del día (0-23).",
    "hora_inicio": "Hora de inicio del servicio.",
    "hora_fin": "Hora de finalización del servicio.",

    # Oferta
    "tot_veh": "Cantidad total de vehículos que operaron en el día, expandida "
               "por el factor de expansión vehicular.",
    "tot_km": "Kilómetros recorridos totales calculados sobre la traza GPS por "
              "UrbanTrips. Incluye expansión vehicular.",
    "tot_km_gps": "Kilómetros recorridos según el odómetro reportado por la "
                  "validadora a bordo. Incluye expansión vehicular.",
    "veh": "Cantidad de vehículos únicos que operaron (sin expansión).",
    "servicios": "Cantidad de servicios despachados en la hora.",

    # Demanda
    "tot_pax": "Cantidad total de pasajeros, expandida por factor_expansion_linea.",
    "pax": "Suma de factor_expansion_linea de las etapas validadas.",
    "factor_expansion_linea": "Factor de expansión que ajusta los pasajeros "
                              "captados a la demanda total estimada de la línea.",

    # DMT - tres variantes
    "dmt_mean_od": "Distancia Media de Viaje del pasajero, calculada como promedio "
                   "ponderado de distance_od (shortest path por red) por "
                   "factor_expansion_linea.",
    "dmt_mean_route": "DMT calculada con la distancia recorrida sobre la traza "
                      "GPS (distance_route), ponderada por pasajeros.",
    "dmt_mean_route_gps": "DMT calculada con la distancia del odómetro "
                          "(distance_route_gps), ponderada por pasajeros.",
    "dmt_median_od": "Mediana ponderada de distance_od.",
    "dmt_median_route": "Mediana ponderada de distance_route.",
    "dmt_median_route_gps": "Mediana ponderada de distance_route_gps.",
    "dmt_route": "DMT a nivel etapa, promedio ponderado de distance_route.",
    "dmt_route_gps": "DMT a nivel etapa, promedio ponderado de distance_route_gps.",

    # Productividad
    "pvd": "Pasajeros por Vehículo-Día. Se calcula como tot_pax / tot_veh.",
    "kvd": "Kilómetros por Vehículo-Día calculados sobre traza GPS. "
           "Se calcula como tot_km / tot_veh.",
    "kvd_gps": "Kilómetros por Vehículo-Día según odómetro. "
               "Se calcula como tot_km_gps / tot_veh.",
    "ipk_route": "Índice Pasajero-Kilómetro: pasajeros transportados por km "
                 "recorrido. Se calcula como tot_pax / tot_km.",
    "ipk_route_gps": "IPK calculado con km del odómetro. "
                     "Se calcula como tot_pax / tot_km_gps.",

    # EKD / EKO / FO
    "ekd_mean_od": "Espacios-Kilómetro Demandados (mean): tot_pax × dmt_mean_od. "
                   "Mide el espacio-kilómetro consumido por los pasajeros.",
    "ekd_mean_route": "Espacios-Kilómetro Demandados con dmt_mean_route.",
    "ekd_mean_route_gps": "Espacios-Kilómetro Demandados con dmt_mean_route_gps.",
    "eko": "Espacios-Kilómetro Ofertados: tot_km × 60. Asume capacidad fija "
           "de 60 espacios por vehículo.",
    "eko_gps": "EKO calculado con tot_km_gps (odómetro): tot_km_gps × 60.",
    "fo_mean_od": "Factor de Ocupación medio: ekd_mean_od / eko. "
                  "Proporción del espacio ofertado que fue efectivamente ocupado.",
    "fo_mean_route": "Factor de Ocupación medio con distancia route: "
                     "ekd_mean_route / eko.",
    "fo_mean_route_gps": "Factor de Ocupación medio con odómetro: "
                         "ekd_mean_route_gps / eko_gps.",
    "fo_median_od": "Factor de Ocupación con mediana de distance_od.",
    "fo_median_route": "Factor de Ocupación con mediana de distance_route.",
    "fo_median_route_gps": "Factor de Ocupación con mediana de distance_route_gps.",

    # Ocupación a nivel vehículo (basic_kpi)
    "eq_pax": "Pasajeros-equivalente: horas-pasajero consumidas. Se calcula "
              "como (distance_route / kmh_route) × factor_expansion_linea, "
              "sumado por vehículo-hora.",
    "eq_pax_gps": "Pasajeros-equivalente con datos del odómetro: usa "
                  "distance_route_gps y kmh_route_gps. Cobertura menor que "
                  "eq_pax porque el odómetro puede faltar y velocidades "
                  "atípicas (>100 km/h) se descartan.",
    "of": "Factor de Ocupación a nivel vehículo-hora: eq_pax / 60 × 100. "
          "Expresado como porcentaje de la capacidad de 60 espacios.",

    # Velocidades / tiempos
    "kmh_route": "Velocidad comercial en km/h calculada sobre la traza GPS.",
    "kmh_route_gps": "Velocidad comercial en km/h según odómetro. NaN si la "
                     "velocidad implícita supera 100 km/h (filtro de atípicos).",
    "kmh_od": "Velocidad implícita asumiendo trayecto recto entre origen y destino.",
    "travel_time_min": "Tiempo de viaje del pasajero en minutos.",
}


LABELS_TWO_LINE = {
    "id_linea": "ID<br>línea",
    "nombre_linea": "Línea",
    "id_ramal": "Ramal",
    "interno": "Interno",
    "service_id": "Servicio",
    "modo": "Modo",
    "empresa": "Empresa",
    "dia": "Día",
    "yr_mo": "Año-Mes",
    "hora": "Hora",
    "hora_inicio": "Hora<br>inicio",
    "hora_fin": "Hora<br>fin",
    "tot_veh": "Vehículos",
    "tot_km": "Km<br>recorrido",
    "tot_km_gps": "Km<br>odómetro",
    "veh": "Vehículos",
    "servicios": "Servicios",
    "tot_pax": "Pasajeros",
    "pax": "Pasajeros",
    "factor_expansion_linea": "Factor<br>expansión",
    "dmt_mean_od": "DMT media<br>red",
    "dmt_mean_route": "DMT media<br>recorrida",
    "dmt_mean_route_gps": "DMT media<br>odóm.",
    "dmt_median_od": "DMT med.<br>red",
    "dmt_median_route": "DMT med.<br>recorrida",
    "dmt_median_route_gps": "DMT med.<br>odóm.",
    "dmt_route": "DMT<br>recorrido",
    "dmt_route_gps": "DMT<br>odómetro",
    "pvd": "PVD",
    "kvd": "KVD<br>recorrido",
    "kvd_gps": "KVD<br>odómetro",
    "ipk_route": "IPK<br>recorrido",
    "ipk_route_gps": "IPK<br>odómetro",
    "ekd_mean_od": "EKD media<br>red",
    "ekd_mean_route": "EKD media<br>recorrida",
    "ekd_mean_route_gps": "EKD media<br>odóm.",
    "eko": "EKO",
    "eko_gps": "EKO<br>odóm.",
    "fo_mean_od": "FO medio<br>red",
    "fo_mean_route": "FO medio<br>recorrida",
    "fo_mean_route_gps": "FO medio<br>odóm.",
    "fo_median_od": "FO med.<br>red",
    "fo_median_route": "FO med.<br>recorrida",
    "fo_median_route_gps": "FO med.<br>odóm.",
    "eq_pax": "Pax<br>equiv.",
    "eq_pax_gps": "Pax equiv.<br>odóm.",
    "of": "Ocupación",
    "kmh_route": "Vel.<br>recorrido",
    "kmh_route_gps": "Vel.<br>odómetro",
    "kmh_od": "Vel.<br>red",
    "travel_time_min": "T. viaje<br>(min)",
}


def label_of(col: str) -> str:
    """Etiqueta amigable; si no está en LABELS devuelve el nombre crudo."""
    return LABELS.get(col, col)


def label_html(col: str) -> str:
    """Etiqueta para tablas HTML (puede contener <br> para dos renglones)."""
    return LABELS_TWO_LINE.get(col, label_of(col))


def help_of(col: str) -> str | None:
    return HELP_TEXTS.get(col)


# -----------------------------------------------------------------------------
# Carga de datos
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_metadata_lineas() -> pd.DataFrame:
    return levanto_tabla_sql("metadata_lineas", "insumos")


@st.cache_data(show_spinner=False)
def load_kpi_by_day_line() -> pd.DataFrame:
    return levanto_tabla_sql("kpi_by_day_line", "data")


@st.cache_data(show_spinner=False)
def load_basic_kpi_by_line_day() -> pd.DataFrame:
    return levanto_tabla_sql("basic_kpi_by_line_day", "data")


@st.cache_data(show_spinner=False)
def load_basic_kpi_by_line_hr() -> pd.DataFrame:
    return levanto_tabla_sql("basic_kpi_by_line_hr", "data")


@st.cache_data(show_spinner=False)
def load_services_by_line_hour() -> pd.DataFrame:
    return levanto_tabla_sql("services_by_line_hour", "data")


def load_kpi_by_service(line_ids: list, day_filter: str) -> pd.DataFrame:
    """Tabla potencialmente grande: se filtra desde SQL para evitar cargar todo."""
    if not line_ids:
        return pd.DataFrame()
    line_ids_str = ",".join(str(int(i)) for i in line_ids)
    where_dia = "" if day_filter == "Todos" else f"AND dia = '{day_filter}'"
    query = f"""
        SELECT * FROM kpi_by_day_line_service
        WHERE id_linea IN ({line_ids_str}) {where_dia}
    """
    return levanto_tabla_sql_local(
        "kpi_by_day_line_service", "data", query=query
    )


metadata = load_metadata_lineas()
kpi_day = load_kpi_by_day_line()
basic_day = load_basic_kpi_by_line_day()
basic_hr = load_basic_kpi_by_line_hr()
services_hr = load_services_by_line_hour()


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or metadata.empty:
        return df
    meta_cols = [c for c in ["id_linea", "nombre_linea", "modo", "empresa"]
                 if c in metadata.columns]
    return df.merge(metadata[meta_cols], on="id_linea", how="left")


kpi_day = add_metadata(kpi_day)
basic_day = add_metadata(basic_day)
basic_hr = add_metadata(basic_hr)
services_hr = add_metadata(services_hr)

# -----------------------------------------------------------------------------
# Filtros globales
# -----------------------------------------------------------------------------

st.markdown("## Indicadores Operativos")

col_f1, col_f2, col_f3 = st.columns([4, 2, 3])

with col_f1:
    if not metadata.empty:
        line_options = (
            metadata.sort_values("nombre_linea")["nombre_linea"]
            .dropna()
            .unique()
            .tolist()
        )
        lineas_sel = st.multiselect(
            "Líneas",
            line_options,
            default=line_options[:1] if line_options else [],
            help="Selección de líneas a analizar. El nombre identifica la línea; "
                 "internamente se usa id_linea.",
        )
        line_ids_sel = metadata.loc[
            metadata["nombre_linea"].isin(lineas_sel), "id_linea"
        ].tolist()
    else:
        st.warning("No hay metadata de líneas disponible.")
        lineas_sel = []
        line_ids_sel = []

with col_f2:
    days = ["Todos"]
    if not kpi_day.empty:
        days += sorted(kpi_day["dia"].dropna().unique().tolist())
    day_sel = st.selectbox(
        "Día / tipo de día", days, index=0,
        help="Fecha específica (YYYY-MM-DD) o agregado por tipo de día. "
             "'weekday' y 'weekend' son promedios calculados a partir de "
             "los días procesados.",
    )

with col_f3:
    distance_options = {
        "Red (od)": "od",
        "Recorrido (route)": "route",
        "Odómetro (route_gps)": "route_gps",
    }
    dist_sel_labels = st.multiselect(
        "Distancias a mostrar",
        list(distance_options.keys()),
        default=["Recorrido (route)"],
        help="Cada KPI de distancia tiene tres variantes:\n"
             "• od: shortest path por red entre origen y destino del pasajero\n"
             "• route: distancia sobre traza GPS calculada por UrbanTrips\n"
             "• route_gps: odómetro reportado por la validadora",
    )
    dist_sel = [distance_options[d] for d in dist_sel_labels]

if not line_ids_sel:
    st.info("Seleccioná al menos una línea para ver los indicadores.")
    st.stop()

# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------

def filter_kpi(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["id_linea"].isin(line_ids_sel)].copy()
    if day_sel != "Todos" and "dia" in out.columns:
        out = out[out["dia"] == day_sel]
    return out


def cols_for_distances(prefix: str, suffixes: list) -> list:
    """Columnas tipo prefix_suffix para las distancias seleccionadas."""
    return [f"{prefix}_{s}" for s in suffixes if s in dist_sel]


def metric_with_help(container, col_key, value_str, custom_label=None):
    """Wrapper de st.metric que toma label y help del diccionario."""
    label = custom_label if custom_label else label_of(col_key)
    container.metric(label, value_str, help=help_of(col_key))


_INT_COLS = {
    "tot_veh", "tot_pax", "veh", "pax", "servicios",
    "hora", "hora_inicio", "hora_fin", "id_linea",
    "id_ramal", "interno", "service_id",
}

_TABLE_CSS = (
    "<style>"
    ".kpi-tbl{width:100%;border-collapse:collapse;font-size:0.82rem}"
    ".kpi-tbl th{background:rgba(120,120,120,.12);text-align:center;"
    "padding:5px 8px;border:1px solid rgba(120,120,120,.25);"
    "white-space:normal;min-width:55px;vertical-align:bottom;line-height:1.3}"
    ".kpi-tbl td{padding:3px 8px;border:1px solid rgba(120,120,120,.15);"
    "white-space:nowrap}"
    ".kpi-tbl tr:nth-child(even){background:rgba(120,120,120,.04)}"
    "</style>"
)


def _fmt_ar(val, col: str) -> str:
    """Formato numérico argentino: punto como miles, coma como decimal."""
    if pd.isna(val):
        return "–"
    if col.startswith("fo_") or col == "of":
        return f"{float(val):.3f}".replace(".", ",")
    if col in _INT_COLS:
        try:
            return f"{int(round(float(val))):,}".replace(",", ".")
        except Exception:
            return str(val)
    try:
        s = f"{float(val):,.2f}"          # "1,234.56"
        int_part, dec_part = s.split(".")
        return f"{int_part.replace(',', '.')},{dec_part}"   # "1.234,56"
    except Exception:
        return str(val)


def show_table(df: pd.DataFrame, key: str):
    """Renderiza tabla HTML con encabezados en dos renglones y formato argentino."""
    if df.empty:
        st.info("No hay datos para los filtros seleccionados.")
        return

    _MAX_ROWS = 2000
    display_df = df.head(_MAX_ROWS) if len(df) > _MAX_ROWS else df
    if len(df) > _MAX_ROWS:
        st.warning(f"Mostrando {_MAX_ROWS:,} de {len(df):,} filas.")

    cols = display_df.columns.tolist()

    header_cells = "".join(
        f'<th title="{_html.escape(help_of(c) or "")}">{label_html(c)}</th>'
        for c in cols
    )

    row_parts = []
    for _, row in display_df.iterrows():
        cells = ""
        for c in cols:
            val = row[c]
            if pd.api.types.is_numeric_dtype(display_df[c]):
                cells += f'<td style="text-align:right">{_fmt_ar(val, c)}</td>'
            else:
                txt = _html.escape(str(val)) if pd.notna(val) else "–"
                cells += f"<td>{txt}</td>"
        row_parts.append(f"<tr>{cells}</tr>")

    st.markdown(
        _TABLE_CSS
        + '<div style="overflow-x:auto;max-height:500px;overflow-y:auto">'
        + '<table class="kpi-tbl"><thead><tr>'
        + header_cells
        + "</tr></thead><tbody>"
        + "".join(row_parts)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        label="Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def format_value(val, col):
    """Formato argentino para st.metric."""
    if pd.isna(val):
        return "–"
    if col.startswith("fo_") or col == "of":
        return f"{float(val):.3f}".replace(".", ",")
    if col in _INT_COLS or (isinstance(val, (int, float)) and float(val).is_integer()):
        try:
            return f"{int(round(float(val))):,}".replace(",", ".")
        except Exception:
            return str(val)
    try:
        s = f"{float(val):,.2f}"
        int_part, dec_part = s.split(".")
        return f"{int_part.replace(',', '.')},{dec_part}"
    except Exception:
        return str(val)


# -----------------------------------------------------------------------------
# 1. KPIs línea-día (kpi_by_day_line)
# -----------------------------------------------------------------------------

with st.expander("KPIs por línea y día", expanded=True):
    df1 = filter_kpi(kpi_day)

    if df1.empty:
        st.info("Sin datos para los filtros seleccionados.")
    else:
        base_cols = [
            "nombre_linea", "modo", "dia",
            "tot_veh", "tot_km", "tot_km_gps", "tot_pax",
            "pvd", "kvd", "kvd_gps",
            "ipk_route", "ipk_route_gps",
        ]
        dmt_cols = (
            cols_for_distances("dmt_mean", ["od", "route", "route_gps"]) +
            cols_for_distances("dmt_median", ["od", "route", "route_gps"])
        )
        fo_cols = (
            cols_for_distances("fo_mean", ["od", "route", "route_gps"]) +
            cols_for_distances("fo_median", ["od", "route", "route_gps"])
        )
        cols_show = [c for c in base_cols + dmt_cols + fo_cols if c in df1.columns]
        df1_show = df1[cols_show].sort_values(["nombre_linea", "dia"])

        # Vista métrica si hay una sola línea y un día concreto
        if len(line_ids_sel) == 1 and day_sel != "Todos" and len(df1_show) == 1:
            row = df1_show.iloc[0]

            st.markdown("#### Oferta y demanda")
            c = st.columns(4)
            metric_with_help(c[0], "tot_veh", format_value(row.tot_veh, "tot_veh"))
            metric_with_help(c[1], "tot_pax", format_value(row.tot_pax, "tot_pax"))
            metric_with_help(c[2], "tot_km", format_value(row.tot_km, "tot_km"))
            metric_with_help(c[3], "tot_km_gps", format_value(row.tot_km_gps, "tot_km_gps"))

            st.markdown("#### Productividad")
            c = st.columns(4)
            metric_with_help(c[0], "pvd", format_value(row.pvd, "pvd"))
            metric_with_help(c[1], "kvd", format_value(row.kvd, "kvd"))
            metric_with_help(c[2], "ipk_route", format_value(row.ipk_route, "ipk_route"))
            metric_with_help(c[3], "ipk_route_gps", format_value(row.ipk_route_gps, "ipk_route_gps"))

            if dmt_cols:
                st.markdown("#### Distancia Media de Viaje (DMT)")
                c = st.columns(min(len(dmt_cols), 6))
                for i, col in enumerate(dmt_cols):
                    metric_with_help(
                        c[i % len(c)], col, format_value(row[col], col)
                    )

            if fo_cols:
                st.markdown("#### Factor de Ocupación")
                c = st.columns(min(len(fo_cols), 6))
                for i, col in enumerate(fo_cols):
                    metric_with_help(
                        c[i % len(c)], col, format_value(row[col], col)
                    )
        else:
            show_table(df1_show, "kpi_by_day_line")

        # Evolución mensual
        if "yr_mo" in df1.columns and df1["yr_mo"].nunique() > 1:
            st.markdown("#### Evolución mensual")
            evo_options = (
                ["tot_pax", "ipk_route", "ipk_route_gps"]
                + cols_for_distances("dmt_mean", ["od", "route", "route_gps"])
                + cols_for_distances("fo_mean", ["od", "route", "route_gps"])
            )
            evo_options = [c for c in evo_options if c in df1.columns]
            metric_to_plot = st.selectbox(
                "Indicador",
                evo_options,
                format_func=label_of,
                key="evo_metric",
                help="Indicador a graficar a lo largo del tiempo.",
            )
            evo = (
                df1.groupby(["yr_mo", "nombre_linea"], as_index=False)[metric_to_plot]
                .mean()
            )
            fig = px.line(
                evo, x="yr_mo", y=metric_to_plot, color="nombre_linea",
                markers=True,
                labels={"yr_mo": "Año-mes",
                        metric_to_plot: label_of(metric_to_plot),
                        "nombre_linea": "Línea"},
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 2. KPIs operativos básicos
# -----------------------------------------------------------------------------

with st.expander("KPIs operativos básicos", expanded=False):
    tab_day, tab_hr = st.tabs(["Por día", "Perfil horario"])

    with tab_day:
        df_bd = filter_kpi(basic_day)
        if df_bd.empty:
            st.info("Sin datos.")
        else:
            cols_show = [
                "nombre_linea", "modo", "dia",
                "veh", "pax", "eq_pax", "eq_pax_gps",
                "dmt_route", "dmt_route_gps",
                "of", "kmh_route",
            ]
            cols_show = [c for c in cols_show if c in df_bd.columns]
            show_table(
                df_bd[cols_show].sort_values(["nombre_linea", "dia"]),
                "basic_kpi_by_line_day",
            )

    with tab_hr:
        df_bh = filter_kpi(basic_hr)
        if df_bh.empty:
            st.info("Sin datos.")
        else:
            hr_options = ["pax", "eq_pax", "of", "kmh_route",
                           "dmt_route", "dmt_route_gps"]
            hr_options = [c for c in hr_options if c in df_bh.columns]
            metric_hr = st.selectbox(
                "Indicador",
                hr_options,
                format_func=label_of,
                key="hr_metric",
                help="Indicador a graficar por hora del día.",
            )
            fig_hr = px.line(
                df_bh.sort_values(["nombre_linea", "hora"]),
                x="hora", y=metric_hr, color="nombre_linea",
                markers=True,
                labels={"hora": "Hora del día",
                        metric_hr: label_of(metric_hr),
                        "nombre_linea": "Línea"},
            )
            st.plotly_chart(fig_hr, use_container_width=True)

            cols_show = [
                "nombre_linea", "dia", "hora",
                "veh", "pax", "eq_pax", "eq_pax_gps",
                "dmt_route", "dmt_route_gps",
                "of", "kmh_route",
            ]
            cols_show = [c for c in cols_show if c in df_bh.columns]
            show_table(
                df_bh[cols_show].sort_values(["nombre_linea", "dia", "hora"]),
                "basic_kpi_by_line_hr",
            )

# -----------------------------------------------------------------------------
# 3. KPIs por servicio
# -----------------------------------------------------------------------------

with st.expander("KPIs por servicio", expanded=False):
    df_srv = load_kpi_by_service(line_ids_sel, day_sel)
    if df_srv.empty:
        st.info("Sin datos de servicios para los filtros seleccionados.")
    else:
        df_srv = add_metadata(df_srv)

        col_r, col_i = st.columns(2)
        with col_r:
            ramales = ["Todos"] + sorted(
                [str(r) for r in df_srv["id_ramal"].dropna().unique().tolist()]
            )
            ramal_sel = st.selectbox(
                "Ramal", ramales, index=0,
                help="Ramal específico dentro de la línea, o 'Todos'.",
            )
        with col_i:
            internos = ["Todos"] + sorted(
                df_srv["interno"].dropna().astype(str).unique().tolist()
            )
            interno_sel = st.selectbox(
                "Interno", internos, index=0,
                help="Vehículo (coche) específico, o 'Todos'.",
            )

        df_srv_f = df_srv.copy()
        if ramal_sel != "Todos":
            df_srv_f = df_srv_f[df_srv_f["id_ramal"].astype(str) == ramal_sel]
        if interno_sel != "Todos":
            df_srv_f = df_srv_f[df_srv_f["interno"].astype(str) == interno_sel]

        cols_show = [
            "nombre_linea", "dia", "id_ramal", "interno", "service_id",
            "hora_inicio", "hora_fin",
            "tot_km", "tot_km_gps", "tot_pax",
        ]
        cols_show += cols_for_distances("dmt_mean", ["od", "route", "route_gps"])
        cols_show += ["ipk_route", "ipk_route_gps"]
        cols_show += cols_for_distances("fo_mean", ["od", "route", "route_gps"])
        cols_show = [c for c in cols_show if c in df_srv_f.columns]

        st.markdown(f"**{len(df_srv_f):,} servicios**")
        show_table(
            df_srv_f[cols_show].sort_values(
                ["nombre_linea", "dia", "id_ramal", "interno", "service_id"]
            ),
            "kpi_by_day_line_service",
        )

        fo_col = next(
            (c for c in cols_for_distances("fo_mean", ["route", "od", "route_gps"])
             if c in df_srv_f.columns),
            None,
        )
        if fo_col and df_srv_f[fo_col].notna().any():
            st.markdown(f"#### Distribución de {label_of(fo_col)}")
            fig_fo = px.histogram(
                df_srv_f, x=fo_col, nbins=40,
                labels={fo_col: label_of(fo_col)},
            )
            st.plotly_chart(fig_fo, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Servicios despachados por hora
# -----------------------------------------------------------------------------

with st.expander("Servicios despachados por hora", expanded=False):
    df_sh = filter_kpi(services_hr)
    if df_sh.empty:
        st.info("Sin datos.")
    else:
        fig_sh = px.bar(
            df_sh.sort_values(["nombre_linea", "hora"]),
            x="hora", y="servicios", color="nombre_linea",
            barmode="group",
            labels={"hora": "Hora del día",
                    "servicios": label_of("servicios"),
                    "nombre_linea": "Línea"},
        )
        st.plotly_chart(fig_sh, use_container_width=True)
        show_table(
            df_sh[["nombre_linea", "dia", "hora", "servicios"]].sort_values(
                ["nombre_linea", "dia", "hora"]
            ),
            "services_by_line_hour",
        )