# -*- coding: utf-8 -*-
"""
Dashboard de KPIs – Indicadores Operativos
=========================================
Panel Streamlit que muestra:

1. **KPIs por línea** (filtros de línea y día).
2. **Totales y promedios del sistema** con filtro de modo.
3. **Base completa** con descarga CSV.

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
from dash_utils import levanto_tabla_sql, get_logo, configurar_selector_dia

# -----------------------------------------------------------------------------
# Configuración global y estilo
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Indicadores Operativos por Línea", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid=\"metric-container\"] > div:first-child {font-size:0.75rem;}
    div[data-testid=\"metric-container\"] > div:nth-child(2) {font-size:1.05rem;}
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
# Carga de datos
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_kpis() -> pd.DataFrame:
    df = levanto_tabla_sql("kpis_lineas", "general")
    df["vehiculos_operativos"] = (
        df["cant_internos_en_gps"].where(df["cant_internos_en_gps"] > 0, df["cant_internos_en_trx"])
        .fillna(df["flota"])
        .astype("Int64")
    )
    return df
    

kpis_df = load_kpis()

# -----------------------------------------------------------------------------
# Etiquetas y grupos
# -----------------------------------------------------------------------------

LABELS = {
    "vehiculos_operativos": "Vehículos operativos",
    "transacciones": "Transacciones",
    "Masculino": "Masculino",
    "Femenino": "Femenino",
    "No informado": "No informado",
    "sin_descuento": "Sin descuento",
    "tarifa_social": "Tarifa social",
    "educacion_jubilacion": "Estudiantes/Jubilados",
    "travel_speed": "Vel. comercial (km/h)",
    "distancia_media_veh": "Dist. media/veh (km)",
    "fo_mean": "Factor Ocupación (media)",
    "fo_median": "Factor Ocupación (mediana)",
    "travel_time_min": "Tiempo promedio viaje (min)",
    "tot_km": "Km recorridos",
    "dmt_mean": "Distancia media Pax (km)",
    "dmt_median": "Distancia mediana Pax (km)",
    "ipk": "IPK",
}

GENERAL_COLS = ["vehiculos_operativos", "transacciones"]
GENDER_COLS = ["Masculino", "Femenino", "No informado"]
TARIFA_COLS = ["sin_descuento", "tarifa_social", "educacion_jubilacion"]
DEMO_COLS = GENDER_COLS + TARIFA_COLS
OPERATIVE_COLS1 = ["travel_speed", "distancia_media_veh", "fo_mean", "fo_median"]
OPERATIVE_COLS2 = ["travel_time_min", "tot_km", "dmt_mean", "dmt_median", "ipk"]
INT_DISPLAY_COLS = set(GENERAL_COLS + DEMO_COLS + ["tot_km"])
TOTAL_SLOTS = 6  # columnas fijas por fila

# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------

def fmt(val, col):
    if pd.isna(val):
        return "–"
    if col in INT_DISPLAY_COLS or (isinstance(val, (int, float)) and float(val).is_integer()):
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


def weighted_means(df: pd.DataFrame) -> pd.Series:
    w = df["transacciones"]
    wtot = w.sum()
    num_cols = df.select_dtypes("number").columns.difference(["transacciones"])
    return pd.Series({c: (df[c] * w).sum() / wtot if wtot else float("nan") for c in num_cols})

# -----------------------------------------------------------------------------
# 1. KPIs por línea
# -----------------------------------------------------------------------------

with st.expander("KPIs por línea", expanded=True):
    col_filt, col_met = st.columns([2, 10])

    with col_filt:

        lines = kpis_df[["id_linea", "nombre_linea"]].drop_duplicates().sort_values("id_linea")
        lines["label"] = lines.apply(lambda x: f"{x.id_linea} – {x.nombre_linea}", axis=1)
        line_label = st.selectbox("Línea", lines["label"], index=0)
        line_id = lines.set_index("label").loc[line_label, "id_linea"]
        days = sorted(kpis_df["dia"].dropna().unique())
        day_sel = st.selectbox("Día", ["Todos"] + days, index=0)

    with col_met:
        df_line = kpis_df[kpis_df["id_linea"] == line_id]
        if day_sel != "Todos":
            df_line = df_line[df_line["dia"] == day_sel]
        if df_line.empty:
            st.warning("Sin datos para los filtros seleccionados.")
        else:
            st.markdown("#### Generales")
            metric_row(df_line, GENERAL_COLS)
            st.divider()

            st.markdown("#### Género y tipo de tarifa")
            metric_row(df_line, DEMO_COLS, pct=True)
            st.divider()

            st.markdown("#### Operativos")
            metric_row(df_line, OPERATIVE_COLS1)
            metric_row(df_line, OPERATIVE_COLS2)

# -----------------------------------------------------------------------------
# 2. Totales y promedios (filtro modo)
# -----------------------------------------------------------------------------

with st.expander("Totales y promedios del sistema", expanded=False):
    col_mode, col_out = st.columns([2, 10])

    with col_mode:
        mode_options = ["Todos"] + sorted(kpis_df["modo"].dropna().unique())
        mode_sel = st.selectbox("Modo", mode_options, index=0)

    with col_out:
        df_sel = kpis_df if mode_sel == "Todos" else kpis_df[kpis_df["modo"] == mode_sel]

        # Totales
        st.markdown("### Totales")
        tot_df = pd.DataFrame({
            "vehiculos_operativos": [df_sel["vehiculos_operativos"].sum()],
            "transacciones": [df_sel["transacciones"].sum()],
            "tot_km": [df_sel["tot_km"].sum()],
        })
        metric_row(tot_df, GENERAL_COLS + ["tot_km"])
        st.divider()
        demo_tot = df_sel[DEMO_COLS].sum().to_frame().T
        demo_tot["transacciones"] = tot_df["transacciones"].iloc[0]
        metric_row(demo_tot, DEMO_COLS, pct=True)
        st.divider()

        # Promedios ponderados
        st.markdown("### Promedios ponderados por transacciones")
        gen_avg = pd.DataFrame({
            "vehiculos_operativos": [df_sel["vehiculos_operativos"].mean()],
            "transacciones": [df_sel["transacciones"].mean()],
        })
        metric_row(gen_avg, GENERAL_COLS)
        st.divider()
        demo_wp = df_sel[DEMO_COLS].sum().to_frame().T
        demo_wp["transacciones"] = df_sel["transacciones"].sum()
        metric_row(demo_wp, DEMO_COLS, pct=True)
        st.divider()
        op_wp = weighted_means(df_sel).to_frame().T
        metric_row(op_wp, OPERATIVE_COLS1)
        metric_row(op_wp, OPERATIVE_COLS2)

# -----------------------------------------------------------------------------
# 3. Base completa
# -----------------------------------------------------------------------------

with st.expander("Base completa", expanded=False):
    st.dataframe(kpis_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Descargar CSV completo",
        data=kpis_df.to_csv(index=False).encode("utf-8"),
        file_name="kpis_lineas_completo.csv",
        mime="text/csv",
    )
