# -*- coding: utf-8 -*-
"""
Dashboard de KPIs de transporte por línea
========================================
Aplicación Streamlit que permite: 
- Filtrar por día y línea.
- Visualizar indicadores agregados del sistema o por modo.
- Descargar el subconjunto de datos en CSV.

Dependencias:
    streamlit, pandas, plotly
    urbantrips.utils.utils (levanto_tabla_sql)

Ejecución:
    streamlit run dashboard_kpis.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from urbantrips.utils.utils import levanto_tabla_sql

# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard KPIs Transporte",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Constantes y utilidades
# -----------------------------------------------------------------------------
LABELS = {
    "dia": "Día",
    "id_linea": "ID Línea",
    "nombre_linea": "Nombre de línea",
    "empresa": "Empresa",
    "modo": "Modo",
    "transacciones": "Transacciones",
    "tot_pax": "Pasajeros",
    "tot_veh": "Vehículos en servicio",
    "tot_km": "Kilómetros recorridos",
    "travel_speed": "Velocidad comercial (km/h)",
    "travel_time_min": "Tiempo medio de viaje (min)",
    "tarifa_social": "Tarifa social (pasajeros)",
    "educacion_jubilacion": "Educ./Jubil. (pasajeros)",
    "ipk": "IPK (ingresos por km)",
    "fo_mean": "Frecuencia media (min)",
    "fo_median": "Frecuencia mediana (min)",
    "dmt_mean": "Dist. media trayecto (km)",
    "dmt_median": "Dist. mediana trayecto (km)",
}

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Carga los KPIs desde la base SQLite usando el helper existente."""
    alias_data="amba_2024_9_18"
    df = levanto_tabla_sql("kpis_lineas", tabla_tipo="dash", alias_db=alias_data)
    st.write(df)
    # Asegurar que la columna día sea tipo fecha (sin hora) para los filtros
    df["dia"] = pd.to_datetime(df["dia"]).dt.date
    return df

# -----------------------------------------------------------------------------
# Carga inicial de datos
# -----------------------------------------------------------------------------
try:
    df_full = load_data()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Barra lateral: filtros
# -----------------------------------------------------------------------------
st.sidebar.header("Filtros")

# Filtro por día
days = sorted(df_full["dia"].unique())
selected_day = st.sidebar.selectbox("Día", options=["Todos"] + days, index=0)

# Filtro por modo (detecta automáticamente)
unique_modes = sorted(df_full["modo"].unique())
selected_modes = st.sidebar.multiselect("Modo", options=unique_modes, default=unique_modes)

# Filtro por línea (muestra ID y nombre juntos para facilitar búsqueda)
line_options = (
    df_full[["id_linea", "nombre_linea"]]
    .drop_duplicates()
    .sort_values("id_linea")
)
line_options["label"] = line_options.apply(
    lambda x: f"{x.id_linea} - {x.nombre_linea}", axis=1
)
line_dict = dict(zip(line_options["label"], line_options["id_linea"]))
selected_lines = st.sidebar.multiselect(
    "Líneas", options=line_options["label"], default=line_options["label"]
)
selected_line_ids = [line_dict[label] for label in selected_lines]

# -----------------------------------------------------------------------------
# Aplicar filtros al DataFrame
# -----------------------------------------------------------------------------
df_filtered = df_full.copy()

if selected_day != "Todos":
    df_filtered = df_filtered[df_filtered["dia"] == selected_day]

if selected_modes:
    df_filtered = df_filtered[df_filtered["modo"].isin(selected_modes)]

if selected_line_ids:
    df_filtered = df_filtered[df_filtered["id_linea"].isin(selected_line_ids)]

# -----------------------------------------------------------------------------
# KPI globales
# -----------------------------------------------------------------------------

def show_kpis(df: pd.DataFrame) -> None:
    """Muestra KPIs agregados a nivel sistema o del subconjunto filtrado."""
    if df.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    tot_trans = int(df["transacciones"].sum())
    tot_pax = int(df["tot_pax"].sum())
    tot_veh = int(df["tot_veh"].sum())
    tot_km = float(df["tot_km"].sum())
    avg_speed = df["travel_speed"].mean()
    pct_tarifa_social = (
        df["tarifa_social"].sum() / tot_trans if tot_trans else 0
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Transacciones", f"{tot_trans:,}")
    col2.metric("Pasajeros", f"{tot_pax:,}")
    col3.metric("Vehículos", f"{tot_veh:,}")
    col4.metric("Km recorridos", f"{tot_km:,.1f}")
    col5.metric("Vel. comercial (km/h)", f"{avg_speed:,.2f}")
    col6.metric("% Tarifa social", f"{pct_tarifa_social:.1%}")

show_kpis(df_filtered)

# -----------------------------------------------------------------------------
# Gráficos por modo
# -----------------------------------------------------------------------------

st.subheader("Distribución de transacciones por modo")
if not df_filtered.empty:
    by_mode = (
        df_filtered.groupby("modo", as_index=False)["transacciones"].sum()
        .sort_values("transacciones", ascending=False)
    )
    fig = px.bar(
        by_mode,
        x="modo",
        y="transacciones",
        labels={"modo": "Modo", "transacciones": "Transacciones"},
        title="Transacciones por modo",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay datos para graficar.")

# -----------------------------------------------------------------------------
# Tabla de detalle
# -----------------------------------------------------------------------------

st.subheader("Detalle por línea")

if not df_filtered.empty:
    # Renombrar columnas para la tabla de usuario (sin alterar cálculos)
    df_display = df_filtered.rename(columns=LABELS)
    # Orden de columnas amigable
    ordered_cols = [
        "Día",
        "ID Línea",
        "Nombre de línea",
        "Modo",
        "Empresa",
        "Transacciones",
        "Pasajeros",
        "Vehículos en servicio",
        "Kilómetros recorridos",
        "Velocidad comercial (km/h)",
        "Tiempo medio de viaje (min)",
        "% Tarifa social",
        "IPK (ingresos por km)",
    ]
    # Añadir % Tarifa social como columna calculada para la vista
    df_display["% Tarifa social"] = (
        df_filtered["tarifa_social"] / df_filtered["transacciones"]
    ).round(3)

    st.dataframe(
        df_display[ordered_cols],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Seleccione al menos un dato para visualizar la tabla.")

# -----------------------------------------------------------------------------
# Descarga de CSV
# -----------------------------------------------------------------------------

st.subheader("Descargar datos")

csv_data = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar CSV",
    data=csv_data,
    file_name="kpis_filtrado.csv",
    mime="text/csv",
)

# -----------------------------------------------------------------------------
# Pie de página
# -----------------------------------------------------------------------------

st.caption(
    "Fuente de datos: kpis_lineas (SQLite). Última actualización automática al momento de recarga del dashboard."
)
