import sys
import os

# Propagate --config flag to all dashboard modules via env var.
# Usage: streamlit run dashboard.py -- --config /path/to/configuraciones_generales.yaml
_argv = sys.argv[1:]
if "--config" in _argv:
    _idx = _argv.index("--config")
    if _idx + 1 < len(_argv):
        os.environ["URBANTRIPS_CONFIG"] = str(_argv[_idx + 1])

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mapclassify
import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import requests
from PIL import Image
from shapely import wkt
import yaml
import sqlite3
from shapely import wkt
from folium import Figure
from shapely.geometry import LineString

from dash_utils import (
    levanto_tabla_sql,
    get_logo,
    traigo_indicadores,
    configurar_selector_dia,
    formatear_columnas_numericas
)





st.set_page_config(layout="wide")

st.sidebar.success("Seleccione página")

logo = get_logo()
st.image(logo)


st.markdown(
    '<div style="text-align: justify;">urbantrips es una biblioteca de código abierto que toma información de un sistema de pago con tarjeta inteligente de transporte público y, a través de un procesamiento de la información que infiere destinos de los viajes y construye las cadenas de viaje para cada usuario, produce matrices de origen-destino y otros indicadores (KPI) para rutas de autobús. El principal objetivo de la librería es producir insumos útiles para la gestión del transporte público a partir de requerimientos mínimos de información y pre-procesamiento. Con sólo una tabla geolocalizada de transacciones económicas proveniente de un sistema de pago electrónico, se podrán generar resultados, que serán más precisos cuanto más información adicional se incorpore al proceso a través de los archivos opcionales. El proceso elabora las matrices, los indicadores y construye una serie de gráficos y mapas de transporte.</div>',
    unsafe_allow_html=True,
)
st.text("")

alias_seleccionado = configurar_selector_dia()


col1, col2, col3 = st.columns([1, 3, 3])

indicadores = levanto_tabla_sql("indicadores", "data")


def _valor(df, tabla, detalle_contains):
    """Devuelve el valor numérico de un indicador puntual, o None si no existe."""
    sub = df[(df.tabla == tabla) & (df.detalle.str.contains(detalle_contains, regex=False))]
    return float(sub.indicador.iloc[0]) if len(sub) else None


def _tabla_indicadores(df, columna_pct=None, entero=True):
    """Arma un DataFrame listo para mostrar: 'Indicador' + 'Valor' (+ '%').

    - columna_pct: si se pasa una Serie/columna de porcentajes, se agrega la
      columna '%' formateada (vacía donde el valor es nulo o 0).
    - entero: formatea 'Valor' como entero (volúmenes) o con 2 decimales.
    """
    out = df[["detalle", "indicador"]].copy()
    out = formatear_columnas_numericas(out, ["indicador"], entero)
    out = out.rename(columns={"detalle": "Indicador", "indicador": "Valor"})
    if columna_pct is not None:
        pct = pd.Series(columna_pct).reset_index(drop=True)
        out = out.reset_index(drop=True)
        out["%"] = pct.apply(
            lambda x: f"{x:.1f}".replace(".", ",") + "%"
            if pd.notna(x) and round(float(x), 1) != 0.0
            else ""
        )
    return out


def _mostrar(col, titulo, df, ayuda=None, **kwargs):
    col.markdown(f"**{titulo}**")
    if ayuda:
        col.caption(ayuda)
    if df is None or len(df) == 0:
        col.caption("— sin datos —")
        return
    col.dataframe(_tabla_indicadores(df, **kwargs), hide_index=True, use_container_width=True)


if len(indicadores) > 0:
    desc_dia_i = col1.selectbox(
        "Dia", options=indicadores.dia.unique(), key="desc_dia_i"
    )

    indicadores = indicadores[(indicadores.dia == desc_dia_i)].copy()

    # ── Valores de referencia para porcentajes calculados ────────────────────
    registros_trx = _valor(indicadores, "transacciones", "Registros")
    total_viajes_exp = _valor(indicadores, "viajes expandidos", "Cantidad total de viajes")

    # ╔═══════════════════════════ Columna izquierda: VOLÚMENES (embudo) ══════╗

    # 1) Transacciones (insumo)
    trx = indicadores.loc[indicadores.tabla == "transacciones"]
    _mostrar(
        col2, "Preprocesamiento de transacciones", trx,
        ayuda="Insumo crudo: cada transacción del sistema de pago es una etapa potencial.",
    )

    # 2) Etapas: validadas (con destino imputado) + expandidas al universo
    etapas = pd.concat([
        indicadores[indicadores.tabla == "etapas"],
        indicadores[indicadores.tabla == "etapas_expandidas"].sort_values("nivel"),
    ])
    pct_etapas = etapas.apply(
        lambda r: (r.indicador / registros_trx * 100)
        if (r.tabla == "etapas" and registros_trx) else r.porcentaje,
        axis=1,
    )
    _mostrar(
        col2, "Etapas", etapas, columna_pct=pct_etapas,
        ayuda="Etapas con destino validado se expanden al total de transacciones. "
              "El % de las validadas es la tasa de validación; el resto, partición modal.",
    )

    # 3) Viajes: validados + expandidos (con % de transferencia y cortos)
    viajes = pd.concat([
        indicadores[indicadores.tabla == "viajes"],
        indicadores[indicadores.tabla == "viajes expandidos"].sort_values("nivel"),
    ])
    _mostrar(
        col2, "Viajes", viajes, columna_pct=viajes["porcentaje"],
        ayuda="Etapas encadenadas en viajes. Los % son sobre el total de viajes expandidos.",
    )

    # 4) Usuarios / tarjetas
    usuarios = indicadores[indicadores.tabla.isin(["usuarios", "usuarios expandidos"])]
    _mostrar(
        col2, "Usuarios (tarjetas)", usuarios,
        ayuda="Tarjetas únicas. 'finales' usa el factor de expansión por tarjeta; "
              "'total' el factor por etapa.",
    )

    # ╔═══════════════════════════ Columna derecha: PROMEDIOS y MODAL ═════════╗

    # 5) Partición modal de viajes (con % calculado sobre el total expandido)
    modal = indicadores[indicadores.tabla == "modos viajes"].copy()
    pct_modal = (
        modal.indicador / total_viajes_exp * 100 if total_viajes_exp else None
    )
    _mostrar(
        col3, "Partición modal de viajes", modal, columna_pct=pct_modal,
        ayuda="Viajes expandidos por modo y su participación sobre el total.",
    )

    # 6) Promedios
    prom = indicadores[(indicadores.tabla == "avg") & (indicadores.detalle.str.contains("promedio"))]
    _mostrar(
        col3, "Promedios", prom, entero=False,
        ayuda="Distancias y etapas promedio (ponderadas por factor de expansión).",
    )

    # 7) Medianas
    med = indicadores[(indicadores.tabla == "avg") & (indicadores.detalle.str.contains("mediana"))]
    _mostrar(col3, "Medianas", med, entero=False)

