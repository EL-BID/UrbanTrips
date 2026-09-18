# -*- coding: utf-8 -*-
"""Agregación de indicadores de línea-día a totales de sistema.

Esta lógica vivía dentro de `dashboard/pages/9_Indicadores Operativos.py`, que
no es importable (ejecuta `st.*` a nivel de módulo) y por lo tanto no podía
cubrirse con tests. Acá quedan las funciones puras —sin Streamlit y sin acceso a
la base— para que la suite las verifique y para que cualquier otra vista use el
mismo criterio.

Las dos reglas que justifican el módulo:

1. **Las columnas extensivas se suman; las intensivas no.** Para un ratio
   `R = A/B` vale `Σ(Rᵢ·Bᵢ)/ΣBᵢ = ΣAᵢ/ΣBᵢ`, o sea que el promedio ponderado
   **por el denominador** ES el ratio agregado, con la ventaja de que las líneas
   sin ese dato entran con peso 0 en lugar de contaminar el resultado.

   No es cosmético: recomputar un ratio "a mano" sobre el total mezclando modos
   da números falsos, porque subte y tren aportan transacciones pero casi ningún
   km (no reportan GPS). Medido sobre AMBA 2026-05-14:

       IPK  Σ(trx)/Σ(km), todos los modos = 3,268   <- inflado 27 %
       IPK  ponderado por km              = 2,567   <- correcto
       IPK propio del tren = 505,98  |  del metro = inf

2. **"Promedio del período" para lo extensivo es el promedio de los totales
   diarios**, no la media de las filas línea-día (que sería el promedio por
   línea, un número sin sentido como total de sistema).
"""

from __future__ import annotations

import pandas as pd

# --- columnas extensivas: se suman entre líneas -----------------------------
GENDER_COLS = ["Masculino", "Femenino", "No informado"]
TARIFA_COLS = ["sin_descuento", "tarifa_social", "educacion_jubilacion"]
DEMO_COLS = GENDER_COLS + TARIFA_COLS

SUM_COLS = [
    "vehiculos_operativos",
    "cant_internos_en_trx",
    "transacciones",
    "tot_km_route",
    "tot_km_route_gps",
] + DEMO_COLS

# Subconjunto acumulable sobre todo el período. Los conteos de vehículos quedan
# fuera a propósito: sumarlos entre días da vehículo-día, no vehículos (un
# interno que operó 28 días contaría 28 veces).
PERIOD_SUM_COLS = ["transacciones", "tot_km_route", "tot_km_route_gps"] + DEMO_COLS

# --- columnas intensivas: (peso, modo de agregación) ------------------------
# Oferta ponderada por oferta —km de su propia familia de distancia, o
# vehículos—, nunca por demanda; demanda por transacciones.
#
# "harmonic" es para km/h: solo tenemos los km, pero las horas de cada línea se
# recuperan como km/velocidad, así que Σkm / Σ(km/vᵢ) = Σkm / Σhoras. Ponderar
# linealmente por km daría 17,56 km/h contra los 16,41 reales: sesga hacia las
# líneas rápidas.
RATIO_AGG = {
    "velocidad_comercial_route": ("tot_km_route", "harmonic"),
    "velocidad_comercial_route_gps": ("tot_km_route_gps", "harmonic"),
    "distancia_media_veh_route": ("vehiculos_operativos", "wmean"),
    "distancia_media_veh_route_gps": ("vehiculos_operativos", "wmean"),
    "ipk_route": ("tot_km_route", "wmean"),
    "ipk_route_gps": ("tot_km_route_gps", "wmean"),
    "fo_mean_route": ("tot_km_route", "wmean"),
    "fo_median_route": ("tot_km_route", "wmean"),
    "fo_mean_route_gps": ("tot_km_route_gps", "wmean"),
    "fo_median_route_gps": ("tot_km_route_gps", "wmean"),
    "kvd_route": ("vehiculos_operativos", "wmean"),
    "kvd_route_gps": ("vehiculos_operativos", "wmean"),
    "pvd": ("vehiculos_operativos", "wmean"),
    # la familia _od no tiene km de oferta asociados: su factor de ocupación
    # también se pondera por transacciones
    "fo_mean_od": ("transacciones", "wmean"),
    "fo_median_od": ("transacciones", "wmean"),
    "dmt_mean_od": ("transacciones", "wmean"),
    "dmt_median_od": ("transacciones", "wmean"),
    "dmt_mean_route": ("transacciones", "wmean"),
    "dmt_median_route": ("transacciones", "wmean"),
    "dmt_mean_route_gps": ("transacciones", "wmean"),
    "dmt_median_route_gps": ("transacciones", "wmean"),
    "distancia_media_pax": ("transacciones", "wmean"),
    "travel_time_min": ("transacciones", "wmean"),
    "kmh_od": ("transacciones", "wmean"),
}

TIPOS_DIA = ["Todos", "Día hábil", "Sábado", "Domingo"]
# (singular, plural), para los títulos
NOMBRE_DIAS = {
    "Todos": ("día", "días"),
    "Día hábil": ("día hábil", "días hábiles"),
    "Sábado": ("sábado", "sábados"),
    "Domingo": ("domingo", "domingos"),
}
_DOW_A_TIPO = {
    0: "Día hábil", 1: "Día hábil", 2: "Día hábil", 3: "Día hábil",
    4: "Día hábil", 5: "Sábado", 6: "Domingo",
}
DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes",
               "sábado", "domingo"]


def tipo_de_dia(dias: pd.Series) -> pd.Series:
    """Tipo de día a partir de la fecha. Lo que no es fecha queda en ""
    (por ejemplo la fila de resumen `dia = "Promedios"`)."""
    dow = pd.to_datetime(dias, errors="coerce").dt.dayofweek
    return dow.map(_DOW_A_TIPO).fillna("")


def etiqueta_dia(dia: str) -> str:
    """'2026-03-14' -> '2026-03-14 (sábado)'. Si no es fecha, la deja igual."""
    ts = pd.to_datetime(dia, errors="coerce")
    if pd.isna(ts):
        return str(dia)
    return f"{dia} ({DIAS_SEMANA[ts.dayofweek]})"


def _wmean(d: pd.DataFrame, col: str, peso: str) -> float:
    """Promedio de `col` ponderado por `peso`, ignorando filas sin ninguno."""
    if col not in d.columns or peso not in d.columns:
        return float("nan")
    m = d[col].notna() & d[peso].notna() & (d[peso] > 0)
    w = d.loc[m, peso]
    return (d.loc[m, col] * w).sum() / w.sum() if w.sum() else float("nan")


def _harmonic(d: pd.DataFrame, col: str, peso: str) -> float:
    """Σpeso / Σ(peso/col). Con `peso` en km y `col` en km/h equivale a
    Σkm / Σhoras, la velocidad media real del conjunto."""
    if col not in d.columns or peso not in d.columns:
        return float("nan")
    m = d[col].notna() & d[peso].notna() & (d[peso] > 0) & (d[col] > 0)
    if not m.any():
        return float("nan")
    return d.loc[m, peso].sum() / (d.loc[m, peso] / d.loc[m, col]).sum()


def agregar_sistema(d: pd.DataFrame, *, promedio_diario: bool) -> pd.DataFrame:
    """Colapsa un df de línea-día a una fila con los indicadores del conjunto.

    Parameters
    ----------
    d : DataFrame
        Filas de `kpis_lineas` (o `kpis_ramales`) ya filtradas. **No debe
        incluir la fila de resumen `dia = "Promedios"`**: es un promedio, no un
        día, y sumarla duplica los totales.
    promedio_diario : bool
        True devuelve, para las columnas extensivas, el promedio de los totales
        diarios; False, la suma del período. Las intensivas se agregan siempre
        ponderadas sobre las filas recibidas.
    """
    out: dict = {}

    for col in SUM_COLS:
        if col not in d.columns:
            continue
        if d[col].notna().sum() == 0:
            # ninguna línea tiene el dato: sum() daría 0, que se lee como "no
            # operó" en vez de "no hay dato"
            out[col] = float("nan")
            continue
        por_dia = d.groupby("dia")[col].sum()
        out[col] = por_dia.mean() if promedio_diario else por_dia.sum()

    for col, (peso, modo_agg) in RATIO_AGG.items():
        if col not in d.columns:
            continue
        out[col] = (
            _harmonic(d, col, peso) if modo_agg == "harmonic" else _wmean(d, col, peso)
        )

    # cobertura: cociente de sumas, no suma de cocientes
    gps, trx = out.get("vehiculos_operativos"), out.get("cant_internos_en_trx")
    out["cobertura_gps"] = gps / trx if (pd.notna(gps) and trx) else float("nan")
    return pd.DataFrame([out])


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas derivadas comunes a `kpis_lineas` y `kpis_ramales`."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # Cobertura GPS: qué proporción de los vehículos que registraron
    # transacciones llegó a tener un servicio GPS válido. Es calidad del dato,
    # no operación: subte y tren no reportan GPS y ahí da NaN aunque el modo
    # haya operado con normalidad.
    df["cobertura_gps"] = df["vehiculos_operativos"] / df[
        "cant_internos_en_trx"
    ].replace(0, pd.NA)

    # `tot_km_route_gps` se persiste desde 2026-08-20; para bases anteriores se
    # reconstruye como distancia media por vehículo × vehículos. Validado contra
    # la familia _route, donde sí estaba el total real: -0,001 % de error.
    if (
        "tot_km_route_gps" not in df.columns
        and "distancia_media_veh_route_gps" in df.columns
    ):
        df["tot_km_route_gps"] = (
            df["distancia_media_veh_route_gps"] * df["vehiculos_operativos"]
        )

    df["tipo_dia"] = tipo_de_dia(df["dia"])
    return df
