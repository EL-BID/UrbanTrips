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

from urbantrips.kpi.cobertura import lineas_dia_sin_oferta, solo_dias_con_oferta

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
    "tot_km_route_estimado",
] + DEMO_COLS

# Subconjunto acumulable sobre todo el período. Los conteos de vehículos quedan
# fuera a propósito: sumarlos entre días da vehículo-día, no vehículos (un
# interno que operó 28 días contaría 28 veces).
PERIOD_SUM_COLS = [
    "transacciones", "tot_km_route", "tot_km_route_gps", "tot_km_route_estimado",
] + DEMO_COLS

# Pesos derivados que arma `preparar`
LUGARES_KM_ROUTE = "_lugares_km_route"
LUGARES_KM_ROUTE_GPS = "_lugares_km_route_gps"

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
    # El FO es pasajeros-km / lugares-km: entre modos de capacidad distinta se
    # pondera por lugares-km (km x capacidad), no solo por km (ver `preparar`)
    "fo_mean_route": (LUGARES_KM_ROUTE, "wmean"),
    "fo_median_route": (LUGARES_KM_ROUTE, "wmean"),
    "fo_mean_route_gps": (LUGARES_KM_ROUTE_GPS, "wmean"),
    "fo_median_route_gps": (LUGARES_KM_ROUTE_GPS, "wmean"),
    "kvd_route": ("vehiculos_operativos", "wmean"),
    "kvd_route_gps": ("vehiculos_operativos", "wmean"),
    "pvd": ("vehiculos_operativos", "wmean"),
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

# Marca, fila a fila, si `cobertura_gps` es la proporción de etapas dentro de
# servicios válidos (bases nuevas) o la medida anterior por vehículos.
COBERTURA_POR_ETAPAS = "_cobertura_por_etapas"

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


def media_por_grupo(d, col, peso, grupos, armonica=False):
    """Media de `col` ponderada por `peso` en cada grupo de `grupos` (lista de
    series alineadas con `d`), ignorando las filas sin valor o sin peso.

    Con `armonica` es Σpeso / Σ(peso/col): con `peso` en km y `col` en km/h
    equivale a Σkm / Σhoras, la velocidad media real del conjunto. Devuelve una
    serie indexada por grupo; vacía si falta `col` o `peso`, y NaN en los
    grupos sin ninguna fila utilizable.
    """
    if col not in d.columns or peso not in d.columns:
        return pd.Series(dtype="float64")
    m = d[col].notna() & d[peso].notna() & (d[peso] > 0)
    if armonica:
        m &= d[col] > 0
    w = d[peso].where(m)
    num, den = (w, w / d[col]) if armonica else (d[col] * w, w)
    num = num.groupby(grupos).sum(min_count=1)
    den = den.groupby(grupos).sum(min_count=1)
    return num / den


def _un_grupo(d, col, peso, armonica=False) -> float:
    """`media_por_grupo` sobre todo `d` como un solo grupo."""
    m = media_por_grupo(d, col, peso, [pd.Series(0, index=d.index)], armonica)
    return float(m.iloc[0]) if len(m) else float("nan")


def _wmean(d: pd.DataFrame, col: str, peso: str) -> float:
    """Promedio de `col` ponderado por `peso`, ignorando filas sin ninguno."""
    return _un_grupo(d, col, peso)


def _harmonic(d: pd.DataFrame, col: str, peso: str) -> float:
    """Σpeso / Σ(peso/col): la velocidad media real del conjunto."""
    return _un_grupo(d, col, peso, armonica=True)


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
    if LUGARES_KM_ROUTE not in d.columns:
        d = d.copy()
        _agregar_lugares_km(d)

    for col in SUM_COLS:
        if col not in d.columns:
            continue
        if d[col].notna().sum() == 0:
            # ninguna línea tiene el dato: sum() daría 0, que se lee como "no
            # operó" en vez de "no hay dato"
            out[col] = float("nan")
            continue
        # min_count=1: un dia en que todas las lineas estan sin dato (p. ej. por
        # cobertura GPS bajo el umbral) es "sin dato", no un dia de 0 km
        por_dia = d.groupby("dia")[col].sum(min_count=1)
        out[col] = por_dia.mean() if promedio_diario else por_dia.sum()

    for col, (peso, modo_agg) in RATIO_AGG.items():
        if col not in d.columns:
            continue
        out[col] = (
            _harmonic(d, col, peso) if modo_agg == "harmonic" else _wmean(d, col, peso)
        )

    if mide_cobertura_por_etapas(d):
        # proporcion de etapas dentro de servicios validos: se agrega ponderando
        # por transacciones (equivale al cociente de sumas de etapas)
        out["cobertura_gps"] = _wmean(d, "cobertura_gps", "transacciones")
    else:
        # bases anteriores: vehiculos con servicio valido sobre internos con
        # transacciones. Cociente de sumas, no suma de cocientes.
        gps, trx = out.get("vehiculos_operativos"), out.get("cant_internos_en_trx")
        out["cobertura_gps"] = gps / trx if (pd.notna(gps) and trx) else float("nan")
    return pd.DataFrame([out])


def filas_promedios(df: pd.DataFrame, claves: list, decimales: int | None = None):
    """Fila `dia = "Promedios"` de cada entidad (`claves`), a partir de sus filas
    diarias.

    Es el mismo criterio que usa el dashboard para agregar (`agregar_sistema`):
    los totales son el promedio de los totales diarios y los cocientes salen de
    los totales del periodo (ponderados por su denominador), no de promediar los
    cocientes de cada dia. Se promedian los mismos dias en toda la fila: los
    dias en que la entidad quedo sin indicadores de oferta (cobertura GPS bajo el
    umbral) no entran, y una entidad que nunca tiene oferta (un modo sin GPS)
    promedia todos sus dias. Las columnas numericas que `agregar_sistema` no
    agrega (velocidades por franja horaria, internos en GPS, columnas externas
    que se hayan unido a la tabla) quedan como media simple de esos mismos dias.

    `df` puede traer filas `dia = "Promedios"` previas: se ignoran. Devuelve una
    fila por entidad, con `dia = "Promedios"`.
    """
    df = df[df["dia"] != "Promedios"]
    if df.empty:
        return pd.DataFrame(columns=list(claves) + ["dia", "mes"])
    df = solo_dias_con_oferta(df, claves)

    # Con una fila por entidad y dia, el total diario de una entidad es el valor
    # de la fila: promediar los totales diarios es la media simple.
    base = df.drop(columns=["dia", "mes"], errors="ignore").groupby(claves).mean(
        numeric_only=True
    )
    prep = preparar(df)
    grupos = [prep[c] for c in claves]
    for col, (peso, modo_agg) in RATIO_AGG.items():
        if col in base.columns:
            base[col] = media_por_grupo(prep, col, peso, grupos, modo_agg == "harmonic")
    if "cobertura_gps" in base.columns:
        # misma regla que `agregar_sistema`
        por_etapas = prep[COBERTURA_POR_ETAPAS].groupby(grupos).any()
        por_etapas_cob = media_por_grupo(prep, "cobertura_gps", "transacciones", grupos)
        por_vehiculos = base["vehiculos_operativos"] / base[
            "cant_internos_en_trx"
        ].replace(0, float("nan"))
        base["cobertura_gps"] = por_etapas_cob.reindex(base.index).where(
            por_etapas.reindex(base.index), por_vehiculos
        )
    if decimales is not None:
        # round() alcanza tambien a las columnas numericas anulables (Float64)
        base = base.round(decimales)
    base = base.reset_index()
    base["dia"] = "Promedios"
    base["mes"] = ""
    return base


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas derivadas comunes a `kpis_lineas` y `kpis_ramales`."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # Cobertura GPS: qué proporción de las etapas ocurre dentro de un servicio
    # GPS válido de su vehículo (ver kpi/cobertura.py). Es calidad del dato, no
    # operación. Las bases anteriores a esa columna no la traen: ahí se usa la
    # medida vieja, vehículos con servicio válido sobre internos con
    # transacciones (subte y tren no reportan GPS y da NaN).
    por_vehiculos = df["vehiculos_operativos"] / df["cant_internos_en_trx"].replace(
        0, pd.NA
    )
    if "cobertura_gps" in df.columns:
        por_etapas = pd.to_numeric(df["cobertura_gps"], errors="coerce")
        df[COBERTURA_POR_ETAPAS] = por_etapas.notna()
        df["cobertura_gps"] = por_etapas.fillna(pd.to_numeric(por_vehiculos, errors="coerce"))
    else:
        df[COBERTURA_POR_ETAPAS] = False
        df["cobertura_gps"] = por_vehiculos

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

    _agregar_lugares_km(df)
    df["tipo_dia"] = tipo_de_dia(df["dia"])
    return df


def _agregar_lugares_km(df: pd.DataFrame) -> None:
    """Agrega a `df` los lugares-km ofrecidos (km x capacidad), peso del factor
    de ocupacion. Con `capacidad` nula en todas las filas (base anterior a esa
    columna) el peso es el km, como antes; si falta solo en algunas filas, esas
    no tienen factor de ocupacion y no pesan."""
    tiene_capacidad = "capacidad" in df.columns and df["capacidad"].notna().any()
    for peso, km in ((LUGARES_KM_ROUTE, "tot_km_route"),
                     (LUGARES_KM_ROUTE_GPS, "tot_km_route_gps")):
        if km in df.columns:
            df[peso] = df[km] * df["capacidad"] if tiene_capacidad else df[km]


def mide_cobertura_por_etapas(df) -> bool:
    """True si alguna fila de `df` (ya pasado por `preparar`) trae la cobertura
    GPS medida con etapas; False en una base anterior a esa columna."""
    return COBERTURA_POR_ETAPAS in df.columns and bool(df[COBERTURA_POR_ETAPAS].any())


def nota_cobertura(df) -> str:
    """Texto que explica la cobertura GPS de las líneas-día de `df` y avisa
    cuántas quedaron sin indicadores de oferta por tener poca cobertura.
    Vacío si `df` no trae la cobertura medida con etapas.

    Las líneas-día sin oferta se cuentan de los datos (cobertura medida y km
    vacíos), no de un umbral: el umbral con el que se calcularon es el de
    tuning.yaml *de ese momento*, que pudo cambiar después.
    """
    if not mide_cobertura_por_etapas(df):
        return ""
    nota = (
        "La **cobertura GPS** es la parte de las etapas que ocurre dentro "
        "de un servicio GPS válido de su vehículo. Pasajeros por km, "
        "factor de ocupación y pasajeros por vehículo comparan los "
        "pasajeros y la oferta de esos mismos servicios. Los **km "
        "estimados** son los km recorridos divididos por la cobertura: "
        "suponen que la operación sin GPS se parece a la que sí tiene."
    )
    vacias = int((df[COBERTURA_POR_ETAPAS] & lineas_dia_sin_oferta(df)).sum())
    if vacias:
        nota += (
            f"  \n⚠️ {vacias} línea(s)-día de esta selección tienen "
            "cobertura menor al mínimo configurado "
            "(`kpi_cobertura_gps_minima`): sus indicadores "
            "de oferta no se calculan y no entran en los valores de arriba."
        )
    return nota


def nota_capacidad_modos(modos, capacidad_modos) -> str:
    """Texto que dice con que capacidad por vehiculo se calculo el factor de
    ocupacion de los `modos` que se estan mirando.

    `capacidad_modos` es el bloque del config de la corrida (modo -> capacidad).
    Avisa de los modos sin capacidad cargada (sus lineas no tienen factor de
    ocupacion) y de las corridas cuyo config no trae el bloque.
    """
    if not isinstance(capacidad_modos, dict):
        return (
            "⚠️ La configuración de esta corrida no tiene `capacidad_modos`: el "
            "factor de ocupación no se puede interpretar con una capacidad conocida."
        )
    capacidades = pd.to_numeric(pd.Series(capacidad_modos), errors="coerce")
    capacidades.index = capacidades.index.astype(str).str.lower()
    modos = sorted({str(m).lower() for m in modos if pd.notna(m)})
    con = [m for m in modos if capacidades.get(m, 0) > 0]
    sin = [m for m in modos if m not in con]

    partes = []
    if con:
        partes.append(
            "Capacidad por vehículo usada para el factor de ocupación: "
            + " · ".join(f"{m} {capacidades[m]:g}" for m in con)
            + "."
        )
    if sin:
        partes.append(
            f"⚠️ Sin capacidad cargada para: {', '.join(sin)}. Sus líneas no "
            "tienen factor de ocupación."
        )
    return "  \n".join(partes)
