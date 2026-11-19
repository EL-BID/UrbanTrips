"""Que vehiculo es cada interno del GPS.

En la tabla `gps` un "interno" es un equipo que reporta posicion, y no siempre
equivale a un vehiculo:

- En los modos guiados (tranvia, tren, metro) cada formacion lleva varios
  equipos y cada uno reporta como un interno distinto. Contar cada interno como
  un vehiculo multiplica los km y los vehiculos de la linea.
- Hay internos bajo los que reportan dos vehiculos a la vez: la traza alterna
  entre dos posiciones lejanas y gran parte de sus tramos son saltos.

`detectar_vehiculos_gps` agrupa, por dia y linea de modo guiado, los internos que
se mueven juntos, elige uno como principal de cada vehiculo y marca los internos cuya
traza no sirve para medir oferta. La tabla `gps` no se modifica: el resultado se
guarda en `gps_vehiculos` y lo usan los servicios y los indicadores.
"""

import logging

import duckdb
import numpy as np
import pandas as pd

from urbantrips.geo.geo import distancia_recta_km
from urbantrips.utils.dataframe import leer_dia
from urbantrips.utils.sql import create_days_sql_filter

logger = logging.getLogger(__name__)

# Las trazas se comparan interpolando la posicion de cada interno cada
# PASO_GRILLA_SEG. Solo se interpola entre dos puntos separados por menos de
# HUECO_MAXIMO_SEG: en un hueco mayor la linea recta no representa el recorrido.
PASO_GRILLA_SEG = 60
HUECO_MAXIMO_SEG = 300
# Un interno esta en movimiento en un instante si en el ultimo minuto se desplazo
# mas de esta distancia (3 km/h). Parados en la misma terminal no cuenta como
# "ir juntos".
MOVIMIENTO_MINIMO_KM = 0.05
# Dos internos van juntos en un instante si estan a menos de esta distancia.
DISTANCIA_JUNTOS_KM = 0.3
# Dos internos son el mismo vehiculo si, de los instantes en que los dos tienen
# posicion y alguno se mueve, van juntos en al menos PROP_JUNTOS_MINIMA; y esos
# instantes son al menos MINUTOS_JUNTOS_MINIMOS y al menos
# PROP_SOLAPAMIENTO_MINIMA del tiempo en movimiento del que menos se mueve.
PROP_JUNTOS_MINIMA = 0.9
MINUTOS_JUNTOS_MINIMOS = 20
PROP_SOLAPAMIENTO_MINIMA = 0.5
# Un interno con esta proporcion de tramos con salto (velocidad imposible en
# linea recta) o mas tiene dos vehiculos mezclados: su traza no mide oferta.
PROP_SALTOS_MAXIMA = 0.2
# Con menos puntos que estos no se evalua la proporcion de saltos.
MIN_PUNTOS_SALTOS = 10

# Modos donde una formacion lleva varios equipos: ahi dos internos que se mueven
# juntos son el mismo vehiculo. En colectivos lo habitual es lo contrario (dos
# vehiculos que circulan pegados), asi que no se agrupan.
MODOS_CON_FORMACIONES = ("tren", "metro", "tranvia")

MOTIVO_SECUNDARIO = "equipo_secundario"
MOTIVO_SALTOS = "gps_con_saltos"

COLUMNAS = [
    "dia", "id_linea", "interno", "interno_vehiculo", "es_principal",
    "n_equipos", "n_puntos", "km", "prop_saltos", "gps_valido", "motivo",
]

# Vehiculo de cada transaccion de un equipo con dos vehiculos mezclados (ver
# `asignar_transacciones_mezcladas`). `id` es el de `transacciones`, que las
# etapas conservan.
COLUMNAS_TRX = ["dia", "id", "id_linea", "interno", "interno_vehiculo"]


def _resumen_por_interno(gps, tope_kmh):
    """Puntos, km y proporcion de tramos con salto de cada (linea, interno)."""
    claves = ["id_linea", "interno"]
    previo = gps.groupby(claves, sort=False)[["fecha", "latitud", "longitud"]].shift(1)
    delta = (gps["fecha"] - previo["fecha"]).to_numpy(dtype="float64")
    recta = distancia_recta_km(
        previo["latitud"], previo["longitud"], gps["latitud"], gps["longitud"]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        salto = recta / (np.maximum(delta, 1.0) / 3600.0) > tope_kmh
    resumen = (
        gps[claves]
        .assign(_salto=salto, _km=pd.to_numeric(gps["distance_km"], errors="coerce"))
        .groupby(claves, sort=False)
        .agg(n_puntos=("_salto", "size"), saltos=("_salto", "sum"), km=("_km", "sum"))
        .reset_index()
    )
    resumen["prop_saltos"] = (resumen["saltos"] / resumen["n_puntos"]).round(3)
    return resumen.drop(columns="saltos")


def _posiciones_en_grilla(linea, grilla, excluir=()):
    """Posicion interpolada de cada interno de una linea en cada instante de
    `grilla`, sin los internos de `excluir`. Devuelve (internos, lat, lon), con
    NaN donde no hay posicion."""
    internos, lats, lons = [], [], []
    for interno, traza in linea.groupby("interno", sort=True):
        if interno in excluir:
            continue
        t = traza["fecha"].to_numpy(dtype="float64")
        # un punto por segundo: np.interp necesita tiempos crecientes
        t, primero = np.unique(t, return_index=True)
        if len(t) < 2:
            continue
        lat = traza["latitud"].to_numpy(dtype="float64")[primero]
        lon = traza["longitud"].to_numpy(dtype="float64")[primero]
        pos = np.searchsorted(t, grilla, side="right")
        dentro = (pos > 0) & (pos < len(t))
        hueco = np.full(len(grilla), np.inf)
        hueco[dentro] = t[pos[dentro]] - t[pos[dentro] - 1]
        valido = dentro & (hueco <= HUECO_MAXIMO_SEG)
        if not valido.any():
            continue
        internos.append(interno)
        lats.append(np.where(valido, np.interp(grilla, t, lat), np.nan))
        lons.append(np.where(valido, np.interp(grilla, t, lon), np.nan))
    if not internos:
        return [], None, None
    return internos, np.vstack(lats), np.vstack(lons)


def _km2_plano(lat1, lon1, lat2, lon2):
    """Distancia al cuadrado (km2) entre puntos cercanos, en proyeccion plana.
    Se compara contra umbrales al cuadrado para no calcular la raiz."""
    dy = (lat2 - lat1) * 111.195
    dx = (lon2 - lon1) * 111.195 * np.cos(np.radians((lat1 + lat2) / 2))
    return dx * dx + dy * dy


def _pares_juntos(internos, lat, lon):
    """Pares de internos de una linea que se mueven juntos (mismo vehiculo)."""
    with np.errstate(invalid="ignore"):
        paso2 = _km2_plano(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        mueve = np.zeros(lat.shape, dtype=bool)
        mueve[:, 1:] = paso2 > MOVIMIENTO_MINIMO_KM**2

    # Un interno que no se mueve en todo el dia no va junto con nadie.
    se_mueve = np.flatnonzero(mueve.any(axis=1))
    internos = [internos[k] for k in se_mueve]
    lat, lon, mueve = lat[se_mueve], lon[se_mueve], mueve[se_mueve]

    minutos = PASO_GRILLA_SEG / 60.0
    minutos_mov = mueve.sum(axis=1) * minutos
    con_posicion = ~np.isnan(lat)
    pares = []
    for i in range(len(internos) - 1):
        otros = slice(i + 1, len(internos))
        comparable = con_posicion[i] & con_posicion[otros] & (mueve[i] | mueve[otros])
        with np.errstate(invalid="ignore"):
            cerca = comparable & (
                _km2_plano(lat[i], lon[i], lat[otros], lon[otros]) < DISTANCIA_JUNTOS_KM**2
            )
        min_comp = comparable.sum(axis=1) * minutos
        min_cerca = cerca.sum(axis=1) * minutos
        menor_mov = np.minimum(minutos_mov[i], minutos_mov[otros])
        juntos = (
            (min_comp >= MINUTOS_JUNTOS_MINIMOS)
            & (min_comp >= PROP_SOLAPAMIENTO_MINIMA * menor_mov)
            & (min_cerca >= PROP_JUNTOS_MINIMA * min_comp)
        )
        for j in np.flatnonzero(juntos):
            pares.append((internos[i], internos[i + 1 + j]))
    return pares


def _componentes(nodos, pares):
    """Componentes conexas: devuelve {nodo: representante}."""
    padre = {n: n for n in nodos}

    def raiz(n):
        while padre[n] != n:
            padre[n] = padre[padre[n]]
            n = padre[n]
        return n

    for a, b in pares:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[rb] = ra
    return {n: raiz(n) for n in nodos}


def detectar_vehiculos_gps(gps, tope_kmh, lineas_con_formaciones=None):
    """Asigna cada interno del GPS de UN DIA a un vehiculo.

    `gps` trae los puntos de un dia con `dia`, `id_linea`, `interno`, `fecha`
    (segundos), `latitud`, `longitud` y `distance_km`. `tope_kmh` es la velocidad
    maxima plausible de un tramo: un tramo es un salto si la supera en linea
    recta entre dos puntos consecutivos del interno (sin separar por ramal).

    `lineas_con_formaciones` son las lineas donde se agrupan los internos que
    se mueven juntos (las de modos guiados); None las agrupa en todas. En el
    resto, dos internos que van juntos son dos vehiculos y no se agrupan.

    Devuelve una fila por (dia, id_linea, interno):

    - `interno_vehiculo`: el interno principal del vehiculo al que pertenece. Es
      el mismo interno si no comparte vehiculo con otro.
    - `es_principal`: True para el interno que representa al vehiculo.
    - `n_equipos`: internos que comparten ese vehiculo.
    - `prop_saltos`: proporcion de tramos con velocidad imposible en linea recta.
    - `gps_valido` y `motivo`: False si la traza no se usa para medir oferta, por
      ser un equipo secundario de un vehiculo (`equipo_secundario`) o por tener
      dos vehiculos mezclados (`gps_con_saltos`).
    """
    if len(gps) == 0:
        return pd.DataFrame(columns=COLUMNAS)

    dia = gps["dia"].iloc[0]
    gps = gps[["id_linea", "interno", "fecha", "latitud", "longitud", "distance_km"]]
    gps = gps.dropna(subset=["id_linea", "interno", "fecha", "latitud", "longitud"])
    gps = gps.sort_values(["id_linea", "interno", "fecha"], kind="stable")

    out = _resumen_por_interno(gps, tope_kmh)
    out["con_saltos"] = (out["n_puntos"] >= MIN_PUNTOS_SALTOS) & (
        out["prop_saltos"] >= PROP_SALTOS_MAXIMA
    )

    t0 = np.floor(gps["fecha"].min() / PASO_GRILLA_SEG) * PASO_GRILLA_SEG
    grilla = np.arange(t0, gps["fecha"].max() + PASO_GRILLA_SEG, PASO_GRILLA_SEG)

    # Los internos con saltos no se comparan: su posicion interpolada no es la de
    # ningun vehiculo.
    con_saltos = (
        out.loc[out["con_saltos"]].groupby("id_linea")["interno"].agg(set).to_dict()
    )
    grupo = {}
    for id_linea, linea in gps.groupby("id_linea", sort=False):
        if lineas_con_formaciones is not None and id_linea not in lineas_con_formaciones:
            continue
        internos, lat, lon = _posiciones_en_grilla(
            linea, grilla, excluir=con_saltos.get(id_linea, ())
        )
        if len(internos) < 2:
            continue
        pares = _pares_juntos(internos, lat, lon)
        en_pares = sorted({i for par in pares for i in par})
        for interno, rep in _componentes(en_pares, pares).items():
            grupo[(id_linea, interno)] = rep

    claves = list(zip(out["id_linea"], out["interno"]))
    out["_grupo"] = [grupo.get(c, c[1]) for c in claves]

    # Principal de cada vehiculo: el de mas km; a igualdad, mas puntos y menor
    # numero de interno.
    orden = out.sort_values(
        ["id_linea", "_grupo", "km", "n_puntos", "interno"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    principal = orden.groupby(["id_linea", "_grupo"], sort=False)["interno"].first()
    out = out.join(principal.rename("interno_vehiculo"), on=["id_linea", "_grupo"])
    out["n_equipos"] = out.groupby(["id_linea", "_grupo"])["interno"].transform("size")
    out["es_principal"] = out["interno"] == out["interno_vehiculo"]

    out["motivo"] = np.select(
        [~out["es_principal"], out["con_saltos"]],
        [MOTIVO_SECUNDARIO, MOTIVO_SALTOS],
        default=None,
    )
    out["gps_valido"] = out["motivo"].isna()
    out["km"] = out["km"].round(3)
    out["dia"] = dia
    return out.reindex(columns=COLUMNAS).reset_index(drop=True)


def lineas_de_formaciones(modos):
    """Lineas de `modos` ({id_linea: modo}) cuyo modo lleva formaciones."""
    return {
        l for l, modo in modos.items()
        if str(modo).strip().lower() in MODOS_CON_FORMACIONES
    }


def equipos_mezclados(vehiculos, lineas_con_formaciones=None):
    """Filas de `vehiculos` (de `gps_vehiculos`) de los equipos con dos vehiculos
    mezclados (`gps_con_saltos`), solo de `lineas_con_formaciones` (None: de
    todas las lineas)."""
    mezclados = vehiculos[vehiculos["motivo"] == MOTIVO_SALTOS]
    if lineas_con_formaciones is not None:
        mezclados = mezclados[mezclados["id_linea"].isin(list(lineas_con_formaciones))]
    return mezclados


def asignar_transacciones_mezcladas(trx, gps, vehiculos, lineas_con_formaciones=None,
                                    servicios=None):
    """Vehiculo de cada transaccion validada en un equipo con dos vehiculos
    mezclados (`gps_con_saltos`), en las lineas con formaciones de un dia.

    `trx` trae `dia`, `id`, `id_linea`, `interno`, `fecha` (segundos, la misma
    base que `fecha` del GPS), `latitud` y `longitud`; `gps`, los puntos del dia;
    `vehiculos`, la salida de `detectar_vehiculos_gps` para ese dia. Devuelve
    una fila por transaccion asignada, con las columnas de COLUMNAS_TRX.

    Bajo el interno de esos equipos reportan dos (o mas) validadoras que van en
    formaciones distintas, asi que el equipo no se puede agrupar entero con
    ninguna. Cada transaccion, en cambio, tiene su propia hora y posicion: se
    asigna al vehiculo valido (interno principal con GPS valido de la misma
    linea) que en ese instante esta mas cerca, si esta a menos de
    DISTANCIA_JUNTOS_KM. La que no tiene ningun vehiculo cerca (posicion rota o
    vehiculo sin GPS) no se asigna y sigue con su interno.

    Solo se asignan las de `lineas_con_formaciones` (None: todas). En un
    colectivo un equipo mezclado son dos vehiculos que reportan como uno, sin
    interno propio: asignar sus transacciones al colectivo mas cercano las
    llevaria a un vehiculo que no es.

    `servicios` (`id_linea`, `interno`, `min_ts`, `max_ts` de los servicios
    validos) limita los candidatos a los vehiculos que estan operando en ese
    instante. Sin ese filtro una transaccion en una terminal o cerca del
    deposito cae en una formacion estacionada, que pasa a contar como vehiculo
    con transacciones por una o dos validaciones. None: sin filtro.
    """
    vacio = pd.DataFrame(columns=COLUMNAS_TRX)
    if len(trx) == 0 or len(gps) == 0 or len(vehiculos) == 0:
        return vacio
    mezclados = equipos_mezclados(vehiculos, lineas_con_formaciones)
    validos = vehiculos[vehiculos["es_principal"] & vehiculos["gps_valido"]]
    if len(mezclados) == 0 or len(validos) == 0:
        return vacio

    trx = trx.dropna(subset=["fecha", "latitud", "longitud"])
    gps = gps[["id_linea", "interno", "fecha", "latitud", "longitud"]].dropna()
    gps = gps.sort_values(["id_linea", "interno", "fecha"], kind="stable")
    trx_por_linea = dict(tuple(trx.groupby("id_linea")))
    gps_por_linea = dict(tuple(gps.groupby("id_linea")))
    partes = []
    for id_linea, internos in mezclados.groupby("id_linea")["interno"]:
        sel = trx_por_linea.get(id_linea, trx.iloc[:0])
        sel = sel[sel["interno"].isin(set(internos))]
        candidatos = set(validos.loc[validos["id_linea"] == id_linea, "interno"])
        if len(sel) == 0 or not candidatos:
            continue
        linea = gps_por_linea.get(id_linea, gps.iloc[:0])
        otros, lat, lon = _posiciones_en_grilla(
            linea[linea["interno"].isin(candidatos)],
            sel["fecha"].to_numpy(dtype="float64"),
        )
        if not otros:
            continue
        with np.errstate(invalid="ignore"):
            d2 = _km2_plano(
                sel["latitud"].to_numpy(dtype="float64")[None, :],
                sel["longitud"].to_numpy(dtype="float64")[None, :],
                lat, lon,
            )
        d2 = np.where(np.isnan(d2), np.inf, d2)
        if servicios is not None:
            d2 = np.where(
                _en_servicio(servicios, id_linea, otros, sel["fecha"].to_numpy()),
                d2, np.inf,
            )
        cerca = d2.min(axis=0) < DISTANCIA_JUNTOS_KM**2
        if not cerca.any():
            continue
        asignadas = sel.loc[cerca, ["dia", "id", "id_linea", "interno"]].copy()
        asignadas["interno_vehiculo"] = np.asarray(otros)[d2.argmin(axis=0)][cerca]
        partes.append(asignadas)
    if not partes:
        return vacio
    return pd.concat(partes, ignore_index=True).reindex(columns=COLUMNAS_TRX)


def _en_servicio(servicios, id_linea, internos, ts):
    """Matriz (internos x ts): el interno tiene un servicio valido de la linea
    que cubre ese instante."""
    serv = dict(tuple(servicios[servicios["id_linea"] == id_linea].groupby("interno")))
    out = np.zeros((len(internos), len(ts)), dtype=bool)
    for k, interno in enumerate(internos):
        propios = serv.get(interno)
        if propios is None:
            continue
        for ini, fin in zip(propios["min_ts"].to_numpy(), propios["max_ts"].to_numpy()):
            out[k] |= (ts >= ini) & (ts <= fin)
    return out


def guardar_vehiculos_de_transacciones(ctx, dias):
    """Asigna a un vehiculo las transacciones de los equipos con dos vehiculos
    mezclados (`asignar_transacciones_mezcladas`) y reemplaza sus filas en
    `trx_vehiculos`, un dia por vez. Necesita `gps_vehiculos`, `gps`,
    `transacciones` y `services` de esos dias. Solo en las lineas de
    `MODOS_CON_FORMACIONES`."""
    from urbantrips.carto.routes import modos_por_linea

    lineas_con_formaciones = lineas_de_formaciones(modos_por_linea(ctx))
    for dia in dias:
        ctx.data.execute(f"DELETE FROM trx_vehiculos WHERE dia = '{dia}'")
        vehiculos = leer_dia(ctx.data.query, "gps_vehiculos", ["*"], dia)
        mezclados = equipos_mezclados(vehiculos, lineas_con_formaciones)
        if len(mezclados) == 0:
            continue
        validos = vehiculos[vehiculos["es_principal"] & vehiculos["gps_valido"]]
        lineas = _lista_sql(mezclados["id_linea"])
        # se leen solo los equipos mezclados y los vehiculos candidatos; el
        # cruce exacto (linea, interno) lo hace asignar_transacciones_mezcladas
        trx = leer_dia(
            ctx.data.query, "transacciones",
            ["dia", "id", "id_linea", "interno", "fecha", "latitud", "longitud"], dia,
            extra_where=f"id_linea IN ({lineas})"
                        f" AND interno IN ({_lista_sql(mezclados['interno'])})",
        )
        candidatos = f"id_linea IN ({lineas})"
        if len(validos):
            candidatos += f" AND interno IN ({_lista_sql(validos['interno'])})"
        gps = leer_dia(
            ctx.data.query, "gps",
            ["id_linea", "interno", "fecha", "latitud", "longitud"], dia,
            extra_where=candidatos,
        )
        servicios = leer_dia(
            ctx.data.query, "services", ["id_linea", "interno", "min_ts", "max_ts"],
            dia, extra_where=f"valid = 1 AND id_linea IN ({lineas})",
        )
        asignadas = asignar_transacciones_mezcladas(
            trx, gps, vehiculos, lineas_con_formaciones, servicios
        )
        if len(asignadas):
            ctx.data.append_raw(asignadas, "trx_vehiculos")
        logger.info(
            "Transacciones de equipos con vehiculos mezclados (%s): %s asignadas "
            "a un vehiculo", dia, f"{len(asignadas):,}",
        )


def _lista_sql(valores):
    """Valores enteros distintos, listos para un `IN (...)`."""
    return ", ".join(str(int(v)) for v in pd.unique(valores))


def leer_vehiculos_de_trx(ctx, dias=None):
    """Filas de `trx_vehiculos` (vehiculo de cada transaccion reasignada).
    Vacio si la tabla no existe (base anterior a esta tabla)."""
    q = "SELECT * FROM trx_vehiculos WHERE TRUE" + create_days_sql_filter(dias)
    try:
        return ctx.data.query(q)
    except (duckdb.CatalogException, NotImplementedError) as e:
        logger.debug("trx_vehiculos no disponible: %s", e)
        return pd.DataFrame(columns=COLUMNAS_TRX)


def guardar_vehiculos_gps(ctx, gps, dia, tope_kmh):
    """Detecta los vehiculos del GPS de un dia y reemplaza sus filas en
    `gps_vehiculos`. Los internos que se mueven juntos se agrupan solo en las
    lineas de `MODOS_CON_FORMACIONES`."""
    # import diferido: carto.routes es pesado y no hace falta para la deteccion
    from urbantrips.carto.routes import modos_por_linea

    lineas_con_formaciones = lineas_de_formaciones(modos_por_linea(ctx))
    vehiculos = detectar_vehiculos_gps(gps, tope_kmh, lineas_con_formaciones)
    ctx.data.execute(f"DELETE FROM gps_vehiculos WHERE dia = '{dia}'")
    if len(vehiculos) == 0:
        return
    ctx.data.append_raw(vehiculos, "gps_vehiculos")
    secundarios = vehiculos["motivo"] == MOTIVO_SECUNDARIO
    con_saltos = vehiculos["motivo"] == MOTIVO_SALTOS
    logger.info(
        "Vehiculos del GPS (%s): %s internos; %s son equipos secundarios de un "
        "vehiculo (%s lineas) y %s tienen dos vehiculos mezclados",
        dia, f"{len(vehiculos):,}", f"{int(secundarios.sum()):,}",
        vehiculos.loc[secundarios, "id_linea"].nunique(), f"{int(con_saltos.sum()):,}",
    )


def leer_gps_no_validos(ctx, dias=None):
    """Filas de `gps_vehiculos` de los internos cuya traza no mide oferta:
    equipos secundarios de un vehiculo y trazas con dos vehiculos mezclados.
    Son las unicas que cambian algo aguas abajo: en el resto el interno es su
    propio vehiculo.

    `dias` acota la lectura a una lista de dias (None o vacia: todos). Devuelve
    vacio si la tabla no existe (base anterior a esta tabla) o si el adapter no
    resuelve consultas con filtro.
    """
    q = "SELECT * FROM gps_vehiculos WHERE NOT gps_valido" + create_days_sql_filter(dias)
    try:
        no_validos = ctx.data.query(q)
    except (duckdb.CatalogException, NotImplementedError) as e:
        logger.debug("gps_vehiculos no disponible: %s", e)
        return pd.DataFrame(columns=COLUMNAS)
    if "interno_vehiculo" not in no_validos.columns:
        return pd.DataFrame(columns=COLUMNAS)
    return no_validos


def clave_interno(df):
    """Clave (dia, id_linea, interno) con tipos normalizados, para cruzar tablas
    que guardan esas columnas con tipos distintos."""
    return pd.MultiIndex.from_arrays(
        [
            df["dia"].astype(str),
            pd.to_numeric(df["id_linea"], errors="coerce").astype("float64"),
            pd.to_numeric(df["interno"], errors="coerce").astype("float64"),
        ]
    )


def buscar_en_no_validos(df, no_validos, columna):
    """Valor de `columna` de `no_validos` para cada fila de `df`, cruzando por
    (dia, id_linea, interno). Nulo en las filas de internos validos.

    La clave se arma solo para las filas cuyo interno aparece en `no_validos`:
    son una fraccion minima y `df` puede tener millones de filas.
    """
    resultado = pd.Series(None, index=df.index, dtype="object")
    if len(df) == 0 or len(no_validos) == 0:
        return resultado
    candidatas = df["interno"].isin(no_validos["interno"]).to_numpy()
    if candidatas.any():
        valores = pd.Series(no_validos[columna].to_numpy(), index=clave_interno(no_validos))
        valores = valores[~valores.index.duplicated()]
        resultado[candidatas] = valores.reindex(clave_interno(df.loc[candidatas])).to_numpy()
    return resultado


def mapear_interno_a_vehiculo(df, no_validos, trx_vehiculos=None):
    """Reemplaza `interno` por el interno principal de su vehiculo.

    Las etapas llevan el interno del equipo donde se valido; para cruzarlas con
    la oferta del vehiculo (servicios, velocidad) tienen que llevar el interno
    principal. Los internos que no estan en `no_validos` quedan igual.

    `trx_vehiculos` (de `leer_vehiculos_de_trx`) asigna transaccion por
    transaccion las de los equipos con dos vehiculos mezclados, que no tienen un
    vehiculo unico. Se cruza por `id` (el de `transacciones`, que las etapas
    conservan) y tiene prioridad; requiere que `df` traiga `id`.
    """
    principal = buscar_en_no_validos(df, no_validos, "interno_vehiculo")
    if trx_vehiculos is not None and len(trx_vehiculos) and "id" in df.columns:
        # como en buscar_en_no_validos: solo se buscan las filas de los equipos
        # con transacciones reasignadas, una fraccion minima de `df`
        candidatas = df["interno"].isin(trx_vehiculos["interno"]).to_numpy()
        if candidatas.any():
            unicas = trx_vehiculos.drop_duplicates("id")
            por_id = pd.Series(
                unicas["interno_vehiculo"].to_numpy(),
                index=pd.to_numeric(unicas["id"], errors="coerce"),
            )
            asignado = pd.to_numeric(df.loc[candidatas, "id"], errors="coerce").map(por_id)
            principal[candidatas] = asignado.fillna(principal[candidatas]).to_numpy()
    cambia = principal.notna()
    if not cambia.any():
        return df
    df = df.copy(deep=False)
    interno = df["interno"].copy()
    interno[cambia] = pd.to_numeric(principal[cambia]).astype(interno.dtype)
    df["interno"] = interno
    return df


def quitar_gps_no_validos(df, no_validos):
    """Saca de `df` las filas de internos cuya traza no mide oferta. Si saca
    alguna devuelve una copia, para que quien la reciba pueda agregarle columnas."""
    no_valido = buscar_en_no_validos(df, no_validos, "motivo").notna()
    return df.loc[~no_valido].copy() if no_valido.any() else df
