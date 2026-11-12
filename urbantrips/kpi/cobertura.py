"""Cobertura GPS de la demanda y universo comun de los cocientes de oferta.

Una etapa prueba que su vehiculo estaba operando en ese momento. Si en ese
momento el vehiculo no tiene un servicio GPS valido, esa operacion no esta en la
oferta medida (km, vehiculos, velocidades). De ahi salen tres cosas:

- **Cobertura GPS** de una linea en un dia: etapas que ocurren dentro de un
  servicio valido de su vehiculo, sobre el total de etapas. Mide cuanta
  operacion se pierde por problemas del GPS.
- **Mismo universo en los cocientes**: pasajeros por km, factor de ocupacion y
  pasajeros por vehiculo dividen los pasajeros que viajaron dentro de servicios
  validos por la oferta de esos mismos servicios. Con todos los pasajeros en el
  numerador el cociente queda inflado en la proporcion en que falta GPS.
- **Umbral**: con cobertura menor a `COBERTURA_GPS_MINIMA` los indicadores de
  oferta describen una parte chica de la operacion y quedan sin calcular.

El vehiculo de una etapa es el interno principal de su formacion (ver
`datamodel/gps_vehiculos.py`).
"""

import pandas as pd

from urbantrips.utils.utils import leer_tuning_num

# Cobertura GPS por debajo de la cual los indicadores de oferta de una linea
# (o ramal) en un dia quedan sin calcular. Override: `kpi_cobertura_gps_minima`
# en configs/tuning.yaml.
COBERTURA_GPS_MINIMA = 0.5

# Piezas para consultas sobre `etapas e`: el vehiculo de la etapa (el interno
# principal de su formacion) y su hora en segundos, comparable con `min_ts` /
# `max_ts` de `services`.
JOIN_VEHICULO = (
    "LEFT JOIN gps_vehiculos veh ON veh.dia = e.dia AND veh.id_linea = e.id_linea"
    " AND veh.interno = e.interno AND NOT veh.es_principal"
)
VEHICULO_DE_LA_ETAPA = "COALESCE(veh.interno_vehiculo, e.interno)"
TS_DE_LA_ETAPA = "epoch(CAST((e.dia||' '||e.tiempo) AS TIMESTAMP))::BIGINT"

# Cocientes que tienen a la oferta medida con GPS como denominador.
COCIENTES_DE_OFERTA = [
    "pvd", "kvd_route", "kvd_route_gps", "ipk_route", "ipk_route_gps",
    "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
    "fo_median_od", "fo_median_route", "fo_median_route_gps",
]
# Todo lo que el umbral deja sin calcular: esos cocientes y la oferta misma.
COLUMNAS_DE_OFERTA = [
    "tot_veh", "vehiculos_operativos",
    "tot_km_route", "tot_km_route_gps", "tot_km_route_estimado",
    "velocidad_comercial_route", "velocidad_comercial_route_gps",
    "velocidad_comercial_am_route", "velocidad_comercial_am_route_gps",
    "velocidad_comercial_pm_route", "velocidad_comercial_pm_route_gps",
    "distancia_media_veh_route", "distancia_media_veh_route_gps",
] + COCIENTES_DE_OFERTA


def cobertura_gps_minima() -> float:
    """Umbral de cobertura GPS (tuning.yaml)."""
    return leer_tuning_num("kpi_cobertura_gps_minima", COBERTURA_GPS_MINIMA)


def en_servicio_valido(filtro_dias: str = "") -> str:
    """Condicion SQL, para una consulta sobre `etapas e` con `JOIN_VEHICULO`:
    la etapa ocurre dentro de un servicio valido de su vehiculo. Una etapa sin
    hora no se puede ubicar en un servicio y da FALSE.

    `filtro_dias` acota los servicios que se miran (p. ej. `AND s.dia IN
    ('2025-10-29')`): `services` es acumulativa y sin el filtro se recorren los
    de toda la base.
    """
    return f"""(
        e.tiempo IS NOT NULL AND EXISTS (
            SELECT 1 FROM services s
            WHERE s.valid = 1 AND s.dia = e.dia AND s.id_linea = e.id_linea
              AND s.interno = {VEHICULO_DE_LA_ETAPA}
              AND {TS_DE_LA_ETAPA} BETWEEN s.min_ts AND s.max_ts
              {filtro_dias}
        )
    )"""


def leer_cobertura_gps(ctx, dias_in: str, entidad: list) -> pd.DataFrame:
    """Cobertura GPS por `entidad` + dia: etapas con hora que ocurren dentro de
    un servicio valido de su vehiculo, sobre las etapas con hora.

    `dias_in` es la lista de dias ya armada para un `IN (...)`. Las etapas sin
    hora no cuentan; si una linea no tiene ninguna, la cobertura queda nula (no
    se puede medir) y no se le aplica el umbral.
    """
    claves = ", ".join(f"e.{c}" for c in ["dia"] + entidad)
    en_servicio = en_servicio_valido(f"AND s.dia IN ({dias_in})")
    return ctx.data.query(f"""
        SELECT {claves},
            round(
                (count(*) FILTER (WHERE {en_servicio}))::DOUBLE
                / nullif(count(*) FILTER (WHERE e.tiempo IS NOT NULL), 0),
                4
            ) AS cobertura_gps
        FROM etapas e
        {JOIN_VEHICULO}
        WHERE e.dia IN ({dias_in})
        GROUP BY {claves}
    """)


def universo_de_oferta(df):
    """Devuelve `(pasajeros, km_estimados)` para las filas de `df`, que trae
    `tot_pax`, `tot_pax_servicio`, `tot_km_route` y `cobertura_gps`. No
    modifica `df`.

    - Pasajeros que entran a los cocientes de oferta: los que viajaron dentro
      de servicios validos. Si la cobertura no se pudo medir (etapas sin hora,
      o una base anterior a estas columnas) se usan todos, como antes.
    - Km estimados: observados / cobertura. Supone que la operacion sin GPS
      tiene los mismos pasajeros por km que la operacion con GPS; nulo si no hay
      cobertura medida.
    """
    cobertura = df["cobertura_gps"]
    pasajeros = df["tot_pax_servicio"].where(cobertura.notna(), df["tot_pax"])
    km_estimados = df["tot_km_route"] / cobertura.where(cobertura > 0)
    return pasajeros, km_estimados


def vaciar_oferta_sin_cobertura(df, minimo=None):
    """Deja sin calcular los indicadores de oferta de las filas con cobertura
    GPS menor a `minimo` (por defecto, el de tuning.yaml). Las filas sin
    cobertura medida no se tocan, y un `df` sin `cobertura_gps` o sin ninguna
    fila bajo el umbral se devuelve tal cual."""
    if "cobertura_gps" not in df.columns:
        return df
    if minimo is None:
        minimo = cobertura_gps_minima()
    sin_cobertura = pd.to_numeric(df["cobertura_gps"], errors="coerce") < minimo
    if not sin_cobertura.any():
        return df
    cols = df.columns.intersection(COLUMNAS_DE_OFERTA)
    df = df.copy()
    df[cols] = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
        .mask(sin_cobertura, axis=0)
    )
    return df
