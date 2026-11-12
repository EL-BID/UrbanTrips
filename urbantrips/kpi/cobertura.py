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

import numpy as np
import pandas as pd

from urbantrips.utils.utils import leer_tuning_num

# Cobertura GPS por debajo de la cual los indicadores de oferta de una linea
# (o ramal) en un dia quedan sin calcular. Override: `kpi_cobertura_gps_minima`
# en configs/tuning.yaml.
COBERTURA_GPS_MINIMA = 0.5

# JOIN y condicion para saber, en una consulta sobre `etapas e`, si la etapa
# ocurre dentro de un servicio valido de su vehiculo. Una etapa sin hora no se
# puede ubicar en un servicio: la condicion da NULL.
JOIN_VEHICULO = (
    "LEFT JOIN gps_vehiculos veh ON veh.dia = e.dia AND veh.id_linea = e.id_linea"
    " AND veh.interno = e.interno AND NOT veh.es_principal"
)
EN_SERVICIO_VALIDO = """(
        e.tiempo IS NOT NULL AND EXISTS (
            SELECT 1 FROM services s
            WHERE s.valid = 1 AND s.dia = e.dia AND s.id_linea = e.id_linea
              AND s.interno = COALESCE(veh.interno_vehiculo, e.interno)
              AND epoch(CAST((e.dia || ' ' || e.tiempo) AS TIMESTAMP))::BIGINT
                  BETWEEN s.min_ts AND s.max_ts
        )
    )"""

# Indicadores que describen la oferta medida con GPS, o que la usan de
# denominador: son los que el umbral deja sin calcular.
COLUMNAS_DE_OFERTA = [
    "tot_veh", "vehiculos_operativos",
    "tot_km_route", "tot_km_route_gps", "tot_km_route_estimado",
    "velocidad_comercial_route", "velocidad_comercial_route_gps",
    "velocidad_comercial_am_route", "velocidad_comercial_am_route_gps",
    "velocidad_comercial_pm_route", "velocidad_comercial_pm_route_gps",
    "distancia_media_veh_route", "distancia_media_veh_route_gps",
    "pvd", "kvd_route", "kvd_route_gps", "ipk_route", "ipk_route_gps",
    "fo_mean_od", "fo_mean_route", "fo_mean_route_gps",
    "fo_median_od", "fo_median_route", "fo_median_route_gps",
]


def cobertura_gps_minima() -> float:
    """Umbral de cobertura GPS (tuning.yaml)."""
    return leer_tuning_num("kpi_cobertura_gps_minima", COBERTURA_GPS_MINIMA)


def leer_cobertura_gps(ctx, dias_in: str, entidad: list) -> pd.DataFrame:
    """Cobertura GPS por `entidad` + dia: etapas con hora que ocurren dentro de
    un servicio valido de su vehiculo, sobre las etapas con hora.

    `dias_in` es la lista de dias ya armada para un `IN (...)`. Las etapas sin
    hora no cuentan; si una linea no tiene ninguna, la cobertura queda nula (no
    se puede medir) y no se le aplica el umbral.
    """
    claves = ", ".join(f"e.{c}" for c in ["dia"] + entidad)
    cobertura = ctx.data.query(f"""
        SELECT {claves},
            count(*) FILTER (WHERE e.tiempo IS NOT NULL) AS etapas_con_hora,
            count(*) FILTER (WHERE {EN_SERVICIO_VALIDO}) AS etapas_en_servicio
        FROM etapas e
        {JOIN_VEHICULO}
        WHERE e.dia IN ({dias_in})
        GROUP BY {claves}
    """)
    con_hora = cobertura["etapas_con_hora"].astype("float64")
    cobertura["cobertura_gps"] = (
        cobertura["etapas_en_servicio"].astype("float64") / con_hora.where(con_hora > 0)
    ).round(4)
    return cobertura[["dia"] + entidad + ["cobertura_gps"]]


def pasajeros_del_universo(tot_pax, tot_pax_servicio, cobertura_gps):
    """Pasajeros que entran a los cocientes de oferta: los que viajaron dentro
    de servicios validos. Si la cobertura no se pudo medir (etapas sin hora, o
    una base anterior a esta columna) se usan todos, como antes."""
    return tot_pax_servicio.where(cobertura_gps.notna(), tot_pax)


def km_estimados(tot_km, cobertura_gps):
    """Km de la linea estimados a partir de los observados: observados /
    cobertura. Supone que la operacion sin GPS tiene los mismos pasajeros por km
    que la operacion con GPS. Nulo si no hay cobertura medida."""
    return tot_km / cobertura_gps.where(cobertura_gps > 0)


def vaciar_oferta_sin_cobertura(df, minimo=None):
    """Deja sin calcular los indicadores de oferta de las filas con cobertura
    GPS menor a `minimo`. Las filas sin cobertura medida no se tocan."""
    if "cobertura_gps" not in df.columns:
        return df
    if minimo is None:
        minimo = cobertura_gps_minima()
    sin_cobertura = pd.to_numeric(df["cobertura_gps"], errors="coerce") < minimo
    if not sin_cobertura.any():
        return df
    df = df.copy()
    for col in COLUMNAS_DE_OFERTA:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df.loc[sin_cobertura, col] = np.nan
    return df
