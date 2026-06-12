"""
Reusable SQL fragments for DuckDB-based dashboard aggregations.

All CTEs reference tables in the *data* database (etapas, viajes,
travel_times_legs, travel_times_trips) and produce derived columns that
match those built by load_and_process_data() in pandas.
"""

# ── shared classification expressions ────────────────────────────────────────

_TARIFA_CASE = """
    CASE
        WHEN {col} IS NULL OR TRIM({col}) IN ('', '-')
            THEN 'sin_descuento'
        WHEN lower({col}) LIKE '%jubilad%'
          OR lower({col}) LIKE '%pensionad%'
          OR lower({col}) LIKE '%escolar%'
            THEN 'educacion_jubilacion'
        ELSE 'tarifa_social'
    END
"""

_GENERO_CASE = """
    CASE
        WHEN lower(trim({col})) IN ('m', 'masculino', 'varón', 'varon', 'hombre')
            THEN 'Masculino'
        WHEN lower(trim({col})) IN ('f', 'femenino', 'mujer')
            THEN 'Femenino'
        ELSE 'No informado'
    END
"""

_TIPO_DIA = "CASE WHEN isodow(CAST({col} AS DATE)) >= 6 THEN 'Fin de Semana' ELSE 'Hábil' END"
_MES      = "LEFT({col}, 7)"
_RANGO_HORA = """
    CASE
        WHEN {col} BETWEEN 13 AND 16 THEN '13-16'
        WHEN {col} BETWEEN 17 AND 24 THEN '17-24'
        ELSE '0-12'
    END
"""


def _fmt(expr: str, col: str) -> str:
    return expr.replace("{col}", col)


# ── viajes CTE ────────────────────────────────────────────────────────────────

VIAJES_PROC_CTE = f"""
viajes_proc AS (
    SELECT
        v.dia,
        {_fmt(_MES, 'v.dia')}                                      AS mes,
        {_fmt(_TIPO_DIA, 'v.dia')}                                 AS tipo_dia,
        v.id_tarjeta,
        v.id_viaje,
        v.tiempo,
        v.hora,
        v.cant_etapas,
        v.modo,
        v.genero,
        v.tarifa,
        v.factor_expansion_linea,
        v.factor_expansion_tarjeta,
        tt.distance_od,
        NULLIF(COALESCE(CAST(tt.travel_time_min AS DOUBLE), 0), 0) AS travel_time_min,
        NULLIF(
            CASE WHEN COALESCE(tt.travel_time_min, 0) > 0
                 THEN ROUND(
                          CAST(tt.distance_od AS DOUBLE)
                          / (CAST(tt.travel_time_min AS DOUBLE) / 60.0), 1)
                 ELSE NULL END,
            0)                                                      AS kmh_od,
        CASE WHEN v.cant_etapas > 1 THEN 1 ELSE 0 END             AS transferencia,
        {_fmt(_RANGO_HORA, 'v.hora')}                              AS rango_hora,
        CASE WHEN tt.distance_od > 5
             THEN 'Viajes largos (>5kms)'
             ELSE 'Viajes cortos (<=5kms)' END                     AS distancia_agregada,
        {_fmt(_TARIFA_CASE, 'v.tarifa')}                           AS tarifa_agregada,
        {_fmt(_GENERO_CASE, 'v.genero')}                           AS genero_agregado,
        -- minutes to the next trip for the same user on the same day
        ROUND((
            LEAD(epoch(TRY_CAST(v.dia || ' ' || v.tiempo AS TIMESTAMP)))
                OVER (PARTITION BY v.dia, v.id_tarjeta ORDER BY v.tiempo)
            - epoch(TRY_CAST(v.dia || ' ' || v.tiempo AS TIMESTAMP))
        ) / 60.0, 0)                                               AS diff_time
    FROM viajes v
    LEFT JOIN travel_times_trips tt
        ON  v.dia        = tt.dia
        AND v.id_tarjeta = tt.id_tarjeta
        AND v.id_viaje   = tt.id_viaje
    WHERE v.od_validado = 1
      AND tt.distance_od IS NOT NULL
)
"""


# ── etapas CTE ────────────────────────────────────────────────────────────────

ETAPAS_PROC_CTE = f"""
etapas_proc AS (
    SELECT
        e.dia,
        {_fmt(_MES, 'e.dia')}                                      AS mes,
        {_fmt(_TIPO_DIA, 'e.dia')}                                 AS tipo_dia,
        e.id_tarjeta,
        e.id_viaje,
        e.modo,
        e.factor_expansion_linea,
        tt.distance_od,
        NULLIF(COALESCE(CAST(tt.travel_time_min AS DOUBLE), 0), 0) AS travel_time_min,
        NULLIF(
            CASE WHEN COALESCE(tt.travel_time_min, 0) > 0
                 THEN ROUND(
                          CAST(tt.distance_od AS DOUBLE)
                          / (CAST(tt.travel_time_min AS DOUBLE) / 60.0), 1)
                 ELSE NULL END,
            0)                                                      AS kmh_od,
        {_fmt(_TARIFA_CASE, 'e.tarifa')}                           AS tarifa_agregada,
        {_fmt(_GENERO_CASE, 'e.genero')}                           AS genero_agregado
    FROM etapas e
    LEFT JOIN travel_times_legs tt ON e.id = tt.id
    WHERE e.od_validado = 1
      AND tt.distance_od IS NOT NULL
)
"""
