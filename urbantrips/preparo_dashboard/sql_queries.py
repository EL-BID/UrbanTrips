"""
Reusable SQL fragments for DuckDB-based dashboard aggregations.

All CTEs reference tables in the *data* database (etapas, viajes,
travel_times_legs, travel_times_trips) and produce derived columns that
match — column for column — those built by load_and_process_data() in pandas.

Parity notes (each one reproduces a specific pandas transform; do not "fix"
without re-running the proc-CTE parity probe):
- viajes.kmh_od is capped at VELOCIDAD_MAXIMA_KMH (>= cap -> NULL) then 0-filled.
  etapas.kmh_od is NOT capped, only 0-filled.
- etapas.travel_time_min is truncated to int (pandas .astype(int)); viajes keeps
  the fractional value. Both are 0-filled.
- distancia_agregada uses the *singular* viaje labels emitted by
  clasificar_distancia_agregada(nivel="viaje").
"""

from urbantrips.utils.utils import VELOCIDAD_MAXIMA_KMH

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


# ── travel_time_min / kmh_od expressions (parity-critical) ───────────────────
#
# viajes: fractional ttm, 0-filled; kmh from fractional ttm, capped >= cap, 0-filled.
_TTM_VIAJE = "COALESCE(CAST(tt.travel_time_min AS DOUBLE), 0)"
# round_even == numpy/pandas half-to-even; plain ROUND would diverge on x.x5 ties.
# Compute the ratio in REAL (float32) to mirror pandas' float32 arithmetic, else
# ~0.3% of values flip by 0.1 at x.x5 ties due to float32-vs-double precision.
_KMH_VIAJE_RATIO = "CAST(tt.distance_od AS REAL) / (CAST(tt.travel_time_min AS REAL) / CAST(60 AS REAL))"
_KMH_VIAJE = f"""
    COALESCE(
        CASE
            WHEN CAST(tt.travel_time_min AS DOUBLE) > 0 THEN
                CASE
                    WHEN round_even({_KMH_VIAJE_RATIO}, 1) >= {VELOCIDAD_MAXIMA_KMH}
                        THEN NULL
                    ELSE round_even({_KMH_VIAJE_RATIO}, 1)
                END
            ELSE NULL
        END, 0)
"""

# etapas: ttm truncated to int (pandas .astype(int)) then 0-filled; kmh from the
# truncated ttm, NOT capped, 0-filled.
_TTM_ETAPA = "TRUNC(COALESCE(CAST(tt.travel_time_min AS DOUBLE), 0))"
_KMH_ETAPA = f"""
    COALESCE(
        CASE
            WHEN {_TTM_ETAPA} > 0 THEN
                round_even(CAST(tt.distance_od AS DOUBLE) / ({_TTM_ETAPA} / 60.0), 1)
            ELSE NULL
        END, 0)
"""


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
        {_TTM_VIAJE}                                               AS travel_time_min,
        {_KMH_VIAJE}                                               AS kmh_od,
        CASE WHEN v.cant_etapas > 1 THEN 1 ELSE 0 END             AS transferencia,
        {_fmt(_RANGO_HORA, 'v.hora')}                              AS rango_hora,
        CASE WHEN tt.distance_od > 5
             THEN 'Viaje largo (>5kms)'
             ELSE 'Viaje corto (<=5kms)' END                       AS distancia_agregada,
        {_fmt(_TARIFA_CASE, 'v.tarifa')}                           AS tarifa_agregada,
        {_fmt(_GENERO_CASE, 'v.genero')}                           AS genero_agregado,
        -- minutes to the next trip for the same user on the same day. Ordered by
        -- id_viaje, the canonical chronological trip sequence (verified: order by
        -- id_viaje == order by tiempo, 0 discrepancies). The legacy pandas path
        -- used the unsorted frame order + .dt.seconds, which turned backward gaps
        -- into ~1380-min artefacts and inflated the mean ~3x — this is the fix.
        round_even((
            LEAD(epoch(TRY_CAST(v.dia || ' ' || v.tiempo AS TIMESTAMP)))
                OVER (PARTITION BY v.dia, v.id_tarjeta
                      ORDER BY v.dia, v.id_tarjeta, v.id_viaje)
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
        e.id,
        e.dia,
        {_fmt(_MES, 'e.dia')}                                      AS mes,
        {_fmt(_TIPO_DIA, 'e.dia')}                                 AS tipo_dia,
        e.id_tarjeta,
        e.id_viaje,
        e.id_linea,
        e.id_ramal,
        e.hora,
        e.modo,
        e.factor_expansion_linea,
        tt.distance_od,
        {_TTM_ETAPA}                                               AS travel_time_min,
        {_KMH_ETAPA}                                               AS kmh_od,
        {_fmt(_TARIFA_CASE, 'e.tarifa')}                           AS tarifa_agregada,
        {_fmt(_GENERO_CASE, 'e.genero')}                           AS genero_agregado
    FROM etapas e
    LEFT JOIN travel_times_legs tt ON e.id = tt.id
    WHERE e.od_validado = 1
      AND tt.distance_od IS NOT NULL
)
"""


# ── one-shot materialisation ─────────────────────────────────────────────────
#
# The proc-CTEs join etapas/viajes against travel_times (30.9M / 26.6M rows) and
# apply the classifier CASEs. The dashboard-prep consumers each scan them several
# times, so recomputing that join per scan dominates the runtime. These helpers
# build the proc relations ONCE as DuckDB temp tables (RAM bounded by memory_limit,
# spills to temp_directory) so every consumer reads pre-joined/classified rows.

ETAPAS_PROC_MAT = "etapas_proc_mat"
VIAJES_PROC_MAT = "viajes_proc_mat"


def materializar_proc_tables(ctx, replace=False):
    """Build {etapas,viajes}_proc_mat temp tables in the data DB.

    replace=True rebuilds fresh (the orchestrator, once per run). The default
    CREATE TEMP TABLE IF NOT EXISTS lets a consumer called standalone build it on
    first use and reuse it; within an orchestrator run the first call already
    materialised it, so the rest are no-ops.
    """
    verb = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE IF NOT EXISTS"
    ctx.data.execute(
        f"{verb} {ETAPAS_PROC_MAT} AS WITH {ETAPAS_PROC_CTE} SELECT * FROM etapas_proc"
    )
    ctx.data.execute(
        f"{verb} {VIAJES_PROC_MAT} AS WITH {VIAJES_PROC_CTE} SELECT * FROM viajes_proc"
    )


def drop_proc_tables(ctx):
    ctx.data.execute(f"DROP TABLE IF EXISTS {ETAPAS_PROC_MAT}")
    ctx.data.execute(f"DROP TABLE IF EXISTS {VIAJES_PROC_MAT}")
