import logging
import multiprocessing
import time
from datetime import datetime

import numpy as np
import pandas as pd

from urbantrips.utils.utils import duracion
from urbantrips.storage.context import StorageContext
from urbantrips.storage.ports import BatchSpec

logger = logging.getLogger(__name__)

def _derive_trip_dia(legs: pd.DataFrame) -> pd.DataFrame:
    """Add trip_dia column = dia of the first leg of each (id_tarjeta, id_viaje)."""
    trip_dia = (
        legs.sort_values(["id_tarjeta", "id_viaje", "dia"])
        .groupby(["id_tarjeta", "id_viaje"], as_index=False)["dia"]
        .first()
        .rename(columns={"dia": "trip_dia"})
    )
    return legs.merge(trip_dia, on=["id_tarjeta", "id_viaje"], how="left")


@duracion
def create_trips_from_legs_and_fex(ctx: StorageContext):
    """
    Loads the legs table from db, updates expansion factors and produces
    trips and users tables.

    Produces 3 expansion factors:
    1. factor_expansion_etapa: expands individually validated legs
       (etapa_validada==1) to match total weighted legs per line.
    2. factor_expansion_tarjeta: redistributes weight from cards with
       invalid OD chains to valid ones, preserving total cards.
    3. factor_expansion_linea: expands legs from fully validated trip
       chains (od_validado==1) to match total weighted legs per line,
       then calibrates against reported transactions per line.
    """

    dias_ultima_corrida = ctx.data.get_run_days()
    if dias_ultima_corrida.empty:
        return
    dias_str = ", ".join(f"'{d}'" for d in dias_ultima_corrida["dia"].tolist())

    logger.info("Calculando factores de expansión por etapa, línea y tarjeta")

    # Se procesa UN DÍA POR VEZ (los factores son separables por día: GROUP BY
    # ... dia, JOIN USING(dia, ...); viajes/usuarios agrupan por dia → por día el
    # resultado es bit-idéntico y la RAM se acota a ~1 día).
    #
    # REBUILD+SWAP en vez de DELETE+INSERT sobre etapas: el patrón anterior
    # (DELETE WHERE dia + INSERT, día a día sobre la misma tabla) degradaba
    # progresivamente — con datos idénticos por semana, los scans WHERE dia del
    # CTAS de factores midieron 56→134→159→207s entre semanas 1→4 (+267%): las
    # filas borradas sin compactar y la fragmentación por reuso de huecos
    # encarecen cada lectura siguiente. Acá cada día se APPENDEA a _ut_etapas_new
    # (tabla fresca, sin índices) y al final un swap atómico la renombra a
    # etapas. La tabla original queda intacta durante todo el loop (los CTAS
    # leen de ella a velocidad constante) y la nueva queda clusterizada por día.
    # Los índices de etapas ya están dropeados (begin_bulk_leg_writes);
    # end_bulk_leg_writes los recrea sobre la tabla nueva tras el swap.
    ctx.data.execute("DROP TABLE IF EXISTS _ut_etapas_new")
    ctx.data.execute("CREATE TABLE _ut_etapas_new AS SELECT * FROM etapas LIMIT 0")

    dias = sorted(dias_ultima_corrida["dia"].tolist())
    # preserve_insertion_order=true (default) serializa los INSERT...SELECT para
    # emitir las filas en orden de origen: el reemplazo de etapas escribía con
    # ~2 de 20 cores (medido a escala mes: 259s/día de los ~360s del día). Acá
    # ningún orden intra-día importa — el clustering por día que #0 necesita lo
    # da la propia estructura del loop (cada día se appendea como bloque en su
    # transacción) — así que se desactiva para paralelizar escritura/compresión
    # de row-groups, y se restaura SIEMPRE al salir: la Fase 2 sí depende del
    # ORDER BY batch_id de sus INSERTs.
    ctx.data.execute("SET preserve_insertion_order = false")
    try:
        _create_trips_day_loop(ctx, dias)

        # Días de otras corridas (si los hay) pasan tal cual a la tabla nueva.
        # En una corrida fresca copia 0 filas; cuesta un solo scan de etapas.
        logger.info("  - Copiando días de otras corridas a la tabla nueva...")
        ctx.data.execute(
            f"INSERT INTO _ut_etapas_new SELECT * FROM etapas WHERE dia NOT IN ({dias_str})"
        )

        # Swap atómico: una sola transacción, o queda la etapas vieja o la nueva.
        logger.info("  - Swap etapas <- _ut_etapas_new...")
        ctx.data.execute(
            """
            BEGIN TRANSACTION;
            DROP TABLE etapas;
            ALTER TABLE _ut_etapas_new RENAME TO etapas;
            COMMIT;
            """
        )
    finally:
        ctx.data.execute("SET preserve_insertion_order = true")

    n_etapas = ctx.data.query(
        f"SELECT COUNT(*) AS n FROM etapas WHERE dia IN ({dias_str})"
    )["n"].iloc[0]
    logger.info("Viajes y usuarios creados a partir de %d etapas", n_etapas)

    totals = ctx.data.query(f"""
        SELECT
            SUM(factor_expansion_original) AS factor_expansion_original,
            SUM(factor_expansion_etapa) AS factor_expansion_etapa,
            SUM(factor_expansion_linea) AS factor_expansion_linea
        FROM etapas
        WHERE dia IN ({dias_str})
    """)
    trx_total = ctx.data.query(f"""
        SELECT
            COUNT(*) AS registros,
            COALESCE(SUM(transacciones), 0) AS transacciones
        FROM transacciones_linea
        WHERE dia IN ({dias_str})
    """)

    logger.info(
        "Verificación de factores de expansión | original=%.0f etapa=%.0f linea=%.0f trx=%.0f",
        totals.factor_expansion_original.iloc[0],
        totals.factor_expansion_etapa.iloc[0],
        totals.factor_expansion_linea.iloc[0],
        trx_total.transacciones.iloc[0],
    )
    if trx_total.registros.iloc[0] == 0:
        logger.warning("transacciones_linea no tiene registros para esta corrida")

    for table in ["_ut_etapas_fex"]:
        ctx.data.execute(f"DROP TABLE IF EXISTS {table}")


def _create_trips_day_loop(ctx: StorageContext, dias: list) -> None:
    """Cuerpo por-día de create_trips: factores, reescritura de etapas, viajes
    y usuarios de cada día. Separado para que el caller pueda encerrarlo en el
    toggle de preserve_insertion_order sin indentar 250 líneas de SQL."""

    def run_step(label: str, sql: str) -> None:
        logger.info("  - %s...", label)
        start = time.perf_counter()
        ctx.data.execute(sql)
        logger.info("    listo en %.2fs", time.perf_counter() - start)

    for i, dia in enumerate(dias, 1):
        logger.info("[create_trips] día %d/%d (%s)", i, len(dias), dia)
        dia_in = f"'{dia}'"
        run_step(
            f"[{dia}] Calculando factores en tabla temporal",
            f"""
            CREATE OR REPLACE TABLE _ut_etapas_fex AS
            WITH base AS (
                SELECT
                    e.*,
                    -- od_base replica las DOS invalidaciones del create_trips
                    -- original (urbantrips_viejo trips.py:84-99), aplicadas ANTES
                    -- de calcular los factores: (1) lat/lon = 0 (GPS faltante) y
                    -- (2) distancia OD nula o cero (par no ruteable o mismo
                    -- hexágono origen-destino). distance_od viene de
                    -- travel_times_legs (assign_time_distances, Fase 3). Sin este
                    -- gate una etapa con destino pero sin distancia medible queda
                    -- validada e infla los factores de expansión.
                    CASE
                        WHEN e.latitud = 0 AND e.longitud = 0 THEN 0
                        WHEN tt.distance_od IS NULL OR tt.distance_od = 0 THEN 0
                        ELSE e.od_validado
                    END AS od_base
                FROM etapas e
                LEFT JOIN travel_times_legs tt
                    ON e.id = tt.id AND e.dia = tt.dia
                WHERE e.dia IN ({dia_in})
            ),
            factor_etapa AS (
                SELECT
                    dia,
                    id_linea,
                    SUM(factor_expansion_original)
                    / NULLIF(
                        SUM(CASE WHEN od_base = 1 THEN factor_expansion_original ELSE 0 END),
                        0
                    ) AS ratio_etapa
                FROM base
                GROUP BY dia, id_linea
            ),
            tarjetas AS (
                SELECT
                    dia,
                    id_tarjeta,
                    AVG(factor_expansion_original) AS factor_expansion_original,
                    MIN(od_base) AS od_validado
                FROM base
                GROUP BY dia, id_tarjeta
            ),
            ajuste_tarjeta AS (
                SELECT
                    dia,
                    SUM(factor_expansion_original) AS peso_total,
                    SUM(CASE WHEN od_validado = 1 THEN factor_expansion_original ELSE 0 END)
                        AS peso_valido
                FROM tarjetas
                GROUP BY dia
            ),
            factor_tarjeta AS (
                SELECT
                    t.dia,
                    t.id_tarjeta,
                    COALESCE(
                        t.factor_expansion_original
                        * (a.peso_total / NULLIF(a.peso_valido, 0))
                        * t.od_validado,
                        0
                    ) AS factor_expansion_tarjeta
                FROM tarjetas t
                JOIN ajuste_tarjeta a USING (dia)
            ),
            base_tarjeta AS (
                SELECT
                    b.*,
                    ft.factor_expansion_tarjeta AS factor_expansion_tarjeta_new,
                    CASE
                        WHEN ft.factor_expansion_tarjeta = 0 THEN 0
                        ELSE b.od_base
                    END AS od_final
                FROM base b
                JOIN factor_tarjeta ft USING (dia, id_tarjeta)
            ),
            factor_linea AS (
                SELECT
                    pt.dia,
                    pt.id_linea,
                    (pt.peso_total / NULLIF(pv.peso_validas, 0))
                    * COALESCE(tl.transacciones / NULLIF(pt.peso_total, 0), 1) AS ratio_final
                FROM (
                    SELECT dia, id_linea, SUM(factor_expansion_original) AS peso_total
                    FROM base
                    GROUP BY dia, id_linea
                ) pt
                LEFT JOIN (
                    SELECT dia, id_linea, SUM(factor_expansion_original) AS peso_validas
                    FROM base_tarjeta
                    WHERE od_final = 1
                    GROUP BY dia, id_linea
                ) pv USING (dia, id_linea)
                LEFT JOIN transacciones_linea tl USING (dia, id_linea)
            )
            SELECT
                bt.id,
                bt.batch_id,
                bt.id_tarjeta,
                bt.dia,
                bt.id_viaje,
                bt.id_etapa,
                bt.tiempo,
                bt.hora,
                bt.modo,
                bt.id_linea,
                bt.id_ramal,
                bt.interno,
                bt.genero,
                bt.tarifa,
                bt.latitud,
                bt.longitud,
                bt.h3_o,
                bt.h3_d,
                bt.od_final AS od_validado,
                bt.od_base AS etapa_validada,
                bt.factor_expansion_original,
                COALESCE(bt.factor_expansion_original * fl.ratio_final * bt.od_final, 0)
                    AS factor_expansion_linea,
                bt.factor_expansion_tarjeta_new AS factor_expansion_tarjeta,
                COALESCE(bt.factor_expansion_original * fe.ratio_etapa * bt.od_base, 0)
                    AS factor_expansion_etapa,
                bt.distancia,
                bt.travel_time_min
            FROM base_tarjeta bt
            LEFT JOIN factor_etapa fe USING (dia, id_linea)
            LEFT JOIN factor_linea fl USING (dia, id_linea)
            """,
        )
        run_step(
            f"[{dia}] Insertando etapas con factores en tabla nueva",
            f"""
            INSERT INTO _ut_etapas_new (
                id, batch_id, id_tarjeta, dia, id_viaje, id_etapa, tiempo,
                hora, modo, id_linea, id_ramal, interno, genero, tarifa,
                latitud, longitud, h3_o, h3_d, od_validado, etapa_validada,
                factor_expansion_original, factor_expansion_linea,
                factor_expansion_tarjeta, factor_expansion_etapa, distancia,
                travel_time_min
            )
            SELECT
                id, batch_id, id_tarjeta, dia, id_viaje, id_etapa, tiempo,
                hora, modo, id_linea, id_ramal, interno, genero, tarifa,
                latitud, longitud, h3_o, h3_d, od_validado, etapa_validada,
                factor_expansion_original, factor_expansion_linea,
                factor_expansion_tarjeta, factor_expansion_etapa, distancia,
                travel_time_min
            FROM _ut_etapas_fex
            """,
        )

        run_step(
            f"[{dia}] Borrando viajes previos",
            f"DELETE FROM viajes WHERE dia IN ({dia_in})",
        )
        run_step(
            f"[{dia}] Insertando viajes",
            f"""
            INSERT INTO viajes (
                id_tarjeta, id_viaje, dia, tiempo, hora, cant_etapas, modo,
                autobus, tren, metro, tranvia, brt, cable, lancha, otros,
                h3_o, h3_d, genero, tarifa, od_validado,
                factor_expansion_linea, factor_expansion_tarjeta
            )
            WITH trips AS (
                SELECT
                    id_tarjeta,
                    id_viaje,
                    dia,
                    ARG_MIN(tiempo, id) AS tiempo,
                    ARG_MIN(hora, id) AS hora,
                    COUNT(*) AS cant_etapas,
                    SUM(CASE WHEN modo = 'autobus' THEN 1 ELSE 0 END) AS autobus,
                    SUM(CASE WHEN modo = 'tren' THEN 1 ELSE 0 END) AS tren,
                    SUM(CASE WHEN modo = 'metro' THEN 1 ELSE 0 END) AS metro,
                    SUM(CASE WHEN modo = 'tranvia' THEN 1 ELSE 0 END) AS tranvia,
                    SUM(CASE WHEN modo = 'brt' THEN 1 ELSE 0 END) AS brt,
                    SUM(CASE WHEN modo = 'cable' THEN 1 ELSE 0 END) AS cable,
                    SUM(CASE WHEN modo = 'lancha' THEN 1 ELSE 0 END) AS lancha,
                    SUM(CASE WHEN modo = 'otros' THEN 1 ELSE 0 END) AS otros,
                    ARG_MIN(h3_o, id) AS h3_o,
                    ARG_MAX(h3_d, id) AS h3_d,
                    ARG_MIN(genero, id) AS genero,
                    ARG_MIN(tarifa, id) AS tarifa,
                    MIN(od_validado) AS od_validado,
                    AVG(factor_expansion_linea) AS factor_expansion_linea,
                    AVG(factor_expansion_tarjeta) AS factor_expansion_tarjeta
                -- Se lee de _ut_etapas_fex (las etapas del día CON los factores
                -- recién calculados), NO de etapas: la tabla original conserva
                -- los factores viejos hasta el swap final.
                FROM _ut_etapas_fex
                WHERE dia IN ({dia_in})
                -- dia DEBE estar en la clave: id_viaje arranca en 1 por tarjeta
                -- cada día, así que sin dia los viajes de una misma tarjeta en
                -- días distintos colisionan y se funden en uno solo.
                GROUP BY id_tarjeta, id_viaje, dia
            ),
            classified AS (
                SELECT
                    *,
                    ((autobus > 0)::INT + (tren > 0)::INT + (metro > 0)::INT
                     + (tranvia > 0)::INT + (brt > 0)::INT + (cable > 0)::INT
                     + (lancha > 0)::INT + (otros > 0)::INT) AS cant_modos,
                    CASE
                        WHEN ((autobus > 0)::INT + (tren > 0)::INT + (metro > 0)::INT
                              + (tranvia > 0)::INT + (brt > 0)::INT + (cable > 0)::INT
                              + (lancha > 0)::INT + (otros > 0)::INT) > 1 THEN 'Multimodal'
                        WHEN cant_etapas > 1 THEN 'Multietapa'
                        WHEN autobus > 0 THEN 'autobus'
                        WHEN tren > 0 THEN 'tren'
                        WHEN metro > 0 THEN 'metro'
                        WHEN tranvia > 0 THEN 'tranvia'
                        WHEN brt > 0 THEN 'brt'
                        WHEN cable > 0 THEN 'cable'
                        WHEN lancha > 0 THEN 'lancha'
                        WHEN otros > 0 THEN 'otros'
                        ELSE ''
                    END AS modo_viaje
                FROM trips
            )
            SELECT
                id_tarjeta, id_viaje, dia, tiempo, hora, cant_etapas, modo_viaje,
                autobus, tren, metro, tranvia, brt, cable, lancha, otros,
                h3_o, h3_d, genero, tarifa, od_validado,
                factor_expansion_linea, factor_expansion_tarjeta
            FROM classified
            """,
        )

        run_step(
            f"[{dia}] Borrando usuarios previos",
            f"DELETE FROM usuarios WHERE dia IN ({dia_in})",
        )
        run_step(
            f"[{dia}] Insertando usuarios",
            f"""
            INSERT INTO usuarios (
                id_tarjeta, dia, od_validado, cant_viajes,
                factor_expansion_linea, factor_expansion_tarjeta
            )
            SELECT
                id_tarjeta,
                dia,
                MIN(od_validado) AS od_validado,
                COUNT(id_viaje) AS cant_viajes,
                AVG(factor_expansion_linea) AS factor_expansion_linea,
                AVG(factor_expansion_tarjeta) AS factor_expansion_tarjeta
            FROM viajes
            WHERE dia IN ({dia_in})
            GROUP BY dia, id_tarjeta
            """,
        )


@duracion
def verificar_integridad_viajes_etapas(ctx: StorageContext, raise_on_error: bool = True):
    """Check that the viajes table is consistent with etapas, day by day.

    Every trip key (dia, id_tarjeta, id_viaje) present in etapas must exist in
    viajes with the same cant_etapas, and vice versa. A mismatch means viajes
    was built from a different state of etapas (partial or interrupted run):
    indicators (built from viajes) would silently diverge from chains_norm and
    the dashboard maps (built from etapas). Fix: re-run `--step legs`.

    Returns a DataFrame with one row per inconsistent day (empty when OK).
    """
    diff = ctx.data.query(
        """
        WITH te AS (
            SELECT dia, id_tarjeta, id_viaje, COUNT(*) AS cant_etapas_e
            FROM etapas GROUP BY 1, 2, 3
        ),
        j AS (
            SELECT COALESCE(te.dia, v.dia) AS dia,
                   CASE WHEN v.dia IS NULL THEN 1 ELSE 0 END AS solo_en_etapas,
                   CASE WHEN te.dia IS NULL THEN 1 ELSE 0 END AS solo_en_viajes,
                   CASE WHEN te.dia IS NOT NULL AND v.dia IS NOT NULL
                             AND te.cant_etapas_e != v.cant_etapas
                        THEN 1 ELSE 0 END AS cant_etapas_distinta
            FROM te
            FULL OUTER JOIN viajes v
              ON te.dia = v.dia AND te.id_tarjeta = v.id_tarjeta
             AND te.id_viaje = v.id_viaje
        )
        SELECT dia,
               SUM(solo_en_etapas) AS viajes_solo_en_etapas,
               SUM(solo_en_viajes) AS viajes_solo_en_viajes,
               SUM(cant_etapas_distinta) AS cant_etapas_distinta
        FROM j
        GROUP BY dia
        HAVING SUM(solo_en_etapas) > 0 OR SUM(solo_en_viajes) > 0
            OR SUM(cant_etapas_distinta) > 0
        ORDER BY dia
        """
    )

    if len(diff) == 0:
        logger.info("Integridad viajes/etapas verificada: OK.")
        return diff

    for row in diff.itertuples(index=False):
        logger.error(
            "Integridad viajes/etapas FALLA para %s: %d viajes solo en etapas, "
            "%d solo en viajes, %d con cant_etapas distinta.",
            row.dia, int(row.viajes_solo_en_etapas),
            int(row.viajes_solo_en_viajes), int(row.cant_etapas_distinta),
        )
    msg = (
        "La tabla viajes no es consistente con etapas para los días: "
        f"{diff['dia'].tolist()}. Los indicadores (desde viajes) divergirían "
        "de chains_norm y los mapas (desde etapas). "
        "Re-correr `python run_all_urbantrips.py --step legs` y luego "
        "`--step dashboard`."
    )
    if raise_on_error:
        raise RuntimeError(msg)
    logger.warning(msg)
    return diff


@duracion
def rearrange_trip_id_same_od(
    ctx: StorageContext, batch: BatchSpec | None = None, dia: str | None = None
):
    """
    Takes a legs dataframe with legs and trips id and splits
    trips with same id into 2 trips with different ids and uploads
    new legs to the db

    Parameters
    ----------
    df : pandas DataFrame
        legs dataframe

    Returns
    ----------

    pandas DataFrame
        legs with new trips ids

    Particionado: `dia` procesa un solo día (preferido: etapas queda físicamente
    clusterizada por dia tras el rebuild de destinos, así el SELECT y el UPDATE
    podan row-groups). `batch` (legacy) particiona por batch_id, que perdió su
    soporte físico con ese mismo rebuild — a escala de mes cada query barría la
    tabla completa. Ambos particionados son equivalentes en resultado: todas las
    operaciones de esta función agrupan por (dia, id_tarjeta, ...) como prefijo,
    y tanto batch (hash de tarjeta) como dia preservan íntegro cada grupo
    (dia, id_tarjeta) → ningún cálculo cruza la frontera de la partición.
    """
    dias_ultima_corrida = ctx.data.get_run_days()

    scope_filter = ""
    if dia is not None:
        scope_filter = f"WHERE e.dia = '{dia}'"
    elif batch is not None:
        scope_filter = f"WHERE e.batch_id = {batch.batch_id}"

    df = ctx.data.query(
        f"""
        SELECT e.*
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        {scope_filter}
        """
    )
    if df.empty:
        return

    cols_df = df.columns.tolist()

    # Ordenar el DataFrame para procesarlo secuencialmente
    # 1) Ordenar
    df = df.sort_values(
        by=["dia", "id_tarjeta", "id_viaje", "tiempo", "hora", "id_etapa"]
    ).reset_index(drop=True)

    # 2) Calcular la línea anterior dentro de cada grupo
    df["id_linea_anterior"] = df.groupby(["dia", "id_tarjeta", "id_viaje"])[
        "id_linea"
    ].shift()

    # 3) Crear columna booleana: True si es la misma línea que la anterior, False en caso contrario
    df["es_igual"] = df["id_linea"] == df["id_linea_anterior"]

    # 4) Hacer la suma acumulada dentro de cada grupo
    df["sum_id_viaje"] = df.groupby(["dia", "id_tarjeta"])["es_igual"].cumsum()

    df["id_viaje"] = df.id_viaje + df.sum_id_viaje
    df["id_etapa"] = df.groupby(["dia", "id_tarjeta", "id_viaje"]).cumcount() + 1

    # Borrar columnas auxiliares
    df.drop(columns=["es_igual", "id_linea_anterior", "sum_id_viaje"], inplace=True)

    # Corrige viajes con origen y destino iguales

    # Crear tabla temporal para detectar viajes con el mismo OD
    df_viajes = df.groupby(["dia", "id_tarjeta", "id_viaje"], as_index=False).agg(
        h3_o=("h3_o", "first"),  # Primer origen
        h3_d=("h3_d", "last"),  # Último destino
        od_validado=("od_validado", "min"),  # Validación mínima del viaje
        cant_etapas=("id_etapa", "count"),  # Número de etapas en el viaje
        id_etapa=("id_etapa", "last"),  # Última etapa del viaje
    )

    # Filtrar viajes con el mismo OD
    mask = (
        (df_viajes.h3_o == df_viajes.h3_d)
        & (df_viajes.od_validado == 1)
        & (df_viajes.cant_etapas > 1)
    )

    # Seleccionar solo las tarjetas con problemas
    df_viajes_problemas = df_viajes.loc[
        mask, ["dia", "id_tarjeta", "id_viaje", "id_etapa"]
    ]
    df_viajes_problemas["mismo_od"] = 1

    # Filtrar las etapas originales con problemas
    df = df.merge(
        df_viajes_problemas[["dia", "id_tarjeta", "id_viaje", "mismo_od"]],
        on=["dia", "id_tarjeta", "id_viaje"],
        how="left",
    )
    df["mismo_od"] = df["mismo_od"].fillna(0)

    df["con_problemas"] = df.groupby(["dia", "id_tarjeta"])["mismo_od"].transform("max")

    df_ok = df[df.con_problemas == 0].copy()
    df_viajes_problemas = df[df.con_problemas == 1].copy()

    # 3) Crear columna booleana: True si es la misma línea que la anterior, False en caso contrario
    df_viajes_problemas["es_igual"] = 0
    df_viajes_problemas.loc[
        (df_viajes_problemas.mismo_od == 1) & (df_viajes_problemas.id_etapa != 1),
        "es_igual",
    ] = 1

    # 4) Hacer la suma acumulada dentro de cada grupo
    df_viajes_problemas["sum_id_viaje"] = df_viajes_problemas.groupby(
        ["dia", "id_tarjeta"]
    )["es_igual"].cumsum()

    df_viajes_problemas["id_viaje"] = (
        df_viajes_problemas.id_viaje + df_viajes_problemas.sum_id_viaje
    )
    df_viajes_problemas["id_etapa"] = (
        df_viajes_problemas.groupby(["dia", "id_tarjeta", "id_viaje"]).cumcount() + 1
    )

    df = pd.concat([df_ok, df_viajes_problemas], ignore_index=True)
    df = df.sort_values(by=["dia", "id_tarjeta", "id_viaje", "id_etapa"]).reset_index(
        drop=True
    )

    # Borrar columnas auxiliares
    df = df[cols_df]

    ctx.data.update_leg_trip_ids(df, dia=dia)


@duracion
def compute_trips_travel_time(ctx: StorageContext):
    """
    This function reads from legs travel time in gps and stations
    and computes travel times for trips
    """

    ctx.data.execute(
        """
        INSERT INTO travel_times_legs (dia, id, id_tarjeta, id_etapa, id_viaje, travel_time_min)
        SELECT e.dia, e.id, e.id_tarjeta, e.id_etapa, e.id_viaje,
        (COALESCE(tg.travel_time_min, 0) + COALESCE(ts.travel_time_min, 0)) AS tt
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        LEFT JOIN travel_times_gps tg ON e.id = tg.id
        LEFT JOIN travel_times_stations ts ON e.id = ts.id
        WHERE e.od_validado = 1
        AND (tg.travel_time_min IS NOT NULL OR ts.travel_time_min IS NOT NULL)
        """
    )

    ctx.data.execute(
        """
        INSERT INTO travel_times_trips (dia, id_tarjeta, id_viaje, travel_time_min)
        SELECT tt.dia, tt.id_tarjeta, tt.id_viaje, SUM(tt.travel_time_min) AS travel_time_min
        FROM travel_times_legs tt
        JOIN dias_ultima_corrida d ON tt.dia = d.dia
        GROUP BY tt.dia, tt.id_tarjeta, tt.id_viaje
        """
    )

# NOTA: la antigua `add_distance_and_travel_time` (viajes) fue ELIMINADA
# (2026-07-17). Recalculaba con compute_od_distances la distancia OD directa del
# viaje y la escribía en viajes.distancia — pero (a) esa métrica no es la que se
# quiere (el diseño usa la distancia de `travel_times_trips`, producida por
# assign_time_distances, que ya corre en Fase 3) y (b) su UPDATE global levantaba
# el mes entero (39+ GB). Los consumidores (persist_indicators) leen ahora
# `travel_times_trips` directamente, igual que el dashboard (preparo_dashboard) y
# que el pipeline original, donde esta función nunca se llamaba.
