import logging
import multiprocessing
import time
from datetime import datetime

import numpy as np
import pandas as pd

from urbantrips.utils.utils import duracion
from urbantrips.storage.context import StorageContext
from urbantrips.storage.ports import BatchSpec
from urbantrips.carto.compute_distances import compute_od_distances

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

    def run_step(label: str, sql: str) -> None:
        logger.debug("  - %s...", label)
        start = time.perf_counter()
        ctx.data.execute(sql)
        logger.debug("    listo en %.2fs", time.perf_counter() - start)

    logger.info("Calculando factores de expansión por etapa, línea y tarjeta")
    run_step(
        "Calculando factores en tabla temporal",
        f"""
        CREATE OR REPLACE TABLE _ut_etapas_fex AS
        WITH base AS (
            SELECT
                e.*,
                CASE
                    WHEN e.latitud = 0 AND e.longitud = 0 THEN 0
                    ELSE e.od_validado
                END AS od_base
            FROM etapas e
            WHERE e.dia IN ({dias_str})
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
        "Reemplazando etapas con factores calculados",
        f"""
        BEGIN TRANSACTION;
        DELETE FROM etapas WHERE dia IN ({dias_str});
        INSERT INTO etapas (
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
        FROM _ut_etapas_fex;
        COMMIT;
        """,
    )

    n_etapas = ctx.data.query(
        f"SELECT COUNT(*) AS n FROM etapas WHERE dia IN ({dias_str})"
    )["n"].iloc[0]
    logger.info("Creando tabla de viajes de %d etapas", n_etapas)

    run_step("Borrando viajes previos", f"DELETE FROM viajes WHERE dia IN ({dias_str})")
    run_step(
        "Insertando viajes",
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
                ARG_MIN(dia, id) AS dia,
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
            FROM etapas
            WHERE dia IN ({dias_str})
            GROUP BY id_tarjeta, id_viaje
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

    run_step("Borrando usuarios previos", f"DELETE FROM usuarios WHERE dia IN ({dias_str})")
    run_step(
        "Insertando usuarios",
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
        WHERE dia IN ({dias_str})
        GROUP BY dia, id_tarjeta
        """,
    )

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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# def _delete_dias(conn, table, dias_ultima_corrida):
#     """Delete rows for given days using parameterized query."""
#     dias = dias_ultima_corrida["dia"].tolist()
#     placeholders = ", ".join("?" * len(dias))
#     conn.execute(f"DELETE FROM {table} WHERE dia IN ({placeholders})", dias)
#     conn.commit()


# def _upload_chunked(df, table, conn, chunk_size=500_000):
#     """Upload large dataframes to SQLite efficiently."""
#     print(f"Subiendo datos a {table} ({len(df)} registros) - {str(datetime.now())[:19]}")

#     # Pragmas de performance
#     conn.execute("PRAGMA journal_mode = WAL")
#     conn.execute("PRAGMA synchronous = OFF")
#     conn.execute("PRAGMA cache_size = -2000000")  # 2GB cache

#     cols = df.columns.tolist()
#     placeholders = ", ".join("?" * len(cols))
#     col_names = ", ".join(cols)
#     sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"

#     cursor = conn.cursor()
#     cursor.execute("BEGIN TRANSACTION")
#     try:
#         for i in range(0, len(df), chunk_size):
#             chunk = df.iloc[i: i + chunk_size]
#             cursor.executemany(sql, chunk.values.tolist())
#             print(f"  {min(i + chunk_size, len(df)):,} / {len(df):,} registros...")
#         conn.commit()
#     except Exception:
#         conn.rollback()
#         raise
#     finally:
#         # Restaurar pragmas seguros
#         conn.execute("PRAGMA synchronous = FULL")

    


@duracion
def rearrange_trip_id_same_od(ctx: StorageContext, batch: BatchSpec | None = None):
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

    """
    dias_ultima_corrida = ctx.data.get_run_days()

    batch_filter = ""
    if batch is not None:
        batch_filter = f"WHERE e.batch_id = {batch.batch_id}"

    df = ctx.data.query(
        f"""
        SELECT e.*
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        {batch_filter}
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
    df["id_linea_anterior"] = df.groupby(["id_tarjeta", "id_viaje"])[
        "id_linea"
    ].shift()

    # 3) Crear columna booleana: True si es la misma línea que la anterior, False en caso contrario
    df["es_igual"] = df["id_linea"] == df["id_linea_anterior"]

    # 4) Hacer la suma acumulada dentro de cada grupo
    df["sum_id_viaje"] = df.groupby(["dia", "id_tarjeta"])["es_igual"].cumsum()

    df["id_viaje"] = df.id_viaje + df.sum_id_viaje
    df["id_etapa"] = df.groupby(["id_tarjeta", "id_viaje"]).cumcount() + 1

    # Borrar columnas auxiliares
    df.drop(columns=["es_igual", "id_linea_anterior", "sum_id_viaje"], inplace=True)

    # Corrige viajes con origen y destino iguales

    # Crear tabla temporal para detectar viajes con el mismo OD
    df_viajes = df.groupby(["id_tarjeta", "id_viaje"], as_index=False).agg(
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
        mask, ["id_tarjeta", "id_viaje", "id_etapa"]
    ]
    df_viajes_problemas["mismo_od"] = 1

    # Filtrar las etapas originales con problemas
    df = df.merge(
        df_viajes_problemas[["id_tarjeta", "id_viaje", "mismo_od"]],
        on=["id_tarjeta", "id_viaje"],
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
        df_viajes_problemas.groupby(["id_tarjeta", "id_viaje"]).cumcount() + 1
    )

    df = pd.concat([df_ok, df_viajes_problemas], ignore_index=True)
    df = df.sort_values(by=["dia", "id_tarjeta", "id_viaje", "id_etapa"]).reset_index(
        drop=True
    )

    # Borrar columnas auxiliares
    df = df[cols_df]

    if batch is None:
        dias_str = ", ".join(f"'{d}'" for d in dias_ultima_corrida["dia"].tolist())
        ctx.data.execute(f"DELETE FROM etapas WHERE dia IN ({dias_str})")
    ctx.data.save_legs(df, batch=batch)


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

@duracion
def add_distance_and_travel_time(ctx: StorageContext):
    """
    This function reads trips data and adds distances and travel times
    from the distances table. It also computes the travel speed.
    """

    trips = ctx.data.query(
        """
        SELECT v.id_tarjeta, v.id_viaje, v.dia, v.h3_d, v.h3_o
        FROM viajes v
        JOIN dias_ultima_corrida d ON v.dia = d.dia
        WHERE od_validado = 1
        """
    )

    trips = compute_od_distances(
        od_df=trips,
        origin_col="h3_o",
        dest_col="h3_d",
        distance_col="distance",
        unit="km",
        db_path="data/matriz_distancia/matriz_distancia.duckdb",
        network_cache_dir="data/matriz_distancia",
        symmetric=False,
        precompute_dist=50_000,
        max_tile_deg=99,
        verbose=False,
    )

    ctx.data.save_raw(trips, "temp_distancias")

    ctx.data.execute(
        """
        UPDATE viajes
        SET distancia = temp_distancias.distance
        FROM temp_distancias
        WHERE viajes.id_tarjeta = temp_distancias.id_tarjeta
        AND viajes.id_viaje = temp_distancias.id_viaje
        AND viajes.dia = temp_distancias.dia
        """
    )

    ctx.data.execute(
        """
        UPDATE viajes
        SET travel_time_min = t.travel_time_min
        FROM (
            SELECT dia, id_tarjeta, id_viaje,
                   SUM(COALESCE(travel_time_min, 0)) AS travel_time_min
            FROM travel_times_legs
            GROUP BY dia, id_tarjeta, id_viaje
        ) t
        WHERE viajes.dia = t.dia
        AND viajes.id_tarjeta = t.id_tarjeta
        AND viajes.id_viaje = t.id_viaje
        """
    )

    ctx.data.execute("DROP TABLE IF EXISTS temp_distancias")
