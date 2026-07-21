import pandas as pd
import numpy as np
import weightedstats as ws
from urbantrips.utils.utils import (
    duracion,
)
from urbantrips.storage.context import StorageContext


def _replace_indicator_rows(ctx: StorageContext, resultado: pd.DataFrame) -> None:
    if resultado.empty:
        return

    resultado = resultado.copy()
    resultado = resultado[resultado.indicador.notna()]
    if resultado.empty:
        return

    resultado["indicador"] = resultado["indicador"].round(2)
    resultado = resultado[["dia", "detalle", "indicador", "tabla", "nivel"]]

    indicadores = ctx.data.get_indicators()
    if not indicadores.empty:
        new_keys = resultado[["dia", "detalle", "tabla"]].drop_duplicates()
        indicadores = indicadores.merge(
            new_keys.assign(_replace=1),
            on=["dia", "detalle", "tabla"],
            how="left",
        )
        indicadores = indicadores[indicadores["_replace"].isna()].drop(
            columns=["_replace"]
        )

    indicadores = pd.concat([indicadores, resultado], ignore_index=True)
    if "porcentaje" not in indicadores.columns:
        indicadores["porcentaje"] = np.nan

    for tabla, nivel in (
        resultado.loc[resultado.nivel > 0, ["tabla", "nivel"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        for dia in indicadores.loc[
            (indicadores.tabla == tabla) & (indicadores.nivel == nivel),
            "dia",
        ].unique():
            denominator = indicadores.loc[
                (indicadores.tabla == tabla)
                & (indicadores.nivel == nivel - 1)
                & (indicadores.dia == dia),
                "indicador",
            ]
            if denominator.empty:
                continue
            mask = (
                (indicadores.tabla == tabla)
                & (indicadores.nivel == nivel)
                & (indicadores.dia == dia)
            )
            indicadores.loc[mask, "porcentaje"] = (
                indicadores.loc[mask, "indicador"] / denominator.iloc[0] * 100
            ).round(1)

    indicadores.fillna(0, inplace=True)
    ctx.data.save_indicators(indicadores)


def _indicator_query(
    ctx: StorageContext,
    sql: str,
    detalle: str,
    tabla: str,
    nivel: int,
) -> pd.DataFrame:
    df = ctx.data.query(sql)
    if df.empty:
        return pd.DataFrame(columns=["dia", "detalle", "indicador", "tabla", "nivel"])
    df["detalle"] = detalle
    df["tabla"] = tabla
    df["nivel"] = nivel
    return df[["dia", "detalle", "indicador", "tabla", "nivel"]]


def _fused_indicators(ctx: StorageContext, sql: str, specs: list) -> list:
    """Ejecuta UNA query (dia + varias columnas agregadas) y la expande en varias
    filas de indicadores. Reduce scans: N indicadores con el MISMO FROM/WHERE/GROUP BY
    dia se computan en un solo scan de la tabla en vez de N. Bit-idéntico a las queries
    separadas — SUM ignora NULLs igual que `WHERE col IS NOT NULL`, y `CASE WHEN cond`
    replica un `WHERE cond` extra. specs = [(col, detalle, tabla, nivel), ...]."""
    df = ctx.data.query(sql)
    out = []
    empty = pd.DataFrame(columns=["dia", "detalle", "indicador", "tabla", "nivel"])
    for col, detalle, tabla, nivel in specs:
        if df.empty or col not in df.columns:
            out.append(empty.copy())
            continue
        d = df[["dia", col]].rename(columns={col: "indicador"}).copy()
        d["detalle"] = detalle
        d["tabla"] = tabla
        d["nivel"] = nivel
        out.append(d[["dia", "detalle", "indicador", "tabla", "nivel"]])
    return out


def _weighted_median_rows(
    df: pd.DataFrame,
    detalle_prefix: str,
    tabla: str,
    by_mode: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dia", "detalle", "indicador", "tabla", "nivel"])

    group_cols = ["dia", "modo"] if by_mode else ["dia"]
    rows = []
    for group_key, group in df.groupby(group_cols):
        if by_mode:
            dia, modo = group_key
            detalle = f"{detalle_prefix} - {modo}"
        else:
            dia = group_key[0] if isinstance(group_key, tuple) else group_key
            detalle = detalle_prefix
        rows.append(
            {
                "dia": dia,
                "detalle": detalle,
                "indicador": ws.weighted_median(
                    group["distance_od"].tolist(),
                    weights=group["factor_expansion_linea"].tolist(),
                ),
                "tabla": tabla,
                "nivel": 0,
            }
        )
    return pd.DataFrame(rows)


@duracion
def persist_indicators(ctx: StorageContext):
    """
    Esta funcion crea tabla de indicatores clave
    """

    indicator_rows = []

    # TRANSACCIONES
    # FUSIÓN (1 scan de transacciones en vez de 2). COUNT(*) = registros; SUM ignora
    # NULLs solo → idéntico al WHERE factor_expansion IS NOT NULL de la query separada.
    indicator_rows.extend(
        _fused_indicators(
            ctx,
            """
            SELECT dia,
                   COUNT(*) AS registros,
                   SUM(factor_expansion) AS totales
            FROM transacciones
            GROUP BY dia
            """,
            [
                ("registros", "Registros en transacciones", "transacciones", 0),
                ("totales", "Cantidad de transacciones totales", "transacciones", 0),
            ],
        )
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, SUM(factor_expansion_linea) AS indicador
            FROM etapas
            WHERE od_validado = 1
              AND factor_expansion_linea IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad total de etapas",
            "etapas_expandidas",
            0,
        )
    )

    etapas_modo = ctx.data.query(
        """
        SELECT dia, modo, SUM(factor_expansion_linea) AS indicador
        FROM etapas
        WHERE od_validado = 1
          AND modo IS NOT NULL
          AND factor_expansion_linea IS NOT NULL
        GROUP BY dia, modo
        """
    )
    if not etapas_modo.empty:
        etapas_modo["detalle"] = "Etapas " + etapas_modo["modo"].astype(str)
        etapas_modo["tabla"] = "etapas_expandidas"
        etapas_modo["nivel"] = 1
        indicator_rows.append(
            etapas_modo[["dia", "detalle", "indicador", "tabla", "nivel"]]
        )

    # FUSIÓN (1 scan de etapas en vez de 2): las dos comparten el GROUP BY dia,id_tarjeta
    # sobre etapas WHERE od_validado=1 → MAX(fex_tarjeta) y MIN(fex_linea) en una pasada.
    # SUM ignora NULLs → idéntico a los WHERE ... IS NOT NULL de las queries separadas.
    indicator_rows.extend(
        _fused_indicators(
            ctx,
            """
            SELECT dia,
                   SUM(fex_tarjeta) AS tarjetas_finales,
                   SUM(fex_linea) AS tarjetas_totales
            FROM (
                SELECT dia, id_tarjeta,
                       MAX(factor_expansion_tarjeta) AS fex_tarjeta,
                       MIN(factor_expansion_linea) AS fex_linea
                FROM etapas
                WHERE od_validado = 1
                GROUP BY dia, id_tarjeta
            )
            GROUP BY dia
            """,
            [
                ("tarjetas_finales", "Cantidad de tarjetas finales", "usuarios", 0),
                ("tarjetas_totales", "Cantidad total de tarjetas", "usuarios expandidos", 0),
            ],
        )
    )

    # VIAJES
    # FUSIÓN (1 scan de viajes en vez de 3; incluye la de transferencia de más abajo).
    # Todas comparten WHERE od_validado=1 GROUP BY dia. COUNT(*)=registros; SUM(fex) ignora
    # NULLs=expandidos; SUM(CASE WHEN cant_etapas>1) replica el WHERE cant_etapas>1.
    indicator_rows.extend(
        _fused_indicators(
            ctx,
            """
            SELECT dia,
                   COUNT(*) AS registros,
                   SUM(factor_expansion_linea) AS expandidos,
                   SUM(CASE WHEN cant_etapas > 1 THEN factor_expansion_linea END) AS transferencia
            FROM viajes
            WHERE od_validado = 1
            GROUP BY dia
            """,
            [
                ("registros", "Cantidad de registros en viajes", "viajes", 0),
                ("expandidos", "Cantidad total de viajes expandidos", "viajes expandidos", 0),
                ("transferencia", "Cantidad de viajes con transferencia", "viajes expandidos", 1),
            ],
        )
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT v.dia, SUM(v.factor_expansion_linea) AS indicador
            FROM viajes v
            JOIN travel_times_trips tt
              ON v.dia = tt.dia
             AND v.id_tarjeta = tt.id_tarjeta
             AND v.id_viaje = tt.id_viaje
            WHERE v.od_validado = 1
              AND tt.distance_od <= 5
              AND v.factor_expansion_linea IS NOT NULL
            GROUP BY v.dia
            """,
            "Cantidad de viajes cortos (<5kms)",
            "viajes expandidos",
            1,
        )
    )

    # (la "Cantidad de viajes con transferencia" se computa en la fusión de viajes de arriba)

    viajes_modo = ctx.data.query(
        """
        SELECT dia, modo, SUM(factor_expansion_linea) AS indicador
        FROM viajes
        WHERE od_validado = 1
          AND modo IS NOT NULL
          AND factor_expansion_linea IS NOT NULL
        GROUP BY dia, modo
        """
    )
    if not viajes_modo.empty:
        viajes_modo["detalle"] = "Viajes " + viajes_modo["modo"].astype(str)
        viajes_modo["tabla"] = "modos viajes"
        viajes_modo["nivel"] = 0
        indicator_rows.append(
            viajes_modo[["dia", "detalle", "indicador", "tabla", "nivel"]]
        )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT
                v.dia,
                SUM(tt.distance_od * v.factor_expansion_linea) / SUM(v.factor_expansion_linea) AS indicador
            FROM viajes v
            JOIN travel_times_trips tt
              ON v.dia = tt.dia
             AND v.id_tarjeta = tt.id_tarjeta
             AND v.id_viaje = tt.id_viaje
            WHERE v.od_validado = 1
              AND tt.distance_od IS NOT NULL
            GROUP BY v.dia
            """,
            "Distancia de los viajes (promedio en kms)",
            "avg",
            0,
        )
    )

    # Mediana ponderada de distancia de viajes: se computa DÍA POR DÍA para no
    # levantar toda la tabla `viajes` del mes a pandas (weightedstats no tiene
    # equivalente SQL). Es separable: _weighted_median_rows agrupa por dia (o
    # dia,modo) → la mediana nunca cruza días, resultado idéntico. Los días se
    # derivan de la MISMA población que la query original (todos los viajes válidos
    # con distancia), no de dias_ultima_corrida, para preservar el alcance exacto.
    # La distancia OD del viaje sale de travel_times_trips (Fase 3), no de una
    # columna en viajes.
    _dias_df = ctx.data.query(
        "SELECT DISTINCT v.dia FROM viajes v "
        "JOIN travel_times_trips tt "
        "ON v.dia = tt.dia AND v.id_tarjeta = tt.id_tarjeta AND v.id_viaje = tt.id_viaje "
        "WHERE v.od_validado = 1 AND tt.distance_od IS NOT NULL"
    )
    dias = sorted(_dias_df["dia"].tolist())
    _median_total = []
    _median_by_mode = []
    for _dia in dias:
        viajes_median = ctx.data.query(
            f"""
            SELECT v.dia, v.modo, tt.distance_od AS distance_od, v.factor_expansion_linea
            FROM viajes v
            JOIN travel_times_trips tt
              ON v.dia = tt.dia
             AND v.id_tarjeta = tt.id_tarjeta
             AND v.id_viaje = tt.id_viaje
            WHERE v.od_validado = 1
              AND tt.distance_od IS NOT NULL
              AND v.dia = '{_dia}'
            """
        )
        if viajes_median.empty:
            continue
        _median_total.append(
            _weighted_median_rows(
                viajes_median,
                "Distancia de los viajes (mediana en kms)",
                "avg",
            )
        )
        _median_by_mode.append(
            _weighted_median_rows(
                viajes_median[viajes_median["modo"].notna()].copy(),
                "Distancia de los viajes (mediana en kms)",
                "avg",
                by_mode=True,
            )
        )
        del viajes_median

    _empty_median = pd.DataFrame(
        columns=["dia", "detalle", "indicador", "tabla", "nivel"]
    )
    indicator_rows.append(
        pd.concat(_median_total, ignore_index=True) if _median_total else _empty_median
    )

    viajes_modo_mean = ctx.data.query(
        """
        SELECT
            v.dia,
            v.modo,
            SUM(tt.distance_od * v.factor_expansion_linea) / SUM(v.factor_expansion_linea) AS indicador
        FROM viajes v
        JOIN travel_times_trips tt
          ON v.dia = tt.dia
         AND v.id_tarjeta = tt.id_tarjeta
         AND v.id_viaje = tt.id_viaje
        WHERE v.od_validado = 1
          AND v.modo IS NOT NULL
          AND tt.distance_od IS NOT NULL
        GROUP BY v.dia, v.modo
        """
    )
    if not viajes_modo_mean.empty:
        viajes_modo_mean["detalle"] = (
            "Distancia de los viajes (promedio en kms) - "
            + viajes_modo_mean["modo"].astype(str)
        )
        viajes_modo_mean["tabla"] = "avg"
        viajes_modo_mean["nivel"] = 0
        indicator_rows.append(
            viajes_modo_mean[["dia", "detalle", "indicador", "tabla", "nivel"]]
        )

    indicator_rows.append(
        pd.concat(_median_by_mode, ignore_index=True) if _median_by_mode else _empty_median
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT
                dia,
                SUM(cant_etapas * factor_expansion_linea) / SUM(factor_expansion_linea) AS indicador
            FROM viajes
            WHERE od_validado = 1
              AND cant_etapas IS NOT NULL
            GROUP BY dia
            """,
            "Etapas promedio de los viajes",
            "avg",
            0,
        )
    )

    # USUARIOS
    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT
                dia,
                SUM(cant_viajes * factor_expansion_linea) / SUM(factor_expansion_linea) AS indicador
            FROM usuarios
            WHERE od_validado = 1
              AND cant_viajes IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad promedio de viajes por tarjeta",
            "avg",
            0,
        )
    )

    _replace_indicator_rows(
        ctx,
        pd.concat(indicator_rows, ignore_index=True)
        if indicator_rows
        else pd.DataFrame(columns=["dia", "detalle", "indicador", "tabla", "nivel"]),
    )
