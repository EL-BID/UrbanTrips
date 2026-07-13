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
    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, COUNT(*) AS indicador
            FROM transacciones
            GROUP BY dia
            """,
            "Registros en transacciones",
            "transacciones",
            0,
        )
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, SUM(factor_expansion) AS indicador
            FROM transacciones
            WHERE factor_expansion IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad de transacciones totales",
            "transacciones",
            0,
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

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, SUM(factor_expansion_tarjeta) AS indicador
            FROM (
                SELECT dia, id_tarjeta, MAX(factor_expansion_tarjeta) AS factor_expansion_tarjeta
                FROM etapas
                WHERE od_validado = 1
                GROUP BY dia, id_tarjeta
            )
            WHERE factor_expansion_tarjeta IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad de tarjetas finales",
            "usuarios",
            0,
        )
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, SUM(factor_expansion_linea) AS indicador
            FROM (
                SELECT dia, id_tarjeta, MIN(factor_expansion_linea) AS factor_expansion_linea
                FROM etapas
                WHERE od_validado = 1
                GROUP BY dia, id_tarjeta
            )
            WHERE factor_expansion_linea IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad total de tarjetas",
            "usuarios expandidos",
            0,
        )
    )

    # VIAJES
    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, COUNT(*) AS indicador
            FROM viajes
            WHERE od_validado = 1
            GROUP BY dia
            """,
            "Cantidad de registros en viajes",
            "viajes",
            0,
        )
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, SUM(factor_expansion_linea) AS indicador
            FROM viajes
            WHERE od_validado = 1
              AND factor_expansion_linea IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad total de viajes expandidos",
            "viajes expandidos",
            0,
        )
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, SUM(factor_expansion_linea) AS indicador
            FROM viajes
            WHERE od_validado = 1
              AND distancia <= 5
              AND factor_expansion_linea IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad de viajes cortos (<5kms)",
            "viajes expandidos",
            1,
        )
    )

    indicator_rows.append(
        _indicator_query(
            ctx,
            """
            SELECT dia, SUM(factor_expansion_linea) AS indicador
            FROM viajes
            WHERE od_validado = 1
              AND cant_etapas > 1
              AND factor_expansion_linea IS NOT NULL
            GROUP BY dia
            """,
            "Cantidad de viajes con transferencia",
            "viajes expandidos",
            1,
        )
    )

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
                dia,
                SUM(distancia * factor_expansion_linea) / SUM(factor_expansion_linea) AS indicador
            FROM viajes
            WHERE od_validado = 1
              AND distancia IS NOT NULL
            GROUP BY dia
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
    _dias_df = ctx.data.query(
        "SELECT DISTINCT dia FROM viajes "
        "WHERE od_validado = 1 AND distancia IS NOT NULL"
    )
    dias = sorted(_dias_df["dia"].tolist())
    _median_total = []
    _median_by_mode = []
    for _dia in dias:
        viajes_median = ctx.data.query(
            f"""
            SELECT dia, modo, distancia AS distance_od, factor_expansion_linea
            FROM viajes
            WHERE od_validado = 1
              AND distancia IS NOT NULL
              AND dia = '{_dia}'
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
            dia,
            modo,
            SUM(distancia * factor_expansion_linea) / SUM(factor_expansion_linea) AS indicador
        FROM viajes
        WHERE od_validado = 1
          AND modo IS NOT NULL
          AND distancia IS NOT NULL
        GROUP BY dia, modo
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
