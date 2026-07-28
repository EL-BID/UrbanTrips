import pandas as pd
import weightedstats as ws


def _ctx(tmp_path):
    from urbantrips.storage.context import StorageContext
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.adapters.memory.adapters import (
        InMemoryDashAdapter,
        InMemoryGeneralAdapter,
        InMemoryInsumoAdapter,
    )

    return StorageContext(
        data=DuckDBDataAdapter(tmp_path / "data.duckdb"),
        insumos=InMemoryInsumoAdapter(),
        dash=InMemoryDashAdapter(),
        general=InMemoryGeneralAdapter(),
    )


def test_persist_indicators_pushdown_outputs_expected_values(tmp_path):
    from urbantrips.datamodel.misc import persist_indicators

    ctx = _ctx(tmp_path)
    day = "2024-01-01"

    # persist_indicators acota sus agregados a dias_ultima_corrida (mismo patrón
    # que create_trips_from_legs_and_fex): sin días registrados no hay nada que
    # calcular y retorna temprano.
    ctx.data.save_run_days(pd.DataFrame({"dia": [day]}))

    ctx.data.save_legs(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "id_tarjeta": ["A", "A", "B"],
                "dia": [day, day, day],
                "id_viaje": [1, 1, 1],
                "id_etapa": [1, 2, 1],
                "tiempo": ["08:00:00", "08:10:00", "09:00:00"],
                "hora": [8, 8, 9],
                "modo": ["bus", "train", "bus"],
                "id_linea": [1, 2, 1],
                "id_ramal": [10, 20, 10],
                "interno": [100, 200, 100],
                "h3_o": ["a", "b", "c"],
                "h3_d": ["b", "c", "d"],
                "od_validado": [1, 1, 1],
                "factor_expansion_linea": [2.0, 3.0, 5.0],
                "factor_expansion_tarjeta": [2.0, 4.0, 5.0],
            }
        )
    )
    ctx.data.save_trips(
        pd.DataFrame(
            {
                "id_tarjeta": ["A", "B", "C"],
                "id_viaje": [1, 1, 1],
                "dia": [day, day, day],
                "tiempo": ["08:00:00", "09:00:00", "10:00:00"],
                "hora": [8, 9, 10],
                "cant_etapas": [1, 2, 3],
                "modo": ["bus", "train", "bus"],
                "od_validado": [1, 1, 1],
                "factor_expansion_linea": [2.0, 3.0, 5.0],
                "factor_expansion_tarjeta": [2.0, 3.0, 5.0],
            }
        )
    )
    ctx.data.save_users(
        pd.DataFrame(
            {
                "id_tarjeta": ["A", "B"],
                "dia": [day, day],
                "od_validado": [1, 1],
                "cant_viajes": [2.0, 4.0],
                "factor_expansion_linea": [2.0, 3.0],
                "factor_expansion_tarjeta": [2.0, 3.0],
            }
        )
    )
    # La distancia OD por viaje vive en travel_times_trips (la produce
    # assign_time_distances en Fase 3); persist_indicators la lee de ahí. La
    # columna viajes.distancia, que espejaba estos valores, ya no existe.
    ctx.data.append_raw(
        pd.DataFrame(
            {
                "dia": [day, day, day],
                "id_tarjeta": ["A", "B", "C"],
                "id_viaje": [1, 1, 1],
                "distance_od": [2.0, 8.0, 4.0],
                "travel_time_min": [10.0, 20.0, 30.0],
            }
        ),
        "travel_times_trips",
    )

    persist_indicators(ctx)

    indicadores = ctx.data.get_indicators()

    def value(detalle, tabla=None, column="indicador"):
        rows = indicadores[indicadores["detalle"] == detalle]
        if tabla is not None:
            rows = rows[rows["tabla"] == tabla]
        assert len(rows) == 1
        return rows.iloc[0][column]

    assert value("Cantidad total de etapas", "etapas_expandidas") == 10.0
    assert value("Etapas bus", "etapas_expandidas") == 7.0
    assert value("Etapas bus", "etapas_expandidas", "porcentaje") == 70.0
    assert value("Cantidad de tarjetas finales", "usuarios") == 9.0
    assert value("Cantidad total de tarjetas", "usuarios expandidos") == 7.0

    assert value("Cantidad de registros en viajes", "viajes") == 3.0
    assert value("Cantidad total de viajes expandidos", "viajes expandidos") == 10.0
    assert value("Cantidad de viajes cortos (<5kms)", "viajes expandidos") == 7.0
    assert value("Cantidad de viajes cortos (<5kms)", "viajes expandidos", "porcentaje") == 70.0
    assert value("Viajes bus", "modos viajes") == 7.0

    expected_median = round(
        ws.weighted_median([2.0, 8.0, 4.0], weights=[2.0, 3.0, 5.0]),
        2,
    )
    assert value("Distancia de los viajes (promedio en kms)", "avg") == 4.8
    assert value("Distancia de los viajes (mediana en kms)", "avg") == expected_median
    assert value("Distancia de los viajes (promedio en kms) - bus", "avg") == 3.43
    assert value("Etapas promedio de los viajes", "avg") == 2.3
    assert value("Cantidad promedio de viajes por tarjeta", "avg") == 3.2


def test_weighted_median_equivale_a_weightedstats():
    """La versión numpy debe dar lo mismo que weightedstats.weighted_median.

    Se reemplazó la implementación pura (Python: listas + sorted + while, ~8x más
    lenta sobre los ~5M de valores por día del pipeline). Este test fija la
    equivalencia, incluidos los bordes donde las dos ramas del algoritmo original
    difieren: empates, un peso que supera el midpoint, y acumulada que cae justo
    sobre el midpoint (promedia dos valores).
    """
    import numpy as np
    import weightedstats as ws

    from urbantrips.datamodel.misc import _weighted_median

    casos = [
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),          # impar simple
        ([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]),  # acumulada justo en midpoint
        ([5.0, 1.0, 3.0], [1.0, 10.0, 1.0]),          # un peso > midpoint
        ([2.0, 2.0, 2.0, 9.0], [1.0, 1.0, 1.0, 2.0]),  # empates en los datos
        ([1.0, 2.0], [0.0, 3.0]),                      # peso cero
        ([7.5], [4.0]),                                # un solo elemento
        ([3.0, 1.0, 2.0], [0.5, 0.25, 0.25]),          # pesos fraccionarios
    ]
    for data, weights in casos:
        esperado = ws.weighted_median(data, weights=list(weights))
        obtenido = _weighted_median(np.array(data), np.array(weights))
        assert obtenido == esperado, (
            f"data={data} weights={weights}: numpy={obtenido} vs pura={esperado}"
        )

    # aleatorio, para cubrir combinaciones que no se me ocurrieron
    rng = np.random.default_rng(20260722)
    for _ in range(200):
        n = int(rng.integers(1, 50))
        data = rng.normal(10, 5, n).round(3)
        weights = rng.random(n).round(3) * 10
        esperado = ws.weighted_median(data.tolist(), weights=weights.tolist())
        obtenido = _weighted_median(data, weights)
        assert obtenido == esperado, (
            f"n={n} data={data.tolist()} weights={weights.tolist()}: "
            f"numpy={obtenido} vs pura={esperado}"
        )
