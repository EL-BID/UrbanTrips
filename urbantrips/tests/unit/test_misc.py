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
                "distancia": [2.0, 8.0, 4.0],
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
