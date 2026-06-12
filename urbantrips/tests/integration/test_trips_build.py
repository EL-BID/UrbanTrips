# urbantrips/tests/integration/test_trips_build.py
"""Regresión: una tarjeta con el mismo id_viaje en días distintos debe
producir un viaje por día (id_viaje arranca en 1 por tarjeta cada día;
sin dia en el GROUP BY los viajes de ambos días se fundían en uno solo,
asignado al primer día con las etapas sumadas)."""
from types import SimpleNamespace

import pandas as pd


def _legs_two_days_same_trip_id() -> pd.DataFrame:
    # tarjeta T001: viaje 1 con 2 etapas el día 1 y 3 etapas el día 2
    rows = []
    next_id = 1
    for dia, n_etapas in [("2024-09-17", 2), ("2024-09-18", 3)]:
        for id_etapa in range(1, n_etapas + 1):
            rows.append({
                "id": next_id,
                "id_tarjeta": "T001",
                "dia": dia,
                "id_viaje": 1,
                "id_etapa": id_etapa,
                "tiempo": f"08:{id_etapa:02d}",
                "hora": 8,
                "modo": "autobus",
                "id_linea": 10 + id_etapa,
                "id_ramal": 1,
                "interno": 100,
                "genero": None,
                "tarifa": None,
                "latitud": -34.6,
                "longitud": -58.4,
                "h3_o": "88c2e312d9fffff",
                "h3_d": "88c2e312d1fffff",
                "od_validado": 1,
                "etapa_validada": 1,
                "factor_expansion_original": 1.0,
            })
            next_id += 1
    return pd.DataFrame(rows)


def test_viajes_no_se_funden_entre_dias(tmp_path):
    from urbantrips.datamodel.trips import (
        create_trips_from_legs_and_fex,
        verificar_integridad_viajes_etapas,
    )
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter.save_run_days(pd.DataFrame({"dia": ["2024-09-17", "2024-09-18"]}))
    adapter.save_legs(_legs_two_days_same_trip_id())
    ctx = SimpleNamespace(data=adapter)

    create_trips_from_legs_and_fex(ctx)

    viajes = adapter.query("SELECT * FROM viajes ORDER BY dia")
    assert len(viajes) == 2, f"esperaba un viaje por día, hay {len(viajes)}"
    assert viajes["dia"].tolist() == ["2024-09-17", "2024-09-18"]
    assert viajes["cant_etapas"].tolist() == [2, 3]

    # el check de integridad del pipeline debe pasar
    diff = verificar_integridad_viajes_etapas(ctx)
    assert diff.empty
