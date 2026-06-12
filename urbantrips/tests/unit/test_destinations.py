import pandas as pd
from types import SimpleNamespace

from urbantrips.destinations.destinations import (
    calcular_indicadores_destinos_etapas,
    imputar_destino_potencial,
)


def test_destinos_potenciales(df_etapas):
    def check(d):
        primer_origen = d.h3_o.iloc[[0]]
        origenes_sig = d.h3_o.iloc[1:]
        destinos = pd.concat([origenes_sig, primer_origen]).values
        return all(d.h3_d.values == destinos)

    result = imputar_destino_potencial(df_etapas)
    assert result.groupby("id_tarjeta").apply(check).all()


def test_calcular_indicadores_destinos_uses_storage_context(monkeypatch):
    from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter

    monkeypatch.setattr(
        "urbantrips.utils.utils.iniciar_conexion_db",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy DB connection should not be used")
        ),
    )

    ctx = SimpleNamespace(data=InMemoryDataAdapter())
    etapas = pd.DataFrame(
        {
            "dia": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "od_validado": [1, 0, 1],
        }
    )

    calcular_indicadores_destinos_etapas(etapas, ctx)

    indicadores = ctx.data.get_indicators()
    assert set(indicadores["dia"]) == {"2024-01-01", "2024-01-02"}
    assert set(indicadores["detalle"]) == {"Cantidad de etapas con destinos validados"}
    assert indicadores.set_index("dia").loc["2024-01-01", "indicador"] == 1
