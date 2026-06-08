# urbantrips/tests/e2e/test_storage_context.py
import pytest
import pandas as pd
from pathlib import Path


def test_storage_context_holds_all_ports(in_memory_ctx):
    from urbantrips.storage.ports import DataPort, InsumoPort, DashPort, GeneralPort
    assert isinstance(in_memory_ctx.data, DataPort)
    assert isinstance(in_memory_ctx.insumos, InsumoPort)
    assert isinstance(in_memory_ctx.dash, DashPort)
    assert isinstance(in_memory_ctx.general, GeneralPort)


def test_build_storage_context_duckdb(tmp_path):
    from urbantrips.storage.context import build_storage_context
    from urbantrips.config.config import Config
    from urbantrips.storage.ports import DataPort

    config = Config(
        alias_db="test",
        alias_db_insumos="test",
        alias_db_dashboard="test",
        corridas=["run01"],
        geolocalizar_trx=False,
        nombre_archivo_trx="f.csv",
        nombre_archivo_gps=None,
        nombres_variables_trx={},
        formato_fecha="%Y-%m-%d",
        columna_hora="tiempo",
        tipo_trx_invalidas=None,
        tolerancia_parada_destino=300,
        resolucion_h3=8,
        ordenamiento_transacciones="tiempo",
        ventana_viajes=90,
        ventana_duplicado=5,
        tiempos_viaje_estaciones=None,
        storage_backend="duckdb",
        n_batches=1,
    )
    ctx = build_storage_context(config, base_dir=tmp_path)
    assert isinstance(ctx.data, DataPort)


def test_context_data_flows_through_adapters(in_memory_ctx):
    legs = pd.DataFrame({
        "id": [1], "id_tarjeta": ["T1"], "dia": ["2024-01-01"],
    })
    in_memory_ctx.data.save_legs(legs)
    result = in_memory_ctx.data.get_legs()
    assert len(result) == 1
