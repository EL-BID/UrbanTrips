"""Regresión: en el pipeline (line_ids None), process_services acota borrado y lectura
a run_days. Antes borraba TODO services y re-clasificaba el GPS histórico congelado en
cada corrida (3.4×, no re-procesable). El path interactivo (line_ids) queda igual.

importorskip: services.py arrastra geopandas al importar.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock

services = pytest.importorskip("urbantrips.datamodel.services")


def test_delete_old_services_data_scopes_to_run_days_in_pipeline():
    ctx = MagicMock()
    services.delete_old_services_data(ctx, None, run_days=["2024-01-01", "2024-01-02"])
    sqls = [c[0][0] for c in ctx.data.execute.call_args_list]
    assert len(sqls) == 3  # services_gps_points, services, services_stats
    for s in sqls:
        assert "WHERE dia IN" in s
        assert "'2024-01-01'" in s and "'2024-01-02'" in s
        assert s.strip() not in (  # NO wipe total
            "DELETE FROM services", "DELETE FROM services_stats",
            "DELETE FROM services_gps_points",
        )


def test_delete_old_services_data_by_line_ignores_run_days():
    ctx = MagicMock()
    services.delete_old_services_data(ctx, "10,20", run_days=["2024-01-01"])
    for c in ctx.data.execute.call_args_list:
        assert "id_linea IN (10,20)" in c[0][0]  # interactivo por línea, no por día


def test_get_stops_and_gps_data_pipeline_reads_only_run_days(monkeypatch):
    ctx = MagicMock()
    captured = {}

    def fake_query(q):
        if "gps_exists" in q:
            return pd.DataFrame({"gps_exists": [1]})
        captured["q"] = q
        return pd.DataFrame()  # gps_points vacío → early return, no toca geopandas

    ctx.data.query.side_effect = fake_query
    monkeypatch.setattr(
        services.utils, "leer_configs_generales",
        lambda *a, **k: {"utilizar_servicios_gps": True, "epsg_m": 5344},
    )

    services.get_stops_and_gps_data(ctx, None, run_days=["2024-01-01", "2024-01-02"])

    q = captured["q"]
    assert "g.dia IN" in q
    assert "'2024-01-01'" in q and "'2024-01-02'" in q
    assert "services_stats" not in q  # sin anti-join en el pipeline
