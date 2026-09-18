"""Regresión del upsert por corrida en compute_kpi (reemplaza el patrón processed_days).

Antes, run_basic_kpi / compute_speed / compute_dispatched usaban `dia NOT IN
(processed_days)` y nunca borraban sus salidas para los run-days → re-procesar un día
ya presente lo SALTEABA (filas stale) y re-leer la salida creciente daba O(n²). Ahora
se borra por corrida antes del loop con _delete_run_days_from.

importorskip: kpi.py arrastra osmnx/compute_od_distances al importar.
"""
import duckdb
import pandas as pd
import pytest
from unittest.mock import MagicMock

kpi = pytest.importorskip("urbantrips.kpi.kpi")


def test_delete_run_days_from_scopes_tolerates_missing_and_mirrors_dash():
    ctx = MagicMock()
    ctx.data.get_run_days.return_value = pd.DataFrame(
        {"dia": ["2024-01-01", "2024-01-02"]}
    )

    # tabla aún no creada (corrida fresca) → CatalogException tolerada (no propaga)
    ctx.data.execute.side_effect = duckdb.CatalogException("Table does not exist")
    kpi._delete_run_days_from(ctx, "basic_kpi_by_line_day")
    sql = ctx.data.execute.call_args[0][0]
    assert "DELETE FROM basic_kpi_by_line_day" in sql
    assert "'2024-01-01'" in sql and "'2024-01-02'" in sql
    ctx.dash.execute.assert_not_called()  # sin also_dash no toca dash

    # also_dash → borra también en la DB dash (services_by_line_hour está espejada)
    ctx.data.execute.side_effect = None
    kpi._delete_run_days_from(ctx, "services_by_line_hour", also_dash=True)
    dash_sql = ctx.dash.execute.call_args[0][0]
    assert "DELETE FROM services_by_line_hour" in dash_sql
    assert "'2024-01-01'" in dash_sql


def test_delete_run_days_from_noop_sin_run_days():
    ctx = MagicMock()
    ctx.data.get_run_days.return_value = pd.DataFrame({"dia": []})
    kpi._delete_run_days_from(ctx, "basic_kpi_by_line_day", also_dash=True)
    ctx.data.execute.assert_not_called()
    ctx.dash.execute.assert_not_called()
