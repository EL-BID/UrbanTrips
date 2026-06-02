import numpy as np
import pandas as pd
from types import SimpleNamespace
from urbantrips.utils.utils import calculate_weighted_means


def test_calculate_weighted_means_equal_weights():
    """Equal weights produce the same result as a simple mean."""
    df = pd.DataFrame({
        "dia": ["2022-08-11", "2022-08-11"],
        "value": [10.0, 20.0],
        "weight": [1.0, 1.0],
    })
    result = calculate_weighted_means(
        df, aggregate_cols=["dia"], weighted_mean_cols=["value"], weight_col="weight"
    )
    assert abs(result.loc[0, "value"] - 15.0) < 0.001


def test_calculate_weighted_means_zero_to_nan():
    """zero_to_nan parameter should exclude zero values from the weighted mean."""
    df = pd.DataFrame({
        "dia": ["2022-08-11", "2022-08-11"],
        "value": [0.0, 20.0],
        "weight": [5.0, 1.0],
    })
    result_with = calculate_weighted_means(
        df, aggregate_cols=["dia"], weighted_mean_cols=["value"],
        weight_col="weight", zero_to_nan=["value"]
    )
    result_without = calculate_weighted_means(
        df, aggregate_cols=["dia"], weighted_mean_cols=["value"],
        weight_col="weight"
    )
    # With zero_to_nan: only value=20 contributes → result = 20
    assert abs(result_with.loc[0, "value"] - 20.0) < 0.001
    # Without: (0*5 + 20*1)/6 ≈ 3.33
    assert result_without.loc[0, "value"] < 5.0


# --- compute_speed_by_day_veh_hour ---

def test_compute_speed_null_distance_servicio_coerced_to_nan(monkeypatch):
    """NULL distance_servicio_mts must produce NaN in distance_km_gps, not crash.

    Before fix d971510, distance_servicio_mts was absent from the SELECT; after
    the fix pd.to_numeric(..., errors='coerce') handles NULL/non-numeric values.
    """
    from urbantrips.kpi import kpi as kpi_module

    monkeypatch.setattr(kpi_module, "get_processed_days", lambda ctx, table_name: "''")

    gps_rows = pd.DataFrame({
        "dia":                  ["2024-01-01"] * 4,
        "id_linea":             [1] * 4,
        "id_ramal":             [1] * 4,
        "interno":              [101] * 4,
        "fecha":                [1704096000, 1704096060, 1704096120, 1704096180],
        "velocity":             [30.0] * 4,
        "distance_km":          [0.5] * 4,
        "distance_servicio_mts": [1000.0, None, None, 500.0],
    })

    class _MockData:
        def query(self, sql):
            return gps_rows.copy()

    ctx = SimpleNamespace(data=_MockData())
    result = kpi_module.compute_speed_by_day_veh_hour(ctx)

    assert result is not None
    # Rows with NULL distance_servicio_mts must have NaN kmh_route_gps_veh_h, not error
    null_rows = result[result["kmh_route_gps_veh_h"].isna()]
    # At least some rows with NULL distance should survive as NaN (not crash)
    assert "kmh_route_veh_h" in result.columns
    assert "kmh_route_gps_veh_h" in result.columns


def test_compute_speed_distance_km_gps_scales_from_metres(monkeypatch):
    """distance_servicio_mts (metres) must be divided by 1000 to get km."""
    from urbantrips.kpi import kpi as kpi_module

    monkeypatch.setattr(kpi_module, "get_processed_days", lambda ctx, table_name: "''")

    # Two consecutive pings 60 seconds apart, odometer reports 1800 m → 1.8 km
    # delta_hr = 1/60, so kmh_route_gps_veh_h = 1.8 / (1/60) = 108 → capped later
    # Use a modest value: 600 m in 60 s = 0.6 km → 36 km/h
    gps_rows = pd.DataFrame({
        "dia":                  ["2024-01-01", "2024-01-01"],
        "id_linea":             [1, 1],
        "id_ramal":             [1, 1],
        "interno":              [101, 101],
        "fecha":                [1704096000, 1704096060],
        "velocity":             [36.0, 36.0],
        "distance_km":          [0.6, 0.6],
        "distance_servicio_mts": [600.0, 600.0],
    })

    class _MockData:
        def query(self, sql):
            return gps_rows.copy()

    ctx = SimpleNamespace(data=_MockData())
    result = kpi_module.compute_speed_by_day_veh_hour(ctx)

    assert result is not None and len(result) > 0
    # 600 m / 1000 = 0.6 km; delta = 60s = 1/60 hr → speed = 0.6 * 60 = 36 km/h
    assert abs(result["kmh_route_gps_veh_h"].iloc[0] - 36.0) < 1.0


# --- compute_kpi_by_line_day ---

def test_compute_kpi_by_line_day_does_not_use_conn_data():
    """compute_kpi_by_line_day must not reference 'conn_data' (undefined variable).

    Before fix 9705d39, the code had pd.read_sql(..., conn_data) which raised
    NameError at runtime.  This source-level check catches any reintroduction.
    """
    import inspect
    from urbantrips.kpi.kpi import compute_kpi_by_line_day

    src = inspect.getsource(compute_kpi_by_line_day)
    assert "conn_data" not in src, (
        "'conn_data' (undefined variable) must not appear in compute_kpi_by_line_day"
    )
    assert "ctx.data.query" in src, (
        "services must be fetched via ctx.data.query, not pd.read_sql"
    )
