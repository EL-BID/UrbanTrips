import pandas as pd
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
