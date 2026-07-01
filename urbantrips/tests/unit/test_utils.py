import pandas as pd
import pytest
from urbantrips.utils.utils import (
    leer_alias,
    is_date_string,
    check_date_type,
    create_line_ids_sql_filter,
    create_branch_ids_sql_filter,
    normalize_vars,
    calculate_weighted_means,
)


# --- leer_alias ---

def test_leer_alias_data(monkeypatch):
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_generales",
        lambda autogenerado=True: {"alias_db_data": "myalias"},
    )
    assert leer_alias("data") == "myalias_"


def test_leer_alias_missing_key_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_generales",
        lambda autogenerado=True: {},
    )
    assert leer_alias("data") == ""


def test_leer_alias_invalid_tipo_raises(monkeypatch):
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_generales",
        lambda autogenerado=True: {},
    )
    with pytest.raises(ValueError):
        leer_alias("invalid_tipo")


def test_leer_alias_already_ends_with_underscore(monkeypatch):
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_generales",
        lambda autogenerado=True: {"alias_db_data": "city"},
    )
    result = leer_alias("data")
    assert result.endswith("_")
    assert not result.endswith("__")


# --- is_date_string ---

def test_is_date_string_valid():
    assert is_date_string("2022-08-11") is True


def test_is_date_string_invalid_format():
    assert is_date_string("08/11/2022") is False


def test_is_date_string_non_padded():
    assert is_date_string("2022-8-11") is False


def test_is_date_string_not_a_date():
    assert is_date_string("weekday") is False


# --- check_date_type ---

def test_check_date_type_weekday():
    check_date_type("weekday")   # should not raise


def test_check_date_type_weekend():
    check_date_type("weekend")   # should not raise


def test_check_date_type_date_string():
    check_date_type("2022-08-11")   # should not raise


def test_check_date_type_invalid_raises():
    with pytest.raises(Exception):
        check_date_type("monday")


# --- create_line_ids_sql_filter ---

def test_create_line_ids_sql_filter_list():
    result = create_line_ids_sql_filter([1, 2, 3])
    assert "id_linea in (1,2,3)" in result


def test_create_line_ids_sql_filter_single_int():
    result = create_line_ids_sql_filter(5)
    assert "id_linea in (5)" in result


def test_create_line_ids_sql_filter_none():
    result = create_line_ids_sql_filter(None)
    assert "id_linea is not NULL" in result


# --- create_branch_ids_sql_filter ---

def test_create_branch_ids_sql_filter_list():
    result = create_branch_ids_sql_filter([10, 20])
    assert "id_ramal in (10,20)" in result


def test_create_branch_ids_sql_filter_none():
    result = create_branch_ids_sql_filter(None)
    assert "id_ramal is not NULL" in result


# --- normalize_vars ---

def test_normalize_vars_weekday():
    df = pd.DataFrame({"day_type": ["weekday", "weekend"]})
    result = normalize_vars(df)
    assert result["day_type"].tolist() == ["Día hábil", "Fin de semana"]


def test_normalize_vars_no_target_columns_unchanged():
    df = pd.DataFrame({"col_a": [1, 2]})
    result = normalize_vars(df)
    assert result["col_a"].tolist() == [1, 2]


def test_normalize_vars_modo_capitalized():
    df = pd.DataFrame({"modo": ["autobus", "metro"]})
    result = normalize_vars(df)
    assert result["modo"].tolist() == ["Autobus", "Metro"]


# --- calculate_weighted_means ---

def test_calculate_weighted_means_basic():
    df = pd.DataFrame({
        "dia": ["2022-08-11", "2022-08-11", "2022-08-12"],
        "value": [10.0, 20.0, 30.0],
        "weight": [1.0, 3.0, 2.0],
    })
    result = calculate_weighted_means(
        df,
        aggregate_cols=["dia"],
        weighted_mean_cols=["value"],
        weight_col="weight",
    )
    # 2022-08-11: (10*1 + 20*3) / 4 = 17.5
    aug11 = result.loc[result.dia == "2022-08-11", "value"].iloc[0]
    assert abs(aug11 - 17.5) < 0.001


def test_calculate_weighted_means_single_row():
    df = pd.DataFrame({"dia": ["2022-08-12"], "value": [30.0], "weight": [2.0]})
    result = calculate_weighted_means(
        df, aggregate_cols=["dia"], weighted_mean_cols=["value"], weight_col="weight"
    )
    assert abs(result.loc[0, "value"] - 30.0) < 0.001


def test_calculate_weighted_means_missing_column_raises():
    df = pd.DataFrame({"dia": ["2022-08-11"], "value": [10.0]})
    with pytest.raises(ValueError):
        calculate_weighted_means(
            df,
            aggregate_cols=["dia"],
            weighted_mean_cols=["value"],
            weight_col="nonexistent",
        )
