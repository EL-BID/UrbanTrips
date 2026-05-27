import pytest
import pandas as pd
import re


def test_duracion_wraps_function():
    from urbantrips.utils.decorators import duracion

    @duracion
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_duracion_logs_start_and_finish_timestamps(caplog):
    import logging
    from urbantrips.utils.decorators import duracion

    @duracion
    def noop():
        return None

    with caplog.at_level(logging.INFO, logger="urbantrips.utils.decorators"):
        noop()

    timestamp = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    assert re.search(rf"Iniciado noop \({timestamp}\)", caplog.text)
    assert re.search(rf"Finalizado noop \({timestamp}\)", caplog.text)


def test_duracion_backward_compat():
    from urbantrips.utils.utils import duracion

    @duracion
    def multiply(a, b):
        return a * b

    assert multiply(3, 4) == 12


def test_create_directories_creates_expected_paths(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    from urbantrips.utils.fs import create_directories
    create_directories()
    assert os.path.isdir("data/db")
    assert os.path.isdir("resultados/tablas")


def test_normalize_vars_renames_day_type():
    from urbantrips.utils.dataframe import normalize_vars
    df = pd.DataFrame({"day_type": ["weekday", "weekend"]})
    result = normalize_vars(df)
    assert list(result["day_type"]) == ["Día hábil", "Fin de semana"]


def test_calculate_weighted_means_basic():
    from urbantrips.utils.dataframe import calculate_weighted_means
    df = pd.DataFrame({
        "group": ["A", "A", "B"],
        "value": [10.0, 20.0, 30.0],
        "weight": [1.0, 1.0, 2.0],
    })
    result = calculate_weighted_means(
        df, aggregate_cols=["group"], weighted_mean_cols=["value"], weight_col="weight"
    )
    assert abs(result.loc[result.group == "A", "value"].values[0] - 15.0) < 0.01
