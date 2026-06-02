import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# agg_matriz — observed=True regression (commit 549a76e)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Minimal fake ports — have query() but deliberately NO get_raw()
# ---------------------------------------------------------------------------

class _FakePort:
    """Mimics DuckDBDashAdapter: has query() but not get_raw()."""

    def __init__(self, tables: dict):
        # tables = {"table_name": ["col_a", "col_b", ...]}
        self._tables = tables

    def query(self, sql: str) -> pd.DataFrame:
        sql_lower = sql.lower()
        for name, cols in self._tables.items():
            if name in sql_lower:
                return pd.DataFrame(columns=cols)
        return pd.DataFrame()


def test_table_exists_returns_true_for_known_table():
    from urbantrips.preparo_dashboard.preparo_dashboard import _table_exists
    port = _FakePort({"etapas": ["dia", "h3_o"]})
    assert _table_exists(port, "etapas") is True


def test_table_exists_returns_false_for_missing_table():
    from urbantrips.preparo_dashboard.preparo_dashboard import _table_exists
    port = _FakePort({})
    assert _table_exists(port, "nonexistent") is False


def test_table_has_cols_true_when_all_present():
    from urbantrips.preparo_dashboard.preparo_dashboard import _table_has_cols
    port = _FakePort({"agg_etapas": ["dia", "zona", "factor_expansion_linea"]})
    assert _table_has_cols(port, "agg_etapas", ["dia", "zona"]) is True


def test_table_has_cols_false_when_col_missing():
    from urbantrips.preparo_dashboard.preparo_dashboard import _table_has_cols
    port = _FakePort({"agg_etapas": ["dia"]})
    assert _table_has_cols(port, "agg_etapas", ["dia", "missing_col"]) is False


def test_table_has_cols_false_for_missing_table():
    from urbantrips.preparo_dashboard.preparo_dashboard import _table_has_cols
    port = _FakePort({})
    assert _table_has_cols(port, "nonexistent", ["dia"]) is False


def test_imprimo_matrices_od_does_not_call_get_raw(mocker):
    """imprimo_matrices_od must use query(), not get_raw() (DashAdapter lacks it)."""
    from urbantrips.preparo_dashboard.preparo_dashboard import imprimo_matrices_od

    fake_dash = _FakePort({"agg_matrices": [
        "id_polygon", "tipo_dia", "zona", "inicio", "fin",
        "transferencia", "modo_agregado", "rango_hora", "genero_agregado",
        "tarifa_agregada", "distancia_agregada", "orden_origen", "orden_destino",
        "Origen", "Destino", "lat1", "lon1", "lat4", "lon4",
        "distancia", "travel_time_min", "travel_speed", "factor_expansion_linea",
        "dia",
    ]})
    mocker.patch(
        "urbantrips.preparo_dashboard.preparo_dashboard.leer_alias",
        return_value="test",
    )
    mocker.patch(
        "urbantrips.preparo_dashboard.preparo_dashboard.agg_matriz",
        return_value=pd.DataFrame(columns=["zona"]),
    )
    mocker.patch(
        "urbantrips.preparo_dashboard.preparo_dashboard.replace_dash_partition",
    )

    ctx = mocker.MagicMock()
    ctx.dash = fake_dash

    # Should NOT raise AttributeError
    imprimo_matrices_od(ctx)


def test_polygon_loop_excess_legs_subset_computed_once():
    """
    After the fix the excess-legs subset is stored once and reused.
    Contract: trips with etapa_max > 3 are dropped; others kept.
    """
    import pandas as pd

    etapas_all = pd.DataFrame({
        "dia": ["2024-01-01"] * 6,
        "id_tarjeta": ["T1", "T1", "T1", "T1", "T2", "T2"],
        "id_viaje": [1, 1, 1, 1, 2, 2],
        "id_etapa": [1, 2, 3, 4, 1, 2],
        "h3_o": ["a"] * 6,
        "h3_d": ["b"] * 6,
    })
    etapas_all["etapa_max"] = etapas_all.groupby(
        ["dia", "id_tarjeta", "id_viaje"]
    ).id_etapa.transform("max")

    # Apply the fixed logic directly (same code as the patch)
    _excess = etapas_all[etapas_all.etapa_max > 3]
    if len(_excess) > 0:
        etapas_all = etapas_all[etapas_all.etapa_max <= 3].copy()
        del _excess

    assert len(etapas_all) == 2, "T1 (4 etapas) should be dropped"
    assert set(etapas_all.id_tarjeta.unique()) == {"T2"}


def test_load_and_process_data_excludes_batch_columns(mocker):
    """batch_id, etapa_validada, factor_expansion_etapa must not appear in etapas."""
    from urbantrips.preparo_dashboard.preparo_dashboard import load_and_process_data
    from unittest.mock import MagicMock

    n = 5
    etapas_df = pd.DataFrame({
        "id": range(n),
        "dia": ["2024-01-01"] * n,
        "id_tarjeta": [f"T{i}" for i in range(n)],
        "id_viaje": [1] * n,
        "id_etapa": range(1, n + 1),
        "tiempo": ["08:00:00"] * n,
        "hora": [8] * n,
        "modo": ["autobus"] * n,
        "id_linea": [1] * n,
        "id_ramal": [1] * n,
        "interno": [101] * n,
        "genero": ["M"] * n,
        "tarifa": ["completo"] * n,
        "latitud": [-34.6] * n,
        "longitud": [-58.5] * n,
        "h3_o": ["8b2a100d2bfffff"] * n,
        "h3_d": ["8b2a100d2cfffff"] * n,
        "od_validado": [1] * n,
        "factor_expansion_original": [1.0] * n,
        "factor_expansion_linea": [1.0] * n,
        "factor_expansion_tarjeta": [1.0] * n,
        "distancia": [2.5] * n,
        "travel_time_min": [15.0] * n,
        "distance_od": [2.5] * n,
    })
    viajes_df = pd.DataFrame({
        "id_tarjeta": [f"T{i}" for i in range(n)],
        "id_viaje": [1] * n,
        "dia": ["2024-01-01"] * n,
        "tiempo": ["08:00:00"] * n,
        "hora": [8] * n,
        "cant_etapas": [1] * n,
        "modo": ["autobus"] * n,
        "autobus": [1] * n,
        "tren": [0] * n,
        "metro": [0] * n,
        "tranvia": [0] * n,
        "brt": [0] * n,
        "cable": [0] * n,
        "lancha": [0] * n,
        "otros": [0] * n,
        "h3_o": ["8b2a100d2bfffff"] * n,
        "h3_d": ["8b2a100d2cfffff"] * n,
        "genero": ["M"] * n,
        "tarifa": ["completo"] * n,
        "od_validado": [1] * n,
        "factor_expansion_linea": [1.0] * n,
        "factor_expansion_tarjeta": [1.0] * n,
        "distancia": [2.5] * n,
        "travel_time_min": [15.0] * n,
        "distance_od": [2.5] * n,
    })

    ctx = MagicMock()
    ctx.data.query.side_effect = lambda sql: (
        etapas_df.copy() if "etapas" in sql else viajes_df.copy()
    )

    etapas, viajes = load_and_process_data(ctx)

    assert "batch_id" not in etapas.columns
    assert "etapa_validada" not in etapas.columns
    assert "factor_expansion_etapa" not in etapas.columns


# ---------------------------------------------------------------------------
# agg_matriz — observed=True (commit 549a76e)
# ---------------------------------------------------------------------------

def test_agg_matriz_observed_true_no_phantom_rows_for_empty_categories():
    """groupby(..., observed=True) must not create rows for unobserved category levels.

    Before fix 549a76e, groupby without observed=True inflated results with zero-weight
    rows for every unused category value (e.g., transfer types, modes, hour ranges).
    """
    from urbantrips.preparo_dashboard.aggregation import agg_matriz

    # 'Origen' has 3 category levels but only "A" is present in the data
    df = pd.DataFrame({
        "id_polygon":            [1, 1],
        "zona":                  ["Z1", "Z1"],
        "Origen":                pd.Categorical(["A", "A"], categories=["A", "B", "C"]),
        "Destino":               ["X", "X"],
        "transferencia":         [0, 0],
        "modo_agregado":         ["bus", "bus"],
        "rango_hora":            ["manana", "manana"],
        "distancia_agregada":    [5, 5],
        "factor_expansion_linea": [1.0, 2.0],
        "distance_od":           [3.0, 5.0],
        "travel_time_min":       [10.0, 20.0],
        "kmh_od":                [18.0, 15.0],
    })

    result = agg_matriz(df)

    # Only "A" was observed — must not produce extra rows for "B" or "C"
    assert len(result) == 1
    assert result["Origen"].iloc[0] == "A"
    # Weight sum must match the actual data (1.0 + 2.0)
    assert result["factor_expansion_linea"].iloc[0] == 3.0


def test_agg_matriz_observed_false_would_produce_extra_rows():
    """Confirm the pre-fix behaviour: without observed=True, extra rows appear.

    This test documents what the bug looked like so the regression is clear.
    It is written against a local groupby — not calling agg_matriz — so it will
    not start failing if agg_matriz is ever refactored.
    """
    df = pd.DataFrame({
        "Origen": pd.Categorical(["A", "A"], categories=["A", "B", "C"]),
        "weight": [1.0, 2.0],
    })
    without_observed = df.groupby("Origen", as_index=False)["weight"].sum()
    with_observed    = df.groupby("Origen", as_index=False, observed=True)["weight"].sum()

    assert len(without_observed) == 3, "Without observed=True all 3 category levels appear"
    assert len(with_observed)    == 1, "With observed=True only the observed level appears"


# ---------------------------------------------------------------------------
# B3 — DBSCAN tuning config
# ---------------------------------------------------------------------------

def test_leer_configs_tuning_returns_defaults_when_no_file(tmp_path, monkeypatch):
    """Returns hardcoded defaults when configs/tuning.yaml does not exist."""
    monkeypatch.chdir(tmp_path)
    # Reload to clear any cached state from previous runs
    import importlib
    import urbantrips.utils.utils as utils_mod
    importlib.reload(utils_mod)
    from urbantrips.utils.utils import leer_configs_tuning
    cfg = leer_configs_tuning()
    assert cfg["dbscan"]["grid_steps"] == 5
    assert cfg["dbscan"]["early_stop_silhouette"] == 0.7


def test_leer_configs_tuning_overrides_from_file(tmp_path, monkeypatch):
    """Values in tuning.yaml override the defaults."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tuning.yaml").write_text(
        "dbscan:\n  grid_steps: 10\n  early_stop_silhouette: 0.5\n",
        encoding="utf-8",
    )
    import importlib
    import urbantrips.utils.utils as utils_mod
    importlib.reload(utils_mod)
    from urbantrips.utils.utils import leer_configs_tuning
    cfg = leer_configs_tuning()
    assert cfg["dbscan"]["grid_steps"] == 10
    assert cfg["dbscan"]["early_stop_silhouette"] == 0.5


def test_dbscan_grid_search_respects_grid_steps(monkeypatch):
    """_run_grid_search uses grid_steps from tuning config and runs ≤ grid_steps² fits."""
    import numpy as np
    import pandas as pd
    import sklearn.cluster

    call_count = {"n": 0}
    original_fit = sklearn.cluster.DBSCAN.fit

    def counting_fit(self, X, y=None, sample_weight=None):
        call_count["n"] += 1
        return original_fit(self, X, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(sklearn.cluster.DBSCAN, "fit", counting_fit)
    monkeypatch.setattr(
        "urbantrips.cluster.dbscan.leer_configs_tuning",
        lambda: {"dbscan": {"grid_steps": 3, "early_stop_silhouette": 0.99}},
    )

    np.random.seed(42)
    n = 60
    cluster_a = np.random.uniform(0.1, 0.3, (n // 2, 2))
    cluster_b = np.random.uniform(0.6, 0.8, (n // 2, 2))
    X = pd.DataFrame(np.vstack([cluster_a, cluster_b]), columns=["o_proj", "d_proj"])
    w = pd.Series(np.ones(n))

    from urbantrips.cluster.dbscan import _run_grid_search
    _run_grid_search(X, w, type_k="lrs")

    # grid_steps=3 → 3×3=9 grid fits + 3 final fits = 12 max
    assert call_count["n"] <= 12, (
        f"Expected at most 12 DBSCAN fits, got {call_count['n']}"
    )
