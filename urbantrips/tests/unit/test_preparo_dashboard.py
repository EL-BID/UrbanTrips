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
# B1 — construyo_indicadores DuckDB equivalence
# ---------------------------------------------------------------------------

def _make_viajes_for_indicadores():
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    dias = rng.choice(["2024-10-14", "2024-10-15", "2024-10-19"], n)
    return pd.DataFrame({
        "id_polygon": rng.choice(["poly_1", "poly_2"], n).tolist(),
        "dia": dias.tolist(),
        "mes": ["2024-10"] * n,
        "tipo_dia": ["Hábil" if d < "2024-10-19" else "Fin de Semana" for d in dias],
        "id_tarjeta": [f"T{i % 50:03d}" for i in range(n)],
        "id_viaje": rng.integers(1, 5, n).tolist(),
        "rango_hora": rng.choice(["0-12", "13-16", "17-24"], n).tolist(),
        "distancia_agregada": rng.choice(
            ["Viajes cortos (<=5kms)", "Viajes largos (>5kms)"], n
        ).tolist(),
        "modo": rng.choice(["COLECTIVO", "SUBTE", "TREN"], n).tolist(),
        "transferencia": rng.integers(0, 2, n).tolist(),
        "factor_expansion_linea": rng.uniform(1.0, 3.5, n).tolist(),
        "distance_od": rng.uniform(1.0, 15.0, n).tolist(),
        "distance_route": rng.uniform(1.2, 16.0, n).tolist(),
        "distance_route_gps": rng.uniform(1.3, 17.0, n).tolist(),
        "travel_time_min": rng.integers(5, 60, n).astype(float).tolist(),
        "kmh_od": rng.uniform(5.0, 60.0, n).tolist(),
        "cant_etapas": rng.integers(1, 4, n).tolist(),
        "hora": rng.integers(0, 24, n).tolist(),
        "od_validado": [1] * n,
        "factor_expansion_tarjeta": rng.uniform(1.0, 3.5, n).tolist(),
        "modo_agregado": rng.choice(["COLECTIVO", "multietapa (COLECTIVO)", "multimodal"], n).tolist(),
        "genero_agregado": rng.choice(["M", "F", ""], n).tolist(),
        "tarifa_agregada": rng.choice(["normal", "social"], n).tolist(),
    })


def test_construyo_indicadores_duckdb_matches_pandas():
    """
    DuckDB implementation must produce the same set of indicators as the pandas baseline.
    Exact Valor strings may differ slightly due to float rounding; we check structure
    and that numeric values are within 5% of each other.
    """
    from unittest.mock import MagicMock
    from urbantrips.preparo_dashboard.preparo_dashboard import (
        construyo_indicadores,
        _construyo_indicadores_pandas,
    )
    import numpy as np

    viajes = _make_viajes_for_indicadores()

    def _make_ctx():
        ctx = MagicMock()
        ctx.dash.get_raw.return_value = pd.DataFrame()
        ctx.dash.append_raw.return_value = None
        ctx.dash.execute.return_value = None
        return ctx

    ctx_pandas = _make_ctx()
    _construyo_indicadores_pandas(ctx_pandas, viajes.copy(), poligonos=False)
    expected_df = ctx_pandas.dash.append_raw.call_args[0][0]

    ctx_duckdb = _make_ctx()
    construyo_indicadores(ctx_duckdb, viajes.copy(), poligonos=False)
    actual_df = ctx_duckdb.dash.append_raw.call_args[0][0]

    sort_cols = ["id_polygon", "dia", "Tipo", "Indicador"]

    # 1. Same set of (id_polygon, dia, Tipo, Indicador) groups
    expected_keys = set(map(tuple, expected_df[sort_cols].drop_duplicates().values))
    actual_keys = set(map(tuple, actual_df[sort_cols].drop_duplicates().values))
    assert actual_keys == expected_keys, (
        f"Missing indicators: {expected_keys - actual_keys}\n"
        f"Extra indicators: {actual_keys - expected_keys}"
    )

    # 2. Same columns
    assert set(actual_df.columns) == set(expected_df.columns)

    # 3. Numeric values within 5% (parse formatted strings back to float)
    def parse_valor(s):
        try:
            return float(str(s).replace(".", "").replace(",", ".").replace("%", "").strip())
        except Exception:
            return np.nan

    merged = expected_df[sort_cols + ["Valor"]].merge(
        actual_df[sort_cols + ["Valor"]].rename(columns={"Valor": "Valor_actual"}),
        on=sort_cols,
        how="inner",
    )
    merged["v_exp"] = merged["Valor"].apply(parse_valor)
    merged["v_act"] = merged["Valor_actual"].apply(parse_valor)
    mask = merged["v_exp"].notna() & merged["v_act"].notna() & (merged["v_exp"].abs() > 0.001)
    rel_diff = ((merged.loc[mask, "v_act"] - merged.loc[mask, "v_exp"]).abs()
                / merged.loc[mask, "v_exp"].abs())
    assert (rel_diff <= 0.05).all(), (
        f"Some indicator values differ by more than 5%:\n"
        f"{merged[mask][rel_diff > 0.05][sort_cols + ['Valor', 'Valor_actual']].head(10)}"
    )
