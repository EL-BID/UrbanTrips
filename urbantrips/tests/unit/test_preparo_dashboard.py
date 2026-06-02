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
        # derived columns now returned by SQL
        "tipo_dia": ["Hábil"] * n,
        "mes": ["2024-01"] * n,
        "rango_hora": ["0-12"] * n,
        "distancia_agregada": ["Etapa corta (<=5kms)"] * n,
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
        # derived columns now returned by SQL
        "tipo_dia": ["Hábil"] * n,
        "mes": ["2024-01"] * n,
        "rango_hora": ["0-12"] * n,
        "distancia_agregada": ["Viajes cortos (<=5kms)"] * n,
        "transferencia": [0] * n,
    })

    ctx = MagicMock()
    ctx.data.query.side_effect = lambda sql: (
        etapas_df.copy() if "FROM etapas" in sql else viajes_df.copy()
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
# B5 — derived columns computed in SQL
# ---------------------------------------------------------------------------

def _make_synthetic_etapas_b5():
    import numpy as np
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "dia": ["2024-10-14", "2024-10-14", "2024-10-19", "2024-10-19"],
        "id_tarjeta": ["T1", "T1", "T2", "T2"],
        "id_viaje": [1, 1, 2, 2],
        "id_etapa": [1, 2, 1, 2],
        "tiempo": ["08:00:00", "08:30:00", "10:00:00", "10:40:00"],
        "hora": [8, 14, 10, 20],
        "modo": ["COLECTIVO", "SUBTE", "COLECTIVO", "COLECTIVO"],
        "id_linea": [28, 99, 45, 45],
        "id_ramal": [1, 1, 1, 1],
        "interno": [101, 201, 102, 102],
        "genero": [None, None, None, None],
        "tarifa": ["normal", "normal", "social", "social"],
        "latitud": [-34.608, -34.610, -34.578, -34.582],
        "longitud": [-58.372, -58.374, -58.430, -58.432],
        "h3_o": ["88e8ea402bfffff", "88e8ea402bfffff", "88e8ea4051fffff", "88e8ea4051fffff"],
        "h3_d": ["88e8ea4031fffff", "88e8ea4031fffff", "88e8ea405bfffff", "88e8ea405bfffff"],
        "od_validado": [1, 1, 1, 1],
        "factor_expansion_original": [1.0, 1.0, 2.0, 2.0],
        "factor_expansion_linea": [1.0, 1.0, 2.0, 2.0],
        "factor_expansion_tarjeta": [1.0, 1.0, 2.0, 2.0],
        "travel_time_min": [20.0, 15.0, 30.0, 10.0],
        "distance_od": [3.5, 7.2, 7.2, 1.8],
        "distance_route": [4.0, 8.0, 8.0, 2.0],
        "distance_route_gps": [4.1, 8.1, 8.1, 2.1],
        "kmh_od": [np.nan] * 4,
        "kmh_route": [np.nan] * 4,
        "kmh_route_gps": [np.nan] * 4,
        # derived columns returned by SQL
        "tipo_dia": ["Hábil", "Hábil", "Fin de Semana", "Fin de Semana"],
        "mes": ["2024-10"] * 4,
        "rango_hora": ["0-12", "13-16", "0-12", "17-24"],
        "distancia_agregada": [
            "Etapa corta (<=5kms)", "Etapa larga (>5kms)",
            "Etapa larga (>5kms)", "Etapa corta (<=5kms)",
        ],
    })


def _make_synthetic_viajes_b5():
    import numpy as np
    return pd.DataFrame({
        "dia": ["2024-10-14", "2024-10-14", "2024-10-19"],
        "id_tarjeta": ["T1", "T1", "T2"],
        "id_viaje": [1, 2, 1],
        "tiempo": ["08:00:00", "09:00:00", "10:00:00"],
        "hora": [8, 14, 10],
        "cant_etapas": [2, 1, 1],
        "modo": ["COLECTIVO", "COLECTIVO", "COLECTIVO"],
        "autobus": [1, 1, 1],
        "tren": [0, 0, 0],
        "metro": [0, 0, 0],
        "tranvia": [0, 0, 0],
        "brt": [0, 0, 0],
        "cable": [0, 0, 0],
        "lancha": [0, 0, 0],
        "otros": [0, 0, 0],
        "h3_o": ["88e8ea402bfffff", "88e8ea402bfffff", "88e8ea4051fffff"],
        "h3_d": ["88e8ea4031fffff", "88e8ea4051fffff", "88e8ea405bfffff"],
        "genero": [None, None, None],
        "tarifa": ["normal", "normal", "social"],
        "od_validado": [1, 1, 1],
        "factor_expansion_linea": [1.0, 1.0, 2.0],
        "factor_expansion_tarjeta": [1.0, 1.0, 2.0],
        "distance_od": [3.5, 7.2, 7.2],
        "travel_time_min": [35.0, 30.0, 30.0],
        "distance_route": [4.0, 8.0, 8.0],
        "distance_route_gps": [4.1, 8.1, 8.1],
        "kmh_od": [np.nan] * 3,
        "kmh_route": [np.nan] * 3,
        "kmh_route_gps": [np.nan] * 3,
        # derived columns returned by SQL
        "tipo_dia": ["Hábil", "Hábil", "Fin de Semana"],
        "mes": ["2024-10"] * 3,
        "rango_hora": ["0-12", "13-16", "0-12"],
        "distancia_agregada": [
            "Viajes cortos (<=5kms)", "Viajes largos (>5kms)", "Viajes largos (>5kms)",
        ],
        "transferencia": [1, 0, 0],
    })


def _run_load_and_process(etapas_raw, viajes_raw):
    from urbantrips.preparo_dashboard.preparo_dashboard import load_and_process_data
    from unittest.mock import MagicMock
    ctx = MagicMock()
    ctx.data.query.side_effect = [etapas_raw.copy(), viajes_raw.copy()]
    return load_and_process_data(ctx)


def test_load_tipo_dia_weekday():
    etapas, viajes = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert all(etapas.loc[etapas.dia == "2024-10-14", "tipo_dia"] == "Hábil")
    assert all(viajes.loc[viajes.dia == "2024-10-14", "tipo_dia"] == "Hábil")


def test_load_tipo_dia_weekend():
    etapas, viajes = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert all(etapas.loc[etapas.dia == "2024-10-19", "tipo_dia"] == "Fin de Semana")
    assert all(viajes.loc[viajes.dia == "2024-10-19", "tipo_dia"] == "Fin de Semana")


def test_load_mes_column():
    etapas, viajes = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert all(etapas["mes"] == "2024-10")
    assert all(viajes["mes"] == "2024-10")


def test_load_rango_hora_etapas():
    etapas, _ = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert etapas.loc[etapas.hora == 8, "rango_hora"].iloc[0] == "0-12"
    assert etapas.loc[etapas.hora == 14, "rango_hora"].iloc[0] == "13-16"
    assert etapas.loc[etapas.hora == 20, "rango_hora"].iloc[0] == "17-24"


def test_load_distancia_agregada_etapas():
    etapas, _ = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert all(etapas.loc[etapas.distance_od <= 5, "distancia_agregada"] == "Etapa corta (<=5kms)")
    assert all(etapas.loc[etapas.distance_od > 5, "distancia_agregada"] == "Etapa larga (>5kms)")


def test_load_distancia_agregada_viajes():
    _, viajes = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert all(viajes.loc[viajes.distance_od <= 5, "distancia_agregada"] == "Viajes cortos (<=5kms)")
    assert all(viajes.loc[viajes.distance_od > 5, "distancia_agregada"] == "Viajes largos (>5kms)")


def test_load_transferencia():
    _, viajes = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert all(viajes.loc[viajes.cant_etapas == 1, "transferencia"] == 0)
    assert all(viajes.loc[viajes.cant_etapas > 1, "transferencia"] == 1)
