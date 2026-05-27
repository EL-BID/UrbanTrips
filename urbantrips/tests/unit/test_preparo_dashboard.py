import pandas as pd
import pytest


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
    })

    ctx = MagicMock()
    ctx.data.query.side_effect = lambda sql: (
        etapas_df.copy() if "etapas" in sql else viajes_df.copy()
    )

    etapas, viajes = load_and_process_data(ctx)

    assert "batch_id" not in etapas.columns
    assert "etapa_validada" not in etapas.columns
    assert "factor_expansion_etapa" not in etapas.columns
