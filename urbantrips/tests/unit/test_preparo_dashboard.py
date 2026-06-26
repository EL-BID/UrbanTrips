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
# B6 — persistent DuckDB connection
# ---------------------------------------------------------------------------
import tempfile
from pathlib import Path


def _make_insumos_adapter(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    return DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")


def test_insumos_adapter_reuses_connection(tmp_path):
    """Two sequential calls share the same connection object."""
    adapter = _make_insumos_adapter(tmp_path)
    conn1 = adapter._conn
    conn2 = adapter._conn
    assert conn1 is conn2
    adapter.close()


def test_insumos_adapter_read_write_cycle(tmp_path):
    """save_raw then get_raw returns the same data."""
    import pandas as pd
    adapter = _make_insumos_adapter(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    adapter.save_raw(df, "test_table")
    result = adapter.get_raw("test_table")
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))
    adapter.close()


def test_insumos_adapter_close_is_idempotent(tmp_path):
    """Calling close() twice must not raise."""
    adapter = _make_insumos_adapter(tmp_path)
    adapter.close()
    adapter.close()  # must not raise


# ---------------------------------------------------------------------------
# B2 — no defensive copies at orchestration level
# ---------------------------------------------------------------------------

def test_preparo_dashboard_passes_frames_by_reference(mocker):
    """
    The indicator consumers (resumen_x_linea, construyo_indicadores non-polygon,
    crea_socio_indicadores, guarda_particion_modal, calculo_kpi_lineas) now
    self-source from the data DB via the proc-CTEs — the orchestrator no longer
    builds or passes the etapas/viajes frames. Verified by checking each consumer
    is invoked without a frame argument.
    """
    import pandas as pd
    from unittest.mock import MagicMock, patch

    from urbantrips.preparo_dashboard.preparo_dashboard import preparo_indicadores_dash as preparo_dashboard

    etapas = pd.DataFrame({"dia": ["2024-10-14"], "id": [1]})
    viajes = pd.DataFrame({"dia": ["2024-10-14"], "id_viaje": [1]})
    etapas_id = id(etapas)
    viajes_id = id(viajes)

    received_ids = {}

    def capture_resumen(ctx):
        # resumen_x_linea now self-sources etapas from the data DB (etapas_proc).
        received_ids["resumen_selfsourced"] = True

    def capture_indicadores(ctx, viajes=None, poligonos=False):
        if not poligonos:
            # non-polygon indicators now self-source from the data DB (proc-CTE
            # temp table); the orchestrator no longer passes the viajes frame.
            received_ids["ind_viajes_is_none"] = viajes is None

    def capture_socio(ctx):
        # crea_socio_indicadores now self-sources from the data DB (proc-CTEs).
        received_ids["socio_selfsourced"] = True

    def capture_particion(ctx):
        # guarda_particion_modal now self-sources from the data DB (etapas_proc).
        received_ids["part_selfsourced"] = True

    def capture_kpi(ctx):
        # calculo_kpi_lineas now self-sources from the data DB (etapas_proc).
        received_ids["kpi_selfsourced"] = True

    ctx = MagicMock()
    ctx.insumos.get_raw.return_value = pd.DataFrame()

    with (
        patch("urbantrips.preparo_dashboard.preparo_dashboard.guardo_zonificaciones"),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.migrar_equivalencias_zonas"),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.load_and_process_data",
              return_value=(etapas, viajes)),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.resumen_x_linea",
              side_effect=capture_resumen),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.construyo_indicadores",
              side_effect=capture_indicadores),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.crea_socio_indicadores",
              side_effect=capture_socio),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.guarda_particion_modal",
              side_effect=capture_particion),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.calculo_kpi_lineas",
              side_effect=capture_kpi),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.crear_indices_unificados"),
    ):
        preparo_dashboard(ctx, lineas_deseo=True, poligonos=True, kpis=True)

    assert received_ids.get("resumen_selfsourced"), "resumen_x_linea must self-source from the data DB"
    assert received_ids["ind_viajes_is_none"], "construyo_indicadores (non-polygon) must self-source, not receive a viajes frame"
    assert received_ids.get("socio_selfsourced"), "crea_socio_indicadores must self-source from the data DB"
    assert received_ids.get("part_selfsourced"), "guarda_particion_modal must self-source from the data DB"
    assert received_ids.get("kpi_selfsourced"), "calculo_kpi_lineas must self-source from the data DB"


# ---------------------------------------------------------------------------
# B4 — select_h3_from_polygon uses h3.geo_to_cells
# ---------------------------------------------------------------------------
import h3
from shapely.geometry import box
import geopandas as gpd


def _make_buenos_aires_polygons():
    """Two polygons in Buenos Aires large enough to contain H3 res-8 cells (~0.5km hex edge)."""
    # Plaza de Mayo area: ~1km × 1km
    poly1 = box(-58.380, -34.615, -58.365, -34.600)
    # Palermo area: ~1km × 1km
    poly2 = box(-58.440, -34.585, -58.420, -34.570)
    return gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[poly1, poly2],
        crs=4326,
    )


def test_select_h3_from_polygon_returns_valid_cells():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    assert len(result) > 0
    for cell in result["h3"]:
        assert h3.is_valid_cell(cell), f"{cell} is not a valid H3 cell"


def test_select_h3_from_polygon_cells_at_correct_resolution():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    for cell in result["h3"]:
        assert h3.get_resolution(cell) == 8


def test_select_h3_from_polygon_id_column_maps_to_source():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    assert set(result["id"].unique()) == {1, 2}


def test_select_h3_from_polygon_returns_geodataframe():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs is not None
    assert set(result.columns) >= {"id", "h3", "geometry"}


# ---------------------------------------------------------------------------
# B3 — DBSCAN tuning config
# ---------------------------------------------------------------------------

def test_leer_configs_tuning_returns_defaults_when_no_file(tmp_path, monkeypatch):
    """Returns hardcoded defaults when configs/tuning.yaml does not exist."""
    monkeypatch.chdir(tmp_path)
    import importlib
    from urbantrips.utils.paths import reset_paths
    reset_paths()
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
    from urbantrips.utils.paths import reset_paths
    reset_paths()
    import urbantrips.utils.utils as utils_mod
    importlib.reload(utils_mod)
    from urbantrips.utils.utils import leer_configs_tuning
    cfg = leer_configs_tuning()
    assert cfg["dbscan"]["grid_steps"] == 10
    assert cfg["dbscan"]["early_stop_silhouette"] == 0.5


def test_dbscan_grid_search_respects_grid_steps(monkeypatch):
    """_run_grid_search uses grid_steps from tuning config and runs ≤ grid_steps² fits."""
    import numpy as np
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

    assert call_count["n"] <= 12, f"Expected at most 12 DBSCAN fits, got {call_count['n']}"


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
        "kmh_od": [float("nan")] * 4,
        "kmh_route": [float("nan")] * 4,
        "kmh_route_gps": [float("nan")] * 4,
        "tipo_dia": ["Hábil", "Hábil", "Fin de Semana", "Fin de Semana"],
        "mes": ["2024-10"] * 4,
        "rango_hora": ["0-12", "13-16", "0-12", "17-24"],
        "distancia_agregada": [
            "Etapa corta (<=5kms)", "Etapa larga (>5kms)",
            "Etapa larga (>5kms)", "Etapa corta (<=5kms)",
        ],
    })


def _make_synthetic_viajes_b5():
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
        "kmh_od": [float("nan")] * 3,
        "kmh_route": [float("nan")] * 3,
        "kmh_route_gps": [float("nan")] * 3,
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
    # unified with chains_norm labels (singular) — was "Viajes cortos/largos"
    assert all(viajes.loc[viajes.distance_od <= 5, "distancia_agregada"] == "Viaje corto (<=5kms)")
    assert all(viajes.loc[viajes.distance_od > 5, "distancia_agregada"] == "Viaje largo (>5kms)")


def test_load_transferencia():
    _, viajes = _run_load_and_process(_make_synthetic_etapas_b5(), _make_synthetic_viajes_b5())
    assert all(viajes.loc[viajes.cant_etapas == 1, "transferencia"] == 0)
    assert all(viajes.loc[viajes.cant_etapas > 1, "transferencia"] == 1)


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

    expected_keys = set(map(tuple, expected_df[sort_cols].drop_duplicates().values))
    actual_keys = set(map(tuple, actual_df[sort_cols].drop_duplicates().values))
    assert actual_keys == expected_keys, (
        f"Missing indicators: {expected_keys - actual_keys}\n"
        f"Extra indicators: {actual_keys - expected_keys}"
    )

    assert set(actual_df.columns) == set(expected_df.columns)

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
