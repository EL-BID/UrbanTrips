# B2 — Remove DataFrame Copies at Orchestration Level

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove three unnecessary `.copy()` calls that duplicate multi-GB DataFrames before passing them to `proceso_lineas_deseo`, `proceso_poligonos`, and `calculo_kpi_lineas`.

**Architecture:** The three called functions write their results to DuckDB tables — they don't mutate the caller's `etapas` or `viajes` references in any way the caller depends on. Internal defensive copies that guard mutation within those functions (e.g. `preparo_dashboard.py:1081`) are left untouched.

**Tech Stack:** Python, pandas, pytest, pytest-mock

---

## File map

- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py` (lines ~2463–2490)
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

---

### Task 1: Write failing test

**Files:**
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

- [ ] **Step 1: Add test that verifies orchestrator does not copy DataFrames**

Append to `test_preparo_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# B2 — no defensive copies at orchestration level
# ---------------------------------------------------------------------------

def test_preparo_dashboard_passes_frames_by_reference(mocker):
    """
    The top-level orchestrator must NOT call .copy() on etapas/viajes before
    passing to proceso_lineas_deseo, proceso_poligonos, or calculo_kpi_lineas.
    Verified by checking that the id() of the DataFrame received inside each
    function matches the id() of the original.
    """
    import pandas as pd
    from unittest.mock import MagicMock, patch

    from urbantrips.preparo_dashboard.preparo_dashboard import preparo_dashboard

    etapas = pd.DataFrame({"dia": ["2024-10-14"], "id": [1]})
    viajes = pd.DataFrame({"dia": ["2024-10-14"], "id_viaje": [1]})
    etapas_id = id(etapas)
    viajes_id = id(viajes)

    received_ids = {}

    def capture_lineas_deseo(ctx, etapas, viajes, **kwargs):
        received_ids["lineas_etapas"] = id(etapas)
        received_ids["lineas_viajes"] = id(viajes)

    def capture_poligonos(ctx, etapas, viajes, **kwargs):
        received_ids["poly_etapas"] = id(etapas)
        received_ids["poly_viajes"] = id(viajes)

    def capture_kpi(ctx, etapas, viajes):
        received_ids["kpi_etapas"] = id(etapas)
        received_ids["kpi_viajes"] = id(viajes)

    ctx = MagicMock()

    with (
        patch("urbantrips.preparo_dashboard.preparo_dashboard.guardo_zonificaciones"),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.load_and_process_data",
              return_value=(etapas, viajes)),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.proceso_lineas_deseo",
              side_effect=capture_lineas_deseo),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.proceso_poligonos",
              side_effect=capture_poligonos),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.calculo_kpi_lineas",
              side_effect=capture_kpi),
        patch("urbantrips.preparo_dashboard.preparo_dashboard.crear_indices_unificados"),
    ):
        ctx.insumos.get_raw.return_value = pd.DataFrame()
        preparo_dashboard(ctx, lineas_deseo=True, poligonos=True, kpis=True)

    assert received_ids["lineas_etapas"] == etapas_id, "proceso_lineas_deseo received a copy of etapas"
    assert received_ids["lineas_viajes"] == viajes_id, "proceso_lineas_deseo received a copy of viajes"
    assert received_ids["poly_etapas"] == etapas_id, "proceso_poligonos received a copy of etapas"
    assert received_ids["poly_viajes"] == viajes_id, "proceso_poligonos received a copy of viajes"
    assert received_ids["kpi_etapas"] == etapas_id, "calculo_kpi_lineas received a copy of etapas"
    assert received_ids["kpi_viajes"] == viajes_id, "calculo_kpi_lineas received a copy of viajes"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_preparo_dashboard_passes_frames_by_reference -v
```

Expected: FAIL — `id()` of received DataFrames differs from originals because `.copy()` is called.

---

### Task 2: Remove the `.copy()` calls

**Files:**
- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`

- [ ] **Step 1: Remove copies in `preparo_dashboard()` function (around line 2463)**

Find this block:

```python
    if lineas_deseo:
        # print("Proceso lineas de deseo")
        proceso_lineas_deseo(
            ctx,
            etapas=etapas.copy(),
            viajes=viajes.copy(),
            zonificaciones=zonificaciones.copy(),
            equivalencias_zonas=equivalencias_zonas.copy(),
            resoluciones=resoluciones,
        )
    if poligonos:
        # print("Proceso Polígonos")
        proceso_poligonos(
            ctx,
            etapas=etapas.copy(),
            viajes=viajes.copy(),
            zonificaciones=zonificaciones.copy(),
            resoluciones=resoluciones,
            poligon_id=poligon_id,
        )

    if kpis:
        # print("Proceso kpis")
        kpis = calculo_kpi_lineas(
            ctx, etapas=etapas.copy(), viajes=viajes.copy()
        )
```

Replace with:

```python
    if lineas_deseo:
        proceso_lineas_deseo(
            ctx,
            etapas=etapas,
            viajes=viajes,
            zonificaciones=zonificaciones,
            equivalencias_zonas=equivalencias_zonas,
            resoluciones=resoluciones,
        )
    if poligonos:
        proceso_poligonos(
            ctx,
            etapas=etapas,
            viajes=viajes,
            zonificaciones=zonificaciones,
            resoluciones=resoluciones,
            poligon_id=poligon_id,
        )

    if kpis:
        kpis = calculo_kpi_lineas(
            ctx, etapas=etapas, viajes=viajes
        )
```

- [ ] **Step 2: Run the new test**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_preparo_dashboard_passes_frames_by_reference -v
```

Expected: PASS.

- [ ] **Step 3: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 3: Commit

- [ ] **Step 1: Commit**

```
git add urbantrips/preparo_dashboard/preparo_dashboard.py urbantrips/tests/unit/test_preparo_dashboard.py
git commit -m "perf(b2): remove defensive DataFrame copies at orchestration level"
```
