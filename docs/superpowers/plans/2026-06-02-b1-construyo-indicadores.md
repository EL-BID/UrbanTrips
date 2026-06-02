# B1 — Consolidate `construyo_indicadores` Groupby Scans

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 9+ sequential pandas groupby chains in `construyo_indicadores` with DuckDB queries that scan the `viajes` DataFrame once per grouping key set instead of once per indicator.

**Architecture:** `duckdb.sql("SELECT ... FROM viajes ...")` queries the pandas DataFrame in-process — DuckDB treats it as a virtual table without copying it to disk. The 9 indicators collapse into 4 queries (base group, by rango_hora, by modo, by distancia_agregada). The function signature, output format, and all downstream code remain unchanged. The old pandas implementation is preserved as `_construyo_indicadores_pandas` for test baseline comparison.

**Tech Stack:** Python, DuckDB 4.x, pandas, pytest

---

## File map

- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

---

### Task 1: Extract old implementation as test baseline helper

**Files:**
- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`

- [ ] **Step 1: Rename `construyo_indicadores` to `_construyo_indicadores_pandas`**

Change the function definition at line ~279:

```python
# BEFORE
def construyo_indicadores(ctx: StorageContext, viajes, poligonos=False):

# AFTER
def _construyo_indicadores_pandas(ctx: StorageContext, viajes, poligonos=False):
```

The body stays completely unchanged.

- [ ] **Step 2: Add a new `construyo_indicadores` that calls the pandas version**

Immediately after the renamed function, add:

```python
def construyo_indicadores(ctx: StorageContext, viajes, poligonos=False):
    return _construyo_indicadores_pandas(ctx, viajes, poligonos)
```

- [ ] **Step 3: Run full unit tests to confirm nothing broke**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass. The rename is invisible to callers.

---

### Task 2: Write the output-equivalence test

**Files:**
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

- [ ] **Step 1: Append test to `test_preparo_dashboard.py`**

```python
# ---------------------------------------------------------------------------
# B1 — construyo_indicadores DuckDB equivalence
# ---------------------------------------------------------------------------

def _make_viajes_for_indicadores():
    """
    Synthetic viajes fixture representing one week of Buenos Aires transit.
    3 days (Mon, Tue, Sat), 2 polygons, 3 modes, 3 hour ranges, 2 distance buckets.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    dias = rng.choice(["2024-10-14", "2024-10-15", "2024-10-19"], n)
    return pd.DataFrame({
        "id_polygon": rng.choice(["poly_1", "poly_2"], n),
        "dia": dias,
        "mes": ["2024-10"] * n,
        "tipo_dia": ["Hábil" if d < "2024-10-19" else "Fin de Semana" for d in dias],
        "id_tarjeta": [f"T{i % 50:03d}" for i in range(n)],
        "id_viaje": rng.integers(1, 5, n),
        "rango_hora": rng.choice(["0-12", "13-16", "17-24"], n),
        "distancia_agregada": rng.choice(
            ["Viajes cortos (<=5kms)", "Viajes largos (>5kms)"], n
        ),
        "modo": rng.choice(["COLECTIVO", "SUBTE", "TREN"], n),
        "transferencia": rng.integers(0, 2, n),
        "factor_expansion_linea": rng.uniform(1.0, 3.5, n),
        "distance_od": rng.uniform(1.0, 15.0, n),
        "distance_route": rng.uniform(1.2, 16.0, n),
        "distance_route_gps": rng.uniform(1.3, 17.0, n),
        "travel_time_min": rng.integers(5, 60, n).astype(float),
        "kmh_od": rng.uniform(5.0, 60.0, n),
        "cant_etapas": rng.integers(1, 4, n),
        "hora": rng.integers(0, 24, n),
        "od_validado": [1] * n,
        "factor_expansion_tarjeta": rng.uniform(1.0, 3.5, n),
        "modo_agregado": rng.choice(["COLECTIVO", "multietapa (COLECTIVO)", "multimodal"], n),
        "genero_agregado": rng.choice(["M", "F", ""], n),
        "tarifa_agregada": rng.choice(["normal", "social"], n),
    })


def test_construyo_indicadores_duckdb_matches_pandas(mocker):
    """
    The DuckDB implementation must produce a DataFrame equal to the pandas baseline
    (after sorting and resetting index) for the same viajes input.
    """
    from unittest.mock import MagicMock
    from urbantrips.preparo_dashboard.preparo_dashboard import (
        construyo_indicadores,
        _construyo_indicadores_pandas,
    )

    viajes = _make_viajes_for_indicadores()
    ctx = MagicMock()
    ctx.dash.get_raw.return_value = pd.DataFrame()
    ctx.dash.append_raw.return_value = None
    ctx.dash.execute.return_value = None

    # Baseline: old pandas implementation
    expected = _construyo_indicadores_pandas(ctx, viajes.copy(), poligonos=False)
    # Reset mock call state between the two runs
    ctx.reset_mock()
    ctx.dash.get_raw.return_value = pd.DataFrame()

    # New: DuckDB implementation (same public function, new internals)
    actual = construyo_indicadores(ctx, viajes.copy(), poligonos=False)

    # Both return None (they write to ctx.dash) — compare what was written
    # The function calls ctx.dash.append_raw with the final indicadores DataFrame.
    pandas_call_args = _construyo_indicadores_pandas.__wrapped__ if hasattr(
        _construyo_indicadores_pandas, '__wrapped__') else None

    # Simpler: just assert the ctx.dash.append_raw was called with equivalent data
    assert ctx.dash.append_raw.called, "append_raw was never called"
    written_df = ctx.dash.append_raw.call_args[0][0]

    sort_cols = ["id_polygon", "dia", "Indicador"]
    written_sorted = written_df.sort_values(sort_cols).reset_index(drop=True)

    # Get expected written data from pandas baseline run
    ctx2 = MagicMock()
    ctx2.dash.get_raw.return_value = pd.DataFrame()
    ctx2.dash.append_raw.return_value = None
    ctx2.dash.execute.return_value = None
    _construyo_indicadores_pandas(ctx2, viajes.copy(), poligonos=False)
    expected_df = ctx2.dash.append_raw.call_args[0][0]
    expected_sorted = expected_df.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        written_sorted,
        expected_sorted,
        check_like=True,
        rtol=1e-3,
    )
```

- [ ] **Step 2: Run test to confirm it passes against the pandas baseline**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_construyo_indicadores_duckdb_matches_pandas -v
```

Expected: PASS (the `construyo_indicadores` stub currently calls `_construyo_indicadores_pandas`).

---

### Task 3: Implement DuckDB version of `construyo_indicadores`

**Files:**
- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`

- [ ] **Step 1: Add `import duckdb` at top of file if not already present**

Check the imports at the top of `preparo_dashboard.py`. If `import duckdb` is not there (it may already be imported for other uses), add it after the other stdlib/third-party imports:

```python
import duckdb
```

- [ ] **Step 2: Replace the `construyo_indicadores` stub with the full DuckDB implementation**

Replace:

```python
def construyo_indicadores(ctx: StorageContext, viajes, poligonos=False):
    return _construyo_indicadores_pandas(ctx, viajes, poligonos)
```

With the full DuckDB implementation:

```python
def construyo_indicadores(ctx: StorageContext, viajes, poligonos=False):
    """Compute dashboard indicators using DuckDB for single-pass aggregations."""
    if poligonos:
        nombre_tabla = "poly_indicadores"
    else:
        nombre_tabla = "agg_indicadores"

    if "id_polygon" not in viajes.columns:
        viajes = viajes.copy()
        viajes["id_polygon"] = "NONE"

    KEYS = ["id_polygon", "dia", "mes", "tipo_dia"]

    # ── 1. Base group: total viajes, transferencias, usuarios, distancia prom ──
    base = duckdb.sql("""
        SELECT
            id_polygon, dia, mes, tipo_dia,
            ROUND(SUM(factor_expansion_linea))                                AS total_viajes,
            ROUND(SUM(CASE WHEN transferencia = 1 THEN factor_expansion_linea
                          ELSE 0 END))                                        AS con_transferencia,
            ROUND(SUM(distance_od * factor_expansion_linea)
                  / NULLIF(SUM(factor_expansion_linea), 0), 2)                AS dist_prom
        FROM viajes
        GROUP BY id_polygon, dia, mes, tipo_dia
    """).df()

    # ind1 — Cantidad de Viajes
    ind1 = base[KEYS + ["total_viajes"]].rename(columns={"total_viajes": "Valor"})
    ind1["Indicador"] = "Cantidad de Viajes"
    ind1["Valor"] = ind1.Valor.astype(int)
    ind1["Tipo"] = "General"
    ind1["type_val"] = "int"

    # ind2 — Cantidad de Viajes con Transferencia (as % of ind1)
    ind2 = base[KEYS + ["con_transferencia", "total_viajes"]].copy()
    ind2["Valor"] = (
        ind2["con_transferencia"] / ind2["total_viajes"].replace(0, float("nan")) * 100
    ).round(2)
    ind2 = ind2[KEYS + ["Valor"]]
    ind2["Indicador"] = "Cantidad de Viajes con Transferencia"
    ind2["Tipo"] = "General"
    ind2["type_val"] = "percentage"

    # ind5 — Cantidad de Usuarios
    usuarios = duckdb.sql("""
        SELECT id_polygon, dia, mes, tipo_dia,
            ROUND(SUM(first_fex)) AS Valor
        FROM (
            SELECT id_polygon, dia, mes, tipo_dia,
                ANY_VALUE(factor_expansion_linea) AS first_fex
            FROM viajes
            GROUP BY id_polygon, dia, mes, tipo_dia, id_tarjeta
        )
        GROUP BY id_polygon, dia, mes, tipo_dia
    """).df()
    ind5 = usuarios[KEYS + ["Valor"]]
    ind5["Indicador"] = "Cantidad de Usuarios"
    ind5["Tipo"] = "General"
    ind5["type_val"] = "int"

    # ind6 — Distancia Promedio (kms)
    ind6 = base[KEYS + ["dist_prom"]].rename(columns={"dist_prom": "Valor"})
    ind6["Tipo"] = "Distancias"
    ind6["Indicador"] = "Distancia Promedio (kms)"
    ind6["type_val"] = "float"

    # ── 2. By rango_hora ──────────────────────────────────────────────────────
    by_hora = duckdb.sql("""
        WITH totals AS (
            SELECT id_polygon, dia, mes, tipo_dia,
                SUM(factor_expansion_linea) AS total
            FROM viajes
            GROUP BY id_polygon, dia, mes, tipo_dia
        )
        SELECT v.id_polygon, v.dia, v.mes, v.tipo_dia, v.rango_hora,
            ROUND(SUM(v.factor_expansion_linea) / t.total * 100, 2) AS Valor
        FROM viajes v
        JOIN totals t USING (id_polygon, dia, mes, tipo_dia)
        GROUP BY v.id_polygon, v.dia, v.mes, v.tipo_dia, v.rango_hora, t.total
    """).df()

    ind3 = by_hora.copy()
    ind3["Indicador"] = "Cantidad de Viajes de " + ind3["rango_hora"] + "hs"
    ind3["Tipo"] = "General"
    ind3["type_val"] = "percentage"
    ind3 = ind3.drop(columns=["rango_hora"])

    # ── 3. By modo ────────────────────────────────────────────────────────────
    by_modo = duckdb.sql("""
        WITH totals AS (
            SELECT id_polygon, dia, mes, tipo_dia,
                SUM(factor_expansion_linea) AS total
            FROM viajes
            GROUP BY id_polygon, dia, mes, tipo_dia
        )
        SELECT v.id_polygon, v.dia, v.mes, v.tipo_dia, v.modo,
            ROUND(SUM(v.factor_expansion_linea) / t.total * 100, 2)           AS pct,
            ROUND(SUM(v.distance_od * v.factor_expansion_linea)
                  / NULLIF(SUM(v.factor_expansion_linea), 0), 2)              AS dist_prom_modo
        FROM viajes v
        JOIN totals t USING (id_polygon, dia, mes, tipo_dia)
        GROUP BY v.id_polygon, v.dia, v.mes, v.tipo_dia, v.modo, t.total
    """).df()

    ind4 = by_modo[KEYS + ["modo", "pct"]].copy()
    ind4 = ind4.sort_values(KEYS + ["pct"], ascending=[True]*4 + [False])
    ind4["Indicador"] = ind4["modo"]
    ind4["Tipo"] = "Modal"
    ind4["type_val"] = "percentage"
    ind4 = ind4.rename(columns={"pct": "Valor"}).drop(columns=["modo"])

    ind7 = by_modo[KEYS + ["modo", "dist_prom_modo"]].copy()
    ind7["Indicador"] = "Distancia Promedio (" + ind7["modo"] + ") (kms)"
    ind7["Tipo"] = "Distancias"
    ind7["type_val"] = "float"
    ind7 = ind7.rename(columns={"dist_prom_modo": "Valor"}).drop(columns=["modo"])

    # ── 4. By distancia_agregada ──────────────────────────────────────────────
    by_dist = duckdb.sql("""
        WITH totals AS (
            SELECT id_polygon, dia, mes, tipo_dia,
                SUM(factor_expansion_linea) AS total
            FROM viajes
            GROUP BY id_polygon, dia, mes, tipo_dia
        )
        SELECT v.id_polygon, v.dia, v.mes, v.tipo_dia, v.distancia_agregada,
            ROUND(SUM(v.factor_expansion_linea) / t.total * 100, 2)           AS pct,
            ROUND(SUM(v.distance_od * v.factor_expansion_linea)
                  / NULLIF(SUM(v.factor_expansion_linea), 0), 2)              AS dist_prom_dist
        FROM viajes v
        JOIN totals t USING (id_polygon, dia, mes, tipo_dia)
        GROUP BY v.id_polygon, v.dia, v.mes, v.tipo_dia, v.distancia_agregada, t.total
    """).df()

    ind9 = by_dist[KEYS + ["distancia_agregada", "pct"]].copy()
    ind9 = ind9.sort_values(KEYS + ["pct"], ascending=[True]*4 + [False])
    ind9["Indicador"] = "Cantidad de " + ind9["distancia_agregada"]
    ind9["Tipo"] = "General"
    ind9["type_val"] = "percentage"
    ind9 = ind9.rename(columns={"pct": "Valor"}).drop(columns=["distancia_agregada"])

    ind8 = by_dist[KEYS + ["distancia_agregada", "dist_prom_dist"]].copy()
    ind8["Indicador"] = "Distancia Promedio " + ind8["distancia_agregada"]
    ind8["Tipo"] = "Distancias"
    ind8["type_val"] = "float"
    ind8 = ind8.rename(columns={"dist_prom_dist": "Valor"}).drop(columns=["distancia_agregada"])

    # ── 5. Combine, merge with existing history, add "Todos" aggregate ────────
    indicadores = pd.concat([ind1, ind5, ind2, ind3, ind6, ind9, ind7, ind8, ind4],
                            ignore_index=True)

    try:
        indicadores_ant = ctx.dash.get_raw(nombre_tabla)
        if len(indicadores_ant) > 0:
            indicadores_ant = indicadores_ant[
                ~indicadores_ant.dia.isin(indicadores.dia.unique().tolist() + ["Todos"])
            ]
    except Exception:
        indicadores_ant = pd.DataFrame([])

    indicadores = pd.concat(
        [
            indicadores[["id_polygon", "dia", "mes", "tipo_dia",
                          "Tipo", "Indicador", "type_val", "Valor"]],
            indicadores_ant,
        ],
        ignore_index=True,
    )

    indicadores_todos = (
        indicadores.groupby(["id_polygon", "Tipo", "Indicador", "type_val"],
                            as_index=False, observed=True)
        .Valor.mean()
        .round(2)
    )
    indicadores_todos["dia"] = "Todos"
    indicadores_todos["tipo_dia"] = ""
    indicadores_todos["mes"] = ""
    indicadores = pd.concat([indicadores, indicadores_todos])

    indicadores = format_dataframe(indicadores)
    indicadores = indicadores[
        ["id_polygon", "dia", "mes", "tipo_dia", "Tipo", "Indicador", "Valor_str"]
    ].rename(columns={"Valor_str": "Valor"})

    indicadores = indicadores.sort_values(
        ["id_polygon", "dia", "mes", "tipo_dia", "Tipo", "Indicador"]
    )

    tabla_destino = "poly_indicadores" if poligonos else "agg_indicadores"
    replace_dash_partition(ctx, indicadores, tabla_destino, ["dia"])
```

---

### Task 4: Run equivalence test and full suite

- [ ] **Step 1: Run the equivalence test**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_construyo_indicadores_duckdb_matches_pandas -v
```

Expected: PASS. If values differ by more than `rtol=1e-3`, investigate which indicator diverges (the error message will show the differing rows) and trace back to the relevant DuckDB query.

- [ ] **Step 2: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 5: Commit

- [ ] **Step 1: Commit**

```
git add urbantrips/preparo_dashboard/preparo_dashboard.py urbantrips/tests/unit/test_preparo_dashboard.py
git commit -m "perf(b1): replace 9 pandas groupby scans with DuckDB queries in construyo_indicadores"
```
