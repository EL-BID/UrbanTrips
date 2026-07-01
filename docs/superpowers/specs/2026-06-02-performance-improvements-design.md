# Performance improvements — architectural bottlenecks (B1–B6)

**Date:** 2026-06-02
**Branch base:** `feat/dash-improvement`
**Context:** A full pipeline run takes ~3 days. The DuckDB migration covered storage only; computation still happens entirely in pandas. This spec covers the six architectural bottlenecks identified in `PERF_REVIEW.md` Part 1.

---

## Scope

Six independent improvements, each in its own git worktree, each with output-equivalence tests.
Micro-optimisations (PERF_REVIEW.md Part 2) are out of scope for this spec.

---

## Overall structure

### Worktrees

| Branch | Improvement | Primary file |
|---|---|---|
| `perf/b1-construyo-indicadores` | B1: consolidate 9 groupby → single DuckDB query | `preparo_dashboard/preparo_dashboard.py` |
| `perf/b2-remove-copies` | B2: remove `.copy()` at orchestration level | `preparo_dashboard/preparo_dashboard.py` |
| `perf/b3-dbscan-grid` | B3: DBSCAN grid reduction + early stopping + tuning config | `cluster/dbscan.py`, `configs/tuning.yaml` |
| `perf/b4-h3-polygon-fill` | B4: replace iterrows+concat with `h3.geo_to_cells` | `preparo_dashboard/geo.py` |
| `perf/b5-derived-columns-sql` | B5: push scalar derived columns into DuckDB SQL | `preparo_dashboard/preparo_dashboard.py` |
| `perf/b6-persistent-connection` | B6: persistent DuckDB connection in adapters | `storage/adapters/duckdb/insumos.py` |

All branches are independent and can be implemented in parallel. Merge order: B2 and B6 first (no logic change), then B4, B3, B5, B1 (ascending risk).

### Test strategy

Each worktree adds tests to the relevant existing test file. The pattern is **output-equivalence**: run the old code path and new code path on the same synthetic input, assert identical output (or identical schema for B3 where cluster label values may differ).

**Synthetic fixture shape (Buenos Aires public transit context):**
- ~300 legs (`etapas`), ~150 trips (`viajes`)
- 3 bus lines (`colectivo`), 1 metro line (`subte`)
- 5 days: Mon–Fri + 1 Saturday
- 2 polygons (e.g. area around Plaza de Mayo and Palermo)
- H3 cells at resolution 8–9, centred on Buenos Aires (~−34.60, −58.45)
- Modes: `COLECTIVO`, `SUBTE`, `TREN`
- Tariff types: standard SUBE and SUBE social (red)
- Expansion factors in range 1.0–3.5

---

## B1 · `construyo_indicadores` — consolidate 9 groupby scans

**File:** `urbantrips/preparo_dashboard/preparo_dashboard.py:279`
**Test file:** `urbantrips/tests/unit/test_preparo_dashboard.py`

### Problem

The function builds `ind1`–`ind9` as 9 independent `viajes.groupby(...)` chains, each scanning the full DataFrame. It is called twice per run (lines 2437 and 2155), totalling 18 full-table scans of `viajes`.

### Solution

Replace the 9 groupby chains with a single `duckdb.sql()` query that references the `viajes` DataFrame directly (DuckDB can query pandas DataFrames in-process without copying to disk). All 9 indicators are computed in one pass using `FILTER (WHERE ...)` conditional aggregates.

The result is reshaped into the same long-format DataFrame (`id_polygon, dia, mes, tipo_dia, Tipo, Indicador, Valor`) as today.

### Interface contract

- **Signature unchanged:** `construyo_indicadores(ctx, viajes, poligonos=False)`
- **Output unchanged:** same columns, same dtypes, same values
- **No callers change**

### Test

1. Build synthetic `viajes` fixture (~150 rows)
2. Extract old pandas implementation as `_construyo_indicadores_pandas()` (test-only helper)
3. Run both implementations on the same input
4. Assert output DataFrames are equal after sorting by `(dia, Indicador)`

---

## B2 · Remove `.copy()` at orchestration level

**File:** `urbantrips/preparo_dashboard/preparo_dashboard.py:2467–2487`
**Test file:** `urbantrips/tests/unit/test_preparo_dashboard.py`

### Problem

The top-level orchestrator passes deep copies of `etapas` and `viajes` to three functions, creating 2–4 GB of unnecessary RAM overhead. All three functions write to DuckDB tables — they don't mutate the caller's DataFrames.

### Solution

Remove the three `.copy()` calls:

```python
# Before
proceso_lineas_deseo(ctx, etapas=etapas.copy(), viajes=viajes.copy(), ...)
proceso_poligonos(   ctx, etapas=etapas.copy(), viajes=viajes.copy(), ...)
calculo_kpi_lineas(  ctx, etapas=etapas.copy(), viajes=viajes.copy())

# After
proceso_lineas_deseo(ctx, etapas=etapas, viajes=viajes, ...)
proceso_poligonos(   ctx, etapas=etapas, viajes=viajes, ...)
calculo_kpi_lineas(  ctx, etapas=etapas, viajes=viajes)
```

Internal `.copy()` calls within those functions that guard against actual mutation (e.g. line 1081) are left untouched.

### Interface contract

No interface changes. Behaviour unchanged.

### Test

Call the orchestrator with a synthetic `StorageContext` and assert the output tables are written correctly — confirming the functions work without defensive copies at the call site.

---

## B3 · DBSCAN grid reduction + early stopping + tuning config

**Files:** `urbantrips/cluster/dbscan.py:232–248`, new `configs/tuning.yaml`, new `configs/tuning.yaml.example`, `urbantrips/utils/utils.py` (or equivalent config loader)
**Test file:** `urbantrips/tests/unit/test_preparo_dashboard.py` or new `urbantrips/tests/unit/test_dbscan.py`

### Problem

The parameter search runs a 20×20 grid (400 DBSCAN fits) per route direction and never exits early. With ~100 directions per run, this is ~40,000 DBSCAN iterations. All constants are magic numbers in source code.

### Solution

**1. Configurable tuning file**

A new optional `configs/tuning.yaml`:

```yaml
# configs/tuning.yaml
# Advanced performance tuning — safe to leave at defaults for most runs.

dbscan:
  grid_steps: 5               # steps per axis; grid size = grid_steps²
  early_stop_silhouette: 0.7  # stop search if silhouette exceeds this threshold
```

A `leer_configs_tuning()` function added to `urbantrips/utils/utils.py` (alongside `leer_configs_generales`) loads the file if present, merging over hardcoded defaults:

```python
TUNING_DEFAULTS = {
    "dbscan": {"grid_steps": 5, "early_stop_silhouette": 0.7},
}
```

`configs/tuning.yaml.example` is committed to the repo as documentation. The `.gitignore` already has `configs/*` with a single exception (`configuraciones_generales.yaml`); a second exception `!configs/tuning.yaml.example` is added so the example file is tracked. The actual `configs/tuning.yaml` (with local overrides) remains gitignored by the existing rule.

**2. Grid reduction**

```python
cfg = leer_configs_tuning()
grid_steps = cfg["dbscan"]["grid_steps"]

min_samples_range = list(map(int, w.sum() * np.linspace(0.01, 0.5, grid_steps)))
eps_range = np.linspace(0.01, 0.5, grid_steps)        # lrs
# eps_range = np.linspace(100, 1000, grid_steps)       # 4d
```

**3. Early stopping**

```python
early_stop = cfg["dbscan"]["early_stop_silhouette"]

for eps in eps_range:
    for min_samples in min_samples_range:
        ...
        if best_silhouette_score >= early_stop:
            break
    if best_silhouette_score >= early_stop:
        break
```

### Interface contract

- Output columns (`k_max_groups`, `k_max_silhouette`, `k_min_noise`) unchanged
- Cluster label **values** may differ from the 20×20 run (acceptable: fresh independent analysis)
- All downstream consumers unchanged

### Test

1. Build synthetic legs DataFrame with two clear spatial clusters along a Buenos Aires corridor
2. Run the new grid search (default `grid_steps=5`)
3. Assert: correct output columns, all labels are integers ≥ −1, runtime with 5×5 is measurably faster than 20×20 (timed with `time.perf_counter`)
4. Assert: `leer_configs_tuning()` returns defaults when no file is present; returns overrides when file exists

---

## B4 · `select_h3_from_polygon` — replace point-grid with `h3.geo_to_cells`

**File:** `urbantrips/preparo_dashboard/geo.py:55–92`
**Test file:** `urbantrips/tests/unit/test_preparo_dashboard.py`

### Problem

The function fills each polygon with a cartesian product of grid `Point` objects (potentially millions), then maps each point to its H3 cell via a spatial join. `pd.concat` inside the polygon loop causes quadratic memory behaviour.

### Solution

Replace the point-grid approach with `h3.geo_to_cells` (H3's native polygon-fill). The outer loop over the handful of polygon rows is kept — only the inner cartesian product and quadratic concat are removed:

```python
from shapely.geometry import mapping

def select_h3_from_polygon(poly, res=8, spacing=0.0001, viz=False):
    poly = poly.reset_index(drop=True).to_crs(4326)
    records = []
    for i, row in poly.iterrows():
        geojson = mapping(row.geometry)
        cells = h3.geo_to_cells(geojson, res)
        records.extend({"id": row.id, "h3": cell} for cell in cells)

    points_result = pd.DataFrame(records)
    gdf_hexs = h3_to_geodataframe(points_result.h3).rename(columns={"h3_index": "h3"})
    gdf_hexs = (
        gdf_hexs.merge(points_result, on="h3")[["id", "h3", "geometry"]]
        .sort_values(["id", "h3"])
        .reset_index(drop=True)
    )
    ...
```

The `spacing` parameter is kept in the signature (with a deprecation note in the docstring) to avoid breaking callers that pass it explicitly.

### Interface contract

- **Signature unchanged:** `select_h3_from_polygon(poly, res=8, spacing=0.0001, viz=False)`
- **Return value unchanged:** GeoDataFrame with columns `id`, `h3`, `geometry`
- All callers unchanged

### Test

1. Build a synthetic GeoDataFrame with 2 Buenos Aires polygons (small areas around Plaza de Mayo and Palermo)
2. Run the new function
3. Assert: all returned H3 cells are valid (`h3.is_valid_cell`), all at the requested resolution, `id` column correctly maps cells to source polygon, result is a GeoDataFrame with CRS 4326

---

## B5 · Derived columns pushed into DuckDB SQL

**File:** `urbantrips/preparo_dashboard/preparo_dashboard.py:65–274`
**Test file:** `urbantrips/tests/unit/test_preparo_dashboard.py`

### Problem

After reading `etapas` and `viajes` from DuckDB, `load_and_process_data` computes ~12 derived columns in pandas on the fully materialised DataFrames. Several of these are scalar expressions that DuckDB can compute before materialisation, reducing the size of the DataFrames that hit Python memory.

### Solution

Move the following columns into the DuckDB `SELECT`:

| Column | Table | SQL expression |
|---|---|---|
| `tipo_dia` | both | `CASE WHEN DAYOFWEEK(CAST(dia AS DATE)) >= 6 THEN 'Fin de Semana' ELSE 'Hábil' END` |
| `mes` | both | `STRFTIME(CAST(dia AS DATE), '%Y-%m')` |
| `rango_hora` | both | `CASE WHEN hora BETWEEN 13 AND 16 THEN '13-16' WHEN hora > 16 THEN '17-24' ELSE '0-12' END` |
| `distancia_agregada` | viajes | `CASE WHEN distance_od > 5 THEN 'Viajes largos (>5kms)' ELSE 'Viajes cortos (<=5kms)' END` |
| `distancia_agregada` | etapas | `CASE WHEN distance_od > 5 THEN 'Etapa larga (>5kms)' ELSE 'Etapa corta (<=5kms)' END` |
| `transferencia` | viajes | `CAST(cant_etapas > 1 AS INTEGER)` |
| `kmh_od` | both | `CASE WHEN travel_time_min > 0 THEN ROUND(distance_od / (travel_time_min / 60.0), 1) END` |

Columns that remain in pandas (require a groupby+merge or external function):
- `tarifa_agregada`, `genero_agregado` — use existing Python classifier functions; inlining would require DuckDB UDF registration
- `modo_agregado` — requires a cross-row groupby; stays as-is
- `Fecha`, `Fecha_next`, `diff_time` — datetime window computation; cleaner in pandas

### Interface contract

- **Signature unchanged:** `load_and_process_data(ctx) → (etapas, viajes)`
- **Output unchanged:** same columns, same dtypes, same values
- All callers unchanged

### Test

1. Build synthetic `etapas` and `viajes` tables in an in-memory DuckDB
2. Run `load_and_process_data` with a mock `StorageContext` pointing at it
3. Assert each moved column matches expected values (e.g. `tipo_dia == 'Hábil'` for Monday, `rango_hora == '13-16'` for `hora=14`)
4. Parametrize to cover boundary cases: weekend rows, boundary hours (12, 13, 16, 17), zero `travel_time_min`

---

## B6 · Persistent DuckDB connection in adapters

**File:** `urbantrips/storage/adapters/duckdb/insumos.py` (and equivalent data/dash adapters if they follow the same pattern)
**Test file:** `urbantrips/tests/unit/test_preparo_dashboard.py` or adapter-level test

### Problem

`_conn()` opens a new DuckDB file connection on every method call and closes it immediately after. Hundreds of open/close cycles per run.

### Solution

Open the connection once at construction, reuse it across all method calls, close on explicit `close()` or `__del__`:

```python
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = duckdb.connect(str(self._path))
    self._apply_schema()

def close(self) -> None:
    if self._conn:
        self._conn.close()
        self._conn = None

def __del__(self) -> None:
    self.close()
```

All methods replace `with self._conn() as conn:` with `self._conn`. The same change is applied to the data and dash adapters if they have the same pattern.

### Interface contract

No public interface changes. All adapter methods behave identically.

### Test

Create a `DuckDBInsumoAdapter` against a temp file, perform several read/write cycles interleaved, assert results are consistent. Assert `close()` is idempotent (calling twice does not raise).

---

## Merge order

1. `perf/b6-persistent-connection` — isolated to adapter, no logic change
2. `perf/b2-remove-copies` — isolated to orchestrator call site, no logic change
3. `perf/b4-h3-polygon-fill` — isolated to geo helper
4. `perf/b3-dbscan-grid` — isolated to cluster module + new config file
5. `perf/b5-derived-columns-sql` — changes SQL queries in `load_and_process_data`
6. `perf/b1-construyo-indicadores` — changes computation in `construyo_indicadores`

---

## Out of scope

- PERF_REVIEW.md Part 2 micro-optimisations (separate effort)
- `tarifa_agregada` / `genero_agregado` as DuckDB UDFs
- `modo_agregado` window function refactor
- Parallelising the polygon loop (`proceso_lineas_deseo`)
- Streaming / chunked processing of `etapas` and `viajes`
