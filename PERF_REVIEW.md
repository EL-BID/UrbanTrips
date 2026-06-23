# Performance review

## Part 1 — Architectural bottlenecks (June 2026)

The project migrated storage from SQLite to DuckDB, but the migration is **at the I/O layer only**.
Heavy computation still happens entirely in pandas. DuckDB is used as a fast file format, not as
a query engine. The bottlenecks below are architectural and compound each other; fixing them is
independent of the micro-optimisation checklist in Part 2.

A full 3-day run on a reasonably-sized dataset is the motivating context.

### Priority table

| # | Location | What | Effort | Impact |
|---|----------|------|--------|--------|
| P0 | `preparo_dashboard.py:279` | 9 sequential groupby scans, called twice | ~3 h | Very high |
| P0 | `preparo_dashboard.py:2467–2487` | 3 large DataFrame `.copy()` calls | 15 min | High |
| P1 | `cluster/dbscan.py:244` | 20×20 DBSCAN parameter grid per route | ~1 h | High |
| P1 | `preparo_dashboard.py:101` | Derived columns computed in pandas, not SQL | ~4 h | Medium-high |
| P2 | `preparo_dashboard/geo.py:62` | `iterrows` + `pd.concat` inside polygon loop | ~2 h | Medium |
| P3 | `storage/adapters/duckdb/insumos.py:23` | New DB connection per adapter call | 30 min | Low |

---

### B1 · `construyo_indicadores` — 9 sequential full-table scans, called twice

**[`preparo_dashboard.py:279`](urbantrips/preparo_dashboard/preparo_dashboard.py)**

`construyo_indicadores` builds `ind1`–`ind9` as 9 independent `viajes.groupby(...)` calls.
Each call does a complete in-memory scan of the full `viajes` DataFrame. The function is called
from two places in the same run:

- line 2437 inside `proceso_lineas_deseo`
- line 2155 inside `proceso_poligonos`

That is **18 full scans of `viajes`** per run. All 9 indicators share the same base grouping keys
(`id_polygon`, `dia`, `mes`, `tipo_dia`) so they can be computed in a single DuckDB query using
conditional aggregates:

```sql
SELECT
  id_polygon, dia, mes, tipo_dia,
  SUM(factor_expansion_linea)                                          AS total_viajes,
  SUM(CASE WHEN transferencia = 1 THEN factor_expansion_linea END)     AS con_transferencia,
  SUM(CASE WHEN rango_hora = '13-16' THEN factor_expansion_linea END)  AS rango_13_16,
  -- …remaining indicators…
FROM viajes
GROUP BY id_polygon, dia, mes, tipo_dia
```

Result materialised once, then reshaped to the long format the dashboard expects. Estimated
speedup: **8–9×** for this function.

---

### B2 · Three large DataFrame copies at the orchestration level

**[`preparo_dashboard.py:2463–2488`](urbantrips/preparo_dashboard/preparo_dashboard.py)**

```python
proceso_lineas_deseo(ctx, etapas=etapas.copy(), viajes=viajes.copy(), ...)  # lines 2467–2468
proceso_poligonos(  ctx, etapas=etapas.copy(), viajes=viajes.copy(), ...)  # lines 2477–2478
calculo_kpi_lineas( ctx, etapas=etapas.copy(), viajes=viajes.copy())        # line 2487
```

Three deep copies of multi-GB DataFrames, sequentially, before any processing starts. The copies
exist to prevent mutation leaking across the three branches, but in practice none of the three
functions mutates the inputs in a way that would affect the others (all writes go to DuckDB tables,
not back to the caller's variables). Removing `.copy()` here eliminates **2–4 GB of peak RAM
pressure** with a trivial change.

---

### B3 · DBSCAN 20×20 parameter grid — 400 fits per route direction

**[`cluster/dbscan.py:232–248`](urbantrips/cluster/dbscan.py)**

```python
min_samples_range = list(map(int, w.sum() * np.linspace(0.01, 0.5, 20)))  # 20 values
eps_range = np.linspace(0.01, 0.5, 20)                                     # 20 values

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(...).fit(X, sample_weight=w)                       # 400 fits
```

Then 3 more `.fit()` calls at line 297 for the best-of-three params. **403 DBSCAN fits per
direction.** With 50 lines × 2 directions = 100 directions, that is ~40,000 DBSCAN iterations
per run. DBSCAN is O(N log N) with a spatial index but degrades to O(N²) for dense or
high-dimensional data.

Two complementary fixes:
1. **Reduce the grid** from 20×20 to 5×5 (25 fits). The response surface for silhouette vs eps/min_samples
   is smooth enough that a coarse grid finds equivalent params.
2. **Early stopping**: break as soon as `silhouette_score > 0.7` (or another threshold). The grid
   currently always exhausts all 400 combinations.

Estimated speedup on the KPI phase: **80–90%**.

---

### B4 · `select_h3_from_polygon` — `iterrows` + `pd.concat` inside loop

**[`preparo_dashboard/geo.py:62–70`](urbantrips/preparo_dashboard/geo.py)**

```python
for i, row in poly.iterrows():
    points = [Point(x, y) for x in x_coords for y in y_coords]   # full cartesian product
    pts = gpd.GeoDataFrame(geometry=points, crs=4326)
    points_result = pd.concat([points_result, pts])               # O(n²) — reallocates every iteration
```

Two anti-patterns stacked: `iterrows` (Python-speed row iteration) and `pd.concat` inside a loop
(reallocates the entire growing DataFrame on each call). For a polygon spanning 0.1° × 0.1° with
`spacing=0.0001` that is ~1 million points; with multiple polygons the concat loop hits quadratic
memory and time.

Quick fix: collect all GeoDataFrames in a list, `pd.concat` once after the loop.

Better fix: use `h3.polyfill` (or `h3.geo_to_cells`) directly — H3's native polygon fill returns
hex IDs without any point sampling, making the whole function a few lines.

---

### B5 · Derived columns computed in pandas instead of SQL

**[`preparo_dashboard.py:101–193`](urbantrips/preparo_dashboard/preparo_dashboard.py)**

After reading `etapas` and `viajes` from DuckDB, `load_and_process_data` computes ~12 derived
columns in pandas on the full materialised DataFrames:

| Column | SQL equivalent |
|--------|---------------|
| `tarifa_agregada` | `CASE WHEN tarifa IN (...) THEN 'social' ELSE 'regular' END` |
| `rango_hora` | `CASE WHEN hora BETWEEN 13 AND 16 THEN '13-16' … END` |
| `distancia_agregada` | `CASE WHEN distance_od > 5 THEN 'largo' ELSE 'corto' END` |
| `tipo_dia` | `CASE WHEN DAYOFWEEK(dia) >= 6 THEN 'Fin de Semana' ELSE 'Hábil' END` |
| `mes` | `STRFTIME(dia, '%Y-%m')` |

Computing these inside the DuckDB `SELECT` means a **smaller DataFrame is materialised in the
first place**. The `modo_agregado` derivation (lines 168–189: groupby + merge back) maps directly
to a SQL window function:

```sql
CASE
  WHEN COUNT(DISTINCT modo) OVER (PARTITION BY dia, id_tarjeta, id_viaje) > 1 THEN 'multimodal'
  WHEN COUNT(*) OVER (PARTITION BY dia, id_tarjeta, id_viaje) > 1 THEN 'multietapa (' || modo || ')'
  ELSE modo
END AS modo_agregado
```

---

### B6 · New DuckDB connection opened per adapter method call

**[`storage/adapters/duckdb/insumos.py:23`](urbantrips/storage/adapters/duckdb/insumos.py)**

```python
def _conn(self) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(self._path))
```

Every adapter method calls `_conn()` inside a `with` block, opening and closing a connection to
the DuckDB file each time. DuckDB handles this cheaply, but for a pipeline that calls the adapter
hundreds of times a persistent connection (opened once in `__init__`, closed in `__del__` or a
context manager) is cleaner and avoids repeated file-open overhead. Minor relative to B1–B5.

---

## Part 2 — Micro-optimisation checklist

Each step covers one file. Read only that file, apply the fix, run tests, move on.
Patterns already fixed in `preparo_dashboard.py` and `utils/dataframe.py` are marked ✅.

**Status (June 2026): all steps complete.** Steps 1–4 and 6–8 were already applied in the
`feat/dash-improvement` branch before this checklist was executed. Step 5 (`viz/viz.py:1582-1583`)
was the only remaining instance and was fixed last.

---

## Patterns to hunt

| ID | Pattern | Fix |
|----|---------|-----|
| A | `series.apply(h3toparent / h3.cell_to_parent, ...)` | list comprehension |
| B | `series.apply(lambda x: pd.Series(h3_to_lat_lon(x)))` | `np.array([...])` |
| C | `groupby().apply(lambda x: np.average(x[col], weights=x[w]))` | DuckDB `SUM(col*w)/SUM(w)` |
| D | duplicate `calculate_weighted_means` | delete, import from `utils.dataframe` |
| E | `series.apply(lambda geom: geom.wkt)` | `.to_wkt()` vectorized method |
| F | `groupby().apply(lambda row: h3.grid_distance(...), axis=1)` | list comprehension or vectorize |

---

## Step 1 — `urbantrips/preparo_dashboard/geo.py` · Pattern A

**Line 138:**
```python
poly_sel[f"zona_{i}"] = poly_sel["h3"].apply(h3toparent, res=resol)
```
**Fix:**
```python
poly_sel[f"zona_{i}"] = [h3toparent(x, res=resol) for x in poly_sel["h3"]]
```
Identical to the fix already applied in `preparo_dashboard.py`. One-liner change.

---

## Step 2 — `urbantrips/geo/geo.py` · Pattern A (function `normalizo_lat_lon`)

**Lines 161, 165:**
```python
df["origin"] = df[h3_o].apply(h3togeo)
df["destination"] = df[h3_d].apply(h3togeo)
```
`h3togeo` is a scalar wrapper around `h3.cell_to_latlng`. Fix:
```python
df["origin"] = [h3togeo(x) for x in df[h3_o]]
df["destination"] = [h3togeo(x) for x in df[h3_d]]
```
Check: what does `h3togeo` return? If it returns a (lat, lon) tuple, the downstream `.apply(bring_latlon, latlon=...)` calls (lines ~162–168) may also need to be replaced with index access.

---

## Step 3 — `urbantrips/dashboard/dash_utils.py` · Pattern D + Pattern C

**Lines 142–192:** Contains a full duplicate `calculate_weighted_means` (older pandas N-pass version).
- Delete it and import from `urbantrips.utils.dataframe`.
- Check all callers in the same file (lines 288, 335, 1154) — they should work unchanged.

**Lines 1479:** `for k in range(k_max, 1, -1):` — read context to check if this is a DataFrame loop or just a numeric iteration (likely numeric, may be fine).

**Lines ~74, ~82 in `urbantrips/viz/helpers.py` · Pattern C:**
```python
.apply(lambda x: np.average(x["longitud"], weights=x["fex"]))
.apply(lambda x: np.average(x["latitud"], weights=x["fex"]))
```
These are grouped weighted means. Replace with a DuckDB query or with pandas:
```python
# pandas vectorized alternative (no DuckDB needed):
grouped["longitud"] = (grouped["longitud"] * grouped["fex"]).groupby(...).sum() / grouped["fex"].groupby(...).sum()
```
Or use `calculate_weighted_means` directly if the calling context suits it.

---

## Step 4 — `urbantrips/viz/helpers.py` · Pattern A + C

**Line 63:**
```python
zonas["h3_o_tmp"] = zonas["h3"].apply(h3.cell_to_parent, res=res)
```
**Fix:** `[h3.cell_to_parent(x, res) for x in zonas["h3"]]`

**Line 88:**
```python
df["h3_o_tmp"] = df[h3_o].apply(h3.cell_to_parent, res=res)
```
Same fix.

**Lines 74, 82 — Pattern C** (same as Step 3 above, fix together with the file).

---

## Step 5 — `urbantrips/viz/viz.py` · Pattern A (6 instances)

**Lines 702–703, 895–896, 1581–1582:**
```python
zonas["h3_r6"] = zonas["h3"].apply(h3.cell_to_parent, res=6)
zonas["h3_r7"] = zonas["h3"].apply(h3.cell_to_parent, res=7)
```
All six are identical. Fix:
```python
zonas["h3_r6"] = [h3.cell_to_parent(x, 6) for x in zonas["h3"]]
zonas["h3_r7"] = [h3.cell_to_parent(x, 7) for x in zonas["h3"]]
```
These are in visualization functions so impact is lower, but the fix is trivial.

**Lines 1985, 1994 — Pattern C:**
```python
.apply(lambda x: np.average(x["longitud"], weights=x["fex"]))
.apply(lambda x: np.average(x["latitud"], weights=x["fex"]))
```
Same weighted-mean-via-apply pattern. Same fix as Step 3/4.

---

## Step 6 — `urbantrips/kpi/kpi.py` · Pattern F

**Line 1567:**
```python
speed.apply(lambda row: h3.grid_distance(row["h3"], row["h3_lag"]), axis=1)
```
Row-wise `axis=1` apply — one of the slowest pandas patterns. Fix with list comprehension:
```python
[h3.grid_distance(h3_, lag) for h3_, lag in zip(speed["h3"], speed["h3_lag"])]
```
Read ~10 lines of context first to confirm column names and that `speed` is not None-guarded.

---

## Step 7 — `urbantrips/viz/overlapping.py` + `urbantrips/storage/adapters/duckdb/insumos.py` · Pattern E

**`overlapping.py` lines 186, 224:**
```python
base_gdf_to_db["wkt"] = base_gdf_to_db["geometry"].apply(lambda geom: geom.wkt)
comp_gdf_to_db["wkt"] = comp_gdf_to_db["geometry"].apply(lambda geom: geom.wkt)
```
**`insumos.py` line 36:**
```python
df["wkt"] = gdf.geometry.apply(lambda g: g.wkt)
```
All three: geopandas has a vectorized `.to_wkt()`:
```python
base_gdf_to_db["wkt"] = base_gdf_to_db.geometry.to_wkt()
df["wkt"] = gdf.geometry.to_wkt()
```

---

## Step 8 — `urbantrips/cluster/dbscan.py` · Pattern A (low priority)

**Line 489:**
```python
geoms = legs.apply(lambda row: LineString([row.o, row.d]), axis=1)
```
Row-wise geometry construction. Fix with list comprehension:
```python
geoms = [LineString([o, d]) for o, d in zip(legs["o"], legs["d"])]
```
Confirm column names `o` and `d` are the actual point columns before changing.

---

## Skipped (not worth touching)

- `dashboard/pages/*.py` formatting lambdas — pure display code, runs once on render
- `utils/utils.py:175,183` — `groupby("dia").apply(...)` — read context; likely a custom aggregation that can't be simplified trivially
- `carto/routes.py`, `carto/stops.py` — geometric smoothing (`lowess_linea`) — algo-level apply, not a simple pattern fix
- `datamodel/services.py` — service detection logic with stateful `.apply()` — complex domain logic
- `kpi/supply_kpi.py`, `kpi/line_od_matrix.py` — supply KPI computation — separate concern

---

## After each step

```
uv run python -m pytest urbantrips/tests/ -x -q
```
