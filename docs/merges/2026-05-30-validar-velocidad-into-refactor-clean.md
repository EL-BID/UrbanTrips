# Merge Report: validar_velocidad → feat/refactor-clean

**Date:** 2026-05-30  
**Branch:** `feat/refactor-clean-merge-vv`  
**Commit:** `bb9ae93`  
**Author:** Martín Anzorena + Claude

---

## Context

`feat/refactor-clean` contained a major architectural refactor (StorageContext, DuckDB, step CLI).  
`validar_velocidad` was branched from `dev` (common ancestor `61f9c5b`) and added GPS-based distance/speed calculations and a comprehensive column rename.

Both branches had 12–13 commits since the ancestor, with heavy overlap in ~20 files.

---

## Merge Strategy

**Rule:** HEAD (`feat/refactor-clean`) wins on architecture. `validar_velocidad` wins on logic/algorithms.

- HEAD introduced: `StorageContext`, DuckDB adapters, `logger` (instead of `print`), `ctx`-based storage calls, step CLI.
- VV introduced: `travel_times_legs/trips` tables, column renames, GPS odometer distance computation, vectorized KPI helpers.

For every conflict, the resolution was:
1. Keep HEAD's function signatures (`ctx: StorageContext`, `logger.*`, `ctx.data.*`)
2. Integrate VV's algorithmic changes (new distance columns, new computations)
3. Discard VV's old SQLite API calls (`iniciar_conexion_db`, `levanto_tabla_sql`, `guardar_tabla_sql`)

---

## Files Resolved (18 total)

| File | Conflicts | Resolution |
|---|---|---|
| `.gitignore` | 1 | HEAD (added `.worktrees/`, benchmark notes) |
| `carto/carto.py` | 2 | HEAD (ctx saves, helper function) |
| `carto/routes.py` | 2 | HEAD + VV better assert message |
| `carto/stops.py` | 2 | HEAD |
| `dashboard/pages/5_Indicadores…py` | 1 (whole file) | HEAD |
| `datamodel/legs.py` | 15 | Blend — see key changes below |
| `datamodel/misc.py` | 3 | HEAD + update queries to use `distance_od` |
| `datamodel/services.py` | 6 | HEAD |
| `datamodel/transactions.py` | 5 | HEAD |
| `datamodel/trips.py` | 5 | HEAD |
| `destinations/destinations.py` | 1 | HEAD |
| `kpi/kpi.py` | 44 | HEAD + VV improvements (see below) |
| `preparo_dashboard/preparo_dashboard.py` | 7 | HEAD + VV travel_times JOIN |
| `run_all_urbantrips.py` | 1 | HEAD |
| `utils/check_configs.py` | 4 | HEAD |
| `utils/run_process.py` | 7 | HEAD + VV step ordering |
| `utils/utils.py` | 4 | HEAD (DuckDB support kept) |
| `viz/viz.py` | 1 (whole file) | HEAD |

---

## Key Functional Changes Integrated from validar_velocidad

### 1. `assign_gps_destination` → `assign_time_distances`

`datamodel/legs.py` — the function was renamed and extended:

- Reads all validated legs and computes `distance_od` via `compute_od_distances`
- If GPS exists: matches legs to GPS pings, then computes:
  - `distance_route`: from cumulative GPS odometer (`distance_km`)
  - `distance_route_gps`: from operator-reported odometer (`distance_servicio_mts`)
  - `kmh_od`, `kmh_route`, `kmh_route_gps`
- Saves per-leg results to `travel_times_legs` table
- Aggregates per-trip results into `travel_times_trips` table
- Saves both using `ctx.data.execute(DELETE)` + `ctx.data.append_raw`

### 2. Processing Order in `procesar_transacciones`

`utils/run_process.py` — order changed per VV's design:

```
OLD (HEAD):
  assign_gps_origin → assign_gps_destination
  compute_trips_travel_time → legs.add_distance_and_travel_time
  rearrange_trip_id_same_od → create_trips_from_legs_and_fex
  trips.add_distance_and_travel_time

NEW (merged):
  assign_gps_origin
  rearrange_trip_id_same_od     ← moved earlier
  assign_time_distances          ← replaces assign_gps_destination
  create_trips_from_legs_and_fex ← redundant travel-time calls removed
```

`compute_trips_travel_time`, `legs.add_distance_and_travel_time`, and `trips.add_distance_and_travel_time` are no longer called — travel data lives in `travel_times_*` tables.

### 3. Dashboard Data Loading

`preparo_dashboard/preparo_dashboard.py` — `load_and_process_data` now reads:

```sql
SELECT e.*, tt.travel_time_min, tt.distance_od, tt.distance_route,
       tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
FROM etapas e
LEFT JOIN travel_times_legs tt ON e.id = tt.id
WHERE e.od_validado = 1
```

Same pattern for `viajes` JOIN `travel_times_trips`.

### 4. KPI Improvements

`kpi/kpi.py`:
- `pd.set_option('future.no_silent_downcasting', True)` added
- `read_data_for_daily_kpi` now JOINs `travel_times_legs` to get pre-computed distances (no more `compute_od_distances` call in the KPI path)
- `compute_kpi_by_line_day` now cleans `inf` values from ratio columns and rounds `tot_pax`

### 5. Column Renames (from validar_velocidad)

Applied across `kpi_lineas.py`, `preparo_dashboard`, `misc.py`:

| Old name | New name |
|---|---|
| `distancia` | `distance_od` |
| `distance_km` | `distance_route` |
| `travel_speed` | `kmh_od` |
| `dmt_mean` | `dmt_mean_od` / `dmt_mean_route` / `dmt_mean_route_gps` |
| `dmt_median` | `dmt_median_od` / `dmt_median_route` / `dmt_median_route_gps` |
| `ipk` | `ipk_route` |
| `fo_mean` / `fo_median` | `fo_mean_od/route/route_gps` / `fo_median_od/route/route_gps` |
| `serv_distance_km` | `serv_distance_route` |

### 6. Config Key Rename

`configs/configuraciones_generales.yaml`:  
`alias_db` → `alias_db_insumos`

New GPS column names added: `distance_servicio_mts_gps`, `distance_servicio_mts_agg_gps`, `id_servicio_gps`, `dominio_gps`.

---

## Notable Decision: run_basic_kpi

VV's version of `run_basic_kpi` assumed `distance_route` and `kmh_route_leg` columns directly on `etapas`, populated by a different upstream step. This is incompatible with the refactored architecture where those columns live in `travel_times_legs`.

**Decision:** `run_basic_kpi` was fully reverted to HEAD's original version, which computes its own `distance` via `compute_od_distances` and uses the legacy speed pipeline. This function may need a separate update in a future pass to use `travel_times_legs` data.

---

## Post-Merge Validation

- All 18 conflict files resolved to zero conflict markers
- Full Python syntax check passed (`ast.parse`) for all `.py` files in `urbantrips/`
- No old API calls (`iniciar_conexion_db`, `levanto_tabla_sql`, `guardar_tabla_sql`) left in the resolved conflict sections

---

## Known Follow-up Items

1. **`run_basic_kpi`** should eventually be updated to read from `travel_times_legs` (has `distance_route`, `distance_route_gps`) instead of re-computing via `compute_od_distances`.
2. **`legs.add_distance_and_travel_time`** and **`trips.add_distance_and_travel_time`** are still in the codebase but no longer called — they can be removed in a cleanup pass.
3. **`trips.compute_trips_travel_time`** same as above.
4. **`assign_gps_destination`** still exists as a stub in `legs.py` alongside `assign_time_distances` — the old function should be removed.
