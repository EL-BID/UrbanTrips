# Step CLI Design

**Date:** 2026-05-29  
**Status:** Approved

## Problem

The full pipeline (`ingest → legs → outputs → dashboard`) can take hours. When a late step fails, users must re-run everything. There is no way to restart from a specific step or stop at a defined checkpoint.

## Steps

Four named steps, in execution order:

| Name | What it runs |
|---|---|
| `ingest` | Phase 1: ingest all days (`_ingest_all_days`) |
| `legs` | Phase 2+3: create legs + enrich (`_create_legs_for_batches` + `_enrich_all_legs`) |
| `outputs` | Phase 4 + routes + KPIs + indicators (`_build_final_outputs`, `infer_routes_geoms`, `build_routes_from_official_inferred`, `compute_kpi`, `persist_indicators`) |
| `dashboard` | Dashboard preparation (`preparo_indicadores_dash`) |

## CLI Interface

Two new mutually exclusive flags added to `run_all_urbantrips.py`:

```
--step <name>      Run exactly one step (validates prerequisites first)
--through <name>   Run from ingest through the named step (inclusive)
```

### Usage examples

```bash
# Existing behaviour (unchanged)
python run_all_urbantrips.py
python run_all_urbantrips.py --no_dashboard
python run_all_urbantrips.py -b all

# New: run from start through outputs (no dashboard)
python run_all_urbantrips.py --through outputs

# New: run from start through legs only
python run_all_urbantrips.py --through legs

# New: run a single step
python run_all_urbantrips.py --step dashboard
python run_all_urbantrips.py --step outputs
```

`--no_dashboard` is kept for backward compatibility. It is equivalent to `--through outputs`.

`--borrar_corrida` is incompatible with `--step` (you cannot delete-and-re-run a single step in isolation). Combining them raises an argument error.

## Prerequisite Validation

Before executing a step, `check_prerequisites(step, ctx)` queries the DB and raises `RuntimeError` with an actionable message if the required data is absent.

| Step | Check | Error message |
|---|---|---|
| `legs` | `etapas` table has rows | `"Step 'legs' requires ingest to have been run first (etapas table is empty). Run with --through legs first."` |
| `outputs` | `etapas` table has rows with `h3` populated | `"Step 'outputs' requires legs to have been run first (etapas.h3 is empty). Run with --through outputs first."` |
| `dashboard` | `viajes` table has rows | `"Step 'dashboard' requires outputs to have been run first (viajes table is empty). Run with --through outputs first."` |

`ingest` has no prerequisites.

`--through` validates only the first step in the sequence that has a prerequisite (e.g., `--through outputs` validates `legs` prerequisites before running `outputs`, but `ingest` and `legs` run unconditionally first).

## Implementation

### `run_process.py`

Extract four public functions from `run_all`:

```python
def run_ingest(ctx, corridas, configs, trx_order_params, n_batches): ...
def run_legs(ctx, corridas, configs, trx_order_params, n_batches): ...
def run_outputs(ctx): ...
def run_dashboard(ctx): ...
def check_prerequisites(step: str, ctx: StorageContext) -> None: ...
```

`run_all` becomes a thin orchestrator calling these in sequence.

### `run_all_urbantrips.py`

- Add `--step` and `--through` as a mutually exclusive group to the argparse parser.
- `main()` accepts `step` and `through` parameters.
- Dispatch logic: resolve which steps to run, call `check_prerequisites` for the first step that requires it, then execute in order.

## Out of Scope

- Resuming a failed step mid-execution (e.g., resuming leg creation partway through batches) — existing idempotency handles re-runs.
- Skipping individual sub-steps within a phase.
- A `--from` flag (run from step X rather than always from `ingest`) — not needed given current use cases.
