# Base-dir isolation

**Date:** 2026-06-09
**Status:** approved

## Goal

Allow fully isolated runs — each with its own config, inputs, databases, and outputs — under a single directory, without changing existing CWD-based workflows.

## Approach

`--base-dir` CLI flag (Option C): sets the root for all path resolution. The config file is found automatically inside it. The config can optionally override specific subdirectories (relative to config or absolute). No `--base-dir` → identical to today's behavior.

## Architecture

### `Paths` dataclass (`urbantrips/utils/paths.py`)

```python
@dataclass
class Paths:
    base: Path
    config_file: Path
    input_dir: Path   # input CSVs         default: {base}/data/data_ciudad
    db_dir: Path      # database files      default: {base}/data/db
    output_dir: Path  # resultados          default: {base}/resultados
```

A module-level singleton is initialized once at startup via `init_paths()` and read everywhere via `get_paths()`. If `init_paths()` was never called, `get_paths()` returns defaults relative to `Path(".")` — full backward compatibility.

### Path resolution order

1. `--base-dir /runs/city_A` (or `URBANTRIPS_BASE` env var) sets `base`
2. Config file searched at:
   - `{base}/configuraciones_generales.yaml`
   - `{base}/configs/configuraciones_generales.yaml`
   - Clear error naming both paths if neither exists
3. Config loaded → `input_dir`, `db_dir`, `output_dir` keys (if present) override defaults
   - Relative values resolved relative to the config file's directory
   - Absolute values used as-is
4. Fully-resolved `Paths` stored as singleton

### New config keys (all optional)

```yaml
input_dir: data/data_ciudad    # relative to config file, or absolute
db_dir: data/db
output_dir: resultados
```

## Components

### New

**`urbantrips/utils/paths.py`**
- `init_paths(base_dir: Path | None, config_overrides: dict) -> Paths`
- `get_paths() -> Paths`
- `_find_config(base: Path) -> Path`

### Modified

| File | Change |
|---|---|
| `run_all_urbantrips.py` | Add `--base-dir` / `-d` flag; set `URBANTRIPS_BASE`; call `init_paths()` before anything else |
| `config/config.py` | Add `input_dir`, `db_dir`, `output_dir` to `Config`, `_KNOWN_FIELDS` (all `str \| None`, default `None`) |
| `utils/utils.py` | `leer_configs_generales()` and `leer_configs_tuning()` use `get_paths().config_file.parent` instead of hardcoded `"configs/"` |
| `utils/fs.py` | `create_directories()` uses `get_paths()` for all directory roots |
| `utils/run_process.py` | `_build_ctx()` and `_ingest_all_days()` use `get_paths().db_dir` and `get_paths().input_dir` |
| `viz/viz.py` | All `os.path.join("resultados", ...)` → `get_paths().output_dir / ...` |
| `kpi/kpi.py` | All `os.path.join("resultados", ...)` → `get_paths().output_dir / ...` |

## Data flow

```
run_all_urbantrips.py
  → parse --base-dir
  → set URBANTRIPS_BASE env var
  → init_paths()          # resolves config, loads overrides, stores singleton
  → _build_ctx()          # calls get_paths().db_dir
  → run_ingest()          # calls get_paths().input_dir
  → run_outputs()         # calls get_paths().output_dir
```

No `Paths` object is threaded through function signatures — all consumers call `get_paths()` directly.

## Error handling

| Situation | Behavior |
|---|---|
| `--base-dir` given but does not exist | `FileNotFoundError` from `init_paths()` with clear message |
| Config not found at either expected location | `FileNotFoundError` naming both paths tried |
| Relative override in config | Resolved relative to config file's directory |
| `get_paths()` called before `init_paths()` | Silent fallback to CWD-relative defaults |

## Testing

- Existing unit tests unaffected: they never call `init_paths()`, so `get_paths()` falls back to CWD defaults
- New unit tests in `tests/unit/test_paths.py`:
  - Base-dir resolution (with and without `--base-dir`)
  - Config search logic (root vs `configs/` subfolder)
  - Override resolution (relative and absolute paths)
  - Fallback default when `init_paths()` not called

## Usage examples

```bash
# Fully isolated run
python run_all_urbantrips.py --base-dir /runs/city_A

# Custom output dir (in config)
# /runs/city_A/configuraciones_generales.yaml:
#   output_dir: /shared/results/city_A

# Backward-compatible (no change needed for existing setups)
python run_all_urbantrips.py
```
