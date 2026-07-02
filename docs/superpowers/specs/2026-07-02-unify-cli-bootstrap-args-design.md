# Unify CLI bootstrap arguments

**Date:** 2026-07-02
**Status:** approved

## Goal

`run_all_urbantrips.py` and `dashboard.py` are the only two script entry points in the repo. Both need `--config` and `--base-dir` to resolve paths, but today they implement this independently: `run_all_urbantrips.py` uses `argparse` with `-c/--config` and `-d/--base-dir`; `dashboard.py` hand-scans `sys.argv` for `--config` only, with no `--base-dir` support and no short flag. Unify them so both scripts accept the same bootstrap flags, defined once.

## Approach

Extract the two bootstrap flags (`-c/--config`, `-d/--base-dir`) into a small shared module. Each script keeps its own `ArgumentParser` (they have different additional flags — `-b`, `-n`, `--step`, `--through` are run_all-only and don't apply to viewing a dashboard), but both call into the shared module to define and apply the common ones. This avoids a second, divergent copy of the flag definitions and env var wiring.

## Architecture

### New module: `urbantrips/utils/cli.py`

```python
def add_bootstrap_args(parser: argparse.ArgumentParser) -> None:
    """Add -c/--config and -d/--base-dir to an existing parser."""

def apply_bootstrap_env(args: argparse.Namespace) -> None:
    """Set URBANTRIPS_CONFIG / URBANTRIPS_BASE env vars from parsed args, if present."""
```

- `add_bootstrap_args` adds exactly the two `add_argument` calls currently inlined in `run_all_urbantrips.build_parser()` (same flags, `dest`, `help` text).
- `apply_bootstrap_env` replaces the two `if args.config: os.environ[...]` / `if args.base_dir: os.environ[...]` lines currently duplicated inline.

## Components

### Modified

| File | Change |
|---|---|
| `urbantrips/run_all_urbantrips.py` | `build_parser()` calls `add_bootstrap_args(parser)` instead of inlining `-c`/`-d`; `__main__` calls `apply_bootstrap_env(args)` instead of the manual env var sets |
| `urbantrips/dashboard/dashboard.py` | Replace the manual `sys.argv` scan (module top, before `import streamlit`) with: build an `argparse.ArgumentParser`, `add_bootstrap_args(parser)`, `args, _ = parser.parse_known_args()`, `apply_bootstrap_env(args)`. Update the module comment to document `--base-dir` alongside `--config`. |

### New

**`urbantrips/utils/cli.py`** — `add_bootstrap_args()`, `apply_bootstrap_env()`.

## Data flow

```
run_all_urbantrips.py __main__:
  build_parser() → add_bootstrap_args(parser)   # adds -c/-d
  parser.parse_args()
  apply_bootstrap_env(args)                      # sets URBANTRIPS_CONFIG / URBANTRIPS_BASE
  init_paths(...)                                # unchanged, explicit

dashboard.py (module top-level, before importing streamlit):
  argparse.ArgumentParser() → add_bootstrap_args(parser)
  parser.parse_known_args()                      # tolerant of unexpected args
  apply_bootstrap_env(args)                      # sets same env vars
  # no explicit init_paths() call — get_paths() lazily reads URBANTRIPS_BASE/URBANTRIPS_CONFIG
```

`parse_known_args` is used in `dashboard.py` (not `parse_args`) defensively, in case Streamlit's invocation ever forwards something unexpected after `--`; `run_all_urbantrips.py` keeps `parse_args` since it's the sole consumer of its own argv.

## Error handling

No behavior change from today: invalid `--base-dir` still surfaces as `FileNotFoundError` from `init_paths()`/`get_paths()`, unrelated to this refactor. Unknown flags to `dashboard.py` are silently ignored via `parse_known_args`, matching today's silent-ignore behavior of the manual `sys.argv` scan.

## Testing

- New `tests/unit/test_cli.py`: `add_bootstrap_args` produces a parser accepting `-c/--config` and `-d/--base-dir`; `apply_bootstrap_env` sets the right env vars (and doesn't set them when args are `None`).
- `tests/unit/test_cli_entrypoint.py`: existing tests continue to pass unchanged (parser behavior for `-c`/`-d` is identical, just sourced from the shared helper).
- `dashboard.py`'s argv-parsing block stays a few lines at the top of the file (before `import streamlit`), calling straight into `add_bootstrap_args`/`apply_bootstrap_env` — no new function is introduced there. Since importing the rest of the file has Streamlit side effects, this block itself is covered indirectly by `test_cli.py`'s coverage of `add_bootstrap_args`/`apply_bootstrap_env`, and is otherwise validated manually per the design's usage examples (Streamlit isn't in the automated unit test surface today either).

## Usage examples

```bash
# Before: only --config worked
streamlit run urbantrips/dashboard/dashboard.py -- --config configs/otra_ciudad.yaml

# After: --base-dir also works, plus the -c/-d short flags
streamlit run urbantrips/dashboard/dashboard.py -- --base-dir /runs/city_A
streamlit run urbantrips/dashboard/dashboard.py -- -c configs/otra_ciudad.yaml
```
