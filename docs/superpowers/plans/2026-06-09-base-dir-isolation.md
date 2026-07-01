# Base-dir Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--base-dir` CLI flag so all UrbanTrips paths (config, inputs, databases, outputs) resolve under a single user-specified directory, enabling fully isolated runs.

**Architecture:** A new `urbantrips/utils/paths.py` module holds a `Paths` dataclass and a module-level singleton initialized by `init_paths()` and read everywhere via `get_paths()`. The singleton falls back to CWD-relative defaults when `init_paths()` was never called, preserving full backward compatibility.

**Tech Stack:** Python stdlib (`pathlib`, `os`, `dataclasses`), PyYAML (already a dependency), pytest.

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Create | `urbantrips/utils/paths.py` | New module: `Paths`, `init_paths`, `get_paths`, `reset_paths` |
| Create | `urbantrips/tests/unit/test_paths.py` | Unit tests for paths module |
| Modify | `urbantrips/config/config.py` | Add `input_dir`, `db_dir`, `output_dir` optional fields |
| Modify | `urbantrips/run_all_urbantrips.py` | Add `--base-dir` flag, call `init_paths()` at startup |
| Modify | `urbantrips/utils/utils.py` | `leer_configs_generales`, `leer_configs_tuning` use `get_paths()` |
| Modify | `urbantrips/utils/fs.py` | `create_directories()` uses `get_paths()` |
| Modify | `urbantrips/utils/check_configs.py` | All `"configs/"` and `"data/data_ciudad"` hardcoded paths |
| Modify | `urbantrips/utils/run_process.py` | `_build_ctx`, `borrar_corridas`, `_ingest_all_days` |
| Modify | `urbantrips/carto/carto.py` | `data/data_ciudad` paths |
| Modify | `urbantrips/carto/routes.py` | `data/data_ciudad` paths |
| Modify | `urbantrips/carto/stops.py` | `data/data_ciudad` paths |
| Modify | `urbantrips/datamodel/transactions.py` | `data/data_ciudad` paths |
| Modify | `urbantrips/viz/viz.py` | All `resultados/` paths |
| Modify | `urbantrips/viz/section_supply.py` | All `resultados/` paths |
| Modify | `urbantrips/viz/line_od_matrix.py` | All `resultados/` paths |
| Modify | `urbantrips/viz/helpers.py` | `resultados/` path |
| Modify | `urbantrips/kpi/line_od_matrix.py` | `resultados/` path |
| Modify | `urbantrips/cluster/dbscan.py` | `resultados/` paths |
| Modify | `urbantrips/preparo_dashboard/preparo_dashboard.py` | `resultados/` paths |
| Modify | `urbantrips/dashboard/dash_storage.py` | Config path, `get_project_root` |
| Modify | `urbantrips/dashboard/dash_utils.py` | DB path resolution |

---

## Task 1: Create `urbantrips/utils/paths.py`

**Files:**
- Create: `urbantrips/utils/paths.py`
- Create: `urbantrips/tests/unit/test_paths.py`

- [ ] **Step 1: Write the failing tests**

```python
# urbantrips/tests/unit/test_paths.py
from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import patch
from urbantrips.utils.paths import init_paths, get_paths, reset_paths, Paths


@pytest.fixture(autouse=True)
def clean_singleton():
    reset_paths()
    yield
    reset_paths()


def test_get_paths_fallback_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("URBANTRIPS_BASE", raising=False)
    monkeypatch.delenv("URBANTRIPS_CONFIG", raising=False)
    p = get_paths()
    assert p.base == tmp_path.resolve()
    assert p.input_dir == tmp_path.resolve() / "data" / "data_ciudad"
    assert p.db_dir == tmp_path.resolve() / "data" / "db"
    assert p.output_dir == tmp_path.resolve() / "resultados"


def test_init_paths_finds_config_at_root(tmp_path):
    config = tmp_path / "configuraciones_generales.yaml"
    config.write_text("alias_db: test\n")
    p = init_paths(tmp_path)
    assert p.config_file == config
    assert p.base == tmp_path


def test_init_paths_finds_config_in_configs_subdir(tmp_path):
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "configuraciones_generales.yaml"
    config.write_text("alias_db: test\n")
    p = init_paths(tmp_path)
    assert p.config_file == config


def test_init_paths_raises_if_base_not_exist():
    with pytest.raises(FileNotFoundError, match="base_dir does not exist"):
        init_paths(Path("/nonexistent/path/xyz"))


def test_init_paths_raises_if_config_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        init_paths(tmp_path)


def test_init_paths_uses_default_subdirs(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text("")
    p = init_paths(tmp_path)
    assert p.input_dir == tmp_path / "data" / "data_ciudad"
    assert p.db_dir == tmp_path / "data" / "db"
    assert p.output_dir == tmp_path / "resultados"


def test_init_paths_relative_override(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text(
        "output_dir: ../outputs\n"
    )
    p = init_paths(tmp_path)
    assert p.output_dir == (tmp_path / "configs" / ".." / "outputs").resolve()


def test_init_paths_absolute_override(tmp_path):
    abs_out = tmp_path / "shared" / "results"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text(
        f"output_dir: {abs_out}\n"
    )
    p = init_paths(tmp_path)
    assert p.output_dir == abs_out


def test_get_paths_returns_singleton(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text("")
    p1 = init_paths(tmp_path)
    p2 = get_paths()
    assert p1 is p2


def test_get_paths_lazy_init_from_env(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text("")
    monkeypatch.setenv("URBANTRIPS_BASE", str(tmp_path))
    p = get_paths()
    assert p.base == tmp_path
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest urbantrips/tests/unit/test_paths.py -v 2>&1 | head -30
```
Expected: ImportError or collection errors — `paths.py` doesn't exist yet.

- [ ] **Step 3: Create `urbantrips/utils/paths.py`**

```python
# urbantrips/utils/paths.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_paths: "Paths | None" = None


@dataclass
class Paths:
    base: Path
    config_file: Path
    input_dir: Path
    db_dir: Path
    output_dir: Path

    @property
    def configs_dir(self) -> Path:
        return self.config_file.parent


def _find_config(base: Path) -> Path:
    candidates = [
        base / "configuraciones_generales.yaml",
        base / "configs" / "configuraciones_generales.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Config file not found in {base}. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def _resolve_dir(value: str | None, config_file: Path, default: Path) -> Path:
    if value is None:
        return default
    p = Path(value)
    return p if p.is_absolute() else (config_file.parent / p).resolve()


def init_paths(base_dir: Path | None = None) -> Paths:
    """Initialize the path singleton from base_dir.

    Reads input_dir / db_dir / output_dir overrides from the config YAML if present.
    Raises FileNotFoundError if base_dir doesn't exist or no config is found.
    """
    global _paths
    base = Path(base_dir).resolve() if base_dir else Path(".").resolve()
    if not base.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base}")

    config_file = _find_config(base)

    overrides: dict = {}
    try:
        with open(config_file, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for key in ("input_dir", "db_dir", "output_dir"):
            if key in raw and raw[key]:
                overrides[key] = raw[key]
    except Exception:
        pass

    _paths = Paths(
        base=base,
        config_file=config_file,
        input_dir=_resolve_dir(overrides.get("input_dir"), config_file, base / "data" / "data_ciudad"),
        db_dir=_resolve_dir(overrides.get("db_dir"), config_file, base / "data" / "db"),
        output_dir=_resolve_dir(overrides.get("output_dir"), config_file, base / "resultados"),
    )
    return _paths


def get_paths() -> Paths:
    """Return the path singleton, lazily initializing from URBANTRIPS_BASE or CWD."""
    global _paths
    if _paths is None:
        env_base = os.environ.get("URBANTRIPS_BASE")
        if env_base:
            init_paths(Path(env_base))
        else:
            _paths = _default_paths()
    return _paths


def reset_paths() -> None:
    """Reset the singleton. For use in tests only."""
    global _paths
    _paths = None


def _default_paths() -> Paths:
    """CWD-relative defaults, respecting URBANTRIPS_CONFIG env var."""
    env_config = os.environ.get("URBANTRIPS_CONFIG")
    if env_config:
        config_file = Path(env_config).resolve()
        base = (
            config_file.parent.parent
            if config_file.parent.name == "configs"
            else config_file.parent
        )
    else:
        base = Path(".").resolve()
        config_file = base / "configs" / "configuraciones_generales.yaml"
    return Paths(
        base=base,
        config_file=config_file,
        input_dir=base / "data" / "data_ciudad",
        db_dir=base / "data" / "db",
        output_dir=base / "resultados",
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest urbantrips/tests/unit/test_paths.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add urbantrips/utils/paths.py urbantrips/tests/unit/test_paths.py
git commit -m "feat: add Paths singleton module for base-dir resolution"
```

---

## Task 2: Add optional path fields to `Config`

**Files:**
- Modify: `urbantrips/config/config.py`
- Test: `urbantrips/tests/unit/test_config.py`

- [ ] **Step 1: Read the test file to understand existing patterns**

```bash
grep -n "input_dir\|db_dir\|output_dir\|_KNOWN_FIELDS" urbantrips/tests/unit/test_config.py | head -20
```

- [ ] **Step 2: Add a test for the new fields**

In `urbantrips/tests/unit/test_config.py`, add to the existing test suite:

```python
def test_path_overrides_are_optional(minimal_config_yaml, tmp_path):
    """input_dir / db_dir / output_dir default to None when absent."""
    cfg = load_config(tmp_path / "config.yaml")  # uses existing minimal fixture
    assert cfg.input_dir is None
    assert cfg.db_dir is None
    assert cfg.output_dir is None


def test_path_overrides_are_loaded(tmp_path):
    """Explicit path override keys are parsed and stored."""
    content = _minimal_yaml() + "\ninput_dir: /my/input\ndb_dir: custom/db\noutput_dir: /out\n"
    p = tmp_path / "config.yaml"
    p.write_text(content)
    cfg = load_config(p)
    assert cfg.input_dir == "/my/input"
    assert cfg.db_dir == "custom/db"
    assert cfg.output_dir == "/out"
```

(Use `_minimal_yaml()` or whatever helper already exists in that test file to build the required fields.)

- [ ] **Step 3: Run new tests to confirm they fail**

```bash
python -m pytest urbantrips/tests/unit/test_config.py -v -k "path_override" 2>&1 | tail -10
```

- [ ] **Step 4: Update `config/config.py`**

Add to `_KNOWN_FIELDS`:
```python
_KNOWN_FIELDS = {
    "alias_db", "alias_db_insumos", "alias_db_dashboard", "corridas",
    "geolocalizar_trx", "nombre_archivo_trx", "nombre_archivo_gps",
    "nombres_variables_trx", "formato_fecha", "columna_hora",
    "tipo_trx_invalidas", "tolerancia_parada_destino", "resolucion_h3",
    "ordenamiento_transacciones", "ventana_viajes", "ventana_duplicado",
    "tiempos_viaje_estaciones", "storage_backend", "n_batches", "parallel_workers",
    "input_dir", "db_dir", "output_dir",
}
```

Add to the `Config` dataclass (after `parallel_workers`):
```python
input_dir: str | None = None
db_dir: str | None = None
output_dir: str | None = None
```

In `_apply_legacy_defaults`, add the three new keys to the `None`-stripping loop:
```python
for opt in ("storage_backend", "n_batches", "parallel_workers", "input_dir", "db_dir", "output_dir"):
    if data.get(opt) is None:
        data.pop(opt, None)
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest urbantrips/tests/unit/test_config.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add urbantrips/config/config.py urbantrips/tests/unit/test_config.py
git commit -m "feat: add optional input_dir/db_dir/output_dir to Config"
```

---

## Task 3: Add `--base-dir` CLI flag

**Files:**
- Modify: `urbantrips/run_all_urbantrips.py`
- Test: `urbantrips/tests/unit/test_cli_entrypoint.py`

- [ ] **Step 1: Check existing CLI tests**

```bash
grep -n "base_dir\|base-dir\|init_paths" urbantrips/tests/unit/test_cli_entrypoint.py
```

- [ ] **Step 2: Write a failing test**

In `urbantrips/tests/unit/test_cli_entrypoint.py`, add:

```python
def test_base_dir_flag_is_accepted():
    from urbantrips.run_all_urbantrips import build_parser
    parser = build_parser()
    args = parser.parse_args(["--base-dir", "/tmp/run_a"])
    assert args.base_dir == "/tmp/run_a"


def test_base_dir_short_flag():
    from urbantrips.run_all_urbantrips import build_parser
    parser = build_parser()
    args = parser.parse_args(["-d", "/tmp/run_b"])
    assert args.base_dir == "/tmp/run_b"


def test_base_dir_defaults_to_none():
    from urbantrips.run_all_urbantrips import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert args.base_dir is None
```

- [ ] **Step 3: Run to confirm failure**

```bash
python -m pytest urbantrips/tests/unit/test_cli_entrypoint.py -v -k "base_dir" 2>&1 | tail -10
```

- [ ] **Step 4: Update `run_all_urbantrips.py`**

In `build_parser()`, add after the existing `--config` argument:

```python
parser.add_argument(
    "-d",
    "--base-dir",
    type=str,
    default=None,
    dest="base_dir",
    help="Project root directory. Config, inputs, databases, and outputs are resolved relative to this path.",
)
```

In `__main__`, after `if args.config: os.environ["URBANTRIPS_CONFIG"] = args.config`, add:

```python
if args.base_dir:
    os.environ["URBANTRIPS_BASE"] = args.base_dir

from pathlib import Path
from urbantrips.utils.paths import init_paths
init_paths(Path(args.base_dir) if args.base_dir else None)
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest urbantrips/tests/unit/test_cli_entrypoint.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add urbantrips/run_all_urbantrips.py urbantrips/tests/unit/test_cli_entrypoint.py
git commit -m "feat: add --base-dir CLI flag, call init_paths at startup"
```

---

## Task 4: Update config loading in `utils/utils.py`

**Files:**
- Modify: `urbantrips/utils/utils.py`

- [ ] **Step 1: Read the current `leer_configs_generales` and `leer_configs_tuning` implementations**

Lines 85–166 of `urbantrips/utils/utils.py`.

- [ ] **Step 2: Replace path resolution in `leer_configs_generales`**

Find and replace the env-var block at the top of `leer_configs_generales`:

```python
# Before (lines 93–102):
env_path = os.environ.get("URBANTRIPS_CONFIG")
if env_path:
    path = env_path
else:
    archivo = (
        "configuraciones_generales_autogenerado.yaml"
        if autogenerado
        else "configuraciones_generales.yaml"
    )
    path = os.path.join("configs", archivo)
```

```python
# After:
from urbantrips.utils.paths import get_paths
_p = get_paths()
if autogenerado:
    path = str(_p.configs_dir / "configuraciones_generales_autogenerado.yaml")
else:
    path = str(_p.config_file)
```

- [ ] **Step 3: Replace path in `leer_configs_tuning`**

Find:
```python
path = os.path.join("configs", "tuning.yaml")
```

Replace with:
```python
from urbantrips.utils.paths import get_paths
path = str(get_paths().configs_dir / "tuning.yaml")
```

- [ ] **Step 4: Verify with existing config tests**

```bash
python -m pytest urbantrips/tests/unit/test_config.py urbantrips/tests/unit/test_misc.py -v 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add urbantrips/utils/utils.py
git commit -m "refactor: use get_paths() in leer_configs_generales and leer_configs_tuning"
```

---

## Task 5: Update `fs.py` and `check_configs.py`

**Files:**
- Modify: `urbantrips/utils/fs.py`
- Modify: `urbantrips/utils/check_configs.py`

- [ ] **Step 1: Rewrite `create_directories()` in `fs.py`**

```python
import os
from pathlib import Path


def create_directories():
    """Creates the standard UrbanTrips directory structure under the active base dir."""
    from urbantrips.utils.paths import get_paths
    p = get_paths()
    dirs = [
        p.db_dir,
        p.input_dir,
        p.configs_dir,
        p.base / "docs",
        p.output_dir / "tablas",
        p.output_dir / "png",
        p.output_dir / "pdf",
        p.output_dir / "matrices",
        p.output_dir / "data",
        p.output_dir / "html",
        p.output_dir / "geojson",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Update `check_configs.py` — config file paths**

In `write_config()` (around line 282):
```python
# Before:
path = os.path.join("configs", filename)
# After:
from urbantrips.utils.paths import get_paths
path = str(get_paths().configs_dir / filename)
```

In `check_configs_file()` (around line 863–873):
```python
# Before:
directory = "configs"
file_name = "configuraciones_generales_autogenerado.yaml"
file_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
# After:
from urbantrips.utils.paths import get_paths
configs_dir = get_paths().configs_dir
file_path = str(configs_dir / "configuraciones_generales_autogenerado.yaml")
configs_dir.mkdir(parents=True, exist_ok=True)
```

In `check_config()` (around lines 994–998), replace hardcoded config paths:
```python
# Before:
corregir_codificacion_a_utf8_sin_modificar_texto(
    "configs/configuraciones_generales.yaml"
)
replace_tabs_with_spaces(os.path.join("configs", "configuraciones_generales.yaml"))
# After:
from urbantrips.utils.paths import get_paths
_config_path = str(get_paths().config_file)
corregir_codificacion_a_utf8_sin_modificar_texto(_config_path)
replace_tabs_with_spaces(_config_path)
```

In `check_config()` (around lines 1028–1040), replace autogenerado paths:
```python
# Before:
corregir_codificacion_a_utf8_sin_modificar_texto(
    "configs/configuraciones_generales_autogenerado.yaml"
)
# ...
base_path = Path() / 'configs'
# After:
from urbantrips.utils.paths import get_paths
_autogen = str(get_paths().configs_dir / "configuraciones_generales_autogenerado.yaml")
corregir_codificacion_a_utf8_sin_modificar_texto(_autogen)
# ...
base_path = get_paths().configs_dir
```

- [ ] **Step 3: Update `check_configs.py` — `data/data_ciudad` paths**

There are 5 occurrences. Replace each `os.path.join("data", "data_ciudad", ...)` with `str(get_paths().input_dir / ...)`:

Line 432: `os.path.join("data", "data_ciudad", nombre_archivo_informacion_lineas)` → `str(get_paths().input_dir / nombre_archivo_informacion_lineas)`

Line 442: `os.path.join("data", "data_ciudad", nombre_archivo_trx)` → `str(get_paths().input_dir / nombre_archivo_trx)`

Line 517: `os.path.join("data", "data_ciudad")` → `str(get_paths().input_dir)`

Line 520: `os.path.join("data", "data_ciudad", nombre_archivo_trx)` → `str(get_paths().input_dir / nombre_archivo_trx)`

Line 754: `os.path.join("data", "data_ciudad", nombre_archivo_informacion_lineas)` → `str(get_paths().input_dir / nombre_archivo_informacion_lineas)`

Line 805: `os.path.join("data", "data_ciudad", nombre_archivo_gps)` → `str(get_paths().input_dir / nombre_archivo_gps)`

Line 848: `os.path.join("data", "data_ciudad", recorridos_geojson)` → `str(get_paths().input_dir / recorridos_geojson)`

Add `from urbantrips.utils.paths import get_paths` at the top of each function that uses it, or once at module level.

- [ ] **Step 4: Run tests**

```bash
python -m pytest urbantrips/tests/unit/ -v 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add urbantrips/utils/fs.py urbantrips/utils/check_configs.py
git commit -m "refactor: use get_paths() in fs.create_directories and check_configs"
```

---

## Task 6: Update `run_process.py`

**Files:**
- Modify: `urbantrips/utils/run_process.py`

- [ ] **Step 1: Update `_build_ctx()`**

Find (around line 32):
```python
base = Path(configs.get("db_path", "data/db"))
base.mkdir(parents=True, exist_ok=True)
return StorageContext(
    data=DuckDBDataAdapter(base / f"{alias_data}_data.duckdb"),
    insumos=DuckDBInsumoAdapter(base / f"{alias_insumos}_insumos.duckdb"),
    dash=DuckDBDashAdapter(base / f"{alias_data}_dash.duckdb"),
    general=DuckDBGeneralAdapter(base / f"{alias_data}_general.duckdb"),
)
```

Replace with:
```python
from urbantrips.utils.paths import get_paths
db_dir = get_paths().db_dir
db_dir.mkdir(parents=True, exist_ok=True)
return StorageContext(
    data=DuckDBDataAdapter(db_dir / f"{alias_data}_data.duckdb"),
    insumos=DuckDBInsumoAdapter(db_dir / f"{alias_insumos}_insumos.duckdb"),
    dash=DuckDBDashAdapter(db_dir / f"{alias_data}_dash.duckdb"),
    general=DuckDBGeneralAdapter(db_dir / f"{alias_data}_general.duckdb"),
)
```

Remove the now-unused `configs.get("db_path", ...)` line (the `configs` dict is still used for aliases above it).

- [ ] **Step 2: Update `borrar_corridas()`**

Find (around line 154):
```python
base = Path(configs_usuario.get("db_path", "data/db"))
```

Replace with:
```python
from urbantrips.utils.paths import get_paths
base = get_paths().db_dir
```

- [ ] **Step 3: Update `_ingest_all_days()`**

Find (around line 297–303):
```python
nombre_archivo_trx = configs.get("nombre_archivo_trx") or f"{corrida}_trx.csv"
csv_path = os.path.join("data", "data_ciudad", nombre_archivo_trx)
if not os.path.exists(csv_path):
    zip_path = csv_path.replace(".csv", ".zip") if csv_path.endswith(".csv") else csv_path + ".zip"
    if os.path.exists(zip_path):
        csv_path = zip_path
```

Replace with:
```python
from urbantrips.utils.paths import get_paths
nombre_archivo_trx = configs.get("nombre_archivo_trx") or f"{corrida}_trx.csv"
csv_path = str(get_paths().input_dir / nombre_archivo_trx)
if not os.path.exists(csv_path):
    zip_path = csv_path.replace(".csv", ".zip") if csv_path.endswith(".csv") else csv_path + ".zip"
    if os.path.exists(zip_path):
        csv_path = zip_path
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest urbantrips/tests/unit/test_run_all_urbantrips.py -v 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add urbantrips/utils/run_process.py
git commit -m "refactor: use get_paths() for db and input paths in run_process"
```

---

## Task 7: Update `carto/` and `datamodel/` input paths

**Files:**
- Modify: `urbantrips/carto/carto.py`
- Modify: `urbantrips/carto/routes.py`
- Modify: `urbantrips/carto/stops.py`
- Modify: `urbantrips/datamodel/transactions.py`
- Modify: `urbantrips/viz/viz.py` (one input path)

The pattern for all replacements: `os.path.join("data", "data_ciudad", fname)` → `str(get_paths().input_dir / fname)`, and `os.path.join("data", "data_ciudad")` → `str(get_paths().input_dir)`.

Add `from urbantrips.utils.paths import get_paths` at the top of each affected function (or once at module level if the import isn't already there).

- [ ] **Step 1: Update `carto/carto.py`**

Line 238: `db_path = os.path.join("data", "data_ciudad", file_zona)` → `db_path = str(get_paths().input_dir / file_zona)`

Line 352: `db_path = os.path.join("data", "data_ciudad", poly_file)` → `db_path = str(get_paths().input_dir / poly_file)`

- [ ] **Step 2: Update `carto/routes.py`**

Line 53: `geojson_path = os.path.join("data", "data_ciudad", geojson_name)` → `geojson_path = str(get_paths().input_dir / geojson_name)`

Line 301: `ruta = os.path.join("data", "data_ciudad", tabla_lineas)` → `ruta = str(get_paths().input_dir / tabla_lineas)`

- [ ] **Step 3: Update `carto/stops.py`**

Line 24: `stops_path = os.path.join("data", "data_ciudad", nombre_archivo_paradas)` → `stops_path = str(get_paths().input_dir / nombre_archivo_paradas)`

Line 91: `data_path = os.path.join("data", "data_ciudad")` → `data_path = str(get_paths().input_dir)`

Line 300: `path = os.path.join("data", "data_ciudad", tts_file_name)` → `path = str(get_paths().input_dir / tts_file_name)`

- [ ] **Step 4: Update `datamodel/transactions.py`**

Line 74: `ruta = os.path.join("data", "data_ciudad", nombre_archivo_trx)` → `ruta = str(get_paths().input_dir / nombre_archivo_trx)`

Line 553: `ruta_trx_eco = os.path.join("data", "data_ciudad", nombre_archivo_trx_eco)` → `ruta_trx_eco = str(get_paths().input_dir / nombre_archivo_trx_eco)`

Line 735: `ruta_gps = os.path.join("data", "data_ciudad", nombre_archivo_gps)` → `ruta_gps = str(get_paths().input_dir / nombre_archivo_gps)`

- [ ] **Step 5: Update `viz/viz.py` (one input path)**

Line 2202: `file = os.path.join("data", "data_ciudad", f"{i[0]}")` → `file = str(get_paths().input_dir / f"{i[0]}")`

- [ ] **Step 6: Run tests**

```bash
python -m pytest urbantrips/tests/unit/ -v 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add urbantrips/carto/carto.py urbantrips/carto/routes.py urbantrips/carto/stops.py urbantrips/datamodel/transactions.py urbantrips/viz/viz.py
git commit -m "refactor: use get_paths().input_dir in carto, datamodel, viz"
```

---

## Task 8: Update output paths in `viz/`

The pattern: `os.path.join("resultados", sub, file)` → `str(get_paths().output_dir / sub / file)`.

Add `from urbantrips.utils.paths import get_paths` at module level (after existing imports) in each file.

**Files:**
- Modify: `urbantrips/viz/viz.py`
- Modify: `urbantrips/viz/section_supply.py`
- Modify: `urbantrips/viz/line_od_matrix.py`
- Modify: `urbantrips/viz/helpers.py`

- [ ] **Step 1: Update `viz/viz.py` (19 occurrences)**

Add to imports at top of file: `from urbantrips.utils.paths import get_paths`

Replace every `os.path.join("resultados", ...)`:

| Line | Before | After |
|------|--------|-------|
| 75 | `os.path.join("resultados", "png", f"{alias}linea_{id_linea}.png")` | `str(get_paths().output_dir / "png" / f"{alias}linea_{id_linea}.png")` |
| 634 | `os.path.join("resultados", frm, archivo)` | `str(get_paths().output_dir / frm / archivo)` |
| 686 | `os.path.join("resultados", "geojson", f_0)` | `str(get_paths().output_dir / "geojson" / f_0)` |
| 687 | `os.path.join("resultados", "geojson", f_1)` | `str(get_paths().output_dir / "geojson" / f_1)` |
| 1276 | `os.path.join("resultados", "png", f"{alias}{savefile_}.png")` | `str(get_paths().output_dir / "png" / f"{alias}{savefile_}.png")` |
| 1279 | `os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")` | `str(get_paths().output_dir / "pdf" / f"{alias}{savefile_}.pdf")` |
| 1364 | `os.path.join("resultados", "png", f"{alias}{savefile_}.png")` | `str(get_paths().output_dir / "png" / f"{alias}{savefile_}.png")` |
| 1367 | `os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")` | `str(get_paths().output_dir / "pdf" / f"{alias}{savefile_}.pdf")` |
| 1457 | `os.path.join("resultados", "png", f"{alias}{savefile_}.png")` | `str(get_paths().output_dir / "png" / f"{alias}{savefile_}.png")` |
| 1460 | `os.path.join("resultados", "pdf", f"{alias}{savefile_}.pdf")` | `str(get_paths().output_dir / "pdf" / f"{alias}{savefile_}.pdf")` |
| 1556 | `os.path.join("resultados", "png", f"{alias}{savefile}.png")` | `str(get_paths().output_dir / "png" / f"{alias}{savefile}.png")` |
| 1559 | `os.path.join("resultados", "pdf", f"{alias}{savefile}.pdf")` | `str(get_paths().output_dir / "pdf" / f"{alias}{savefile}.pdf")` |
| 1875 | `os.path.join("resultados", "png", f"{savefile}.png")` | `str(get_paths().output_dir / "png" / f"{savefile}.png")` |
| 1878 | `os.path.join("resultados", "pdf", f"{savefile}.pdf")` | `str(get_paths().output_dir / "pdf" / f"{savefile}.pdf")` |
| 1881 | `os.path.join("resultados", "matrices", f"{savefile}.xlsx")` | `str(get_paths().output_dir / "matrices" / f"{savefile}.xlsx")` |
| 2100 | `os.path.join("resultados", "png", f"{savefile}.png")` | `str(get_paths().output_dir / "png" / f"{savefile}.png")` |
| 2103 | `os.path.join("resultados", "pdf", f"{savefile}.pdf")` | `str(get_paths().output_dir / "pdf" / f"{savefile}.pdf")` |
| 2301 | `os.path.join("resultados", frm, archivo)` | `str(get_paths().output_dir / frm / archivo)` |
| 2455 | `os.path.join("resultados", frm, archivo)` | `str(get_paths().output_dir / frm / archivo)` |

- [ ] **Step 2: Update `viz/section_supply.py`**

Add `from urbantrips.utils.paths import get_paths` after existing imports.

| Line | Before | After |
|------|--------|-------|
| 338 | `os.path.join("resultados", frm, archivo)` | `str(get_paths().output_dir / frm / archivo)` |
| 349 | `os.path.join("resultados", "geojson", f_0)` | `str(get_paths().output_dir / "geojson" / f_0)` |
| 350 | `os.path.join("resultados", "geojson", f_1)` | `str(get_paths().output_dir / "geojson" / f_1)` |
| 709 | `os.path.join("resultados", frm, archivo)` | `str(get_paths().output_dir / frm / archivo)` |
| 773 | `os.path.join("resultados", "geojson", f_0)` | `str(get_paths().output_dir / "geojson" / f_0)` |
| 774 | `os.path.join("resultados", "geojson", f_1)` | `str(get_paths().output_dir / "geojson" / f_1)` |

- [ ] **Step 3: Update `viz/line_od_matrix.py`**

Add `from urbantrips.utils.paths import get_paths` after existing imports.

| Line | Before | After |
|------|--------|-------|
| 449 | `os.path.join("resultados", frm, archivo)` | `str(get_paths().output_dir / frm / archivo)` |
| 592 | `os.path.join("resultados", "html", savefile)` | `str(get_paths().output_dir / "html" / savefile)` |

- [ ] **Step 4: Update `viz/helpers.py`**

Add `from urbantrips.utils.paths import get_paths` after existing imports.

Line 121: `os.path.join("resultados", "html", savefile)` → `str(get_paths().output_dir / "html" / savefile)`

- [ ] **Step 5: Run tests**

```bash
python -m pytest urbantrips/tests/unit/ -v 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add urbantrips/viz/viz.py urbantrips/viz/section_supply.py urbantrips/viz/line_od_matrix.py urbantrips/viz/helpers.py
git commit -m "refactor: use get_paths().output_dir in viz modules"
```

---

## Task 9: Update output paths in `kpi/`, `cluster/`, `preparo_dashboard/`

**Files:**
- Modify: `urbantrips/kpi/line_od_matrix.py`
- Modify: `urbantrips/cluster/dbscan.py`
- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`

- [ ] **Step 1: Update `kpi/line_od_matrix.py`**

Add `from urbantrips.utils.paths import get_paths` after existing imports.

Line 288: `os.path.join("resultados", "matrices", archivo)` → `str(get_paths().output_dir / "matrices" / archivo)`

- [ ] **Step 2: Update `cluster/dbscan.py`**

Add `from urbantrips.utils.paths import get_paths` after existing imports.

Line 497: `os.path.join("resultados", frm, file_name)` → `str(get_paths().output_dir / frm / file_name)`

Line 668: `os.path.join("resultados", frm, file_name)` → `str(get_paths().output_dir / frm / file_name)`

- [ ] **Step 3: Update `preparo_dashboard/preparo_dashboard.py`**

Add `from urbantrips.utils.paths import get_paths` after existing imports.

Line 833: `os.path.join("resultados", "matrices", f"{savefile}.xlsx")` → `str(get_paths().output_dir / "matrices" / f"{savefile}.xlsx")`

Line 849–851:
```python
# Before:
os.path.join(
    "resultados", "matrices", f"{savefile}_normalizada.xlsx"
)
# After:
str(get_paths().output_dir / "matrices" / f"{savefile}_normalizada.xlsx")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest urbantrips/tests/unit/ -v 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add urbantrips/kpi/line_od_matrix.py urbantrips/cluster/dbscan.py urbantrips/preparo_dashboard/preparo_dashboard.py
git commit -m "refactor: use get_paths().output_dir in kpi, cluster, preparo_dashboard"
```

---

## Task 10: Update dashboard path resolution

**Files:**
- Modify: `urbantrips/dashboard/dash_storage.py`
- Modify: `urbantrips/dashboard/dash_utils.py`

- [ ] **Step 1: Update `dash_storage.py`**

Replace `_get_base_config_path()` to delegate to `get_paths()`:

```python
def _get_base_config_path() -> Path:
    """Return the non-autogenerated config path."""
    from urbantrips.utils.paths import get_paths
    return get_paths().config_file
```

Replace `get_project_root()` to delegate to `get_paths()`:

```python
def get_project_root() -> Path:
    """Return the project root."""
    from urbantrips.utils.paths import get_paths
    return get_paths().base
```

- [ ] **Step 2: Update `dash_utils.py`**

Find the `iniciar_conexion_db` / `resolve_db_path` function (around line 98–113) that builds a list of path candidates:

```python
# Before:
project_root = get_project_root()
candidates = [
    project_root / "data" / "db" / f"{alias_db}{tipo}.duckdb",
    project_root / "data" / "db" / f"{alias_db}{tipo}.sqlite",
    Path("data") / "db" / f"{alias_db}{tipo}.duckdb",
    Path("/data/db") / f"{alias_db}{tipo}.duckdb",
    Path("data") / "db" / f"{alias_db}{tipo}.sqlite",
    Path("/data/db") / f"{alias_db}{tipo}.sqlite",
]
```

```python
# After:
from urbantrips.utils.paths import get_paths
db_dir = get_paths().db_dir
candidates = [
    db_dir / f"{alias_db}{tipo}.duckdb",
    db_dir / f"{alias_db}{tipo}.sqlite",
    Path("data") / "db" / f"{alias_db}{tipo}.duckdb",
    Path("/data/db") / f"{alias_db}{tipo}.duckdb",
    Path("data") / "db" / f"{alias_db}{tipo}.sqlite",
    Path("/data/db") / f"{alias_db}{tipo}.sqlite",
]
```

(Keeping the legacy fallback candidates for any existing deployments that haven't migrated.)

- [ ] **Step 3: Run tests**

```bash
python -m pytest urbantrips/tests/unit/test_dashboard_dash_utils.py -v 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 4: Full unit test suite**

```bash
python -m pytest urbantrips/tests/unit/ -v 2>&1 | tail -30
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add urbantrips/dashboard/dash_storage.py urbantrips/dashboard/dash_utils.py
git commit -m "refactor: use get_paths() in dashboard path resolution"
```

---

## Task 11: Smoke test end-to-end

- [ ] **Step 1: Verify help output shows new flag**

```bash
python urbantrips/run_all_urbantrips.py --help 2>&1 | grep -A2 "base-dir"
```
Expected output contains: `--base-dir`, `-d`, `Project root directory`.

- [ ] **Step 2: Verify backward-compat (no flag)**

```bash
python -c "
from urbantrips.utils.paths import get_paths, reset_paths
reset_paths()
p = get_paths()
print('base:', p.base)
print('input_dir:', p.input_dir)
print('db_dir:', p.db_dir)
print('output_dir:', p.output_dir)
"
```
Expected: all paths are relative to the current working directory.

- [ ] **Step 3: Verify `--base-dir` routing**

```bash
mkdir -p /tmp/ut_test_run/configs
echo "alias_db: test" > /tmp/ut_test_run/configs/configuraciones_generales.yaml
python -c "
import os
os.environ['URBANTRIPS_BASE'] = '/tmp/ut_test_run'
from urbantrips.utils.paths import get_paths, reset_paths
reset_paths()
p = get_paths()
print('base:', p.base)
print('input_dir:', p.input_dir)
print('output_dir:', p.output_dir)
"
```
Expected: all paths under `/tmp/ut_test_run/`.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest urbantrips/tests/unit/ -v 2>&1 | tail -30
```
Expected: all pass.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: base-dir isolation — smoke test verified"
```
