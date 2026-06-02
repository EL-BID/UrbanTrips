# B3 — DBSCAN Grid Reduction + Early Stopping + Tuning Config

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the DBSCAN parameter grid from 20×20 to a configurable default of 5×5, add early stopping on silhouette score, and expose both values via an optional `configs/tuning.yaml` file so they are never magic numbers.

**Architecture:** A new `leer_configs_tuning()` function in `urbantrips/utils/utils.py` reads `configs/tuning.yaml` if present, merging its values over hardcoded defaults. `cluster/dbscan.py` calls it to get `grid_steps` and `early_stop_silhouette`. The `.gitignore` already excludes `configs/*`; a `!configs/tuning.yaml.example` exception is added so the example file is tracked.

**Tech Stack:** Python, scikit-learn, numpy, PyYAML, pytest

---

## File map

- Modify: `urbantrips/utils/utils.py` (add `leer_configs_tuning`)
- Modify: `urbantrips/cluster/dbscan.py` (use tuning config, add early stopping)
- Create: `configs/tuning.yaml.example`
- Modify: `.gitignore` (add exception for `tuning.yaml.example`)
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py` (add DBSCAN + config tests)

---

### Task 1: Write failing tests

**Files:**
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

- [ ] **Step 1: Append tests to `test_preparo_dashboard.py`**

```python
# ---------------------------------------------------------------------------
# B3 — DBSCAN tuning config + grid reduction
# ---------------------------------------------------------------------------
import time


def test_leer_configs_tuning_returns_defaults_when_no_file(tmp_path, monkeypatch):
    """Returns hardcoded defaults when configs/tuning.yaml does not exist."""
    monkeypatch.chdir(tmp_path)
    from urbantrips.utils.utils import leer_configs_tuning
    cfg = leer_configs_tuning()
    assert cfg["dbscan"]["grid_steps"] == 5
    assert cfg["dbscan"]["early_stop_silhouette"] == 0.7


def test_leer_configs_tuning_overrides_from_file(tmp_path, monkeypatch):
    """Values in tuning.yaml override the defaults."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tuning.yaml").write_text(
        "dbscan:\n  grid_steps: 10\n  early_stop_silhouette: 0.5\n"
    )
    import importlib
    import urbantrips.utils.utils as utils_mod
    importlib.reload(utils_mod)  # reload so leer_configs_tuning sees fresh import
    from urbantrips.utils.utils import leer_configs_tuning
    cfg = leer_configs_tuning()
    assert cfg["dbscan"]["grid_steps"] == 10
    assert cfg["dbscan"]["early_stop_silhouette"] == 0.5


def test_dbscan_grid_respects_grid_steps(monkeypatch):
    """cluster_legs_lrs uses grid_steps from tuning config."""
    import numpy as np
    from unittest.mock import patch

    call_count = {"n": 0}
    original_dbscan_fit = None

    # Patch DBSCAN.fit to count calls
    import sklearn.cluster
    original_fit = sklearn.cluster.DBSCAN.fit

    def counting_fit(self, X, y=None, sample_weight=None):
        call_count["n"] += 1
        return original_fit(self, X, y=y, sample_weight=sample_weight)

    # Build synthetic legs for Buenos Aires colectivo line 28
    # Two clear clusters along Avenida Corrientes
    np.random.seed(42)
    n = 60
    # Cluster A: legs between 0.1 and 0.3 on the LRS
    cluster_a = np.random.uniform(0.1, 0.3, (n // 2, 2))
    # Cluster B: legs between 0.6 and 0.8 on the LRS
    cluster_b = np.random.uniform(0.6, 0.8, (n // 2, 2))
    X = pd.DataFrame(
        np.vstack([cluster_a, cluster_b]),
        columns=["o_proj", "d_proj"],
    )
    w = pd.Series(np.ones(n))

    with patch("urbantrips.utils.utils.leer_configs_tuning",
               return_value={"dbscan": {"grid_steps": 3, "early_stop_silhouette": 0.99}}):
        with patch.object(sklearn.cluster.DBSCAN, "fit", counting_fit):
            from urbantrips.cluster.dbscan import cluster_legs_lrs
            # cluster_legs_lrs expects X as DataFrame with o_proj/d_proj cols
            # We call the internal grid-search helper directly
            from urbantrips.cluster.dbscan import _run_grid_search
            _run_grid_search(X, w, type_k="lrs")

    # grid_steps=3 means 3×3=9 fits maximum (may exit earlier with early stop)
    assert call_count["n"] <= 9 + 3, (
        f"Expected at most 12 DBSCAN fits (9 grid + 3 final), got {call_count['n']}"
    )
```

- [ ] **Step 2: Run tests to see them fail**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_leer_configs_tuning_returns_defaults_when_no_file urbantrips/tests/unit/test_preparo_dashboard.py::test_leer_configs_tuning_overrides_from_file -v
```

Expected: FAIL — `leer_configs_tuning` does not exist yet.

---

### Task 2: Add `leer_configs_tuning` to `utils.py`

**Files:**
- Modify: `urbantrips/utils/utils.py`

- [ ] **Step 1: Add the function after `leer_configs_generales`**

Find the end of `leer_configs_generales` (around line 122 where it `return {}`). Add immediately after:

```python
_TUNING_DEFAULTS: dict = {
    "dbscan": {
        "grid_steps": 5,
        "early_stop_silhouette": 0.7,
    },
}


def leer_configs_tuning() -> dict:
    """
    Load optional performance-tuning parameters from configs/tuning.yaml.
    Returns hardcoded defaults for any key not present in the file.
    The file is optional — if absent, all defaults apply.
    """
    import copy

    def _deep_merge(base: dict, overrides: dict) -> dict:
        result = copy.deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    path = os.path.join("configs", "tuning.yaml")
    if not os.path.exists(path):
        return copy.deepcopy(_TUNING_DEFAULTS)

    try:
        with open(path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        return _deep_merge(_TUNING_DEFAULTS, overrides)
    except Exception as e:
        logger.warning("Could not load configs/tuning.yaml: %s — using defaults", e)
        return copy.deepcopy(_TUNING_DEFAULTS)
```

Note: `os`, `yaml`, and `logger` are already imported at the top of `utils.py`.

- [ ] **Step 2: Run config tests**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_leer_configs_tuning_returns_defaults_when_no_file urbantrips/tests/unit/test_preparo_dashboard.py::test_leer_configs_tuning_overrides_from_file -v
```

Expected: both PASS.

---

### Task 3: Extract `_run_grid_search` and apply tuning config in `dbscan.py`

**Files:**
- Modify: `urbantrips/cluster/dbscan.py`

- [ ] **Step 1: Add import of `leer_configs_tuning` at top of `dbscan.py`**

```python
from urbantrips.utils.utils import leer_configs_tuning
```

- [ ] **Step 2: Extract the grid-search loop into `_run_grid_search`**

The current grid search is embedded inside `cluster_legs_lrs` and `cluster_legs_4d`. Extract it into a module-level helper. Locate the block that starts at `# set initial benchmarks` (around line 220) and ends just before `logger.debug(...)` (around line 283). Extract it as:

```python
def _run_grid_search(X, w, type_k: str):
    """
    Run DBSCAN grid search over eps and min_samples ranges.
    Returns a dict with keys 'max_groups', 'max_silhouette', 'min_noise',
    each mapping to the (eps, min_samples) tuple that scored best.
    """
    cfg = leer_configs_tuning()
    grid_steps = cfg["dbscan"]["grid_steps"]
    early_stop_silhouette = cfg["dbscan"]["early_stop_silhouette"]

    best_num_clusters = 0
    best_num_noise = float("inf")
    best_silhouette_score = -1

    max_groups_params = None
    max_silhouette_params = None
    min_noise_params = None

    min_samples_range = list(map(int, w.sum() * np.linspace(0.01, 0.5, grid_steps)))

    if type_k == "lrs":
        eps_range = np.linspace(0.01, 0.5, grid_steps)
    elif type_k == "4d":
        eps_range = np.linspace(100, 1000, grid_steps)
    else:
        raise ValueError(f"Unknown type_k: {type_k!r}")

    done = False
    for eps in eps_range:
        if done:
            break
        for min_samples in min_samples_range:
            params = (eps, min_samples)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X, sample_weight=w)
            labels = dbscan.labels_

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            num_noise_df = pd.DataFrame({"labels": labels, "w": w})
            num_noise_df = num_noise_df.groupby("labels").sum()
            try:
                num_noise = num_noise_df.loc[-1, "w"]
            except KeyError:
                num_noise = 0

            if num_clusters > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = -1
                num_noise = float("inf")

            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                max_silhouette_params = params

            if num_clusters > best_num_clusters:
                best_num_clusters = num_clusters
                max_groups_params = params

            if num_noise < best_num_noise:
                best_num_noise = num_noise
                min_noise_params = params

            if best_silhouette_score >= early_stop_silhouette:
                done = True
                break

    logger.debug(
        "Max clusters=%d eps=%s min_samples=%s | Min noise=%d eps=%s min_samples=%s | "
        "Max silhouette=%.3f eps=%s min_samples=%s",
        best_num_clusters,
        max_groups_params[0] if max_groups_params else None,
        max_groups_params[1] if max_groups_params else None,
        best_num_noise,
        min_noise_params[0] if min_noise_params else None,
        min_noise_params[1] if min_noise_params else None,
        best_silhouette_score,
        max_silhouette_params[0] if max_silhouette_params else None,
        max_silhouette_params[1] if max_silhouette_params else None,
    )

    return {
        "max_groups": max_groups_params,
        "max_silhouette": max_silhouette_params,
        "min_noise": min_noise_params,
    }
```

- [ ] **Step 3: Replace the embedded grid-search block in both `cluster_legs_lrs` and `cluster_legs_4d`**

In each function, replace the block from `# set initial benchmarks` through the `logger.debug(...)` call with:

```python
    params = _run_grid_search(X, w, type_k=type_k)
```

Then the existing `for p in params:` loop (lines ~296–303) continues unchanged — it uses the returned dict directly.

- [ ] **Step 4: Run the DBSCAN test**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_dbscan_grid_respects_grid_steps -v
```

Expected: PASS.

- [ ] **Step 5: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 4: Create `configs/tuning.yaml.example` and update `.gitignore`

**Files:**
- Create: `configs/tuning.yaml.example`
- Modify: `.gitignore`

- [ ] **Step 1: Create `configs/tuning.yaml.example`**

```yaml
# configs/tuning.yaml — Advanced performance tuning for UrbanTrips.
# Copy this file to configs/tuning.yaml and edit to override defaults.
# This file is safe to leave absent; all keys fall back to the defaults below.

dbscan:
  # Number of steps per axis in the DBSCAN parameter grid search.
  # Total DBSCAN fits = grid_steps². Default 5 → 25 fits per route direction.
  # Increase for higher-quality clustering at the cost of longer run times.
  grid_steps: 5

  # Stop the parameter search early if silhouette score exceeds this threshold.
  # Range: -1.0 (worst) to 1.0 (perfect). 0.7 is strong separation.
  # Set to 1.0 to disable early stopping.
  early_stop_silhouette: 0.7
```

- [ ] **Step 2: Add exception to `.gitignore`**

Find the existing line in `.gitignore`:
```
!configs/configuraciones_generales.yaml
```

Add immediately after it:
```
!configs/tuning.yaml.example
```

- [ ] **Step 3: Verify example file is tracked**

```
git status configs/tuning.yaml.example
```

Expected: shows as untracked (new file, not ignored).

---

### Task 5: Commit

- [ ] **Step 1: Commit**

```
git add urbantrips/utils/utils.py urbantrips/cluster/dbscan.py configs/tuning.yaml.example .gitignore urbantrips/tests/unit/test_preparo_dashboard.py
git commit -m "perf(b3): DBSCAN grid reduction, early stopping, and tuning config"
```
