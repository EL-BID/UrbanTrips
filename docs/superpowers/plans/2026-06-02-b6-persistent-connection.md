# B6 — Persistent DuckDB Connection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-call `duckdb.connect()` in all three DuckDB adapters with a single persistent connection opened at construction time.

**Architecture:** Each adapter stores `self._conn` as a live `DuckDBPyConnection`. All methods that previously did `with self._conn() as conn:` now use `self._conn` directly. A `close()` method and `__del__` guard ensure cleanup.

**Tech Stack:** Python, DuckDB 4.x, pytest

---

## File map

- Modify: `urbantrips/storage/adapters/duckdb/insumos.py`
- Modify: `urbantrips/storage/adapters/duckdb/data.py`
- Modify: `urbantrips/storage/adapters/duckdb/dash.py`
- Modify: `urbantrips/storage/adapters/duckdb/general.py`
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py` (add adapter tests)

---

### Task 1: Write failing tests for persistent connection

**Files:**
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

- [ ] **Step 1: Add tests to `test_preparo_dashboard.py`**

Append this to the end of the file:

```python
# ---------------------------------------------------------------------------
# B6 — persistent DuckDB connection
# ---------------------------------------------------------------------------
import tempfile
from pathlib import Path


def _make_insumos_adapter(tmp_path):
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    return DuckDBInsumoAdapter(tmp_path / "insumos.duckdb")


def test_insumos_adapter_reuses_connection(tmp_path):
    """Two sequential calls share the same connection object."""
    adapter = _make_insumos_adapter(tmp_path)
    conn1 = adapter._conn
    conn2 = adapter._conn
    assert conn1 is conn2
    adapter.close()


def test_insumos_adapter_read_write_cycle(tmp_path):
    """save_raw then get_raw returns the same data."""
    import pandas as pd
    adapter = _make_insumos_adapter(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    adapter.save_raw(df, "test_table")
    result = adapter.get_raw("test_table")
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))
    adapter.close()


def test_insumos_adapter_close_is_idempotent(tmp_path):
    """Calling close() twice must not raise."""
    adapter = _make_insumos_adapter(tmp_path)
    adapter.close()
    adapter.close()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_insumos_adapter_reuses_connection urbantrips/tests/unit/test_preparo_dashboard.py::test_insumos_adapter_read_write_cycle urbantrips/tests/unit/test_preparo_dashboard.py::test_insumos_adapter_close_is_idempotent -v
```

Expected: FAIL — `adapter._conn` is currently a method, not a connection object.

---

### Task 2: Refactor `DuckDBInsumoAdapter` to persistent connection

**Files:**
- Modify: `urbantrips/storage/adapters/duckdb/insumos.py`

- [ ] **Step 1: Replace `_conn()` method with persistent `self._conn` attribute**

Replace the `__init__` and `_conn` method:

```python
# BEFORE
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._apply_schema()

def _conn(self) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(self._path))
```

```python
# AFTER
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = duckdb.connect(str(self._path))
    self._apply_schema()

def close(self) -> None:
    if self._conn is not None:
        self._conn.close()
        self._conn = None

def __del__(self) -> None:
    self.close()
```

- [ ] **Step 2: Replace all `with self._conn() as conn:` blocks**

Every method in the file uses `with self._conn() as conn:`. Replace ALL occurrences with direct use of `self._conn`. The pattern is:

```python
# BEFORE
with self._conn() as conn:
    conn.execute(...)

# AFTER
self._conn.execute(...)
```

For methods that return a value:

```python
# BEFORE
with self._conn() as conn:
    return conn.execute("SELECT ...").fetchdf()

# AFTER
return self._conn.execute("SELECT ...").fetchdf()
```

For methods with `conn.register` / `conn.unregister`:

```python
# BEFORE
with self._conn() as conn:
    conn.register("_df", flat)
    try:
        conn.execute("INSERT INTO lines_geoms SELECT id_linea, wkt FROM _df")
    finally:
        conn.unregister("_df")

# AFTER
self._conn.register("_df", flat)
try:
    self._conn.execute("INSERT INTO lines_geoms SELECT id_linea, wkt FROM _df")
finally:
    self._conn.unregister("_df")
```

Apply this to every method: `_apply_schema`, `get_routes`, `get_stops`, `get_distances`, `get_metadata_lineas`, `get_metadata_ramales`, `get_matrix_validation`, `get_travel_times_stations`, `save_routes`, `save_stops`, `save_distances`, `save_matrix_validation`, `save_travel_times_stations`, `save_metadata_lineas`, `save_metadata_ramales`, `execute`, `query`, `save_raw`, `get_raw`, `append_raw`.

- [ ] **Step 3: Run tests**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_insumos_adapter_reuses_connection urbantrips/tests/unit/test_preparo_dashboard.py::test_insumos_adapter_read_write_cycle urbantrips/tests/unit/test_preparo_dashboard.py::test_insumos_adapter_close_is_idempotent -v
```

Expected: all 3 PASS.

---

### Task 3: Apply same refactor to `DuckDBDataAdapter`

**Files:**
- Modify: `urbantrips/storage/adapters/duckdb/data.py`

- [ ] **Step 1: Replace `__init__` and `_conn` in `DuckDBDataAdapter`**

```python
# BEFORE
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._read_only = False
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._apply_schema()

def _conn(self) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(self._path), read_only=getattr(self, "_read_only", False))
```

```python
# AFTER
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._read_only = False
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = duckdb.connect(str(self._path), read_only=self._read_only)
    self._apply_schema()

def close(self) -> None:
    if self._conn is not None:
        self._conn.close()
        self._conn = None

def __del__(self) -> None:
    self.close()
```

- [ ] **Step 2: Replace all `with self._conn() as conn:` throughout `data.py`**

Same pattern as Task 2 Step 2 — replace every `with self._conn() as conn:` block, using `self._conn` directly.

- [ ] **Step 3: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 4: Apply same refactor to `DuckDBGeneralAdapter`

**Files:**
- Modify: `urbantrips/storage/adapters/duckdb/general.py`

- [ ] **Step 1: Replace `__init__` and `_conn` in `DuckDBGeneralAdapter`**

```python
# BEFORE
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._apply_schema()

def _conn(self) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(self._path))
```

```python
# AFTER
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = duckdb.connect(str(self._path))
    self._apply_schema()

def close(self) -> None:
    if self._conn is not None:
        self._conn.close()
        self._conn = None

def __del__(self) -> None:
    self.close()
```

- [ ] **Step 2: Replace all `with self._conn() as conn:` throughout `general.py`**

Same pattern as previous adapters.

- [ ] **Step 3: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 5: Apply same refactor to `DuckDBDashAdapter`

**Files:**
- Modify: `urbantrips/storage/adapters/duckdb/dash.py`

- [ ] **Step 1: Replace `__init__` and `_conn` in `DuckDBDashAdapter`**

```python
# BEFORE
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._apply_schema()

def _conn(self) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(self._path))
```

```python
# AFTER
def __init__(self, db_path: Path) -> None:
    self._path = Path(db_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = duckdb.connect(str(self._path))
    self._apply_schema()

def close(self) -> None:
    if self._conn is not None:
        self._conn.close()
        self._conn = None

def __del__(self) -> None:
    self.close()
```

- [ ] **Step 2: Replace all `with self._conn() as conn:` throughout `dash.py`**

Same pattern as previous adapters.

- [ ] **Step 3: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 6: Commit

- [ ] **Step 1: Commit**

```
git add urbantrips/storage/adapters/duckdb/insumos.py urbantrips/storage/adapters/duckdb/data.py urbantrips/storage/adapters/duckdb/dash.py urbantrips/storage/adapters/duckdb/general.py urbantrips/tests/unit/test_preparo_dashboard.py
git commit -m "perf(b6): persistent DuckDB connection in all adapters"
```
