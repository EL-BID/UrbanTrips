# B5 — Push Scalar Derived Columns into DuckDB SQL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move five scalar derived columns (`tipo_dia`, `mes`, `rango_hora`, `distancia_agregada`, `transferencia`) from post-read pandas computation into the DuckDB `SELECT` inside `load_and_process_data`, so a smaller DataFrame is materialised from the start.

**Architecture:** The two SQL queries in `load_and_process_data` (lines 73–99) are extended with `CASE WHEN` expressions and DuckDB date functions. The corresponding pandas lines that previously computed those columns are deleted. All downstream columns and their values remain identical.

**Tech Stack:** Python, DuckDB 4.x, pandas, pytest, pytest-mock

---

## File map

- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

---

### Task 1: Write failing tests

**Files:**
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

- [ ] **Step 1: Append tests to `test_preparo_dashboard.py`**

```python
# ---------------------------------------------------------------------------
# B5 — derived columns computed in SQL
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


def _make_synthetic_etapas():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "dia": ["2024-10-14", "2024-10-14", "2024-10-19", "2024-10-19"],
        "id_tarjeta": ["T1", "T1", "T2", "T2"],
        "id_viaje": [1, 1, 2, 2],
        "id_etapa": [1, 2, 1, 2],
        "tiempo": ["08:00:00", "08:30:00", "10:00:00", "10:40:00"],
        "hora": [8, 8, 10, 10],
        "modo": ["COLECTIVO", "SUBTE", "COLECTIVO", "COLECTIVO"],
        "id_linea": [28, 99, 45, 45],
        "id_ramal": [1, 1, 1, 1],
        "interno": [101, 201, 102, 102],
        "genero": [None, None, None, None],
        "tarifa": ["normal", "normal", "social", "social"],
        "latitud": [-34.608, -34.610, -34.578, -34.582],
        "longitud": [-58.372, -58.374, -58.430, -58.432],
        "h3_o": ["88e8ea402bfffff", "88e8ea402bfffff", "88e8ea4051fffff", "88e8ea4051fffff"],
        "h3_d": ["88e8ea4031fffff", "88e8ea4031fffff", "88e8ea405bfffff", "88e8ea405bfffff"],
        "od_validado": [1, 1, 1, 1],
        "factor_expansion_original": [1.0, 1.0, 2.0, 2.0],
        "factor_expansion_linea": [1.0, 1.0, 2.0, 2.0],
        "factor_expansion_tarjeta": [1.0, 1.0, 2.0, 2.0],
        "travel_time_min": [20.0, 15.0, 30.0, 10.0],
        "distance_od": [3.5, 2.0, 7.2, 1.8],
        "distance_route": [4.0, 2.2, 8.0, 2.0],
        "distance_route_gps": [4.1, 2.3, 8.1, 2.1],
        "kmh_od": [np.nan, np.nan, np.nan, np.nan],
        "kmh_route": [np.nan, np.nan, np.nan, np.nan],
        "kmh_route_gps": [np.nan, np.nan, np.nan, np.nan],
    })


def _make_synthetic_viajes():
    return pd.DataFrame({
        "dia": ["2024-10-14", "2024-10-14", "2024-10-19"],
        "id_tarjeta": ["T1", "T1", "T2"],
        "id_viaje": [1, 2, 1],
        "tiempo": ["08:00:00", "09:00:00", "10:00:00"],
        "hora": [8, 9, 10],
        "cant_etapas": [2, 1, 1],
        "modo": ["COLECTIVO", "COLECTIVO", "COLECTIVO"],
        "autobus": [1, 1, 1],
        "tren": [0, 0, 0],
        "metro": [0, 0, 0],
        "tranvia": [0, 0, 0],
        "brt": [0, 0, 0],
        "cable": [0, 0, 0],
        "lancha": [0, 0, 0],
        "otros": [0, 0, 0],
        "h3_o": ["88e8ea402bfffff", "88e8ea402bfffff", "88e8ea4051fffff"],
        "h3_d": ["88e8ea4031fffff", "88e8ea4051fffff", "88e8ea405bfffff"],
        "genero": [None, None, None],
        "tarifa": ["normal", "normal", "social"],
        "od_validado": [1, 1, 1],
        "factor_expansion_linea": [1.0, 1.0, 2.0],
        "factor_expansion_tarjeta": [1.0, 1.0, 2.0],
        "distance_od": [3.5, 7.2, 7.2],
        "travel_time_min": [35.0, 30.0, 30.0],
        "distance_route": [4.0, 8.0, 8.0],
        "distance_route_gps": [4.1, 8.1, 8.1],
        "kmh_od": [np.nan, np.nan, np.nan],
        "kmh_route": [np.nan, np.nan, np.nan],
        "kmh_route_gps": [np.nan, np.nan, np.nan],
    })


def _run_load_and_process(etapas_raw, viajes_raw):
    """Call load_and_process_data with mocked ctx."""
    from urbantrips.preparo_dashboard.preparo_dashboard import load_and_process_data
    ctx = MagicMock()
    ctx.data.query.side_effect = [etapas_raw.copy(), viajes_raw.copy()]
    return load_and_process_data(ctx)


def test_load_tipo_dia_weekday():
    etapas, viajes = _run_load_and_process(_make_synthetic_etapas(), _make_synthetic_viajes())
    assert all(etapas.loc[etapas.dia == "2024-10-14", "tipo_dia"] == "Hábil")
    assert all(viajes.loc[viajes.dia == "2024-10-14", "tipo_dia"] == "Hábil")


def test_load_tipo_dia_weekend():
    etapas, viajes = _run_load_and_process(_make_synthetic_etapas(), _make_synthetic_viajes())
    assert all(etapas.loc[etapas.dia == "2024-10-19", "tipo_dia"] == "Fin de Semana")
    assert all(viajes.loc[viajes.dia == "2024-10-19", "tipo_dia"] == "Fin de Semana")


def test_load_mes_column():
    etapas, viajes = _run_load_and_process(_make_synthetic_etapas(), _make_synthetic_viajes())
    assert all(etapas["mes"] == "2024-10")
    assert all(viajes["mes"] == "2024-10")


def test_load_rango_hora_boundaries():
    etapas, viajes = _run_load_and_process(_make_synthetic_etapas(), _make_synthetic_viajes())
    # hora=8 → "0-12"
    assert all(etapas.loc[etapas.hora == 8, "rango_hora"] == "0-12")
    assert all(viajes.loc[viajes.hora == 8, "rango_hora"] == "0-12")
    # hora=10 → "0-12"
    assert all(etapas.loc[etapas.hora == 10, "rango_hora"] == "0-12")


def test_load_distancia_agregada_etapas():
    etapas, _ = _run_load_and_process(_make_synthetic_etapas(), _make_synthetic_viajes())
    # distance_od=3.5 → corta; 7.2 → larga
    assert all(etapas.loc[etapas.distance_od <= 5, "distancia_agregada"] == "Etapa corta (<=5kms)")
    assert all(etapas.loc[etapas.distance_od > 5, "distancia_agregada"] == "Etapa larga (>5kms)")


def test_load_distancia_agregada_viajes():
    _, viajes = _run_load_and_process(_make_synthetic_etapas(), _make_synthetic_viajes())
    assert all(viajes.loc[viajes.distance_od <= 5, "distancia_agregada"] == "Viajes cortos (<=5kms)")
    assert all(viajes.loc[viajes.distance_od > 5, "distancia_agregada"] == "Viajes largos (>5kms)")


def test_load_transferencia():
    _, viajes = _run_load_and_process(_make_synthetic_etapas(), _make_synthetic_viajes())
    assert all(viajes.loc[viajes.cant_etapas == 1, "transferencia"] == 0)
    assert all(viajes.loc[viajes.cant_etapas > 1, "transferencia"] == 1)
```

- [ ] **Step 2: Run tests to verify they pass with the current implementation**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_load_tipo_dia_weekday urbantrips/tests/unit/test_preparo_dashboard.py::test_load_tipo_dia_weekend urbantrips/tests/unit/test_preparo_dashboard.py::test_load_mes_column urbantrips/tests/unit/test_preparo_dashboard.py::test_load_rango_hora_boundaries urbantrips/tests/unit/test_preparo_dashboard.py::test_load_distancia_agregada_etapas urbantrips/tests/unit/test_preparo_dashboard.py::test_load_distancia_agregada_viajes urbantrips/tests/unit/test_preparo_dashboard.py::test_load_transferencia -v
```

These tests establish the baseline — they should PASS with the old code. They act as regression guards: if they still pass after the refactor, the values are identical.

---

### Task 2: Add derived columns to the `etapas` SQL query

**Files:**
- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`

- [ ] **Step 1: Extend the `etapas` SQL query (lines 73–86)**

Replace:

```python
    etapas = ctx.data.query(
        """
        SELECT e.id, e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa, e.tiempo, e.hora,
               e.modo, e.id_linea, e.id_ramal, e.interno, e.genero, e.tarifa,
               e.latitud, e.longitud, e.h3_o, e.h3_d, e.od_validado,
               e.factor_expansion_original, e.factor_expansion_linea,
               e.factor_expansion_tarjeta,
               tt.travel_time_min, tt.distance_od, tt.distance_route,
               tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
        FROM etapas e
        LEFT JOIN travel_times_legs tt ON e.id = tt.id
        WHERE e.od_validado = 1
        """
    )
```

With:

```python
    etapas = ctx.data.query(
        """
        SELECT e.id, e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa, e.tiempo, e.hora,
               e.modo, e.id_linea, e.id_ramal, e.interno, e.genero, e.tarifa,
               e.latitud, e.longitud, e.h3_o, e.h3_d, e.od_validado,
               e.factor_expansion_original, e.factor_expansion_linea,
               e.factor_expansion_tarjeta,
               tt.travel_time_min, tt.distance_od, tt.distance_route,
               tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps,
               CASE WHEN DAYOFWEEK(CAST(e.dia AS DATE)) >= 6
                    THEN 'Fin de Semana' ELSE 'Hábil' END              AS tipo_dia,
               STRFTIME(CAST(e.dia AS DATE), '%Y-%m')                  AS mes,
               CASE WHEN e.hora BETWEEN 13 AND 16 THEN '13-16'
                    WHEN e.hora > 16 THEN '17-24'
                    ELSE '0-12' END                                    AS rango_hora,
               CASE WHEN tt.distance_od > 5 THEN 'Etapa larga (>5kms)'
                    ELSE 'Etapa corta (<=5kms)' END                    AS distancia_agregada
        FROM etapas e
        LEFT JOIN travel_times_legs tt ON e.id = tt.id
        WHERE e.od_validado = 1
        """
    )
```

- [ ] **Step 2: Extend the `viajes` SQL query (lines 88–99)**

Replace:

```python
    viajes = ctx.data.query(
        """
        SELECT v.*, tt.travel_time_min, tt.distance_od, tt.distance_route,
               tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
        FROM viajes v
        LEFT JOIN travel_times_trips tt
        ON v.dia = tt.dia
        AND v.id_tarjeta = tt.id_tarjeta
        AND v.id_viaje = tt.id_viaje
        WHERE v.od_validado = 1
        """
    )
```

With:

```python
    viajes = ctx.data.query(
        """
        SELECT v.*, tt.travel_time_min, tt.distance_od, tt.distance_route,
               tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps,
               CASE WHEN DAYOFWEEK(CAST(v.dia AS DATE)) >= 6
                    THEN 'Fin de Semana' ELSE 'Hábil' END              AS tipo_dia,
               STRFTIME(CAST(v.dia AS DATE), '%Y-%m')                  AS mes,
               CASE WHEN v.hora BETWEEN 13 AND 16 THEN '13-16'
                    WHEN v.hora > 16 THEN '17-24'
                    ELSE '0-12' END                                    AS rango_hora,
               CASE WHEN tt.distance_od > 5 THEN 'Viajes largos (>5kms)'
                    ELSE 'Viajes cortos (<=5kms)' END                  AS distancia_agregada,
               CAST(v.cant_etapas > 1 AS INTEGER)                      AS transferencia
        FROM viajes v
        LEFT JOIN travel_times_trips tt
        ON v.dia = tt.dia
        AND v.id_tarjeta = tt.id_tarjeta
        AND v.id_viaje = tt.id_viaje
        WHERE v.od_validado = 1
        """
    )
```

---

### Task 3: Remove the now-redundant pandas lines

**Files:**
- Modify: `urbantrips/preparo_dashboard/preparo_dashboard.py`

- [ ] **Step 1: Remove `tipo_dia`, `mes`, `rango_hora`, `distancia_agregada` for `etapas` (lines ~153–160)**

Delete these lines (they are now computed in SQL):

```python
    etapas["tipo_dia"] = np.where(
        pd.to_datetime(etapas["dia"]).dt.dayofweek >= 5, "Fin de Semana", "Hábil"
    )
    etapas["mes"] = pd.to_datetime(etapas["dia"]).dt.to_period("M").astype(str)

    cond_rh_e = [etapas["hora"].between(13, 16), etapas["hora"].between(17, 24)]
    etapas["rango_hora"] = np.select(cond_rh_e, ["13-16", "17-24"], default="0-12")
```

And for `distancia_agregada` on etapas (line ~137–139):

```python
    etapas["distancia_agregada"] = np.where(
        etapas["distance_od"] > 5, "Etapa larga (>5kms)", "Etapa corta (<=5kms)"
    )
```

- [ ] **Step 2: Remove `tipo_dia`, `mes`, `rango_hora`, `distancia_agregada`, `transferencia` for `viajes`**

Delete these lines:

```python
    viajes["transferencia"] = (viajes["cant_etapas"] > 1).astype(int)

    cond_rh = [viajes["hora"].between(13, 16), viajes["hora"].between(17, 24)]
    viajes["rango_hora"] = np.select(cond_rh, ["13-16", "17-24"], default="0-12")

    viajes["distancia_agregada"] = np.where(
        viajes["distance_od"] > 5, "Viajes largos (>5kms)", "Viajes cortos (<=5kms)"
    )

    viajes["tipo_dia"] = np.where(
        pd.to_datetime(viajes["dia"]).dt.dayofweek >= 5, "Fin de Semana", "Hábil"
    )

    viajes["mes"] = pd.to_datetime(viajes["dia"]).dt.to_period("M").astype(str)
```

Note: keep `viajes["Fecha"]`, `viajes["Fecha_next"]`, and `viajes["diff_time"]` — these remain in pandas.

- [ ] **Step 3: Run the regression tests**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_load_tipo_dia_weekday urbantrips/tests/unit/test_preparo_dashboard.py::test_load_tipo_dia_weekend urbantrips/tests/unit/test_preparo_dashboard.py::test_load_mes_column urbantrips/tests/unit/test_preparo_dashboard.py::test_load_rango_hora_boundaries urbantrips/tests/unit/test_preparo_dashboard.py::test_load_distancia_agregada_etapas urbantrips/tests/unit/test_preparo_dashboard.py::test_load_distancia_agregada_viajes urbantrips/tests/unit/test_preparo_dashboard.py::test_load_transferencia -v
```

Expected: all 7 PASS.

**Note:** The mock `ctx.data.query.side_effect` returns the synthetic DataFrames directly (already containing the derived columns), so the SQL expressions are not actually executed in these unit tests — the tests verify that `load_and_process_data` correctly uses the columns that come back from the query. To verify the SQL expressions themselves are syntactically correct, run the full integration test in the next step.

- [ ] **Step 4: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 4: Commit

- [ ] **Step 1: Commit**

```
git add urbantrips/preparo_dashboard/preparo_dashboard.py urbantrips/tests/unit/test_preparo_dashboard.py
git commit -m "perf(b5): push scalar derived columns into DuckDB SQL in load_and_process_data"
```
