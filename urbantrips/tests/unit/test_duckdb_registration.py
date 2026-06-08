# urbantrips/tests/unit/test_duckdb_registration.py
"""
Regression tests for DuckDB Arrow-registration memory leaks.

Root cause
----------
``conn.register(name, df)`` creates an Arrow-based virtual relation that holds
a raw pointer into the pandas DataFrame's numpy memory.  If ``conn.unregister``
is **not** called before the connection closes, DuckDB 1.5.x may retain that
pointer in its MVCC cleanup queue.  When Python later frees the DataFrame,
DuckDB's deferred cleanup touches freed memory → malloc free-list corruption →
process crash at an unrelated point (often many minutes later).

What these tests verify
-----------------------
1. Every ``conn.register`` call in every DuckDB adapter save-method is matched
   by ``conn.unregister`` before the method returns (success path).
2. ``conn.unregister`` is still called even when ``conn.execute`` raises an
   exception (the ``finally`` block works).
3. ``save_legs`` — rewritten to use parquet staging — issues **zero**
   ``conn.register`` calls at all.

Strategy
--------
We inject a lightweight tracking proxy between the adapter and its real DuckDB
connection.  The proxy intercepts ``register``/``unregister`` calls and keeps a
running list of "active" registrations.  After each adapter method returns the
list must be empty.  Using a real (temp-file) DuckDB database ensures the full
code path executes; zero-row DataFrames are used so NOT-NULL constraints are
never violated.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import duckdb
import pandas as pd
import pytest


# ─── Tracking infrastructure ─────────────────────────────────────────────────


def _setup_tracking(adapter) -> list[str]:
    """
    Replace *adapter._conn* with a factory that returns tracking proxies.

    All proxies created by the same call share a single ``active`` list, so
    dangling registrations accumulate across multiple ``_conn()`` calls within
    one adapter method (there is always exactly one, but the design is safe for
    any number).

    Returns the shared ``active`` list; callers assert it is ``[]`` after each
    save-method call.
    """
    db_path = adapter._path
    active: list[str] = []

    def _conn():
        real = duckdb.connect(str(db_path))

        class _Proxy:
            # ── context manager ────────────────────────────────────────────
            def __enter__(self_):
                return self_          # must return proxy, not real

            def __exit__(self_, *args):
                real.close()
                return False

            # ── tracked ───────────────────────────────────────────────────
            def register(self_, name: str, df) -> None:
                active.append(name)
                return real.register(name, df)

            def unregister(self_, name: str) -> None:
                try:
                    active.remove(name)
                except ValueError:
                    pass              # already removed (shouldn't happen)
                return real.unregister(name)

            # ── forward everything else ───────────────────────────────────
            def __getattr__(self_, item):
                return getattr(real, item)

        return _Proxy()

    adapter._conn = _conn
    return active


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def data_adapter(tmp_path):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    active = _setup_tracking(adapter)
    return adapter, active


# ─── Minimal zero-row DataFrames for each save method ────────────────────────

def _empty(**cols) -> pd.DataFrame:
    """Return a 0-row DataFrame with the given column→dtype mapping."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


_RUN_DAYS_DF = _empty(dia=str)

_TRX_COLS = [
    "id", "batch_id", "fecha", "id_original", "id_tarjeta", "dia", "tiempo",
    "hora", "modo", "id_linea", "id_ramal", "interno", "orden_trx", "genero",
    "tarifa", "latitud", "longitud", "factor_expansion",
]
_TRX_RAW_COLS = [
    "id_original", "id_tarjeta", "dia", "tiempo", "hora", "modo",
    "id_linea", "id_ramal", "interno", "orden_trx", "genero", "tarifa",
    "latitud", "longitud", "fecha_ts", "factor_expansion_raw",
]
_VIAJES_COLS = [
    "id_tarjeta", "id_viaje", "dia", "tiempo", "hora", "cant_etapas", "modo",
    "autobus", "tren", "metro", "tranvia", "brt", "cable", "lancha", "otros",
    "h3_o", "h3_d", "genero", "tarifa", "od_validado",
    "factor_expansion_linea", "factor_expansion_tarjeta", "distancia",
    "travel_time_min",
]
_USUARIOS_COLS = [
    "id_tarjeta", "dia", "od_validado", "cant_viajes",
    "factor_expansion_linea", "factor_expansion_tarjeta",
]
_GPS_COLS = [
    "id", "id_original", "dia", "id_linea", "id_ramal", "interno",
    "fecha", "latitud", "longitud", "velocity", "service_type",
    "distance_km", "h3",
]


# ─── Success-path tests ───────────────────────────────────────────────────────


def test_save_run_days_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_run_days(_RUN_DAYS_DF)
    assert active == [], f"Dangling registrations: {active}"


def test_save_transactions_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_transactions(_empty(**{c: object for c in _TRX_COLS}))
    assert active == [], f"Dangling registrations: {active}"


def test_save_raw_chunk_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_raw_chunk(_empty(**{c: object for c in _TRX_RAW_COLS}))
    assert active == [], f"Dangling registrations: {active}"


def test_save_trips_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_trips(_empty(**{c: object for c in _VIAJES_COLS}))
    assert active == [], f"Dangling registrations: {active}"


def test_save_users_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_users(_empty(**{c: object for c in _USUARIOS_COLS}))
    assert active == [], f"Dangling registrations: {active}"


def test_save_gps_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_gps(_empty(**{c: object for c in _GPS_COLS}))
    assert active == [], f"Dangling registrations: {active}"


def test_save_indicators_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_indicators(_empty(dia=str, valor=float))
    assert active == [], f"Dangling registrations: {active}"


def test_save_raw_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.save_raw(_empty(x=float, y=float), "test_raw_table")
    assert active == [], f"Dangling registrations: {active}"


def test_append_raw_no_dangling(data_adapter):
    adapter, active = data_adapter
    adapter.append_raw(_empty(x=float, y=float), "test_append_table")
    assert active == [], f"Dangling registrations: {active}"


# ─── Exception-path test ─────────────────────────────────────────────────────


def test_unregister_called_even_on_execute_failure(data_adapter):
    """
    The ``finally: conn.unregister(name)`` guard must fire even when
    ``conn.execute`` raises.  This test forces a DuckDB error by making
    the first INSERT call raise via a custom proxy's execute method.
    """
    adapter, active = data_adapter
    db_path = adapter._path

    def failing_conn():
        real = duckdb.connect(str(db_path))
        fail_flag = [True]

        class FailingTrackingProxy:
            def __enter__(self_):
                return self_

            def __exit__(self_, *args):
                real.close()
                return False

            def register(self_, name: str, df) -> None:
                active.append(name)
                return real.register(name, df)

            def unregister(self_, name: str) -> None:
                try:
                    active.remove(name)
                except ValueError:
                    pass
                return real.unregister(name)

            def execute(self_, sql: str, *args, **kwargs):
                # Fail on the first data-manipulation statement that
                # references the registered view (CREATE OR REPLACE or
                # INSERT … SELECT … FROM _raw_df).
                if fail_flag[0] and "_raw_df" in sql:
                    fail_flag[0] = False
                    raise duckdb.Error("Simulated INSERT failure")
                return real.execute(sql, *args, **kwargs)

            def __getattr__(self_, item):
                return getattr(real, item)

        return FailingTrackingProxy()

    adapter._conn = failing_conn

    with pytest.raises(duckdb.Error, match="Simulated INSERT failure"):
        adapter.save_raw(_empty(x=float), "will_fail")

    assert active == [], (
        f"register/unregister imbalance after exception: {active}"
    )


# ─── save_legs must NOT use conn.register at all ─────────────────────────────


def test_save_legs_uses_no_arrow_registration(data_adapter):
    """
    save_legs was rewritten to use parquet staging instead of conn.register().
    This test verifies the parquet path is in use: zero register calls are made
    regardless of data size.
    """
    from urbantrips.storage.ports import BatchSpec

    adapter, active = data_adapter

    legs = _empty(**{
        "id": int, "batch_id": int, "id_tarjeta": object, "dia": object,
        "id_viaje": int, "id_etapa": int, "tiempo": object, "hora": int,
        "modo": object, "id_linea": int, "id_ramal": int, "interno": int,
        "genero": object, "tarifa": object, "latitud": float, "longitud": float,
        "h3_o": object, "h3_d": object, "od_validado": int,
        "etapa_validada": int, "factor_expansion_original": float,
        "factor_expansion_linea": float, "factor_expansion_tarjeta": float,
        "factor_expansion_etapa": float, "distancia": float,
        "travel_time_min": float,
    })

    batch = BatchSpec(batch_id=0, total_batches=1)
    adapter.save_legs(legs, batch=batch)

    assert active == [], (
        f"save_legs issued unexpected register calls: {active}"
    )


# ─── compute_distances cache functions ───────────────────────────────────────


def _make_tracking_proxy(real_con, active: list[str]):
    """
    Wrap a real DuckDB connection in a tracking proxy.

    DuckDB's C-extension connection object has read-only ``register`` /
    ``unregister`` attributes, so we cannot monkey-patch them directly.
    Instead we build a lightweight proxy class that intercepts those two
    calls and forwards everything else to the real connection.
    """

    class _ConProxy:
        def register(self_, name: str, df) -> None:
            active.append(name)
            return real_con.register(name, df)

        def unregister(self_, name: str) -> None:
            try:
                active.remove(name)
            except ValueError:
                pass
            return real_con.unregister(name)

        def close(self_) -> None:
            return real_con.close()

        def __getattr__(self_, item):
            return getattr(real_con, item)

    return _ConProxy()


def test_query_duckdb_no_dangling(tmp_path):
    """_query_duckdb must unregister _tmp_q before returning."""
    from urbantrips.carto.compute_distances import _init_duckdb, _query_duckdb

    real_con = _init_duckdb(str(tmp_path / "dist.duckdb"))
    active: list[str] = []
    proxy = _make_tracking_proxy(real_con, active)

    pairs = pd.DataFrame({"o_norm": pd.Series(dtype=str),
                          "d_norm": pd.Series(dtype=str)})
    _query_duckdb(pairs, proxy)
    real_con.close()

    assert active == [], f"_query_duckdb left dangling: {active}"


def test_store_duckdb_no_dangling(tmp_path):
    """_store_duckdb must unregister _tmp_ins before returning."""
    from urbantrips.carto.compute_distances import _init_duckdb, _store_duckdb

    real_con = _init_duckdb(str(tmp_path / "dist.duckdb"))
    active: list[str] = []
    proxy = _make_tracking_proxy(real_con, active)

    rows = pd.DataFrame({
        "o_norm": pd.Series(dtype=str),
        "d_norm": pd.Series(dtype=str),
        "distance_m": pd.Series(dtype=float),
    })
    _store_duckdb(rows, proxy)
    real_con.close()

    assert active == [], f"_store_duckdb left dangling: {active}"
