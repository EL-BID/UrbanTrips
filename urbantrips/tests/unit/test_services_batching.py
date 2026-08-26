"""Lectura del gps por lotes de líneas en process_services (fix de OOM a escala mes).

Antes se bajaba TODO el gps de la corrida en un solo DataFrame con `SELECT g.*`: a
escala mes (105 M pings, 5 columnas TEXT que en pandas son un `str` por fila) son
~42 GB y el proceso muere por OOM. Ahora se lee por lotes de líneas y solo las
columnas que el módulo usa.

importorskip: services.py arrastra geopandas al importar.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock

services = pytest.importorskip("urbantrips.datamodel.services")


def _counts(pairs):
    return pd.DataFrame(pairs, columns=["id_linea", "n"])


# ── _line_batches ───────────────────────────────────────────────────────────────

def test_lotes_respetan_el_tope_de_filas():
    counts = _counts([(1, 40), (2, 40), (3, 40), (4, 10)])
    batches = services._line_batches(counts, max_rows=100)
    assert batches == [[1, 2], [3, 4]]


def test_una_linea_mas_grande_que_el_tope_va_sola():
    """La línea es indivisible: process_line_services clasifica su traza completa."""
    counts = _counts([(1, 10), (2, 500), (3, 10)])
    batches = services._line_batches(counts, max_rows=100)
    assert batches == [[1], [2], [3]]


def test_todas_las_lineas_entran_exactamente_una_vez():
    counts = _counts([(i, 30) for i in range(1, 21)])
    batches = services._line_batches(counts, max_rows=100)
    planas = [line for b in batches for line in b]
    assert sorted(planas) == list(range(1, 21))
    assert len(planas) == len(set(planas))
    assert all(sum(30 for _ in b) <= 100 for b in batches)


def test_sin_lineas_no_hay_lotes():
    assert services._line_batches(_counts([]), max_rows=100) == []


# ── _gps_scope_sql ──────────────────────────────────────────────────────────────

def test_scope_pipeline_acota_a_run_days():
    sql = services._gps_scope_sql(None, run_days=["2024-01-01", "2024-01-02"])
    assert "g.dia IN ('2024-01-01', '2024-01-02')" in sql
    assert "services_stats" not in sql  # sin anti-join sobre el gps acumulado


def test_scope_interactivo_no_se_acota_por_dia():
    """Por línea el borrado es de TODOS los días: acotar la lectura borraría historia."""
    sql = services._gps_scope_sql("10,20", run_days=["2024-01-01"])
    assert "services_stats" in sql
    assert "g.id_linea IN (10,20)" in sql
    assert "2024-01-01" not in sql


def test_batch_lines_se_compone_con_run_days():
    """El bug del parche del cliente: pasar la línea por line_ids_str perdía run_days."""
    sql = services._gps_scope_sql(None, run_days=["2024-01-01"], batch_lines=[7, 8])
    assert "g.dia IN ('2024-01-01')" in sql
    assert "g.id_linea IN (7, 8)" in sql
    assert "services_stats" not in sql


# ── proyección de columnas ──────────────────────────────────────────────────────

def _ctx_con_gps(monkeypatch, utilizar_servicios_gps):
    """ctx cuya query registra el SQL y devuelve un gps vacío."""
    monkeypatch.setattr(
        services.utils, "leer_configs_generales",
        lambda *a, **k: {"utilizar_servicios_gps": utilizar_servicios_gps, "epsg_m": 9265},
    )
    capturado = {}

    def fake_query(q):
        if "gps_exists" in q:
            return pd.DataFrame({"gps_exists": [1]})
        capturado["sql"] = q
        return pd.DataFrame()

    ctx = MagicMock()
    ctx.data.query.side_effect = fake_query
    return ctx, capturado


def test_no_se_lee_select_estrella(monkeypatch):
    ctx, cap = _ctx_con_gps(monkeypatch, True)
    services.get_stops_and_gps_data(ctx, None, run_days=["2024-01-01"])
    assert "g.*" not in cap["sql"]
    for col in services.GPS_COLS:
        assert f"g.{col}" in cap["sql"]
    # las columnas anchas que no se usan quedan afuera
    for col in ("id_original", "h3", "id_servicio", "distance_servicio_mts_agg"):
        assert f"g.{col}" not in cap["sql"]


def test_latitud_y_longitud_solo_en_el_camino_geo(monkeypatch):
    ctx, cap = _ctx_con_gps(monkeypatch, True)
    services.get_stops_and_gps_data(ctx, None, run_days=["2024-01-01"])
    assert "g.latitud" not in cap["sql"]

    ctx, cap = _ctx_con_gps(monkeypatch, False)
    services.get_stops_and_gps_data(ctx, None, run_days=["2024-01-01"])
    assert "g.latitud" in cap["sql"] and "g.longitud" in cap["sql"]


def test_el_order_by_se_conserva(monkeypatch):
    """Los cumsum por servicio dependen del orden dentro de (linea, dia, ramal, interno)."""
    ctx, cap = _ctx_con_gps(monkeypatch, True)
    services.get_stops_and_gps_data(ctx, None, run_days=["2024-01-01"])
    assert "ORDER BY g.id_linea, g.dia, g.id_ramal, g.interno, g.fecha, g.id" in cap["sql"]
