"""Day-scoping incremental del dashboard (#5+#6).

El orquestador materializa etapas_proc_mat/viajes_proc_mat acotadas a los días
de la corrida; los consumidores escriben solo esas particiones y acotan sus
lecturas de tablas acumulativas a los días del mat. Estos tests cubren:
- materializar_proc_tables(run_days=...) filtra ambas tablas.
- proc_mat_days devuelve los días presentes en el mat.
- _upsert_indicator_por_dia preserva los días congelados (distribucion /
  viajes_hora eran reemplazo TOTAL → pérdida de días viejos).
"""

from types import SimpleNamespace

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _DuckPort:
    """Port mínimo sobre una conexión DuckDB real (execute + query)."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql):
        self._conn.execute(sql)

    def query(self, sql):
        return self._conn.execute(sql).fetchdf()


def _ctx_con_dos_dias():
    """StorageContext falso: data DB con etapas/viajes/travel_times de 2 días."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE etapas AS
        SELECT * FROM (VALUES
            (1, '2026-03-09', 'T1', 1, 10, 1, 8, 'autobus', 1.0, '', 'M', 1),
            (2, '2026-03-27', 'T1', 1, 10, 1, 9, 'autobus', 1.0, '', 'F', 1)
        ) AS t(id, dia, id_tarjeta, id_viaje, id_linea, id_ramal, hora, modo,
               factor_expansion_linea, tarifa, genero, od_validado)
    """)
    conn.execute("""
        CREATE TABLE travel_times_legs AS
        SELECT * FROM (VALUES
            (1, 5.0, 20.0),
            (2, 6.0, 25.0)
        ) AS t(id, distance_od, travel_time_min)
    """)
    conn.execute("""
        CREATE TABLE viajes AS
        SELECT * FROM (VALUES
            ('2026-03-09', 'T1', 1, '08:00:00', 8, 1, 'autobus', 'M', '', 1.0, 1.0, 1),
            ('2026-03-27', 'T1', 1, '09:00:00', 9, 1, 'autobus', 'F', '', 1.0, 1.0, 1)
        ) AS t(dia, id_tarjeta, id_viaje, tiempo, hora, cant_etapas, modo,
               genero, tarifa, factor_expansion_linea, factor_expansion_tarjeta,
               od_validado)
    """)
    conn.execute("""
        CREATE TABLE travel_times_trips AS
        SELECT * FROM (VALUES
            ('2026-03-09', 'T1', 1, 5.0, 20.0),
            ('2026-03-27', 'T1', 1, 6.0, 25.0)
        ) AS t(dia, id_tarjeta, id_viaje, distance_od, travel_time_min)
    """)
    return SimpleNamespace(data=_DuckPort(conn))


# ---------------------------------------------------------------------------
# materializar_proc_tables + proc_mat_days
# ---------------------------------------------------------------------------

def test_materializar_proc_tables_day_scoped():
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, proc_mat_days, ETAPAS_PROC_MAT, VIAJES_PROC_MAT,
    )

    ctx = _ctx_con_dos_dias()
    materializar_proc_tables(ctx, replace=True, run_days=["2026-03-27"])

    etapas_dias = ctx.data.query(f"SELECT DISTINCT dia FROM {ETAPAS_PROC_MAT}")
    viajes_dias = ctx.data.query(f"SELECT DISTINCT dia FROM {VIAJES_PROC_MAT}")
    assert etapas_dias["dia"].tolist() == ["2026-03-27"]
    assert viajes_dias["dia"].tolist() == ["2026-03-27"]
    assert proc_mat_days(ctx) == ["2026-03-27"]


def test_materializar_proc_tables_sin_run_days_trae_todo():
    from urbantrips.preparo_dashboard.sql_queries import (
        materializar_proc_tables, proc_mat_days,
    )

    ctx = _ctx_con_dos_dias()
    materializar_proc_tables(ctx, replace=True)
    assert proc_mat_days(ctx) == ["2026-03-09", "2026-03-27"]


def test_dias_where_clause():
    from urbantrips.preparo_dashboard.sql_queries import dias_where_clause

    assert dias_where_clause([]) == ""
    assert dias_where_clause(None) == ""
    assert dias_where_clause(["2026-03-27"]) == " WHERE dia IN ('2026-03-27')"
    assert (
        dias_where_clause(["a'b"], prefix="AND")
        == " AND dia IN ('a''b')"
    )


# ---------------------------------------------------------------------------
# _upsert_indicator_por_dia (distribucion / viajes_hora)
# ---------------------------------------------------------------------------

def _ctx_dash():
    from urbantrips.storage.adapters.memory.adapters import InMemoryDashAdapter
    return SimpleNamespace(dash=InMemoryDashAdapter())


def test_upsert_indicator_preserva_dias_congelados():
    from urbantrips.preparo_dashboard.preparo_dashboard import _upsert_indicator_por_dia

    ctx = _ctx_dash()
    historia = pd.DataFrame({
        "Día": ["2026-03-09", "2026-03-09", "2026-03-10"],
        "Modo": ["autobus", "Todos", "autobus"],
        "Distancia (kms)": [5, 5, 7],
        "Viajes": [100, 100, 50],
    })
    ctx.dash.save_indicator(historia, "distribucion")

    nuevo = pd.DataFrame({
        "Día": ["2026-03-10", "2026-03-27"],
        "Modo": ["autobus", "autobus"],
        "Distancia (kms)": [7, 6],
        "Viajes": [55, 200],
    })
    _upsert_indicator_por_dia(ctx, nuevo, "distribucion")

    out = ctx.dash.get_indicator("distribucion")
    # días congelados intactos
    frozen = out[out["Día"] == "2026-03-09"].sort_values("Modo").reset_index(drop=True)
    assert frozen["Viajes"].tolist() == [100, 100]
    # día re-corrido reemplazado (55, no 50) y día nuevo presente
    assert out[out["Día"] == "2026-03-10"]["Viajes"].tolist() == [55]
    assert out[out["Día"] == "2026-03-27"]["Viajes"].tolist() == [200]
    assert len(out) == 4


def test_upsert_indicator_sin_historia_escribe_directo():
    from urbantrips.preparo_dashboard.preparo_dashboard import _upsert_indicator_por_dia

    ctx = _ctx_dash()
    nuevo = pd.DataFrame({
        "Día": ["2026-03-27"], "Modo": ["autobus"], "Hora": [8], "Viajes": [10],
    })
    _upsert_indicator_por_dia(ctx, nuevo, "viajes_hora")
    out = ctx.dash.get_indicator("viajes_hora")
    assert out["Viajes"].tolist() == [10]


def test_viajes_poligonos_acota_chains_a_run_days():
    """_viajes_poligonos_desde_chains lee chains_norm SOLO de los dias de la
    corrida, y ademas de a un dia por vez: la proyeccion cuesta 717 B/fila
    medidos, o sea ~72 GB si se levantara el mes entero de una."""
    from unittest.mock import MagicMock
    from urbantrips.preparo_dashboard.preparo_dashboard import (
        _viajes_poligonos_desde_chains,
    )

    ctx = MagicMock()
    ctx.insumos.query.return_value = pd.DataFrame(
        {"h3": ["abc"], "zona": ["P1"], "tipo": ["poligono"]}
    )
    run_days = ["2026-03-27", "2026-03-28"]
    ctx.data.get_run_days.return_value = pd.DataFrame({"dia": run_days})
    capturadas = []

    def fake_dash_query(sql):
        capturadas.append(sql)
        return pd.DataFrame(columns=[
            "dia", "mes", "tipo_dia", "id_tarjeta", "id_viaje",
            "h3_inicio_norm", "h3_fin_norm", "modo_agregado", "rango_hora",
            "transferencia", "distancia_agregada", "distance_od",
            "factor_expansion_linea",
        ])

    ctx.dash.query.side_effect = fake_dash_query

    _viajes_poligonos_desde_chains(ctx)

    lecturas = [s for s in capturadas if "LIMIT 0" not in s]
    # una lectura por dia de la corrida, cada una acotada a ese dia
    assert len(lecturas) == len(run_days), lecturas
    for dia, sql in zip(run_days, lecturas):
        assert "from chains_norm where dia = '{}'".format(dia) in sql.lower(), sql
    # y ninguna lectura sin filtro de dia
    assert not [s for s in lecturas if "where dia" not in s.lower()]


def test_upsert_indicator_historia_legacy_sin_dia_reemplaza_entera():
    """Tabla pre-creada por el schema con shape legacy (desc_dia/...): no hay
    columna Día que mergear — se reemplaza entera, sin romper."""
    from urbantrips.preparo_dashboard.preparo_dashboard import _upsert_indicator_por_dia

    ctx = _ctx_dash()
    legacy = pd.DataFrame({
        "desc_dia": ["hábil"], "tipo_dia": ["x"], "Distancia": [1], "Viajes": [2],
    })
    ctx.dash.save_indicator(legacy, "distribucion")

    nuevo = pd.DataFrame({
        "Día": ["2026-03-27"], "Modo": ["autobus"],
        "Distancia (kms)": [6], "Viajes": [200],
    })
    _upsert_indicator_por_dia(ctx, nuevo, "distribucion")
    out = ctx.dash.get_indicator("distribucion")
    assert out.columns.tolist() == ["Día", "Modo", "Distancia (kms)", "Viajes"]
    assert len(out) == 1
