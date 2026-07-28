"""Log de progreso por corrida/día en el general DB (register_step, migración).

Cubre el adapter DuckDB (con migración legacy) y el in-memory en paralelo.
"""
import duckdb
import pandas as pd
import pytest

from urbantrips.storage.adapters.memory.adapters import InMemoryGeneralAdapter


def _duck_adapter(tmp_path):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    return DuckDBGeneralAdapter(tmp_path / "g.duckdb")


@pytest.fixture(params=["duck", "memory"])
def gen(request, tmp_path):
    if request.param == "duck":
        a = _duck_adapter(tmp_path)
        yield a
        a.close()
    else:
        yield InMemoryGeneralAdapter()


# ── register_step: crea filas en ingest, actualiza en steps siguientes ─────────

def test_register_step_crea_y_actualiza(gen):
    gen.register_step("miAlias", "semana1", ["2026-03-09", "2026-03-10"],
                      "ingest", config_yaml="cfg.yaml")
    log = gen.get_run_log()
    assert len(log) == 2
    assert set(log["dia"]) == {"2026-03-09", "2026-03-10"}
    assert log["ingest_ts"].notna().all()
    assert log["legs_ts"].isna().all()
    assert set(log["corrida"]) == {"semana1"}
    assert set(log["config_yaml"]) == {"cfg.yaml"}

    # el step siguiente actualiza las MISMAS filas, no crea nuevas
    gen.register_step("miAlias", "semana1", ["2026-03-09", "2026-03-10"], "legs")
    log = gen.get_run_log()
    assert len(log) == 2
    assert log["legs_ts"].notna().all()


def test_register_step_dias_vacio_es_noop(gen):
    gen.register_step("a", "c", [], "ingest")
    assert len(gen.get_run_log()) == 0


def test_register_step_step_desconocido(gen):
    with pytest.raises(ValueError):
        gen.register_step("a", "c", ["2026-03-09"], "nonexistent")


def test_delete_corrida_log(gen):
    gen.register_step("a", "c1", ["2026-03-09"], "ingest")
    gen.register_step("a", "c2", ["2026-03-10"], "ingest")
    gen.delete_corrida_log("a", ["c1"])
    log = gen.get_run_log()
    assert set(log["corrida"]) == {"c2"}


def test_register_step_purga_placeholder_legacy(gen):
    # simula una corrida migrada legacy: fila completa con dia=NULL (via register_run)
    gen.register_run("semana1", "transactions_completed")
    log0 = gen.get_run_log()
    assert log0["dia"].isna().all()  # placeholder sin día real

    # al ingestar días reales de esa corrida, el placeholder dia=NULL se purga
    gen.register_step("al", "semana1", ["2026-03-09", "2026-03-10"], "ingest")
    log = gen.get_run_log()
    assert set(log["dia"].dropna()) == {"2026-03-09", "2026-03-10"}
    assert not log["dia"].isna().any()  # no quedó el placeholder


def test_delete_corrida_log_borra_legacy_alias_null(gen):
    # fila legacy (alias NULL) + fila nueva de la misma corrida → delete borra ambas
    gen.register_run("semana1", "transactions_completed")   # alias NULL, dia NULL
    gen.register_step("al", "otra", ["2026-03-09"], "ingest")
    gen.delete_corrida_log("al", ["semana1"])
    assert set(gen.get_run_log()["corrida"]) == {"otra"}


def test_clear_runs(gen):
    gen.register_step("a", "c", ["2026-03-09"], "ingest")
    gen.clear_runs()
    assert len(gen.get_run_log()) == 0


def test_compat_register_run_marca_completa(gen):
    gen.register_run("run01", "transactions_completed")
    log = gen.get_run_log()
    assert gen.run_exists("run01")
    row = log[log["corrida"] == "run01"].iloc[0]
    for c in ("ingest_ts", "legs_ts", "outputs_ts", "dashboard_ts"):
        assert pd.notna(row[c])


# ── migración del formato viejo (solo DuckDB) ──────────────────────────────────

def test_migracion_legacy_a_long(tmp_path):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter

    # fabricar una base con el esquema VIEJO (corrida, process, date)
    p = tmp_path / "legacy.duckdb"
    con = duckdb.connect(str(p))
    con.execute("CREATE TABLE corridas (corrida TEXT, process TEXT, date TEXT)")
    con.execute(
        "INSERT INTO corridas VALUES "
        "('run01','transactions_completed','2026-01-01 10:00:00'),"
        "('run02','transactions_completed','2026-01-02 10:00:00')"
    )
    con.close()

    # abrir con el adapter nuevo → migra
    a = DuckDBGeneralAdapter(p)
    try:
        cols = [r[1] for r in a._conn.execute("PRAGMA table_info('corridas')").fetchall()]
        assert "process" not in cols
        assert "ingest_ts" in cols and "dashboard_ts" in cols
        log = a.get_run_log()
        assert set(log["corrida"]) == {"run01", "run02"}
        # sembradas como COMPLETAS (los 4 ts), dia NULL
        assert log["ingest_ts"].notna().all()
        assert log["dashboard_ts"].notna().all()
        assert log["dia"].isna().all()
        assert a.run_exists("run01")
    finally:
        a.close()


def test_esquema_nuevo_no_se_re_migra(tmp_path):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    p = tmp_path / "new.duckdb"
    a = DuckDBGeneralAdapter(p)
    a.register_step("al", "c", ["2026-03-09"], "ingest")
    a.close()
    # reabrir: no debe romper ni borrar lo existente
    a2 = DuckDBGeneralAdapter(p)
    try:
        assert len(a2.get_run_log()) == 1
    finally:
        a2.close()
