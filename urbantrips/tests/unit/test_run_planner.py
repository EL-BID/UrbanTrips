"""Planner de corridas: decide qué correr según el log de progreso.

Cubre los escenarios que definió el usuario:
- crecer el yaml (dia1,dia2 completas + dia3,dia4 nuevas → solo corren las nuevas)
- corridas multi-día (semana1 = varios días)
- reprocesar explícito (fuerza desde ingest)
- recuperar un crash (resume desde el step que faltó)
"""
import pandas as pd

from urbantrips.utils.run_planner import (
    estado_corrida, planificar, scope_y_step, dias_de_corridas, STEP_ORDER,
)

TS = "2026-03-27 10:00:00"


def _log(rows):
    """rows: lista de dicts parciales; completa columnas faltantes con None."""
    cols = ["config_yaml", "alias", "corrida", "dia",
            "ingest_ts", "legs_ts", "outputs_ts", "dashboard_ts", "date"]
    full = []
    for r in rows:
        d = {c: None for c in cols}
        d.update(r)
        d["date"] = TS
        full.append(d)
    return pd.DataFrame(full, columns=cols)


def _completa(corrida, dia):
    return {"corrida": corrida, "dia": dia, "ingest_ts": TS, "legs_ts": TS,
            "outputs_ts": TS, "dashboard_ts": TS}


# ── estado_corrida ─────────────────────────────────────────────────────────────

def test_estado_nueva_sin_log():
    assert estado_corrida(pd.DataFrame(), "c")[0] == "nueva"
    assert estado_corrida(_log([_completa("otra", "2026-03-09")]), "c")[0] == "nueva"


def test_estado_completa():
    log = _log([_completa("c", "2026-03-09"), _completa("c", "2026-03-10")])
    estado, step, dias = estado_corrida(log, "c")
    assert estado == "completa"
    assert step is None
    assert dias == ["2026-03-09", "2026-03-10"]


def test_estado_incompleta_desde_primer_step_pendiente():
    # ingest+legs OK, outputs NULL → resume desde outputs
    log = _log([{"corrida": "c", "dia": "2026-03-09",
                 "ingest_ts": TS, "legs_ts": TS}])
    estado, step, dias = estado_corrida(log, "c")
    assert estado == "incompleta"
    assert step == "outputs"
    assert dias == ["2026-03-09"]


def test_estado_incompleta_si_algun_dia_falta_step():
    # un día completo y otro a medias → la corrida es incompleta en el step del peor
    log = _log([
        _completa("semana1", "2026-03-09"),
        {"corrida": "semana1", "dia": "2026-03-10", "ingest_ts": TS},
    ])
    estado, step, dias = estado_corrida(log, "semana1")
    assert estado == "incompleta"
    assert step == "legs"
    assert dias == ["2026-03-09", "2026-03-10"]


# ── planificar: crecer el yaml ────────────────────────────────────────────────

def test_crecer_yaml_solo_corren_las_nuevas():
    log = _log([_completa("dia1", "2026-03-01"), _completa("dia2", "2026-03-02")])
    plan = planificar(log, ["dia1", "dia2", "dia3", "dia4"])
    assert plan["to_ingest"] == ["dia3", "dia4"]
    assert plan["skip"] == ["dia1", "dia2"]
    assert plan["resume"] == []
    assert plan["forzadas"] == []


def test_semanas_multidia_completa_se_saltea():
    log = _log([_completa("semana1", f"2026-03-0{d}") for d in range(1, 8)])
    plan = planificar(log, ["semana1", "semana2"])
    assert plan["to_ingest"] == ["semana2"]
    assert plan["skip"] == ["semana1"]


# ── planificar: reprocesar explícito ──────────────────────────────────────────

def test_reprocesar_fuerza_corrida_completa():
    log = _log([_completa("dia1", "2026-03-01"), _completa("dia2", "2026-03-02")])
    plan = planificar(log, ["dia1", "dia2"], reprocesar=["dia1"])
    assert plan["to_ingest"] == ["dia1"]
    assert plan["forzadas"] == ["dia1"]      # existía → hay que limpiar antes
    assert plan["skip"] == ["dia2"]


def test_reprocesar_subconjunto_de_varias():
    # yaml con 4 corridas completas; --reprocesar 09 y 11 → solo esas se rehacen,
    # 10 y 12 quedan congeladas. Es el caso que preguntó el usuario.
    log = _log([
        _completa("ut_amba_20260309", "2026-03-09"),
        _completa("ut_amba_20260310", "2026-03-10"),
        _completa("ut_amba_20260311", "2026-03-11"),
        _completa("ut_amba_20260312", "2026-03-12"),
    ])
    plan = planificar(
        log,
        ["ut_amba_20260309", "ut_amba_20260310",
         "ut_amba_20260311", "ut_amba_20260312"],
        reprocesar=["ut_amba_20260309", "ut_amba_20260311"],
    )
    assert plan["to_ingest"] == ["ut_amba_20260309", "ut_amba_20260311"]
    assert plan["forzadas"] == ["ut_amba_20260309", "ut_amba_20260311"]
    assert plan["skip"] == ["ut_amba_20260310", "ut_amba_20260312"]
    assert plan["resume"] == []


def test_reprocesar_corrida_nueva_no_es_forzada():
    # pasar en --reprocesar algo que no está en el log: se ingesta, pero no
    # requiere limpieza previa (no hay data vieja)
    plan = planificar(pd.DataFrame(), ["dia1"], reprocesar=["dia1"])
    assert plan["to_ingest"] == ["dia1"]
    assert plan["forzadas"] == []


# ── planificar: recuperar crash ───────────────────────────────────────────────

def test_resume_desde_step_que_falto():
    # dia3,dia4 ingestaron+legs pero crashearon antes de outputs
    log = _log([
        {"corrida": "dia3", "dia": "2026-03-03", "ingest_ts": TS, "legs_ts": TS},
        {"corrida": "dia4", "dia": "2026-03-04", "ingest_ts": TS, "legs_ts": TS},
    ])
    plan = planificar(log, ["dia3", "dia4"])
    assert plan["to_ingest"] == []
    assert plan["skip"] == []
    corridas = [c for c, _, _ in plan["resume"]]
    assert corridas == ["dia3", "dia4"]
    assert all(step == "outputs" for _, step, _ in plan["resume"])


# ── scope_y_step ──────────────────────────────────────────────────────────────

def test_scope_step_toma_el_mas_temprano():
    # dia3 nueva ya ingestada (falta legs), dia5 incompleta (falta outputs)
    # → start = legs (el más temprano), scope = ambos días
    log = _log([
        {"corrida": "dia3", "dia": "2026-03-03", "ingest_ts": TS},
        {"corrida": "dia5", "dia": "2026-03-05", "ingest_ts": TS, "legs_ts": TS},
    ])
    start, dias = scope_y_step(log, ["dia3", "dia5"])
    assert start == "legs"
    assert dias == ["2026-03-03", "2026-03-05"]


def test_scope_vacio_si_todo_completo():
    log = _log([_completa("dia1", "2026-03-01")])
    start, dias = scope_y_step(log, ["dia1"])
    assert start is None
    assert dias == []


def test_scope_resume_puro_outputs():
    log = _log([{"corrida": "dia3", "dia": "2026-03-03",
                 "ingest_ts": TS, "legs_ts": TS}])
    start, dias = scope_y_step(log, ["dia3"])
    assert start == "outputs"
    assert dias == ["2026-03-03"]


def test_dias_de_corridas():
    log = _log([_completa("dia1", "2026-03-01"), _completa("dia2", "2026-03-02")])
    assert dias_de_corridas(log, ["dia1"]) == ["2026-03-01"]
    assert dias_de_corridas(log, ["dia1", "dia2"]) == ["2026-03-01", "2026-03-02"]
    assert dias_de_corridas(pd.DataFrame(), ["dia1"]) == []
