"""Decisiones de orquestación (run_all/_marcar_step) con adapters en memoria.

No ingesta CSVs reales: siembra el log + dias_ultima_corrida y verifica que
run_all arranque en el step correcto, sea no-op cuando no hay pendientes, y que
_marcar_step registre el progreso agrupando por corrida.
"""
import pandas as pd
import pytest

from urbantrips.storage.adapters.memory.adapters import InMemoryGeneralAdapter


class _FakeData:
    """DataPort mínimo para las decisiones de run_all/_marcar_step."""
    def __init__(self, run_days=None):
        self._run_days = pd.DataFrame({"dia": run_days or []})

    def get_run_days(self):
        return self._run_days

    def save_run_days(self, df):
        self._run_days = df.copy()


class _Ctx:
    def __init__(self, data, general):
        self.data = data
        self.general = general


TS = "2026-03-27 10:00:00"


def _row(corrida, dia, **ts):
    d = {"config_yaml": "cfg.yaml", "alias": "al", "corrida": corrida, "dia": dia,
         "ingest_ts": None, "legs_ts": None, "outputs_ts": None,
         "dashboard_ts": None, "date": TS}
    d.update(ts)
    return d


# ── _marcar_step: agrupa por corrida usando el log ────────────────────────────

def test_marcar_step_agrupa_por_corrida(monkeypatch):
    from urbantrips.utils import run_process

    gen = InMemoryGeneralAdapter()
    gen.register_step("al", "semana1", ["2026-03-09", "2026-03-10"], "ingest")
    gen.register_step("al", "semana2", ["2026-03-16"], "ingest")
    ctx = _Ctx(_FakeData(["2026-03-09", "2026-03-10", "2026-03-16"]), gen)

    monkeypatch.setattr(run_process, "_alias_actual", lambda: "al")
    monkeypatch.setattr(run_process, "_config_yaml_name", lambda: "cfg.yaml")

    run_process._marcar_step(ctx, "legs")

    log = gen.get_run_log()
    # legs marcado en los 3 días, respetando su corrida
    assert log[log.dia == "2026-03-09"].iloc[0]["legs_ts"] is not None
    assert log[log.dia == "2026-03-16"].iloc[0]["legs_ts"] is not None
    assert set(log.corrida) == {"semana1", "semana2"}
    assert len(log) == 3  # no creó filas nuevas


def test_marcar_step_run_days_vacio_es_noop(monkeypatch):
    from urbantrips.utils import run_process
    gen = InMemoryGeneralAdapter()
    ctx = _Ctx(_FakeData([]), gen)
    monkeypatch.setattr(run_process, "_alias_actual", lambda: "al")
    run_process._marcar_step(ctx, "legs")
    assert len(gen.get_run_log()) == 0


# ── run_all: no-op con gracia + arranque desde el step correcto ───────────────

def _patch_run_all(monkeypatch, gen, run_days_iniciales, config_corridas):
    """Parcha run_all para no ingestar ni construir _build_ctx; captura qué steps
    corren. run_ingest se vuelve no-op (el ingest real no aplica en este test)."""
    from urbantrips.utils import run_process

    data = _FakeData(run_days_iniciales)
    ctx = _Ctx(data, gen)
    corridos = []

    monkeypatch.setattr(run_process, "_build_ctx", lambda: ctx)
    monkeypatch.setattr(run_process, "borrar_corridas", lambda *a, **k: None)
    monkeypatch.setattr(run_process, "_config_corridas", lambda: config_corridas)
    monkeypatch.setattr(run_process, "run_ingest", lambda c, reprocesar=None: None)
    monkeypatch.setattr(run_process, "run_legs", lambda c: corridos.append("legs"))
    monkeypatch.setattr(run_process, "run_outputs", lambda c: corridos.append("outputs"))
    monkeypatch.setattr(run_process, "run_dashboard", lambda c: corridos.append("dashboard"))
    return ctx, corridos


def test_run_all_noop_si_todo_completo(monkeypatch):
    from urbantrips.utils import run_process
    gen = InMemoryGeneralAdapter()
    gen._log.append(_row("dia1", "2026-03-01", ingest_ts=TS, legs_ts=TS,
                         outputs_ts=TS, dashboard_ts=TS))
    _, corridos = _patch_run_all(monkeypatch, gen, [], ["dia1"])

    run_process.run_all()
    assert corridos == []  # nada que hacer


def test_run_all_dia_nuevo_corre_desde_legs(monkeypatch):
    from urbantrips.utils import run_process
    gen = InMemoryGeneralAdapter()
    # simula que run_ingest ya dejó dia2 con ingest_ts (nuevo)
    gen._log.append(_row("dia2", "2026-03-02", ingest_ts=TS))
    _, corridos = _patch_run_all(monkeypatch, gen, [], ["dia2"])

    run_process.run_all()
    assert corridos == ["legs", "outputs", "dashboard"]


def test_run_all_resume_desde_outputs(monkeypatch):
    from urbantrips.utils import run_process
    gen = InMemoryGeneralAdapter()
    # dia3 crasheó tras legs → resume desde outputs (legs NO se re-corre)
    gen._log.append(_row("dia3", "2026-03-03", ingest_ts=TS, legs_ts=TS))
    _, corridos = _patch_run_all(monkeypatch, gen, [], ["dia3"])

    run_process.run_all()
    assert corridos == ["outputs", "dashboard"]


def test_run_all_resume_solo_dashboard(monkeypatch):
    from urbantrips.utils import run_process
    gen = InMemoryGeneralAdapter()
    gen._log.append(_row("dia3", "2026-03-03", ingest_ts=TS, legs_ts=TS, outputs_ts=TS))
    _, corridos = _patch_run_all(monkeypatch, gen, [], ["dia3"])

    run_process.run_all()
    assert corridos == ["dashboard"]


def test_run_all_no_dashboard_flag(monkeypatch):
    from urbantrips.utils import run_process
    gen = InMemoryGeneralAdapter()
    gen._log.append(_row("dia2", "2026-03-02", ingest_ts=TS))
    _, corridos = _patch_run_all(monkeypatch, gen, [], ["dia2"])

    run_process.run_all(crear_dashboard=False)
    assert corridos == ["legs", "outputs"]  # dashboard omitido
