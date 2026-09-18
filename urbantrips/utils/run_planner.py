"""Decide, a partir del log de progreso (corridas), qué correr en esta invocación.

Funciones puras sobre un DataFrame de log (formato long: una fila por
(corrida, dia) con un ts por step). No tocan storage ni ctx → testeables solas.

Reglas (acordadas con el usuario):
- Corrida del config SIN filas en el log → NUEVA → se corre completa desde ingest.
- Corrida en el log con algún step sin terminar → INCOMPLETA → se resume desde el
  primer step sin terminar (recuperación de un crash).
- Corrida con los 4 steps en todas sus filas → COMPLETA → se saltea, SALVO que
  venga en --reprocesar, que la fuerza desde ingest (borra y regenera sus días).

El general DB es por-alias (un archivo {alias}_general.duckdb), así que el log
contiene un solo alias y el match es por nombre de corrida.
"""
from __future__ import annotations

STEP_ORDER = ["ingest", "legs", "outputs", "dashboard"]
_STEP_TS = {
    "ingest": "ingest_ts", "legs": "legs_ts",
    "outputs": "outputs_ts", "dashboard": "dashboard_ts",
}


def estado_corrida(log, corrida):
    """('nueva'|'incompleta'|'completa', primer_step_pendiente|None, [dias]).

    'completa' = TODAS las filas de la corrida tienen los 4 ts. 'incompleta' =
    alguna fila tiene algún ts NULL; primer_step = el más temprano pendiente.
    'nueva' = sin filas.
    """
    if log is None or len(log) == 0 or "corrida" not in log.columns:
        return ("nueva", None, [])
    rows = log[log["corrida"].astype(str) == str(corrida)]
    if len(rows) == 0:
        return ("nueva", None, [])
    dias = sorted(d for d in rows["dia"].dropna().astype(str).unique())
    pendientes = [s for s in STEP_ORDER if rows[_STEP_TS[s]].isna().any()]
    if not pendientes:
        return ("completa", None, dias)
    return ("incompleta", pendientes[0], dias)


def planificar(log, config_corridas, reprocesar=None):
    """Clasifica las corridas del config en:
      to_ingest: [corrida]              — nuevas + forzadas (re-leen CSV desde cero)
      resume:    [(corrida, step, dias)] — incompletas, se terminan desde `step`
      skip:      [corrida]              — completas y no forzadas
      forzadas:  [corrida]              — subconjunto de to_ingest que YA existía
                                          (hay que limpiar su data+log antes)
    """
    reprocesar = set(reprocesar or [])
    to_ingest, resume, skip, forzadas = [], [], [], []
    for c in config_corridas:
        estado, step, dias = estado_corrida(log, c)
        if c in reprocesar:
            to_ingest.append(c)
            if estado != "nueva":
                forzadas.append(c)
            continue
        if estado == "nueva":
            to_ingest.append(c)
        elif estado == "incompleta":
            resume.append((c, step, dias))
        else:
            skip.append(c)
    return {"to_ingest": to_ingest, "resume": resume, "skip": skip,
            "forzadas": forzadas}


def scope_y_step(log, corridas):
    """Dado el log YA actualizado por el ingest, devuelve (start_step, [dias]):
    el step más temprano a correr y la unión de días de las corridas que no están
    completas. Si no hay nada pendiente → (None, [])."""
    dias, steps = set(), set()
    for c in corridas:
        estado, step, ds = estado_corrida(log, c)
        if estado == "completa":
            continue
        dias.update(ds)
        if step is not None:
            steps.add(step)
    if not steps:
        return (None, sorted(dias))
    start = min(steps, key=STEP_ORDER.index)
    return (start, sorted(dias))


def dias_de_corridas(log, corridas):
    """Días conocidos (del log) de las corridas dadas — para limpiar antes de un
    reproceso forzado."""
    if log is None or len(log) == 0 or "corrida" not in log.columns:
        return []
    sel = log[log["corrida"].astype(str).isin({str(c) for c in corridas})]
    return sorted(d for d in sel["dia"].dropna().astype(str).unique())
