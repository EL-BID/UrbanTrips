# urbantrips/storage/schema/general.py

# `corridas` es el LOG de progreso por (alias, corrida, dia). Formato long: una
# fila por día procesado, con un timestamp por step (NULL = ese step no terminó).
# Permite (a) saltear corridas ya completas, (b) resumir una corrida que crasheó
# desde el primer step sin terminar, (c) reprocesar días puntuales borrándolos y
# regenerándolos. `corrida` puede abarcar varios días (semana1 → 7 filas).
STEP_COLUMNS = ["ingest_ts", "legs_ts", "outputs_ts", "dashboard_ts"]
# nombre de step (el que usa --step / _STEP_ORDER) → columna de timestamp del log
STEP_TS_COLUMN = {
    "ingest": "ingest_ts",
    "legs": "legs_ts",
    "outputs": "outputs_ts",
    "dashboard": "dashboard_ts",
}

CORRIDAS = """
CREATE TABLE IF NOT EXISTS corridas (
    config_yaml  TEXT,
    alias        TEXT,
    corrida      TEXT NOT NULL,
    dia          TEXT,
    ingest_ts    TEXT,
    legs_ts      TEXT,
    outputs_ts   TEXT,
    dashboard_ts TEXT,
    date         TEXT NOT NULL
)
"""

ALL_TABLES = [CORRIDAS]
