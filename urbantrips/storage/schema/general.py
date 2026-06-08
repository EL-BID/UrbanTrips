# urbantrips/storage/schema/general.py

CORRIDAS = """
CREATE TABLE IF NOT EXISTS corridas (
    corrida TEXT NOT NULL,
    process TEXT NOT NULL,
    date    TEXT NOT NULL
)
"""

ALL_TABLES = [CORRIDAS]
