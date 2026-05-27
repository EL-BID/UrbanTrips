import sqlite3
import pytest

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transacciones (
    id INT NOT NULL,
    fecha INT NOT NULL,
    id_original TEXT,
    id_tarjeta TEXT,
    dia TEXT,
    tiempo TEXT,
    hora INT,
    modo TEXT,
    id_linea BIGINT,
    id_ramal BIGINT,
    interno INT,
    orden_trx INT,
    genero TEXT,
    tarifa TEXT,
    latitud REAL,
    longitud REAL,
    factor_expansion REAL
);
CREATE TABLE IF NOT EXISTS dias_ultima_corrida (
    dia TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS etapas (
    id INT PRIMARY KEY NOT NULL,
    id_tarjeta TEXT,
    dia TEXT,
    id_viaje INT,
    id_etapa INT,
    tiempo TEXT,
    hora INT,
    modo TEXT,
    id_linea BIGINT,
    id_ramal BIGINT,
    interno INT,
    genero TEXT,
    tarifa TEXT,
    latitud REAL,
    longitud REAL,
    h3_o TEXT,
    h3_d TEXT,
    factor_expansion REAL
);
CREATE TABLE IF NOT EXISTS indicadores (
    dia TEXT,
    detalle TEXT,
    indicador REAL,
    tabla TEXT,
    nivel INT
);
"""


class _PersistentConn:
    """Wraps sqlite3.Connection. close() is a no-op so in-memory data survives
    calls from production code that close the connection after each use.

    Migration path: replace the lambda in patched_db with a real backend
    connection factory — no other test changes needed."""

    def __init__(self, conn):
        self._conn = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture
def db_conn():
    """In-memory SQLite connection with the full schema. Each test gets a fresh DB."""
    conn = sqlite3.connect(":memory:")
    for stmt in _SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    yield _PersistentConn(conn)
    conn.close()


@pytest.fixture
def patched_db(db_conn, monkeypatch):
    """Returns db_conn AND patches iniciar_conexion_db in all relevant modules
    so every production call to that function returns the same in-memory DB.

    Migration path: replace the lambda with a real backend factory when the
    DB abstraction layer is built."""
    conn_factory = lambda tipo="data", alias_db="": db_conn
    monkeypatch.setattr("urbantrips.utils.utils.iniciar_conexion_db", conn_factory)
    monkeypatch.setattr(
        "urbantrips.datamodel.legs.iniciar_conexion_db",
        conn_factory,
        raising=False,
    )
    monkeypatch.setattr(
        "urbantrips.datamodel.transactions.iniciar_conexion_db",
        conn_factory,
        raising=False,
    )
    monkeypatch.setattr(
        "urbantrips.destinations.destinations.iniciar_conexion_db",
        conn_factory,
        raising=False,
    )
    return db_conn
