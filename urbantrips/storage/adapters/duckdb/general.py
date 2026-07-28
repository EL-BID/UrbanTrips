# urbantrips/storage/adapters/duckdb/general.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from urbantrips.storage.identifiers import validate_table_name
from urbantrips.storage.schema import general as schema


class DuckDBGeneralAdapter:
    """Implements GeneralPort using DuckDB."""

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._path = Path(db_path)
        self._read_only = read_only
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path), read_only=self._read_only)
        if not read_only:
            self._migrate_corridas_if_legacy()
            self._apply_schema()

    def close(self) -> None:
        # getattr: si duckdb.connect falla en __init__ (base tomada por otro
        # proceso) el atributo no existe y __del__ no tiene que romper.
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def _apply_schema(self) -> None:
        for ddl in schema.ALL_TABLES:
            self._conn.execute(ddl)

    def _migrate_corridas_if_legacy(self) -> None:
        """El `corridas` viejo era (corrida, process, date): una fila por corrida
        marcada tras el ingest. El nuevo es long por (alias, corrida, dia) con un
        ts por step. Si detecto el esquema viejo, migro: cada corrida vieja se
        siembra como COMPLETA (los 4 ts seteados) con dia=NULL. Así el skip por
        nombre de corrida sigue andando; un --reprocesar de una de ellas re-ingesta
        desde el CSV y reaprende los días (no necesita el mapeo viejo)."""
        try:
            # PRAGMA table_info devuelve (cid, name, type, ...): el nombre es r[1]
            cols = [
                r[1]
                for r in self._conn.execute("PRAGMA table_info('corridas')").fetchall()
            ]
        except duckdb.CatalogException:
            return  # no existe todavía; _apply_schema la crea nueva
        if not cols or "process" not in cols:
            return  # ya es el esquema nuevo (o vacío)

        legacy = self._conn.execute(
            "SELECT DISTINCT corrida, date FROM corridas"
        ).fetchdf()
        self._conn.execute("DROP TABLE corridas")
        self._apply_schema()
        if len(legacy) == 0:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = pd.DataFrame({
            "config_yaml": None,
            "alias": None,
            "corrida": legacy["corrida"].astype(str),
            "dia": None,
            "ingest_ts": ts,
            "legs_ts": ts,
            "outputs_ts": ts,
            "dashboard_ts": ts,
            "date": legacy["date"].astype(str),
        })
        self._conn.register("_legacy", rows)
        try:
            self._conn.execute("INSERT INTO corridas SELECT * FROM _legacy")
        finally:
            self._conn.unregister("_legacy")

    # ── run log (progreso por corrida/día) ────────────────────────────────────

    def get_run_log(self) -> pd.DataFrame:
        try:
            return self._conn.execute("SELECT * FROM corridas").fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame()

    def register_step(
        self,
        alias: str,
        corrida: str,
        dias: list[str],
        step: str,
        config_yaml: str | None = None,
    ) -> None:
        """Marca `step` como terminado para (alias, corrida) en cada día de `dias`.

        Si el día ya tiene fila (steps previos), la ACTUALIZA (setea el ts del
        step). Si no existe (típicamente el ingest, que estrena las filas), la
        crea. `dias` vacío es no-op."""
        ts_col = schema.STEP_TS_COLUMN.get(step)
        if ts_col is None:
            raise ValueError(f"step desconocido: {step!r}")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Al aparecer los días reales de una corrida, purgar su placeholder
        # dia=NULL (sembrado por la migración legacy) — ya no representa nada.
        if any(d is not None for d in dias):
            self._conn.execute(
                "DELETE FROM corridas WHERE corrida = ? AND dia IS NULL", [corrida]
            )
        for dia in dias:
            existe = self._conn.execute(
                "SELECT 1 FROM corridas WHERE alias IS NOT DISTINCT FROM ? "
                "AND corrida = ? AND dia IS NOT DISTINCT FROM ? LIMIT 1",
                [alias, corrida, dia],
            ).fetchone()
            if existe:
                self._conn.execute(
                    f"UPDATE corridas SET {ts_col} = ?, date = ? "
                    "WHERE alias IS NOT DISTINCT FROM ? AND corrida = ? "
                    "AND dia IS NOT DISTINCT FROM ?",
                    [now, now, alias, corrida, dia],
                )
            else:
                vals = {c: None for c in schema.STEP_COLUMNS}
                vals[ts_col] = now
                self._conn.execute(
                    "INSERT INTO corridas (config_yaml, alias, corrida, dia, "
                    "ingest_ts, legs_ts, outputs_ts, dashboard_ts, date) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [config_yaml, alias, corrida, dia,
                     vals["ingest_ts"], vals["legs_ts"], vals["outputs_ts"],
                     vals["dashboard_ts"], now],
                )

    def save_config_snapshot(
        self, alias: str | None, corrida: str, archivo: str | None, contenido: str
    ) -> None:
        """Guarda el yaml que produjo esta corrida dentro de la propia base.

        Deja la base auto-descriptiva: con el alias alcanza para saber con qué
        config se generaron los datos, sin depender de que el archivo siga
        existiendo (ni sin editar) en `configs/`. Lo consume el selector de
        corridas del dashboard.

        Es un upsert por (alias, corrida): re-correr la misma corrida actualiza
        la copia en vez de acumular filas.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            "DELETE FROM config_snapshot "
            "WHERE alias IS NOT DISTINCT FROM ? AND corrida = ?",
            [alias, corrida],
        )
        self._conn.execute(
            "INSERT INTO config_snapshot (alias, corrida, archivo, contenido, date) "
            "VALUES (?, ?, ?, ?, ?)",
            [alias, corrida, archivo, contenido, now],
        )

    def get_config_snapshot(self) -> pd.DataFrame:
        """Snapshots de config guardados, el más reciente primero.

        Devuelve vacío (no rompe) si la tabla no existe: las bases anteriores a
        esta feature no la tienen y sólo se crea al abrir en escritura.
        """
        try:
            return self._conn.execute(
                "SELECT alias, corrida, archivo, contenido, date "
                "FROM config_snapshot ORDER BY date DESC"
            ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(
                columns=["alias", "corrida", "archivo", "contenido", "date"]
            )

    def delete_corrida_log(self, alias: str, corridas: list[str]) -> None:
        """Borra del log TODAS las filas de esas corridas — usado al reprocesar
        para que se re-registren desde cero. Match por nombre de corrida (el
        general DB es por-alias; las filas migradas legacy tienen alias NULL, así
        que filtrar por alias las dejaría stale)."""
        for corrida in corridas:
            self._conn.execute("DELETE FROM corridas WHERE corrida = ?", [corrida])

    def clear_runs(self) -> None:
        try:
            self._conn.execute("DELETE FROM corridas")
        except duckdb.CatalogException:
            pass

    # ── compat legacy (write_transactions_to_db / tests) ──────────────────────
    # El path monolítico procesar_transacciones marca la corrida entera con un
    # solo evento. Se mapea a "los 4 steps completos, dia=NULL" en el log nuevo.

    def get_completed_runs(self) -> pd.DataFrame:
        return self.get_run_log()

    def register_run(self, alias: str, process: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            "INSERT INTO corridas (config_yaml, alias, corrida, dia, "
            "ingest_ts, legs_ts, outputs_ts, dashboard_ts, date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [None, None, alias, None, now, now, now, now, now],
        )

    def run_exists(self, alias: str) -> bool:
        runs = self.get_run_log()
        if runs.empty or "corrida" not in runs.columns:
            return False
        return alias in runs["corrida"].values

    # ── genéricos ─────────────────────────────────────────────────────────────

    def execute(self, sql: str) -> None:
        self._conn.execute(sql)

    def query(self, sql: str) -> pd.DataFrame:
        try:
            return self._conn.execute(sql).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame()

    def append_raw(self, df: pd.DataFrame, table_name: str) -> None:
        table_name = validate_table_name(table_name)
        self._conn.register("_raw_df", df)
        try:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} AS "
                f"SELECT * FROM _raw_df WHERE FALSE"
            )
            self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM _raw_df")
        finally:
            self._conn.unregister("_raw_df")

    def get_raw(self, table_name: str) -> pd.DataFrame:
        table_name = validate_table_name(table_name)
        try:
            return self._conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except Exception:
            return pd.DataFrame()
