# urbantrips/config/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Extension per backend for DB file names
_EXTENSION: dict[str, str] = {
    "duckdb": ".duckdb",
    "sqlite": ".sqlite",
    "postgresql": "",
}

_KNOWN_FIELDS = {
    "alias_db", "alias_db_insumos", "alias_db_dashboard", "corridas",
    "geolocalizar_trx", "nombre_archivo_trx", "nombre_archivo_gps",
    "nombres_variables_trx", "formato_fecha", "columna_hora",
    "tipo_trx_invalidas", "tolerancia_parada_destino", "resolucion_h3",
    "ordenamiento_transacciones", "ventana_viajes", "ventana_duplicado",
    "tiempos_viaje_estaciones", "storage_backend", "n_batches", "parallel_workers",
}

_REQUIRED_FIELDS = {
    "alias_db",
    "corridas",
    "geolocalizar_trx",
    "nombres_variables_trx",
    "formato_fecha",
    "columna_hora",
    "tipo_trx_invalidas",
    "tolerancia_parada_destino",
    "resolucion_h3",
    "ordenamiento_transacciones",
    "ventana_viajes",
    "ventana_duplicado",
    "tiempos_viaje_estaciones",
}

_VALID_BACKENDS = {"duckdb", "sqlite", "postgresql"}


@dataclass
class Config:
    alias_db: str
    alias_db_insumos: str
    alias_db_dashboard: str
    corridas: list[str]
    geolocalizar_trx: bool
    nombre_archivo_trx: str
    nombre_archivo_gps: str | None
    nombres_variables_trx: dict[str, str]
    formato_fecha: str
    columna_hora: str
    tipo_trx_invalidas: Any
    tolerancia_parada_destino: float
    resolucion_h3: int
    ordenamiento_transacciones: str
    ventana_viajes: int
    ventana_duplicado: int
    tiempos_viaje_estaciones: Any
    storage_backend: str = "duckdb"
    n_batches: int = 1
    parallel_workers: int | None = None
    # Remaining YAML fields preserved for backward compatibility
    # during domain migration (Plan 2 will type these progressively)
    raw: dict = field(default_factory=dict, repr=False)

    def db_path(self, tipo: str, alias: str | None = None, base_dir: Path | None = None) -> Path:
        """Return the database file path for the given database type.

        Parameters
        ----------
        tipo : str
            One of 'data', 'insumos', 'dash', 'general'.
        alias : str | None
            Run alias (for 'data' and 'dash'). Defaults to the first corrida.
        base_dir : Path | None
            Project root. Defaults to current working directory.
        """
        base = base_dir or Path(".")
        ext = _EXTENSION.get(self.storage_backend, ".duckdb")

        if tipo == "general":
            name = f"{self.alias_db}_general{ext}"
        elif tipo == "insumos":
            name = f"{self.alias_db_insumos}_insumos{ext}"
        elif tipo in ("data", "dash"):
            run_alias = alias or (self.corridas[0] if self.corridas else self.alias_db)
            name = f"{run_alias}_{tipo}{ext}"
        else:
            raise ValueError(f"tipo invalido: {tipo!r}")

        return base / "data" / "db" / name


def load_config(path: Path) -> Config:
    """Load configuration from a YAML file and return a typed Config."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            raw = yaml.safe_load(f) or {}

    known = {k: v for k, v in raw.items() if k in _KNOWN_FIELDS}
    extra = {k: v for k, v in raw.items() if k not in _KNOWN_FIELDS}
    known = _apply_legacy_defaults(known, raw)
    _validate_config_data(known, path)

    return Config(**known, raw=extra)


def _apply_legacy_defaults(known: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    data = known.copy()
    alias_db = data.get("alias_db")
    if alias_db:
        data.setdefault("alias_db_insumos", alias_db)
        data.setdefault("alias_db_dashboard", alias_db)
    data.setdefault("nombre_archivo_trx", "[CORRIDA]_trx.csv")
    if raw.get("usa_archivo_gps", False):
        data.setdefault("nombre_archivo_gps", "[CORRIDA]_gps.csv")
    else:
        data.setdefault("nombre_archivo_gps", None)
    return data


def _validate_config_data(data: dict[str, Any], path: Path) -> None:
    missing = sorted(field for field in _REQUIRED_FIELDS if field not in data)
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"Missing required config field(s) in {path}: {fields}")

    if not isinstance(data["corridas"], list) or not data["corridas"]:
        raise ValueError("Config field 'corridas' must be a non-empty list.")

    if not isinstance(data["nombres_variables_trx"], dict):
        raise ValueError("Config field 'nombres_variables_trx' must be a mapping.")

    backend = data.get("storage_backend", "duckdb")
    if backend not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(
            f"Config field 'storage_backend' must be one of: {valid}."
        )

    for field in ("resolucion_h3", "ventana_viajes", "ventana_duplicado"):
        if not isinstance(data[field], int):
            raise ValueError(f"Config field {field!r} must be an integer.")

    if not isinstance(data["tolerancia_parada_destino"], int | float):
        raise ValueError("Config field 'tolerancia_parada_destino' must be numeric.")

    n_batches = data.get("n_batches", 1)
    if not isinstance(n_batches, int) or n_batches < 1:
        raise ValueError("Config field 'n_batches' must be an integer greater than 0.")
