from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_paths: "Paths | None" = None


@dataclass
class Paths:
    base: Path
    config_file: Path
    input_dir: Path
    db_dir: Path
    output_dir: Path

    @property
    def configs_dir(self) -> Path:
        return self.config_file.parent


def _find_config(base: Path) -> Path:
    candidates = [
        base / "configuraciones_generales.yaml",
        base / "configs" / "configuraciones_generales.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Config file not found in {base}. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def _resolve_dir(value: str | None, config_file: Path, default: Path) -> Path:
    if value is None:
        return default
    p = Path(value)
    return p if p.is_absolute() else (config_file.parent / p).resolve()


def init_paths(base_dir: Path | None = None) -> Paths:
    """Initialize the path singleton from base_dir.

    Reads input_dir / db_dir / output_dir overrides from the config YAML if present.
    Raises FileNotFoundError if base_dir doesn't exist or no config is found.
    """
    global _paths
    base = Path(base_dir).resolve() if base_dir else Path(".").resolve()
    if not base.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base}")

    config_file = _find_config(base)

    overrides: dict = {}
    try:
        with open(config_file, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for key in ("input_dir", "db_dir", "output_dir"):
            if key in raw and raw[key]:
                overrides[key] = raw[key]
    except Exception:
        pass

    _paths = Paths(
        base=base,
        config_file=config_file,
        input_dir=_resolve_dir(overrides.get("input_dir"), config_file, base / "data" / "data_ciudad"),
        db_dir=_resolve_dir(overrides.get("db_dir"), config_file, base / "data" / "db"),
        output_dir=_resolve_dir(overrides.get("output_dir"), config_file, base / "resultados"),
    )
    return _paths


def get_paths() -> Paths:
    """Return the path singleton, lazily initializing from URBANTRIPS_BASE or CWD."""
    global _paths
    if _paths is None:
        env_base = os.environ.get("URBANTRIPS_BASE")
        if env_base:
            init_paths(Path(env_base))
        else:
            _paths = _default_paths()
    return _paths


def reset_paths() -> None:
    """Reset the singleton. For use in tests only."""
    global _paths
    _paths = None


def _default_paths() -> Paths:
    """CWD-relative defaults, respecting URBANTRIPS_CONFIG env var."""
    env_config = os.environ.get("URBANTRIPS_CONFIG")
    if env_config:
        config_file = Path(env_config).resolve()
        base = (
            config_file.parent.parent
            if config_file.parent.name == "configs"
            else config_file.parent
        )
    else:
        base = Path(".").resolve()
        config_file = base / "configs" / "configuraciones_generales.yaml"
    return Paths(
        base=base,
        config_file=config_file,
        input_dir=base / "data" / "data_ciudad",
        db_dir=base / "data" / "db",
        output_dir=base / "resultados",
    )
