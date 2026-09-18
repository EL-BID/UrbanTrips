from __future__ import annotations

import os
import tempfile
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
    tmp_dir: Path

    @property
    def configs_dir(self) -> Path:
        return self.config_file.parent


def _default_tmp_dir() -> Path:
    """Scratch dir used when the config leaves ``tmp_dir`` empty.

    The system temp dir, not something under `base`: temporary artifacts
    (DuckDB spill, parquet staging) can reach tens of GB and the project
    disk is often the small one. Set `tmp_dir` in the YAML to move them.
    """
    return Path(tempfile.gettempdir()) / "urbantrips_tmp"


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


_DIR_OVERRIDE_KEYS = ("input_dir", "db_dir", "output_dir", "tmp_dir")


def _read_dir_overrides(config_file: Path) -> dict:
    """Read the directory-override keys out of the config YAML.

    Raises FileNotFoundError when the file is absent; callers decide whether
    that is fatal.
    """
    with open(config_file, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {k: raw[k] for k in _DIR_OVERRIDE_KEYS if raw.get(k)}


def init_paths(base_dir: Path | None = None, config_file: Path | None = None) -> Paths:
    """Initialize the path singleton from base_dir.

    Reads input_dir / db_dir / output_dir / tmp_dir overrides from the config YAML
    if present.
    Raises FileNotFoundError if base_dir doesn't exist or no config is found.
    """
    global _paths

    env_config = os.environ.get("URBANTRIPS_CONFIG")
    if config_file is None and env_config:
        config_file = Path(env_config).resolve()

    if config_file is not None:
        config_file = Path(config_file).resolve()
        base = (
            config_file.parent.parent
            if config_file.parent.name == "configs"
            else config_file.parent
        )
    else:
        base = Path(base_dir).resolve() if base_dir else Path(".").resolve()
        if not base.exists():
            raise FileNotFoundError(f"base_dir does not exist: {base}")
        config_file = _find_config(base)

    base = Path(base_dir).resolve() if base_dir else base
    if not base.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base}")

    try:
        overrides = _read_dir_overrides(config_file)
    except FileNotFoundError:
        overrides = {}  # config file optional — defaults apply
    except Exception as exc:
        raise ValueError(f"Could not read config file {config_file}: {exc}") from exc

    _paths = Paths(
        base=base,
        config_file=config_file,
        input_dir=_resolve_dir(overrides.get("input_dir"), config_file, base / "data" / "data_ciudad"),
        db_dir=_resolve_dir(overrides.get("db_dir"), config_file, base / "data" / "db"),
        output_dir=_resolve_dir(overrides.get("output_dir"), config_file, base / "resultados"),
        tmp_dir=_resolve_dir(overrides.get("tmp_dir"), config_file, _default_tmp_dir()),
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


def get_tmp_dir() -> Path:
    """Return the scratch dir for every temporary artifact, creating it.

    Single entry point for DuckDB spill (``temp_directory``) and for the
    parquet staging dirs. Honours ``tmp_dir`` in the config YAML; falls back
    to the system temp dir when it is absent or blank.
    """
    tmp = get_paths().tmp_dir
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


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
    # Honour the directory overrides even on this lazy path: spawned workers
    # (Windows uses spawn, so they re-import and never call init_paths) inherit
    # URBANTRIPS_CONFIG and land here. Without this they would spill DuckDB and
    # stage parquet somewhere other than the configured tmp_dir.
    try:
        overrides = _read_dir_overrides(config_file)
    except Exception:
        overrides = {}  # config missing or unreadable — plain defaults

    return Paths(
        base=base,
        config_file=config_file,
        input_dir=_resolve_dir(
            overrides.get("input_dir"), config_file, base / "data" / "data_ciudad"
        ),
        db_dir=_resolve_dir(overrides.get("db_dir"), config_file, base / "data" / "db"),
        output_dir=_resolve_dir(
            overrides.get("output_dir"), config_file, base / "resultados"
        ),
        tmp_dir=_resolve_dir(overrides.get("tmp_dir"), config_file, _default_tmp_dir()),
    )
