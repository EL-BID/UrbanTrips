# urbantrips/tests/unit/test_config.py
from pathlib import Path
import tempfile
import os

import pytest
import yaml

FIXTURE_CONFIG = Path(__file__).parent.parent / "fixtures" / "configs" / "test_config.yaml"


def test_load_config_returns_config_object():
    from urbantrips.config.config import load_config, Config
    cfg = load_config(FIXTURE_CONFIG)
    assert isinstance(cfg, Config)


def test_config_fields():
    from urbantrips.config.config import load_config
    cfg = load_config(FIXTURE_CONFIG)
    assert cfg.alias_db == "test_ciudad"
    assert cfg.storage_backend == "duckdb"
    assert cfg.n_batches == 2
    assert cfg.corridas == ["corrida_01"]


def test_config_defaults_storage_backend_to_duckdb():
    """Configs without storage_backend default to duckdb."""
    from urbantrips.config.config import load_config

    minimal = {
        "alias_db": "ciudad",
        "corridas": ["r1"],
        "geolocalizar_trx": False,
        "nombre_archivo_trx": "f.csv",
        "nombre_archivo_gps": None,
        "nombres_variables_trx": {},
        "formato_fecha": "%Y-%m-%d",
        "columna_hora": "tiempo",
        "tipo_trx_invalidas": None,
        "tolerancia_parada_destino": 300,
        "resolucion_h3": 8,
        "ordenamiento_transacciones": "tiempo",
        "ventana_viajes": 90,
        "ventana_duplicado": 5,
        "tiempos_viaje_estaciones": None,
        "alias_db_insumos": "ciudad",
        "alias_db_dashboard": "ciudad",
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(minimal, f)
        path = f.name
    try:
        cfg = load_config(Path(path))
        assert cfg.storage_backend == "duckdb"
        assert cfg.n_batches == 1
    finally:
        os.unlink(path)


def test_config_db_path_returns_path_object():
    from urbantrips.config.config import load_config
    cfg = load_config(FIXTURE_CONFIG)
    p = cfg.db_path("data", base_dir=Path("/tmp/test"))
    assert p == Path("/tmp/test/data/db/corrida_01_data.duckdb")


def test_config_defaults_legacy_aliases_and_input_filenames():
    from urbantrips.config.config import load_config

    minimal = _minimal_config()
    minimal.pop("alias_db_insumos")
    minimal.pop("alias_db_dashboard")
    minimal.pop("nombre_archivo_trx")
    minimal.pop("nombre_archivo_gps")

    cfg = _load_from_temp_yaml(minimal)

    assert cfg.alias_db_insumos == "ciudad"
    assert cfg.alias_db_dashboard == "ciudad"
    assert cfg.nombre_archivo_trx == "[CORRIDA]_trx.csv"
    assert cfg.nombre_archivo_gps is None


def test_config_validation_reports_missing_required_field():
    from urbantrips.config.config import load_config

    minimal = _minimal_config()
    minimal.pop("corridas")
    path = _write_temp_yaml(minimal)
    try:
        with pytest.raises(ValueError, match="Missing required config field"):
            load_config(path)
    finally:
        os.unlink(path)


def test_config_validation_rejects_invalid_backend():
    from urbantrips.config.config import load_config

    minimal = _minimal_config()
    minimal["storage_backend"] = "not-a-backend"
    path = _write_temp_yaml(minimal)
    try:
        with pytest.raises(ValueError, match="storage_backend"):
            load_config(path)
    finally:
        os.unlink(path)


def _minimal_config():
    return {
        "alias_db": "ciudad",
        "alias_db_insumos": "ciudad",
        "alias_db_dashboard": "ciudad",
        "corridas": ["r1"],
        "geolocalizar_trx": False,
        "nombre_archivo_trx": "f.csv",
        "nombre_archivo_gps": None,
        "nombres_variables_trx": {},
        "formato_fecha": "%Y-%m-%d",
        "columna_hora": "tiempo",
        "tipo_trx_invalidas": None,
        "tolerancia_parada_destino": 300,
        "resolucion_h3": 8,
        "ordenamiento_transacciones": "tiempo",
        "ventana_viajes": 90,
        "ventana_duplicado": 5,
        "tiempos_viaje_estaciones": None,
    }


def _write_temp_yaml(data):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        return Path(f.name)


def _load_from_temp_yaml(data):
    path = _write_temp_yaml(data)
    try:
        from urbantrips.config.config import load_config

        return load_config(path)
    finally:
        os.unlink(path)


def test_path_overrides_are_optional():
    """input_dir / db_dir / output_dir default to None when absent."""
    from urbantrips.config.config import load_config

    minimal = _minimal_config()
    cfg = _load_from_temp_yaml(minimal)
    assert cfg.input_dir is None
    assert cfg.db_dir is None
    assert cfg.output_dir is None


def test_path_overrides_are_loaded():
    """Explicit path override keys are parsed and stored."""
    from urbantrips.config.config import load_config

    minimal = _minimal_config()
    minimal["input_dir"] = "/my/input"
    minimal["db_dir"] = "custom/db"
    minimal["output_dir"] = "/out"

    cfg = _load_from_temp_yaml(minimal)
    assert cfg.input_dir == "/my/input"
    assert cfg.db_dir == "custom/db"
    assert cfg.output_dir == "/out"
