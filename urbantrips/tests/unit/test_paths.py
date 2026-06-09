from __future__ import annotations
import pytest
from pathlib import Path
from urbantrips.utils.paths import init_paths, get_paths, reset_paths, Paths


@pytest.fixture(autouse=True)
def clean_singleton():
    reset_paths()
    yield
    reset_paths()


def test_get_paths_fallback_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("URBANTRIPS_BASE", raising=False)
    monkeypatch.delenv("URBANTRIPS_CONFIG", raising=False)
    p = get_paths()
    assert p.base == tmp_path.resolve()
    assert p.input_dir == tmp_path.resolve() / "data" / "data_ciudad"
    assert p.db_dir == tmp_path.resolve() / "data" / "db"
    assert p.output_dir == tmp_path.resolve() / "resultados"


def test_init_paths_finds_config_at_root(tmp_path):
    config = tmp_path / "configuraciones_generales.yaml"
    config.write_text("alias_db: test\n")
    p = init_paths(tmp_path)
    assert p.config_file == config
    assert p.base == tmp_path


def test_init_paths_finds_config_in_configs_subdir(tmp_path):
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "configuraciones_generales.yaml"
    config.write_text("alias_db: test\n")
    p = init_paths(tmp_path)
    assert p.config_file == config


def test_init_paths_raises_if_base_not_exist():
    with pytest.raises(FileNotFoundError, match="base_dir does not exist"):
        init_paths(Path("/nonexistent/path/xyz"))


def test_init_paths_raises_if_config_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        init_paths(tmp_path)


def test_init_paths_uses_default_subdirs(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text("")
    p = init_paths(tmp_path)
    assert p.input_dir == tmp_path.resolve() / "data" / "data_ciudad"
    assert p.db_dir == tmp_path.resolve() / "data" / "db"
    assert p.output_dir == tmp_path.resolve() / "resultados"


def test_init_paths_relative_override(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text(
        "output_dir: ../outputs\n"
    )
    p = init_paths(tmp_path)
    assert p.output_dir == (tmp_path / "configs" / ".." / "outputs").resolve()


def test_init_paths_absolute_override(tmp_path):
    abs_out = tmp_path / "shared" / "results"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text(
        f"output_dir: {abs_out}\n"
    )
    p = init_paths(tmp_path)
    assert p.output_dir == abs_out


def test_get_paths_returns_singleton(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text("")
    p1 = init_paths(tmp_path)
    p2 = get_paths()
    assert p1 is p2


def test_get_paths_lazy_init_from_env(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "configuraciones_generales.yaml").write_text("")
    monkeypatch.setenv("URBANTRIPS_BASE", str(tmp_path))
    p = get_paths()
    assert p.base == tmp_path
