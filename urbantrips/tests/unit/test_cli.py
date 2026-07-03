import argparse
import os

from urbantrips.utils.cli import add_bootstrap_args, apply_bootstrap_env


def test_add_bootstrap_args_accepts_config_and_base_dir():
    parser = argparse.ArgumentParser()
    add_bootstrap_args(parser)

    args = parser.parse_args(["--config", "configs/x.yaml", "--base-dir", "/tmp/city_a"])

    assert args.config == "configs/x.yaml"
    assert args.base_dir == "/tmp/city_a"


def test_add_bootstrap_args_short_flags():
    parser = argparse.ArgumentParser()
    add_bootstrap_args(parser)

    args = parser.parse_args(["-c", "configs/x.yaml", "-d", "/tmp/city_b"])

    assert args.config == "configs/x.yaml"
    assert args.base_dir == "/tmp/city_b"


def test_add_bootstrap_args_defaults_to_none():
    parser = argparse.ArgumentParser()
    add_bootstrap_args(parser)

    args = parser.parse_args([])

    assert args.config is None
    assert args.base_dir is None


def test_apply_bootstrap_env_sets_both_vars(monkeypatch):
    monkeypatch.delenv("URBANTRIPS_CONFIG", raising=False)
    monkeypatch.delenv("URBANTRIPS_BASE", raising=False)

    args = argparse.Namespace(config="configs/x.yaml", base_dir="/tmp/city_a")
    apply_bootstrap_env(args)

    try:
        assert os.environ["URBANTRIPS_CONFIG"] == "configs/x.yaml"
        assert os.environ["URBANTRIPS_BASE"] == "/tmp/city_a"
    finally:
        os.environ.pop("URBANTRIPS_CONFIG", None)
        os.environ.pop("URBANTRIPS_BASE", None)


def test_apply_bootstrap_env_leaves_env_untouched_when_args_none(monkeypatch):
    monkeypatch.delenv("URBANTRIPS_CONFIG", raising=False)
    monkeypatch.delenv("URBANTRIPS_BASE", raising=False)

    args = argparse.Namespace(config=None, base_dir=None)
    apply_bootstrap_env(args)

    assert "URBANTRIPS_CONFIG" not in os.environ
    assert "URBANTRIPS_BASE" not in os.environ
