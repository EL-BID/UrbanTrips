from __future__ import annotations

import argparse
import os


def add_bootstrap_args(parser: argparse.ArgumentParser) -> None:
    """Add -c/--config and -d/--base-dir to an existing parser."""
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Ruta al archivo de configuración YAML (por defecto: configs/configuraciones_generales.yaml)",
    )

    parser.add_argument(
        "-d",
        "--base-dir",
        type=str,
        default=None,
        dest="base_dir",
        help="Project root directory. Config, inputs, databases, and outputs are resolved relative to this path.",
    )


def apply_bootstrap_env(args: argparse.Namespace) -> None:
    """Set URBANTRIPS_CONFIG / URBANTRIPS_BASE env vars from parsed args, if present."""
    if args.config:
        os.environ["URBANTRIPS_CONFIG"] = args.config

    if args.base_dir:
        os.environ["URBANTRIPS_BASE"] = args.base_dir
