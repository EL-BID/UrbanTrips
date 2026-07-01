import argparse
import logging
import time

from urbantrips.utils.run_process import (
    _build_ctx,
    check_prerequisites,
    run_ingest,
    run_legs,
    run_outputs,
    run_dashboard,
    run_all,
)

"""
────────────────────────────────────────────────────────────────────────────
Ejemplos de uso desde consola:

python run_all_urbantrips.py
    → Corre todo (ingest → legs → outputs → dashboard)

python run_all_urbantrips.py --no_dashboard
    → Corre ingest → legs → outputs (sin dashboard)

python run_all_urbantrips.py --through outputs
    → Corre ingest → legs → outputs (sin dashboard)

python run_all_urbantrips.py --through legs
    → Corre solo ingest + legs

python run_all_urbantrips.py --step dashboard
    → Corre solo el dashboard (valida que outputs haya corrido antes)

python run_all_urbantrips.py --step outputs
    → Corre solo outputs (valida que legs haya corrido antes)

python run_all_urbantrips.py --borrar_corrida all
    → Borra todo y vuelve a correr desde cero, creando dashboard

python run_all_urbantrips.py --config configs/otra_ciudad.yaml
    → Usa un archivo de configuración alternativo
────────────────────────────────────────────────────────────────────────────
"""

_STEP_ORDER = ["ingest", "legs", "outputs", "dashboard"]

_STEP_FNS = {
    "ingest": run_ingest,
    "legs": run_legs,
    "outputs": run_outputs,
    "dashboard": run_dashboard,
}


def _run_step(step: str) -> None:
    ctx = _build_ctx()
    check_prerequisites(step, ctx)
    _STEP_FNS[step](ctx)


def _run_through(through: str) -> None:
    ctx = _build_ctx()
    steps = _STEP_ORDER[: _STEP_ORDER.index(through) + 1]
    for step in steps:
        check_prerequisites(step, ctx)
        _STEP_FNS[step](ctx)


def main(borrar_corrida="", crear_dashboard=True, step=None, through=None):
    if step is not None:
        _run_step(step)
    elif through is not None:
        _run_through(through)
    else:
        run_all(borrar_corrida=borrar_corrida, crear_dashboard=crear_dashboard)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ejecuta corridas de UrbanTrips con opciones de borrado y dashboard."
    )

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

    parser.add_argument(
        "-b",
        "--borrar_corrida",
        type=str,
        default="",
        help="Opciones: '' (vacío), 'all' (todo), o un alias específico",
    )

    parser.add_argument(
        "-n",
        "--no_dashboard",
        action="store_true",
        help="Omite la creación del dashboard. Por defecto se crea.",
    )

    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--step",
        choices=_STEP_ORDER,
        default=None,
        help="Ejecuta un único paso (ingest | legs | outputs | dashboard).",
    )
    exclusive.add_argument(
        "--through",
        choices=_STEP_ORDER,
        default=None,
        help="Ejecuta desde ingest hasta el paso indicado (inclusive).",
    )

    return parser


def _validate_args(args):
    if args.step is not None and args.borrar_corrida:
        raise SystemExit(
            "error: --step and --borrar_corrida are incompatible. "
            "Cannot delete-and-re-run a single step in isolation."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import os
    from pathlib import Path
    from urbantrips.utils.paths import init_paths

    inicio = time.time()
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)

    if args.config:
        os.environ["URBANTRIPS_CONFIG"] = args.config

    if args.base_dir:
        os.environ["URBANTRIPS_BASE"] = args.base_dir

    init_paths(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        config_file=Path(args.config) if args.config else None,
    )

    main(
        borrar_corrida=args.borrar_corrida,
        crear_dashboard=not args.no_dashboard,
        step=args.step,
        through=args.through,
    )
    fin = time.time()
    logging.info("tiempo total de la corrida: %.2f min", (fin - inicio) / 60)
