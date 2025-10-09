import argparse
from urbantrips.utils.run_process import run_all

"""
────────────────────────────────────────────────────────────
📝 Ejemplos de uso desde consola (Windows o Linux):

python run_all_urbantrips.py
    → Corre solo las pendientes y crea el dashboard

python run_all_urbantrips.py --borrar_corrida all
    → Borra todo y vuelve a correr desde cero, creando dashboard

python run_all_urbantrips.py --borrar_corrida alias1
    → Borra y vuelve a correr el alias 'alias1' y lo que falte, creando dashboard

python run_all_urbantrips.py --no_dashboard
    → Corre pendientes sin crear el dashboard

python run_all_urbantrips.py --borrar_corrida alias1 --no_dashboard
    → Borra 'alias1', corre lo que falte, y no crea dashboard
────────────────────────────────────────────────────────────
"""


def main(borrar_corrida="", crear_dashboard=True):
    """
    Ejecuta el proceso principal de UrbanTrips.

    Parámetros:
    ----------
    borrar_corrida : str
        - ''     : Corre solo las corridas pendientes (no corridas previamente).
        - 'all'  : Borra todas las corridas y corre todo de nuevo.
        - alias  : Borra la corrida con el alias especificado y vuelve a correr lo faltante.

    crear_dashboard : bool
        Si es True, también ejecuta la creación del dashboard asociado (valor por defecto).
    """

    run_all(borrar_corrida=borrar_corrida, crear_dashboard=crear_dashboard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta corridas de UrbanTrips con opciones de borrado y dashboard."
    )

    parser.add_argument(
        "--borrar_corrida",
        type=str,
        default="",
        help="Opciones: '' (vacío, corre solo pendientes), 'all' (corre todo desde cero), o un alias específico",
    )

    parser.add_argument(
        "--no_dashboard",
        action="store_true",
        help="Si se incluye, se omite la creación del dashboard. Por defecto se crea.",
    )

    args = parser.parse_args()

    main(
        borrar_corrida=args.borrar_corrida,
        crear_dashboard=not args.no_dashboard,  # por defecto es True, salvo que se indique --no_dashboard
    )
