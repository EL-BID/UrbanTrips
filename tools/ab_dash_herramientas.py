"""A/B de los cuatro pasos de "Herramientas interactivas" del dashboard.

Corre `run_basic_kpi`, `compute_lines_od_matrix`, `compute_route_section_supply`
y `compute_route_section_load` para una línea, vuelca las tablas de salida a CSV
y las compara contra otra corrida. Sirve para verificar que acotar por días y
resolver los parámetros de sección no movió ningún número.

Se corre dos veces, cada una contra su propia copia de la base:

    # version anterior, desde un worktree en el commit viejo
    git worktree add /tmp/ut_head HEAD --detach
    set UT_RAIZ=/tmp/ut_head
    python tools/ab_dash_herramientas.py abtestold salida_old

    # version nueva
    set UT_RAIZ=<repo>
    python tools/ab_dash_herramientas.py abtestnew salida_new --dias

    # comparación
    python tools/ab_dash_herramientas.py --comparar salida_old salida_new

Dos cosas que hacen falta y no son obvias:

- **Correr siempre con el cwd del repo principal**, aun cuando el código salga de
  un worktree: `compute_od_distances` cachea la red de OSM en `cache/osm`
  relativo al cwd, y sin ese cache se queda bajando de Overpass.
- **La línea tiene que tener un recorrido sano.** Las líneas sin recorrido
  oficial caen al inferido, que mide miles de km: la versión nueva las rechaza y
  la vieja devuelve cualquier cosa, así que no hay nada que comparar.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

RAIZ = Path(os.environ.get("UT_RAIZ", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(RAIZ))

DB_DIR = Path(__file__).resolve().parents[1] / "data" / "db"


# Clave natural de cada tabla de salida. Comparar ordenando por TODAS las
# columnas desalinea las filas justo cuando hay diferencias, que es el caso que
# importa: hay varias filas por clave (una por día) y los empates se rompen
# distinto en cada lado.
CLAVES = {
    "lines_od_matrix_by_section": [
        "id_linea", "yr_mo", "day_type", "n_sections", "hour_min", "hour_max",
        "section_id_o", "section_id_d",
    ],
    "ocupacion_por_linea_tramo": [
        "id_linea", "yr_mo", "day_type", "n_sections", "section_meters",
        "sentido", "section_id", "hour_min", "hour_max",
    ],
    "supply_stats_by_section_id": [
        "id_linea", "yr_mo", "dia", "day_type", "n_sections", "section_meters",
        "sentido", "section_id",
    ],
    "basic_kpi_by_line_day": ["dia", "id_linea"],
    "basic_kpi_by_line_hr": ["dia", "id_linea", "hora"],
    "basic_kpi_by_vehicle_hr": ["dia", "id_linea", "id_ramal", "interno", "hora"],
}


def comparar(dir_a: Path, dir_b: Path) -> int:
    """Compara los CSV de las dos corridas. Devuelve la cantidad de diferencias."""
    nombres = sorted({p.name for p in dir_a.glob("*.csv")} | {p.name for p in dir_b.glob("*.csv")})
    if not nombres:
        raise SystemExit("no hay CSV para comparar")

    problemas = 0
    for nombre in nombres:
        pa, pb = dir_a / nombre, dir_b / nombre
        if not pa.exists() or not pb.exists():
            print(f"{nombre:34} SOLO EN {'A' if pa.exists() else 'B'}")
            problemas += 1
            continue

        a = pd.read_csv(pa)
        b = pd.read_csv(pb)
        if a.shape != b.shape:
            print(f"{nombre:34} distinta forma: {a.shape} vs {b.shape}")
            problemas += 1
            continue

        tabla = nombre.removesuffix(".csv")
        llave = [c for c in CLAVES.get(tabla, sorted(a.columns)) if c in a.columns]
        a = a.sort_values(llave).reset_index(drop=True)
        b = b.sort_values(llave).reset_index(drop=True)

        difs = []
        for col in a.columns:
            if pd.api.types.is_numeric_dtype(a[col]):
                distintas = int(
                    ((a[col].fillna(-999) - b[col].fillna(-999)).abs() > 1e-9).sum()
                )
            else:
                distintas = int((a[col].fillna("") != b[col].fillna("")).sum())
            if distintas:
                difs.append(f"{col} ({distintas})")

        if difs:
            print(f"{nombre:34} DISTINTO en {', '.join(difs)}")
            problemas += 1
        else:
            print(f"{nombre:34} idéntico ({len(a)} filas)")

    return problemas


def correr(alias: str, salida: Path, usar_dias: bool, linea: int,
           n_sections: int, day_type: str, epsg_m: int | None) -> None:
    salida.mkdir(parents=True, exist_ok=True)

    if epsg_m is not None:
        # La configuración activa puede ser de otra ciudad que la base del A/B.
        # Se fija a mano para que los dos lados midan igual.
        from urbantrips.geo import geo

        geo.get_epsg_m = lambda: epsg_m

    from urbantrips.kpi.kpi import compute_route_section_load, run_basic_kpi
    from urbantrips.kpi.line_od_matrix import compute_lines_od_matrix
    from urbantrips.kpi.supply_kpi import compute_route_section_supply
    from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
    from urbantrips.storage.context import StorageContext

    conn = duckdb.connect(str(DB_DIR / f"{alias}_data.duckdb"), read_only=True)
    todos = conn.execute("select distinct dia from etapas order by 1").fetchdf().dia.tolist()
    conn.close()
    quiero_habiles = day_type == "weekday"
    dias = [d for d in todos if (pd.Timestamp(d).dayofweek < 5) == quiero_habiles]
    print(f"alias={alias}  días={dias}")

    ctx = StorageContext(
        data=DuckDBDataAdapter(str(DB_DIR / f"{alias}_data.duckdb"), read_only=False),
        insumos=DuckDBInsumoAdapter(str(DB_DIR / f"{alias}_insumos.duckdb"), read_only=False),
        dash=DuckDBDashAdapter(str(DB_DIR / f"{alias}_dash.duckdb"), read_only=False),
        general=DuckDBGeneralAdapter(str(DB_DIR / f"{alias}_general.duckdb"), read_only=False),
    )
    extra = {"dias": dias} if usar_dias else {}

    print("1/4 run_basic_kpi")
    run_basic_kpi(ctx, id_linea=[linea], **extra)

    print("2/4 compute_lines_od_matrix")
    compute_lines_od_matrix(
        ctx, line_ids=[linea], hour_range=False, n_sections=n_sections,
        section_meters=None, day_type=day_type, save_csv=False, **extra,
    )

    print("3/4 compute_route_section_supply")
    compute_route_section_supply(
        ctx, line_ids=[linea], hour_range=False, n_sections=n_sections,
        section_meters=None, day_type=day_type, **extra,
    )

    print("4/4 compute_route_section_load")
    compute_route_section_load(
        ctx, line_ids=[linea], hour_range=False, n_sections=n_sections,
        section_meters=None, day_type=day_type, **extra,
    )

    tablas = [
        "lines_od_matrix_by_section",
        "ocupacion_por_linea_tramo",
        "supply_stats_by_section_id",
        "basic_kpi_by_line_day",
        "basic_kpi_by_line_hr",
        "basic_kpi_by_vehicle_hr",
    ]
    for tabla in tablas:
        try:
            df = ctx.data.query(f"select * from {tabla}")
        except Exception as exc:
            print(f"  {tabla}: sin tabla ({type(exc).__name__})")
            continue
        df = df.sort_values(list(df.columns)).reset_index(drop=True)
        df.to_csv(salida / f"{tabla}.csv", index=False)
        print(f"  {tabla}: {len(df)} filas")

    for adapter in (ctx.data, ctx.insumos, ctx.dash, ctx.general):
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias", nargs="?", help="alias de la base a procesar")
    parser.add_argument("salida", nargs="?", help="directorio donde dejar los CSV")
    parser.add_argument(
        "--dias", action="store_true",
        help="usar la firma nueva (dias= en las cuatro funciones)",
    )
    parser.add_argument("--linea", type=int, default=900)
    parser.add_argument("--n-sections", type=int, default=10)
    parser.add_argument("--day-type", default="weekday")
    parser.add_argument(
        "--epsg-m", type=int, default=None,
        help="CRS en metros a forzar (5344 para Mendoza, 9265 para AMBA)",
    )
    parser.add_argument(
        "--comparar", nargs=2, metavar=("DIR_A", "DIR_B"),
        help="comparar dos directorios de salida en vez de correr",
    )
    args = parser.parse_args()

    if args.comparar:
        problemas = comparar(Path(args.comparar[0]), Path(args.comparar[1]))
        print()
        if problemas:
            raise SystemExit(f"{problemas} tabla(s) con diferencias")
        print("todas las tablas idénticas")
        return

    if not args.alias or not args.salida:
        parser.error("hacen falta alias y salida (o --comparar)")

    correr(
        args.alias, Path(args.salida), args.dias, args.linea,
        args.n_sections, args.day_type, args.epsg_m,
    )
    print("listo")


if __name__ == "__main__":
    main()
