"""A/B de `add_od_lrs_to_legs_from_route`: versión anterior (un Point de shapely
por etapa y una proyección por fila) contra la nueva (una proyección por celda h3).

La proyección de una etapa sobre el recorrido depende SOLO de la celda h3 y de la
geometría del recorrido, y las celdas se repiten muchísimo: en la línea más
cargada del AMBA, 1.982.279 etapas de un mes usan 828 orígenes y 852 destinos
distintos. Proyectar una vez por celda da el mismo resultado y evita las cuatro
pasadas fila por fila.

La versión anterior se copia acá tal cual era, para que el A/B siga corriendo
aunque el código de producción cambie.

Uso:
    python tools/ab_lrs_dedup.py
    python tools/ab_lrs_dedup.py --corrida marzo_mes_completo_2026_fix --linea 359
    python tools/ab_lrs_dedup.py --limite 200000
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import duckdb
import numpy as np
from shapely import wkt

from urbantrips.carto.routes import get_route_section_id
from urbantrips.geo import geo
from urbantrips.kpi.kpi import add_od_lrs_to_legs_from_route

DB_DIR = Path(__file__).resolve().parents[1] / "data" / "db"


def version_anterior(legs_df, route_geom):
    """`add_od_lrs_to_legs_from_route` tal como era antes del dedup."""
    legs_df = legs_df.copy()
    legs_df["o"] = legs_df["h3_o"].map(geo.create_point_from_h3)
    legs_df["d"] = legs_df["h3_d"].map(geo.create_point_from_h3)
    legs_df["o_proj"] = list(
        map(get_route_section_id, legs_df["o"], itertools.repeat(route_geom))
    )
    legs_df["d_proj"] = list(
        map(get_route_section_id, legs_df["d"], itertools.repeat(route_geom))
    )
    return legs_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", default="marzo_mes_completo_2026_fix")
    parser.add_argument("--linea", type=int, default=359)
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="cuántas etapas leer; sin esto, todas las de la línea",
    )
    parser.add_argument("--duckdb-mem", default="8GB")
    args = parser.parse_args()

    insumos = duckdb.connect(
        str(DB_DIR / f"{args.corrida}_insumos.duckdb"), read_only=True
    )
    fila = insumos.execute(
        f"select * from lines_geoms where direction=0 and id_linea={args.linea}"
    ).fetchdf()
    insumos.close()
    if fila.empty:
        raise SystemExit(f"la línea {args.linea} no tiene recorrido en {args.corrida}")
    route_geom = wkt.loads(fila.wkt.item())
    print(f"línea {args.linea}: recorrido con {len(route_geom.coords)} vértices")

    data = duckdb.connect(str(DB_DIR / f"{args.corrida}_data.duckdb"), read_only=True)
    data.execute(f"SET memory_limit='{args.duckdb_mem}'")
    q = (
        "select dia, h3_o, h3_d, factor_expansion_linea from etapas "
        f"where od_validado=1 and id_linea={args.linea}"
    )
    if args.limite:
        q += f" limit {args.limite}"
    legs = data.execute(q).fetchdf()
    data.close()

    print(f"etapas: {len(legs):,}".replace(",", "."))
    print(
        f"celdas h3 distintas: {legs.h3_o.nunique()} orígenes / "
        f"{legs.h3_d.nunique()} destinos"
    )

    t = time.time()
    viejo = version_anterior(legs, route_geom)
    t_viejo = time.time() - t

    t = time.time()
    nuevo = add_od_lrs_to_legs_from_route(legs.copy(), route_geom)
    t_nuevo = time.time() - t

    iguales_o = np.array_equal(viejo.o_proj.values, nuevo.o_proj.values)
    iguales_d = np.array_equal(viejo.d_proj.values, nuevo.d_proj.values)

    print()
    print(f"viejo: {t_viejo:8.1f} s")
    print(f"nuevo: {t_nuevo:8.1f} s   ({t_viejo / max(t_nuevo, 1e-9):.0f}x)")
    print(f"o_proj idéntico: {iguales_o}")
    print(f"d_proj idéntico: {iguales_d}")

    if not (iguales_o and iguales_d):
        distintos = int((viejo.o_proj.values != nuevo.o_proj.values).sum())
        distintos += int((viejo.d_proj.values != nuevo.d_proj.values).sum())
        raise SystemExit(f"NO son equivalentes: {distintos} proyecciones distintas")


if __name__ == "__main__":
    main()
