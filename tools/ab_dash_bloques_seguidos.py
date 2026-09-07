# -*- coding: utf-8 -*-
"""Los bloques pesados del dashboard, SEGUIDOS EN UN SOLO PROCESO.

`ab_resumen_lineas.py` y `ab_socio_poligonos.py` miden cada bloque en un proceso
limpio, pero en produccion los cuatro corren uno atras del otro en el mismo
proceso. Si la memoria que un bloque libera no vuelve al SO, el pico del paso es
mayor que el mayor de los picos individuales -- medido sobre el mes: 18,83 GB
contra 11,50 del bloque mas grande solo. **El pico del paso no es el maximo de
los picos individuales**, y por eso este arnes existe.

El orden es el de `preparo_indicadores_dash`: mat -> resumen_x_linea ->
poligonos -> socio -> kpi por linea -> kpi por ramal. Fuera quedan `chains`, las
matrices y el resto, que son SQL adentro de DuckDB.

Uso:
    python tools/ab_dash_bloques_seguidos.py --alias marzo_mes_completo_2026_fix \
        --duckdb-mem 9GB

Las bases se abren read-only; el mat se materializa como TEMP en la conexion del
proceso, asi que no se toca nada en disco.
"""
import argparse
import os
import sys
import time

import duckdb

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import ab_resumen_lineas as abr  # noqa: E402
import ab_socio_poligonos as abs_  # noqa: E402
from urbantrips.preparo_dashboard.sql_queries import VIAJES_PROC_CTE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    ap.add_argument("--db-dir", default="data/db")
    ap.add_argument(
        "--duckdb-mem", default="9GB",
        help="memory_limit de DuckDB; por defecto el del cliente",
    )
    ap.add_argument(
        "--dias", choices=["corrida", "todos"], default="todos",
        help="'todos' es el caso del cliente: el mes entero en una sola corrida",
    )
    args = ap.parse_args()

    p = lambda suf: os.path.join(  # noqa: E731
        args.db_dir, "{}_{}.duckdb".format(args.alias, suf))

    con_data = duckdb.connect(p("data"), read_only=True)
    con_data.execute("SET memory_limit='{}'".format(args.duckdb_mem))
    q = lambda sql: con_data.execute(sql).fetchdf()  # noqa: E731

    # UN SOLO Pico para todos los bloques: es el punto del arnes
    pico = abs_.Pico()
    print("base: {} | duckdb mem: {} | RSS inicial {:.2f} GB\n".format(
        args.alias, args.duckdb_mem, abs_.rss_gb()))

    def marca(nombre, t0):
        mem_db = abs_._mem_duckdb_gb(q)
        r = abs_.rss_gb()
        pico.marca()
        print("{:<28} {:>6.0f}s | RSS {:5.2f} | duckdb {:5.2f} | python {:5.2f} "
              "| PICO ACUMULADO {:5.2f} GB".format(
                  nombre, time.time() - t0, r, mem_db, r - mem_db, pico.max),
              flush=True)

    if args.dias == "todos":
        sql_dias = ("SELECT DISTINCT dia AS dia FROM viajes "
                    "WHERE dia IS NOT NULL ORDER BY 1")
    else:
        sql_dias = "SELECT dia FROM dias_ultima_corrida ORDER BY 1"
    dias = [str(d) for d in q(sql_dias)["dia"].tolist()]

    t0 = time.time()
    con_data.execute(
        "CREATE TEMP TABLE {} AS WITH {} SELECT * FROM viajes_proc WHERE dia IN ({})"
        .format(abs_.MAT, VIAJES_PROC_CTE, ", ".join("'" + d + "'" for d in dias)))
    marca("materializar mat ({} dias)".format(len(dias)), t0)

    t0 = time.time()
    abr.resumen_nuevo(q, pico)
    marca("resumen_x_linea", t0)

    con_ins = duckdb.connect(p("insumos"), read_only=True)
    equivalencias = con_ins.execute(
        "SELECT h3, zona, tipo FROM equivalencias_zonas "
        "WHERE tipo IN ('poligono', 'cuenca')").fetchdf()
    con_ins.close()
    if len(equivalencias):
        con_dash = duckdb.connect(p("dash"), read_only=True)
        con_dash.execute("SET memory_limit='{}'".format(args.duckdb_mem))
        t0 = time.time()
        abs_.poligonos_nuevo(
            lambda sql: con_dash.execute(sql).fetchdf(), equivalencias, dias, pico)
        marca("poligonos", t0)
        con_dash.close()
    else:
        print("{:<28} sin poligonos en equivalencias_zonas: se saltea".format("poligonos"))

    t0 = time.time()
    abs_.socio_nuevo(q, pico)
    marca("crea_socio_indicadores", t0)

    for entidad, nombre in ((["id_linea"], "levanto_data[linea]"),
                            (["id_linea", "id_ramal"], "levanto_data[ramal]")):
        t0 = time.time()
        abr.kpi_nuevo(q, pico, entidad)
        marca(nombre, t0)

    con_data.close()
    print("\nPICO DEL PROCESO COMPLETO: {:.2f} GB".format(pico.max))


if __name__ == "__main__":
    main()
