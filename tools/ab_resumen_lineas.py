# -*- coding: utf-8 -*-
"""A/B de los agregados de transacciones y gps del dashboard.

Compara la version vieja (levantar las tablas enteras en pandas) contra la
nueva (leerlas de a un dia y combinar los parciales) sobre una base real, para
los dos consumidores que hacian esas lecturas:

  * resumen_x_linea -> trx_agg / cant_internos_en_trx / cant_internos_en_gps
    por linea y por linea+ramal (groupby con observed=True)
  * levanto_data -> flota / cant_internos_en_gps / cant_internos_en_trx, con el
    dia derivado de fecha (puede partir grupos entre chunks; es el caso que
    ejercita combinar_suma / combinar_conteo_distintos)

Uso:
    python tools/ab_resumen_lineas.py --db data/db/<alias>_data.duckdb
    python tools/ab_resumen_lineas.py --db ... --modo new

--modo new existe para las bases donde la version vieja no entra en RAM (el
mes): ahi no hay contra que comparar, pero se mide el pico y se comprueba que
la nueva corre.
"""
import argparse
import os
import sys
import time

import duckdb
import pandas as pd
import psutil
from pandas.testing import assert_frame_equal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbantrips.utils.dataframe import (  # noqa: E402
    combinar_conteo_distintos,
    combinar_suma,
    dias_para_leer_por_dia,
    leer_dia,
)

COLS_LINEA = ["dia", "id_linea"]
COLS_RAMAL = ["dia", "id_linea", "id_ramal"]
PROC = psutil.Process(os.getpid())


def rss_gb():
    return PROC.memory_info().rss / 1e9


class Pico:
    """Pico de RSS del proceso, muestreado alrededor de cada bloque."""

    def __init__(self):
        self.max = rss_gb()

    def marca(self):
        self.max = max(self.max, rss_gb())
        return self.max


# --------------------------------------------------------------------------
# resumen_x_linea
# --------------------------------------------------------------------------
def resumen_viejo(q, pico):
    gps = q("SELECT dia, id_linea, id_ramal, interno FROM gps")
    pico.marca()
    trx = q(
        "SELECT dia, id_linea, id_ramal, modo, interno, factor_expansion "
        "FROM transacciones"
    )
    pico.marca()
    out = {}
    for nivel, cols in (("linea", COLS_LINEA), ("ramal", COLS_RAMAL)):
        out[nivel] = dict(
            trx_agg=(
                trx.groupby(cols + ["modo"], as_index=False, observed=True)
                .factor_expansion.sum()
                .rename(columns={"factor_expansion": "transacciones"})
            ),
            internos_agg=(
                trx.groupby(cols + ["interno"], as_index=False, observed=True)
                .size()
                .groupby(cols, as_index=False, observed=True)
                .size()
                .rename(columns={"size": "cant_internos_en_trx"})
            ),
            gps_agg=(
                gps.groupby(cols + ["interno"], as_index=False, observed=True)
                .size()
                .groupby(cols, as_index=False, observed=True)
                .size()
                .rename(columns={"size": "cant_internos_en_gps"})
            ),
        )
        pico.marca()
    del gps, trx
    return out


def resumen_nuevo(q, pico, dias=None):
    niveles = (("linea", COLS_LINEA), ("ramal", COLS_RAMAL))
    trx_parc = {"linea": [], "ramal": []}
    internos_parc = {"linea": [], "ramal": []}
    for dia in dias_para_leer_por_dia(q, "transacciones", dias):
        dia_trx = leer_dia(
            q, "transacciones",
            ["dia", "id_linea", "id_ramal", "modo", "interno", "factor_expansion"],
            dia,
        )
        pico.marca()
        for nivel, cols in niveles:
            trx_parc[nivel].append(
                dia_trx.groupby(cols + ["modo"], as_index=False, observed=True)
                .factor_expansion.sum()
            )
            internos_parc[nivel].append(
                dia_trx.groupby(cols + ["interno"], as_index=False, observed=True).size()
            )
        pico.marca()
        del dia_trx

    gps_parc = {"linea": [], "ramal": []}
    for dia in dias_para_leer_por_dia(q, "gps", dias):
        dia_gps = leer_dia(q, "gps", ["dia", "id_linea", "id_ramal", "interno"], dia)
        pico.marca()
        for nivel, cols in niveles:
            gps_parc[nivel].append(
                dia_gps.groupby(cols + ["interno"], as_index=False, observed=True).size()
            )
        pico.marca()
        del dia_gps

    return {
        nivel: dict(
            trx_agg=combinar_suma(
                trx_parc[nivel], cols + ["modo"], "factor_expansion"
            ).rename(columns={"factor_expansion": "transacciones"}),
            internos_agg=combinar_conteo_distintos(
                internos_parc[nivel], cols, "cant_internos_en_trx"
            ),
            gps_agg=combinar_conteo_distintos(
                gps_parc[nivel], cols, "cant_internos_en_gps"
            ),
        )
        for nivel, cols in niveles
    }


# --------------------------------------------------------------------------
# levanto_data (camino KPI): el dia sale de fecha, no de la columna dia
# --------------------------------------------------------------------------
def kpi_viejo(q, pico, entidad):
    cols = ["dia"] + entidad
    gps = q("SELECT fecha, id_linea, id_ramal, interno FROM gps")
    pico.marca()
    gps["fecha"] = pd.to_datetime(gps["fecha"], unit="s")
    gps["dia"] = gps["fecha"].dt.strftime("%Y-%m-%d")
    pico.marca()
    flota = (
        gps.groupby(cols, as_index=False).size().rename(columns={"size": "flota"})
    )
    gps_agg = (
        gps.groupby(cols + ["interno"], as_index=False)
        .size()
        .groupby(cols, as_index=False)
        .size()
        .rename(columns={"size": "cant_internos_en_gps"})
    )
    del gps
    trx = q("SELECT dia, id_linea, id_ramal, interno FROM transacciones")
    pico.marca()
    internos_agg = (
        trx.groupby(cols + ["interno"], as_index=False)
        .size()
        .groupby(cols, as_index=False)
        .size()
        .rename(columns={"size": "cant_internos_en_trx"})
    )
    del trx
    return dict(flota=flota, gps_agg=gps_agg, internos_agg=internos_agg)


def kpi_nuevo(q, pico, entidad, dias=None):
    cols = ["dia"] + entidad
    flota_parc, gps_parc = [], []
    for dia in dias_para_leer_por_dia(q, "gps", dias):
        dia_gps = leer_dia(q, "gps", ["fecha", "id_linea", "id_ramal", "interno"], dia)
        dia_gps["fecha"] = pd.to_datetime(dia_gps["fecha"], unit="s")
        dia_gps["dia"] = dia_gps["fecha"].dt.strftime("%Y-%m-%d")
        pico.marca()
        flota_parc.append(dia_gps.groupby(cols, as_index=False).size())
        gps_parc.append(dia_gps.groupby(cols + ["interno"], as_index=False).size())
        del dia_gps
    flota = combinar_suma(flota_parc, cols, "size", observed=False).rename(
        columns={"size": "flota"}
    )
    gps_agg = combinar_conteo_distintos(
        gps_parc, cols, "cant_internos_en_gps", observed=False
    )
    internos_parc = []
    for dia in dias_para_leer_por_dia(q, "transacciones", dias):
        dia_trx = leer_dia(
            q, "transacciones", ["dia", "id_linea", "id_ramal", "interno"], dia
        )
        pico.marca()
        internos_parc.append(dia_trx.groupby(cols + ["interno"], as_index=False).size())
        del dia_trx
    internos_agg = combinar_conteo_distintos(
        internos_parc, cols, "cant_internos_en_trx", observed=False
    )
    return dict(flota=flota, gps_agg=gps_agg, internos_agg=internos_agg)


# --------------------------------------------------------------------------
def comparar(viejo, nuevo, etiqueta):
    ok = True
    for k in sorted(viejo):
        a, b = viejo[k], nuevo[k]
        try:
            assert_frame_equal(a, b, check_exact=True, check_dtype=True)
            print(
                "  [OK]   {}.{}: {:,} filas identicas ({})".format(
                    etiqueta, k, len(a), ", ".join(a.columns)
                )
            )
        except AssertionError as exc:
            ok = False
            print("  [DIFF] {}.{}: {}".format(etiqueta, k, str(exc).splitlines()[0]))
            print("         viejo {} {}".format(a.shape, dict(a.dtypes.astype(str))))
            print("         nuevo {} {}".format(b.shape, dict(b.dtypes.astype(str))))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--modo", choices=["both", "new"], default="both")
    ap.add_argument(
        "--solo", default=None,
        help="corre un solo escenario: resumen | kpi_linea | kpi_ramal",
    )
    ap.add_argument(
        "--duckdb-mem", default=None,
        help="memory_limit de DuckDB; poner el del cliente al validar su camino",
    )
    args = ap.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    if args.duckdb_mem:
        con.execute("SET memory_limit='{}'".format(args.duckdb_mem))

    def q(sql):
        return con.execute(sql).fetchdf()

    n_trx = con.execute("SELECT count(*) FROM transacciones").fetchone()[0]
    n_gps = con.execute("SELECT count(*) FROM gps").fetchone()[0]
    print("base: {}".format(args.db))
    print(
        "transacciones {:,} | gps {:,} | RSS inicial {:.2f} GB\n".format(
            n_trx, n_gps, rss_gb()
        )
    )

    escenarios = [
        ("resumen", "resumen_x_linea", resumen_viejo, resumen_nuevo),
        (
            "kpi_linea", "levanto_data[linea]",
            lambda q_, p: kpi_viejo(q_, p, ["id_linea"]),
            lambda q_, p: kpi_nuevo(q_, p, ["id_linea"]),
        ),
        (
            "kpi_ramal", "levanto_data[ramal]",
            lambda q_, p: kpi_viejo(q_, p, ["id_linea", "id_ramal"]),
            lambda q_, p: kpi_nuevo(q_, p, ["id_linea", "id_ramal"]),
        ),
    ]
    if args.solo:
        escenarios = [e for e in escenarios if e[0] == args.solo]

    todo_ok = True
    for _clave, nombre, viejo_fn, nuevo_fn in escenarios:
        print("== {}".format(nombre))
        pico_n = Pico()
        t0 = time.time()
        nuevo = nuevo_fn(q, pico_n)
        print(
            "  nuevo: {:6.1f}s  pico RSS {:.2f} GB".format(
                time.time() - t0, pico_n.max
            )
        )

        if args.modo == "new":
            for k, v in sorted(nuevo.items()):
                if isinstance(v, dict):
                    for k2, v2 in sorted(v.items()):
                        print("    {}.{}: {:,} filas".format(k, k2, len(v2)))
                else:
                    print("    {}: {:,} filas".format(k, len(v)))
            print()
            continue

        pico_v = Pico()
        t0 = time.time()
        viejo = viejo_fn(q, pico_v)
        print(
            "  viejo: {:6.1f}s  pico RSS {:.2f} GB".format(
                time.time() - t0, pico_v.max
            )
        )

        if nombre == "resumen_x_linea":
            for nivel in ("linea", "ramal"):
                todo_ok &= comparar(
                    viejo[nivel], nuevo[nivel], "{}[{}]".format(nombre, nivel)
                )
        else:
            todo_ok &= comparar(viejo, nuevo, nombre)
        del viejo, nuevo
        print()

    con.close()
    if args.modo == "both":
        print("RESULTADO:", "IDENTICO" if todo_ok else "HAY DIFERENCIAS")
        raise SystemExit(0 if todo_ok else 1)


if __name__ == "__main__":
    main()
