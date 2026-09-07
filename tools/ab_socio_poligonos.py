# -*- coding: utf-8 -*-
"""A/B de los otros dos bloques del dashboard que levantaban una tabla entera.

  * crea_socio_indicadores -> el bloque "viajes promedio por usuario", que leia
    viajes_proc_mat proyectado a 8 columnas (475 B/fila medidos)
  * _viajes_poligonos_desde_chains -> la seleccion de viajes por poligono, que
    leia chains_norm proyectado a 13 columnas (717 B/fila medidos)

Compara la version vieja (tabla entera en pandas) contra la nueva (un dia por
vez) sobre una base real, sin tocarla: las bases se abren read-only y el
viajes_proc_mat se materializa como TEMP en la conexion del proceso.

Uso:
    python tools/ab_socio_poligonos.py --alias semana_prueba_ramales_20260820
    python tools/ab_socio_poligonos.py --alias mza_oct --solo poligonos
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

from urbantrips.preparo_dashboard.sql_queries import VIAJES_PROC_CTE  # noqa: E402
from urbantrips.utils.dataframe import (  # noqa: E402
    calculate_weighted_means,
    dias_para_leer_por_dia,
    leer_dia,
)

PROC = psutil.Process(os.getpid())
MAT = "viajes_proc_mat"
AGG_COLS_USER = ["dia", "mes", "tipo_dia", "genero_agregado", "tarifa_agregada"]
COLS_USER = [
    "dia", "mes", "tipo_dia", "id_tarjeta", "genero_agregado",
    "tarifa_agregada", "factor_expansion_tarjeta", "factor_expansion_linea",
]
COLS_CHAINS = [
    "dia", "mes", "tipo_dia", "id_tarjeta", "id_viaje",
    "h3_inicio_norm", "h3_fin_norm", "modo_agregado", "rango_hora",
    "transferencia", "distancia_agregada", "distance_od",
    "factor_expansion_linea",
]


def rss_gb():
    return PROC.memory_info().rss / 1e9


class Pico:
    def __init__(self):
        self.max = rss_gb()

    def marca(self):
        self.max = max(self.max, rss_gb())
        return self.max


# --------------------------------------------------------------------------
# socio: viajes promedio por usuario
# --------------------------------------------------------------------------
def _userx_desde_frame(viajes_user):
    """El bloque pandas comun a las dos versiones (por-tarjeta -> por-dia)."""
    _userx_clean = viajes_user[["dia", "id_tarjeta"]].copy()
    _userx_clean["tarifa_agregada"] = viajes_user["tarifa_agregada"].str.replace("-", "")
    _tarifa_agg = duckdb.sql("""
        SELECT dia, id_tarjeta,
               COALESCE(STRING_AGG(DISTINCT NULLIF(tarifa_agregada, ''), '-'), '-') AS tarifa_agregada_agg
        FROM _userx_clean
        GROUP BY dia, id_tarjeta
    """).df()
    _orden_tarifas = {
        v: "-".join(sorted(v.split("-")))
        for v in _tarifa_agg["tarifa_agregada_agg"].unique()
    }
    _tarifa_agg["tarifa_agregada_agg"] = _tarifa_agg["tarifa_agregada_agg"].map(
        _orden_tarifas
    )
    userx = viajes_user[
        ["dia", "mes", "tipo_dia", "id_tarjeta", "genero_agregado",
         "factor_expansion_tarjeta", "factor_expansion_linea"]
    ].merge(_tarifa_agg, how="left")
    return (
        userx.groupby(
            ["dia", "mes", "tipo_dia", "id_tarjeta", "genero_agregado",
             "tarifa_agregada_agg"],
            as_index=False, observed=True)
        .agg({"factor_expansion_tarjeta": "count", "factor_expansion_linea": "mean"})
        .rename(columns={"factor_expansion_tarjeta": "cant_viajes"})
        .rename(columns={"tarifa_agregada_agg": "tarifa_agregada"})
    )


def _wm(df, sumado):
    return calculate_weighted_means(
        df,
        aggregate_cols=AGG_COLS_USER,
        weighted_mean_cols=["cant_viajes"],
        weight_col="factor_expansion_linea",
        var_fex_summed=sumado,
    ).round(3)


def socio_viejo(q, pico):
    viajes_user = q("SELECT {} FROM {}".format(", ".join(COLS_USER), MAT))
    pico.marca()
    userx = _userx_desde_frame(viajes_user)
    del viajes_user
    pico.marca()
    userx = _wm(userx, True)
    userx = _wm(userx, False)
    return {"userx": userx}


def _mem_duckdb_gb(q):
    """Lo que la conexion tiene en su buffer manager (tablas TEMP incluidas).

    Separa "esto lo tiene DuckDB" de "esto lo tiene el heap de Python": sin el
    corte, un RSS que sube no dice cual de los dos crece.
    """
    try:
        df = q("SELECT sum(memory_usage_bytes) AS b FROM duckdb_memory()")
        return float(df["b"].iloc[0] or 0) / 1e9
    except Exception:
        return float("nan")


def socio_nuevo(q, pico, traza=False):
    """La version nueva es la funcion de produccion, llamada tal cual.

    No se replica su logica aca: el `query_fn` que se le pasa envuelve al real y
    aprovecha que todo lo que hace pasa por ahi para trazar RSS por dia sin
    tocar el codigo de produccion.
    """
    from urbantrips.preparo_dashboard.preparo_dashboard import (
        _socio_userx_por_dia,
    )

    n = {"dias": 0}

    def q_traza(sql):
        df = q(sql)
        pico.marca()
        if traza and "dia_mat" in sql:
            n["dias"] += 1
            mem_db = _mem_duckdb_gb(q)
            r = rss_gb()
            print("    dia {:>2} | {:>4} filas devueltas | RSS {:5.2f} | duckdb "
                  "{:5.2f} | python {:5.2f} | pico {:5.2f} GB".format(
                      n["dias"], len(df), r, mem_db, r - mem_db, pico.max),
                  flush=True)
        return df

    userx = _socio_userx_por_dia(q_traza, MAT)
    pico.marca()
    userx = _wm(userx, False)
    return {"userx": userx}


# --------------------------------------------------------------------------
# poligonos: seleccion de viajes desde chains_norm
# --------------------------------------------------------------------------
def _mascara(chains, h3_poly, tipo):
    en_origen = chains["h3_inicio_norm"].isin(h3_poly)
    en_destino = chains["h3_fin_norm"].isin(h3_poly)
    return (en_origen & en_destino) if tipo == "cuenca" else (en_origen | en_destino)


def poligonos_viejo(q_dash, equivalencias, dias, pico):
    where = ""
    if dias:
        valores = ", ".join("'" + d + "'" for d in dias)
        where = " WHERE dia IN ({})".format(valores)
    chains = q_dash(
        "SELECT {} FROM chains_norm{}".format(", ".join(COLS_CHAINS), where)
    )
    pico.marca()
    frames = []
    for (zona, tipo), grupo in equivalencias.groupby(["zona", "tipo"], observed=True):
        mask = _mascara(chains, set(grupo["h3"]), tipo)
        if not mask.any():
            continue
        frames.append(
            chains.loc[mask]
            .drop(columns=["h3_inicio_norm", "h3_fin_norm"])
            .assign(id_polygon=zona)
        )
        pico.marca()
    del chains
    if not frames:
        return {"viajes": pd.DataFrame([])}
    viajes = pd.concat(frames, ignore_index=True).rename(
        columns={"modo_agregado": "modo"}
    )
    return {"viajes": viajes}


def poligonos_nuevo(q_dash, equivalencias, dias, pico):
    poligonos_h3 = {
        (zona, tipo): set(grupo["h3"])
        for (zona, tipo), grupo in equivalencias.groupby(["zona", "tipo"], observed=True)
    }
    seleccion = {clave: [] for clave in poligonos_h3}
    for dia in dias_para_leer_por_dia(q_dash, "chains_norm", dias):
        chains = leer_dia(q_dash, "chains_norm", COLS_CHAINS, dia)
        pico.marca()
        if len(chains) == 0:
            continue
        for clave, h3_poly in poligonos_h3.items():
            zona, tipo = clave
            mask = _mascara(chains, h3_poly, tipo)
            if not mask.any():
                continue
            seleccion[clave].append(
                chains.loc[mask]
                .drop(columns=["h3_inicio_norm", "h3_fin_norm"])
                .assign(id_polygon=zona)
            )
        del chains
        pico.marca()
    frames = [
        pd.concat(partes, ignore_index=True)
        for partes in seleccion.values()
        if partes
    ]
    if not frames:
        return {"viajes": pd.DataFrame([])}
    viajes = pd.concat(frames, ignore_index=True).rename(
        columns={"modo_agregado": "modo"}
    )
    return {"viajes": viajes}


# --------------------------------------------------------------------------
def comparar(viejo, nuevo, etiqueta):
    ok = True
    for k in sorted(viejo):
        a, b = viejo[k], nuevo[k]
        if len(a) == 0 and len(b) == 0:
            print("  [OK]   {}.{}: ambos vacios".format(etiqueta, k))
            continue
        # el orden de fila no es parte del contrato (todo lo que sigue es un
        # GROUP BY): se compara el contenido, ordenado por todas las columnas
        cols = list(a.columns)
        # ordenar por las columnas NO flotantes: si se ordena por una metrica,
        # una diferencia numerica reordena las filas y el error sale reportado
        # sobre una columna de texto, que confunde el diagnostico
        claves = [c for c in cols if not pd.api.types.is_float_dtype(a[c])] or cols
        a_s = a.sort_values(claves).reset_index(drop=True)
        b_s = b[cols].sort_values(claves).reset_index(drop=True)
        try:
            assert_frame_equal(a_s, b_s, check_exact=True, check_dtype=True)
            print("  [OK]   {}.{}: {:,} filas identicas ({})".format(
                etiqueta, k, len(a), ", ".join(cols)))
        except AssertionError as exc:
            ok = False
            print("  [DIFF] {}.{}: {}".format(etiqueta, k, str(exc).splitlines()[0]))
            print("         viejo {} / nuevo {}".format(a.shape, b.shape))
            try:
                assert_frame_equal(a_s, b_s, check_exact=False, rtol=1e-9, atol=1e-9)
                print("         (coinciden con tolerancia 1e-9: diferencia de ULP)")
            except AssertionError:
                print("         (tampoco coinciden con tolerancia 1e-9)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    ap.add_argument("--db-dir", default="data/db")
    ap.add_argument("--solo", default=None, choices=["socio", "poligonos"])
    ap.add_argument("--modo", choices=["both", "new"], default="both")
    ap.add_argument(
        "--traza", action="store_true",
        help="imprime RSS por dia en el bloque socio (para ver si el pico es "
             "de un dia o acumulado)",
    )
    ap.add_argument(
        "--duckdb-mem", default="8GB",
        help="memory_limit de DuckDB; poner el del cliente al validar su camino",
    )
    ap.add_argument(
        "--dias", choices=["corrida", "todos"], default="corrida",
        help="'corrida' usa dias_ultima_corrida (como quedo la base); 'todos' "
             "toma los dias presentes en las tablas, que es como lo ve el "
             "cliente cuando corre el mes entero en una sola corrida",
    )
    args = ap.parse_args()

    p = lambda suf: os.path.join(args.db_dir, "{}_{}.duckdb".format(args.alias, suf))  # noqa: E731
    con_data = duckdb.connect(p("data"), read_only=True)
    con_data.execute("SET memory_limit='{}'".format(args.duckdb_mem))

    def q_data(sql):
        return con_data.execute(sql).fetchdf()

    if args.dias == "todos":
        sql_dias = "SELECT DISTINCT dia AS dia FROM viajes WHERE dia IS NOT NULL ORDER BY 1"
    else:
        sql_dias = "SELECT dia FROM dias_ultima_corrida ORDER BY 1"
    run_days = [str(d) for d in con_data.execute(sql_dias).fetchdf()["dia"].tolist()]
    print("base: {} | dias: {} ({}) | duckdb mem: {}".format(
        args.alias, len(run_days), args.dias, args.duckdb_mem))

    todo_ok = True

    if args.solo in (None, "socio"):
        print("\n== crea_socio_indicadores [viajes promedio por usuario]")
        t0 = time.time()
        dias_sql = ", ".join("'" + d + "'" for d in run_days)
        con_data.execute(
            "CREATE TEMP TABLE {} AS WITH {} SELECT * FROM viajes_proc "
            "WHERE dia IN ({})".format(MAT, VIAJES_PROC_CTE, dias_sql)
        )
        n = con_data.execute("SELECT count(*) FROM {}".format(MAT)).fetchone()[0]
        print("  {} materializado: {:,} filas ({:.0f}s)".format(
            MAT, n, time.time() - t0))

        pico_n = Pico()
        t0 = time.time()
        nuevo = socio_nuevo(q_data, pico_n, traza=args.traza)
        print("  nuevo: {:6.1f}s  pico RSS {:.2f} GB".format(
            time.time() - t0, pico_n.max))
        if args.modo == "new":
            for k, v in sorted(nuevo.items()):
                print("    {}: {:,} filas".format(k, len(v)))
            del nuevo
        else:
            pico_v = Pico()
            t0 = time.time()
            viejo = socio_viejo(q_data, pico_v)
            print("  viejo: {:6.1f}s  pico RSS {:.2f} GB".format(
                time.time() - t0, pico_v.max))
            todo_ok &= comparar(viejo, nuevo, "socio")
            del viejo, nuevo

    if args.solo in (None, "poligonos"):
        print("\n== _viajes_poligonos_desde_chains")
        con_ins = duckdb.connect(p("insumos"), read_only=True)
        equivalencias = con_ins.execute(
            "SELECT h3, zona, tipo FROM equivalencias_zonas "
            "WHERE tipo IN ('poligono', 'cuenca')"
        ).fetchdf()
        con_ins.close()
        if len(equivalencias) == 0:
            print("  sin poligonos en equivalencias_zonas: nada que comparar")
        else:
            con_dash = duckdb.connect(p("dash"), read_only=True)
            con_dash.execute("SET memory_limit='{}'".format(args.duckdb_mem))

            def q_dash(sql):
                return con_dash.execute(sql).fetchdf()

            pico_n = Pico()
            t0 = time.time()
            nuevo = poligonos_nuevo(q_dash, equivalencias, run_days, pico_n)
            print("  nuevo: {:6.1f}s  pico RSS {:.2f} GB".format(
                time.time() - t0, pico_n.max))
            if args.modo == "new":
                for k, v in sorted(nuevo.items()):
                    print("    {}: {:,} filas".format(k, len(v)))
                del nuevo
            else:
                pico_v = Pico()
                t0 = time.time()
                viejo = poligonos_viejo(q_dash, equivalencias, run_days, pico_v)
                print("  viejo: {:6.1f}s  pico RSS {:.2f} GB".format(
                    time.time() - t0, pico_v.max))
                todo_ok &= comparar(viejo, nuevo, "poligonos")
            con_dash.close()

    con_data.close()
    if args.modo == "new":
        # no se comparo nada contra la version vieja: decirlo, para que nadie
        # lea un "IDENTICO" que no hubo
        print("\nRESULTADO: solo corrio la version nueva (sin comparacion)")
        raise SystemExit(0)
    print("\nRESULTADO:", "IDENTICO" if todo_ok else "HAY DIFERENCIAS")
    raise SystemExit(0 if todo_ok else 1)


if __name__ == "__main__":
    main()
