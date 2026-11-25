"""Mide cuánta RAM consume de verdad procesar UN día, para calibrar el divisor de
`_parallel_day_workers` (`_PANDAS_COPY_AMPLIFICATION` en `datamodel/legs.py`).

Corre el cómputo de un día en un proceso hijo — el mismo que corre en un worker del
camino paralelo — y muestrea su RSS desde el padre. Devuelve el factor real:

    factor = pico_RSS_del_worker / insumos_estimados

`_estimated_day_footprint_gb` estima los insumos exactamente (filas x ancho de fila
derivado del esquema); lo único que estaba a ojo era ese multiplicador. Con este
número se reemplaza la constante por una medición.

**Abre la base en modo READ ONLY.** No escribe nada: las funciones de worker
(`_infer_destinations_dia_worker`, `_gps_destino_y_tiempos_dia`) son pandas puro, y
el parquet del worker de destinos va a un temp que se borra.

Uso:
    python tools/medir_footprint_dia.py marzo_mes_completo_2026_fix
    python tools/medir_footprint_dia.py <alias> --etapa tiempos --dia 2026-04-01
    python tools/medir_footprint_dia.py <alias> --etapa chains

`--etapa chains` calibra `_CHAINS_COPY_AMPLIFICATION` (preparo_dashboard/chains.py),
que es aparte: chains genera mucho más que lo que lee. Ahí los insumos de la tabla
son los que ESTIMA el autotune (no los de pandas), así el factor se usa tal cual.
"""
import argparse
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("footprint")

DB_DIR = Path(__file__).resolve().parents[1] / "data" / "db"
GB = float(2**30)


def _abrir(alias):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    data = DuckDBDataAdapter(DB_DIR / f"{alias}_data.duckdb", read_only=True)
    insumos = DuckDBInsumoAdapter(DB_DIR / f"{alias}_insumos.duckdb", read_only=True)
    return data, insumos


def _dia_mas_grande(data):
    df = data.query(
        "SELECT dia, count(*) AS n FROM etapas GROUP BY 1 ORDER BY n DESC LIMIT 1"
    )
    return df["dia"].iloc[0], int(df["n"].iloc[0])


def _dia_mas_grande_chains(data):
    """El día que dimensiona el autotune de chains: el de más etapas od_validado
    dentro de dias_ultima_corrida (lo que lee _fetch_chains_inputs_dia)."""
    df = data.query(
        "SELECT e.dia, count(*) AS n FROM etapas e "
        "JOIN dias_ultima_corrida d ON e.dia = d.dia "
        "WHERE e.od_validado = 1 GROUP BY 1 ORDER BY n DESC LIMIT 1"
    )
    if df.empty:
        raise SystemExit("chains: no hay etapas od_validado en dias_ultima_corrida")
    return df["dia"].iloc[0], int(df["n"].iloc[0])


def _mem_gb(*frames):
    return sum(f.memory_usage(deep=True).sum() for f in frames if f is not None) / GB


# ── el cómputo que se mide, ya en el hijo ───────────────────────────────────────
def _hijo_destinos(dia, min_dist, etapas, metadata_lineas, matriz_val, modos_ramal, q):
    from urbantrips.destinations.destinations import _infer_destinations_dia_worker

    stage = tempfile.mkdtemp(prefix="footprint_")
    try:
        _infer_destinations_dia_worker(
            dia, 0, min_dist, etapas, metadata_lineas, matriz_val, modos_ramal, stage
        )
        q.put("ok")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _hijo_chains(etapas, viajes, res, max_etapas, q):
    from urbantrips.preparo_dashboard.chains import _compute_chains_dia

    _compute_chains_dia(etapas, viajes, res, max_etapas, False)
    q.put("ok")


def _hijo_tiempos(args_tuple, q):
    from urbantrips.datamodel.legs import _gps_destino_y_tiempos_dia

    _gps_destino_y_tiempos_dia(*args_tuple)
    q.put("ok")


def _correr_midiendo(target, args, etiqueta):
    """Lanza `target` en un hijo y muestrea su RSS hasta que termina."""
    import psutil

    ctx_mp = mp.get_context("spawn")
    q = ctx_mp.Queue()
    p = ctx_mp.Process(target=target, args=args + (q,))
    t0 = time.time()
    p.start()
    proc = psutil.Process(p.pid)
    pico = 0.0
    base = None
    try:
        while p.is_alive():
            try:
                rss = proc.memory_info().rss / GB
            except psutil.Error:
                break
            if base is None and rss > 0:
                base = rss  # intérprete + imports, antes de recibir los datos
            pico = max(pico, rss)
            time.sleep(0.2)
    finally:
        p.join()
    dur = time.time() - t0
    logger.info(
        "%s: pico RSS del worker %.2f GB (arranque %.2f GB) en %.1f min%s",
        etiqueta, pico, base or 0.0, dur / 60,
        "" if p.exitcode == 0 else f" — EXITCODE {p.exitcode}",
    )
    return pico, dur, p.exitcode


# ── armado de insumos (en el padre, igual que el pipeline) ──────────────────────
def medir_destinos(data, insumos, dia):
    from urbantrips.destinations.destinations import (
        _fetch_etapas_dia_infer, _prep_matriz_infer,
    )
    from urbantrips.utils.utils import leer_configs_generales, modos_con_ramal

    ctx = SimpleNamespace(data=data, insumos=insumos)

    configs = leer_configs_generales(autogenerado=False)
    # El camino min_distancia pasa una matriz con `parada` y hace la busqueda de
    # minima distancia: es bastante mas caro que el de validacion. Tomarlo del
    # config y no hardcodearlo, o se mide el camino equivocado.
    min_dist = configs.get("imputar_destinos_min_distancia", False)

    etapas = _fetch_etapas_dia_infer(ctx, dia)
    metadata_lineas = insumos.get_metadata_lineas()[["id_linea", "id_linea_agg"]]
    modos_ramal = modos_con_ramal(configs)
    matriz_val = _prep_matriz_infer(insumos.get_matrix_validation(), min_dist)

    ins = _mem_gb(etapas, metadata_lineas, matriz_val)
    logger.info(
        "infer_destinations — insumos del %s: %s etapas, %.2f GB en pandas "
        "(imputar_destinos_min_distancia=%s)",
        dia, f"{len(etapas):,}", ins, min_dist,
    )
    pico, _, exitcode = _correr_midiendo(
        _hijo_destinos, (dia, min_dist, etapas, metadata_lineas, matriz_val, modos_ramal),
        "infer_destinations",
    )
    return ins, pico, exitcode


def medir_tiempos(data, insumos, dia, next_dia):
    import h3
    from urbantrips.utils.utils import (
        leer_configs_generales, modos_con_ramal, RAMAL_SENTINEL,
    )

    configs = leer_configs_generales(autogenerado=False)
    legs_h3_res = configs["resolucion_h3"]

    # `distance_od` la agrega compute_od_distances, que escribe en su cache DuckDB.
    # Para no tocar la base se lee la ya calculada de travel_times_legs: el objetivo
    # es medir RAM, y la columna pesa lo mismo venga de donde venga.
    legs_all = data.query(
        f"""
        SELECT e.*, t.distance_od
        FROM etapas e
        LEFT JOIN travel_times_legs t ON t.id = e.id AND t.dia = e.dia
        WHERE e.etapa_validada = 1 AND e.dia = '{dia}'
        ORDER BY e.dia, e.id_tarjeta, e.id_viaje, e.id_etapa, e.id_linea, e.id_ramal, e.interno
        """
    )
    gps_dias = ", ".join(f"'{d}'" for d in [dia] + ([next_dia] if next_dia else []))
    gps = data.query(
        f"SELECT g.* FROM gps g WHERE g.dia IN ({gps_dias}) "
        "ORDER BY dia, id_linea, id_ramal, interno, fecha"
    )
    legs_to_gps_o = data.query(
        f"SELECT lo.id_legs, lo.id_gps AS id_gps_o FROM legs_to_gps_origin lo "
        f"WHERE lo.dia = '{dia}'"
    )

    metadata_lineas = insumos.get_metadata_lineas()[["id_linea", "id_linea_agg", "modo"]]
    mv = insumos.get_matrix_validation()
    matriz = mv[["id_linea_agg", "id_ramal", "parada", "area_influencia"]].drop_duplicates()
    matriz["id_ramal"] = matriz["id_ramal"].fillna(RAMAL_SENTINEL).astype("int64")
    matriz["ring"] = matriz.apply(
        lambda row: h3.grid_distance(row.parada, row.area_influencia), axis=1
    )
    lado_m = h3.average_hexagon_edge_length(res=legs_h3_res, unit="m")
    ring_max = max(1, round(configs.get("tolerancia_destino_gps", 1000) / (lado_m * 2)))
    matriz = matriz[matriz.ring <= ring_max]

    ins = _mem_gb(legs_all, gps, legs_to_gps_o, metadata_lineas, matriz)
    logger.info(
        "assign_time_distances — insumos del %s: %s etapas + %s gps, %.2f GB en pandas",
        dia, f"{len(legs_all):,}", f"{len(gps):,}", ins,
    )
    pico, _, exitcode = _correr_midiendo(
        _hijo_tiempos,
        ((dia, next_dia, legs_all, gps, legs_to_gps_o, metadata_lineas, matriz,
          modos_con_ramal(configs), legs_h3_res),),
        "assign_time_distances",
    )
    return ins, pico, exitcode


def medir_chains(data, insumos, dia):
    from urbantrips.datamodel.legs import _insumos_bytes
    from urbantrips.preparo_dashboard.chains import (
        RES_CHAINS_NORM, _CHAINS_COPY_AMPLIFICATION, _chains_tablas_footprint,
        _fetch_chains_inputs_dia,
    )

    ctx = SimpleNamespace(data=data, insumos=insumos)

    # mismos argumentos que preparo_dashboard: res=RES_CHAINS_NORM, max_etapas=3
    lineas = insumos.query("SELECT id_linea, nombre_linea FROM metadata_lineas")
    inp = _fetch_chains_inputs_dia(ctx, dia, lineas)
    if inp is None:
        raise SystemExit(f"chains: el día {dia} no tiene etapas od_validado en dias_ultima_corrida")
    etapas, viajes = inp

    # la misma base que multiplica el autotune (sin factor, piso ni override)
    fijas, del_dia, _ = _insumos_bytes(ctx, [dia], _chains_tablas_footprint())
    estimados = (fijas + del_dia) / GB
    logger.info(
        "chains — insumos del %s: %s etapas + %s viajes, %.2f GB en pandas, "
        "%.2f GB según el autotune (factor actual %.1f)",
        dia, f"{len(etapas):,}", f"{len(viajes):,}", _mem_gb(etapas, viajes),
        estimados, _CHAINS_COPY_AMPLIFICATION,
    )
    pico, _, exitcode = _correr_midiendo(
        _hijo_chains, (etapas, viajes, RES_CHAINS_NORM, 3), "chains"
    )
    return estimados, pico, exitcode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alias", help="alias de la base (se abre READ ONLY)")
    ap.add_argument(
        "--dia",
        help="día a medir (default: el de más etapas; en chains, el de más etapas "
             "od_validado de la última corrida)",
    )
    ap.add_argument(
        "--etapa", choices=["destinos", "tiempos", "chains", "ambas"], default="ambas",
        help="'ambas' = destinos + tiempos (comparten _PANDAS_COPY_AMPLIFICATION); "
             "chains tiene su propio factor y se mide aparte",
    )
    args = ap.parse_args()

    data, insumos = _abrir(args.alias)
    try:
        if args.dia:
            dia = args.dia
            n = int(data.query(
                f"SELECT count(*) AS n FROM etapas WHERE dia = '{dia}'"
            )["n"].iloc[0])
        elif args.etapa == "chains":
            dia, n = _dia_mas_grande_chains(data)
        else:
            dia, n = _dia_mas_grande(data)
        siguientes = data.query(
            f"SELECT min(dia) AS d FROM etapas WHERE dia > '{dia}'"
        )["d"]
        next_dia = None if siguientes.empty or pd.isna(siguientes.iloc[0]) else siguientes.iloc[0]
        logger.info("Día medido: %s (%s etapas) — siguiente: %s", dia, f"{n:,}", next_dia)

        filas = []
        if args.etapa in ("destinos", "ambas"):
            filas.append(("infer_destinations",) + medir_destinos(data, insumos, dia))
        if args.etapa in ("tiempos", "ambas"):
            filas.append(("assign_time_distances",) + medir_tiempos(data, insumos, dia, next_dia))
        if args.etapa == "chains":
            filas.append(("chains",) + medir_chains(data, insumos, dia))
    finally:
        data.close()
        insumos.close()

    print()
    print(f"{'etapa':26s} {'insumos GB':>11s} {'pico worker GB':>15s} {'factor':>8s}")
    invalidas = False
    for nombre, ins, pico, exitcode in filas:
        if exitcode == 0:
            factor = f"{pico / max(ins, 1e-9):8.2f}"
        else:
            # el hijo murió (p.ej. sin RAM): el pico es parcial y el factor no sirve
            factor = f"{'INVÁLIDO':>8s}"
            invalidas = True
        print(f"{nombre:26s} {ins:11.2f} {pico:15.2f} {factor}")
    print()
    if invalidas:
        print("Alguna medición no terminó (EXITCODE != 0, ver log): su pico es parcial.")
        print("No usar ese factor; liberar RAM y volver a medir.")
        print()
    if args.etapa == "chains":
        print("El factor es lo que hay que poner en _CHAINS_COPY_AMPLIFICATION")
        print("(preparo_dashboard/chains.py). Los insumos son los que estima el")
        print("autotune, así que el factor se aplica tal cual. Dejar algo de margen.")
    else:
        print("El factor es lo que hay que poner en _PANDAS_COPY_AMPLIFICATION")
        print("(legs.py). Tomar el MAYOR de destinos y tiempos: comparten ese")
        print("multiplicador. chains tiene el suyo (--etapa chains).")


if __name__ == "__main__":
    main()
