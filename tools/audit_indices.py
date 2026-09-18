"""Auditoría empírica de índices ART (candidata D del plan de optimización).

Mide, sobre una COPIA de un backup de la base data, el tiempo de las formas de
query canónicas del pipeline CON los índices presentes y DESPUÉS de dropearlos.
Si los tiempos no cambian, el índice es puro costo de mantenimiento en las
escrituras y puede eliminarse del schema.

Uso (con la corrida del mes TERMINADA, máquina libre):

    conda activate trips6
    python tools/audit_indices.py <ruta_al_data.duckdb_de_backup> [copia_de_trabajo]

    # ejemplo:
    python tools/audit_indices.py data/db/bk_step2_legs_20260717/data.duckdb E:/tmp/audit_copy.duckdb

NUNCA apuntar a la base viva ni al backup directo: el script DROPea índices,
por eso trabaja sobre una copia (la crea si no existe; ~63GB, tarda unos min).

Salida: tabla comparativa con/sin índice por query. Interpretación: los scans
por `dia` deberían dar igual (los sirve el zonemap con la tabla day-clustered);
si alguna query empeora >2x sin el índice, ese índice se rescata y se documenta
su consumidor.
"""

import shutil
import sys
import time
from pathlib import Path

import duckdb

# Índices de storage/schema/data.py (mantener sincronizado a mano si cambia el schema)
INDEXES = [
    "idx_trx_batch",
    "idx_etapas_batch",
    "idx_etapas_id",
    "idx_gps_line_day",
    "idx_etapas_dia_od_validado",
    "idx_etapas_dia_line_ramal_interno",
    "idx_gps_dia_line_ramal_interno_fecha",
    "idx_travel_times_stations_id",
    "idx_services_stats_line_day",
]


def build_queries(con):
    """Formas de query canónicas del pipeline, parametrizadas con datos reales."""
    dia = con.execute("SELECT MIN(dia) FROM etapas").fetchone()[0]
    linea, ramal, interno = con.execute(
        f"SELECT id_linea, id_ramal, interno FROM etapas "
        f"WHERE dia='{dia}' AND id_linea IS NOT NULL LIMIT 1"
    ).fetchone()
    batch = con.execute("SELECT MIN(batch_id) FROM transacciones").fetchone()[0]

    q = {
        # create_trips / kpi / chains: scan de un día completo
        "etapas WHERE dia": (
            f"SELECT COUNT(*), SUM(factor_expansion_original) FROM etapas WHERE dia='{dia}'"
        ),
        # destinations / dashboard: día + od_validado
        "etapas dia+od_validado": (
            f"SELECT COUNT(*) FROM etapas WHERE dia='{dia}' AND od_validado=1"
        ),
        # kpi por línea: día + línea/ramal/interno
        "etapas dia+linea+ramal+int": (
            f"SELECT COUNT(*) FROM etapas WHERE dia='{dia}' AND id_linea={linea} "
            f"AND id_ramal IS NOT DISTINCT FROM {ramal} AND interno IS NOT DISTINCT FROM {interno}"
        ),
        # update_leg_destinations / rearrange: join masivo por id
        "etapas JOIN por id (100k)": (
            "SELECT COUNT(*) FROM etapas e JOIN "
            "(SELECT id FROM etapas USING SAMPLE 100000 ROWS) s USING (id)"
        ),
        # process_services / kpi: gps por día+línea
        "gps dia+linea": (
            f"SELECT COUNT(*) FROM gps WHERE dia='{dia}' AND id_linea={linea}"
        ),
        # assign_time_distances / kpi: gps del día ordenado (el ORDER BY que 'espeja' el índice 5-col)
        "gps dia ORDER BY 5col": (
            f"SELECT COUNT(*) FROM (SELECT * FROM gps WHERE dia='{dia}' "
            f"ORDER BY dia, id_linea, id_ramal, interno, fecha)"
        ),
        # standardize: trx por batch (el único caso con hipótesis de rescate)
        "trx WHERE batch_id": (
            f"SELECT COUNT(*) FROM transacciones WHERE batch_id={batch}"
        ),
        # trips.py:565: join travel_times por id
        "ttimes_gps JOIN por id (100k)": (
            "SELECT COUNT(*) FROM travel_times_gps t JOIN "
            "(SELECT id FROM travel_times_gps USING SAMPLE 100000 ROWS) s USING (id)"
        ),
    }
    return q


def bench(con, queries, label, reps=2):
    """Corre cada query `reps` veces y devuelve el mejor tiempo (mitiga cache frío)."""
    out = {}
    for name, sql in queries.items():
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            try:
                con.execute(sql).fetchall()
                dt = time.perf_counter() - t0
            except Exception as e:  # tabla ausente en este backup, etc.
                dt = None
                print(f"  [{label}] {name}: SKIP ({e})")
                break
            best = dt if best is None else min(best, dt)
        if best is not None:
            out[name] = best
            print(f"  [{label}] {name:32s} {best:8.2f}s")
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / "audit_indices_copy.duckdb"

    if not work.exists():
        print(f"Copiando {src} -> {work} (una vez, ~minutos)...")
        shutil.copy2(src, work)

    con = duckdb.connect(str(work))
    con.execute("SET memory_limit='17GB'")

    existing = {
        r[0] for r in con.execute(
            "SELECT index_name FROM duckdb_indexes()"
        ).fetchall()
    }
    print(f"Índices presentes en la copia: {sorted(existing & set(INDEXES))}\n")

    queries = build_queries(con)

    print("== CON índices ==")
    with_idx = bench(con, queries, "con")

    for idx in INDEXES:
        con.execute(f"DROP INDEX IF EXISTS {idx}")
    print("\n== SIN índices (todos dropeados) ==")
    without_idx = bench(con, queries, "sin")

    print("\n== RESUMEN (sin/con; >2.0 = el índice ayudaba a esa query) ==")
    for name in with_idx:
        if name in without_idx and with_idx[name] > 0:
            ratio = without_idx[name] / with_idx[name]
            flag = "  <-- RESCATAR índice asociado" if ratio > 2 else ""
            print(f"  {name:32s} con={with_idx[name]:7.2f}s  sin={without_idx[name]:7.2f}s  x{ratio:5.2f}{flag}")

    con.close()
    print(f"\nCopia de trabajo en {work} — borrarla al terminar (no es un backup).")


if __name__ == "__main__":
    main()
