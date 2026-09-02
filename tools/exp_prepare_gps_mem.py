"""Experimento OOM ronda 4 (2026-09-02): prepare viejo (sentencia unica) vs nuevo
(etapas) sobre el MISMO gps_raw del mes, con memory_limit apretado.

A diferencia de ab_gps_ingest.py, aca la variante vieja SI es una copia congelada
(_sql_viejo): la sentencia unica de prepare_gps_from_raw nunca llego a commitearse
(la ronda 3 se reemplazo antes del commit), asi que no hay ref de git de donde
cargarla y esta copia es su unico registro.

Medido 2026-09-02 sobre los 28 dias de AMBA (103,6 M filas en gps_raw, duckdb
1.5.3, checksums identicos): old/new @ 4GB: 615 s / 330 s; @ 2GB: 1102 s / 364 s;
@ 1GB fallan las dos con OutOfMemoryException.

Uso (un proceso por variante; el allocator no devuelve RAM al SO):
    python tools/exp_prepare_gps_mem.py build     # arma el staging UNA vez (codigo real)
    python tools/exp_prepare_gps_mem.py old 4GB   # sentencia unica sobre el staging
    python tools/exp_prepare_gps_mem.py new 4GB   # prepare_gps_from_raw sobre el staging

`build` corre process_and_upload_gps_table de verdad (chunks reales, bbox, NAs,
factores) y aborta justo cuando llega a prepare_gps_from_raw, dejando gps_raw
persistido en un directorio de trabajo bajo el temp del sistema (~15 GB entre csv
y base; borrarlo a mano al terminar). Las variantes abren esa base, corren solo el
paso de preparacion y reportan filas, tiempo, pico de RSS y un checksum.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
SCRATCH = Path(tempfile.gettempdir()) / "exp_prepare_gps_mem"
SCRATCH.mkdir(parents=True, exist_ok=True)
DB = SCRATCH / "exp_gps_raw_mes.duckdb"
CSV = SCRATCH / "exp_mes_gps.csv"

# los 28 dias del csv (verificados en la corrida del arnes)
DIAS = (
    [f"2026-03-{d:02d}" for d in range(9, 32)]
    + [f"2026-04-{d:02d}" for d in range(1, 6)]
)

SUBSET_DEDUP = ["dia", "id_linea", "id_ramal", "interno", "fecha", "latitud", "longitud"]
ODOMETRO = "diff"  # amba: hay distance_servicio_mts_agg


def _cargar_arnes():
    spec = importlib.util.spec_from_file_location(
        "ab_gps_ingest", RAIZ / "tools" / "ab_gps_ingest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ab_gps_ingest"] = mod
    spec.loader.exec_module(mod)
    return mod


class _StagingListo(Exception):
    pass


def build():
    ab = _cargar_arnes()
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.context import StorageContext
    import urbantrips.datamodel.transactions as trx
    import pandas as pd

    if DB.exists():
        DB.unlink()
    wal = DB.with_suffix(".duckdb.wal")
    if wal.exists():
        wal.unlink()

    if not CSV.exists():
        rutas = ab.dias_disponibles("amba")[:28]
        assert len(rutas) == 28, f"solo hay {len(rutas)} dias"
        print("Armando csv multi-dia...", flush=True)
        ab.armar_csv_multidia(rutas, CSV)
    print(f"csv: {CSV.stat().st_size / 2**30:.2f} GB", flush=True)

    cfg = ab.configs_ab("amba")
    adapter = DuckDBDataAdapter(DB, memory_limit="9GB")
    adapter.save_run_days(pd.DataFrame({"dia": DIAS}))
    ctx = StorageContext(
        data=adapter, insumos=ab.InsumosSinZonas(), dash=None, general=None
    )

    def _frenar(self, *a, **kw):
        raise _StagingListo()

    paths = SimpleNamespace(input_dir=CSV.parent)
    t0 = time.time()
    with (
        patch.object(trx, "leer_configs_generales", return_value=cfg),
        patch.object(trx, "get_paths", return_value=paths),
        patch("urbantrips.utils.utils.leer_configs_generales", return_value=cfg),
        patch.object(DuckDBDataAdapter, "prepare_gps_from_raw", _frenar),
    ):
        try:
            trx.process_and_upload_gps_table(
                ctx=ctx,
                nombre_archivo_gps=CSV.name,
                nombres_variables_gps=dict(cfg["nombres_variables_gps"]),
                formato_fecha=cfg["formato_fecha"],
            )
        except _StagingListo:
            pass
        else:
            sys.exit("no llego a prepare_gps_from_raw — revisar")
    n = adapter.query("SELECT COUNT(*) AS n FROM gps_raw")["n"].iloc[0]
    adapter.close()
    print(f"staging listo: {n:,} filas en gps_raw — {time.time() - t0:.0f}s", flush=True)


def _medir_rss():
    proc = psutil.Process()
    pico = [proc.memory_info().rss / 2**30]
    vivo = [True]

    def loop():
        while vivo[0]:
            pico[0] = max(pico[0], proc.memory_info().rss / 2**30)
            time.sleep(0.01)

    threading.Thread(target=loop, daemon=True).start()
    return pico, vivo


def _sql_viejo(particion: str, id_offset: int) -> str:
    """La sentencia unica tal como estaba antes del fix (odometro='diff')."""
    ventana = "PARTITION BY id_linea, id_ramal, interno ORDER BY fecha, orden_archivo"
    delta = (
        "distance_servicio_mts_agg - lag(distance_servicio_mts_agg) "
        f"OVER ({ventana})"
    )
    calculadas = (
        f"CASE WHEN ({delta}) IS NULL OR ({delta}) < 0 THEN 0 "
        f"ELSE ({delta}) END AS distance_servicio_mts, "
        "distance_servicio_mts_agg"
    )
    return f"""
        CREATE TABLE gps_prep AS
        WITH sin_duplicados AS (
            SELECT * FROM gps_raw
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY {particion} ORDER BY orden_archivo
            ) = 1
        ),
        con_id AS (
            SELECT
                (ROW_NUMBER() OVER (ORDER BY orden_archivo) - 1 + {id_offset})
                    AS id,
                *
            FROM sin_duplicados
        )
        SELECT
            id, orden_archivo, id_original, dia, id_linea, id_ramal, interno,
            fecha, latitud, longitud, velocity, id_servicio, service_type,
            {calculadas}
        FROM con_id
        """


def correr_variante(variante: str, mem: str):
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    adapter = DuckDBDataAdapter(DB, memory_limit=mem)
    for t in ("gps_prep", "_gps_prep_ids", "_gps_prep_odo"):
        adapter.execute(f"DROP TABLE IF EXISTS {t}")

    pico, vivo = _medir_rss()
    base = pico[0]
    t0 = time.time()
    try:
        if variante == "old":
            adapter.execute(_sql_viejo(", ".join(SUBSET_DEDUP), 0))
            filas = adapter.query("SELECT COUNT(*) AS n FROM gps_prep")["n"].iloc[0]
        else:
            filas = adapter.prepare_gps_from_raw(
                dedup_subset=SUBSET_DEDUP, id_offset=0, odometro=ODOMETRO
            )
    except Exception as e:
        vivo[0] = False
        print(
            f"{variante} @ {mem}: FALLO a los {time.time() - t0:.0f}s "
            f"(pico RSS +{pico[0] - base:.2f} GB)\n  {type(e).__name__}: {e}",
            flush=True,
        )
        return
    segundos = time.time() - t0
    vivo[0] = False
    # checksum barato para comparar variantes sin bajar la tabla a pandas
    chk = adapter.query(
        "SELECT SUM(id * 31 + orden_archivo) AS a, "
        "       ROUND(SUM(distance_servicio_mts), 3) AS b, "
        "       ROUND(SUM(distance_servicio_mts_agg), 3) AS c, "
        "       COUNT(DISTINCT dia) AS d FROM gps_prep"
    ).iloc[0]
    for t in ("gps_prep", "_gps_prep_ids", "_gps_prep_odo"):
        adapter.execute(f"DROP TABLE IF EXISTS {t}")
    adapter.close()
    print(
        f"{variante} @ {mem}: {int(filas):,} filas — {segundos:.0f}s — "
        f"pico RSS +{pico[0] - base:.2f} GB\n"
        f"  checksum: a={chk['a']} b={chk['b']} c={chk['c']} dias={chk['d']}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "old", "new"):
        sys.exit(__doc__)
    if sys.argv[1] == "build":
        build()
    else:
        correr_variante(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "4GB")
