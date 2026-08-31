"""A/B del ingest de gps: versión anterior (todo el csv en un DataFrame) contra la
nueva (chunks a `gps_raw` + dedup/id/odómetro en SQL).

Corre las DOS implementaciones sobre el MISMO csv real, cada una contra su propia base
DuckDB, y compara fila por fila la tabla `gps`, la de factores de expansión y —lo que
cierra el argumento— los frames que cada versión le entrega a `compute_distance_km_gps`,
que es donde termina el camino que se cambió.

El csv de entrada se arma concatenando N días de AMBA en UN SOLO archivo: es la forma
en que corre el cliente (una corrida = un mes) y la que ninguna de nuestras corridas
ejercita, porque nosotros partimos el mes en 28 corridas de un día.

La versión anterior NO se copia acá: se carga el módulo entero tal como está en el
commit que se le pasa (`--ref`, por defecto HEAD), así que no envejece con el código.

Uso:
    python tools/ab_gps_ingest.py --dias 3
    python tools/ab_gps_ingest.py --dias 3 --chunk 500000 --ref HEAD
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter  # noqa: E402
from urbantrips.storage.context import StorageContext  # noqa: E402
from urbantrips.utils.utils import leer_configs_generales  # noqa: E402

DATOS = RAIZ / "data" / "data_ciudad"

# Dos ciudades porque ejercitan las dos ramas del config: AMBA tiene ramales, odómetro
# acumulado e id_servicio; Mendoza no tiene ninguno de los tres (id_ramal se copia de
# id_linea, el dedup no lo incluye y no hay odómetro que calcular).
CIUDADES = {
    "amba": {
        "config": "configuraciones_generales_2026mescompleto.yaml",
        "archivos": [f"mes_completo_2026{m:02d}{d:02d}_gps.csv"
                     for m, tope in ((3, 31), (4, 30)) for d in range(1, tope + 1)],
        # bbox que loguea la corrida del cliente, sin el buffer de 0.009*30 que le
        # agrega eliminar_trx_fuera_bbox. Se inyecta por config para que las dos
        # versiones filtren igual sin depender de la tabla de zonificaciones.
        "bbox": {
            "minx": -60.27343 + 0.27,
            "miny": -35.70824 + 0.27,
            "maxx": -57.39263 - 0.27,
            "maxy": -33.70926 - 0.27,
        },
    },
    "mza": {
        "config": "configuraciones_generales_mza.yaml",
        "archivos": [f"mza_{d}_gps.csv" for d in
                     ("lunes", "martes", "miércoles", "jueves", "viernes",
                      "sábado", "domingo")],
        "bbox": None,  # el config de mza ya trae filtro_latlong_bbox
    },
}


def dias_disponibles(ciudad):
    return [DATOS / n for n in CIUDADES[ciudad]["archivos"] if (DATOS / n).exists()]


def armar_csv_multidia(rutas, destino):
    """Concatena varios _gps.csv diarios en uno solo, como el archivo del cliente."""
    with open(destino, "w", encoding="utf-8", newline="") as salida:
        for i, ruta in enumerate(rutas):
            with open(ruta, "r", encoding="utf-8") as f:
                encabezado = f.readline()
                if i == 0:
                    salida.write(encabezado)
                for linea in f:
                    salida.write(linea)
    return destino


def cargar_modulo_de_commit(ref: str, destino: Path):
    """Carga urbantrips/datamodel/transactions.py tal como está en `ref`."""
    fuente = subprocess.run(
        ["git", "show", f"{ref}:urbantrips/datamodel/transactions.py"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    destino.write_text(fuente, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("transactions_ref", destino)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["transactions_ref"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def configs_ab(ciudad):
    ficha = CIUDADES[ciudad]
    cfg = dict(leer_configs_generales_desde(RAIZ / "configs" / ficha["config"]))
    if ficha["bbox"]:
        cfg["filtro_latlong_bbox"] = ficha["bbox"]
    return cfg


def leer_configs_generales_desde(ruta):
    import yaml

    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class InsumosSinZonas:
    """Fuerza a las dos versiones por la rama del bbox del config."""

    def get_zones(self):
        return gpd.GeoDataFrame(geometry=[])


def correr(modulo, csv_path: Path, base: Path, cfg, dias, chunk=None,
           solo_medir=False, duckdb_mem=None):
    """Corre una implementación y devuelve (tabla gps, factores, frames por día)."""
    adapter = DuckDBDataAdapter(base, memory_limit=duckdb_mem)
    adapter.save_run_days(pd.DataFrame({"dia": dias}))
    ctx = StorageContext(
        data=adapter, insumos=InsumosSinZonas(), dash=None, general=None
    )

    # Con --solo no se guardan los frames: la medición de RSS es sobre el ingest, no
    # sobre el arnés (quedarse los 28 días del mes sería más memoria que el ingest).
    capturados = []
    guardar_frames = not solo_medir

    # Stub de compute_distance_km_gps. Replica el `sort_values(...).reset_index()` que
    # hace la función real —es lo que fija el orden en que queda guardada la tabla, y
    # tiene que entrar en la comparación— y saltea sólo el cálculo de distancias de red,
    # que necesita las matrices de insumos y es código que este cambio no toca.
    orden_cols = (
        ["dia", "id_linea", "id_ramal", "interno", "fecha"]
        if cfg["lineas_contienen_ramales"]
        else ["dia", "id_linea", "interno", "fecha"]
    )

    def capturar(df, ctx=None):
        if guardar_frames:
            capturados.append(df.copy())
        return (
            df.sort_values(orden_cols).reset_index(drop=True).assign(distance_km=np.nan)
        )

    paths = SimpleNamespace(input_dir=csv_path.parent)
    parches = [
        patch.object(modulo, "leer_configs_generales", return_value=cfg),
        patch.object(modulo, "get_paths", return_value=paths),
        # `new=` y NO `side_effect=`: con side_effect, patch crea un MagicMock que
        # guarda cada llamada en call_args_list, o sea que se queda con una
        # referencia al frame de CADA día. Medido: 2 M bloques vivos retenidos por
        # llamada. Con --solo eso falsea el pico y hace parecer que el loop por día
        # tiene una fuga que no tiene.
        patch.object(modulo, "compute_distance_km_gps", capturar),
        patch(
            "urbantrips.utils.utils.leer_configs_generales",
            return_value=cfg,
        ),
    ]
    if chunk is not None and hasattr(modulo, "_GPS_CHUNK_ROWS"):
        parches.append(patch.object(modulo, "_GPS_CHUNK_ROWS", chunk))

    t0 = time.time()
    for p in parches:
        p.start()
    try:
        modulo.process_and_upload_gps_table(
            ctx=ctx,
            nombre_archivo_gps=csv_path.name,
            nombres_variables_gps=dict(cfg["nombres_variables_gps"]),
            formato_fecha=cfg["formato_fecha"],
        )
    finally:
        for p in reversed(parches):
            p.stop()
    segundos = time.time() - t0

    # Sin ORDER BY: DuckDB devuelve el orden de inserción, así que esto compara
    # también en qué orden quedó guardada la tabla, no solo su contenido.
    if solo_medir:
        # Bajar la tabla entera a pandas para compararla es más memoria que el propio
        # ingest y falsearía el pico. Midiendo solo se cuentan las filas.
        n = adapter.query("SELECT COUNT(*) AS n FROM gps")["n"].iloc[0]
        return pd.DataFrame(index=range(int(n))), None, capturados, segundos

    gps = adapter.query("SELECT * FROM gps")
    veh = adapter.query(
        "SELECT * FROM vehicle_expansion_factors ORDER BY id_linea, dia"
    )
    return gps, veh, capturados, segundos


def comparar_frames(nombre, a, b):
    """Compara dos DataFrames fila por fila. Devuelve True si son idénticos."""
    if list(a.columns) != list(b.columns):
        print(f"  {nombre}: DIFIEREN las columnas\n    A={list(a.columns)}\n    B={list(b.columns)}")
        return False
    if len(a) != len(b):
        print(f"  {nombre}: DIFIERE la cantidad de filas: {len(a):,} vs {len(b):,}")
        return False

    ok = True
    for col in a.columns:
        sa, sb = a[col].reset_index(drop=True), b[col].reset_index(drop=True)
        if sa.equals(sb):
            continue
        # Diferencias de REPRESENTACIÓN con los mismos valores. Son esperables en los
        # frames intermedios: la versión nueva trae del staging como texto las dos
        # columnas de id que la tabla gps guarda en TEXT, y como DOUBLE las que guarda
        # en FLOAT. Lo que importa es que la tabla `gps` final coincida, y eso se
        # compara aparte.
        try:
            if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
                if np.array_equal(
                    sa.to_numpy(dtype="float64"),
                    sb.to_numpy(dtype="float64"),
                    equal_nan=True,
                ):
                    print(f"  {nombre}.{col}: mismos valores, dtype {sa.dtype} vs {sb.dtype}")
                    continue
        except (TypeError, ValueError):
            pass
        if pd.api.types.is_object_dtype(sa) != pd.api.types.is_object_dtype(sb):
            ta = sa.astype("string")
            tb = sb.astype("string")
            if ta.equals(tb):
                print(
                    f"  {nombre}.{col}: mismos valores como texto, "
                    f"dtype {sa.dtype} vs {sb.dtype}"
                )
                continue
        distintas = (sa != sb) & ~(sa.isna() & sb.isna())
        print(f"  {nombre}.{col}: DIFIERE en {int(distintas.sum()):,} de {len(sa):,} filas")
        muestra = sa[distintas].head(3).tolist(), sb[distintas].head(3).tolist()
        print(f"    A={muestra[0]}  B={muestra[1]}")
        ok = False
    if ok:
        print(f"  {nombre}: idéntico ({len(a):,} filas x {len(a.columns)} columnas)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ciudad", choices=sorted(CIUDADES), default="amba",
        help="amba ejercita ramales + odómetro; mza, la rama sin ramales ni odómetro",
    )
    ap.add_argument("--dias", type=int, default=3, help="cuántos días concatenar")
    ap.add_argument("--chunk", type=int, default=None, help="filas por chunk de lectura")
    ap.add_argument("--ref", default="HEAD", help="commit con la versión de referencia")
    ap.add_argument(
        "--duckdb-mem",
        default=None,
        help="memory_limit de DuckDB, ej 2GB. Fijarlo hace comparable la medición "
        "y simula una máquina apretada: al tope, DuckDB derrama a disco.",
    )
    ap.add_argument(
        "--solo",
        choices=["ref", "nueva"],
        help="corre una sola versión y reporta su pico de RSS (para medir memoria hay "
        "que usar un proceso por versión: el allocator no le devuelve la RAM al SO)",
    )
    args = ap.parse_args()

    rutas = dias_disponibles(args.ciudad)[: args.dias]
    if len(rutas) < args.dias:
        sys.exit(f"Solo hay {len(rutas)} días de gps de {args.ciudad} en {DATOS}")

    cfg = configs_ab(args.ciudad)
    with tempfile.TemporaryDirectory(prefix="ab_gps_") as tmp:
        tmp = Path(tmp)
        csv_path = tmp / "ab_gps.csv"
        print(f"Armando csv multi-día con {len(rutas)} días…")
        armar_csv_multidia(rutas, csv_path)
        print(f"  {csv_path.stat().st_size / 2**30:.2f} GB")

        col_fecha = cfg["nombres_variables_gps"]["fecha_gps"]
        dias = sorted(
            pd.to_datetime(
                pd.read_csv(csv_path, usecols=[col_fecha])[col_fecha],
                format=cfg["formato_fecha"],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
            .dropna()
            .unique()
        )
        print(f"  días: {dias}")

        modulo_ref = cargar_modulo_de_commit(args.ref, tmp / "transactions_ref.py")
        import urbantrips.datamodel.transactions as modulo_nuevo

        if args.solo:
            import threading

            import psutil

            proc = psutil.Process()
            pico = [0.0]
            vivo = [True]

            def muestrear():
                while vivo[0]:
                    pico[0] = max(pico[0], proc.memory_info().rss / 2**30)
                    time.sleep(0.01)

            serie = []

            def muestrear2():
                while vivo[0]:
                    serie.append(
                        (time.time(), proc.memory_info().rss / 2**30)
                    )
                    time.sleep(1.0)

            threading.Thread(target=muestrear, daemon=True).start()
            threading.Thread(target=muestrear2, daemon=True).start()
            base = proc.memory_info().rss / 2**30
            modulo = modulo_ref if args.solo == "ref" else modulo_nuevo
            gps, _, _, segundos = correr(
                modulo,
                csv_path,
                tmp / "solo.duckdb",
                cfg,
                dias,
                chunk=args.chunk,
                solo_medir=True,
                duckdb_mem=args.duckdb_mem,
            )
            vivo[0] = False
            if serie:
                t0s = serie[0][0]
                print("\n  traza de RSS (s, GB):")
                paso = max(1, len(serie) // 30)
                print(
                    "   "
                    + "  ".join(
                        f"{int(t - t0s)}s:{r:.1f}" for t, r in serie[::paso]
                    )
                )
            print(
                f"\n{args.solo}: {len(gps):,} filas — {segundos:.1f}s — "
                f"pico RSS +{pico[0] - base:.2f} GB sobre {len(dias)} días"
            )
            return 0

        print(f"\nCorriendo versión de referencia ({args.ref})…")
        gps_a, veh_a, frames_a, t_a = correr(
            modulo_ref, csv_path, tmp / "a.duckdb", cfg, dias,
            duckdb_mem=args.duckdb_mem
        )
        print(f"  {len(gps_a):,} filas en gps — {t_a:.1f}s")

        print("\nCorriendo versión nueva…")
        gps_b, veh_b, frames_b, t_b = correr(
            modulo_nuevo, csv_path, tmp / "b.duckdb", cfg, dias, chunk=args.chunk,
            duckdb_mem=args.duckdb_mem
        )
        print(f"  {len(gps_b):,} filas en gps — {t_b:.1f}s")

        print("\nComparación:")
        ok = True
        # Contenido, fila por fila, en orden de id
        ok &= comparar_frames(
            "gps (contenido)",
            gps_a.sort_values("id").reset_index(drop=True),
            gps_b.sort_values("id").reset_index(drop=True),
        )
        # Orden físico DENTRO de cada día (gps_a/gps_b vienen sin ORDER BY, o sea en
        # orden de inserción). El orden en que se insertan los días ENTRE sí puede
        # diferir —la versión de referencia recorría los días en el orden en que
        # aparecían en el frame ya ordenado por (id_linea, id_ramal, interno, fecha), y
        # la nueva en orden de archivo— y no es un resultado: ninguna consulta del
        # pipeline depende del orden físico de la tabla.
        dias_a = list(dict.fromkeys(gps_a["dia"]))
        dias_b = list(dict.fromkeys(gps_b["dia"]))
        if dias_a != dias_b:
            print(
                "  gps: los días se insertan en distinto orden"
                f"\n    A={dias_a}\n    B={dias_b}"
            )
        for d in dias_a:
            ok &= comparar_frames(
                f"gps (orden dentro de {d})",
                gps_a[gps_a["dia"] == d].reset_index(drop=True),
                gps_b[gps_b["dia"] == d].reset_index(drop=True),
            )
        ok &= comparar_frames("vehicle_expansion_factors", veh_a, veh_b)

        if len(frames_a) != len(frames_b):
            print(f"  frames por día: DIFIEREN {len(frames_a)} vs {len(frames_b)}")
            ok = False
        else:
            for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
                cols = [c for c in fa.columns if c in fb.columns]
                # Ordenados por id: el frame que recibe compute_distance_km_gps puede
                # venir en distinto orden (la versión de referencia reordenaba el frame
                # entero antes del loop por día), pero eso da lo mismo porque
                # compute_distance_km_gps lo re-ordena por
                # (dia, id_linea, id_ramal, interno, fecha) apenas entra. Que el orden
                # final coincida lo verifica la comparación de la tabla `gps`, que se
                # hace sin ORDER BY.
                ok &= comparar_frames(
                    f"frame día {i} -> compute_distance_km_gps",
                    fa[cols].sort_values("id").reset_index(drop=True),
                    fb[cols].sort_values("id").reset_index(drop=True),
                )

        print(f"\n{'IDÉNTICO' if ok else 'HAY DIFERENCIAS'}   (ref {t_a:.1f}s / nueva {t_b:.1f}s)")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
