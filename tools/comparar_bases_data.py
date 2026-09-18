"""Compara dos bases `_data.duckdb` sobre los mismos días, tabla por tabla.

Pensado para validar un cambio de código: se corre el mismo período con las dos versiones
y esto responde si el resultado es idéntico. Todo se resuelve en SQL —las tablas de una
semana de AMBA son decenas de millones de filas y no entran en pandas—, y las dos bases se
abren **READ_ONLY**, así que es seguro apuntarle a una base validada.

Primero un `EXCEPT` en los dos sentidos sobre todas las columnas comunes: si da 0 y 0, las
tablas son idénticas como conjunto de filas y no hace falta mirar más. Si da distinto de 0,
recién ahí desglosa por columna, emparejando por clave natural (`dia, id_tarjeta, id_viaje,
id_etapa` en etapas, y su equivalente en cada tabla) y no por el id sustituto.

Uso:
    python tools/comparar_bases_data.py --a verif_semana1_20260828 \\
        --b marzo_mes_completo_2026_fix --dias 2026-03-09,...,2026-03-15
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb

RAIZ = Path(__file__).resolve().parents[1]

# Clave natural por tabla: la que identifica la fila por su contenido, no por el id
# sustituto que se reasigna en cada corrida.
CLAVES = {
    "etapas": ["dia", "id_tarjeta", "id_viaje", "id_etapa"],
    "viajes": ["dia", "id_tarjeta", "id_viaje"],
    "usuarios": ["dia", "id_tarjeta"],
    # OJO: tiene que ser el subset del drop_duplicates del ingest, que incluye
    # latitud/longitud. Sin ellas la clave NO es única —dos pings del mismo vehículo en
    # el mismo segundo con coordenadas distintas sobreviven al dedup—, el join se
    # multiplica y el desglose por columna da diferencias que no existen.
    "gps": ["dia", "id_linea", "id_ramal", "interno", "fecha", "latitud", "longitud"],
    "transacciones": ["dia", "id_tarjeta", "id_original"],
    "vehicle_expansion_factors": ["dia", "id_linea"],
}


# OJO: al hacer ATTACH ... AS a, las tablas quedan en el CATALOGO `a`, esquema `main`.
# Filtrar por table_schema='a' no devuelve nada y hace que se saltee todo en silencio
# (o sea, un "IDÉNTICAS" falso). Va por table_catalog.
def columnas(con, catalogo, tabla):
    filas = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_catalog = ? AND table_name = ? ORDER BY ordinal_position",
        [catalogo, tabla],
    ).fetchall()
    return [f[0] for f in filas]


def existe(con, catalogo, tabla):
    return bool(
        con.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_catalog = ? AND table_name = ?",
            [catalogo, tabla],
        ).fetchone()
    )


def comparar_tabla(con, tabla, dias_sql, solo_conteos=False):
    if not (existe(con, "a", tabla) and existe(con, "b", tabla)):
        print(f"  {tabla:28s} — no está en las dos bases, se saltea")
        return None

    cols_a, cols_b = columnas(con, "a", tabla), columnas(con, "b", tabla)
    comunes = [c for c in cols_a if c in cols_b]
    faltan = set(cols_a) ^ set(cols_b)
    if faltan:
        print(f"  {tabla:28s} — OJO, columnas distintas: {sorted(faltan)}")

    tiene_dia = "dia" in comunes
    where = f" WHERE dia IN ({dias_sql})" if tiene_dia else ""

    n_a = con.execute(f"SELECT COUNT(*) FROM a.{tabla}{where}").fetchone()[0]
    n_b = con.execute(f"SELECT COUNT(*) FROM b.{tabla}{where}").fetchone()[0]
    if n_a != n_b:
        print(f"  {tabla:28s} DIFIERE la cantidad de filas: {n_a:,} vs {n_b:,}")
        return False

    sel = ", ".join(comunes)
    sql_a = f"SELECT {sel} FROM a.{tabla}{where}"
    sql_b = f"SELECT {sel} FROM b.{tabla}{where}"
    solo_a = con.execute(f"SELECT COUNT(*) FROM ({sql_a} EXCEPT {sql_b})").fetchone()[0]
    solo_b = con.execute(f"SELECT COUNT(*) FROM ({sql_b} EXCEPT {sql_a})").fetchone()[0]

    if solo_a == 0 and solo_b == 0:
        print(f"  {tabla:28s} idéntica ({n_a:,} filas x {len(comunes)} columnas)")
        return True

    print(
        f"  {tabla:28s} DIFIERE: {solo_a:,} filas solo en A, {solo_b:,} solo en B "
        f"(de {n_a:,})"
    )
    if solo_conteos:
        return False

    # desglose por columna, emparejando por clave natural
    clave = [c for c in CLAVES.get(tabla, []) if c in comunes]
    if not clave:
        print(f"    (sin clave natural definida para {tabla}, no se desglosa)")
        return False
    on = " AND ".join(f"x.{c} IS NOT DISTINCT FROM y.{c}" for c in clave)
    print(f"    desglose por columna (clave: {', '.join(clave)}):")
    for col in comunes:
        if col in clave:
            continue
        n = con.execute(
            f"SELECT COUNT(*) FROM ({sql_a}) x JOIN ({sql_b}) y ON {on} "
            f"WHERE x.{col} IS DISTINCT FROM y.{col}"
        ).fetchone()[0]
        if n:
            print(f"      {col:32s} {n:>14,} filas distintas")
    return False


def _temp_dir() -> Path:
    """Dónde spillea DuckDB cuando la comparación no entra en `--memoria`.

    OBLIGATORIO en una conexión in-memory: sin esto el default de DuckDB 1.5.3 es
    `temp_directory='.tmp'` RELATIVO AL CWD, o sea adentro del repo si se corre desde la
    raíz (una conexión sobre archivo no tiene el problema: usa `<base>.duckdb.tmp`). Y acá
    se spillea mucho: comparar `etapas` a escala mes son 253,9 M filas contra otras tantas
    con un techo de 12 GB. El 2026-08-29 dejó 54 GB en `UrbanTrips/.tmp/`, se comieron el
    disco y aparecían en `git status`.

    Directorio propio por PID porque DuckDB NO mete el PID en el nombre del archivo de
    spill (`duckdb_temp_storage_DEFAULT-0.tmp`): compartir el temp con una corrida viva
    sería pisarse. Se borra al salir, y si el proceso muere de golpe lo que queda cae en el
    temp del sistema, no en el repo.
    """
    d = Path(tempfile.gettempdir()) / f"urbantrips_comparar_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    atexit.register(shutil.rmtree, d, True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="alias de la base nueva")
    ap.add_argument("--b", required=True, help="alias de la base de referencia")
    ap.add_argument("--dias", required=True, help="días a comparar, separados por coma")
    ap.add_argument("--tablas", default=None, help="subconjunto de tablas, separadas por coma")
    ap.add_argument("--solo-conteos", action="store_true", help="no desglosa por columna")
    ap.add_argument("--memoria", default="12GB")
    args = ap.parse_args()

    db = RAIZ / "data" / "db"
    ruta_a, ruta_b = db / f"{args.a}_data.duckdb", db / f"{args.b}_data.duckdb"
    for r in (ruta_a, ruta_b):
        if not r.exists():
            sys.exit(f"No existe {r}")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memoria}'")
    con.execute(f"SET temp_directory='{_temp_dir().as_posix()}'")
    # READ_ONLY en las dos: la base de referencia no se toca ni por accidente
    con.execute(f"ATTACH '{ruta_a.as_posix()}' AS a (READ_ONLY)")
    con.execute(f"ATTACH '{ruta_b.as_posix()}' AS b (READ_ONLY)")

    dias = [d.strip() for d in args.dias.split(",") if d.strip()]
    dias_sql = ", ".join(f"'{d}'" for d in dias)
    print(f"A = {args.a}\nB = {args.b}\ndías = {len(dias)} ({dias[0]} … {dias[-1]})\n")

    tablas = (
        [t.strip() for t in args.tablas.split(",")]
        if args.tablas
        else ["gps", "transacciones", "etapas", "viajes", "usuarios",
              "vehicle_expansion_factors"]
    )
    resultados = [comparar_tabla(con, t, dias_sql, args.solo_conteos) for t in tablas]
    comparadas = [r for r in resultados if r is not None]

    # Si no se comparó NADA, esto no es un OK: es que no se encontraron las tablas.
    # Pasó en el ensayo de este script (filtraba por esquema en vez de por catálogo) y
    # reportó "IDÉNTICAS" habiendo salteado todo.
    if not comparadas:
        print("\nNO SE COMPARÓ NINGUNA TABLA — revisar nombres de base y de tablas")
        return 2

    todo_ok = all(comparadas)
    print(
        f"\n{'IDÉNTICAS' if todo_ok else 'HAY DIFERENCIAS'} "
        f"({len(comparadas)} de {len(tablas)} tablas comparadas)"
    )
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
