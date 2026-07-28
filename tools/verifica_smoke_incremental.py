"""Verifica una corrida incremental: que dias_ultima_corrida quede SOLO con los días
nuevos, que los días previos se preserven en las tablas de datos, y que sus
indicadores no cambien de valor.

Reutilizable para cualquier incremento (run2 = 16/17, run3 = 23/24, etc.).

Qué prueba, y por qué importa:
  1. `dias_ultima_corrida` == solo los días nuevos del incremento.
     Es la premisa de los filtros por día de verificar_integridad/persist_indicators
     Y el síntoma directo del bug de run_process._ingest_all_days (que la llenaba con
     TODOS los días acumulados → Fase 3 reprocesaba los viejos).
  2. Los días previos SIGUEN en etapas/viajes/usuarios (rebuilds los arrastran).
  3. Los indicadores de los días previos sobreviven Y no cambian de valor
     (upsert por (dia, detalle, tabla) en _replace_indicator_rows).

Uso:
    # ANTES del incremento (captura el estado a preservar)
    python tools/verifica_smoke_incremental.py snapshot
    # DESPUÉS del incremento, pasando los días nuevos:
    python tools/verifica_smoke_incremental.py verify 2026-03-23,2026-03-24
"""
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

# Alias de la DB del smoke. Override con SMOKE_ALIAS para probar otra ciudad
# (p.ej. Mendoza) sin tocar el default histórico.
ALIAS = os.environ.get("SMOKE_ALIAS", "smoke_inc_2026")
DB = Path(__file__).resolve().parents[1] / "data" / "db" / f"{ALIAS}_data.duckdb"
SNAP = Path(__file__).resolve().parent / f".{ALIAS}_snapshot.parquet"
SNAP_DAYS = Path(__file__).resolve().parent / f".{ALIAS}_snapshot_days.txt"

ok = True


def check(cond, msg, detalle=""):
    global ok
    print(f"  {'OK  ' if cond else 'FALLA'}  {msg}")
    if not cond:
        ok = False
        if detalle:
            print(f"          {detalle}")


def conectar():
    if not DB.exists():
        sys.exit(f"No existe {DB} — ¿corrió alguna corrida con alias {ALIAS}?")
    return duckdb.connect(str(DB), read_only=True)


def dias_de(con, tabla):
    try:
        return sorted(r[0] for r in con.execute(
            f"SELECT DISTINCT dia FROM {tabla}").fetchall())
    except Exception as e:
        return f"<error: {e}>"


def indicadores(con):
    return con.execute(
        "SELECT dia, detalle, tabla, indicador FROM indicadores ORDER BY 1,2,3"
    ).df()


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else ""

    if modo == "snapshot":
        con = conectar()
        print(f"\n=== SNAPSHOT del estado actual ({DB.name}) ===")
        dias_previos = dias_de(con, "etapas")
        for t in ("etapas", "viajes", "usuarios"):
            print(f"  {t:12} días: {dias_de(con, t)}")
        print(f"  dias_ultima_corrida: {dias_de(con, 'dias_ultima_corrida')}")
        ind = indicadores(con)
        ind.to_parquet(SNAP)
        SNAP_DAYS.write_text(",".join(dias_previos), encoding="utf-8")
        print(f"\n  {len(ind)} indicadores guardados ({SNAP.name})")
        print(f"  días a preservar: {dias_previos}")
        print("\nAhora corré el incremento y volvé con: "
              "verify <dia_nuevo1>,<dia_nuevo2>")

    elif modo == "verify":
        if len(sys.argv) < 3:
            sys.exit("Falta la lista de días nuevos. Ej: verify 2026-03-23,2026-03-24")
        if not SNAP.exists() or not SNAP_DAYS.exists():
            sys.exit(f"Falta el snapshot — ¿corriste 'snapshot' antes del incremento?")

        dias_nuevos = sorted(d.strip() for d in sys.argv[2].split(",") if d.strip())
        dias_previos = sorted(
            d for d in SNAP_DAYS.read_text(encoding="utf-8").split(",") if d)
        esperados = sorted(set(dias_previos) | set(dias_nuevos))
        prev = pd.read_parquet(SNAP)

        con = conectar()
        print(f"\n=== VERIFICACIÓN incremento {dias_nuevos} ({DB.name}) ===")
        print(f"  días previos (a preservar): {dias_previos}\n")

        # 1. dias_ultima_corrida = SOLO los días nuevos  ← el check del bug
        duc = dias_de(con, "dias_ultima_corrida")
        check(duc == dias_nuevos,
              f"dias_ultima_corrida == {dias_nuevos} (solo días nuevos)",
              f"encontrado: {duc}  ← si incluye días previos, la Fase 3 los reprocesó")

        # 2. las tablas de datos tienen previos + nuevos
        for t in ("etapas", "viajes", "usuarios"):
            d = dias_de(con, t)
            check(d == esperados, f"{t} tiene {esperados}", f"encontrado: {d}")

        # 3. indicadores: están todos los días
        ind = indicadores(con)
        dias_ind = sorted(ind["dia"].unique())
        check(dias_ind == esperados,
              "indicadores tiene todos los días", f"encontrado: {dias_ind}")

        # 4. los indicadores previos NO cambiaron
        v_antes = prev[prev["dia"].isin(dias_previos)]
        v_ahora = ind[ind["dia"].isin(dias_previos)]
        merged = v_antes.merge(
            v_ahora, on=["dia", "detalle", "tabla"],
            how="outer", suffixes=("_antes", "_ahora"), indicator=True)
        faltantes = merged[merged["_merge"] == "left_only"]
        check(faltantes.empty,
              "ningún indicador de los días previos desapareció",
              f"{len(faltantes)} desaparecidos: "
              f"{faltantes[['dia','detalle','tabla']].head(5).to_dict('records')}")

        comunes = merged[merged["_merge"] == "both"].copy()
        comunes["dif"] = (
            comunes["indicador_antes"] - comunes["indicador_ahora"]).abs()
        distintos = comunes[comunes["dif"] > 1e-9]
        check(distintos.empty,
              "los valores de los días previos no cambiaron",
              f"{len(distintos)} cambiaron:\n"
              f"{distintos[['dia','detalle','indicador_antes','indicador_ahora']].head(5)}")

        print(f"\n  (comparados {len(comunes)} indicadores de días previos)")
        print("\n" + ("=== INCREMENTAL OK ===" if ok
                      else "=== HAY FALLAS, ver arriba ==="))
        sys.exit(0 if ok else 1)

    else:
        sys.exit(__doc__)
