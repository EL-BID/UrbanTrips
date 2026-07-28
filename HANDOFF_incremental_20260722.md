# HANDOFF — Corridas incrementales (semana a semana) — 2026-07-22

## Objetivo del usuario
Correr semana por semana (cada una con su config, MISMO alias de DB) y que **agregar una
semana NO aumente el tiempo de procesamiento** proporcional a lo acumulado. Cada corrida
debería costar ~lo mismo, procesando sólo los días nuevos.

## Decisión metodológica tomada: CONGELAR los días viejos
No re-imputar destinos de días ya corridos (`infer_destinations` es el paso más caro). Con
la 1ra corrida = una semana entera, `matriz_validacion` ya tiene cobertura suficiente. Los
números de un día quedan fijos al momento en que se calcularon (bueno para un dashboard: la
fecha no se mueve al cargar más datos). `matriz_validacion` SÍ se sigue reconstruyendo cada
corrida (barato, 1.15×) y alimenta sólo los días nuevos.

---

## EL BUG que se encontró y arregló
`run_process.py` (`_ingest_all_days`) llenaba `dias_ultima_corrida` con
`SELECT DISTINCT dia FROM transacciones`. Pero `transacciones` es ACUMULATIVA (nunca se
limpia por corrida), así que en incremental devolvía TODOS los días históricos. Downstream,
toda la Fase 3 (infer_destinations, assign_*, rearrange, create_trips, compute_kpi, chains)
lee `get_run_days()` → reprocesaba los días viejos.

**FIX aplicado** (`run_process.py:410-472`): preservar los días de esta corrida en
`run_dias` (desde `transacciones_raw`, que sólo tiene los nuevos) y acotar la re-derivación
a `WHERE dia IN (run_dias)`. Un solo fix raíz que arregla en cascada toda la Fase 3.

Confirmado en el smoke run2: procesó 4 días cuando la corrida era de 2 → el bug. Con el fix
+ test nuevo (`test_ingest_incremental_run_days_solo_dias_nuevos`) → deja sólo los nuevos.

---

## ESTADO: cambios aplicados (SIN commitear, 239 tests verdes)
| Archivo | Cambio |
|---|---|
| `urbantrips/utils/run_process.py:410-472` | **FIX run_days** (el crítico) |
| `urbantrips/datamodel/misc.py` | `_weighted_median` numpy (8×, idéntico) + filtros por día + logs `_paso` de progreso |
| `urbantrips/datamodel/trips.py:407` | `verificar_integridad_viajes_etapas` acotado a run_days |
| `urbantrips/carto/stops.py` | default `direction=0` si el CSV no la trae (bug de otra sesión: feature `direction`) |
| `urbantrips/datamodel/transactions.py:420` | `.copy()` para el SettingWithCopyWarning |
| tests | +test incremental run_days, +test weighted_median, +save_run_days en test_misc |

Correr suite: `conda run -n trips6 python -m pytest urbantrips/tests -q`

---

## MAÑANA: correr run3 del smoke (la medición definitiva)

El smoke ya hizo run1 (días 9,10 fresco, 58min) y run2 (16,17 incremental, PERO con el bug
→ reprocesó los 4, 128min). La DB `smoke_inc_2026` tiene hoy los 4 días. run3 agrega 23/24
YA con el fix puesto.

> 23/24 = lun/mar, MISMO tamaño exacto de archivo que 9/10 y 16/17 → comparación limpia.

```bash
# 1. snapshot fresco (captura el estado de 4 días actual; pisa el de 2 días)
python tools/verifica_smoke_incremental.py snapshot

# 2. correr el incremento 23/24 SIN -b  (con -b all borra la DB y se pierde el test)
python -m urbantrips.run_all_urbantrips -c configs/smoke_inc_run3.yaml

# 3. verificar — ahora TODO debe dar OK (antes fallaba dias_ultima_corrida por el bug)
python tools/verifica_smoke_incremental.py verify 2026-03-23,2026-03-24

# 4. mirar el log nuevo (logs/run_<ts>.log): la Fase 3 debe decir "2 día(s)", NO 6
```

El `verify` chequea: `dias_ultima_corrida == [23,24]`; las tablas tienen los 6 días; los
indicadores de los días viejos NO cambiaron (congelados).

---

## LA PREGUNTA que run3 responde
Con run_days arreglado, `WHERE dia IN (23,24)` selecciona 2 días de una tabla de 6.
- ¿DuckDB **poda** a esos 2 días en los JOINs de `persist_indicators` y en `create_trips`?
  → costo **constante** por incremento = **objetivo cumplido**.
- ¿O **escanea** los 6 días acumulados? → falta el fix estructural de create_trips.

Los scans SIMPLES ya se probó que podan (A/B: 1.12× para tabla 4× más grande). El pruning
en **JOINs/GROUP BY** es lo único sin verificar. run3 lo mide.

### Los 2 acumuladores medidos (run1 2d vs run2 4d, esperado 2× si lineal)
- `create_trips`: 234s → 812s = **3.47×**. PERO run2 tenía el bug (recalculó 4 días). Con el
  fix, el recálculo de factores de días viejos YA no pasa (se copian). Queda el rebuild+swap
  que reescribe toda `etapas` (copia días viejos, O(acumulado) pero COPIA no recálculo).
- `persist_indicators`: 27s → 178s = **6.5×**. Las medianas numpy escalan lineal (bien); los
  que explotan son los JOINs `viajes×travel_times_trips` y GROUP BY sobre etapas (6-8×).

---

## PENDIENTES (medir con run3 antes de tocar)

1. **create_trips estructural** (usuario: "importante, no rehaga días ya hechos"). El fix
   run_days ya frena el RECÁLCULO. Falta cambiar rebuild+swap → **append de días nuevos**
   (no reescribir toda la tabla). Delicado: el rebuild se eligió contra fragmentación
   (comentario en trips.py:53-58). MEDIR cuánto cuesta la copia sola en run3 primero.

2. **lowess** (`infer_routes_geoms`, routes.py:429) — variante B: recalcular solo líneas
   SIN geometría. ⚠️ `save_raw` (insumos.py:190) hace CREATE OR REPLACE → hay que leer
   existentes + concat + guardar la unión (NO solo las nuevas). Diseño completo en la
   memoria `escalado-mes-vs-semana-20260722.md`.

3. **agregados dashboard nivel 2** (resumen_x_linea, construyo_indicadores, socio,
   particion). Separables por día (escriben con replace_dash_partition por día), PERO el
   costo real es `materializar_proc_tables` (etapas entera) COMPARTIDA por varios
   consumidores. ~15 min. Baja prioridad, hacer por-función.

4. **assign_stations_od** (legs.py:1450) — deriva días de `SELECT DISTINCT dia FROM etapas`,
   no run_days. TRIVIAL (2.5s), no es correctitud. Muy baja prioridad.

## Por diseño, NO tocar
- `update_stations_catchment_area` (rebuild matriz_validacion) — barato, alimenta días
  nuevos. Es lo que se decidió al congelar.
- dbscan, páginas dashboard, viz — fuera de `run_all`.

---

## Archivos clave
- Configs smoke: `configs/smoke_inc_run{1,2,3}.yaml` (run3 = días 23/24, listo)
- Verificador: `tools/verifica_smoke_incremental.py` (`snapshot` / `verify <dias>`)
- DB smoke: `data/db/smoke_inc_2026_*.duckdb` (4 días: 9,10,16,17)
- Memoria detallada: `~/.claude/.../memory/escalado-mes-vs-semana-20260722.md`

## Nota
Fix SIN commitear a propósito (el usuario commitea). Cuando run3 valide, preguntar si
commitear todo el set.
