# AUDIT — Day-scoping del pipeline incremental (2026-07-23)

## Objetivo
Que procesar una corrida cueste **O(días de la corrida)**, no O(acumulado), y que se pueda
**re-procesar un subconjunto de días sin romper el resto** (recuperación de crashes,
correcciones, prod con un año de datos). Las tablas `transacciones/etapas/viajes/usuarios/
gps/travel_times_*` son ACUMULATIVAS; los días de la corrida están en `dias_ultima_corrida`
(`ctx.data.get_run_days()`). Los días viejos están "congelados" (no se re-imputan).

Auditado con 3 barridos (enrichment+trips / KPI+indicadores / dashboard-prep). Clasificación:
✅ day-scoped · 🔵 full-table-pero-preserva (correcto, perf ∝ acumulado) · 🔴 acumulador o riesgo.

## TL;DR
- **La mayoría del pipeline ya es correctitud-safe** para re-procesar (chains, assign_*,
  rearrange, verificar_integridad, create_trips y el write-back de destinos preservan los
  congelados; persist_indicators y kpi_by_day_line/_service hacen upsert por día).
- **PERO hay 2 bugs de CORRECTITUD que HOY impiden re-procesar** días ya presentes → son el
  **prerequisito** de la feature de re-procesamiento robusto.
- El resto es **PERF**: casi todo cuelga de 3-4 raíces con fix de 1-2 líneas, + 2 estructurales
  (rebuilds de `etapas`) que necesitan decisión de diseño (particionar/append).
- lowess (`infer_routes_geoms`) ya se arregló esta sesión.

---

## 🔴 PRIORIDAD 1 — CORRECTITUD (bloquean el re-procesamiento; arreglar PRIMERO)

### 1. `kpi.py` — patrón `processed_days` (3 funciones)
`run_basic_kpi` (kpi.py:1320-1338), `compute_speed_by_day_veh_hour` (kpi.py:1257-1264),
`compute_dispatched_services_by_line_hour_day` (kpi.py:1760-1783) usan un guard legacy
`dia NOT IN (processed_days)` y **nunca borran sus tablas de salida para los run-days**
(`basic_kpi_by_line_day/_hr`, `basic_kpi_by_vehicle_hr`, `services_by_line_hour`).
- **Correctitud:** re-procesar un día ya presente → `NOT IN(processed_days)` lo **excluye** →
  la query no devuelve nada → quedan las **filas viejas STALE, nunca recomputadas**. Estos KPIs
  NO soportan re-procesar (a diferencia de `kpi_by_day_line/_service`, que sí hacen upsert).
- **Perf (O(n²)):** corren dentro de un loop por-día; cada iteración re-lee la salida acumulada
  (`get_processed_days`/`get_raw`, kpi.py:1722) que crece, y arma un `NOT IN` cada vez más grande
  → Σk = O(N²). **Es la causa de la escalera** (run_basic_kpi día1 108s→día26 396s; dispatched
  "escalera perfecta"). NO es el scan de la fuente (etapas/gps/services están dia-clustered
  `ORDER BY dia` → el `dia='X'` poda por zonemap).
- **Fix (un solo patrón):** antes de cada loop, `_delete_run_days_from(...)` esas 4 tablas
  (como ya hace `kpi_by_day_line`), y **eliminar toda la maquinaria `processed_days`**, apoyándose
  solo en el scope `dia = '{dia}'`. Arregla correctitud **y** O(n²) de una.

### 2. `services.py` — `delete_old_services_data(None)` + anti-join
`process_services(ctx, line_ids=None)` (run_process.py:672) → `delete_old_services_data`
(services.py:81-92) con `line_ids=None` **borra TODA la tabla `services`/`services_gps_points`/
`services_stats`** (todos los días) y reclasifica todo el histórico GPS desde cero.
- Además `get_stops_and_gps_data` (services.py:110-124) hace un anti-join
  `gps LEFT JOIN services_stats ... WHERE ss IS NULL` sobre el **gps acumulado entero** (el
  "Descargando paradas y gps" que medimos 3.4×). Entre el wipe-total y el anti-join, services
  solo puede **wipe-all** o **skip-existing**, nunca re-procesar limpio los run-days.
- **Fix:** acotar el DELETE a `WHERE dia IN (run_days)` y el `gps_query` a `AND g.dia IN
  (run_days)` (soltando la dependencia wipe→anti-join). → O(run-days) y re-procesable.

### 3. (latente) `save_indicator` de `distribucion` / `viajes_hora`
`preparo_dashboard.py:1248-1249` → `save_indicator` = `CREATE OR REPLACE TABLE` (reemplazo
ENTERO). Hoy es correcto **porque** el proc materializa todos los días. Si se day-scopea el proc
(PERF #7) **sin** convertir estas 2 a upsert por-día, **perderían los días viejos**. Único cambio
que DEBE ir coordinado con el day-scoping del proc. (`construyo_indicadores` NO tiene la trampa:
su lógica `indicadores_ant`+"Todos" ya reconstruye desde el histórico.)

### 4. (latente, código muerto) `compute_trips_travel_time` (trips.py:642-664)
`INSERT INTO travel_times_legs/_trips ... JOIN dias_ultima_corrida` **sin DELETE previo** →
duplicaría filas en re-run. **No se llama en run_all** (código muerto). Dejar muerto, o agregar
DELETE por run-days antes de cablearlo.

---

## 🔴 PRIORIDAD 2 — PERF, barato y de alto impacto (day-scope, casi todo 1-2 líneas)

### 5. `assign_stations_od` (legs.py:1446) — el peor de Fase 3
Deriva los días de `SELECT DISTINCT dia FROM etapas` (**todos los acumulados**) en vez de
`get_run_days()`, y borra+reprocesa cada día (legs.py:1448-1474). Re-toca días congelados (con
valores idénticos → no corrompe) pero cuesta ∝ acumulado. **Fix trivial, bit-idéntico:**
`dias = sorted(get_run_days()["dia"])` (station assignment es day-separable).

### 6. `misc.py` — JOINs `viajes × travel_times_trips` (persist_indicators, el 3.6×)
misc.py:323, 367, 407, 419 (en loop), 459: filtran solo `v.dia IN (run_days)`; DuckDB **no
propaga** el filtro al lado `travel_times_trips` (acumulado, sin dia-cluster ni índice) → lo
escanea entero. El del loop de medianas (419) es **O(n²)**. **Fix (1 línea c/u):** agregar el
predicado simétrico `AND tt.dia IN (dias_str)` (o `AND tt.dia = '{_dia}'` en el loop).

### 7. `materializar_proc_tables` (sql_queries.py:184-198) — raíz del dashboard-prep 1.6×
Los CTE `ETAPAS_PROC_CTE`/`VIAJES_PROC_CTE` (sql_queries.py:132/164) filtran solo
`od_validado=1`, sin `dia` → materializan `etapas`/`viajes` **enteras** cada corrida. Es el motor
del costo de `construyo_indicadores`, `crea_socio_indicadores`, `guarda_particion_modal`,
`resumen_x_linea`, `calculo_kpi_lineas`. **Fix:** filtro run_days en los 2 CTE
(`JOIN dias_ultima_corrida` o `WHERE dia IN (SELECT dia FROM dias_ultima_corrida)`).
⚠️ **Coordinar con #3** (save_indicator) o se pierden días en `distribucion`/`viajes_hora`.
Los consumidores que escriben con `replace_dash_partition` (agg_indicadores, socio_indicadores,
datos_particion_modal, resumen_lineas*) se vuelven ✅ automáticamente al scopear el proc.

### 8. Lecturas full en `resumen_x_linea` y `kpi_lineas.py`
- `resumen_x_linea`: `gps` (preparo_dashboard.py:1350), `transacciones` (1356), `get_raw
  kpi_by_day_line`/`services` (1352-53) sin filtro dia.
- `kpi_lineas.py:levanto_data`: `gps` (146), `transacciones` (148), `services` (157) sin filtro;
  y `calculo_kpi_lineas` reescribe `kpis_lineas` para TODOS los días acumulados.
- **Fix:** `WHERE dia IN (run_days)` en esas queries + upsert solo run-days en `kpis_lineas`
  (recomputar la fila "Promedios" desde el `kpis_lineas` ya persistido).

### 9. `services.py:110` anti-join gps (ya cubierto en #2)
### 10. Serial `get_transactions(batch)` (legs.py:72 / data.py:276)
Solo el **camino serial** de legs (parallel ya arreglado con `get_transactions_for_chunk`). Carga
todos los días del batch en RAM y pandas descarta. **Fix:** `AND dia IN (run_days)` en
`get_transactions`, o rutear el serial por `get_transactions_for_chunk`. Baja prioridad.

---

## 🔵 PRIORIDAD 3 — ESTRUCTURAL (correcto, pero O(acumulado); decisión de diseño)

Después de day-scopear todo lo de arriba, quedan **2 rebuilds de `etapas` entera por corrida**:

### 11. `create_trips` rebuild+swap (trips.py:83-96)
`INSERT INTO _ut_etapas_new SELECT * FROM etapas WHERE dia NOT IN (run_days)` + `DROP/RENAME`.
Copia cada día congelado tal cual (correcto) pero reescribe toda `etapas` (+ re-clustering) cada
corrida. Elegido contra fragmentación (trips.py:53-58).

### 12. `update_leg_destinations_from_parquet` (data.py:600-681)
Rebuild completo de `etapas` (`CREATE etapas_new AS SELECT ... LEFT JOIN parquet` + swap +
recrear índices). `COALESCE` preserva los congelados (correcto), pero reescribe todo + reconstruye
todos los índices cada corrida.

**Opción de diseño para 11 y 12:** **particionar `etapas` por período** (tabla/parquet por
mes/día) → una corrida nueva nunca toca las particiones viejas → escritura O(días) real, sin
copiar ni fragmentar. Es el cambio más grande pero resuelve los dos de raíz (alinea con la nota
"create_trips append vs rebuild" de la memoria). Alternativa por-op: append-only de días nuevos
con compactación periódica.

---

## Ya ✅ (referencia)
chains (day-scoped end-to-end), assign_time_distances, assign_gps_origin, rearrange,
verificar_integridad, los `UPDATE etapas` del adapter (update_leg_trip_ids / destinations),
save_legs (fix de hoy), get_transactions_for_chunk (fix de hoy), persist_indicators single-table
aggregates, kpi_by_day_line/_service (upsert), compute_kpi_by_service, lowess (variante B de hoy).

## Orden de ataque sugerido
1. **P1 correctitud (#1 kpi processed_days, #2 services)** — habilitan re-procesar sin stale.
   Bonus: #1 mata el O(n²) de compute_kpi.
2. **P2 perf barato (#5 assign_stations_od, #6 misc JOINs)** — mayor ratio impacto/esfuerzo.
3. **#7 proc + #3 save_indicator (coordinados)** — el grueso del dashboard-prep.
4. **#8, #10** — resto de lecturas full.
5. **P3 estructural (#11, #12)** — decidir particionado vs append; el residual O(acumulado).

Cada fix con su regresión (como save_legs / Fase 2 / lowess). Nada commiteado.
</content>
