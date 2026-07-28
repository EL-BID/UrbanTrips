# DISEÑO — Los 2 rebuilds de `etapas` (#8 estructural del audit)

## Problema
Tras day-scopear todo lo demás, quedan **2 operaciones que reescriben `etapas` ENTERA en
cada corrida** → costo O(acumulado), no O(días de la corrida):

1. **`update_leg_destinations_from_parquet`** (`data.py:600-681`) — write-back de destinos.
   `CREATE etapas_new AS SELECT e.* / COALESCE(u.col,e.col) FROM etapas e LEFT JOIN
   read_parquet(stage) u ON e.id=u.id AND e.dia=u.dia ORDER BY e.dia` → swap → recrear índices.
2. **`create_trips_from_legs_and_fex`** (`trips.py:64-96`) — factores + viajes.
   Loop por-día llena `_ut_etapas_new`, copia los días congelados (`INSERT ... WHERE dia NOT IN
   run_days`, `trips.py:84`) y hace swap.

Ambos son **correctitud-safe** (preservan los días congelados tal cual) — el costo es solo perf.

## Por qué existen los rebuilds (restricciones a respetar)
- **write-back:** un `UPDATE etapas ... FROM parquet` degradaba a fila-por-fila (MVCC, mantiene
  PK/índices por fila) a escala mes (253M) → >75min. El rebuild bulk es ~36min.
- **create_trips:** el patrón viejo (DELETE WHERE dia + INSERT, **día a día en loop sobre la
  misma tabla**) fragmentaba: los scans `WHERE dia` del CTAS de factores midieron 56→207s entre
  semanas 1→4 (+267%) por huecos sin compactar (`trips.py:53-58`).
- **DOBLE FUNCIÓN del write-back (clave):** el `ORDER BY dia` **re-clusteriza etapas por día**.
  Fase 2 escribe `ORDER BY batch_id` (un batch abarca todos los días → mezclado). El rebuild deja
  etapas dia-clusterizada → todo lo downstream (`WHERE dia=X`) poda por zonemap de row-group. Si
  se day-scopea SIN preservar este clustering, se rompe el pruning de create_trips/compute_kpi/etc.

## Insight que habilita el day-scoping
Con el **congelamiento**, los días viejos NO cambian tras su corrida. Ambos rebuilds solo tocan
los días congelados para (a) preservarlos en la tabla monolítica y (b) re-clusterizar. Si cada
corrida escribe SOLO su slice de run-days **como bloques contiguos por día** y lo appendea, la
tabla queda "clusterizada por corrida, y por día dentro de la corrida" ≈ dia-clusterizada (cada
día = un bloque contiguo) → el pruning sigue funcionando **sin** reescribir los congelados.

## Opciones

### A. Day-scope de los 2 rebuilds (RECOMENDADA como primer paso, bajo riesgo)
Reescribir SOLO el slice de run-days, appendeado como bloques por día; nunca tocar los congelados.

**write-back day-scoped:** el parquet de infer_destinations solo tiene run-days (infer corre solo
run-days). Entonces basta actualizar los destinos de los run-days:
```sql
-- en vez de rebuild de toda etapas:
CREATE TABLE _ut_dest AS
  SELECT e.* REPLACE(COALESCE(u.h3_d,e.h3_d) AS h3_d, COALESCE(u.od_validado,e.od_validado) AS od_validado,
                     COALESCE(u.etapa_validada,e.etapa_validada) AS etapa_validada)
  FROM etapas e LEFT JOIN read_parquet(stage) u ON e.id=u.id AND e.dia=u.dia
  WHERE e.dia IN (run_days) ORDER BY e.dia;      -- solo run-days, ordenado por día
DELETE FROM etapas WHERE dia IN (run_days);       -- un DELETE de run-days
INSERT INTO etapas SELECT * FROM _ut_dest;        -- append de bloque contiguo por día
```
Los índices NO se reconstruyen enteros (hoy sí, cada corrida) — se mantienen o se recrean solo si
hace falta (política actual: etapas sin índices en el hot path).

**create_trips day-scoped:** ya computa los run-days en un loop; **eliminar el copy-back de días
congelados** (`trips.py:84`) y el swap → hacer `DELETE FROM etapas WHERE dia IN (run_days)` +
`INSERT` del `_ut_etapas_new` (que ya viene ordenado por día). Sin copiar los N-2 días congelados.

**Fragmentación:** el patrón viejo degradaba porque **re-procesaba los MISMOS días repetidamente**
(el smoke re-corría datos idénticos). En el incremental real cada día se reescribe UNA vez (su
corrida) y después se congela (nunca se re-borra) → la fragmentación NO se compone sobre los
congelados; es un hueco por corrida, acotado. Mitigación si aparece: `CHECKPOINT` periódico.

**Pros:** O(run-days), poco invasivo (2 funciones), preserva clustering (bloques por día), correcto
por construcción (congelados intactos). **Contras:** reintroduce DELETE+INSERT (riesgo de
fragmentación a validar a escala de muchas corridas); requiere cuidar el orden por día en el append.

### B. Particionar `etapas` físicamente por período (ROBUSTA a largo plazo)
Una tabla física por mes (`etapas_2026_03`, …) + vista `etapas` = UNION ALL. Una corrida nueva solo
crea/escribe la partición de su período; las viejas nunca se tocan → O(run-days) REAL y **cero
fragmentación** (cada partición se escribe una vez).
**Pros:** resuelve de raíz ambos problemas; ideal para un año de datos. **Contras:** MUY invasivo —
la vista no es escribible en DuckDB, así que TODOS los sitios que hacen INSERT/DELETE/UPDATE etapas
(decenas) deben enrutar a la partición correcta; y las corridas que cruzan meses complican el ruteo.

### C. `etapas` como parquet particionado (hive) por dia/mes
Escribir etapas como parquet `dia=YYYY-MM-DD/` y leer con `read_parquet(..., hive_partitioning=1)`
(pruning por partición nativo). Una corrida escribe sus particiones; las viejas inmutables.
**Pros:** particionado nativo, inmutable = sin fragmentación, escala a años. **Contras:** cambio
arquitectónico grande (etapas deja de ser tabla DuckDB con índices/UPDATE in-place); hay que
reescribir todo el read/write path de etapas. Alineado con el "congelar" (particiones inmutables).

## Recomendación
1. **Ahora: Opción A** (day-scope los 2 rebuilds). Convierte O(acumulado)→O(run-days) con cambio
   acotado a 2 funciones, correcto por el congelamiento, preservando el dia-clustering vía bloques
   por día. Validar fragmentación a escala de muchas corridas (medir el tiempo del CTAS de factores
   de create_trips corrida tras corrida — si NO degrada como el 56→207s viejo, alcanza).
2. **Si la fragmentación degrada a escala año: Opción C** (parquet particionado) — es el destino
   natural del modelo "congelar" (particiones inmutables por día). Opción B queda como alternativa
   si se quiere seguir en tablas DuckDB puras.

## Validación de A (cuando se implemente)
- Correctitud: comparar `etapas`/`viajes` bit-a-bit vs una corrida con el rebuild viejo (mismos
  días) → idénticos (el day-scope no cambia resultados, solo qué filas se reescriben).
- Clustering: confirmar que un `WHERE dia=X` sigue podando (EXPLAIN ANALYZE, o el timing de
  compute_kpi por día se mantiene ~plano vs run3).
- Perf: create_trips y write-back deben pasar de ~2× (run3) a ~plano vs run1.
- Fragmentación: medir el timing del CTAS de factores a lo largo de varias corridas incrementales.

## Riesgo / nota
Este es el ÚNICO ítem del audit que es decisión de diseño (no parche). Todo lo demás (P1+P2) es
day-scoping directo. Conviene decidir A vs C recién con los números de run4 en mano (cuánto queda
escalando create_trips + write-back tras los otros fixes).
</content>
