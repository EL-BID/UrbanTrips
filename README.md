![logo](https://github.com/EL-BID/UrbanTrips/blob/dev/docs/img/logo_readme.png)

# UrbanTrips

**UrbanTrips** es una librería de Python de código abierto que procesa datos de transacciones de sistemas de pago con tarjeta inteligente para transporte público. A partir de esos datos infiere destinos de viaje, encadena etapas en viajes completos, y produce matrices origen-destino, indicadores operativos (KPIs) y visualizaciones.

El único requisito mínimo es un CSV con transacciones geolocalizadas y un archivo de configuración YAML. Los resultados mejoran progresivamente a medida que se incorporan insumos opcionales (datos GPS, trazados de líneas, zonificaciones).

> Para una discusión metodológica sobre la imputación de destinos y la construcción de matrices OD, ver el [documento metodológico](https://github.com/EL-BID/UrbanTrips/blob/dev/docs/Metodologia_UrbanTrips.pdf).
> Documentación completa: [el-bid.github.io/UrbanTrips](https://el-bid.github.io/UrbanTrips/).

---

## Índice

- [Prerequisitos](#prerequisitos)
- [Instalación](#instalación)
- [Inicio rápido](#inicio-rápido)
- [Cómo funciona](#cómo-funciona)
- [Referencia de configuración](#referencia-de-configuración)
- [Esquema de datos de entrada](#esquema-de-datos-de-entrada)
- [Estructura de directorios](#estructura-de-directorios)
- [Ejecución del pipeline](#ejecución-del-pipeline)
- [Configuración para desarrollo](#configuración-para-desarrollo)
- [Agradecimientos](#agradecimientos)
- [Licencia](#licencia)

---

## Prerequisitos

| Requisito | Versión |
|---|---|
| Python | ≥ 3.12 |
| pip o [uv](https://docs.astral.sh/uv/) | cualquier versión reciente |

La librería utiliza **DuckDB** para almacenamiento (no requiere servidor de base de datos) y hexágonos **H3** para indexado espacial. El procesamiento geoespacial se apoya en GeoPandas, Shapely y Fiona, por lo que se recomienda fuertemente usar un entorno virtual.

---

## Instalación

### Con pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install urbantrips
```

### Con uv

```bash
uv venv
source .venv/bin/activate
uv pip install urbantrips
```

### Con conda (útil para evitar problemas con GDAL)

```bash
conda create -n urbantrips -c conda-forge python=3.12
conda activate urbantrips
pip install urbantrips
```

---

## Inicio rápido

1. **Colocá el CSV de transacciones** en `data/data_ciudad/`.
2. **Editá `configs/configuraciones_generales.yaml`** para mapear los nombres de columnas y definir los nombres de las corridas.
3. **Ejecutá el pipeline completo:**

```bash
python urbantrips/run_all_urbantrips.py
```

Los resultados se guardan en `resultados/` y el dashboard de Streamlit se lanza automáticamente al finalizar.

Hay un dataset de ejemplo disponible (transacciones SUBE de Buenos Aires) [aquí](https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/transacciones.csv). Descargarlo en `data/data_ciudad/transacciones.csv` y usar la [configuración de ejemplo](#configuración-de-ejemplo) más abajo.

---

## Cómo funciona

El pipeline corre cuatro pasos secuenciales:

```
ingest  →  legs  →  outputs  →  dashboard
```

| Paso | Descripción |
|---|---|
| **ingest** | Lee el CSV de transacciones, valida y limpia los registros, geolocaliza las transacciones (desde coordenadas o tabla GPS), y escribe etapas normalizadas en la base de datos DuckDB. |
| **legs** | Agrupa etapas en viajes por tarjeta por día, imputa destinos faltantes usando la siguiente transacción como referencia, y aplica filtros de tolerancia espacial. |
| **outputs** | Calcula matrices OD para cada nivel de zonificación configurado, KPIs (IPK, factor de carga, tiempo de viaje, etc.), líneas de deseo, y exporta resultados a CSV, PNG, PDF y HTML. |
| **dashboard** | Prepara tablas agregadas y lanza un dashboard de Streamlit para exploración interactiva. |

Cada corrida procesa uno o más días definidos en la clave `corridas` del archivo de configuración. Las corridas sucesivas se acumulan en las mismas bases de datos DuckDB, permitiendo procesar `semana1`, `semana2`, etc. por separado y consultarlas en conjunto.

### Imputación de destinos

Para cada etapa, la librería busca el siguiente tap-on de la misma tarjeta y encuentra la parada de la línea utilizada que minimiza la distancia a esa siguiente transacción, sujeto a una tolerancia configurable (`tolerancia_parada_destino`, por defecto 2200 m). Si ninguna parada candidata cae dentro de la tolerancia, el destino queda sin imputar.

### Líneas y ramales

Una línea puede tener múltiples ramales (variantes de recorrido). Con `lineas_contienen_ramales: True`, la imputación considera las paradas de todos los ramales de una línea, no solo el ramal registrado en la transacción. Esto permite manejar sistemas de metro (donde los egresos no siempre se registran por ramal) y redes de buses donde el ramal registrado puede ser poco confiable.

---

## Referencia de configuración

Toda la configuración vive en `configs/configuraciones_generales.yaml`. Las secciones principales son:

### Definición de corridas

```yaml
corridas: ['20251015', '20251018']   # Nombres de corridas — deben coincidir con los prefijos de los archivos trx y gps
alias_db_insumos: "ciudad_insumos"   # Base de datos compartida de insumos (recorridos, paradas, zonificaciones)
```

Para cada nombre de corrida `X` se espera un archivo `data/data_ciudad/X_trx.csv` (y opcionalmente `X_gps.csv`).

### Mapeo de columnas

```yaml
nombres_variables_trx:
    id_trx: "ID_TRX"
    fecha_trx: "FECHATRX"
    id_tarjeta_trx: "NROTARJETA"
    modo_trx: "MODO"
    hora_trx:                        # dejar vacío si la hora está incluida en fecha_trx
    id_linea_trx: "IDLINEA"
    id_ramal_trx: "RAMAL"
    interno_trx: "INTERNO"
    orden_trx:                       # orden secuencial por tarjeta por día; requerido si no hay timestamp con minutos
    latitud_trx: "LATITUDE"
    longitud_trx: "LONGITUDE"
    factor_expansion:                # factor de expansión muestral (opcional)
```

### Procesamiento de transacciones

```yaml
formato_fecha: "%Y-%m-%d %H:%M:%S"  # formato strftime del campo de fecha
columna_hora: False                  # True si la hora está en una columna entera separada
ordenamiento_transacciones: "fecha_completa"  # o "orden_trx"
ventana_viajes: 120                  # minutos — máxima brecha entre etapas del mismo viaje
ventana_duplicado: 5                 # minutos — ventana para detectar taps simultáneos duplicados
```

### Geografía

```yaml
resolucion_h3: 9       # Resolución H3: 8 ≈ 460 m de lado, 9 ≈ 174 m, 10 ≈ 65 m
epsg_m: 5347           # CRS proyectado (en metros) para cálculos de distancia
filtro_latlong_bbox:
    minx: -59.3
    miny: -35.5
    maxx: -57.5
    maxy: -34.0
```

### Insumos opcionales

```yaml
recorridos_geojson: "recorridos.geojson"                   # Geometrías de recorridos de líneas
nombre_archivo_informacion_lineas: "lineas.csv"            # Metadata de líneas (nombre, modo, empresa)
lineas_contienen_ramales: True

zonificaciones:
    geo1: "partidos.geojson"
    var1: "nombre_partido"
    orden1: ['Zona A', 'Zona B', 'Zona C']

usa_archivo_gps: True
```

### Configuración de ejemplo

Para el dataset de muestra SUBE de Buenos Aires:

```yaml
corridas: ['transacciones']
alias_db_insumos: "amba"
resolucion_h3: 8
tolerancia_parada_destino: 2200
imputar_destinos_min_distancia: True
lineas_contienen_ramales: True
nombre_archivo_informacion_lineas: "lineas_amba.csv"
formato_fecha: "%d/%m/%Y"
columna_hora: True
ordenamiento_transacciones: orden_trx
geolocalizar_trx: False

nombres_variables_trx:
    id_trx: id
    fecha_trx: fecha
    id_tarjeta_trx: id_tarjeta
    modo_trx: modo
    hora_trx: hora
    id_linea_trx: id_linea
    id_ramal_trx: id_ramal
    interno_trx: interno_bus
    orden_trx: etapa_red_sube
    latitud_trx: lat
    longitud_trx: lon
    factor_expansion:

modos:
    autobus: COL
    tren: TRE
    metro: SUB

filtro_latlong_bbox:
    minx: -59.3
    miny: -35.5
    maxx: -57.5
    maxy: -34.0
```

---

## Esquema de datos de entrada

### Transacciones (obligatorio)

| Campo | Tipo | Notas |
|---|---|---|
| `fecha_trx` | strftime | **Obligatorio.** Timestamp completo o solo fecha. |
| `id_tarjeta_trx` | int/str | **Obligatorio.** Identificador de tarjeta (único por día). |
| `id_linea_trx` | int | **Obligatorio.** Identificador de línea. |
| `latitud_trx` | float | **Obligatorio.** Latitud del tap-on. |
| `longitud_trx` | float | **Obligatorio.** Longitud del tap-on. |
| `hora_trx` | int (0–23) | Obligatorio si `fecha_trx` no tiene componente horario. |
| `orden_trx` | int | Obligatorio si `fecha_trx` no tiene precisión de minutos. |
| `interno_trx` | int | Identificador del vehículo. Obligatorio si se usa geolocalización GPS. |
| `id_ramal_trx` | int | Identificador de ramal. Opcional. |
| `modo_trx` | str | Modo de transporte. Por defecto `autobus` si está ausente. |
| `factor_expansion` | float | Factor de expansión muestral. Opcional. |

### GPS (opcional)

| Campo | Tipo | Notas |
|---|---|---|
| `id_linea_gps` | int | **Obligatorio.** |
| `interno_gps` | int | **Obligatorio.** |
| `fecha_gps` | strftime | **Obligatorio.** Debe incluir día, hora y minuto. |
| `latitud_gps` | float | **Obligatorio.** |
| `longitud_gps` | float | **Obligatorio.** |
| `id_ramal_gps` | int | Obligatorio si las líneas tienen ramales. |

### Información de líneas (opcional)

| Campo | Tipo | Notas |
|---|---|---|
| `id_linea` | int | **Obligatorio.** |
| `nombre_linea` | str | **Obligatorio.** |
| `modo` | str | **Obligatorio.** |
| `id_ramal` | int | Obligatorio si las líneas tienen ramales. |
| `nombre_ramal` | str | Obligatorio si las líneas tienen ramales. |
| `empresa` | str | Opcional. |

### Geometrías de recorridos (GeoJSON opcional)

| Campo | Tipo | Notas |
|---|---|---|
| `id_linea` | int | **Obligatorio.** |
| `id_ramal` | str | Obligatorio si las líneas tienen ramales. |
| `geometry` | LineString 2D | **Obligatorio.** Debe ser un LineString simple, no MultiLineString. |
| `stops_distance` | int | Opcional. Metros entre paradas interpoladas. |
| `line_stops_buffer` | int | Opcional. Buffer para fusionar paradas cercanas. |

---

## Estructura de directorios

```
UrbanTrips/
├── configs/
│   └── configuraciones_generales.yaml
├── data/
│   ├── data_ciudad/          # Archivos de entrada (CSV, GeoJSON)
│   │   ├── 20251015_trx.csv
│   │   ├── 20251015_gps.csv
│   │   ├── recorridos.geojson
│   │   └── partidos.geojson
│   └── db/                   # Bases de datos DuckDB (se crean automáticamente)
│       ├── ciudad_data.duckdb
│       └── ciudad_insumos.duckdb
├── resultados/               # Todos los resultados (se crean automáticamente)
│   ├── data/                 # Exportaciones CSV
│   ├── html/                 # Mapas interactivos
│   ├── matrices/             # Archivos de matrices OD
│   ├── pdf/
│   ├── png/
│   └── tablas/
└── urbantrips/               # Código fuente de la librería
```

---

## Ejecución del pipeline

### Corrida completa

```bash
python urbantrips/run_all_urbantrips.py
```

### Corridas parciales

```bash
# Detener después de legs (sin outputs ni dashboard)
python urbantrips/run_all_urbantrips.py --through legs

# Ejecutar un único paso
python urbantrips/run_all_urbantrips.py --step outputs

# Sin dashboard
python urbantrips/run_all_urbantrips.py --no_dashboard
```

### Usar un archivo de configuración alternativo

```bash
python urbantrips/run_all_urbantrips.py --config configs/otra_ciudad.yaml
```

### Volver a correr desde cero

```bash
python urbantrips/run_all_urbantrips.py --borrar_corrida all
```

### Lanzar el dashboard manualmente

```bash
streamlit run urbantrips/dashboard/dashboard.py
```

#### Acceso concurrente a las bases

El dashboard abre las bases DuckDB **en modo solo lectura**, con conexiones de
vida corta. Eso permite:

- tener **varios dashboards abiertos** sobre la misma base al mismo tiempo;
- que el pipeline pueda arrancar aunque haya dashboards abiertos.

Las pocas acciones del dashboard que escriben (procesar indicadores de línea en
*Herramientas interactivas*, comparar líneas, procesar un polígono en *Estimar
demanda*, guardar escenarios/clusters) abren una **ventana de escritura**
acotada: se cierran las conexiones de lectura, se escribe, y se vuelve a solo
lectura. Si en ese momento otra corrida u otro dashboard tiene la base tomada,
se muestra un aviso y se reintenta, en vez de fallar con un error de DuckDB.

Límite conocido: DuckDB admite un solo escritor por archivo y no permite
lectores mientras hay un escritor. **Mientras corre el pipeline, los dashboards
no van a poder leer esa base.**

Para forzar el modo solo lectura en cualquier proceso (por ejemplo un script de
consulta) se puede exportar `URBANTRIPS_DB_READ_ONLY=1`.

---

## Configuración para desarrollo

```bash
git clone https://github.com/EL-BID/UrbanTrips.git
cd UrbanTrips
uv sync
```

O con pip:

```bash
pip install -e ".[dev]"
```

El `Makefile` replica los mismos chequeos que corren en GitHub Actions:

| Comando | Descripción |
|---|---|
| `make lint` | Verificación de estilo con Ruff |
| `make unit` | Tests unitarios con cobertura |
| `make integration` | Tests de integración con cobertura |
| `make build` | Construcción del paquete |
| `make ci` | Todos los pasos anteriores en secuencia |

---

## Agradecimientos

Agradecemos la colaboración de los gobiernos de Ciudad de Buenos Aires, Córdoba, Mendoza y Bariloche, que proveyeron datos y participaron de valiosas conversaciones para mejorar esta librería.

---

## Licencia

Copyright © 2023. Banco Interamericano de Desarrollo ("BID"). Uso autorizado. [AM-331-A3](LICENSE.md)

## Autores

Felipe González ([@alephcero](https://github.com/alephcero/))  
Sebastián Anapolsky ([@sanapolsky](https://github.com/sanapolsky/))
