![logo](https://github.com/EL-BID/UrbanTrips/blob/dev/docs/img/logo_readme.png)

# UrbanTrips

**UrbanTrips** is an open-source Python library that processes smart card transaction data from public transit systems. It infers trip destinations, chains stages into complete journeys, and produces origin-destination matrices, KPIs, and visualizations — all from a single geolocated CSV of tap-on events.

The library is designed for cities with minimal data infrastructure: the only hard requirements are a transactions file and a configuration YAML. Results get progressively richer as optional inputs (GPS data, route shapes, zoning layers) are added.

> For a detailed methodological discussion on destination imputation and OD matrix construction, see the [methodology document](https://github.com/EL-BID/UrbanTrips/blob/dev/docs/Metodologia_UrbanTrips.pdf) (Spanish).
> Full documentation: [el-bid.github.io/UrbanTrips](https://el-bid.github.io/UrbanTrips/).

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration reference](#configuration-reference)
- [Input data schemas](#input-data-schemas)
- [Directory structure](#directory-structure)
- [Running the pipeline](#running-the-pipeline)
- [Development setup](#development-setup)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.12 |
| pip or [uv](https://docs.astral.sh/uv/) | any recent version |

The library uses **DuckDB** for storage (no database server needed) and **H3** hexagons for spatial indexing. Heavy geospatial work is handled by GeoPandas, Shapely, and Fiona, so a working GDAL stack is required — using a virtual environment is strongly recommended.

---

## Installation

### With pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install urbantrips
```

### With uv

```bash
uv venv
source .venv/bin/activate
uv pip install urbantrips
```

### With conda (avoids GDAL build issues on some platforms)

```bash
conda create -n urbantrips -c conda-forge python=3.12
conda activate urbantrips
pip install urbantrips
```

---

## Quick start

1. **Place your transaction CSV** in `data/data_ciudad/`.
2. **Edit `configs/configuraciones_generales.yaml`** to map your column names and set the run names.
3. **Run the full pipeline:**

```bash
python urbantrips/run_all_urbantrips.py
```

Results land in `resultados/` and the Streamlit dashboard launches automatically at the end.

A ready-to-use sample dataset (Buenos Aires SUBE transactions) is available [here](https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/transacciones.csv). Download it to `data/data_ciudad/transacciones.csv` and use the [sample configuration](#sample-configuration) below.

---

## How it works

The pipeline runs four sequential steps:

```
ingest  →  legs  →  outputs  →  dashboard
```

| Step | What it does |
|---|---|
| **ingest** | Reads the transaction CSV, validates and cleans records, geolocates transactions (from coordinates or a GPS table), and writes normalised stages to the DuckDB data database. |
| **legs** | Groups stages into trips per card per day, imputes missing destinations using the next transaction as a proxy, and applies spatial tolerance filters. |
| **outputs** | Computes OD matrices at each configured zoning level, KPIs (IPK, load factor, travel time, etc.), desire lines, and exports results to CSV, PNG, PDF, and HTML. |
| **dashboard** | Prepares aggregated tables and launches a Streamlit dashboard for interactive exploration. |

Each run processes one or more days defined in the `corridas` config key. Multiple runs accumulate into the same DuckDB databases, so you can process `week1`, `week2`, etc. separately and query across all of them.

### Destination imputation

For each stage, the library looks at the next tap-on of the same card and finds the stop on the used line that minimises the distance to that next tap-on, subject to a configurable tolerance (`tolerancia_parada_destino`, default 2200 m). If no candidate stop falls within tolerance, the destination is left unimputed.

### Lines and branches

A single line can contain multiple branches (route variants). When `lineas_contienen_ramales: True`, the imputation considers stops from all branches of a line, not just the specific branch recorded in the transaction. This handles metro systems (where exits are not always recorded per branch) and bus networks where the recorded branch may be unreliable.

---

## Configuration reference

All configuration lives in `configs/configuraciones_generales.yaml`. The key sections are:

### Run definition

```yaml
corridas: ['20251015', '20251018']   # Run names — must match transaction/GPS file prefixes
alias_db_insumos: "city_inputs"      # Shared inputs database (routes, stops, zoning)
```

Each run name `X` expects a file `data/data_ciudad/X_trx.csv` (and optionally `X_gps.csv`).

The list is **incremental**: you can add new runs and re-run — only the new ones are processed, already-finished ones are skipped. See [Incremental runs, recovery and reprocessing](#incremental-runs-recovery-and-reprocessing).

### Column mapping

```yaml
nombres_variables_trx:
    id_trx: "ID_TRX"
    fecha_trx: "FECHATRX"
    id_tarjeta_trx: "NROTARJETA"
    modo_trx: "MODO"
    hora_trx:                        # leave blank if hour is embedded in fecha_trx
    id_linea_trx: "IDLINEA"
    id_ramal_trx: "RAMAL"
    interno_trx: "INTERNO"
    orden_trx:                       # sequential order per card per day; required if no minute-level timestamp
    latitud_trx: "LATITUDE"
    longitud_trx: "LONGITUDE"
    factor_expansion:                # optional sampling weight
```

### Transaction processing

```yaml
formato_fecha: "%Y-%m-%d %H:%M:%S"  # strftime format of the date field
columna_hora: False                  # True if hour is in a separate integer column
ordenamiento_transacciones: "fecha_completa"  # or "orden_trx"
ventana_viajes: 120                  # minutes — max gap between stages in the same trip
ventana_duplicado: 5                 # minutes — window to detect simultaneous duplicate taps
```

### Geography

```yaml
resolucion_h3: 9       # H3 resolution: 8 ≈ 460 m sides, 9 ≈ 174 m, 10 ≈ 65 m
epsg_m: 5347           # Projected CRS (metres) for distance calculations
filtro_latlong_bbox:
    minx: -59.3
    miny: -35.5
    maxx: -57.5
    maxy: -34.0
```

### Optional inputs

```yaml
recorridos_geojson: "routes.geojson"               # Line route geometries
nombre_archivo_informacion_lineas: "lines.csv"     # Line metadata (name, mode, company)
lineas_contienen_ramales: True

zonificaciones:
    geo1: "districts.geojson"
    var1: "district_name"
    orden1: ['Zone A', 'Zone B', 'Zone C']

usa_archivo_gps: True
```

### Sample configuration

For the Buenos Aires SUBE sample dataset:

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

## Input data schemas

### Transactions (mandatory)

| Field | Type | Notes |
|---|---|---|
| `fecha_trx` | strftime | **Required.** Full timestamp or date only. |
| `id_tarjeta_trx` | int/str | **Required.** Card identifier (unique per day). |
| `id_linea_trx` | int | **Required.** Line identifier. |
| `latitud_trx` | float | **Required.** Tap-on latitude. |
| `longitud_trx` | float | **Required.** Tap-on longitude. |
| `hora_trx` | int (0–23) | Required when `fecha_trx` has no time component. |
| `orden_trx` | int | Required when `fecha_trx` has no minute-level precision. |
| `interno_trx` | int | Vehicle identifier. Required when GPS geolocation is used. |
| `id_ramal_trx` | int | Branch identifier. Optional. |
| `modo_trx` | str | Transport mode. Defaults to `autobus` if absent. |
| `factor_expansion` | float | Sampling expansion factor. Optional. |

### GPS (optional)

| Field | Type | Notes |
|---|---|---|
| `id_linea_gps` | int | **Required.** |
| `interno_gps` | int | **Required.** |
| `fecha_gps` | strftime | **Required.** Must include day, hour, and minute. |
| `latitud_gps` | float | **Required.** |
| `longitud_gps` | float | **Required.** |
| `id_ramal_gps` | int | Required if lines have branches. |

### Line information (optional)

| Field | Type | Notes |
|---|---|---|
| `id_linea` | int | **Required.** |
| `nombre_linea` | str | **Required.** |
| `modo` | str | **Required.** |
| `id_ramal` | int | Required if lines have branches. |
| `nombre_ramal` | str | Required if lines have branches. |
| `empresa` | str | Optional. |

### Route geometries (optional GeoJSON)

| Field | Type | Notes |
|---|---|---|
| `id_linea` | int | **Required.** |
| `id_ramal` | str | Required if lines have branches. |
| `geometry` | 2D LineString | **Required.** Must be a single LineString, not MultiLineString. |
| `stops_distance` | int | Optional. Metres between interpolated stops. |
| `line_stops_buffer` | int | Optional. Buffer for merging nearby stops. |

---

## Directory structure

```
UrbanTrips/
├── configs/
│   └── configuraciones_generales.yaml
├── data/
│   ├── data_ciudad/          # Input files (CSV, GeoJSON)
│   │   ├── 20251015_trx.csv
│   │   ├── 20251015_gps.csv
│   │   ├── routes.geojson
│   │   └── districts.geojson
│   └── db/                   # DuckDB databases (auto-created)
│       ├── city_data.duckdb
│       └── city_inputs.duckdb
├── resultados/               # All outputs (auto-created)
│   ├── data/                 # CSV exports
│   ├── html/                 # Interactive maps
│   ├── matrices/             # OD matrix files
│   ├── pdf/
│   ├── png/
│   └── tablas/
└── urbantrips/               # Library source
```

---

## Running the pipeline

### Full run

```bash
python urbantrips/run_all_urbantrips.py
```

### Partial runs

```bash
# Stop after legs (skip outputs and dashboard)
python urbantrips/run_all_urbantrips.py --through legs

# Run only one step
python urbantrips/run_all_urbantrips.py --step outputs

# Skip dashboard
python urbantrips/run_all_urbantrips.py --no_dashboard
```

### Using an alternate config file

```bash
python urbantrips/run_all_urbantrips.py --config configs/other_city.yaml
```

### Re-running from scratch

```bash
python urbantrips/run_all_urbantrips.py --borrar_corrida all
```

### Incremental runs, recovery and reprocessing

The pipeline keeps a **per-run, per-step progress log** in
`{alias}_general.duckdb` (one row per processed day, recording when each step —
ingest, legs, outputs, dashboard — completed). Using that log, a full run
(`run_all_urbantrips.py` without `--step`/`--through`) decides what to do with
each run listed under `corridas:` in the config:

- **New** (not in the log) → processed in full.
- **Incomplete** (interrupted by a crash) → **resumed from the step that was
  missing**, without redoing completed work.
- **Complete** → **skipped**.

This makes the following safe:

```bash
# Add new days: only the NEW ones run; already-finished days are left untouched.
# (extend the `corridas:` list in the yaml and re-run)
python urbantrips/run_all_urbantrips.py --config configs/my_city.yaml

# Resume an interrupted run (reboot, crash): re-run the same command — it picks up
# where it left off. Re-running already-present days no longer crashes.
python urbantrips/run_all_urbantrips.py --config configs/my_city.yaml
```

With nothing pending it ends with *"No hay días pendientes de procesar"* without
doing anything (this used to crash).

#### Reprocessing finished runs: `--reprocesar`

To **rebuild from scratch** one or more already-complete runs (e.g. the source
data changed), pass their names — **items of the `corridas:` list**, not the yaml
filename — comma-separated **without spaces**. It deletes those runs' days and
regenerates them; **every other day is left intact**.

```bash
# config with corridas: ['dia1', 'dia2', 'dia3', 'dia4']

# reprocess only dia1  (dia2, dia3, dia4 stay frozen)
python urbantrips/run_all_urbantrips.py --config configs/my_city.yaml \
  --reprocesar dia1

# reprocess two runs
python urbantrips/run_all_urbantrips.py --config configs/my_city.yaml \
  --reprocesar dia1,dia3
```

Notes:

- The **unit is the run** (an item of `corridas:`, which maps to a
  `<corrida>_trx.csv` file). If a run spans several days, **all** of its days are
  reprocessed.
- The names must be in the config's `corridas:` list.
- `--reprocesar` is incompatible with `--borrar_corrida` (that already rebuilds
  everything).

### Launching the dashboard manually

On Windows, via the `.bat`:

```bat
dashboard.bat                                        REM configuraciones_generales.yaml
dashboard.bat configuraciones_generales_2024.yaml     REM looked up in configs/
dashboard.bat configs\other.yaml                     REM relative or absolute path
```

Or directly:

```bash
streamlit run urbantrips/dashboard/dashboard.py
streamlit run urbantrips/dashboard/dashboard.py -- --config configs/configuraciones_generales_2024.yaml
```

The bare `--` is **required**: without it, streamlit keeps the argument instead of
forwarding it to the script. The `.bat` handles that for you.

The chosen config determines which databases are opened (via `alias_db_insumos`),
and supplies `resolucion_h3`, `epsg_m` and `lineas_contienen_ramales`.

#### Switching runs without restarting

To switch between runs from within the dashboard, create `configs/corridas.yaml`
listing the aliases you want to see:

```yaml
corridas:
  - corrida_2024
  - corrida_2025
```

With that file, a **selector appears in the sidebar**. Picking another run
repoints the configuration and reloads: the effect is the same as having launched
the dashboard with a different `--config`, but without restarting the process.

The file is **optional**: without it there is no selector and the dashboard opens
the run of the config it was launched with, as before. See the
`configs/corridas.yaml.example` template.

The alias alone is enough because the dashboard finds its configuration on its own:

1. **From the copy the run itself stored** in `{alias}_general.duckdb`. Every run
   archives the yaml it was processed with, so that is the configuration that
   actually produced the data — even if the original file was later edited or
   deleted.
2. If that database predates this feature, from the yaml in `configs/` declaring
   that alias.

The sidebar shows which of the two it came from. A run that cannot be resolved, or
that is missing any of its four databases, still appears in the list but disabled
and with the reason, instead of silently vanishing.

---

## Development setup

```bash
git clone https://github.com/EL-BID/UrbanTrips.git
cd UrbanTrips
uv sync
```

Or with pip:

```bash
pip install -e ".[dev]"
```

The `Makefile` mirrors the CI checks:

| Command | Description |
|---|---|
| `make lint` | Style check with Ruff |
| `make unit` | Unit tests with coverage |
| `make integration` | Integration tests with coverage |
| `make build` | Build the package |
| `make ci` | All of the above in sequence |

---

## Acknowledgements

We thank the governments of Buenos Aires, Córdoba, Mendoza, and Bariloche for providing data and participating in discussions that shaped this library.

---

## License

Copyright © 2023. Inter-American Development Bank (IDB). Authorized use. [AM-331-A3](LICENSE.md)

## Authors

Felipe González ([@alephcero](https://github.com/alephcero/))  
Sebastián Anapolsky ([@sanapolsky](https://github.com/sanapolsky/))
