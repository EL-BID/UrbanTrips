"""`process_and_upload_gps_table` de punta a punta contra una base DuckDB real.

Lo que fija: el odómetro (`distance_servicio_mts`) se calcula sobre TODO el archivo y
no día por día —la traza de un vehículo cruza la medianoche—, aunque el guardado sí sea
por día. Es la propiedad que había que preservar al sacar el `sort_values` del mes
entero, que era lo que se llevaba la RAM en una corrida de un mes en un solo archivo.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
from urbantrips.storage.context import StorageContext


_NOMBRES_VARIABLES_GPS = {
    "id": "id_gps",
    "latitud": "latitud_gps",
    "longitud": "longitud_gps",
    "id_linea": "id_linea_gps",
    "id_ramal": "id_ramal_gps",
    "interno": "interno_gps",
    "fecha": "fecha_gps",
    "distance_servicio_mts_agg": "distancia_serv_mts",
}

_CONFIGS = {"lineas_contienen_ramales": True, "resolucion_h3": 8}

# Dos vehículos de líneas distintas, cada uno con pings a los dos lados de la
# medianoche y el odómetro acumulado corriendo entre días. Las filas van
# DESORDENADAS a propósito (así llega el csv del cliente).
_FILAS = [
    # id, linea, ramal, interno, fecha, odómetro acumulado
    (4, 10, 101, 7, "02/07/2026 00:30:00", 1500),
    (1, 10, 101, 7, "01/07/2026 22:00:00", 1000),
    (7, 20, 201, 8, "02/07/2026 01:00:00", 900),
    (3, 10, 101, 7, "01/07/2026 23:30:00", 1200),
    (5, 20, 201, 8, "01/07/2026 22:30:00", 500),
    (2, 10, 101, 7, "01/07/2026 23:00:00", 1100),
    (6, 20, 201, 8, "01/07/2026 23:45:00", 700),
]


@pytest.fixture
def ctx_y_csv(tmp_path):
    csv = tmp_path / "gps.csv"
    filas = ["id_gps,id_linea_gps,id_ramal_gps,interno_gps,fecha_gps,latitud_gps,longitud_gps,distancia_serv_mts"]
    for id_gps, linea, ramal, interno, fecha, odo in _FILAS:
        filas.append(f"{id_gps},{linea},{ramal},{interno},{fecha},-34.6,-58.4,{odo}")
    csv.write_text("\n".join(filas))

    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter.save_run_days(pd.DataFrame({"dia": ["2026-07-01", "2026-07-02"]}))
    ctx = StorageContext(data=adapter, insumos=None, dash=None, general=None)
    return ctx, csv


def _correr(ctx, csv):
    from urbantrips.datamodel.transactions import process_and_upload_gps_table

    with patch(
        "urbantrips.datamodel.transactions.leer_configs_generales",
        return_value=_CONFIGS,
    ), patch(
        "urbantrips.datamodel.transactions.bbox_area_estudio",
        return_value=(-99.0, -99.0, 99.0, 99.0),
    ), patch(
        "urbantrips.datamodel.transactions.compute_distance_km_gps",
        side_effect=lambda df, ctx: df.assign(distance_km=0.0),
    ), patch(
        "urbantrips.datamodel.transactions.geo.referenciar_h3",
        side_effect=lambda df, *a, **kw: df.assign(h3="h3"),
    ):
        process_and_upload_gps_table(
            ctx,
            nombre_archivo_gps=str(csv),
            nombres_variables_gps=dict(_NOMBRES_VARIABLES_GPS),
            formato_fecha="%d/%m/%Y %H:%M:%S",
        )


def test_odometro_se_calcula_cruzando_la_medianoche(ctx_y_csv):
    ctx, csv = ctx_y_csv
    _correr(ctx, csv)

    gps = ctx.data.query(
        "SELECT id_original, dia, id_linea, interno, distance_servicio_mts, "
        "distance_servicio_mts_agg FROM gps ORDER BY id_linea, interno, id_original"
    )

    # línea 10: 1000 -> 1100 -> 1200 -> 1500 (el último ping ya es del día 2)
    linea10 = gps[gps.id_linea == 10]
    assert linea10["distance_servicio_mts"].tolist() == [0, 100, 100, 300]
    assert linea10["dia"].tolist() == ["2026-07-01"] * 3 + ["2026-07-02"]
    # el salto de 300 es entre el último ping del día 1 y el primero del día 2:
    # calculado día por día habría dado 0
    assert linea10.iloc[-1]["distance_servicio_mts"] == 300

    # línea 20: 500 -> 700 -> 900
    linea20 = gps[gps.id_linea == 20]
    assert linea20["distance_servicio_mts"].tolist() == [0, 200, 200]


def test_dia_se_guarda_como_fecha_iso(ctx_y_csv):
    ctx, csv = ctx_y_csv
    _correr(ctx, csv)

    dias = ctx.data.query("SELECT DISTINCT dia FROM gps ORDER BY dia")["dia"].tolist()
    assert dias == ["2026-07-01", "2026-07-02"]


def test_solo_se_guardan_los_dias_de_la_corrida(ctx_y_csv):
    """El loop por día filtra por run_days; el odómetro igual usa todo el archivo."""
    ctx, csv = ctx_y_csv
    ctx.data.execute("DELETE FROM dias_ultima_corrida WHERE dia = '2026-07-01'")
    _correr(ctx, csv)

    gps = ctx.data.query("SELECT dia, distance_servicio_mts FROM gps")
    assert gps["dia"].unique().tolist() == ["2026-07-02"]
    # el ping del día 2 conserva el salto contra el último del día 1
    assert sorted(gps["distance_servicio_mts"].tolist()) == [200, 300]
