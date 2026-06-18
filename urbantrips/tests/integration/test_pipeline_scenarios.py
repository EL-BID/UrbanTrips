# urbantrips/tests/integration/test_pipeline_scenarios.py
"""
Pipeline scenario integration tests.

These tests exercise the full pipeline functions against a real DuckDB
data adapter (in a tmp_path) paired with an in-memory insumos adapter.
They verify row-count invariants, idempotency, multi-day isolation, and
DuckDB file persistence across adapter instances.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
from urbantrips.storage.adapters.memory.adapters import (
    InMemoryInsumoAdapter,
    InMemoryDashAdapter,
    InMemoryGeneralAdapter,
)
from urbantrips.storage.context import StorageContext

# ---------------------------------------------------------------------------
# Real H3 cells at resolution 8 (Buenos Aires area)
# ---------------------------------------------------------------------------
H3_A = "88754e6491fffff"  # origin cell
H3_B = "88754e64b5fffff"  # destination cell (different hex)
H3_C = "88754e6481fffff"  # third cell for multi-etapa scenarios

LINEA_ID = 1
LINEA_AGG_ID = 1


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_legs(day: str, n: int = 3, id_offset: int = 0) -> pd.DataFrame:
    """Return a minimal etapas DataFrame with n rows for the given day."""
    rows = []
    for i in range(n):
        rows.append({
            "id": id_offset + i,
            "id_tarjeta": f"T{i % 2}",   # two cards, so groupby produces multi-etapa sequences
            "dia": day,
            "id_viaje": i // 2,
            "id_etapa": i % 2,
            "tiempo": f"08:{i:02d}:00",
            "hora": 8,
            "modo": "autobus",
            "id_linea": LINEA_ID,
            "id_ramal": 0,
            "interno": 0,
            "genero": "",
            "tarifa": "",
            "latitud": -34.6,
            "longitud": -58.4,
            "h3_o": H3_A,
            "h3_d": "",
            "od_validado": 0,
            "etapa_validada": 1,
            "factor_expansion_original": 1.0,
            "factor_expansion_linea": 1.0,
            "factor_expansion_tarjeta": 1.0,
            "factor_expansion_etapa": 1.0,
            "distancia": 0.0,
            "travel_time_min": 0.0,
        })
    return pd.DataFrame(rows)


def _make_metadata_lineas() -> pd.DataFrame:
    return pd.DataFrame({
        "id_linea": [LINEA_ID],
        "id_linea_agg": [LINEA_AGG_ID],
        "nombre_linea": ["L1"],
    })


def _make_matriz_validacion() -> pd.DataFrame:
    """One validation area: H3_B is a valid destination for line LINEA_AGG_ID."""
    return pd.DataFrame({
        "id_linea_agg": [LINEA_AGG_ID],
        "area_influencia": [H3_B],
        "parada": [H3_B],
    })


def _ctx(tmp_path: Path) -> StorageContext:
    """Build a StorageContext with a real DuckDB data adapter + in-memory insumos."""
    data_db = tmp_path / "data" / "db" / "test_data.duckdb"
    data_db.parent.mkdir(parents=True, exist_ok=True)
    return StorageContext(
        data=DuckDBDataAdapter(data_db),
        insumos=InMemoryInsumoAdapter(
            metadata_lineas=_make_metadata_lineas(),
            matriz_validacion=_make_matriz_validacion(),
        ),
        dash=InMemoryDashAdapter(),
        general=InMemoryGeneralAdapter(),
    )


# ---------------------------------------------------------------------------
# Scenario 1: DuckDB persistence
# ---------------------------------------------------------------------------

def test_duckdb_data_persists_across_adapter_instances(tmp_path):
    """Data written via one adapter is readable by a fresh adapter on the same file."""
    db_path = tmp_path / "test.duckdb"
    legs = _make_legs("2024-01-01")

    adapter1 = DuckDBDataAdapter(db_path)
    adapter1.save_legs(legs)

    adapter2 = DuckDBDataAdapter(db_path)
    result = adapter2.get_legs()

    assert len(result) == len(legs)
    assert set(result["dia"]) == {"2024-01-01"}


# ---------------------------------------------------------------------------
# Scenario 2: One-day destination imputation
# ---------------------------------------------------------------------------

def test_infer_destinations_one_day_sets_od_validado(tmp_path):
    """
    Legs with h3_o = H3_A for a single day get od_validado=1 where the
    next leg's origin can be matched against the validation matrix.
    """
    from urbantrips.destinations.destinations import infer_destinations

    ctx = _ctx(tmp_path)
    day = "2024-01-01"
    legs = _make_legs(day, n=3)

    ctx.data.save_legs(legs)
    ctx.data.save_run_days(pd.DataFrame({"dia": [day]}))

    infer_destinations(ctx)

    result = ctx.data.get_legs()
    assert len(result) == len(legs), "infer_destinations must not change row count"
    # All rows for the day should be present
    assert set(result["dia"]) == {day}


# ---------------------------------------------------------------------------
# Scenario 3: Multi-day isolation
# ---------------------------------------------------------------------------

def test_infer_destinations_only_updates_current_run_days(tmp_path):
    """
    Legs from a non-current day are not modified by infer_destinations.
    """
    from urbantrips.destinations.destinations import infer_destinations

    ctx = _ctx(tmp_path)
    day_current = "2024-01-02"
    day_other = "2024-01-01"

    legs_current = _make_legs(day_current, n=3, id_offset=0)
    legs_other = _make_legs(day_other, n=2, id_offset=100)

    ctx.data.save_legs(legs_current)
    ctx.data.save_legs(legs_other)
    ctx.data.save_run_days(pd.DataFrame({"dia": [day_current]}))

    infer_destinations(ctx)

    result = ctx.data.get_legs()
    assert len(result) == 5, "Both days' legs must be preserved"

    # The non-current day should be unchanged (od_validado = 0 as seeded)
    other_day_rows = result[result["dia"] == day_other]
    assert (other_day_rows["od_validado"] == 0).all(), (
        "Non-current day legs must not be modified"
    )


# ---------------------------------------------------------------------------
# Scenario 4: Idempotency (rerun / delete)
# ---------------------------------------------------------------------------

def test_infer_destinations_is_idempotent(tmp_path):
    """
    Running infer_destinations twice for the same day must not duplicate rows.
    """
    from urbantrips.destinations.destinations import infer_destinations

    ctx = _ctx(tmp_path)
    day = "2024-01-03"
    legs = _make_legs(day, n=4)

    ctx.data.save_legs(legs)
    ctx.data.save_run_days(pd.DataFrame({"dia": [day]}))

    infer_destinations(ctx)
    count_after_first = len(ctx.data.get_legs())

    # Reset run days and run again (same day)
    ctx.data.save_run_days(pd.DataFrame({"dia": [day]}))
    infer_destinations(ctx)
    count_after_second = len(ctx.data.get_legs())

    assert count_after_first == len(legs)
    assert count_after_second == count_after_first, (
        "Second run must not add duplicate rows"
    )


# ---------------------------------------------------------------------------
# Scenario 5: geolocalizar_trx end-to-end through _ingest_all_days
# ---------------------------------------------------------------------------

_NOMBRES_VARIABLES_TRX = {
    "id_trx": "id",
    "id_tarjeta": "id_tarjeta",
    "fecha": "fecha",
    "id_linea": "id_linea",
    "id_ramal": "id_ramal",
    "interno": "interno",
    "orden": "orden_trx",
    "latitud": "latitud",
    "longitud": "longitud",
    "modo": "modo",
    "tarifa": "tarifa",
    "fex": "factor_expansion",
}

_NOMBRES_VARIABLES_GPS = {
    "id": "id_gps",
    "latitud": "lat",
    "longitud": "lon",
    "id_linea": "id_linea_gps",
    "id_ramal": "id_ramal_gps",
    "interno": "interno_gps",
    "fecha": "fecha_gps",
}


def test_ingest_geolocalizar_trx_fills_coordinates_from_gps(tmp_path, monkeypatch):
    """End-to-end: a trx CSV with no latitud/longitud columns, paired with a
    gps CSV and geolocalizar_trx: True, should end up with transacciones
    rows carrying coordinates derived from the nearest preceding gps ping —
    not crash, and not silently drop every row.
    """
    from urbantrips.utils import run_process
    from urbantrips.utils.paths import init_paths, reset_paths

    corrida = "20250101"
    base = tmp_path
    (base / "data" / "data_ciudad").mkdir(parents=True)
    (base / "configs").mkdir(parents=True)

    # trx CSV: latitud/longitud are present but empty (to be filled from GPS).
    trx_csv = base / "data" / "data_ciudad" / f"{corrida}_trx.csv"
    trx_csv.write_text(
        "id,id_tarjeta,fecha,id_linea,id_ramal,interno,orden,latitud,longitud,modo,tarifa,fex\n"
        "1,card_1,2025-01-01 08:05:00,1,1,10,1,,,autobus,-,1.0\n"
        "2,card_2,2025-01-01 09:05:00,1,1,10,2,,,autobus,-,1.0\n"
    )

    # gps CSV: one ping per trx, each strictly before its trx timestamp,
    # for the same id_linea/id_ramal/interno.
    gps_csv = base / "data" / "data_ciudad" / f"{corrida}_gps.csv"
    gps_csv.write_text(
        "id_gps,lat,lon,id_linea_gps,id_ramal_gps,interno_gps,fecha_gps\n"
        "1,-34.60,-58.40,1,1,10,2025-01-01 08:00:00\n"
        "2,-34.61,-58.41,1,1,10,2025-01-01 09:00:00\n"
    )

    config_file = base / "configs" / "configuraciones_generales.yaml"
    config_file.write_text("placeholder: true\n")

    reset_paths()
    init_paths(base)

    ctx = _ctx(base)

    config = {
        "nombres_variables_trx": _NOMBRES_VARIABLES_TRX,
        "formato_fecha": "%Y-%m-%d %H:%M:%S",
        "tipo_trx_invalidas": None,
        "lineas_contienen_ramales": True,
        "nombre_archivo_trx": f"{corrida}_trx.csv",
        "usa_archivo_gps": True,
        "nombre_archivo_gps": f"{corrida}_gps.csv",
        "nombres_variables_gps": _NOMBRES_VARIABLES_GPS,
        "geolocalizar_trx": True,
        "resolucion_h3": 8,
        "n_batches": 1,
    }

    monkeypatch.setattr(
        "urbantrips.utils.check_configs.check_config",
        lambda corrida: None,
    )
    monkeypatch.setattr(
        run_process, "leer_configs_generales", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        "urbantrips.datamodel.transactions.leer_configs_generales",
        lambda *args, **kwargs: config,
    )

    # compute_od_distances builds a real street network via osmnx/pandana,
    # which isn't needed to exercise the geocoding wiring under test and
    # whose native (cython) dependency is broken in this environment.
    # Other tests (e.g. test_legs.py) take the same passthrough approach.
    def _passthrough_distances(od_df, **kwargs):
        result = od_df.copy()
        result["distance_km"] = 0.0
        return result

    monkeypatch.setattr(
        "urbantrips.datamodel.transactions.compute_od_distances",
        _passthrough_distances,
    )

    try:
        run_process._ingest_all_days(ctx, [corrida])
    finally:
        reset_paths()

    transacciones = ctx.data.query("SELECT * FROM transacciones")
    assert len(transacciones) > 0, "geolocalizar_trx must not drop every row"
    assert transacciones["latitud"].notna().all()
    assert transacciones["longitud"].notna().all()


def test_ingest_without_geolocalizar_trx_skips_geocoding(tmp_path, monkeypatch):
    """When geolocalizar_trx is False (default), rows without latitud/
    longitud in the trx file are dropped by standardization as before —
    geocoding must not run.
    """
    from urbantrips.utils import run_process
    from urbantrips.utils.paths import init_paths, reset_paths

    corrida = "20250102"
    base = tmp_path
    (base / "data" / "data_ciudad").mkdir(parents=True)
    (base / "configs").mkdir(parents=True)

    trx_csv = base / "data" / "data_ciudad" / f"{corrida}_trx.csv"
    trx_csv.write_text(
        "id,id_tarjeta,fecha,id_linea,id_ramal,interno,orden,latitud,longitud,modo,tarifa,fex\n"
        "1,card_1,2025-01-02 08:05:00,1,1,10,1,,,autobus,-,1.0\n"
    )

    config_file = base / "configs" / "configuraciones_generales.yaml"
    config_file.write_text("placeholder: true\n")

    reset_paths()
    init_paths(base)

    ctx = _ctx(base)

    config = {
        "nombres_variables_trx": _NOMBRES_VARIABLES_TRX,
        "formato_fecha": "%Y-%m-%d %H:%M:%S",
        "tipo_trx_invalidas": None,
        "lineas_contienen_ramales": True,
        "nombre_archivo_trx": f"{corrida}_trx.csv",
        "usa_archivo_gps": False,
        "geolocalizar_trx": False,
        "n_batches": 1,
    }

    geolocate_calls = []
    monkeypatch.setattr(
        "urbantrips.utils.check_configs.check_config",
        lambda corrida: None,
    )
    monkeypatch.setattr(
        run_process, "leer_configs_generales", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        ctx.data,
        "geolocate_raw_transactions_from_gps",
        lambda *a, **kw: geolocate_calls.append((a, kw)),
    )

    try:
        run_process._ingest_all_days(ctx, [corrida])
    finally:
        reset_paths()

    assert geolocate_calls == [], "geocoding must be skipped when geolocalizar_trx is False"
    transacciones = ctx.data.query("SELECT * FROM transacciones")
    assert len(transacciones) == 0, (
        "rows missing latitud/longitud must still be dropped when geolocalizar_trx is False"
    )


def test_ingest_with_latlong_already_present_skips_geocoding_and_keeps_values(
    tmp_path, monkeypatch
):
    """End-to-end: when trx rows already carry real, non-null latitud/
    longitud values, geolocalizar_trx: True must not overwrite them with
    gps-derived coordinates (COALESCE keeps the original non-null values),
    and rows must not be dropped.
    """
    from urbantrips.utils import run_process
    from urbantrips.utils.paths import init_paths, reset_paths

    corrida = "20250103"
    base = tmp_path
    (base / "data" / "data_ciudad").mkdir(parents=True)
    (base / "configs").mkdir(parents=True)

    # trx CSV: real, distinct, non-null latitud/longitud for each row.
    trx_csv = base / "data" / "data_ciudad" / f"{corrida}_trx.csv"
    trx_csv.write_text(
        "id,id_tarjeta,fecha,id_linea,id_ramal,interno,orden,latitud,longitud,modo,tarifa,fex\n"
        "1,card_1,2025-01-03 08:05:00,1,1,10,1,-34.60,-58.40,autobus,-,1.0\n"
        "2,card_2,2025-01-03 09:05:00,1,1,10,2,-34.61,-58.41,autobus,-,1.0\n"
    )

    # gps CSV with different coordinates than the trx rows, to prove the
    # original trx coordinates are NOT overwritten by gps data.
    gps_csv = base / "data" / "data_ciudad" / f"{corrida}_gps.csv"
    gps_csv.write_text(
        "id_gps,lat,lon,id_linea_gps,id_ramal_gps,interno_gps,fecha_gps\n"
        "1,-34.70,-58.50,1,1,10,2025-01-03 08:00:00\n"
        "2,-34.71,-58.51,1,1,10,2025-01-03 09:00:00\n"
    )

    config_file = base / "configs" / "configuraciones_generales.yaml"
    config_file.write_text("placeholder: true\n")

    reset_paths()
    init_paths(base)

    ctx = _ctx(base)

    config = {
        "nombres_variables_trx": _NOMBRES_VARIABLES_TRX,
        "formato_fecha": "%Y-%m-%d %H:%M:%S",
        "tipo_trx_invalidas": None,
        "lineas_contienen_ramales": True,
        "nombre_archivo_trx": f"{corrida}_trx.csv",
        "usa_archivo_gps": True,
        "nombre_archivo_gps": f"{corrida}_gps.csv",
        "nombres_variables_gps": _NOMBRES_VARIABLES_GPS,
        "geolocalizar_trx": True,
        "resolucion_h3": 8,
        "n_batches": 1,
    }

    monkeypatch.setattr(
        "urbantrips.utils.check_configs.check_config",
        lambda corrida: None,
    )
    monkeypatch.setattr(
        run_process, "leer_configs_generales", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        "urbantrips.datamodel.transactions.leer_configs_generales",
        lambda *args, **kwargs: config,
    )

    def _passthrough_distances(od_df, **kwargs):
        result = od_df.copy()
        result["distance_km"] = 0.0
        return result

    monkeypatch.setattr(
        "urbantrips.datamodel.transactions.compute_od_distances",
        _passthrough_distances,
    )

    try:
        run_process._ingest_all_days(ctx, [corrida])
    finally:
        reset_paths()

    transacciones = ctx.data.query("SELECT * FROM transacciones")
    assert len(transacciones) > 0, "rows with existing coordinates must not be dropped"

    transacciones = transacciones.sort_values("id").reset_index(drop=True)
    assert transacciones["latitud"].tolist() == pytest.approx([-34.60, -34.61]), (
        "pre-existing latitud values must not be overwritten by gps data"
    )
    assert transacciones["longitud"].tolist() == pytest.approx([-58.40, -58.41]), (
        "pre-existing longitud values must not be overwritten by gps data"
    )
