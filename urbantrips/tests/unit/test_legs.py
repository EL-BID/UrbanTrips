import pandas as pd
import pytest
from types import SimpleNamespace
from urbantrips.datamodel.legs import (
    crear_viaje_id_acumulada,
    asignar_id_viaje_etapa_fecha_completa,
    asignar_id_viaje_etapa_orden_trx,
    cambiar_id_tarjeta_trx_simul_fecha,
    crear_delta_trx,
    assign_gps_origin,
)


# --- crear_viaje_id_acumulada ---

def test_crear_viaje_id_acumulada_120min(df_test_id_viaje):
    dia = df_test_id_viaje.dia == "2022-08-11"
    tarj = df_test_id_viaje.id_tarjeta == 1
    sub = df_test_id_viaje.loc[dia & tarj]
    assert crear_viaje_id_acumulada(sub, ventana_viajes=120 * 60) == [1, 1, 2, 3, 3]


def test_crear_viaje_id_acumulada_150min(df_test_id_viaje):
    dia = df_test_id_viaje.dia == "2022-08-11"
    tarj = df_test_id_viaje.id_tarjeta == 1
    sub = df_test_id_viaje.loc[dia & tarj]
    assert crear_viaje_id_acumulada(sub, ventana_viajes=150 * 60) == [1, 1, 1, 2, 2]


def test_crear_viaje_id_acumulada_30min(df_test_id_viaje):
    dia = df_test_id_viaje.dia == "2022-08-11"
    tarj = df_test_id_viaje.id_tarjeta == 1
    sub = df_test_id_viaje.loc[dia & tarj]
    assert crear_viaje_id_acumulada(sub, ventana_viajes=30 * 60) == [1, 1, 2, 3, 4]


def test_crear_viaje_id_acumulada_29min(df_test_id_viaje):
    dia = df_test_id_viaje.dia == "2022-08-11"
    tarj = df_test_id_viaje.id_tarjeta == 1
    sub = df_test_id_viaje.loc[dia & tarj]
    assert crear_viaje_id_acumulada(sub, ventana_viajes=29 * 60) == [1, 2, 3, 4, 5]


# --- asignar_id_viaje_etapa_fecha_completa ---

def test_asignar_id_viaje_etapa_fecha_completa(df_test_id_viaje):
    trx = df_test_id_viaje.copy().rename(columns={"fecha_dt": "fecha"})
    result = asignar_id_viaje_etapa_fecha_completa(trx, ventana_viajes=120)
    # With groupby=["id_tarjeta"] trip IDs are continuous across days per card.
    # card 1: day1=[1,1,2,3,3], day2 (delta=0 extends trip3)=[3,3,4,5,5]
    # card 2: day1=[1,1], day2 (delta=0 extends trip1)=[1,1]
    assert (result.id_viaje == [1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 1, 1, 1, 1]).all()
    assert (result.id_etapa == [1, 2, 1, 1, 2, 3, 4, 1, 1, 2, 1, 2, 3, 4]).all()


# --- asignar_id_viaje_etapa_orden_trx ---

def test_asignar_id_viaje_etapa_orden_trx_simple(df_trx):
    df_trx["tiempo"] = None
    result = asignar_id_viaje_etapa_orden_trx(df_trx)

    simple = result.loc[result.id_tarjeta == "37030208"]
    assert len(simple) == 4
    assert (simple.id_viaje == [1, 2, 3, 4]).all()
    assert simple.id_etapa.unique()[0] == 1


def test_asignar_id_viaje_etapa_orden_trx_multimodal(df_trx):
    df_trx["tiempo"] = None
    result = asignar_id_viaje_etapa_orden_trx(df_trx)

    multim = result.loc[result.id_tarjeta == "3839538659"]
    assert (multim.id_viaje == [1] * 3 + [2] * 3).all()
    assert (multim.id_etapa == [1, 2, 3] * 2).all()


def test_asignar_id_viaje_etapa_orden_trx_checkout(df_trx):
    df_trx["tiempo"] = None
    result = asignar_id_viaje_etapa_orden_trx(df_trx)

    chkout = result.loc[result.id_tarjeta == "37035823"]
    chkout = chkout.loc[chkout.id_viaje.isin([2, 3])]
    assert (chkout.id_viaje == [2, 2, 3]).all()
    assert (chkout.id_etapa == [1, 2, 1]).all()


# --- cambiar_id_tarjeta_trx_simul_fecha ---

def _make_dup_trx(df_test_id_viaje):
    extra = pd.DataFrame({
        "id": 17, "fecha_dt": "2022-08-12 09:33:00", "id_tarjeta": 2,
        "dia": "2022-08-12", "hora_shift": "2022-08-12 09:30:00",
        "delta": 3 * 60, "hora": "09:33:00",
    }, index=[0])
    trx = pd.concat([df_test_id_viaje, extra]).reset_index(drop=True).copy()
    trx["id_tarjeta"] = trx["id_tarjeta"].map(str)
    trx["id_linea"] = 5
    trx["interno"] = 10
    trx = trx.rename(columns={"fecha_dt": "fecha"}).reset_index(drop=True)
    return trx


def test_cambiar_id_tarjeta_5min_window(df_test_id_viaje):
    trx = _make_dup_trx(df_test_id_viaje)
    result, dupes = cambiar_id_tarjeta_trx_simul_fecha(trx, ventana_duplicado=5)
    assert len(dupes) == 2
    assert (dupes.id_tarjeta_original == ["2", "2"]).all()
    assert (dupes.id_tarjeta_nuevo == ["2_1", "2_2"]).all()
    assert (result.loc[result["id"].isin([15, 16, 17]), "id_tarjeta"] == ["2_0", "2_1", "2_2"]).all()


def test_cambiar_id_tarjeta_1min_window(df_test_id_viaje):
    trx = _make_dup_trx(df_test_id_viaje)
    result, dupes = cambiar_id_tarjeta_trx_simul_fecha(trx, ventana_duplicado=1)
    assert len(dupes) == 1
    assert (dupes.id_tarjeta_original == ["2"]).all()
    assert (dupes.id_tarjeta_nuevo == ["2_1"]).all()
    assert (result.loc[result["id"].isin([15, 16, 17]), "id_tarjeta"] == ["2_0", "2_1", "2_0"]).all()


# --- crear_delta_trx ---

def test_crear_delta_trx_basic():
    trx = pd.DataFrame({
        "dia": ["2022-08-11"] * 3,
        "id_tarjeta": ["1"] * 3,
        "fecha": pd.to_datetime(["2022-08-11 09:00", "2022-08-11 09:30", "2022-08-11 11:00"]),
    })
    result = crear_delta_trx(trx)
    assert result["delta"].iloc[0] == 0
    assert result["delta"].iloc[1] == 30 * 60
    assert result["delta"].iloc[2] == 90 * 60


def test_crear_delta_trx_resets_across_days():
    trx = pd.DataFrame({
        "dia": ["2022-08-11", "2022-08-12"],
        "id_tarjeta": ["1", "1"],
        "fecha": pd.to_datetime(["2022-08-11 09:00", "2022-08-12 09:00"]),
    })
    result = crear_delta_trx(trx)
    assert result.loc[result.dia == "2022-08-11", "delta"].iloc[0] == 0
    assert result.loc[result.dia == "2022-08-12", "delta"].iloc[0] == 0


# --- multi-day correctness contracts ---

def test_multiday_trip_ids_restart_per_day(df_trx_multiday):
    """id_viaje is continuous per card across days (no longer resets per day).
    Day 1 starts at 1; day 2 continues from where day 1 left off."""
    result = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    # Day 1: trips 1-3 (fixture gaps: 0,45,75min → trip1; 6h → trip2; 90min → trip3)
    # Day 2: continues from trip 3 (delta=0 from day reset), then trips 4,5
    # Day 3: continues from trip 5 (delta=0 from day reset), then trips 6,7
    day1_min = result.loc[result.dia == "2022-08-11"].groupby("id_tarjeta")["id_viaje"].min()
    assert (day1_min == 1).all(), "First day trips must start at 1"
    day2_min = result.loc[result.dia == "2022-08-12"].groupby("id_tarjeta")["id_viaje"].min()
    assert (day2_min > 1).all(), "Subsequent days must continue trip IDs from previous day"


def test_multiday_leg_ids_restart_per_trip(df_trx_multiday):
    """id_etapa must start at 1 for each (id_tarjeta, id_viaje) trip start."""
    result = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    min_leg_ids = result.groupby(["id_tarjeta", "id_viaje"])["id_etapa"].min()
    assert (min_leg_ids == 1).all()


def test_multiday_trip_count_matches_perday_processing(df_trx_multiday):
    """With card-level groupby, trips span days and exact window boundaries
    remain in the current trip. Total unique trips per card: 6."""
    result = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    total_trips = result.groupby("id_tarjeta")["id_viaje"].max()
    assert (total_trips == 6).all(), f"Expected 6 trips per card, got: {total_trips.tolist()}"


def test_multiday_known_trip_structure(df_trx_multiday):
    """Fixture has 5 trx per (day, card). With card-level groupby, trip IDs are
    continuous across days. Day 1: [1,1,1,2,2]; day 2 continues from trip 2:
    [2,3,3,4,4]; day 3 continues from trip 4: [4,5,5,6,6]."""
    result = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    expected = {
        "2022-08-11": [1, 1, 1, 2, 2],
        "2022-08-12": [2, 3, 3, 4, 4],
        "2022-08-13": [4, 5, 5, 6, 6],
    }
    for (dia, card), group in result.groupby(["dia", "id_tarjeta"]):
        group = group.sort_values(["id_viaje", "id_etapa"])
        assert group["id_viaje"].tolist() == expected[dia], \
            f"Unexpected trip structure for dia={dia}, card={card}"


def test_midnight_crossing_trip_stays_unified():
    """A trip that starts before midnight and continues after should be one trip."""
    df = pd.DataFrame({
        "id": [1, 2],
        "id_tarjeta": ["card_A", "card_A"],
        "dia": ["2022-08-11", "2022-08-12"],
        "fecha": pd.to_datetime(["2022-08-11 23:45:00", "2022-08-12 00:10:00"]),
        "hora": [23, 0],
    })
    df["hora_shift"] = df.groupby("id_tarjeta")["fecha"].shift(1)
    df["delta"] = (df["fecha"] - df["hora_shift"]).dt.total_seconds().fillna(0).astype(int)

    result = asignar_id_viaje_etapa_fecha_completa(df, ventana_viajes=90)
    # Both legs should be in trip 1 — 25-minute gap is within 90-minute window
    assert result["id_viaje"].tolist() == [1, 1]


def test_assign_gps_origin_uses_narrow_sql_reads(monkeypatch):
    from urbantrips.datamodel import legs as legs_module

    monkeypatch.setattr(
        legs_module,
        "leer_configs_generales",
        lambda: {"nombre_archivo_gps": "gps.csv"},
    )

    class _Data:
        def __init__(self):
            self.queries = []
            self.saved = {}

        def query(self, sql):
            self.queries.append(sql)
            if "FROM etapas" in sql:
                return pd.DataFrame(
                    {
                        "dia": ["2024-01-01"],
                        "id_linea": [1],
                        "id_ramal": [10],
                        "interno": [100],
                        "tiempo": ["08:00:00"],
                        "id": [11],
                    }
                )
            if "FROM gps" in sql:
                return pd.DataFrame(
                    {
                        "dia": ["2024-01-01"],
                        "id_linea": [1],
                        "id_ramal": [10],
                        "interno": [100],
                        "fecha": [1704096000],
                        "id": [22],
                    }
                )
            raise AssertionError(sql)

        def get_legs(self):
            raise AssertionError("assign_gps_origin should not call get_legs")

        def get_gps(self):
            raise AssertionError("assign_gps_origin should not call get_gps")

        def save_raw(self, df, table_name):
            self.saved[table_name] = df.copy()

    data = _Data()
    assign_gps_origin(SimpleNamespace(data=data))

    assert len(data.queries) == 2
    assert "SELECT e.dia, e.id_linea, e.id_ramal, e.interno, e.tiempo, e.id" in data.queries[0]
    assert "SELECT g.dia, g.id_linea, g.id_ramal, g.interno, g.fecha, g.id" in data.queries[1]
    assert "SELECT e.*" not in data.queries[0]
    assert "SELECT g.*" not in data.queries[1]
    result = data.saved["legs_to_gps_origin"]
    assert result[["id_legs", "id_gps"]].iloc[0].tolist() == [11, 22]


def test_same_day_trips_split_correctly_after_midnight_fix():
    """Normal same-day trip splitting is unaffected by the groupby change."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "id_tarjeta": ["card_B", "card_B", "card_B"],
        "dia": ["2022-08-11"] * 3,
        "fecha": pd.to_datetime(["2022-08-11 08:00", "2022-08-11 09:00", "2022-08-11 16:00"]),
        "hora": [8, 9, 16],
    })
    df["hora_shift"] = df.groupby("id_tarjeta")["fecha"].shift(1)
    df["delta"] = (df["fecha"] - df["hora_shift"]).dt.total_seconds().fillna(0).astype(int)

    result = asignar_id_viaje_etapa_fecha_completa(df, ventana_viajes=90)
    # 8:00→9:00 = 60min (within window) → trip 1; 9:00→16:00 = 420min → trip 2
    assert result["id_viaje"].tolist() == [1, 1, 2]
