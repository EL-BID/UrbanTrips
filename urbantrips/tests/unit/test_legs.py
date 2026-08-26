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
    assign_time_distances,
    pago_doble_tarjeta,
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
    """El dia es un corte duro: cada dia reinicia la numeracion.

    Los dos dias del fixture tienen los mismos horarios para la tarjeta 1, asi
    que tienen que dar la misma estructura. Antes del fix de 2026-08-14 el dia 2
    continuaba la numeracion del dia 1 ([3,3,4,5,5] en vez de [1,1,2,3,3]).
    """
    trx = df_test_id_viaje.copy().rename(columns={"fecha_dt": "fecha"})
    # se ordena en el test para no depender del sort interno de la funcion
    result = asignar_id_viaje_etapa_fecha_completa(trx, ventana_viajes=120).sort_values(
        ["dia", "id_tarjeta", "fecha"]
    )
    # 08-11: tarjeta 1 [12:00,12:30 | 14:30 | 18:30,19:30], tarjeta 2 [09:30,10:30]
    # 08-12: idem tarjeta 1; tarjeta 2 [09:30, 09:31]
    assert result.id_viaje.tolist() == [1, 1, 2, 3, 3, 1, 1,
                                        1, 1, 2, 3, 3, 1, 1]
    assert result.id_etapa.tolist() == [1, 2, 1, 1, 2, 1, 2,
                                        1, 2, 1, 1, 2, 1, 2]


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
    """id_viaje reinicia en 1 en CADA dia. El dia es un corte duro.

    Antes del fix de 2026-08-14 el groupby era solo por id_tarjeta y la
    numeracion se arrastraba de un dia al siguiente, porque crear_delta_trx deja
    delta=0 en la primera trx de cada dia y eso hacia invisible la frontera.
    """
    result = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    for dia in ("2022-08-11", "2022-08-12", "2022-08-13"):
        minimos = result.loc[result.dia == dia].groupby("id_tarjeta")["id_viaje"].min()
        assert (minimos == 1).all(), f"{dia}: los viajes deben empezar en 1"


def test_multiday_leg_ids_restart_per_trip(df_trx_multiday):
    """id_etapa must start at 1 for each (id_tarjeta, id_viaje) trip start."""
    result = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    min_leg_ids = result.groupby(["id_tarjeta", "id_viaje"])["id_etapa"].min()
    assert (min_leg_ids == 1).all()


def test_multiday_trip_count_matches_perday_processing(df_trx_multiday):
    """Procesar N dias juntos da lo mismo que procesarlos de a uno.

    Es el contrato central del fix: el resultado no puede depender de cuantos
    dias entran en la corrida.
    """
    juntos = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    for dia in sorted(df_trx_multiday.dia.unique()):
        solo = asignar_id_viaje_etapa_fecha_completa(
            df_trx_multiday[df_trx_multiday.dia == dia].copy(), ventana_viajes=120
        )
        a = juntos[juntos.dia == dia].sort_values("id")[["id", "id_viaje", "id_etapa"]]
        b = solo.sort_values("id")[["id", "id_viaje", "id_etapa"]]
        assert a.reset_index(drop=True).equals(b.reset_index(drop=True)), (
            f"{dia}: procesar junto con otros dias cambio el resultado"
        )


def test_multiday_known_trip_structure(df_trx_multiday):
    """Fixture: 5 trx por (dia, tarjeta), huecos 0/45/75min, 6h, 90min.
    Con ventana=120 y corte por dia, TODOS los dias dan [1,1,1,2,2]."""
    result = asignar_id_viaje_etapa_fecha_completa(df_trx_multiday.copy(), ventana_viajes=120)
    for (dia, card), group in result.groupby(["dia", "id_tarjeta"]):
        group = group.sort_values(["id_viaje", "id_etapa"])
        assert group["id_viaje"].tolist() == [1, 1, 1, 2, 2], \
            f"Unexpected trip structure for dia={dia}, card={card}"


def test_midnight_crossing_trip_splits_by_day():
    """El dia es un corte DURO: un viaje NO cruza la medianoche.

    Decision del usuario (2026-08-14). Antes estos dos taps quedaban en un solo
    viaje. Ojo: no alcanza con mirar id_viaje —cada dia reinicia en 1, asi que
    ambos dan 1—; hay que verificar que son viajes DISTINTOS, o sea (dia, id_viaje).
    """
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
    assert result.groupby(["dia", "id_viaje"]).ngroups == 2, \
        "las dos etapas deben quedar en viajes distintos, uno por dia"
    assert result["id_etapa"].tolist() == [1, 1], \
        "cada una es la primera etapa de su propio viaje"


def test_trip_ids_ignore_days_outside_the_run_window():
    """Taps de dias MUY separados nunca pueden caer en el mismo viaje.

    Reproduccion del bug de 2026-08-14: con delta=0 en la primera trx de cada dia
    y groupby solo por tarjeta, dos taps separados por SIETE DIAS terminaban en el
    mismo viaje.
    """
    filas = []
    for dia in ("2026-03-09", "2026-03-16"):
        for hhmm in ("08:00:00", "08:20:00", "18:00:00"):
            filas.append({"id_tarjeta": "CARD_A", "dia": dia,
                          "fecha": pd.Timestamp(f"{dia} {hhmm}")})
    df = pd.DataFrame(filas)
    df = crear_delta_trx(df)

    result = asignar_id_viaje_etapa_fecha_completa(df, ventana_viajes=60)
    d1 = result[result.dia == "2026-03-09"].sort_values("fecha")
    d2 = result[result.dia == "2026-03-16"].sort_values("fecha")
    assert d1["id_viaje"].tolist() == d2["id_viaje"].tolist() == [1, 1, 2]
    assert d1["id_etapa"].tolist() == d2["id_etapa"].tolist() == [1, 2, 1]


def test_assign_gps_origin_uses_narrow_sql_reads(monkeypatch):
    from urbantrips.datamodel import legs as legs_module

    monkeypatch.setattr(
        legs_module,
        "leer_configs_generales",
        # acepta kwargs: assign_gps_origin llama leer_configs_generales(autogenerado=False)
        lambda *a, **k: {"nombre_archivo_gps": "gps.csv", "usa_archivo_gps": True},
    )

    class _Data:
        def __init__(self):
            self.queries = []
            self.saved = {}

        def get_run_days(self):
            # assign_gps_origin procesa día por día acotado a la corrida
            return pd.DataFrame({"dia": ["2024-01-01"]})

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

        def append_raw(self, df, table_name):
            # el write pasó a ser append por día (día-scoping incremental)
            prev = self.saved.get(table_name)
            self.saved[table_name] = (
                pd.concat([prev, df], ignore_index=True) if prev is not None
                else df.copy()
            )

        def execute(self, sql):
            pass

    data = _Data()
    assign_gps_origin(SimpleNamespace(data=data))

    # Las queries de _estimated_day_footprint_gb (LIMIT 0 para dtypes, count por
    # día) no leen datos del día: se descartan para chequear las lecturas reales.
    lecturas = [
        q for q in data.queries
        if "LIMIT 0" not in q and "count(*)" not in q
    ]
    assert len(lecturas) == 2
    assert "SELECT e.dia, e.id_linea, e.id_ramal, e.interno, e.tiempo, e.id" in lecturas[0]
    assert "SELECT g.dia, g.id_linea, g.id_ramal, g.interno, g.fecha, g.id" in lecturas[1]
    assert "SELECT e.*" not in lecturas[0]
    assert "SELECT g.*" not in lecturas[1]
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


# --- assign_time_distances (no-GPS branch) ---

def test_assign_time_distances_no_gps_id_etapa_not_null(monkeypatch):
    """Without GPS, travel_times_legs must preserve id_etapa from legs_all.

    Before the fix, id_etapa was omitted from the column selection in the else
    branch, so reindex() filled the whole column with NaN.
    """
    from urbantrips.datamodel import legs as legs_module

    monkeypatch.setattr(
        legs_module,
        "leer_configs_generales",
        lambda autogenerado=True: {"usa_archivo_gps": False},
    )

    def _passthrough_distances(od_df, **kwargs):
        result = od_df.copy()
        result["distance_od"] = 3.5
        return result

    monkeypatch.setattr(legs_module, "compute_od_distances", _passthrough_distances)

    legs_df = pd.DataFrame({
        "dia":        ["2024-01-01", "2024-01-01", "2024-01-01"],
        "id":         [1, 2, 3],
        "id_tarjeta": ["CARD1", "CARD1", "CARD1"],
        "id_viaje":   [1, 1, 2],
        "id_etapa":   [1, 2, 1],
    })

    saved = {}

    class _MockData:
        def query(self, sql):
            return legs_df.copy()

        def get_run_days(self):
            return pd.DataFrame({"dia": ["2024-01-01"]})

        def execute(self, sql):
            pass

        def append_raw(self, df, table):
            saved[table] = df.copy()

    assign_time_distances(SimpleNamespace(data=_MockData()))

    assert "travel_times_legs" in saved, "travel_times_legs was never written"
    result = saved["travel_times_legs"]
    assert "id_etapa" in result.columns
    assert result["id_etapa"].notna().all(), "id_etapa must not be null — fix regression"
    assert result["id_etapa"].tolist() == [1, 2, 1]


def test_assign_time_distances_no_gps_distance_od_stored(monkeypatch):
    """Without GPS, distance_od computed by compute_od_distances must reach the table."""
    from urbantrips.datamodel import legs as legs_module

    monkeypatch.setattr(
        legs_module,
        "leer_configs_generales",
        lambda autogenerado=True: {"usa_archivo_gps": False},
    )

    def _passthrough_distances(od_df, **kwargs):
        result = od_df.copy()
        result["distance_od"] = [1.0, 2.0, 5.0]
        return result

    monkeypatch.setattr(legs_module, "compute_od_distances", _passthrough_distances)

    legs_df = pd.DataFrame({
        "dia":        ["2024-01-01"] * 3,
        "id":         [1, 2, 3],
        "id_tarjeta": ["CARD1", "CARD1", "CARD2"],
        "id_viaje":   [1, 1, 1],
        "id_etapa":   [1, 2, 1],
    })

    saved = {}

    class _MockData:
        def query(self, sql):
            return legs_df.copy()

        def get_run_days(self):
            return pd.DataFrame({"dia": ["2024-01-01"]})

        def execute(self, sql):
            pass

        def append_raw(self, df, table):
            saved[table] = df.copy()

    assign_time_distances(SimpleNamespace(data=_MockData()))

    legs_result = saved["travel_times_legs"]
    assert legs_result["distance_od"].tolist() == [1.0, 2.0, 5.0]

    trips_result = saved["travel_times_trips"]
    card1_total = trips_result.loc[trips_result.id_tarjeta == "CARD1", "distance_od"].iloc[0]
    assert card1_total == 3.0  # 1.0 + 2.0


# --- pago_doble_tarjeta ---

def _make_single_trx(id_tarjeta="CARD1", fecha="2024-01-01 08:00:00"):
    return pd.DataFrame({
        "dia":        ["2024-01-01"],
        "id_tarjeta": [id_tarjeta],
        "id_linea":   [10],
        "fecha":      [fecha],
        "orden_trx":  [1],
        "hora":       [8],
    })


def test_pago_doble_tarjeta_id_tarjeta_nuevo_not_in_output_columns():
    """id_tarjeta_nuevo is an internal temp column and must be dropped by reindex.
    The direct-assignment fix (a1a69e7) must preserve this column contract."""
    trx = _make_single_trx()
    original_cols = set(trx.columns)  # capture before function mutates trx in-place
    params = {"criterio": "fecha_completa", "ventana_duplicado": 5}
    result, _ = pago_doble_tarjeta(trx, params)
    assert "id_tarjeta_nuevo" not in result.columns
    assert set(result.columns) == original_cols


def test_pago_doble_tarjeta_updates_id_tarjeta_to_new_value():
    """id_tarjeta in the result must reflect the id_tarjeta_nuevo computed inside
    the function.  A single non-duplicated transaction gets the '_0' suffix."""
    trx = _make_single_trx()
    params = {"criterio": "fecha_completa", "ventana_duplicado": 5}
    result, dupes = pago_doble_tarjeta(trx, params)
    # nro=0 for the only transaction → id_tarjeta_nuevo = "CARD1_0"
    assert result["id_tarjeta"].iloc[0] == "CARD1_0"
    # No duplicates detected → tarjetas_duplicadas is empty
    assert len(dupes) == 0


def test_pago_doble_tarjeta_splits_concurrent_transactions():
    """Two transactions from the same card on the same line within the window
    are tagged as duplicates and get distinct suffixes."""
    trx = pd.DataFrame({
        "dia":        ["2024-01-01", "2024-01-01"],
        "id_tarjeta": ["CARD1",      "CARD1"],
        "id_linea":   [10,           10],
        "fecha":      ["2024-01-01 08:00:00", "2024-01-01 08:00:30"],
        "orden_trx":  [1,            2],
        "hora":       [8,            8],
    })
    params = {"criterio": "fecha_completa", "ventana_duplicado": 1}
    result, dupes = pago_doble_tarjeta(trx, params)
    # Both rows are within 1-minute window → duplicates → get _0 and _1 suffixes
    suffixes = sorted(result["id_tarjeta"].tolist())
    assert suffixes == ["CARD1_0", "CARD1_1"]
    assert len(dupes) == 1
    assert dupes["id_tarjeta_original"].iloc[0] == "CARD1"
