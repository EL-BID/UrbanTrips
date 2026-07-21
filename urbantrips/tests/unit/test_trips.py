# urbantrips/tests/unit/test_trips.py
import pandas as pd
from types import SimpleNamespace
from urbantrips.datamodel.trips import _derive_trip_dia, rearrange_trip_id_same_od


def test_derive_trip_dia_uses_first_leg_dia():
    legs = pd.DataFrame({
        "id_tarjeta": ["card_A", "card_A", "card_A"],
        "id_viaje": [1, 1, 2],
        "dia": ["2022-08-11", "2022-08-12", "2022-08-12"],
    })
    result = _derive_trip_dia(legs)
    assert result.loc[result.id_viaje == 1, "trip_dia"].iloc[0] == "2022-08-11"
    assert result.loc[result.id_viaje == 2, "trip_dia"].iloc[0] == "2022-08-12"


def test_derive_trip_dia_preserves_all_rows():
    legs = pd.DataFrame({
        "id_tarjeta": ["A", "A", "B"],
        "id_viaje": [1, 1, 1],
        "dia": ["2022-01-01", "2022-01-02", "2022-01-01"],
        "other_col": [10, 20, 30],
    })
    result = _derive_trip_dia(legs)
    assert len(result) == 3
    assert "trip_dia" in result.columns
    assert "other_col" in result.columns


# --- rearrange_trip_id_same_od ---

def _make_legs_ctx(legs_df):
    """Return a minimal mock StorageContext for rearrange_trip_id_same_od."""
    saved = {}

    class _MockData:
        def get_run_days(self):
            dias = legs_df["dia"].unique().tolist()
            return pd.DataFrame({"dia": dias})

        def query(self, sql):
            return legs_df.copy()

        def execute(self, sql):
            pass

        def save_legs(self, df, batch=None):
            saved["result"] = df.copy()

        def update_leg_trip_ids(self, df, dia=None):
            saved["result"] = df.copy()

    return SimpleNamespace(data=_MockData()), saved


def _minimal_legs(dias, id_tarjeta, id_viaje, id_linea, h3_o, h3_d):
    """Build a minimal etapas DataFrame row-by-row from parallel lists."""
    n = len(dias)
    return pd.DataFrame({
        "dia":         dias,
        "id":          list(range(1, n + 1)),
        "id_tarjeta":  id_tarjeta,
        "id_viaje":    id_viaje,
        "id_etapa":    [1] * n,
        "id_linea":    id_linea,
        "tiempo":      ["08:00:00"] * n,
        "hora":        [8] * n,
        "h3_o":        h3_o,
        "h3_d":        h3_d,
        "od_validado": [1] * n,
    })


def test_rearrange_trip_id_same_od_isolates_linea_anterior_by_day():
    """id_linea_anterior must not bleed across days for the same (id_tarjeta, id_viaje).

    Before the fix (645f2e0), groupby omitted 'dia', so the last leg of day N was
    treated as the predecessor of the first leg of day N+1 for the same trip ID.
    This caused spurious es_igual=True, which bumped id_viaje on the second day.
    """
    # Same card, same trip ID (1), same line (10), across two days.
    # h3_o != h3_d so the same-OD filter doesn't fire.
    legs_df = _minimal_legs(
        dias=       ["2024-01-01", "2024-01-02"],
        id_tarjeta= ["CARD_A",     "CARD_A"],
        id_viaje=   [1,            1],
        id_linea=   [10,           10],
        h3_o=       ["hex_A",      "hex_C"],
        h3_d=       ["hex_B",      "hex_D"],
    )
    ctx, saved = _make_legs_ctx(legs_df)
    rearrange_trip_id_same_od(ctx)

    result = saved["result"]
    day2_row = result[result["dia"] == "2024-01-02"]
    assert len(day2_row) == 1
    # id_viaje must remain 1 — the cross-day same-line continuity must NOT split the trip
    assert day2_row["id_viaje"].iloc[0] == 1, (
        "id_viaje was incorrectly bumped — dia was missing from id_linea_anterior groupby"
    )


def test_rearrange_trip_id_same_od_same_line_within_day_does_split():
    """Within a single day, consecutive legs on the same line DO trigger a trip split.
    This verifies the same-OD correction logic still fires correctly after the fix."""
    # Trip 1: two legs on line_10 with same h3_o/h3_d → same-OD condition
    legs_df = _minimal_legs(
        dias=       ["2024-01-01", "2024-01-01"],
        id_tarjeta= ["CARD_B",     "CARD_B"],
        id_viaje=   [1,            1],
        id_linea=   [10,           10],
        h3_o=       ["hex_A",      "hex_A"],
        h3_d=       ["hex_A",      "hex_A"],  # same OD → triggers split
    )
    ctx, saved = _make_legs_ctx(legs_df)
    rearrange_trip_id_same_od(ctx)

    result = saved["result"]
    # Both legs had same OD and > 1 leg → must be split into separate trips
    assert result["id_viaje"].nunique() == 2, (
        "Same-OD trips with multiple legs must be split into distinct trip IDs"
    )
