# urbantrips/tests/unit/test_trips.py
import pandas as pd
from urbantrips.datamodel.trips import _derive_trip_dia


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
