import pandas as pd
import pytest

from urbantrips.storage.ports import BatchSpec, DataPort


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "fecha": [20240101, 20240101, 20240102],
            "id_original": ["a", "b", "c"],
            "id_tarjeta": ["T001", "T002", "T001"],
            "dia": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "tiempo": ["08:00", "09:00", "08:30"],
            "hora": [8, 9, 8],
            "modo": ["bus", "bus", "metro"],
            "id_linea": [1, 2, 3],
            "id_ramal": [10, 20, 30],
            "interno": [100, 200, 300],
            "orden_trx": [1, 1, 1],
            "genero": [None, None, None],
            "tarifa": [None, None, None],
            "latitud": [-34.6, -34.7, -34.8],
            "longitud": [-58.4, -58.5, -58.6],
            "factor_expansion": [1.0, 1.0, 1.0],
        }
    )


def _sample_legs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "id_tarjeta": ["T001", "T002", "T001"],
            "dia": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "id_viaje": [1, 2, 1],
            "id_etapa": [1, 1, 2],
            "tiempo": ["08:00", "09:00", "08:30"],
            "hora": [8, 9, 8],
            "modo": ["bus", "bus", "metro"],
            "id_linea": [1, 2, 3],
            "id_ramal": [10, 20, 30],
            "interno": [100, 200, 300],
            "genero": [None, None, None],
            "tarifa": [None, None, None],
            "latitud": [-34.6, -34.7, -34.8],
            "longitud": [-58.4, -58.5, -58.6],
            "h3_o": ["882a100d2bfffff", "882a100d3bfffff", "882a100d4bfffff"],
            "h3_d": ["882a100d3bfffff", "882a100d4bfffff", "882a100d5bfffff"],
            "od_validado": [1, 1, 1],
            "etapa_validada": [1, 1, 1],
            "factor_expansion_original": [1.0, 1.0, 1.0],
            "factor_expansion_linea": [1.0, 1.0, 1.0],
            "factor_expansion_tarjeta": [1.0, 1.0, 1.0],
            "factor_expansion_etapa": [1.0, 1.0, 1.0],
            "distancia": [500.0, 750.0, 1000.0],
            "travel_time_min": [15.0, 20.0, 30.0],
        }
    )


def _sample_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_tarjeta": ["T001", "T002"],
            "id_viaje": [1, 2],
            "dia": ["2024-01-01", "2024-01-01"],
            "tiempo": ["08:00", "09:00"],
            "hora": [8, 9],
            "cant_etapas": [2, 1],
            "modo": ["bus", "metro"],
            "autobus": [1, 0],
            "tren": [0, 0],
            "metro": [0, 1],
            "tranvia": [0, 0],
            "brt": [0, 0],
            "cable": [0, 0],
            "lancha": [0, 0],
            "otros": [0, 0],
            "h3_o": ["882a100d2bfffff", "882a100d3bfffff"],
            "h3_d": ["882a100d3bfffff", "882a100d4bfffff"],
            "genero": [None, None],
            "tarifa": [None, None],
            "od_validado": [1, 1],
            "factor_expansion_linea": [1.0, 1.0],
            "factor_expansion_tarjeta": [1.0, 1.0],
            "distancia": [500.0, 750.0],
            "travel_time_min": [15.0, 20.0],
        }
    )


def _sample_users() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_tarjeta": ["T001", "T002"],
            "dia": ["2024-01-01", "2024-01-01"],
            "od_validado": [1, 0],
            "cant_viajes": [2.0, 1.0],
            "factor_expansion_linea": [1.0, 1.0],
            "factor_expansion_tarjeta": [1.0, 1.0],
        }
    )


def _sample_gps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "id_original": ["gps-1", "gps-2"],
            "dia": ["2024-01-01", "2024-01-02"],
            "id_linea": [1, 2],
            "id_ramal": [10, 20],
            "interno": [100, 200],
            "fecha": [20240101, 20240102],
            "latitud": [-34.6, -34.7],
            "longitud": [-58.4, -58.5],
            "velocity": [10.0, 20.0],
            "service_type": ["regular", "regular"],
            "distance_km": [1.0, 2.0],
            "h3": ["882a100d2bfffff", "882a100d3bfffff"],
        }
    )


@pytest.fixture(params=["memory", "duckdb"])
def data_adapter(request, tmp_path) -> DataPort:
    if request.param == "memory":
        from urbantrips.storage.adapters.memory.adapters import InMemoryDataAdapter

        return InMemoryDataAdapter()

    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter

    return DuckDBDataAdapter(tmp_path / "contract_data.duckdb")


def test_data_port_contract_core_roundtrips(data_adapter):
    assert isinstance(data_adapter, DataPort)

    run_days = pd.DataFrame({"dia": ["2024-01-01", "2024-01-02"]})
    data_adapter.save_run_days(run_days)
    assert set(data_adapter.get_run_days()["dia"]) == set(run_days["dia"])

    transactions = _sample_transactions()
    data_adapter.save_transactions(transactions)
    assert set(data_adapter.get_transactions()["id"]) == set(transactions["id"])

    legs = _sample_legs()
    data_adapter.save_legs(legs)
    assert set(data_adapter.get_legs()["id"]) == set(legs["id"])

    trips = _sample_trips()
    data_adapter.save_trips(trips)
    assert len(data_adapter.get_trips()) == len(trips)

    users = _sample_users()
    data_adapter.save_users(users)
    assert set(data_adapter.get_users()["id_tarjeta"]) == set(users["id_tarjeta"])

    gps = _sample_gps()
    data_adapter.save_gps(gps)
    assert set(data_adapter.get_gps()["id"]) == set(gps["id"])


def test_data_port_contract_replace_and_delete_by_day(data_adapter):
    data_adapter.save_transactions(_sample_transactions())
    data_adapter.save_legs(_sample_legs())
    data_adapter.save_gps(_sample_gps())

    replacement = _sample_legs().iloc[[0]].copy()
    replacement["h3_d"] = "882a100d6bfffff"
    data_adapter.replace_legs_for_days(replacement, ["2024-01-01"])

    legs = data_adapter.get_legs()
    assert len(legs[legs["dia"] == "2024-01-01"]) == 1
    assert legs.loc[legs["dia"] == "2024-01-01", "h3_d"].iloc[0] == "882a100d6bfffff"
    assert len(legs[legs["dia"] == "2024-01-02"]) == 1

    data_adapter.delete_run_days(["2024-01-02"])
    assert "2024-01-02" not in set(data_adapter.get_transactions()["dia"])
    assert "2024-01-02" not in set(data_adapter.get_legs()["dia"])
    assert "2024-01-02" not in set(data_adapter.get_gps()["dia"])


def test_data_port_contract_generic_table_helpers(data_adapter):
    raw = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    more_raw = pd.DataFrame({"id": [3], "value": ["c"]})

    data_adapter.save_raw(raw, "contract_raw")
    data_adapter.append_raw(more_raw, "contract_raw")

    result = data_adapter.get_raw("contract_raw").sort_values("id").reset_index(drop=True)
    expected = pd.concat([raw, more_raw], ignore_index=True)
    pd.testing.assert_frame_equal(result[expected.columns], expected)

    assert data_adapter.has_rows("contract_raw")
    assert data_adapter.has_rows("contract_raw", "value = 'b'")
    assert not data_adapter.has_rows("contract_raw", "value = 'missing'")
    assert data_adapter.get_max_id("contract_raw") == 4


def test_data_port_contract_rejects_invalid_raw_table_names(data_adapter):
    raw = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError, match="Invalid table name"):
        data_adapter.save_raw(raw, "contract_raw; DROP TABLE etapas")

    with pytest.raises(ValueError, match="Invalid table name"):
        data_adapter.get_raw("contract raw")


def test_data_port_contract_batches_are_complete(data_adapter):
    data_adapter.save_transactions(_sample_transactions())
    batches = data_adapter.get_user_batches(2)

    assert batches == [BatchSpec(batch_id=0, total_batches=2), BatchSpec(batch_id=1, total_batches=2)]

    batched_ids = []
    for batch in batches:
        batched_ids.extend(data_adapter.get_transactions(batch=batch)["id"].tolist())

    assert sorted(batched_ids) == [1, 2, 3]
