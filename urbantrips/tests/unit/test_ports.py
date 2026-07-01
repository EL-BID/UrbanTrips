# urbantrips/tests/unit/test_ports.py
from dataclasses import FrozenInstanceError
import pytest


def test_batchspec_is_immutable():
    from urbantrips.storage.ports import BatchSpec
    b = BatchSpec(batch_id=0, total_batches=4)
    assert b.batch_id == 0
    assert b.total_batches == 4
    with pytest.raises((FrozenInstanceError, AttributeError)):
        b.batch_id = 1  # type: ignore


def test_batchspec_equality():
    from urbantrips.storage.ports import BatchSpec
    assert BatchSpec(0, 4) == BatchSpec(0, 4)
    assert BatchSpec(0, 4) != BatchSpec(1, 4)


def test_port_protocols_importable():
    from urbantrips.storage.ports import DataPort, InsumoPort, DashPort, GeneralPort
    import typing
    assert hasattr(DataPort, "__protocol_attrs__") or hasattr(DataPort, "_is_protocol")
    attrs = getattr(DataPort, "__protocol_attrs__", getattr(DataPort, "_protocol_attrs", set()))
    assert "has_rows" in attrs


def test_memory_adapter_satisfies_general_port():
    """A minimal class implementing GeneralPort passes runtime isinstance check."""
    from urbantrips.storage.ports import GeneralPort
    import pandas as pd

    class _MinimalGeneral:
        def get_completed_runs(self):
            return pd.DataFrame()
        def register_run(self, alias, process):
            pass
        def execute(self, sql):
            pass
        def query(self, sql):
            return pd.DataFrame()
        def append_raw(self, df, table_name):
            pass
        def get_raw(self, table_name):
            return pd.DataFrame()
        def run_exists(self, alias):
            return False
        def clear_runs(self):
            pass

    obj = _MinimalGeneral()
    assert isinstance(obj, GeneralPort)


from urbantrips.storage.schema import data as schema


def test_transacciones_schema_has_batch_id():
    assert "batch_id" in schema.TRANSACCIONES


def test_etapas_schema_has_batch_id():
    assert "batch_id" in schema.ETAPAS


def test_transacciones_raw_table_exists_in_schema():
    assert hasattr(schema, "TRANSACCIONES_RAW")
    assert "transacciones_raw" in schema.TRANSACCIONES_RAW.lower()


def test_index_ddl_constants_exist():
    assert hasattr(schema, "IDX_TRX_BATCH")
    assert hasattr(schema, "IDX_ETAPAS_BATCH")
    assert hasattr(schema, "IDX_GPS_LINE_DAY")
    assert hasattr(schema, "IDX_ETAPAS_DIA_OD_VALIDADO")
    assert hasattr(schema, "IDX_ETAPAS_DIA_LINE_RAMAL_INTERNO")
    assert hasattr(schema, "IDX_GPS_DIA_LINE_RAMAL_INTERNO_FECHA")
    assert hasattr(schema, "IDX_TRAVEL_TIMES_GPS_ID")
    assert hasattr(schema, "IDX_TRAVEL_TIMES_STATIONS_ID")
    assert hasattr(schema, "IDX_SERVICES_STATS_LINE_DAY")
    assert hasattr(schema, "ALL_INDEXES")
    assert "idx_trx_batch" in schema.IDX_TRX_BATCH.lower()
    assert "idx_etapas_batch" in schema.IDX_ETAPAS_BATCH.lower()
    assert "idx_gps_line_day" in schema.IDX_GPS_LINE_DAY.lower()
    assert "etapas(dia, od_validado)" in schema.IDX_ETAPAS_DIA_OD_VALIDADO.lower()
    assert "etapas(dia, id_linea, id_ramal, interno)" in schema.IDX_ETAPAS_DIA_LINE_RAMAL_INTERNO.lower()
    assert "gps(dia, id_linea, id_ramal, interno, fecha)" in schema.IDX_GPS_DIA_LINE_RAMAL_INTERNO_FECHA.lower()
    assert "travel_times_gps(id)" in schema.IDX_TRAVEL_TIMES_GPS_ID.lower()
    assert "travel_times_stations(id)" in schema.IDX_TRAVEL_TIMES_STATIONS_ID.lower()
    assert "idx_services_stats_line_day" in schema.IDX_SERVICES_STATS_LINE_DAY.lower()
