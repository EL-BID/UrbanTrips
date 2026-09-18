"""`clear_raw` vacía el staging con DROP + CREATE, no con DELETE.

Corre en el `finally` de la ingesta, o sea justo cuando el proceso puede venir de una
excepción con la RAM al tope: en la corrida del mes del cliente el `DELETE FROM
transacciones_raw` sobre 105 M filas murió con "Out of Memory Error: Allocation
failure" y tapó el error original. El DROP no escanea la tabla ni materializa el vector
de borrado, y además libera el espacio en el archivo. Lo que estos tests fijan es que la
tabla queda igual de usable después.
"""

import pandas as pd

from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
from urbantrips.storage.schema import data as schema


def _raw_row(**overrides):
    row = {
        "id_original": "1", "id_tarjeta": "card_1", "dia": "2022-01-01",
        "tiempo": "08:00:00", "hora": 8, "modo": "autobus",
        "id_linea": 1, "id_ramal": 1, "interno": 10, "orden_trx": 1,
        "genero": "-", "tarifa": "-", "latitud": -34.6, "longitud": -58.4,
        "fecha_ts": 1641024000, "factor_expansion_raw": 1.0,
    }
    row.update(overrides)
    return row


def _tipos(adapter):
    return adapter.query(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'transacciones_raw' ORDER BY ordinal_position"
    )


def test_clear_raw_deja_la_tabla_vacia_y_con_el_mismo_esquema(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    cols = schema.TRANSACCIONES_RAW_COLUMNS

    adapter.save_raw_chunk(
        pd.DataFrame([_raw_row(id_original="1"), _raw_row(id_original="2")]).reindex(
            columns=cols
        )
    )
    antes = _tipos(adapter)
    assert adapter.has_rows("transacciones_raw")

    adapter.clear_raw()

    assert not adapter.has_rows("transacciones_raw")
    pd.testing.assert_frame_equal(_tipos(adapter), antes)


def test_clear_raw_deja_el_staging_reutilizable(tmp_path):
    """Después de vaciar, la ingesta siguiente tiene que poder insertar de nuevo."""
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    cols = schema.TRANSACCIONES_RAW_COLUMNS

    adapter.save_raw_chunk(pd.DataFrame([_raw_row()]).reindex(columns=cols))
    adapter.clear_raw()
    adapter.save_raw_chunk(
        pd.DataFrame([_raw_row(id_original="9", dia="2022-01-02")]).reindex(columns=cols)
    )

    quedaron = adapter.query("SELECT id_original, dia FROM transacciones_raw")
    assert quedaron["id_original"].tolist() == ["9"]
    assert quedaron["dia"].tolist() == ["2022-01-02"]


def test_clear_raw_es_idempotente(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter.clear_raw()
    adapter.clear_raw()
    assert not adapter.has_rows("transacciones_raw")
