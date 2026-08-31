"""`prepare_gps_from_raw` contra la implementación pandas que reemplazó.

Los tres pasos del ingest de gps que necesitan ver el archivo entero —dedup, id interno
correlativo y odómetro por vehículo— pasaron de pandas a SQL para que el pico de memoria
no dependa de cuántos días traiga el csv. `_referencia_pandas` es una copia CONGELADA de
la versión anterior: no se toca cuando cambia el adapter, es el patrón contra el que se
compara.
"""

import numpy as np
import pandas as pd
import pytest

from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
from urbantrips.storage.schema.data import GPS_RAW_COLUMNS

_CLAVE_ODOMETRO = ["id_linea", "id_ramal", "interno"]

_SUBSET_DEDUP = [
    "dia", "id_linea", "id_ramal", "interno", "fecha", "latitud", "longitud",
]


def _referencia_pandas(gps, subset_dedup, id_offset, odometro):
    """Versión anterior, en pandas y sobre el archivo entero. Congelada."""
    gps = gps.sort_values("orden_archivo").reset_index(drop=True)
    gps = gps.drop_duplicates(subset=subset_dedup)
    gps["id"] = np.arange(id_offset, id_offset + len(gps), dtype="int64")

    ordenado = gps.sort_values(_CLAVE_ODOMETRO + ["fecha", "orden_archivo"])
    if odometro == "diff":
        d = ordenado.groupby(_CLAVE_ODOMETRO)["distance_servicio_mts_agg"].diff()
        d = d.reindex(gps.index)
        gps["distance_servicio_mts"] = d.fillna(0)
        gps.loc[gps["distance_servicio_mts"] < 0, "distance_servicio_mts"] = 0
    elif odometro == "cumsum":
        c = ordenado.groupby(_CLAVE_ODOMETRO)["distance_servicio_mts"].cumsum()
        c = c.reindex(gps.index)
        gps["distance_servicio_mts_agg"] = c.fillna(0)
        gps.loc[
            gps["distance_servicio_mts_agg"] < 0, "distance_servicio_mts_agg"
        ] = 0
    return gps.reset_index(drop=True)


def _gps_raw_sintetico(n_lineas=5, n_internos=4, n_pings=30, seed=0):
    """Staging desordenado, con empates de fecha dentro del mismo vehículo, días
    cruzados, duplicados exactos y odómetros que retroceden (cambio de servicio)."""
    rng = np.random.default_rng(seed)
    filas = []
    for linea in range(1, n_lineas + 1):
        for ramal in (linea * 100, linea * 100 + 1):
            for interno in range(1, n_internos + 1):
                fechas = np.sort(rng.integers(0, 3 * 86400, size=n_pings))
                fechas[1] = fechas[0]  # empate de fecha en el mismo vehículo
                odo = np.cumsum(rng.integers(0, 500, size=n_pings)).astype(float)
                odo[n_pings // 2] = 0.0  # el odómetro se reinicia: da un delta < 0
                for f, o in zip(fechas, odo):
                    filas.append(
                        {
                            "id_original": str(len(filas)),
                            "dia": f"2026-07-{1 + int(f) // 86400:02d}",
                            "id_linea": linea,
                            "id_ramal": ramal,
                            "interno": interno,
                            "fecha": int(f),
                            "latitud": -34.6 + interno / 1000,
                            "longitud": -58.4 + linea / 1000,
                            "velocity": 30.0,
                            "id_servicio": "1",
                            "service_type": "start_service",
                            "distance_servicio_mts": float(rng.integers(0, 400)),
                            "distance_servicio_mts_agg": o,
                        }
                    )
    gps = pd.DataFrame(filas)
    # duplicados exactos sobre el subset del dedup, intercalados
    gps = pd.concat([gps, gps.iloc[::17]], ignore_index=True)
    gps = gps.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    gps["orden_archivo"] = np.arange(len(gps), dtype="int64")
    return gps.reindex(columns=GPS_RAW_COLUMNS)


def _cargar(adapter, gps, n_chunks=3):
    adapter.reset_gps_raw()
    for pedazo in np.array_split(gps, n_chunks):
        adapter.save_gps_raw_chunk(pedazo)


@pytest.mark.parametrize("odometro", ["diff", "cumsum", None])
@pytest.mark.parametrize("id_offset", [0, 1000])
def test_prepare_gps_from_raw_es_identico_a_la_version_pandas(
    tmp_path, odometro, id_offset
):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    gps = _gps_raw_sintetico()
    _cargar(adapter, gps)

    filas = adapter.prepare_gps_from_raw(
        dedup_subset=_SUBSET_DEDUP, id_offset=id_offset, odometro=odometro
    )
    obtenido = adapter.query("SELECT * FROM gps_prep ORDER BY id")
    esperado = _referencia_pandas(gps, _SUBSET_DEDUP, id_offset, odometro)

    assert filas == len(esperado)
    assert obtenido["id"].tolist() == esperado["id"].tolist()
    assert obtenido["orden_archivo"].tolist() == esperado["orden_archivo"].tolist()
    for col in ("distance_servicio_mts", "distance_servicio_mts_agg"):
        np.testing.assert_allclose(
            obtenido[col].to_numpy(dtype=float),
            esperado[col].to_numpy(dtype=float),
            rtol=0,
            atol=0,
            err_msg=f"difiere {col} con odometro={odometro}",
        )


def test_el_dedup_se_queda_con_la_primera_aparicion(tmp_path):
    """pandas `drop_duplicates` conserva la primera; el ROW_NUMBER tiene que hacer lo
    mismo, si no cambia qué fila sobrevive (y con ella su id_original)."""
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    base = {
        "dia": "2026-07-01", "id_linea": 1, "id_ramal": 10, "interno": 5,
        "fecha": 100, "latitud": -34.6, "longitud": -58.4, "velocity": 1.0,
        "id_servicio": "1", "service_type": None,
        "distance_servicio_mts": None, "distance_servicio_mts_agg": None,
    }
    gps = pd.DataFrame(
        [
            {**base, "orden_archivo": 2, "id_original": "tercera"},
            {**base, "orden_archivo": 0, "id_original": "primera"},
            {**base, "orden_archivo": 1, "id_original": "segunda"},
        ]
    ).reindex(columns=GPS_RAW_COLUMNS)
    _cargar(adapter, gps, n_chunks=1)

    adapter.prepare_gps_from_raw(_SUBSET_DEDUP, id_offset=0, odometro=None)
    quedaron = adapter.query("SELECT id_original FROM gps_prep")
    assert quedaron["id_original"].tolist() == ["primera"]


def test_el_odometro_desempata_por_orden_de_archivo(tmp_path):
    """Dos pings del mismo vehículo en el mismo segundo: el orden entre ellos lo fija
    el archivo, igual que el sort estable de pandas."""
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    base = {
        "dia": "2026-07-01", "id_linea": 1, "id_ramal": 10, "interno": 5,
        "latitud": -34.6, "velocity": 1.0, "id_servicio": "1", "service_type": None,
        "distance_servicio_mts": None,
    }
    gps = pd.DataFrame(
        [
            {**base, "orden_archivo": 0, "id_original": "a", "fecha": 100,
             "longitud": -58.40, "distance_servicio_mts_agg": 1000.0},
            {**base, "orden_archivo": 1, "id_original": "b", "fecha": 100,
             "longitud": -58.41, "distance_servicio_mts_agg": 1300.0},
            {**base, "orden_archivo": 2, "id_original": "c", "fecha": 200,
             "longitud": -58.42, "distance_servicio_mts_agg": 1900.0},
        ]
    ).reindex(columns=GPS_RAW_COLUMNS)
    _cargar(adapter, gps, n_chunks=1)

    adapter.prepare_gps_from_raw(_SUBSET_DEDUP, id_offset=0, odometro="diff")
    r = adapter.query(
        "SELECT id_original, distance_servicio_mts FROM gps_prep ORDER BY orden_archivo"
    )
    assert r["distance_servicio_mts"].tolist() == [0.0, 300.0, 600.0]


def test_los_dias_salen_en_orden_de_primera_aparicion(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    base = {
        "id_linea": 1, "id_ramal": 10, "interno": 5, "latitud": -34.6,
        "velocity": 1.0, "id_servicio": "1", "service_type": None,
        "distance_servicio_mts": None, "distance_servicio_mts_agg": None,
    }
    gps = pd.DataFrame(
        [
            {**base, "orden_archivo": 0, "id_original": "a", "dia": "2026-07-03",
             "fecha": 1, "longitud": -58.1},
            {**base, "orden_archivo": 1, "id_original": "b", "dia": "2026-07-01",
             "fecha": 2, "longitud": -58.2},
            {**base, "orden_archivo": 2, "id_original": "c", "dia": "2026-07-03",
             "fecha": 3, "longitud": -58.3},
        ]
    ).reindex(columns=GPS_RAW_COLUMNS)
    _cargar(adapter, gps, n_chunks=1)
    adapter.prepare_gps_from_raw(_SUBSET_DEDUP, id_offset=0, odometro=None)

    assert adapter.gps_prep_days() == ["2026-07-03", "2026-07-01"]


def test_dedup_subset_invalido_no_llega_a_sql(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    adapter.reset_gps_raw()
    with pytest.raises(ValueError):
        adapter.prepare_gps_from_raw(["dia; DROP TABLE gps"], 0, None)
    with pytest.raises(ValueError):
        adapter.prepare_gps_from_raw([], 0, None)


def test_clear_gps_staging_deja_el_staging_vacio_y_reutilizable(tmp_path):
    adapter = DuckDBDataAdapter(tmp_path / "data.duckdb")
    gps = _gps_raw_sintetico(n_lineas=1, n_internos=1, n_pings=5)
    _cargar(adapter, gps, n_chunks=1)
    adapter.prepare_gps_from_raw(_SUBSET_DEDUP, 0, "diff")

    adapter.clear_gps_staging()

    assert not adapter.has_rows("gps_raw")
    assert not adapter.has_rows("gps_prep")
    _cargar(adapter, gps, n_chunks=1)
    assert adapter.has_rows("gps_raw")
