# urbantrips/tests/unit/test_matriz_validacion_fixture.py
"""
Tests del subset minimo `matriz_validacion_amba_test.csv`.

Le dan proposito al fixture `matriz_validacion_test_amba` (antes definido pero
nunca consumido) y blindan contra el schema drift que ya rompio CI una vez: el
refactor del 2026-06-17 agrego la columna `id_ramal` a matriz_validacion y los
fixtures viejos de 3 columnas dejaron de matchear con la tabla real.
"""
import pandas as pd

from urbantrips.destinations.destinations import (
    _imputar_destino_min_distancia_con_matriz,
)

# Columnas que la tabla matriz_validacion tiene post-refactor
# (ver urbantrips/storage/schema/insumos.py).
SCHEMA_COLS = ["id_linea_agg", "id_ramal", "parada", "area_influencia"]


def test_matriz_subset_matches_current_schema(matriz_validacion_test_amba):
    """El subset debe tener exactamente las columnas de la tabla real."""
    assert list(matriz_validacion_test_amba.columns) == SCHEMA_COLS


def test_matriz_subset_id_ramal_is_null_for_non_ramal_modes(matriz_validacion_test_amba):
    """id_ramal vacio se lee como NaN (NULL): modos que no validan por ramal."""
    assert matriz_validacion_test_amba["id_ramal"].isna().all()


def test_matriz_subset_imputes_known_destination(matriz_validacion_test_amba):
    """
    Una etapa cuyo destino potencial cae en un area_influencia del subset
    debe imputar la parada correspondiente y quedar od_validado=1.
    """
    # area_influencia 88c2e314a3fffff -> parada 88c2e314b1fffff (linea agg 1)
    etapas = pd.DataFrame({
        "id": [1],
        "id_linea_agg": [1],
        "id_ramal": [0],
        "modo": ["autobus"],  # no valida por ramal -> id_ramal efectivo = sentinela
        "h3_d": ["88c2e314a3fffff"],
    })

    out = _imputar_destino_min_distancia_con_matriz(
        etapas, matriz_validacion_test_amba, modos_ramal=set()
    )

    assert out.loc[out["id"] == 1, "od_validado"].iloc[0] == 1
    assert out.loc[out["id"] == 1, "h3_d"].iloc[0] == "88c2e314b1fffff"


def test_matriz_subset_unknown_destination_is_not_validated(matriz_validacion_test_amba):
    """Un destino potencial fuera de toda area_influencia queda od_validado=0."""
    etapas = pd.DataFrame({
        "id": [1],
        "id_linea_agg": [1],
        "id_ramal": [0],
        "modo": ["autobus"],
        "h3_d": ["8801234567fffff"],  # celda inexistente en el subset
    })

    out = _imputar_destino_min_distancia_con_matriz(
        etapas, matriz_validacion_test_amba, modos_ramal=set()
    )

    assert out.loc[out["id"] == 1, "od_validado"].iloc[0] == 0
    assert pd.isna(out.loc[out["id"] == 1, "h3_d"].iloc[0])
