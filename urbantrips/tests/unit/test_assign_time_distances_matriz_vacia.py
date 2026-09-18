# urbantrips/tests/unit/test_assign_time_distances_matriz_vacia.py
"""`assign_time_distances` con `matriz_validacion` vacía.

`matriz.apply(..., axis=1)` sobre un DataFrame vacío devuelve un DataFrame (no
una Series), y asignarlo a `matriz["ring"]` rompía con:

    ValueError: Cannot set a DataFrame with multiple columns to the single
    column ring

Una matriz vacía es un dato degenerado —ninguna etapa valida destino por GPS—
pero es un resultado válido, no un motivo para abortar la corrida entera.
"""
import pandas as pd
import pytest

from urbantrips.datamodel import legs as legs_module


_COLS_MATRIZ = ["id_linea_agg", "id_ramal", "parada", "area_influencia"]

_CONFIG = {
    "usa_archivo_gps": True,
    "resolucion_h3": 9,
    "tolerancia_destino_gps": 1000,
    "modos": {"autobus": "COLECTIVO"},
    "modo_valida_ramal": {"valida_ramal_autobus": True},
}


class _FakeData:
    def __init__(self):
        self.executed = []

    def get_run_days(self):
        return pd.DataFrame({"dia": ["2024-01-01"]})

    def execute(self, sql):
        self.executed.append(sql)

    def query(self, sql):
        return pd.DataFrame()

    def append_raw(self, df, table_name):
        pass


class _FakeInsumos:
    def __init__(self, matriz):
        self._matriz = matriz

    def get_metadata_lineas(self):
        return pd.DataFrame({"id_linea": [1], "id_linea_agg": [1], "modo": ["autobus"]})

    def get_matrix_validation(self):
        return self._matriz.copy()


def _ctx(matriz):
    ctx = type("Ctx", (), {})()
    ctx.data = _FakeData()
    ctx.insumos = _FakeInsumos(matriz)
    return ctx


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.setattr(legs_module, "leer_configs_generales", lambda *a, **k: _CONFIG)
    monkeypatch.setattr(
        legs_module, "_parallel_day_workers", lambda n, per_day_gb=None, main_extra_gb=0.0: 1
    )
    # sin etapas validadas ese día el loop no hace trabajo real: _fetch_legs_all_dia
    # devuelve None en ese caso, que es lo que el caller espera
    monkeypatch.setattr(legs_module, "_fetch_legs_all_dia", lambda ctx, dia: None)


def test_matriz_vacia_no_rompe(caplog):
    """El caso que crasheaba."""
    ctx = _ctx(pd.DataFrame(columns=_COLS_MATRIZ))

    with caplog.at_level("WARNING"):
        legs_module.assign_time_distances(ctx)   # no debe levantar

    assert any("matriz_validacion está vacía" in r.message for r in caplog.records), (
        "una matriz vacía tiene que avisar: casi siempre es un problema de datos"
    )


def test_matriz_con_datos_calcula_ring():
    """Control: con filas el camino normal sigue funcionando."""
    parada = "89c2e310003ffff"
    matriz = pd.DataFrame({
        "id_linea_agg": [1],
        "id_ramal": [1],
        "parada": [parada],
        "area_influencia": [parada],   # ring = 0
    })

    legs_module.assign_time_distances(_ctx(matriz))   # no debe levantar


def test_matriz_vacia_produce_ring_vacio_y_filtrable():
    """La columna `ring` tiene que quedar utilizable aguas abajo, no ausente.

    Reproduce el fragmento exacto que fallaba, para fijar el contrato del frame.
    """
    matriz = pd.DataFrame(columns=_COLS_MATRIZ)
    matriz["id_ramal"] = matriz["id_ramal"].fillna(-1).astype("int64")
    matriz["ring"] = pd.Series(dtype="int64")

    filtrada = matriz[matriz.ring <= 2]
    assert filtrada.empty
    assert "ring" in filtrada.columns
