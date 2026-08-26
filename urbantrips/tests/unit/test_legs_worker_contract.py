"""Contrato entre assign_time_distances (main) y su worker.

Por qué este archivo y no test_legs.py: conftest.py excluye unit/test_legs.py del
collect (collect_ignore_glob), así que nada de lo que se agregue ahí corre en la suite.
Este archivo sí se recolecta.

Qué protege: _gps_destino_y_tiempos_dia corre en un worker process que NO hereda el
FileHandler del main, así que sus logger.info() no llegan al archivo de log. Por eso
devuelve un dict `diag` y el main lo emite. Si alguien cambia la aridad de ese return
sin actualizar el caller, el camino PARALELO —el que corre en producción cuando el
autotune da más de 1 worker— rompería recién en vivo, a mitad de una corrida larga.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

# Celda h3 real: grid_distance(c, c) == 0, así pasa el filtro de ring del main.
H3_CELL = "89c2e3a95c3ffff"


def _ctx(legs_df):
    class _Data:
        def query(self, sql):
            return legs_df.copy()

        def get_run_days(self):
            return pd.DataFrame({"dia": ["2024-01-01"]})

        def execute(self, sql):
            pass

        def append_raw(self, df, table):
            pass

    insumos = SimpleNamespace(
        get_metadata_lineas=lambda: pd.DataFrame(
            {"id_linea": [1], "id_linea_agg": [1], "modo": ["autobus"]}
        ),
        get_matrix_validation=lambda: pd.DataFrame({
            "id_linea_agg": [1],
            "id_ramal": [None],
            "parada": [H3_CELL],
            "area_influencia": [H3_CELL],
        }),
    )
    return SimpleNamespace(data=_Data(), insumos=insumos)


def test_camino_paralelo_desempaqueta_los_4_valores_del_worker(monkeypatch):
    legs = pytest.importorskip("urbantrips.datamodel.legs")

    monkeypatch.setattr(
        legs, "leer_configs_generales",
        lambda autogenerado=True: {"usa_archivo_gps": True, "resolucion_h3": 9},
    )
    monkeypatch.setattr(legs, "modos_con_ramal", lambda cfg: set())
    monkeypatch.setattr(
        legs, "_parallel_day_workers", lambda n, per_day_gb=None, main_extra_gb=0.0: 2
    )  # fuerza paralelo

    legs_df = pd.DataFrame({
        "dia": ["2024-01-01"], "id": [1], "id_tarjeta": ["C1"],
        "id_viaje": [1], "id_etapa": [1],
    })
    monkeypatch.setattr(legs, "_fetch_legs_all_dia", lambda ctx, dia: legs_df.copy())
    monkeypatch.setattr(
        legs, "_fetch_time_distance_inputs_dia",
        lambda ctx, dia, nxt: (pd.DataFrame(), pd.DataFrame()),
    )

    tt = pd.DataFrame({"dia": ["2024-01-01"], "id": [1], "distance_od": [1.0]})
    monkeypatch.setattr(
        legs, "_gps_destino_y_tiempos_dia",
        lambda *a, **k: (tt.copy(), tt.copy(), None, {"pct_gps_imputado": 88.5}),
    )

    guardado = []
    monkeypatch.setattr(
        legs, "_save_travel_times_dia",
        lambda ctx, dia, a, b, c: guardado.append(dia),
    )

    # Executor sincrónico: ejercita el submit()/result() real sin levantar subprocesos
    class _Fut:
        def __init__(self, r):
            self._r = r

        def result(self):
            return self._r

    class _Exec:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args, **kw):
            return _Fut(fn(*args, **kw))

    monkeypatch.setattr(legs, "ProcessPoolExecutor", _Exec)
    monkeypatch.setattr(legs, "as_completed", lambda fs: list(fs))

    mensajes = []
    real_info = legs.logger.info
    monkeypatch.setattr(
        legs.logger, "info",
        lambda msg, *a: (mensajes.append(msg % a if a else msg), real_info(msg, *a))[0],
    )

    legs.assign_time_distances(_ctx(legs_df))

    assert guardado == ["2024-01-01"], "el día no se guardó por el camino paralelo"
    assert any("GPS imputado" in m and "88.5" in m for m in mensajes), (
        "el main debe loguear el diagnóstico que devuelve el worker; "
        f"mensajes={mensajes}"
    )


def test_todos_los_returns_del_worker_tienen_la_misma_aridad():
    """TODOS los return de _gps_destino_y_tiempos_dia deben devolver 4 valores.

    La función tiene un return temprano (día sin destinos GPS imputados) además del
    final. Si uno queda en 3 y el otro en 4, el caller falla solo para ciertos días
    —justo el caso difícil de reproducir en una corrida—. Se verifica de forma
    estática para no depender de armar todos los insumos del worker.
    """
    import ast
    import inspect
    import textwrap

    legs = pytest.importorskip("urbantrips.datamodel.legs")
    src = textwrap.dedent(inspect.getsource(legs._gps_destino_y_tiempos_dia))
    fn = ast.parse(src).body[0]

    aridades = [
        len(node.value.elts)
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    ]

    assert aridades, "no se encontró ningún return con tupla"
    assert set(aridades) == {4}, (
        f"todos los return deben devolver 4 valores; se encontró {aridades}"
    )
