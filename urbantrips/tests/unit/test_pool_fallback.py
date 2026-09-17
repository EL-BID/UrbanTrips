# urbantrips/tests/unit/test_pool_fallback.py
"""Un worker muerto degrada a serie en vez de tirar abajo la corrida.

El OOM killer mata un worker con SIGKILL y `concurrent.futures` lo traduce a
`BrokenProcessPool` en el `future.result()`. Hasta ahora nadie lo atrapaba y se
perdían horas de cómputo. Pasó dos veces con el mismo cliente: el 2026-08-23 en
`infer_destinations` y el 2026-09-15 en `assign_time_distances`.

Cómo se testea sin matar procesos: se monkeypatchea `ProcessPoolExecutor` por un
executor SINCRÓNICO (el patrón ya estaba en `test_legs_worker_contract.py:87-105`)
cuyos futuros pueden levantar `BrokenProcessPool` a pedido. Ejercita el
`submit()`/`result()` real, en milisegundos y sin señales. El test 8 valida contra
un `BrokenProcessPool` GENUINO que el fake no esté mintiendo.
"""
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import pandas as pd
import pytest

# Celda h3 real: grid_distance(c, c) == 0, así pasa el filtro de ring del main.
H3_CELL = "89c2e3a95c3ffff"


# ─────────────────────────── andamio ───────────────────────────

class _Fut:
    """Futuro sincrónico: ya trae el resultado, o la excepción a levantar."""

    def __init__(self, resultado=None, exc=None):
        self._r = resultado
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._r


def _exec_factory(romper_desde_submit=None, exc=None, contador=None):
    """Executor sincrónico. `romper_desde_submit` = nro de submit (1-based) desde
    el cual los futuros levantan; None = ninguno rompe."""

    class _Exec:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args, **kw):
            if contador is not None:
                contador.append(1)
            n = len(contador) if contador is not None else 0
            if romper_desde_submit is not None and n >= romper_desde_submit:
                return _Fut(exc=exc or BrokenProcessPool("worker muerto"))
            return _Fut(resultado=fn(*args, **kw))

    return _Exec


def _ctx(dias):
    legs_df = pd.DataFrame({
        "dia": dias, "id": list(range(1, len(dias) + 1)),
        "id_tarjeta": ["C1"] * len(dias),
        "id_viaje": [1] * len(dias), "id_etapa": [1] * len(dias),
    })

    class _Data:
        def query(self, sql):
            return legs_df.copy()

        def get_run_days(self):
            return pd.DataFrame({"dia": dias})

        def execute(self, sql):
            pass

        def append_raw(self, df, table):
            pass

    insumos = SimpleNamespace(
        get_metadata_lineas=lambda: pd.DataFrame(
            {"id_linea": [1], "id_linea_agg": [1], "modo": ["autobus"]}
        ),
        get_matrix_validation=lambda: pd.DataFrame({
            "id_linea_agg": [1], "id_ramal": [None],
            "parada": [H3_CELL], "area_influencia": [H3_CELL],
        }),
    )
    return SimpleNamespace(data=_Data(), insumos=insumos)


@pytest.fixture
def entorno(monkeypatch):
    """`assign_time_distances` con los insumos falsos y el cómputo stubbeado.

    Devuelve (legs, guardados, submits, procesados_en_serie).
    """
    legs = pytest.importorskip("urbantrips.datamodel.legs")
    import urbantrips.utils.parallel as parallel_mod

    monkeypatch.setattr(
        legs, "leer_configs_generales",
        lambda autogenerado=True: {"usa_archivo_gps": True, "resolucion_h3": 9},
    )
    monkeypatch.setattr(legs, "modos_con_ramal", lambda cfg: set())
    monkeypatch.setattr(
        legs, "_fetch_time_distance_inputs_dia",
        lambda ctx, dia, nxt: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(parallel_mod, "as_completed", lambda fs: list(fs))

    en_serie = []
    real_fetch = legs._fetch_legs_all_dia

    def _fetch(ctx, dia):
        return pd.DataFrame({"dia": [dia], "id": [1]})

    monkeypatch.setattr(legs, "_fetch_legs_all_dia", _fetch)

    def _computo(dia, *a, **k):
        # El resultado depende del día: así el test de bit-identidad compara algo.
        tt = pd.DataFrame({"dia": [dia], "id": [hash(dia) % 1000], "distance_od": [1.0]})
        return tt, tt.copy(), None, {"pct_gps_imputado": 50.0}

    monkeypatch.setattr(legs, "_gps_destino_y_tiempos_dia", _computo)

    guardados = []
    monkeypatch.setattr(
        legs, "_save_travel_times_dia",
        lambda ctx, dia, tt, ttt, lg: guardados.append((dia, tt.copy())),
    )
    return SimpleNamespace(
        legs=legs, guardados=guardados, en_serie=en_serie, real_fetch=real_fetch
    )


def _forzar_workers(monkeypatch, legs, n):
    monkeypatch.setattr(
        legs, "_parallel_day_workers",
        lambda nd, per_day_gb=None, main_extra_gb=0.0: n,
    )


# ─────────────────────────── los tests ───────────────────────────

def test_1_el_pool_muerto_no_aborta_la_corrida(monkeypatch, entorno):
    """Antes: BrokenProcessPool subía hasta run_all y se perdían horas."""
    legs = entorno.legs
    dias = ["2024-03-11", "2024-03-12", "2024-03-13"]
    _forzar_workers(monkeypatch, legs, 3)
    monkeypatch.setattr(
        legs, "ProcessPoolExecutor", _exec_factory(romper_desde_submit=1, contador=[])
    )

    legs.assign_time_distances(_ctx(dias))  # no debe levantar

    assert sorted(d for d, _ in entorno.guardados) == dias


def test_2_ningun_dia_se_guarda_dos_veces(monkeypatch, entorno):
    """El hallazgo que define la mecánica del fallback.

    `_save_travel_times_dia` es un `append_raw` sin DELETE, y los DELETE corren una
    sola vez ANTES del loop. Si el pool muere con el día 1 ya guardado y los días
    2-3 en vuelo, rehacer el CHUNK entero duplicaría el día 1. Por eso el fallback
    es por día pendiente.
    """
    legs = entorno.legs
    dias = ["2024-03-11", "2024-03-12", "2024-03-13"]
    _forzar_workers(monkeypatch, legs, 3)
    # el primer submit va bien, los dos siguientes rompen
    monkeypatch.setattr(
        legs, "ProcessPoolExecutor",
        _exec_factory(romper_desde_submit=2, contador=[]),
    )

    legs.assign_time_distances(_ctx(dias))

    guardados = [d for d, _ in entorno.guardados]
    assert sorted(guardados) == dias
    for dia in dias:
        assert guardados.count(dia) == 1, (
            f"{dia} se guardó {guardados.count(dia)} veces; el append duplicaría filas"
        )


def test_3_sticky_no_reintenta_paralelo_despues_de_la_rotura(monkeypatch, entorno):
    """Reintentar es apostar a que el OOM killer elija otra vez un worker."""
    legs = entorno.legs
    dias = [f"2024-03-{d:02d}" for d in range(11, 17)]  # 6 días, 3 chunks de 2
    _forzar_workers(monkeypatch, legs, 2)
    submits = []
    monkeypatch.setattr(
        legs, "ProcessPoolExecutor",
        _exec_factory(romper_desde_submit=1, contador=submits),
    )

    legs.assign_time_distances(_ctx(dias))

    # sólo se intentó el primer chunk (2 submits); el resto fue en serie
    assert len(submits) == 2, f"se siguió submitteando tras la rotura: {len(submits)}"
    assert sorted(d for d, _ in entorno.guardados) == dias


def test_4_bit_identidad_serial_vs_fallback(monkeypatch, entorno):
    """La restricción dura: el fallback no puede mover un número.

    Se corre el mismo fixture por el camino serial puro y por el paralelo-que-rompe,
    y se comparan los frames que llegaron a `_save_travel_times_dia`.
    """
    legs = entorno.legs
    dias = ["2024-03-11", "2024-03-12", "2024-03-13"]

    _forzar_workers(monkeypatch, legs, 1)      # serial puro
    legs.assign_time_distances(_ctx(dias))
    serial = {d: df for d, df in entorno.guardados}

    entorno.guardados.clear()
    _forzar_workers(monkeypatch, legs, 3)
    monkeypatch.setattr(
        legs, "ProcessPoolExecutor",
        _exec_factory(romper_desde_submit=2, contador=[]),
    )
    legs.assign_time_distances(_ctx(dias))
    fallback = {d: df for d, df in entorno.guardados}

    assert set(serial) == set(fallback) == set(dias)
    for dia in dias:
        pd.testing.assert_frame_equal(
            serial[dia].sort_values(["dia", "id"]).reset_index(drop=True),
            fallback[dia].sort_values(["dia", "id"]).reset_index(drop=True),
        )


def test_5_las_excepciones_reales_siguen_propagando(monkeypatch, entorno):
    """Un ValueError en el worker es un BUG, no algo para rehacer en serie.

    Si el fallback se lo comiera, el cómputo correría dos veces y fallaría igual,
    pero sin decir por qué.
    """
    legs = entorno.legs
    dias = ["2024-03-11", "2024-03-12"]
    _forzar_workers(monkeypatch, legs, 2)
    monkeypatch.setattr(
        legs, "ProcessPoolExecutor",
        _exec_factory(
            romper_desde_submit=1, exc=ValueError("bug de verdad"), contador=[]
        ),
    )

    with pytest.raises(ValueError, match="bug de verdad"):
        legs.assign_time_distances(_ctx(dias))


def test_6_el_diagnostico_dice_que_hacer(monkeypatch, entorno, caplog):
    """Antes el único rastro era el traceback: ni memoria, ni qué días, ni qué tocar."""
    legs = entorno.legs
    dias = ["2024-03-11", "2024-03-12", "2024-03-13"]
    _forzar_workers(monkeypatch, legs, 3)
    monkeypatch.setattr(
        legs, "ProcessPoolExecutor",
        _exec_factory(romper_desde_submit=2, contador=[]),
    )

    with caplog.at_level("ERROR"):
        legs.assign_time_distances(_ctx(dias))

    msg = "\n".join(r.message for r in caplog.records)
    assert "assign_time_distances" in msg
    assert "OOM killer" in msg
    assert "2024-03-12" in msg and "2024-03-13" in msg, "faltan los días en vuelo"
    assert "2024-03-11" in msg, "falta el día ya guardado"
    assert "tuning.yaml" in msg
    assert "parallel_day_gb" in msg
    assert "parallel_day_workers: 1" in msg
    assert "NO se aborta" in msg


# ─────────────────── unit de la pieza compartida ───────────────────

@pytest.fixture
def as_completed_falso(monkeypatch):
    """`as_completed` real exige Futures reales; acá los futuros son de mentira."""
    from urbantrips.utils import parallel as P

    monkeypatch.setattr(P, "as_completed", lambda fs: list(fs))


def test_7a_cosechar_es_generador_de_verdad(as_completed_falso):
    """Si acumulara resultados retendría n_workers días a la vez — el recurso que
    nos mató. Tiene que yieldear el primer OK antes de tocar el último futuro."""
    from urbantrips.utils import parallel as P

    tocados = []

    class _F:
        def __init__(self, v):
            self.v = v

        def result(self):
            tocados.append(self.v)
            return self.v

    futures = {_F(1): "a", _F(2): "b", _F(3): "c"}
    gen = P.cosechar(futures, [], etapa="t", n_workers=3)
    primero = next(gen)

    assert primero[1] in (1, 2, 3)
    assert len(tocados) == 1, (
        f"consumió {len(tocados)} futuros antes del primer yield: no es streaming"
    )


def test_7b_cosechar_aparta_los_rotos_y_deja_pasar_el_resto(as_completed_falso):
    from urbantrips.utils import parallel as P

    class _F:
        def __init__(self, v, exc=None):
            self.v, self.exc = v, exc

        def result(self):
            if self.exc:
                raise self.exc
            return self.v

    futures = {
        _F("ok1"): "dia1",
        _F(None, BrokenProcessPool("x")): "dia2",
        _F(None, MemoryError("malloc")): "dia3",
    }
    pendientes = []
    salidas = list(P.cosechar(futures, pendientes, etapa="t", n_workers=3))

    assert salidas == [("dia1", "ok1")]
    assert sorted(pendientes) == ["dia2", "dia3"], (
        "MemoryError también es muerte por memoria y va al fallback"
    )


def test_7c_cosechar_no_se_come_otras_excepciones(as_completed_falso):
    from urbantrips.utils import parallel as P

    class _F:
        def result(self):
            raise KeyError("bug")

    with pytest.raises(KeyError):
        list(P.cosechar({_F(): "dia1"}, [], etapa="t", n_workers=1))


def test_7d_sugerencia_de_parallel_day_gb():
    """El número concreto es la diferencia entre 'tocá un yaml' y 'poné 28'."""
    from urbantrips.utils.parallel import _parallel_day_gb_sugerido

    # 3 workers x 18.4 GB => presupuesto ~55 GB; para 2 workers hace falta ~27.6 -> 28
    assert _parallel_day_gb_sugerido(3, 18.4) == 28
    assert _parallel_day_gb_sugerido(1, 18.4) is None   # no hay worker que sacar
    assert _parallel_day_gb_sugerido(3, None) is None


def test_7e_el_diagnostico_nunca_levanta():
    """Corre DENTRO del handler de una excepción: si rompe, tapa el error real."""
    from urbantrips.utils.parallel import log_pool_muerto

    # modelo basura a propósito
    log_pool_muerto("etapa", ["d1"], [], 2, object(), BrokenProcessPool("x"))


@pytest.mark.slow
def test_8_brokenprocesspool_genuino(monkeypatch):
    """Valida que el executor falso de los tests 1-6 no miente sobre la semántica.

    Un worker que hace os._exit(1) produce un BrokenProcessPool REAL, sin OOM y sin
    señales. Es el único test que necesita procesos de verdad.
    """
    from concurrent.futures import ProcessPoolExecutor
    from urbantrips.utils import parallel as P

    with ProcessPoolExecutor(max_workers=1) as ex:
        futures = {ex.submit(_suicida): "dia1"}
        pendientes = []
        salidas = list(P.cosechar(futures, pendientes, etapa="t", n_workers=1))

    assert salidas == []
    assert pendientes == ["dia1"], (
        "un BrokenProcessPool real tiene que caer en el fallback igual que el falso"
    )


def _suicida():
    """Top-level para que sea pickleable."""
    import os

    os._exit(1)
