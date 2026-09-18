# urbantrips/tests/unit/test_worker_pool.py
"""`worker_pool`: apagado ordenado de los multiprocessing.Pool.

`with multiprocessing.Pool(...)` llama a `terminate()` al salir en vez de
`close()` + `join()`. El helper hace el apagado documentado y, sobre todo,
garantiza el `join()` en ambos caminos (éxito y error) para no dejar workers
sueltos.
"""
import multiprocessing

import pytest

from urbantrips.utils.utils import worker_pool


class _PoolFalso:
    """Registra el orden de las llamadas del ciclo de vida."""

    instancias = []

    def __init__(self, processes=None):
        self.processes = processes
        self.llamadas = []
        _PoolFalso.instancias.append(self)

    def close(self):
        self.llamadas.append("close")

    def terminate(self):
        self.llamadas.append("terminate")

    def join(self):
        self.llamadas.append("join")


@pytest.fixture
def pool_falso(monkeypatch):
    _PoolFalso.instancias = []
    monkeypatch.setattr(multiprocessing, "Pool", _PoolFalso)
    return _PoolFalso


def test_camino_feliz_cierra_y_espera(pool_falso):
    with worker_pool(4) as pool:
        assert pool.processes == 4

    assert pool_falso.instancias[0].llamadas == ["close", "join"], (
        "el camino de éxito tiene que cerrar ordenadamente, no terminar"
    )


def test_ante_excepcion_termina_pero_igual_espera(pool_falso):
    with pytest.raises(RuntimeError, match="boom"):
        with worker_pool(4):
            raise RuntimeError("boom")

    # terminate porque el cómputo se abortó, pero join igual: es lo que evita
    # que queden procesos huérfanos
    assert pool_falso.instancias[0].llamadas == ["terminate", "join"]


def test_keyboardinterrupt_tambien_limpia(pool_falso):
    """BaseException, no sólo Exception: un Ctrl-C no debe dejar workers vivos."""
    with pytest.raises(KeyboardInterrupt):
        with worker_pool(2):
            raise KeyboardInterrupt

    assert pool_falso.instancias[0].llamadas == ["terminate", "join"]


def test_pool_real_no_deja_procesos_vivos():
    """Con procesos de verdad: ninguno queda vivo al salir del contexto."""
    with worker_pool(2) as pool:
        assert pool.map(abs, [-1, -2, -3]) == [1, 2, 3]
        hijos = list(pool._pool)

    assert hijos, "el pool tendría que haber levantado workers"
    for p in hijos:
        p.join(timeout=10)
        assert not p.is_alive(), f"worker {p.pid} sigue vivo tras salir del contexto"
