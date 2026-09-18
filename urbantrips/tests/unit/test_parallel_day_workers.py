# urbantrips/tests/unit/test_parallel_day_workers.py
"""Escalón del autotune de `_parallel_day_workers`.

El divisor (~12 GB por día) es una estimación, pero el cálculo usaba un floor
duro: 1,95 días de presupuesto daban 1 worker y 2,05 daban 2. Como
`psutil.available` fluctúa, una misma corrida decidía 1, 2 y 1. La tolerancia
del 10% redondea hacia arriba sólo cuando falta menos de ~1,2 GB.
"""
import pytest

from urbantrips.datamodel import legs as legs_module


PER_DAY_GB = 12.0
RESERVA_GB = 8.0


@pytest.fixture
def sin_override(monkeypatch, tmp_path):
    """Neutraliza el override de tuning.yaml y fija la reserva del main."""
    class _Paths:
        base = tmp_path  # sin configs/tuning.yaml -> no hay override

    monkeypatch.setattr("urbantrips.utils.paths.get_paths", lambda: _Paths())
    monkeypatch.setattr(legs_module, "_duckdb_memory_limit_gb", lambda: RESERVA_GB)


def _workers(monkeypatch, presupuesto_gb, n_days=7):
    """Corre el autotune con un presupuesto dado (RAM libre − reserva)."""
    import psutil

    class _VM:
        available = int((presupuesto_gb + RESERVA_GB) * 2**30)

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    return legs_module._parallel_day_workers(n_days)


@pytest.mark.parametrize("dias_de_presupuesto, esperado", [
    (0.5, 1),    # menos de un día -> el piso es 1
    (1.0, 1),
    (1.5, 1),    # lejos del escalón, no se redondea
    (1.89, 1),   # justo por debajo de la tolerancia
    (1.95, 2),   # EL CASO DEL BUG: antes daba 1
    (2.0, 2),
    (2.05, 2),
    (2.95, 3),
    (5.0, 3),    # tope de 3
])
def test_escalon_con_tolerancia(monkeypatch, sin_override, dias_de_presupuesto, esperado):
    got = _workers(monkeypatch, dias_de_presupuesto * PER_DAY_GB)
    assert got == esperado, (
        f"con {dias_de_presupuesto} días de presupuesto se esperaban {esperado} "
        f"worker(s), no {got}"
    )


def test_la_tolerancia_no_sobre_commitea_de_mas(monkeypatch, sin_override):
    """No debe redondear hacia arriba con más del 10% faltante.

    Sobre-committear fue lo que produjo el thrashing de la corrida 2026-07-17,
    así que el margen tiene que quedarse chico.
    """
    assert _workers(monkeypatch, 1.5 * PER_DAY_GB) == 1
    assert _workers(monkeypatch, 2.5 * PER_DAY_GB) == 2


def test_nunca_baja_de_uno_ni_supera_los_dias(monkeypatch, sin_override):
    assert _workers(monkeypatch, 0.0) == 1
    assert _workers(monkeypatch, -100.0) == 1        # presupuesto negativo
    assert _workers(monkeypatch, 5 * PER_DAY_GB, n_days=1) == 1  # 1 solo día
