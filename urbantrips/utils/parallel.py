# urbantrips/utils/parallel.py
"""Que un worker muerto degrade a serie en vez de tirar abajo la corrida.

El OOM killer del kernel mata un worker con SIGKILL (sin traceback), y
`concurrent.futures` traduce eso a `BrokenProcessPool` en el `future.result()`.
Hasta ahora nadie lo atrapaba: la excepción subía hasta `run_all` y se perdían
horas de cómputo. Pasó dos veces con el mismo cliente — el 2026-08-23 en
`infer_destinations` y el 2026-09-15 en `assign_time_distances`.

Las etapas día-por-día ya tienen un camino serial implementado y declarado
bit-idéntico al paralelo; sólo que se elegía *a priori* por presupuesto de RAM y
no había vuelta atrás. Estas dos funciones son el puente: `cosechar` separa los
días que el pool alcanzó a terminar de los que hay que rehacer, y
`log_pool_muerto` explica qué pasó y qué tocar para la próxima.
"""
import logging
import math
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool

logger = logging.getLogger(__name__)


# `MemoryError` va acá porque un worker también puede morir por un `malloc`
# fallido DENTRO de Python: eso es una excepción normal que viaja por el futuro,
# no una muerte por señal, pero la causa y la respuesta son las mismas.
#
# Todo lo demás se propaga igual que siempre. Un `ValueError` en el worker es un
# bug, no algo para rehacer en serie: taparlo con un fallback lo convertiría en
# un cómputo que corre dos veces y falla igual, pero sin decir por qué.
POOL_MUERTO = (BrokenProcessPool, MemoryError)


def cosechar(futures, pendientes, *, etapa, n_workers, modelo=None):
    """Consume `futures` en streaming, apartando lo que el pool no pudo terminar.

    Generador de `(item, resultado)` por cada futuro que completó bien. Los que
    rompen porque el pool murió NO propagan: su `item` se agrega a `pendientes`
    —una lista del caller— para que lo rehaga en proceso.

    **Es un generador a propósito.** Devolver `list[(item, resultado)]` obligaría
    a retener los resultados de `n_workers` días a la vez, que es exactamente el
    recurso que nos mató. El caller tiene que poder seguir consumiendo-guardando-
    liberando de a uno.

    Parameters
    ----------
    futures : dict
        {future: item}. `item` es lo que el caller necesita para rehacer ese
        trabajo en serie (el día, o `(día, idx)` donde el idx importa).
    pendientes : list
        Lista del CALLER, se le appendean los items a rehacer. Out-param en vez
        de valor de retorno porque un generador no puede devolver las dos cosas.
    etapa : str
        Nombre para el log (p. ej. "assign_time_distances").
    n_workers : int
        Con cuántos workers se estaba corriendo, para el diagnóstico.
    modelo : _ModeloRam or None
        El modelo de memoria que decidió `n_workers`, para poder sugerir un
        `parallel_day_gb` concreto.
    """
    ya_hechos = []
    roto = None

    for future in as_completed(futures):
        item = futures[future]
        try:
            resultado = future.result()
        except POOL_MUERTO as exc:
            # Cuando el pool se rompe, TODOS los futuros que quedaban pendientes
            # levantan. Se juntan todos y se loguea una sola vez al final, con la
            # lista completa.
            if roto is None:
                roto = exc
            pendientes.append(item)
            continue
        ya_hechos.append(item)
        yield item, resultado

    if roto is not None:
        log_pool_muerto(etapa, pendientes, ya_hechos, n_workers, modelo, roto)


def _ram_libre():
    """(libre_gb, total_gb, swap_gb) o None. psutil es opcional en este proyecto."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return (vm.available / 2**30, vm.total / 2**30, psutil.swap_memory().used / 2**30)
    except Exception:
        return None


def _parallel_day_gb_sugerido(n_workers, por_dia):
    """Qué poner en `parallel_day_gb` para que el autotune elija un worker menos.

    El autotune hace `workers = int(presupuesto / por_dia + 0.1)`. Si con `por_dia`
    dio `n_workers`, entonces el presupuesto ronda `n_workers * por_dia`. Para que
    dé `n_workers - 1` hace falta un divisor de `presupuesto / (n_workers - 1)`.

    Es aproximado —no conocemos el presupuesto exacto acá— pero la diferencia
    entre "andá a tocar un yaml" y "poné 28" justifica el redondeo.
    """
    if not por_dia or n_workers < 2:
        return None
    return int(math.ceil(n_workers * por_dia / (n_workers - 1)))


def log_pool_muerto(etapa, pendientes, ya_hechos, n_workers, modelo, exc):
    """Explica la muerte del pool y qué hacer, en un solo bloque.

    Lo único que quedaba antes era el traceback de `BrokenProcessPool`, que no
    dice ni que fue memoria, ni qué días se perdieron, ni que la corrida puede
    seguir.

    Nunca levanta: corre DENTRO del manejo de una excepción, y que el
    diagnóstico rompa sería peor que no tenerlo.
    """
    try:
        por_dia = getattr(modelo, "por_dia", None)

        lineas = [
            f"[{etapa}] el pool de procesos murió ({type(exc).__name__}). Casi seguro",
            "  el OOM killer del kernel mató a un worker (Python no puede confirmarlo:",
            "  el worker muere por SIGKILL, sin traceback).",
        ]

        if pendientes:
            lineas.append(
                "  Días EN VUELO perdidos, se rehacen: "
                + ", ".join(str(p) for p in pendientes)
            )
        if ya_hechos:
            lineas.append(
                "  Días de este chunk ya guardados, no se rehacen: "
                + ", ".join(str(p) for p in ya_hechos)
            )

        if por_dia:
            lineas.append(
                f"  Decisión de paralelismo: {n_workers} worker(s) × {por_dia:.1f} GB/día "
                "estimados (ver [day_footprint])."
            )
        else:
            lineas.append(f"  Decisión de paralelismo: {n_workers} worker(s).")

        ram = _ram_libre()
        if ram is not None:
            libre, total, swap = ram
            lineas.append(
                f"  RAM libre AHORA: {libre:.1f} GB / {total:.1f} GB; "
                f"swap en uso {swap:.1f} GB."
            )

        lineas.append(
            "  → Se continúa EN SERIE el resto de la etapa (más lento, resultados "
            "idénticos)."
        )
        lineas.append("    La corrida NO se aborta.")
        lineas.append("  → Para la próxima corrida, en configs/tuning.yaml:")

        sugerido = _parallel_day_gb_sugerido(n_workers, por_dia)
        if sugerido is not None:
            lineas.append(
                f"        parallel_day_gb: {sugerido}      # el estimador quedó corto; "
                f"esto daría {n_workers - 1} worker(s)"
            )
        lineas.append(
            "        parallel_day_workers: 1  # serial desde el arranque"
        )

        logger.error("\n".join(lineas))
    except Exception:  # pragma: no cover - el diagnóstico nunca debe romper
        logger.error(
            "[%s] el pool de procesos murió (%s); se continúa en serie.",
            etapa, type(exc).__name__,
        )
