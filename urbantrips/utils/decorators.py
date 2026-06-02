import datetime
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

_pipeline_start: float | None = None


def reset_timer() -> None:
    global _pipeline_start
    _pipeline_start = None


def duracion(f):
    @wraps(f)
    def wrap(*args, **kw):
        global _pipeline_start
        started_at = datetime.datetime.now()
        if _pipeline_start is None:
            _pipeline_start = time.perf_counter()

        logger.info("Iniciado %s (%s)", f.__name__, started_at.strftime("%Y-%m-%d %H:%M:%S"))

        ts = time.perf_counter()
        try:
            result = f(*args, **kw)
        except BaseException:
            te = time.perf_counter()
            failed_at = datetime.datetime.now()
            elapsed = te - ts
            total = te - _pipeline_start
            logger.error(
                "Falló %s (%s). Tardó %.2fs (total acumulado: %.2fs)",
                f.__name__, failed_at.strftime("%Y-%m-%d %H:%M:%S"), elapsed, total,
            )
            raise
        else:
            te = time.perf_counter()
            finished_at = datetime.datetime.now()
            elapsed = te - ts
            total = te - _pipeline_start
            logger.info(
                "Finalizado %s (%s). Tardó %.2fs (total acumulado: %.2fs)",
                f.__name__, finished_at.strftime("%Y-%m-%d %H:%M:%S"), elapsed, total,
            )
            return result
    return wrap
