import pytest

from urbantrips.utils.run_process import _resolve_n_batches
from urbantrips.storage.context import StorageContext
from urbantrips.storage.adapters.memory.adapters import (
    InMemoryDashAdapter,
    InMemoryDataAdapter,
    InMemoryGeneralAdapter,
    InMemoryInsumoAdapter,
)


def test_resolve_n_batches_uses_config_when_set(monkeypatch):
    monkeypatch.setattr(
        "urbantrips.utils.run_process.leer_configs_generales",
        lambda *args, **kwargs: {"n_batches": 42},
    )

    class _Ctx:
        pass

    assert _resolve_n_batches(_Ctx()) == 42


@pytest.mark.parametrize("raw", [None, "", 0])
def test_resolve_n_batches_auto_when_null_empty_or_zero(monkeypatch, raw):
    """A null/empty value or 0 all fall back to auto-resolution."""
    import pandas as pd
    from urbantrips.utils import run_process

    monkeypatch.setattr(
        "urbantrips.utils.run_process.leer_configs_generales",
        lambda *args, **kwargs: {"n_batches": raw},
    )

    class _Data:
        def query(self, sql):
            # nothing stamped yet → MAX(batch_id) is NULL → auto-tune path
            return pd.DataFrame({"m": [None]})

    class _Ctx:
        data = _Data()

    monkeypatch.setattr(
        run_process,
        "_auto_n_batches",
        lambda ctx, safety_factor=0.4: 99,
    )

    assert _resolve_n_batches(_Ctx()) == 99


def test_resolve_n_batches_missing_key_auto(monkeypatch):
    """A config with no n_batches key falls back to auto-resolution."""
    import pandas as pd
    from urbantrips.utils import run_process

    monkeypatch.setattr(
        "urbantrips.utils.run_process.leer_configs_generales",
        lambda *args, **kwargs: {},  # no n_batches key
    )

    class _Data:
        def query(self, sql):
            return pd.DataFrame({"m": [None]})  # nothing stamped

    class _Ctx:
        data = _Data()

    monkeypatch.setattr(
        run_process,
        "_auto_n_batches",
        lambda ctx, safety_factor=0.4: 99,
    )

    assert _resolve_n_batches(_Ctx()) == 99


def test_resolve_n_batches_inherits_stamped_partition(monkeypatch):
    """When auto and transacciones is already stamped, reuse MAX(batch_id) + 1."""
    import pandas as pd
    from urbantrips.utils import run_process

    monkeypatch.setattr(
        "urbantrips.utils.run_process.leer_configs_generales",
        lambda *args, **kwargs: {},  # auto mode
    )

    auto_calls = []

    class _Data:
        def query(self, sql):
            # transacciones stamped with batch_ids 0..29 → MAX = 29
            return pd.DataFrame({"m": [29]})

    class _Ctx:
        data = _Data()

    monkeypatch.setattr(
        run_process,
        "_auto_n_batches",
        lambda ctx, safety_factor=0.4: auto_calls.append(1) or 99,
    )

    assert _resolve_n_batches(_Ctx()) == 30  # 29 + 1, not the auto-tuned 99
    assert auto_calls == []  # stamped count wins, auto-tune never runs


def test_ingest_all_days_uploads_gps_after_run_days(monkeypatch):
    import pandas as pd

    from urbantrips.utils import run_process

    calls = []

    class _Data:
        def clear_raw(self):
            calls.append("clear_raw")

        def get_max_id(self, table):
            assert table == "transacciones"
            return 0

        def standardize_raw_to_transacciones(self, n_batches, id_offset):
            calls.append(("standardize", n_batches, id_offset))

        def query(self, sql):
            assert "SELECT DISTINCT dia" in sql
            calls.append("query_days")
            return pd.DataFrame({"dia": ["2025-10-15"]})

        def save_run_days(self, days):
            calls.append(("save_run_days", days["dia"].tolist()))

    class _Ctx:
        data = _Data()

    config = {
        "nombres_variables_trx": {},
        "formato_fecha": "%Y-%m-%d %H:%M:%S",
        "tipo_trx_invalidas": None,
        "lineas_contienen_ramales": True,
        "nombre_archivo_trx": "20251015_trx.csv",
        "usa_archivo_gps": True,
        "nombre_archivo_gps": "20251015_gps.csv",
        "nombres_variables_gps": {},
        "n_batches": 30,
    }

    monkeypatch.setattr(
        "urbantrips.utils.check_configs.check_config",
        lambda corrida: calls.append(("check_config", corrida)),
    )
    monkeypatch.setattr(
        run_process, "leer_configs_generales", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        "urbantrips.datamodel.ingestion.ingest_day_csv",
        lambda **kwargs: calls.append(("ingest_day_csv", kwargs["csv_path"])),
    )
    monkeypatch.setattr(
        "urbantrips.datamodel.transactions.process_and_upload_gps_table",
        lambda **kwargs: calls.append(("gps", kwargs["nombre_archivo_gps"])),
    )

    run_process._ingest_all_days(_Ctx(), ["20251015"])

    assert calls.index(("save_run_days", ["2025-10-15"])) < calls.index(
        ("gps", "20251015_gps.csv")
    )


def test_run_all_splits_batch_and_global_phases(monkeypatch):
    from urbantrips.storage.ports import BatchSpec
    from urbantrips.utils import run_process

    calls = []
    ctx = StorageContext(
        data=InMemoryDataAdapter(),
        insumos=InMemoryInsumoAdapter(),
        dash=InMemoryDashAdapter(),
        general=InMemoryGeneralAdapter(),
    )

    config = {
        "ordenamiento_transacciones": "fecha_completa",
        "ventana_viajes": 90,
        "ventana_duplicado": 5,
        "n_batches": 2,
    }

    monkeypatch.setattr(run_process, "borrar_corridas", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_process, "inicializo_ambiente", lambda ctx: ["run01"])
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_generales",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(run_process, "_resolve_n_batches", lambda ctx: 2)
    monkeypatch.setattr(
        run_process,
        "_ingest_all_days",
        lambda ctx, corridas: calls.append(("ingest", tuple(corridas))),
    )
    monkeypatch.setattr(
        ctx.data,
        "get_user_batches",
        lambda n_batches: [BatchSpec(0, n_batches), BatchSpec(1, n_batches)],
    )
    monkeypatch.setattr(
        run_process,
        "_create_legs_for_batch",
        lambda ctx, batch, params: calls.append(("legs", batch.batch_id)),
    )
    monkeypatch.setattr(
        run_process,
        "_enrich_all_legs",
        lambda ctx, configs, batches=None: calls.append(
            ("enrich", tuple(batch.batch_id for batch in batches))
        ),
    )
    monkeypatch.setattr(
        run_process,
        "_build_final_outputs",
        lambda ctx: calls.append(("final", None)),
    )
    monkeypatch.setattr(
        "urbantrips.carto.routes.infer_routes_geoms",
        lambda ctx: calls.append(("routes_infer", None)),
    )
    monkeypatch.setattr(
        "urbantrips.carto.routes.build_routes_from_official_inferred",
        lambda ctx: calls.append(("routes_build", None)),
    )
    monkeypatch.setattr(
        "urbantrips.kpi.kpi.compute_kpi",
        lambda ctx: calls.append(("kpi", None)),
    )
    monkeypatch.setattr(
        "urbantrips.datamodel.misc.persist_indicators",
        lambda ctx: calls.append(("indicators", None)),
    )
    monkeypatch.setattr(
        "urbantrips.preparo_dashboard.preparo_dashboard.preparo_indicadores_dash",
        lambda ctx: calls.append(("dashboard", None)),
    )

    run_process.run_all(ctx=ctx, crear_dashboard=True)

    assert calls == [
        ("ingest", ("run01",)),
        ("legs", 0),
        ("legs", 1),
        ("enrich", (0, 1)),
        ("final", None),
        ("routes_infer", None),
        ("routes_build", None),
        ("kpi", None),
        ("indicators", None),
        ("dashboard", None),
    ]


def test_create_legs_for_batches_uses_parallel_workers(monkeypatch):
    import pandas as pd

    from urbantrips.storage.ports import BatchSpec
    from urbantrips.utils import run_process

    saved = []
    ctx = StorageContext(
        data=InMemoryDataAdapter(),
        insumos=InMemoryInsumoAdapter(),
        dash=InMemoryDashAdapter(),
        general=InMemoryGeneralAdapter(),
    )
    batches = [BatchSpec(0, 2), BatchSpec(1, 2)]

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def add_done_callback(self, fn):
            fn(self)

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            batch = args[0]
            legs_df = pd.DataFrame({"id_tarjeta": [f"card-{batch.batch_id}"]})
            dupes_df = pd.DataFrame()
            return ImmediateFuture((batch, legs_df, dupes_df))

    # _batch_id=0 → batch 0, _batch_id=1 → batch 1
    chunk_trx = pd.DataFrame({
        "id_tarjeta": ["cardA", "cardA", "cardB", "cardB"],
        "_batch_id":  [0,       0,       1,       1      ],
    })

    monkeypatch.setattr(run_process, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(run_process, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(run_process, "_can_parallelize_batches", lambda ctx: True)
    monkeypatch.setattr(
        run_process,
        "leer_configs_generales",
        lambda *args, **kwargs: {"resolucion_h3": 8},
    )
    monkeypatch.setattr(
        ctx.data, "get_transactions_for_chunk",
        lambda batch_ids, total_batches: chunk_trx[chunk_trx["_batch_id"].isin(batch_ids)].copy(),
    )
    monkeypatch.setattr(
        ctx.data,
        "save_legs",
        lambda df, batch=None: saved.append((batch.batch_id, df["id_tarjeta"].iloc[0])),
    )

    run_process._create_legs_for_batches(ctx, batches, {}, parallel_workers=2)

    assert saved == [(0, "card-0"), (1, "card-1")]


def test_check_prerequisites_legs_raises_when_etapas_empty(monkeypatch):
    from urbantrips.utils import run_process

    class _Data:
        def has_rows(self, table, where=None):
            return False

    class _Ctx:
        data = _Data()

    import pytest
    with pytest.raises(RuntimeError, match="ingest"):
        run_process.check_prerequisites("legs", _Ctx())


def test_check_prerequisites_outputs_raises_when_h3_empty(monkeypatch):
    """check_prerequisites for 'outputs' must check h3_o (not h3) — etapas has h3_o/h3_d."""
    from urbantrips.utils import run_process

    class _Data:
        def has_rows(self, table, where=None):
            if where == "h3_o IS NOT NULL":
                return False
            return True

    class _Ctx:
        data = _Data()

    import pytest
    with pytest.raises(RuntimeError, match="legs"):
        run_process.check_prerequisites("outputs", _Ctx())


def test_check_prerequisites_dashboard_raises_when_viajes_empty(monkeypatch):
    from urbantrips.utils import run_process

    class _Data:
        def has_rows(self, table, where=None):
            return False

    class _Ctx:
        data = _Data()

    import pytest
    with pytest.raises(RuntimeError, match="outputs"):
        run_process.check_prerequisites("dashboard", _Ctx())


def test_check_prerequisites_passes_when_data_present():
    from urbantrips.utils import run_process

    class _Data:
        def has_rows(self, table, where=None):
            return True

    class _Ctx:
        data = _Data()

    run_process.check_prerequisites("legs", _Ctx())
    run_process.check_prerequisites("outputs", _Ctx())
    run_process.check_prerequisites("dashboard", _Ctx())


def test_check_prerequisites_ingest_never_raises():
    from urbantrips.utils import run_process

    class _Ctx:
        pass

    run_process.check_prerequisites("ingest", _Ctx())
