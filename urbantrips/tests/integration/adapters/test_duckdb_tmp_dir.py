"""Todo lo temporal de DuckDB va al `tmp_dir` del config.

Sin `temp_directory` seteado, una conexion sobre archivo derrama al lado del
.duckdb (o sea al disco de db_dir) y una in-memory a `.tmp` relativo al cwd
--- eso dejo 54 GB en la raiz del repo el 2026-08-29. `tmp_dir` en el yaml
manda todo a un disco elegido; en blanco, al temp del sistema.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from urbantrips.storage.adapters.duckdb.dash import DuckDBDashAdapter
from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter
from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter
from urbantrips.utils import paths as paths_mod


@pytest.fixture
def proyecto_con_tmp_dir(tmp_path, monkeypatch):
    """Proyecto cuyo yaml apunta `tmp_dir` a un directorio propio."""
    configs = tmp_path / "configs"
    configs.mkdir()
    tmp_dir = tmp_path / "otro_disco" / "ut_tmp"
    config_file = configs / "configuraciones_generales.yaml"
    config_file.write_text(f"tmp_dir: {tmp_dir}\n", encoding="utf-8")

    monkeypatch.setenv("URBANTRIPS_CONFIG", str(config_file))
    paths_mod.reset_paths()
    paths_mod.init_paths(tmp_path)
    yield tmp_path, tmp_dir
    paths_mod.reset_paths()


def _temp_directory(conn) -> Path:
    return Path(conn.execute("SELECT current_setting('temp_directory')").fetchone()[0])


@pytest.mark.parametrize(
    "adapter_cls, nombre",
    [
        (DuckDBDataAdapter, "data"),
        (DuckDBInsumoAdapter, "insumos"),
        (DuckDBDashAdapter, "dash"),
        (DuckDBGeneralAdapter, "general"),
    ],
)
def test_adaptadores_derraman_en_tmp_dir(proyecto_con_tmp_dir, adapter_cls, nombre):
    base, tmp_dir = proyecto_con_tmp_dir
    db_dir = base / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    adapter = adapter_cls(db_dir / f"x_{nombre}.duckdb")
    try:
        assert _temp_directory(adapter._conn) == tmp_dir
    finally:
        adapter.close()


def test_adaptador_read_only_tambien(proyecto_con_tmp_dir):
    """Los workers abren en read_only; DuckDB acepta el SET igual."""
    base, tmp_dir = proyecto_con_tmp_dir
    db_dir = base / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / "x_data.duckdb"

    DuckDBDataAdapter(path).close()

    adapter = DuckDBDataAdapter(path, read_only=True)
    try:
        assert _temp_directory(adapter._conn) == tmp_dir
    finally:
        adapter.close()


def test_configure_global_duckdb_usa_tmp_dir(proyecto_con_tmp_dir):
    """La conexion implicita de `duckdb.sql(...)`, la que usan los workers."""
    import duckdb
    from urbantrips.storage.adapters.duckdb.data import configure_global_duckdb

    _, tmp_dir = proyecto_con_tmp_dir
    configure_global_duckdb(force=True)
    assert _temp_directory(duckdb) == tmp_dir


def test_tmp_dir_se_crea_solo(proyecto_con_tmp_dir):
    base, tmp_dir = proyecto_con_tmp_dir
    db_dir = base / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    assert not tmp_dir.exists()
    DuckDBDataAdapter(db_dir / "x_data.duckdb").close()
    assert tmp_dir.is_dir()


# ── staging de parquet, no solo el spill ─────────────────────────────────────


def _rastrear_staging(monkeypatch, tmp_dir):
    """Envuelve TemporaryDirectory/mkdtemp para anotar donde se crean de verdad."""
    import tempfile as _tempfile

    creados: list[Path] = []
    real_td = _tempfile.TemporaryDirectory
    real_mkdtemp = _tempfile.mkdtemp

    class _TD(real_td):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            creados.append(Path(self.name))

    def _mkdtemp(*a, **kw):
        d = real_mkdtemp(*a, **kw)
        creados.append(Path(d))
        return d

    monkeypatch.setattr(_tempfile, "TemporaryDirectory", _TD)
    monkeypatch.setattr(_tempfile, "mkdtemp", _mkdtemp)
    return creados


def test_save_legs_stagea_el_parquet_en_tmp_dir(proyecto_con_tmp_dir, monkeypatch):
    import pandas as pd

    base, tmp_dir = proyecto_con_tmp_dir
    db_dir = base / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    adapter = DuckDBDataAdapter(db_dir / "x_data.duckdb")
    creados = _rastrear_staging(monkeypatch, tmp_dir)
    try:
        adapter.save_legs(
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "id_tarjeta": ["a", "b"],
                    "dia": ["2026-03-02", "2026-03-02"],
                    "id_viaje": [1, 1],
                    "id_etapa": [1, 2],
                }
            )
        )
    finally:
        adapter.close()

    assert creados, "save_legs no creo ningun directorio de staging"
    assert all(d.parent == tmp_dir for d in creados), creados


def test_upsert_chains_norm_stagea_el_parquet_en_tmp_dir(
    proyecto_con_tmp_dir, monkeypatch
):
    import pandas as pd

    base, tmp_dir = proyecto_con_tmp_dir
    db_dir = base / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    adapter = DuckDBDashAdapter(db_dir / "x_dash.duckdb")
    cols = [
        c[1]
        for c in adapter._conn.execute(
            "SELECT * FROM pragma_table_info('chains_norm')"
        ).fetchall()
    ]
    fila = {c: None for c in cols}
    fila.update(dia="2026-03-02", id_tarjeta="a", id_viaje=1)

    creados = _rastrear_staging(monkeypatch, tmp_dir)
    try:
        adapter.upsert_chains_norm(pd.DataFrame([fila]), ["2026-03-02"])
    finally:
        adapter.close()

    assert creados, "upsert_chains_norm no creo ningun directorio de staging"
    assert all(d.parent == tmp_dir for d in creados), creados


def test_infer_destinations_stagea_en_tmp_dir(proyecto_con_tmp_dir, monkeypatch):
    """El staging de destinos usa mkdtemp y lo pasa a los workers por argumento."""
    import pandas as pd

    from urbantrips.destinations.destinations import infer_destinations
    from urbantrips.tests.integration.test_pipeline_scenarios import _ctx, _make_legs

    base, tmp_dir = proyecto_con_tmp_dir
    ctx = _ctx(base)
    dia = "2024-01-01"
    ctx.data.save_legs(_make_legs(dia, n=3))
    ctx.data.save_run_days(pd.DataFrame({"dia": [dia]}))

    creados = _rastrear_staging(monkeypatch, tmp_dir)
    infer_destinations(ctx)

    assert creados, "infer_destinations no creo ningun directorio de staging"
    assert all(d.parent == tmp_dir for d in creados), creados
