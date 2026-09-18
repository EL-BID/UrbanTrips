"""Estimación del costo de RAM por día que alimenta `_parallel_day_workers`.

El divisor era la constante 12.0 GB: el presupuesto se adaptaba a la máquina (RAM
libre) pero la demanda no se adaptaba a los datos. En AMBA (~7,5-9 M etapas/día,
pico real 18-22 GB/día) eso elegía 2 workers y el pool moría por OOM.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from urbantrips.datamodel import legs as legs_module


DIAS = ["2024-01-01", "2024-01-02"]


@pytest.fixture
def sin_override(monkeypatch):
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_tuning", lambda: {}
    )


def _ctx(rows_por_tabla, dtypes_por_tabla):
    """ctx que responde el LIMIT 0 (dtypes) y el max(count) por día."""
    def fake_query(sql):
        tabla = next(t for t in rows_por_tabla if f"FROM {t} " in sql)
        if "LIMIT 0" in sql:
            return pd.DataFrame(
                {f"c{i}": pd.Series(dtype=dt)
                 for i, dt in enumerate(dtypes_por_tabla[tabla])}
            )
        return pd.DataFrame({"n": [rows_por_tabla[tabla]]})

    ctx = MagicMock()
    ctx.data.query.side_effect = fake_query
    ctx.insumos.query.side_effect = fake_query
    return ctx


def test_el_override_de_tuning_gana(monkeypatch):
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_tuning",
        lambda: {"parallel_day_gb": 20},
    )
    ctx = MagicMock()
    got = legs_module._estimated_day_footprint_gb(ctx, DIAS, [legs_module._tabla("etapas")])
    assert got == 20.0
    ctx.data.query.assert_not_called()  # no se mide nada si está forzado


def test_escala_con_las_filas_del_dia(sin_override):
    """Lineal en filas/día (por encima del piso, que aplana los volúmenes chicos)."""
    dtypes = {"etapas": ["object"] * 8 + ["int64"] * 16}
    chico = legs_module._estimated_day_footprint_gb(
        _ctx({"etapas": 5_000_000}, dtypes), DIAS, [legs_module._tabla("etapas")]
    )
    grande = legs_module._estimated_day_footprint_gb(
        _ctx({"etapas": 10_000_000}, dtypes), DIAS, [legs_module._tabla("etapas")]
    )
    assert chico > legs_module._MIN_DAY_FOOTPRINT_GB
    assert grande == pytest.approx(chico * 2, rel=0.01)


def test_las_columnas_object_pesan_mas_que_las_numericas(sin_override):
    """El costo real de `etapas`/`gps` está en las columnas TEXT."""
    solo_texto = legs_module._estimated_day_footprint_gb(
        _ctx({"t": 10_000_000}, {"t": ["object"] * 8}), DIAS, [legs_module._tabla("t")]
    )
    solo_numeros = legs_module._estimated_day_footprint_gb(
        _ctx({"t": 10_000_000}, {"t": ["int64"] * 8}), DIAS, [legs_module._tabla("t")]
    )
    assert solo_texto > solo_numeros


def test_el_factor_de_dias_multiplica(sin_override):
    """assign_time_distances lee el gps del día Y la madrugada del siguiente."""
    dtypes = {"gps": ["object"] * 5 + ["int64"] * 11}
    un_dia = legs_module._estimated_day_footprint_gb(
        _ctx({"gps": 4_000_000}, dtypes), DIAS, [legs_module._tabla("gps")]
    )
    dos_dias = legs_module._estimated_day_footprint_gb(
        _ctx({"gps": 4_000_000}, dtypes), DIAS, [legs_module._tabla("gps", dias=2)]
    )
    assert dos_dias == pytest.approx(un_dia * 2, rel=0.01)


def test_hay_un_piso(sin_override):
    got = legs_module._estimated_day_footprint_gb(
        _ctx({"etapas": 10}, {"etapas": ["int64"]}), DIAS, [legs_module._tabla("etapas")]
    )
    assert got == legs_module._MIN_DAY_FOOTPRINT_GB


def test_amba_cae_en_el_rango_medido(sin_override):
    """~9 M etapas/día + ~4 M gps x2 días tiene que dar cerca de los 18-22 GB observados.

    Es la calibración que evita el OOM: con el 12.0 viejo el autotune elegía 2
    workers con ~30 GB de presupuesto; con esto elige 1.
    """
    rows = {"etapas": 9_000_000, "gps": 4_200_000}
    dtypes = {
        "etapas": ["object"] * 8 + ["int64"] * 16,   # esquema real de etapas
        "gps": ["object"] * 5 + ["int64"] * 11,      # esquema real de gps
    }
    got = legs_module._estimated_day_footprint_gb(
        _ctx(rows, dtypes), DIAS, [legs_module._tabla("etapas"), legs_module._tabla("gps", dias=2)]
    )
    assert 16 <= got <= 24, got


def test_una_tabla_que_no_existe_no_rompe(sin_override):
    ctx = MagicMock()
    ctx.data.query.side_effect = Exception("Catalog Error")
    got = legs_module._estimated_day_footprint_gb(ctx, DIAS, [legs_module._tabla("no_existe")])
    assert got == legs_module._MIN_DAY_FOOTPRINT_GB


def test_parallel_day_workers_usa_el_per_day_gb_que_le_pasan(monkeypatch, tmp_path):
    class _Paths:
        base = tmp_path  # sin tuning.yaml -> sin override de parallel_day_workers

    monkeypatch.setattr("urbantrips.utils.paths.get_paths", lambda: _Paths())
    monkeypatch.setattr(legs_module, "_duckdb_memory_limit_gb", lambda: 8.0)

    import psutil

    class _VM:
        available = int((8.0 + 36.0) * 2**30)  # 36 GB de presupuesto

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())

    assert legs_module._parallel_day_workers(7, per_day_gb=12.0) == 3
    assert legs_module._parallel_day_workers(7, per_day_gb=20.0) == 1
    # sin argumento se cae al 12.0 histórico (compatibilidad)
    assert legs_module._parallel_day_workers(7) == 3


def test_el_where_acota_el_conteo(sin_override):
    """La etapa que lee solo `etapa_validada = 1` no debe pagar por las demás."""
    capturado = {}

    def fake_query(sql):
        if "LIMIT 0" in sql:
            return pd.DataFrame({"c": pd.Series(dtype="int64")})
        capturado["sql"] = sql
        return pd.DataFrame({"n": [1_000_000]})

    ctx = MagicMock()
    ctx.data.query.side_effect = fake_query
    legs_module._estimated_day_footprint_gb(
        ctx, DIAS, [legs_module._tabla("etapas", where="etapa_validada = 1")]
    )
    assert "(etapa_validada = 1)" in capturado["sql"]


def test_sin_where_no_se_agrega_filtro(sin_override):
    capturado = {}

    def fake_query(sql):
        if "LIMIT 0" in sql:
            return pd.DataFrame({"c": pd.Series(dtype="int64")})
        capturado["sql"] = sql
        return pd.DataFrame({"n": [1_000_000]})

    ctx = MagicMock()
    ctx.data.query.side_effect = fake_query
    legs_module._estimated_day_footprint_gb(ctx, DIAS, [legs_module._tabla("etapas")])
    assert "AND (" not in capturado["sql"]


def test_las_columnas_pedidas_llegan_al_limit_0(sin_override):
    """Cobrar la tabla entera cuando la etapa lee una proyección sobreestima 1,7x."""
    vistos = []

    def fake_query(sql):
        if "LIMIT 0" in sql:
            vistos.append(sql)
            return pd.DataFrame({"c": pd.Series(dtype="int64")})
        return pd.DataFrame({"n": [1_000_000]})

    ctx = MagicMock()
    ctx.data.query.side_effect = fake_query
    legs_module._estimated_day_footprint_gb(
        ctx, DIAS, [legs_module._tabla("etapas", ["id", "dia", "h3_o"])]
    )
    assert "SELECT id, dia, h3_o FROM etapas LIMIT 0" in vistos[0]


def test_tabla_fija_se_cuenta_entera_y_desde_insumos(sin_override):
    """La matriz de validación no es day-scoped, pero se picklea a cada worker.

    Medida sobre AMBA es 4,25 GB — más que las etapas del día. Ignorarla es lo que
    hacía que el autotune committeara de más en infer_destinations.
    """
    vistos = {"data": [], "insumos": []}

    def hacer(destino):
        def fake_query(sql):
            vistos[destino].append(sql)
            if "LIMIT 0" in sql:
                return pd.DataFrame({"c": pd.Series(dtype="int64")})
            return pd.DataFrame({"n": [29_284_295]})
        return fake_query

    ctx = MagicMock()
    ctx.data.query.side_effect = hacer("data")
    ctx.insumos.query.side_effect = hacer("insumos")

    legs_module._estimated_day_footprint_gb(ctx, DIAS, [
        legs_module._tabla("matriz_validacion", ["a", "b"], dias=0, fuente="insumos"),
    ])
    assert vistos["data"] == [], "una tabla fuente='insumos' no debe tocar ctx.data"
    conteo = [q for q in vistos["insumos"] if "count(*)" in q][0]
    assert "WHERE dia IN" not in conteo, "una tabla fija no se acota por día"


def test_tabla_fija_con_distinct(sin_override):
    capturado = {}

    def fake_query(sql):
        if "LIMIT 0" in sql:
            return pd.DataFrame({"c": pd.Series(dtype="int64")})
        capturado["sql"] = sql
        return pd.DataFrame({"n": [2_466_091]})

    ctx = MagicMock()
    ctx.insumos.query.side_effect = fake_query
    legs_module._estimated_day_footprint_gb(ctx, DIAS, [
        legs_module._tabla("matriz_validacion", ["a", "b"], dias=0,
                           fuente="insumos", distinct=True),
    ])
    assert "SELECT DISTINCT a, b FROM matriz_validacion" in capturado["sql"]


def test_el_main_extra_separa_fijas_de_dia(sin_override):
    """`reserva_main` era solo el memory_limit de DuckDB e ignoraba el pandas del main.

    En infer_destinations eso son la matriz de validación (4,25 GB medidos, se carga
    una vez y vive todo el loop) más el día en vuelo con su buffer de pickle. Es el
    término que faltaba cuando el pool del cliente murió con 2 workers.
    """
    def fake_query(sql):
        if "LIMIT 0" in sql:
            return pd.DataFrame({"c": pd.Series(dtype="int64")})
        return pd.DataFrame({"n": [10_000_000]})

    ctx = MagicMock()
    ctx.data.query.side_effect = fake_query
    ctx.insumos.query.side_effect = fake_query

    dtype_1obj = ["object"]
    solo_dia = legs_module._day_memory_model(
        ctx, DIAS, [legs_module._tabla("etapas", ["a"])]
    )
    solo_fija = legs_module._day_memory_model(
        ctx, DIAS, [legs_module._tabla("mv", ["a"], dias=0, fuente="insumos")]
    )
    # mismo tamaño de insumos -> mismo por_dia
    assert solo_dia.por_dia == pytest.approx(solo_fija.por_dia)
    # pero el día en vuelo se cobra dos veces al main (frame + pickle) y la fija una
    assert solo_dia.main_extra == pytest.approx(2 * solo_fija.main_extra)


def test_main_extra_se_descuenta_del_presupuesto(monkeypatch, tmp_path):
    class _Paths:
        base = tmp_path

    monkeypatch.setattr("urbantrips.utils.paths.get_paths", lambda: _Paths())
    monkeypatch.setattr(legs_module, "_duckdb_memory_limit_gb", lambda: 8.0)

    import psutil

    class _VM:
        available = int((8.0 + 30.0) * 2**30)  # 30 GB antes de descontar el main

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())

    assert legs_module._parallel_day_workers(7, per_day_gb=10.0) == 3
    assert legs_module._parallel_day_workers(7, per_day_gb=10.0, main_extra_gb=10.0) == 2
    assert legs_module._parallel_day_workers(7, per_day_gb=10.0, main_extra_gb=25.0) == 1


def test_el_override_no_suma_reserva_del_main(monkeypatch):
    """Si el operador fuerza parallel_day_gb, se hace cargo: no se le suma nada."""
    monkeypatch.setattr(
        "urbantrips.utils.utils.leer_configs_tuning",
        lambda: {"parallel_day_gb": 20},
    )
    modelo = legs_module._day_memory_model(
        MagicMock(), DIAS, [legs_module._tabla("etapas")]
    )
    assert modelo == (20.0, 0.0)
