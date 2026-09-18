"""matriz_paradas: el camino incremental tiene que dar la MISMA matriz_validacion
que la reconstrucción total, que es la que veníamos usando.

La clave del diseño: los conteos crudos se acumulan (n_trx, n_gps) y el filtro de
outliers se aplica sobre esos acumulados, así que sumar el día nuevo es equivalente
a recontar todo el histórico.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from urbantrips.carto import carto


ETAPAS_COLS = ["id", "id_tarjeta", "dia", "id_viaje", "id_etapa", "modo",
               "id_linea", "id_ramal", "h3_o"]
GPS_COLS = ["id", "dia", "id_linea", "id_ramal", "interno", "h3"]

# 2 líneas del mismo modo (sin validación por ramal) con paradas de distinta densidad.
# La parada "rara" queda cerca del umbral para que el filtro tenga algo que decidir.
H3 = {
    "a": "89c2e3a95c3ffff",
    "b": "89c2e3105d7ffff",
    "c": "89c2e3374dbffff",
    "raro": "89c2e3b5eabffff",
}


def _etapas(dia, filas):
    """filas = [(id_linea, id_ramal, h3_o, cantidad), ...]"""
    out, i = [], 0
    for id_linea, id_ramal, h3, n in filas:
        for _ in range(n):
            i += 1
            out.append({
                "id": hash((dia, id_linea, h3, i)) % 10**9,
                "id_tarjeta": f"t{i}", "dia": dia, "id_viaje": 1, "id_etapa": 1,
                "modo": "autobus", "id_linea": id_linea, "id_ramal": id_ramal,
                "h3_o": h3,
            })
    return pd.DataFrame(out, columns=ETAPAS_COLS)


def _gps(dia, filas):
    out, i = [], 0
    for id_linea, id_ramal, h3, n in filas:
        for _ in range(n):
            i += 1
            out.append({
                "id": hash((dia, id_linea, h3, i, "g")) % 10**9, "dia": dia,
                "id_linea": id_linea, "id_ramal": id_ramal, "interno": 1, "h3": h3,
            })
    return pd.DataFrame(out, columns=GPS_COLS)


DIA1 = "2026-03-01"
DIA2 = "2026-03-02"

ETAPAS = {
    DIA1: _etapas(DIA1, [(1, 10, H3["a"], 40), (1, 10, H3["b"], 30),
                         (2, 20, H3["c"], 25), (2, 20, H3["raro"], 1)]),
    DIA2: _etapas(DIA2, [(1, 10, H3["a"], 35), (1, 10, H3["b"], 28),
                         (2, 20, H3["c"], 22), (2, 20, H3["raro"], 1)]),
}
GPS = {
    DIA1: _gps(DIA1, [(1, 10, H3["a"], 60), (1, 10, H3["b"], 50),
                      (2, 20, H3["c"], 40), (2, 20, H3["raro"], 3)]),
    DIA2: _gps(DIA2, [(1, 10, H3["a"], 55), (1, 10, H3["b"], 45),
                      (2, 20, H3["c"], 38), (2, 20, H3["raro"], 2)]),
}


@pytest.fixture
def ctx_factory(tmp_path, monkeypatch):
    """Crea un ctx con adaptadores DuckDB reales y la config mínima que lee carto."""
    from urbantrips.storage.adapters.duckdb.data import DuckDBDataAdapter
    from urbantrips.storage.adapters.duckdb.insumos import DuckDBInsumoAdapter

    monkeypatch.setattr(
        carto, "leer_configs_generales",
        lambda autogenerado=False: {"frac_mediana_gps": 0.25},
    )
    monkeypatch.setattr(carto, "modos_con_ramal", lambda configs: set())

    creados = []

    def make(nombre):
        data = DuckDBDataAdapter(tmp_path / f"{nombre}_data.duckdb")
        insumos = DuckDBInsumoAdapter(tmp_path / f"{nombre}_insumos.duckdb")
        insumos.save_metadata_lineas(pd.DataFrame({
            "id_linea": [1, 2], "nombre_linea": ["L1", "L2"],
            "id_linea_agg": [1, 2], "nombre_linea_agg": ["L1", "L2"],
            "modo": ["autobus", "autobus"], "empresa": [None, None],
            "descripcion": [None, None],
        }))
        ctx = SimpleNamespace(data=data, insumos=insumos)
        creados.append(ctx)
        return ctx

    yield make


def _cargar(ctx, dias):
    for dia in dias:
        ctx.data.append_raw(ETAPAS[dia], "etapas")
        ctx.data.append_raw(GPS[dia], "gps")
    ctx.data.save_run_days(pd.DataFrame({"dia": list(dias)}))


def _matriz(ctx):
    m = ctx.insumos.get_matrix_validation()
    return m.sort_values(list(m.columns)).reset_index(drop=True)


def test_incremental_da_la_misma_matriz_que_la_reconstruccion_total(ctx_factory):
    # Camino A (referencia): los 2 días de una, reconstrucción total.
    total = ctx_factory("total")
    _cargar(total, [DIA1, DIA2])
    carto.update_stations_catchment_area(2, total)

    # Camino B: día 1 y después el día 2 en incremental.
    inc = ctx_factory("inc")
    _cargar(inc, [DIA1])
    carto.update_stations_catchment_area(2, inc)
    _cargar(inc, [DIA2])
    carto.update_stations_catchment_area(2, inc)

    pd.testing.assert_frame_equal(_matriz(total), _matriz(inc))


def test_el_dia_2_se_lee_solo_a_si_mismo(ctx_factory, monkeypatch):
    """La corrida incremental no debe escanear los días ya incorporados."""
    inc = ctx_factory("scan")
    _cargar(inc, [DIA1])
    carto.update_stations_catchment_area(2, inc)

    vistos = []
    original = carto._conteos_paradas_crudos

    def espia(ctx, dias=None):
        vistos.append(dias)
        return original(ctx, dias)

    monkeypatch.setattr(carto, "_conteos_paradas_crudos", espia)
    _cargar(inc, [DIA2])
    carto.update_stations_catchment_area(2, inc)

    assert vistos == [[DIA2]], f"debía leer solo el día nuevo, leyó {vistos}"


def test_reproceso_no_duplica_conteos(ctx_factory):
    """Re-correr un día YA incorporado no puede sumar sus conteos dos veces."""
    ctx = ctx_factory("reproc")
    _cargar(ctx, [DIA1, DIA2])
    carto.update_stations_catchment_area(2, ctx)
    antes = ctx.insumos.get_matriz_paradas()

    # mismo run_days, sin datos nuevos: cae al camino de reconstrucción total
    ctx.data.save_run_days(pd.DataFrame({"dia": [DIA2]}))
    carto.update_stations_catchment_area(2, ctx)
    despues = ctx.insumos.get_matriz_paradas()

    key = ["id_linea", "id_ramal", "parada"]
    a = antes.sort_values(key).reset_index(drop=True)
    b = despues.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_guarda_las_descartadas_con_valido_0_y_su_evidencia(ctx_factory):
    ctx = ctx_factory("valido")
    _cargar(ctx, [DIA1])
    carto.update_stations_catchment_area(2, ctx)

    mp = ctx.insumos.get_matriz_paradas()
    # todas las candidatas están, válidas y descartadas
    assert set(mp["parada"]) == set(H3.values())
    assert set(mp["valido"].unique()) <= {0, 1}
    # los conteos crudos se conservan para las descartadas
    raro = mp[mp["parada"] == H3["raro"]].iloc[0]
    assert raro["n_trx"] == 1 and raro["n_gps"] == 3
    # y la matriz de validación solo contiene válidas
    validas = set(mp.loc[mp["valido"] == 1, "parada"])
    assert set(ctx.insumos.get_matrix_validation()["parada"]) <= validas


def test_una_parada_descartada_puede_volver_a_calificar(ctx_factory):
    """Como no se borra nada, si acumula evidencia vuelve sola a valido=1."""
    ctx = ctx_factory("recupera")
    _cargar(ctx, [DIA1])
    carto.update_stations_catchment_area(2, ctx)
    mp1 = ctx.insumos.get_matriz_paradas()
    raro1 = mp1[mp1["parada"] == H3["raro"]].iloc[0]

    # un día que aporta muchos puntos justo a esa parada
    dia3 = "2026-03-03"
    ETAPAS[dia3] = _etapas(dia3, [(2, 20, H3["raro"], 30), (2, 20, H3["c"], 20)])
    GPS[dia3] = _gps(dia3, [(2, 20, H3["raro"], 50), (2, 20, H3["c"], 30)])
    try:
        _cargar(ctx, [dia3])
        carto.update_stations_catchment_area(2, ctx)
    finally:
        ETAPAS.pop(dia3, None)
        GPS.pop(dia3, None)

    mp2 = ctx.insumos.get_matriz_paradas()
    raro2 = mp2[mp2["parada"] == H3["raro"]].iloc[0]
    assert raro2["n_trx"] > raro1["n_trx"]  # la evidencia se acumuló
    assert raro2["valido"] == 1
