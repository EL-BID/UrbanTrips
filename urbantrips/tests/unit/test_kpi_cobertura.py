"""Cobertura GPS, universo comun de los cocientes y umbral (`kpi/cobertura.py`).

Una etapa dentro de un servicio GPS valido de su vehiculo esta "cubierta". Los
cocientes de oferta usan solo los pasajeros cubiertos, y con cobertura menor al
umbral los indicadores de oferta de la linea en ese dia quedan sin calcular.
"""

from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from urbantrips.kpi import cobertura, kpi
from urbantrips.kpi.agregacion_indicadores import (
    COBERTURA_POR_ETAPAS,
    agregar_sistema,
    nota_cobertura,
    preparar,
)

DIA = "2025-10-29"


def _epoch(hora):
    return int(pd.Timestamp(f"{DIA} {hora}").timestamp())


@pytest.fixture
def ctx():
    """Base minima: la linea 100 (tranvia) con una formacion de dos equipos y un
    vehiculo sin GPS; la linea 200 sin horas en las etapas."""
    con = duckdb.connect()
    con.execute(
        "create table services (dia text, id_linea bigint, id_ramal bigint, interno int,"
        " service_id int, min_ts int, max_ts int, valid int)"
    )
    con.execute(
        "create table gps_vehiculos (dia text, id_linea bigint, interno int,"
        " interno_vehiculo int, es_principal boolean)"
    )
    con.execute(
        "create table etapas (id int, dia text, id_linea bigint, id_ramal bigint,"
        " interno int, tiempo text)"
    )
    # el interno 1 es el principal de la formacion y tiene un servicio valido de
    # 8 a 10; su servicio de la tarde no es valido
    con.execute(
        "insert into services values (?, 100, 100, 1, 1, ?, ?, 1), (?, 100, 100, 1, 2, ?, ?, 0)",
        [DIA, _epoch("08:00:00"), _epoch("10:00:00"), DIA, _epoch("15:00:00"), _epoch("17:00:00")],
    )
    con.execute(
        "insert into gps_vehiculos values (?, 100, 1, 1, true), (?, 100, 2, 1, false)",
        [DIA, DIA],
    )
    etapas = [
        (1, 100, 1, "08:30:00"),   # en el servicio valido
        (2, 100, 2, "09:00:00"),   # equipo secundario: va al servicio del principal
        (3, 100, 1, "12:00:00"),   # fuera de todo servicio
        (4, 100, 1, "16:00:00"),   # dentro de un servicio no valido
        (5, 100, 9, "08:30:00"),   # vehiculo sin GPS
        (6, 100, 1, None),         # sin hora: no cuenta
        (7, 200, 5, None),
        (8, 200, 5, None),
    ]
    for id_, linea, interno, tiempo in etapas:
        con.execute(
            "insert into etapas values (?, ?, ?, ?, ?, ?)",
            [id_, DIA, linea, linea, interno, tiempo],
        )
    return SimpleNamespace(data=SimpleNamespace(query=lambda q: con.execute(q).df()))


def test_cobertura_es_la_parte_de_las_etapas_dentro_de_un_servicio_valido(ctx):
    cob = cobertura.leer_cobertura_gps(ctx, f"'{DIA}'", ["id_linea"]).set_index("id_linea")
    # linea 100: 2 de las 5 etapas con hora caen en el servicio valido
    assert cob.loc[100, "cobertura_gps"] == pytest.approx(0.4)
    # linea 200: ninguna etapa tiene hora, la cobertura no se puede medir
    assert np.isnan(cob.loc[200, "cobertura_gps"])


def test_cobertura_por_ramal(ctx):
    cob = cobertura.leer_cobertura_gps(ctx, f"'{DIA}'", ["id_linea", "id_ramal"])
    assert set(cob.columns) == {"dia", "id_linea", "id_ramal", "cobertura_gps"}
    assert len(cob) == 2


@pytest.mark.parametrize("filtro", ["", f"AND s.dia = '{DIA}'"])
def test_las_etapas_traen_si_estan_en_un_servicio_valido(ctx, filtro):
    en_servicio = ctx.data.query(
        f"select e.id, {cobertura.en_servicio_valido(filtro)} en_servicio from etapas e"
        f" {cobertura.JOIN_VEHICULO} order by e.id"
    )["en_servicio"].tolist()
    assert en_servicio == [True, True, False, False, False, False, False, False]


def _day_stats(cobertura_gps, pax_servicio):
    return pd.DataFrame(
        {
            "id_linea": [1], "dia": DIA, "tot_veh": [10],
            "tot_km_route": [1000.0], "tot_km_route_gps": [0.0],
            "tot_pax": [3000.0], "tot_pax_servicio": [pax_servicio],
            "cobertura_gps": [cobertura_gps],
            "dmt_mean_od": 5.0, "dmt_mean_route": 6.0, "dmt_mean_route_gps": 6.0,
            "dmt_median_od": 4.0, "dmt_median_route": 5.0, "dmt_median_route_gps": 5.0,
        }
    )


def _ratios(cobertura_gps, pax_servicio):
    out = kpi._kpi_ratios_y_orden(_day_stats(cobertura_gps, pax_servicio), ["id_linea"], {1: 60.0})
    return out.iloc[0]


def test_los_cocientes_usan_los_pasajeros_dentro_de_servicios_validos():
    fila = _ratios(0.8, 2400.0)
    # 2400 pasajeros cubiertos sobre 1000 km y 10 vehiculos de esos servicios
    assert fila["ipk_route"] == pytest.approx(2.4)
    assert fila["pvd"] == pytest.approx(240.0)
    assert fila["fo_mean_route"] == pytest.approx(2400 * 6 / (1000 * 60))
    # la demanda total y los km observados no cambian
    assert fila["tot_pax"] == 3000
    assert fila["tot_km_route"] == pytest.approx(1000.0)
    assert fila["tot_km_route_estimado"] == pytest.approx(1250.0)
    assert fila["cobertura_gps"] == pytest.approx(0.8)


def test_con_cobertura_baja_la_oferta_queda_sin_calcular():
    fila = _ratios(0.4, 1200.0)
    for col in ("tot_veh", "tot_km_route", "tot_km_route_estimado", "ipk_route", "pvd",
                "kvd_route", "fo_mean_route", "fo_mean_od"):
        assert np.isnan(fila[col]), col
    # la demanda y la propia cobertura se mantienen
    assert fila["tot_pax"] == 3000
    assert fila["dmt_mean_od"] == pytest.approx(5.0)
    assert fila["cobertura_gps"] == pytest.approx(0.4)


def test_justo_en_el_umbral_se_calcula():
    assert _ratios(0.5, 1500.0)["ipk_route"] == pytest.approx(1.5)


def test_sin_cobertura_medida_se_usan_todos_los_pasajeros():
    # etapas sin hora: no hay forma de saber cuales estan cubiertas
    fila = _ratios(np.nan, 0.0)
    assert fila["ipk_route"] == pytest.approx(3.0)
    assert np.isnan(fila["tot_km_route_estimado"])
    assert fila["tot_km_route"] == pytest.approx(1000.0)


def test_el_umbral_es_configurable(monkeypatch):
    monkeypatch.setattr(cobertura, "cobertura_gps_minima", lambda: 0.9)
    assert np.isnan(_ratios(0.8, 2400.0)["ipk_route"])


def test_vaciar_oferta_no_toca_filas_sin_cobertura_medida():
    df = pd.DataFrame(
        {"cobertura_gps": [0.2, np.nan, 0.9], "tot_km_route": [10.0, 20.0, 30.0],
         "transacciones": [5, 6, 7]}
    )
    out = cobertura.vaciar_oferta_sin_cobertura(df, minimo=0.5)
    assert np.isnan(out["tot_km_route"].iloc[0])
    assert out["tot_km_route"].tolist()[1:] == [20.0, 30.0]
    assert out["transacciones"].tolist() == [5, 6, 7]
    assert df["tot_km_route"].tolist() == [10.0, 20.0, 30.0]


def _kpis(**extra):
    base = {
        "dia": [DIA, DIA], "id_linea": ["1", "2"], "modo": "autobus",
        "transacciones": [1000, 3000], "vehiculos_operativos": [5.0, 10.0],
        "cant_internos_en_trx": [10.0, 10.0], "tot_km_route": [100.0, 300.0],
    }
    base.update(extra)
    return pd.DataFrame(base)


def test_dashboard_agrega_la_cobertura_ponderando_por_transacciones():
    df = preparar(_kpis(cobertura_gps=[0.6, 1.0], tot_km_route_estimado=[166.7, 300.0]))
    assert df[COBERTURA_POR_ETAPAS].all()
    agg = agregar_sistema(df, promedio_diario=False)
    assert agg["cobertura_gps"].iloc[0] == pytest.approx((0.6 * 1000 + 1.0 * 3000) / 4000)
    assert agg["tot_km_route_estimado"].iloc[0] == pytest.approx(466.7)


def test_dashboard_con_una_base_anterior_usa_la_medida_por_vehiculos():
    df = preparar(_kpis())
    assert not df[COBERTURA_POR_ETAPAS].any()
    assert df["cobertura_gps"].tolist() == [0.5, 1.0]
    agg = agregar_sistema(df, promedio_diario=False)
    # cociente de sumas: 15 vehiculos con servicio sobre 20 internos
    assert agg["cobertura_gps"].iloc[0] == pytest.approx(0.75)


def test_nota_del_dashboard_avisa_las_lineas_bajo_el_umbral():
    df = preparar(
        _kpis(
            cobertura_gps=[0.3, 1.0], tot_km_route=[np.nan, 300.0],
            tot_km_route_estimado=[np.nan, 300.0],
        )
    )
    nota = nota_cobertura(df)
    assert "cobertura GPS" in nota and "1 línea(s)-día" in nota
    # una linea con km no esta vacia aunque su cobertura sea baja: el umbral con
    # el que se calculo pudo ser otro
    con_km = preparar(_kpis(cobertura_gps=[0.3, 1.0], tot_km_route_estimado=[333.0, 300.0]))
    assert "línea(s)-día" not in nota_cobertura(con_km)
    # una base anterior no trae la cobertura por etapas: no hay nota
    assert nota_cobertura(preparar(_kpis())) == ""


def test_vaciar_oferta_devuelve_el_mismo_frame_si_no_hay_nada_que_vaciar():
    sin_columna = pd.DataFrame({"tot_veh": [1]})
    assert cobertura.vaciar_oferta_sin_cobertura(sin_columna, 0.5) is sin_columna
    bien_cubierta = pd.DataFrame({"cobertura_gps": [0.9], "tot_veh": [3]})
    out = cobertura.vaciar_oferta_sin_cobertura(bien_cubierta, 0.5)
    assert out is bien_cubierta and out["tot_veh"].dtype.kind == "i"


def test_universo_de_oferta_no_modifica_el_frame():
    df = _day_stats(0.8, 2400.0)
    columnas = list(df.columns)
    pax, km_estimados = cobertura.universo_de_oferta(df)
    assert list(df.columns) == columnas
    assert pax.tolist() == [2400.0] and km_estimados.tolist() == [1250.0]


def test_la_cobertura_no_depende_de_la_division_entera(ctx):
    # con division entera en la conexion, 2 / 5 daria 0 y vaciaria toda la oferta
    consulta = ctx.data.query
    ctx.data.query = lambda q: consulta("SET integer_division = true; " + q)
    cob = cobertura.leer_cobertura_gps(ctx, f"'{DIA}'", ["id_linea"]).set_index("id_linea")
    assert cob.loc[100, "cobertura_gps"] == pytest.approx(0.4)


def test_vaciar_oferta_sigue_las_filas_dadas_y_no_el_umbral():
    df = pd.DataFrame(
        {"cobertura_gps": [0.9, 0.2], "tot_veh": [3.0, 4.0], "tot_pax": [10, 20]}
    )
    # por umbral se vacia la segunda; con `filas` se vacia la que se indique
    por_filas = cobertura.vaciar_oferta_sin_cobertura(
        df, filas=pd.Series([True, False])
    )
    assert por_filas["tot_veh"].isna().tolist() == [True, False]
    assert por_filas["tot_pax"].tolist() == [10, 20]
    sin_filas = cobertura.vaciar_oferta_sin_cobertura(df, filas=pd.Series([False, False]))
    assert sin_filas is df


def _diario(**columnas):
    """Indicadores diarios de kpi_by_day_line: todo vacio salvo lo que se dé."""
    n = len(columnas["dia"])
    base = {
        c: [np.nan] * n for c in (
            "tot_veh", "tot_km_route", "tot_km_route_gps", "tot_pax_servicio",
            "cobertura_gps", "dmt_mean_od", "dmt_mean_route", "dmt_mean_route_gps",
            "dmt_median_od", "dmt_median_route", "dmt_median_route_gps",
        )
    }
    base.update(columnas)
    return pd.DataFrame(base)


def _ctx_con_dias(diario):
    """Contexto falso: `query` devuelve los indicadores diarios y `append_raw`
    guarda lo que se sube."""
    subido = {}
    data = SimpleNamespace(
        execute=lambda q: None,
        query=lambda q: diario.copy(),
        append_raw=lambda df, tabla: subido.update(df=df.copy(), tabla=tabla),
    )
    return SimpleNamespace(data=data), subido


def test_promedio_por_tipo_de_dia_recalcula_los_cocientes_con_un_solo_universo():
    # una linea: lunes y martes con oferta; el miercoles quedo sin oferta
    # (cobertura bajo el umbral) pero con 5000 pasajeros
    diario = _diario(
        id_linea=[1, 1, 1], dia=["2024-01-01", "2024-01-02", "2024-01-03"],
        tot_veh=[10.0, 10.0, np.nan], tot_km_route=[1000.0, 1000.0, np.nan],
        tot_pax=[4000, 2000, 5000], tot_pax_servicio=[4000.0, 2000.0, 1000.0],
        cobertura_gps=[1.0, 1.0, 0.2],
        dmt_mean_od=[4.0, 4.0, 9.0], dmt_mean_route=[5.0, 5.0, 9.0],
        dmt_median_od=[4.0, 4.0, 9.0], dmt_median_route=[5.0, 5.0, 9.0],
    )
    ctx, subido = _ctx_con_dias(diario)
    out = kpi.compute_kpi_by_line_typeday(ctx, capacidades={1: 60.0})
    fila = out.set_index("dia").loc["weekday"]
    assert subido["tabla"] == "kpi_by_day_line"
    # el miercoles sin oferta no entra ni en la demanda ni en la oferta
    assert fila["tot_pax"] == 3000 and fila["tot_km_route"] == 1000.0
    assert fila["ipk_route"] == pytest.approx(3.0)
    assert fila["pvd"] == pytest.approx(300.0)
    assert fila["cobertura_gps"] == pytest.approx(1.0)
    # el factor de ocupacion sale de los mismos totales: 3000 * 5 km / (1000 * 60)
    assert fila["fo_mean_route"] == pytest.approx(0.25)


def test_promedio_por_tipo_de_dia_de_una_linea_que_nunca_tiene_oferta():
    diario = _diario(
        id_linea=[2, 2], dia=["2024-01-01", "2024-01-02"], tot_pax=[100, 300],
        tot_pax_servicio=[0.0, 0.0], cobertura_gps=[0.0, 0.0],
        dmt_mean_od=[2.0, 4.0], dmt_median_od=[2.0, 4.0],
    )
    ctx, _ = _ctx_con_dias(diario)
    fila = kpi.compute_kpi_by_line_typeday(ctx, capacidades={}).iloc[0]
    # sin oferta: promedia la demanda de todos sus dias, ponderando la distancia
    assert fila["tot_pax"] == 200
    assert fila["dmt_mean_od"] == pytest.approx((2 * 100 + 4 * 300) / 400)
    assert pd.isna(fila["ipk_route"]) and pd.isna(fila["fo_mean_route"])
