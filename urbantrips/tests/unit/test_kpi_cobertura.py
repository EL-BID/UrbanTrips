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


def test_las_etapas_traen_si_estan_en_un_servicio_valido(ctx):
    en_servicio = ctx.data.query(
        f"select e.id, {cobertura.EN_SERVICIO_VALIDO} en_servicio from etapas e"
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
