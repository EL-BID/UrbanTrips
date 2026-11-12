"""Capacidad del vehiculo por modo para el factor de ocupacion.

El factor de ocupacion dividia por una capacidad fija de 60 lugares, que no
vale para una formacion de tranvia o tren. Ahora sale de `capacidad_modos` del
config, segun el modo de cada linea; sin capacidad, queda sin calcular.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from urbantrips.kpi import kpi


def _ctx():
    lineas = pd.DataFrame(
        {"id_linea": [1, 100, 500, 900], "modo": ["autobus", "tranvia", "cable", "otros"]}
    )
    return SimpleNamespace(insumos=SimpleNamespace(get_metadata_lineas=lambda: lineas))


def _config(monkeypatch, capacidad_modos):
    cfg = {} if capacidad_modos is None else {"capacidad_modos": capacidad_modos}
    monkeypatch.setattr(kpi, "leer_configs_generales", lambda autogenerado=False: cfg)


def test_capacidad_por_linea_segun_el_modo(monkeypatch):
    _config(monkeypatch, {"autobus": 60, "tranvia": "200", "cable": 0, "otros": None})
    assert kpi.capacidad_por_linea(_ctx()) == {1: 60.0, 100: 200.0}


def test_sin_capacidad_modos_en_el_config_avisa_y_no_calcula(monkeypatch, caplog):
    _config(monkeypatch, None)
    with caplog.at_level("WARNING"):
        assert kpi.capacidad_por_linea(_ctx()) == {}
    assert "capacidad_modos" in caplog.text


@pytest.fixture
def ratios():
    """Colectivo (1), tranvia (100) y cable sin capacidad (500), con la misma
    demanda y la misma oferta."""
    day_stats = pd.DataFrame(
        {
            "id_linea": [1, 100, 500],
            "dia": "2025-10-29",
            "tot_veh": [10, 10, 10],
            "tot_km_route": [1000.0, 1000.0, 1000.0],
            "tot_km_route_gps": [1000.0, 0.0, 1000.0],
            "tot_pax": [3000.0, 3000.0, 3000.0],
            "dmt_mean_od": 5.0, "dmt_mean_route": 6.0, "dmt_mean_route_gps": 6.0,
            "dmt_median_od": 4.0, "dmt_median_route": 5.0, "dmt_median_route_gps": 5.0,
        }
    )
    out = kpi._kpi_ratios_y_orden(day_stats, ["id_linea"], {1: 60.0, 100: 200.0})
    return out.set_index("id_linea")


def test_factor_de_ocupacion_usa_la_capacidad_de_la_linea(ratios):
    # colectivo: 3000 pax * 6 km / (1000 km * 60 lugares)
    assert ratios.loc[1, "fo_mean_route"] == pytest.approx(0.3)
    # tranvia: lo mismo con 200 lugares
    assert ratios.loc[100, "fo_mean_route"] == pytest.approx(0.09)
    assert ratios.loc[100, "fo_mean_od"] == pytest.approx(0.075)
    # sin km por odometro no hay factor por odometro
    assert np.isnan(ratios.loc[100, "fo_mean_route_gps"])


def test_linea_sin_capacidad_queda_sin_factor_de_ocupacion(ratios):
    cable = ratios.loc[500]
    assert cable[[c for c in ratios.columns if c.startswith("fo_")]].isna().all()
    # los indicadores que no dependen de la capacidad se calculan igual
    assert cable["ipk_route"] == pytest.approx(3.0)
    assert cable["pvd"] == pytest.approx(300.0)


def test_sin_ninguna_capacidad_el_factor_queda_nulo_y_numerico():
    cap = kpi._capacidad_de(pd.Series([1, 100]), {})
    assert cap.isna().all() and cap.dtype == "float64"


def test_avisa_las_lineas_sin_capacidad(monkeypatch, caplog):
    # cable en 0, otros vacio y un valor mal escrito para el tranvia
    _config(monkeypatch, {"autobus": 60, "tranvia": "200 pax", "cable": 0, "otros": None})
    with caplog.at_level("INFO"):
        assert kpi.capacidad_por_linea(_ctx()) == {1: 60.0}
    assert "autobus 60" in caplog.text
    assert "3 lineas sin capacidad" in caplog.text
    for modo in ("tranvia", "cable", "otros"):
        assert modo in caplog.text


def test_con_todas_las_lineas_con_capacidad_no_avisa(monkeypatch, caplog):
    _config(monkeypatch, {"autobus": 60, "tranvia": 200, "cable": 30, "otros": 40})
    with caplog.at_level("WARNING"):
        kpi.capacidad_por_linea(_ctx())
    assert caplog.text == ""


def test_nota_del_dashboard_segun_los_modos_seleccionados():
    from urbantrips.kpi.agregacion_indicadores import nota_capacidad_modos

    capacidad_modos = {"autobus": 60, "tranvia": 200, "cable": 0, "otros": None}
    todo = nota_capacidad_modos(["autobus", "tranvia", "autobus", None], capacidad_modos)
    assert "autobus 60 · tranvia 200" in todo and "Sin capacidad" not in todo
    assert "tranvia" not in nota_capacidad_modos(["autobus"], capacidad_modos)
    con_cable = nota_capacidad_modos(["autobus", "cable", "tren"], capacidad_modos)
    assert "autobus 60" in con_cable and "Sin capacidad cargada para: cable, tren" in con_cable
    assert "no tiene `capacidad_modos`" in nota_capacidad_modos(["autobus"], None)
