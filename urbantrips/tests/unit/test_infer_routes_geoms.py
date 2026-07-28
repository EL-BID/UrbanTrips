"""Incremental de infer_routes_geoms: no recalcular líneas ya inferidas.

importorskip: routes.py arrastra el stack pesado (geo/statsmodels lowess) al importar,
así que el módulo se skipea limpio donde esas deps no estén.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock

routes = pytest.importorskip("urbantrips.carto.routes")


def _existing(lineas):
    rows = []
    for linea in lineas:
        for d in (0, 1):
            rows.append(
                {"id_linea": linea, "direction": d, "wkt": "LINESTRING(0 0,1 1)"}
            )
    return pd.DataFrame(rows)


def test_infer_routes_geoms_skips_existing_lines():
    """Si todas las líneas ya tienen geometría inferida, la query de etapas filtra
    por id_linea NOT IN (existentes) y, al no quedar líneas nuevas, saltea sin
    recalcular ni reescribir (lowess es caro)."""
    ctx = MagicMock()
    ctx.insumos.get_raw.return_value = _existing([10, 20])
    captured = {}

    def fake_query(q):
        captured["q"] = q
        return pd.DataFrame(columns=["id_linea", "longitud", "latitud"])

    ctx.data.query.side_effect = fake_query

    routes.infer_routes_geoms(ctx)

    q = captured["q"].lower()
    assert "not in" in q, "no acotó la lectura de etapas a las líneas nuevas"
    assert "10" in q and "20" in q
    ctx.insumos.save_raw.assert_not_called()  # nada nuevo → no reescribe


def test_infer_routes_geoms_fresh_computes_all():
    """Corrida fresca (sin inferidas previas): la query NO filtra líneas."""
    ctx = MagicMock()
    ctx.insumos.get_raw.return_value = pd.DataFrame()
    captured = {}

    def fake_query(q):
        captured["q"] = q
        return pd.DataFrame(columns=["id_linea", "longitud", "latitud"])

    ctx.data.query.side_effect = fake_query

    routes.infer_routes_geoms(ctx)

    assert "not in" not in captured["q"].lower()


# ── lowess best-effort: líneas no-inferibles (lowess devuelve None) se saltean ──

def _etapas_dos_lineas():
    # dos líneas con puntos; el contenido no importa, lowess_linea está mockeado
    return pd.DataFrame({
        "id_linea": [100, 100, 100, 200, 200, 200],
        "longitud": [-58.1, -58.2, -58.3, -58.4, -58.5, -58.6],
        "latitud": [-34.1, -34.2, -34.3, -34.4, -34.5, -34.6],
    })


def _geom_valida(monkeypatch):
    import geopandas as gpd
    from shapely import LineString

    def fake(grupo):
        # línea 100 no se puede inferir (None); línea 200 sí
        if grupo["id_linea"].iloc[0] == 100:
            return None
        return gpd.GeoDataFrame(
            {"geometry": [LineString([(-58.4, -34.4), (-58.6, -34.6)])]},
            geometry="geometry", crs=4326,
        )
    monkeypatch.setattr(routes.geo, "lowess_linea", fake)


def test_infer_routes_salta_linea_no_inferible(monkeypatch):
    """Una línea cuyo lowess devuelve None se saltea; la otra se infiere y guarda.
    Antes esto crasheaba con AttributeError: 'DataFrame' has no attribute 'geometry'."""
    ctx = MagicMock()
    ctx.insumos.get_raw.return_value = pd.DataFrame()
    ctx.data.query.return_value = _etapas_dos_lineas()
    _geom_valida(monkeypatch)

    routes.infer_routes_geoms(ctx)  # no debe romper

    ctx.insumos.save_raw.assert_called_once()
    guardado = ctx.insumos.save_raw.call_args[0][0]
    # solo la línea 200, en sus dos direcciones (0 y 1)
    assert set(guardado["id_linea"]) == {200}
    assert set(guardado["direction"]) == {0, 1}


def test_infer_routes_todas_no_inferibles_conserva_existentes(monkeypatch):
    """Si NINGUNA línea nueva infiere, no reescribe (conserva las existentes)."""
    ctx = MagicMock()
    ctx.insumos.get_raw.return_value = _existing([10])
    ctx.data.query.return_value = _etapas_dos_lineas()
    monkeypatch.setattr(routes.geo, "lowess_linea", lambda grupo: None)

    routes.infer_routes_geoms(ctx)  # no rompe

    ctx.insumos.save_raw.assert_not_called()
