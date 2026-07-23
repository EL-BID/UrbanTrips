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
