"""Contrato de eliminar_trx_fuera_bbox (única llave de borrado geográfico).

Vive fuera de test_transactions.py a propósito: ese archivo está en el
collect_ignore_glob del conftest (import pesado de osmnx/pandana vía el módulo
transactions). Acá usamos importorskip para que el test CORRA en entornos con el
stack completo (p.ej. trips6) y SKIPEE limpio donde falten las deps, sin quedar
excluido de la suite estándar.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock

# Si el stack pesado no está, se skipea el módulo entero (no rompe la colección).
tx = pytest.importorskip("urbantrips.datamodel.transactions")


def _bbox_ctx():
    """ctx mínimo sin zonificaciones → eliminar_trx_fuera_bbox cae al bbox del config."""
    ctx = MagicMock()
    ctx.insumos.get_zones.return_value = pd.DataFrame()  # len 0 → usa config
    return ctx


def test_eliminar_trx_fuera_bbox_conserva_latlon_cero_con_fex_cero(monkeypatch):
    """Contrato:
    - fuera-de-bbox con coords reales → SE ELIMINA;
    - lat/lon == 0 → SE CONSERVA pero con factor_expansion = 0 (inválida);
    - válida dentro del bbox → intacta;
    - geo_valido no persiste.
    """
    bbox = {"minx": -59.0, "miny": -35.0, "maxx": -58.0, "maxy": -34.0}
    monkeypatch.setattr(
        tx, "leer_configs_generales", lambda *a, **k: {"filtro_latlong_bbox": bbox}
    )
    monkeypatch.setattr(tx, "agrego_indicador", lambda *a, **k: None)

    trx = pd.DataFrame({
        "id_tarjeta": ["A", "B", "C"],
        "latitud": [-34.6, 0.0, -10.0],     # dentro | lat/lon=0 | fuera-de-bbox
        "longitud": [-58.4, 0.0, -10.0],
        "factor_expansion": [1.0, 5.0, 3.0],
    })

    out = tx.eliminar_trx_fuera_bbox(trx.copy(), ctx=_bbox_ctx())

    # C (fuera-de-bbox real) se elimina; A (válida) y B (lat/lon=0) se conservan
    assert set(out["id_tarjeta"]) == {"A", "B"}
    # A mantiene su fex; B queda inválida con fex=0
    assert out.loc[out["id_tarjeta"] == "A", "factor_expansion"].iloc[0] == 1.0
    assert out.loc[out["id_tarjeta"] == "B", "factor_expansion"].iloc[0] == 0.0
    # geo_valido es un flag transitorio, no debe persistir
    assert "geo_valido" not in out.columns


def test_eliminar_trx_fuera_bbox_sin_bbox_ni_zonificaciones_aborta(monkeypatch):
    """Sin bbox ni zonificaciones la llave no puede filtrar → aborta (fail-closed),
    en vez del viejo fail-open silencioso que dejaba pasar todo sin filtrar."""
    monkeypatch.setattr(tx, "leer_configs_generales", lambda *a, **k: {})  # sin bbox

    trx = pd.DataFrame({
        "id_tarjeta": ["A"],
        "latitud": [-34.6],
        "longitud": [-58.4],
        "factor_expansion": [1.0],
    })

    with pytest.raises(ValueError):
        tx.eliminar_trx_fuera_bbox(trx, ctx=_bbox_ctx())
