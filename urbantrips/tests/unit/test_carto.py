import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from urbantrips.carto.carto import (
    _normalize_zone_ids,
    _to_wgs84,
    _with_wkt_geometry,
)


def test_to_wgs84_reproyecta_una_capa_en_metros_bien_declarada():
    # Villa Maria en Gauss-Kruger faja 4 (POSGAR 2007)
    zonas = gpd.GeoDataFrame(
        {"id": ["a"]},
        geometry=[box(4471605, 6410762, 4482864, 6417755)],
        crs="EPSG:5346",
    )

    result = _to_wgs84(zonas, "barrios.geojson")

    assert result.crs.to_epsg() == 4326
    minx, miny, maxx, maxy = result.total_bounds
    assert -63.4 < minx < maxx < -63.1
    assert -32.5 < miny < maxy < -32.3


def test_to_wgs84_deja_intacta_una_capa_que_ya_esta_en_4326():
    zonas = gpd.GeoDataFrame(
        {"id": ["a"]}, geometry=[box(-63.3, -32.4, -63.2, -32.3)], crs="EPSG:4326"
    )

    result = _to_wgs84(zonas, "barrios.geojson")

    assert result is zonas


def test_to_wgs84_corta_con_mensaje_si_declara_grados_pero_trae_metros():
    # El caso real: GeoJSON exportado sin reproyectar, etiquetado CRS84.
    # Sin este corte, to_crs da inf y buffer() revienta GEOS sin traceback.
    zonas = gpd.GeoDataFrame(
        {"id": ["a"]},
        geometry=[box(4471605, 6410762, 4482864, 6417755)],
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="barrios.geojson.*no son grados"):
        _to_wgs84(zonas, "barrios.geojson")


def test_to_wgs84_corta_con_mensaje_si_la_capa_no_tiene_crs():
    zonas = gpd.GeoDataFrame({"id": ["a"]}, geometry=[box(0, 0, 1, 1)], crs=None)

    with pytest.raises(ValueError, match="barrios.geojson.*no declara"):
        _to_wgs84(zonas, "barrios.geojson")


def test_normalize_zone_ids_preserves_text_and_cleans_integer_floats():
    result = _normalize_zone_ids(pd.Series([123.0, "045", "zona-a", 7]))

    assert result.tolist() == ["123", "45", "zona-a", "7"]


def test_with_wkt_geometry_returns_plain_dataframe_without_geopandas_warning():
    zones = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(1, 2)],
        crs="EPSG:4326",
    )

    result = _with_wkt_geometry(zones)

    assert isinstance(result, pd.DataFrame)
    assert not isinstance(result, gpd.GeoDataFrame)
    assert result.loc[0, "geometry"] == "POINT (1 2)"


def test_with_wkt_geometry_preserves_existing_wkt_strings():
    poly = pd.DataFrame(
        {
            "id": ["existing", "new"],
            "geometry": ["POINT (0 0)", Point(1, 2)],
        }
    )

    result = _with_wkt_geometry(poly)

    assert result["geometry"].tolist() == ["POINT (0 0)", "POINT (1 2)"]
