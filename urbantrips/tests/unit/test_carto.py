import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from urbantrips.carto.carto import _normalize_zone_ids, _with_wkt_geometry


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
