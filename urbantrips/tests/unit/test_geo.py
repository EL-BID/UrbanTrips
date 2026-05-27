import math
import numpy as np
import pandas as pd
import pytest
from urbantrips.geo.geo import (
    referenciar_h3,
    h3_from_row,
    convert_h3_to_resolution,
    get_h3_buffer_ring_size,
    normalizo_lat_lon,
)


def test_h3_from_row_known_value(df_latlng):
    row = df_latlng.iloc[0]
    assert h3_from_row(row, res=8, lat="latitud", lng="longitud") == "88c2e312b9fffff"


def test_referenciar_h3_known_value(df_latlng):
    result = referenciar_h3(df=df_latlng, res=8, nombre_h3="h3")
    assert result["h3"].iloc[1] == "8839540a87fffff"


def test_referenciar_h3_adds_column(df_latlng):
    result = referenciar_h3(df=df_latlng, res=8, nombre_h3="hex_id")
    assert "hex_id" in result.columns
    assert len(result) == len(df_latlng)


def test_referenciar_h3_custom_lat_lon_cols():
    df = pd.DataFrame({"lat": [-34.6158037], "lon": [-58.5033381]})
    result = referenciar_h3(df=df, res=8, nombre_h3="h3", lat="lat", lon="lon")
    assert result["h3"].iloc[0] == "88c2e312b9fffff"


def test_convert_h3_to_resolution_coarser():
    import h3 as h3lib
    fine = "88c2e312b9fffff"   # resolution 8
    parent = convert_h3_to_resolution(fine, target_resolution=6)
    assert h3lib.get_resolution(parent) == 6


def test_convert_h3_to_resolution_invalid_returns_nan():
    result = convert_h3_to_resolution("not_a_valid_h3_index", target_resolution=6)
    assert result is np.nan or (isinstance(result, float) and math.isnan(result))


def test_get_h3_buffer_ring_size_zero_for_small_buffer():
    ring_size = get_h3_buffer_ring_size(resolucion_h3=8, buffer_meters=1)
    assert ring_size == 0


def test_get_h3_buffer_ring_size_positive_for_large_buffer():
    ring_size = get_h3_buffer_ring_size(resolucion_h3=8, buffer_meters=5000)
    assert ring_size > 0


def test_get_h3_buffer_ring_size_buff_max_covers_buffer():
    import h3 as h3lib
    buffer_meters = 3000
    ring_size = get_h3_buffer_ring_size(resolucion_h3=8, buffer_meters=buffer_meters)
    side = round(h3lib.average_hexagon_edge_length(res=8, unit="m"))
    buff_max = (side * 2 * ring_size) + side
    assert buff_max >= buffer_meters


def test_normalizo_lat_lon_adds_norm_columns():
    df = pd.DataFrame({
        "h3_o": ["88c2e312b9fffff", "88c2e312b9fffff"],
        "h3_d": ["8839540a87fffff", "88c2e312b9fffff"],
    })
    result = normalizo_lat_lon(df, h3_o="h3_o", h3_d="h3_d")
    assert "h3_o_norm" in result.columns
    assert "h3_d_norm" in result.columns
