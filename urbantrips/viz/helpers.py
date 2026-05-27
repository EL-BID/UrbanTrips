"""
Pure utility helpers for viz — no DB access, no Streamlit, no ctx.
"""
import logging
import os

import geopandas as gpd
import h3
import mapclassify
import folium
import numpy as np
import pandas as pd
from folium import Figure
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def standarize_size(series, min_size, max_size):
    if series.min() == series.max():
        return pd.Series([min_size] * len(series))
    return min_size + (max_size - min_size) * (series - series.min()) / (
        series.max() - series.min()
    )


def create_squared_polygon(min_x, min_y, max_x, max_y, epsg):
    width = max(max_x - min_x, max_y - min_y)
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    square_bbox_min_x = center_x - width / 2
    square_bbox_min_y = center_y - width / 2
    square_bbox_max_x = center_x + width / 2
    square_bbox_max_y = center_y + width / 2

    square_bbox_coords = [
        (square_bbox_min_x, square_bbox_min_y),
        (square_bbox_max_x, square_bbox_min_y),
        (square_bbox_max_x, square_bbox_max_y),
        (square_bbox_min_x, square_bbox_max_y),
    ]
    p = Polygon(square_bbox_coords)
    return gpd.GeoSeries([p], crs=f"EPSG:{epsg}")


def format_num(num, lpad=10):
    fnum = "{:,}".format(num).replace(".", "*").replace(",", ".").replace("*", ",")
    if lpad > 0:
        fnum = fnum.rjust(lpad, " ")
    return fnum


def extract_hex_colors_from_cmap(cmap, n=5):
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, n))
    return [mcolors.rgb2hex(color) for color in colors]


def crea_df_burbujas(df, zonas, h3_o="h3_o", var_fex="", porc_viajes=100, res=7):
    zonas["h3_o_tmp"] = zonas["h3"].apply(h3.cell_to_parent, res=res)

    hexs = (
        zonas[(zonas.fex.notna()) & (zonas.fex != 0)]
        .groupby("h3_o_tmp", as_index=False)
        .size()
        .drop(["size"], axis=1)
    )
    hexs = hexs.merge(
        zonas[(zonas.fex.notna()) & (zonas.fex != 0)]
        .groupby("h3_o_tmp")
        .apply(lambda x: np.average(x["longitud"], weights=x["fex"]))
        .reset_index()
        .rename(columns={0: "longitud"}),
        how="left",
    )
    hexs = hexs.merge(
        zonas[(zonas.fex.notna()) & (zonas.fex != 0)]
        .groupby("h3_o_tmp")
        .apply(lambda x: np.average(x["latitud"], weights=x["fex"]))
        .reset_index()
        .rename(columns={0: "latitud"}),
        how="left",
    )

    df["h3_o_tmp"] = df[h3_o].apply(h3.cell_to_parent, res=res)
    df_agg = df.groupby(["dia", "h3_o_tmp"], as_index=False).agg({var_fex: "sum"})
    df_agg = df_agg.groupby(["h3_o_tmp"], as_index=False).agg({var_fex: "mean"})
    df_agg = df_agg.merge(hexs.rename(columns={"latitud": "lat_o", "longitud": "lon_o"}))
    df_agg = gpd.GeoDataFrame(
        df_agg,
        geometry=gpd.points_from_xy(df_agg["lon_o"], df_agg["lat_o"]),
        crs=4326,
    )
    df_agg = df_agg.sort_values(var_fex, ascending=False).reset_index(drop=True)
    df_agg["cumsum"] = round(df_agg[var_fex].cumsum() / df_agg[var_fex].sum() * 100)
    df_agg = df_agg[df_agg["cumsum"] < porc_viajes]
    return df_agg


def crear_mapa_folium(df_agg, cmap, var_fex, savefile, k_jenks=5):
    bins = [df_agg[var_fex].min() - 1] + mapclassify.FisherJenks(
        df_agg[var_fex], k=k_jenks
    ).bins.tolist()
    range_bins = range(0, len(bins) - 1)
    bins_labels = [f"{int(bins[n])} a {int(bins[n+1])} viajes" for n in range_bins]
    df_agg["cuts"] = pd.cut(df_agg[var_fex], bins=bins, labels=bins_labels)

    fig = Figure(width=800, height=800)
    m = folium.Map(
        location=[df_agg.lat_o.mean(), df_agg.lon_o.mean()],
        zoom_start=9,
        tiles="cartodbpositron",
    )
    title_html = '<h3 align="center" style="font-size:20px"><b>Your map title</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    line_w = 0.5
    colors = extract_hex_colors_from_cmap(cmap=cmap, n=k_jenks)
    for n, i in enumerate(bins_labels):
        df_agg[df_agg.cuts == i].explore(
            m=m,
            color=colors[n],
            style_kwds={"fillOpacity": 0.3, "weight": line_w},
            name=i,
            tooltip=False,
        )
        line_w += 3

    folium.LayerControl(name="xx").add_to(m)
    fig.add_child(m)
    db_path = os.path.join("resultados", "html", savefile)
    m.save(db_path)
