"""
Pure geometry and H3 helpers for the dashboard preparation layer.
No DB access — all functions take DataFrames / GeoDataFrames and return them.
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Point

from urbantrips.geo.geo import (
    h3_to_geodataframe,
    h3toparent,
    normalizo_lat_lon,
    point_to_h3,
)

logger = logging.getLogger(__name__)


def fix_mixed_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Corrige mezcla Polygon/MultiPolygon para que GeoPandas overlay funcione.
    No cambia la geometría, solo el tipo.
    """
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: MultiPolygon([g]) if g.geom_type == "Polygon" else g
    )
    return gdf


def ensure_geodataframe(df, crs=4326):
    """
    Reconstruye GeoDataFrames desde tablas raw,
    donde geometry puede venir como objeto shapely o WKT.
    """

    if df is None or len(df) == 0 or not hasattr(df, "columns"):
        return gpd.GeoDataFrame(df, geometry=[], crs=crs)

    gdf = df.copy()

    # Si existe WKT pero no geometry
    if "geometry" not in gdf.columns and "wkt" in gdf.columns:
        gdf["geometry"] = gdf["wkt"].apply(
            lambda x: wkt.loads(x) if isinstance(x, str) and x.strip() else None
        )

    # Si geometry viene como string WKT
    elif "geometry" in gdf.columns:
        gdf["geometry"] = gdf["geometry"].apply(
            lambda x: wkt.loads(x) if isinstance(x, str) and x.strip() else x
        )

    # Si todavía no hay geometry
    if "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(gdf, geometry=[], crs=crs)

    return gpd.GeoDataFrame(
        gdf,
        geometry="geometry",
        crs=getattr(df, "crs", None) or crs,
    )


def select_h3_from_polygon(poly, res=8, spacing=0.0001, viz=False):
    """Fill a polygon with points and return the H3 hexagons that cover it."""
    if "id" not in poly.columns:
        poly = poly.reset_index().rename(columns={"index": "id"})

    points_result = pd.DataFrame([])
    poly = poly.reset_index(drop=True).to_crs(4326)
    for i, row in poly.iterrows():
        polygon = poly.geometry[i]
        minx, miny, maxx, maxy = polygon.buffer(0.008).bounds
        x_coords = list(np.arange(minx, maxx, spacing))
        y_coords = list(np.arange(miny, maxy, spacing))
        points = [Point(x, y) for x in x_coords for y in y_coords]
        pts = gpd.GeoDataFrame(geometry=points, crs=4326)
        pts["polygon_number"] = row.id
        points_result = pd.concat([points_result, pts])

    points_result = gpd.sjoin(points_result, poly)
    points_result["h3"] = points_result.apply(point_to_h3, axis=1, resolution=res)
    points_result = (
        points_result.groupby(["polygon_number", "h3"], as_index=False)
        .size()
        .drop(["size"], axis=1)
        .rename(columns={"h3_index": "h3"})
    )

    gdf_hexs = h3_to_geodataframe(points_result.h3).rename(columns={"h3_index": "h3"})
    gdf_hexs = (
        gdf_hexs.merge(points_result, on="h3")[["polygon_number", "h3", "geometry"]]
        .sort_values(["polygon_number", "h3"])
        .reset_index(drop=True)
    )

    if viz:
        ax = poly.boundary.plot(linewidth=1.5, figsize=(15, 15))
        gdf_hexs.plot(ax=ax, alpha=0.6)

    return gdf_hexs.rename(columns={"polygon_number": "id"})


def select_cases_from_polygons(etapas, viajes, polygons, res=8):
    """Filter legs and trips to those whose origin falls within each polygon."""
    polygons = ensure_geodataframe(polygons)
    etapas_selec = gpd.GeoDataFrame([])
    viajes_selec = gpd.GeoDataFrame([])
    polygons_h3 = gpd.GeoDataFrame([])

    for _, poly in polygons.iterrows():
        poly_gdf = gpd.GeoDataFrame([poly], geometry="geometry", crs=polygons.crs)
        gdf_hexs = select_h3_from_polygon(poly_gdf, res=res, viz=False)
        gdf_hexs["id_polygon"] = poly["id"]

        etapas_poly = etapas[etapas.h3_o.isin(gdf_hexs.h3)].copy()
        etapas_poly["id_polygon"] = poly["id"]
        etapas_poly["coincidencias"] = etapas_poly.h3_o.map(
            gdf_hexs.set_index("h3").id_polygon
        )

        viajes_poly = viajes[viajes.h3_o.isin(gdf_hexs.h3)].copy()
        viajes_poly["id_polygon"] = poly["id"]
        viajes_poly["coincidencias"] = viajes_poly.h3_o.map(
            gdf_hexs.set_index("h3").id_polygon
        )

        etapas_selec = pd.concat([etapas_selec, etapas_poly])
        viajes_selec = pd.concat([viajes_selec, viajes_poly])
        polygons_h3 = pd.concat([polygons_h3, gdf_hexs])

    return etapas_selec, viajes_selec, polygons, polygons_h3


def creo_h3_equivalencias(polygons_h3, polygon, res, zonificaciones):
    # El consumidor (preparo_lineas_deseo, rama "cuenca") espera un h3 por fila
    # llamado "h3_o" y, por cada zona, las columnas zona_{res}, lat_{res} y
    # lon_{res}. polygons_h3 trae la columna "h3", la renombramos a "h3_o".
    poly_sel = h3_to_geodataframe(polygons_h3, "h3").rename(columns={"h3": "h3_o"})
    poly_sel = fix_mixed_polygons(poly_sel)
    polygon = fix_mixed_polygons(polygon)

    poly_sel_all = pd.DataFrame([])

    if "res_" in res:
        resol = int(res.replace("res_", ""))
        i = f"res_{resol}"
        poly_sel = poly_sel[["h3_o", "geometry"]].copy()
        poly_sel[f"zona_{i}"] = poly_sel["h3_o"].apply(h3toparent, res=resol)
        poly_2 = h3_to_geodataframe(poly_sel, f"zona_{i}")
        poly_ovl = gpd.overlay(
            poly_sel[["h3_o", "geometry"]],
            poly_2,
            how="intersection",
            keep_geom_type=False,
        )
        poly_ovl = poly_ovl.dissolve(by=f"zona_{i}", as_index=False)
        poly_ovl = poly_ovl[poly_ovl.geom_type.isin(["Polygon", "MultiPolygon"])]
        poly_ovl = gpd.overlay(
            poly_ovl,
            polygon[["geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        if len(poly_ovl) > 0:
            # El recorte con el polígono de cuenca puede partir un hexágono padre
            # en varias piezas: consolido a una fila por zona antes del centroide.
            poly_ovl = fix_mixed_polygons(poly_ovl)
            poly_ovl = poly_ovl.dissolve(by=f"zona_{i}", as_index=False)
            poly_ovl["geometry"] = poly_ovl.geometry.to_crs(4326).representative_point()
            poly_ovl[f"lat_{i}"] = poly_ovl.geometry.y
            poly_ovl[f"lon_{i}"] = poly_ovl.geometry.x
            poly_sel_all = poly_sel.merge(
                poly_ovl[[f"zona_{i}", f"lat_{i}", f"lon_{i}"]],
                on=f"zona_{i}",
                how="left",
            )
    else:
        poly_sel = poly_sel[["h3_o", "geometry"]].copy()
        poly_ovl = gpd.overlay(
            poly_sel,
            zonificaciones[zonificaciones.zona == res][["id", "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        if len(poly_ovl) > 0:
            poly_ovl_agg = poly_ovl.dissolve(by="id", as_index=False)
            poly_ovl_agg = fix_mixed_polygons(poly_ovl_agg)
            poly_ovl_agg = gpd.overlay(
                poly_ovl_agg,
                polygon[["geometry"]],
                how="intersection",
                keep_geom_type=False,
            )
            if len(poly_ovl_agg) > 0:
                poly_ovl_agg = poly_ovl_agg.dissolve(by="id", as_index=False)
                poly_ovl_agg["geometry"] = poly_ovl_agg.geometry.representative_point()
                poly_ovl_agg[f"lat_{res}"] = poly_ovl_agg.geometry.y
                poly_ovl_agg[f"lon_{res}"] = poly_ovl_agg.geometry.x
                poly_ovl_agg[f"zona_{res}"] = poly_ovl_agg.id

                poly_sel_all = poly_ovl.merge(
                    poly_ovl_agg[
                        ["id", f"zona_{res}", f"lat_{res}", f"lon_{res}"]
                    ],
                    on="id",
                    how="left",
                )

    return poly_sel_all


def normalizo_zona(df, zonificaciones):
    if len(zonificaciones) > 0:
        cols = df.columns

        zonificaciones["latlon"] = (
            zonificaciones.geometry.representative_point().y.astype(str)
            + ", "
            + zonificaciones.geometry.representative_point().x.astype(str)
        )
        zonificaciones["aux"] = 1

        zonificacion_tmp1 = zonificaciones[["id", "aux", "geometry"]].rename(
            columns={"id": "tmp_o"}
        )
        zonificacion_tmp1["geometry"] = zonificacion_tmp1[
            "geometry"
        ].representative_point()
        zonificacion_tmp1["h3_o"] = zonificacion_tmp1.apply(
            point_to_h3, axis=1, resolution=8
        )
        zonificacion_tmp1["lat_o"] = zonificacion_tmp1.geometry.y
        zonificacion_tmp1["lon_o"] = zonificacion_tmp1.geometry.x
        zonificacion_tmp1 = zonificacion_tmp1.drop(["geometry"], axis=1)

        zonificacion_tmp2 = zonificaciones[["id", "aux", "geometry"]].rename(
            columns={"id": "tmp_d"}
        )
        zonificacion_tmp2["geometry"] = zonificacion_tmp2[
            "geometry"
        ].representative_point()
        zonificacion_tmp2["h3_d"] = zonificacion_tmp2.apply(
            point_to_h3, axis=1, resolution=8
        )
        zonificacion_tmp1["lat_d"] = zonificacion_tmp2.geometry.y
        zonificacion_tmp1["lon_d"] = zonificacion_tmp2.geometry.x
        zonificacion_tmp2 = zonificacion_tmp2.drop(["geometry"], axis=1)

        zonificacion_tmp = zonificacion_tmp1.merge(zonificacion_tmp2, on="aux")
        zonificacion_tmp = normalizo_lat_lon(zonificacion_tmp, h3_o="h3_o", h3_d="h3_d")
        zonificacion_tmp = zonificacion_tmp[
            ["tmp_o", "tmp_d", "h3_o", "h3_d", "h3_o_norm", "h3_d_norm"]
        ]
        zonificacion_tmp1 = zonificacion_tmp[
            zonificacion_tmp.h3_o == zonificacion_tmp.h3_o_norm
        ].copy()
        zonificacion_tmp1["tmp_o_norm"] = zonificacion_tmp1["tmp_o"]
        zonificacion_tmp1["tmp_d_norm"] = zonificacion_tmp1["tmp_d"]
        zonificacion_tmp2 = zonificacion_tmp[
            zonificacion_tmp.h3_o != zonificacion_tmp.h3_o_norm
        ].copy()
        zonificacion_tmp2["tmp_o_norm"] = zonificacion_tmp2["tmp_d"]
        zonificacion_tmp2["tmp_d_norm"] = zonificacion_tmp2["tmp_o"]
        zonificacion_tmp = pd.concat(
            [zonificacion_tmp1, zonificacion_tmp2], ignore_index=True
        )
        zonificacion_tmp = zonificacion_tmp[
            ["tmp_o", "tmp_d", "tmp_o_norm", "tmp_d_norm"]
        ].rename(columns={"tmp_o": "inicio_norm", "tmp_d": "fin_norm"})

        df = df.merge(zonificacion_tmp, how="left", on=["inicio_norm", "fin_norm"])
        tmp1 = df[df.inicio_norm == df.tmp_o_norm]
        tmp2 = df[df.inicio_norm != df.tmp_o_norm]
        tmp2 = tmp2.rename(
            columns={
                "inicio_norm": "fin_norm",
                "fin_norm": "inicio_norm",
                "poly_inicio_norm": "poly_fin_norm",
                "poly_fin_norm": "poly_inicio_norm",
                "lat1_norm": "lat4_norm",
                "lon1_norm": "lon4_norm",
                "lat4_norm": "lat1_norm",
                "lon4_norm": "lon1_norm",
            }
        )
        tmp2_a = tmp2.loc[tmp2.transfer2_norm == ""]
        tmp2_b = tmp2.loc[tmp2.transfer2_norm != ""]
        tmp2_b = tmp2_b.rename(
            columns={
                "transfer1_norm": "transfer2_norm",
                "transfer2_norm": "transfer1_norm",
                "poly_transfer1_norm": "poly_transfer2_norm",
                "poly_transfer2_norm": "poly_transfer1_norm",
                "lat2_norm": "lat3_norm",
                "lon2_norm": "lon3_norm",
                "lat3_norm": "lat2_norm",
                "lon3_norm": "lon2_norm",
            }
        )

        tmp1 = tmp1[cols]
        tmp2_a = tmp2_a[cols]
        tmp2_b = tmp2_b[cols]

        df = pd.concat([tmp1, tmp2_a, tmp2_b], ignore_index=True)
    return df
