"""Builders for the long-format equivalencias_zonas table.

The equivalencias_zonas table maps H3 cells to zones in LONG format:

    h3 | zona | id | tipo | res

where ``tipo`` takes the values:

    'zonificacion' — administrative zoning layers (partidos, comunas, res_6, ...)
    'poligono'     — analysis area (trips with origin OR destination inside)
    'cuenca'       — catchment basin (trips with origin AND destination inside)

For zoning layers ``zona`` is the layer name (e.g. 'partidos') and ``id`` is the
zone value (e.g. 'La Matanza'). For polygons and basins ``zona`` and ``id`` are
both the polygon id, so dashboards can always filter with WHERE zona = '...'.

The table may contain cells generated at more than one H3 resolution (column
``res``). Joins against trip tables are safe because H3 indexes at different
resolutions never collide.
"""

import logging

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

TIPOS_POLIGONO = ("poligono", "cuenca")


def h3_a_poligono(cell):
    """Return the shapely Polygon (lng, lat) of an H3 cell."""
    borde = h3.cell_to_boundary(cell)
    return Polygon([(lng, lat) for lat, lng in borde])


def _celdas_center(geom, resolucion):
    """H3 cells whose center falls inside the geometry."""
    return set(h3.h3shape_to_cells(h3.geo_to_h3shape(geom), res=resolucion))


def _celdas_overlap(geom, resolucion):
    """H3 cells that intersect the geometry (covering mode)."""
    if not hasattr(h3, "polygon_to_cells_experimental"):
        logger.warning(
            "h3.polygon_to_cells_experimental no disponible; "
            "se usa modo center como aproximación."
        )
        return _celdas_center(geom, resolucion)

    poligonos = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    celdas = set()
    for poly in poligonos:
        ext = [(lat, lng) for lng, lat in poly.exterior.coords]
        agujeros = [[(lat, lng) for lng, lat in r.coords] for r in poly.interiors]
        h3poly = h3.LatLngPoly(ext, *agujeros)
        celdas |= set(h3.polygon_to_cells_experimental(h3poly, resolucion, "overlap"))
    return celdas


def _celdas_de_zona(geom, zona_id, resolucion, modo):
    """H3 cells for one zone, using cell_to_children when the zone is itself
    an H3 cell of coarser resolution (res_6/res_7 layers) — much faster than
    polygon filling and exact."""
    if isinstance(zona_id, str) and h3.is_valid_cell(zona_id):
        res_celda = h3.get_resolution(zona_id)
        if resolucion >= res_celda:
            return set(h3.cell_to_children(zona_id, resolucion))

    if modo == "center":
        return _celdas_center(geom, resolucion)
    if modo == "overlap":
        return _celdas_overlap(geom, resolucion)
    raise ValueError("modo debe ser 'center' u 'overlap'.")


def construir_equivalencias_zonas(
    gdf_zonas=None,
    gdf_poligonos=None,
    resoluciones=8,
    modo_zonas="center",
    modo_poligonos="overlap",
    incluir_geometry=False,
):
    """Build the long-format equivalencias_zonas rows for zoning layers
    and/or analysis polygons.

    Parameters
    ----------
    gdf_zonas : GeoDataFrame, optional
        Zoning layers with columns 'zona', 'id', 'geometry'. Each H3 cell is
        assigned to at most one zone per layer (mode 'center' by default).
    gdf_poligonos : GeoDataFrame, optional
        Analysis polygons with columns 'id', 'geometry' and optionally 'tipo'
        ('poligono' or 'cuenca'; missing values default to 'poligono').
        A cell may belong to several overlapping polygons (mode 'overlap').
    resoluciones : int or iterable of int, default 8
        H3 resolutions to generate. Cells for every resolution are emitted.
    modo_zonas : str, default 'center'
        H3 assignment mode for zoning layers.
    modo_poligonos : str, default 'overlap'
        H3 assignment mode for polygons.
    incluir_geometry : bool, default False
        If True, return a GeoDataFrame (EPSG:4326) with each cell's polygon.
        Geometry is expensive and the stored table does not need it.

    Returns
    -------
    DataFrame (or GeoDataFrame) with columns: h3, zona, id, tipo, res.
    """
    if isinstance(resoluciones, int):
        resoluciones = [resoluciones]
    resoluciones = sorted(set(int(r) for r in resoluciones))

    partes = []

    if gdf_zonas is not None and len(gdf_zonas) > 0:
        if gdf_zonas.crs is None:
            raise ValueError("gdf_zonas no tiene CRS definido.")
        gdf = gdf_zonas.to_crs(epsg=4326)

        filas = []
        for row in gdf.itertuples(index=False):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            for res in resoluciones:
                celdas = _celdas_de_zona(geom, row.id, res, modo_zonas)
                filas += [
                    (c, row.zona, row.id, "zonificacion", res) for c in celdas
                ]

        df_zonas = pd.DataFrame(
            filas, columns=["h3", "zona", "id", "tipo", "res"]
        )
        # each H3 cell belongs to exactly one zone within a layer
        df_zonas = df_zonas.drop_duplicates(subset=["zona", "h3"], keep="first")
        partes.append(df_zonas)

    if gdf_poligonos is not None and len(gdf_poligonos) > 0:
        if gdf_poligonos.crs is None:
            raise ValueError("gdf_poligonos no tiene CRS definido.")
        gdf = gdf_poligonos.to_crs(epsg=4326)
        if "tipo" not in gdf.columns:
            gdf["tipo"] = "poligono"
        gdf["tipo"] = gdf["tipo"].fillna("poligono")

        filas = []
        for row in gdf.itertuples(index=False):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if row.tipo not in TIPOS_POLIGONO:
                logger.warning(
                    "Tipo de polígono desconocido '%s' para id '%s', se omite.",
                    row.tipo, row.id,
                )
                continue
            for res in resoluciones:
                celdas = _celdas_de_zona(geom, row.id, res, modo_poligonos)
                # zona = polygon id so dashboards filter WHERE zona = '{id}'
                filas += [(c, row.id, row.id, row.tipo, res) for c in celdas]

        df_poligonos = pd.DataFrame(
            filas, columns=["h3", "zona", "id", "tipo", "res"]
        )
        # a cell may belong to two overlapping polygons — no dedup across zonas
        df_poligonos = df_poligonos.drop_duplicates(
            subset=["zona", "h3"], keep="first"
        )
        partes.append(df_poligonos)

    if not partes:
        raise ValueError("Debe pasarse al menos gdf_zonas o gdf_poligonos.")

    df = pd.concat(partes, ignore_index=True)

    if incluir_geometry:
        geoms = [h3_a_poligono(c) for c in df["h3"].values]
        return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")

    return df


def _asegurar_formato_long(df):
    """Convert a wide equivalencias_zonas DataFrame to long format.

    Wide format: h3 | <layer columns...> | latitud | longitud.
    Already-long frames (with a 'zona' column) are returned untouched.
    """
    if len(df) == 0 or "zona" in df.columns:
        return df

    value_cols = [c for c in df.columns if c not in ("h3", "latitud", "longitud")]
    largo = df.melt(
        id_vars=["h3"],
        value_vars=value_cols,
        var_name="zona",
        value_name="id",
    )
    largo = largo[largo["id"].notna() & (largo["id"].astype(str) != "")]
    largo["tipo"] = "zonificacion"
    if len(largo) > 0:
        largo["res"] = h3.get_resolution(largo["h3"].iloc[0])
    else:
        largo["res"] = pd.Series(dtype="int64")
    return largo.reset_index(drop=True)


def _quote_sql_str(value):
    """Escape a string for inclusion in a SQL literal."""
    return str(value).replace("'", "''")


def upsert_equivalencias_zonas(equiv_nuevo, ctx=None, db_path="insumos"):
    """Replace in equivalencias_zonas only the zonas present in equiv_nuevo,
    preserving every other zona. Allows adding or refreshing a zone or polygon
    without recomputing the full table.

    Parameters
    ----------
    equiv_nuevo : DataFrame or GeoDataFrame
        Output of construir_equivalencias_zonas for the zonas to update.
    ctx : StorageContext, optional
        When provided, persistence goes through ctx.insumos (pipeline path).
    db_path : str
        Database alias used when ctx is None (dashboard/notebook path).
    """
    if len(equiv_nuevo) == 0:
        return

    equiv_nuevo = pd.DataFrame(equiv_nuevo.drop(columns="geometry", errors="ignore"))
    zonas_nuevas = equiv_nuevo["zona"].astype(str).unique().tolist()

    if ctx is not None:
        existentes = ctx.insumos.get_raw("equivalencias_zonas")
        existentes = _asegurar_formato_long(existentes)
        if len(existentes) > 0:
            existentes = existentes[~existentes["zona"].isin(zonas_nuevas)]
        resultado = pd.concat([existentes, equiv_nuevo], ignore_index=True)
        ctx.insumos.save_raw(resultado, "equivalencias_zonas")
        return

    from urbantrips.dashboard.dash_utils import levanto_tabla_sql, guardar_tabla_sql

    existentes = levanto_tabla_sql("equivalencias_zonas", db_path)
    existentes = _asegurar_formato_long(existentes)
    if len(existentes) > 0:
        existentes = existentes[~existentes["zona"].isin(zonas_nuevas)]
    resultado = pd.concat([existentes, equiv_nuevo], ignore_index=True)
    guardar_tabla_sql(resultado, "equivalencias_zonas", db_path, modo="replace")


def sincronizar_equivalencias_dash(ctx=None):
    """Copy equivalencias_zonas from insumos to the dash database.

    Dashboards join chains_norm (dash) with equivalencias_zonas inside a
    single SQL connection, so the table must live in the same file. insumos
    remains the canonical source; this just refreshes the dash copy.
    """
    if ctx is not None:
        eq = ctx.insumos.get_raw("equivalencias_zonas")
        if len(eq) == 0:
            logger.warning(
                "sincronizar_equivalencias_dash: equivalencias_zonas vacía en insumos."
            )
            return
        eq = _asegurar_formato_long(eq)
        ctx.dash.save_raw(eq, "equivalencias_zonas")
        logger.info(
            "sincronizar_equivalencias_dash: %d filas copiadas a dash.", len(eq)
        )
        return

    from urbantrips.dashboard.dash_utils import levanto_tabla_sql, guardar_tabla_sql

    eq = levanto_tabla_sql("equivalencias_zonas", "insumos")
    if len(eq) == 0:
        logger.warning(
            "sincronizar_equivalencias_dash: equivalencias_zonas vacía en insumos."
        )
        return
    eq = _asegurar_formato_long(eq)
    guardar_tabla_sql(eq, "equivalencias_zonas", "dash", modo="replace")
    logger.info("sincronizar_equivalencias_dash: %d filas copiadas a dash.", len(eq))


def migrar_equivalencias_zonas(ctx=None, db_path="insumos"):
    """One-shot migration of equivalencias_zonas from wide to long format.

    Safe to re-run: if the table is missing, empty or already long
    (it has a 'zona' column), nothing is written.
    """
    if ctx is not None:
        actual = ctx.insumos.get_raw("equivalencias_zonas")
    else:
        from urbantrips.dashboard.dash_utils import levanto_tabla_sql
        actual = levanto_tabla_sql("equivalencias_zonas", db_path)

    if len(actual) == 0:
        logger.info("migrar_equivalencias_zonas: tabla vacía o inexistente, no se migra.")
        return
    if "zona" in actual.columns:
        logger.info("migrar_equivalencias_zonas: la tabla ya está en formato long.")
        return

    largo = _asegurar_formato_long(actual)
    logger.info(
        "migrar_equivalencias_zonas: %d filas wide -> %d filas long.",
        len(actual), len(largo),
    )

    if ctx is not None:
        ctx.insumos.save_raw(largo, "equivalencias_zonas")
    else:
        from urbantrips.dashboard.dash_utils import guardar_tabla_sql
        guardar_tabla_sql(largo, "equivalencias_zonas", db_path, modo="replace")
