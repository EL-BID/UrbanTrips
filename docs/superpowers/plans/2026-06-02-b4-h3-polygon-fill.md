# B4 — Replace Point-Grid with `h3.geo_to_cells` in `select_h3_from_polygon`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cartesian-product point-grid + quadratic `pd.concat` loop in `select_h3_from_polygon` with H3's native `h3.geo_to_cells`, eliminating millions of intermediate Point objects and O(n²) memory behaviour.

**Architecture:** `h3.geo_to_cells(geojson_dict, resolution)` returns the set of H3 cell IDs that cover a GeoJSON polygon directly — no point sampling needed. A `shapely.geometry.mapping(polygon)` call converts the Shapely geometry to the GeoJSON dict H3 expects. The outer loop over the handful of polygon rows is kept; only the inner cartesian product and concat are replaced.

**Tech Stack:** Python, H3 4.3.0, GeoPandas, Shapely, pytest

---

## File map

- Modify: `urbantrips/preparo_dashboard/geo.py`
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

---

### Task 1: Write failing tests

**Files:**
- Modify: `urbantrips/tests/unit/test_preparo_dashboard.py`

- [ ] **Step 1: Add tests to `test_preparo_dashboard.py`**

Append to the file:

```python
# ---------------------------------------------------------------------------
# B4 — select_h3_from_polygon uses h3.geo_to_cells
# ---------------------------------------------------------------------------
import h3
from shapely.geometry import box
import geopandas as gpd


def _make_buenos_aires_polygons():
    """Two small polygons in Buenos Aires at H3 res 8."""
    # Plaza de Mayo area (~100m × 100m)
    poly1 = box(-58.374, -34.609, -58.370, -34.606)
    # Palermo area (~200m × 200m)
    poly2 = box(-58.432, -34.578, -58.426, -34.573)
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[poly1, poly2],
        crs=4326,
    )
    return gdf


def test_select_h3_from_polygon_returns_valid_cells():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    assert len(result) > 0
    for cell in result["h3"]:
        assert h3.is_valid_cell(cell), f"{cell} is not a valid H3 cell"


def test_select_h3_from_polygon_cells_at_correct_resolution():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    for cell in result["h3"]:
        assert h3.get_resolution(cell) == 8


def test_select_h3_from_polygon_id_column_maps_to_source():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    assert set(result["id"].unique()) == {1, 2}


def test_select_h3_from_polygon_returns_geodataframe():
    from urbantrips.preparo_dashboard.geo import select_h3_from_polygon
    gdf = _make_buenos_aires_polygons()
    result = select_h3_from_polygon(gdf, res=8)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs is not None
    assert set(result.columns) >= {"id", "h3", "geometry"}
```

- [ ] **Step 2: Run tests to see them fail**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_returns_valid_cells urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_cells_at_correct_resolution urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_id_column_maps_to_source urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_returns_geodataframe -v
```

Expected: tests may PASS with the old implementation (they test output shape, not internals) — that's fine, they serve as regression guards for the rewrite.

---

### Task 2: Rewrite `select_h3_from_polygon`

**Files:**
- Modify: `urbantrips/preparo_dashboard/geo.py`

- [ ] **Step 1: Add `mapping` import at top of `geo.py`**

`geo.py` already imports from `shapely.geometry`. Add `mapping` to that import:

```python
# BEFORE
from shapely.geometry import MultiPolygon, Point

# AFTER
from shapely.geometry import MultiPolygon, Point, mapping
```

- [ ] **Step 2: Replace `select_h3_from_polygon` body**

Replace the entire function body (keep the signature unchanged):

```python
def select_h3_from_polygon(poly, res=8, spacing=0.0001, viz=False):
    """Fill a polygon with H3 hexagons at the given resolution.

    Parameters
    ----------
    poly : GeoDataFrame
        Polygons to fill. Must have an 'id' column.
    res : int
        H3 resolution.
    spacing : float
        Deprecated — ignored. Kept for backward compatibility with callers
        that pass it as a keyword argument.
    viz : bool
        If True, plot the result.

    Returns
    -------
    GeoDataFrame with columns ['id', 'h3', 'geometry'].
    """
    if "id" not in poly.columns:
        poly = poly.reset_index().rename(columns={"index": "id"})

    poly = poly.reset_index(drop=True).to_crs(4326)
    records = []
    for _, row in poly.iterrows():
        geojson = mapping(row.geometry)
        cells = h3.geo_to_cells(geojson, res)
        records.extend({"id": row.id, "h3": cell} for cell in cells)

    if not records:
        return gpd.GeoDataFrame(columns=["id", "h3", "geometry"], crs=4326)

    points_result = pd.DataFrame(records)
    gdf_hexs = h3_to_geodataframe(points_result["h3"].tolist()).rename(columns={"h3_index": "h3"})
    gdf_hexs = (
        gdf_hexs.merge(points_result, on="h3")[["id", "h3", "geometry"]]
        .sort_values(["id", "h3"])
        .reset_index(drop=True)
    )

    if viz:
        ax = poly.boundary.plot(linewidth=1.5, figsize=(15, 15))
        gdf_hexs.plot(ax=ax, alpha=0.6)

    return gdf_hexs
```

Note: `h3` is already imported at the top of `geo.py` (`import h3`). Confirm this before saving.

- [ ] **Step 3: Run all four new tests**

```
uv run pytest urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_returns_valid_cells urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_cells_at_correct_resolution urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_id_column_maps_to_source urbantrips/tests/unit/test_preparo_dashboard.py::test_select_h3_from_polygon_returns_geodataframe -v
```

Expected: all 4 PASS.

- [ ] **Step 4: Run full unit tests**

```
uv run pytest urbantrips/tests/unit/ -x -q
```

Expected: all existing tests pass.

---

### Task 3: Commit

- [ ] **Step 1: Commit**

```
git add urbantrips/preparo_dashboard/geo.py urbantrips/tests/unit/test_preparo_dashboard.py
git commit -m "perf(b4): replace point-grid with h3.geo_to_cells in select_h3_from_polygon"
```
