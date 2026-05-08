"""
od_distances.py
---------------
Calculo de distancias de red entre pares OD con cache persistente.

Motor de computo : pandana  (Contraction Hierarchies, C++)
Fuente de red    : pandana.loaders.osm.pdna_network_from_bbox
                   (el mismo loader que usa UrbanTrips, sin osmnx ni pyrosm)
Backends de cache: DuckDB (recomendado) o SQLite
Identificacion   : H3 hexagonos (compatible v3 y v4)

Uso
---
    from compute_distances import compute_od_distances

    df_result = compute_od_distances(
        od_df      = df,
        origin_col = "h3_o",
        dest_col   = "h3_d",
        db_path    = "cache/od.duckdb",
    )

    # Solo la Serie
    df["distance_od"] = compute_od_distances(...)["distance_m"]

Uso para areas grandes (AMBA y similares)
-----------------------------------------
    # Activar tiling automatico y reducir precompute_dist para ahorrar RAM
    df_result = compute_od_distances(
        od_df             = df,
        origin_col        = "h3",
        dest_col          = "h3_lag",
        unit              = "km",
        db_path           = "cache/od.duckdb",
        network_cache_dir = "cache/osm",
        symmetric         = False,
        precompute_dist   = 30_000,   # buffer geografico del tiling (no llama precompute)
        max_tile_deg      = 0.3,      # activa tiling para bbox > 0.3 grados
    )
"""

from __future__ import annotations

import gc
import json
import sqlite3
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import warnings


warnings.filterwarnings(
    "ignore",
    message="Unsigned integer: shortest path distance is trying to be calculated",
    category=UserWarning,
    module="pandana.network",
)
# ══════════════════════════════════════════════════════════════════════════════
# Helpers: H3
# ══════════════════════════════════════════════════════════════════════════════

def _h3_to_geo(h: str) -> tuple[float, float]:
    """Retorna (lat, lon) compatible con h3 v3 y v4."""
    import h3 as h3lib
    if hasattr(h3lib, "cell_to_latlng"):
        return h3lib.cell_to_latlng(h)   # v4+
    return h3lib.h3_to_geo(h)            # v3


def _bbox_from_h3(h3_ids: np.ndarray, buffer_deg: float = 0.02) -> tuple:
    """
    Retorna (ymin, xmin, ymax, xmax) desde centroides H3.
    Convencion pandana: y=lat, x=lon.
    """
    lats, lons = zip(*[_h3_to_geo(h) for h in h3_ids])
    return (
        min(lats) - buffer_deg,   # ymin
        min(lons) - buffer_deg,   # xmin
        max(lats) + buffer_deg,   # ymax
        max(lons) + buffer_deg,   # xmax
    )


def _h3_to_coords(h3_ids: np.ndarray) -> dict[str, tuple[float, float]]:
    """Mapea {h3_id -> (lon, lat)} para todos los H3 unicos."""
    result = {}
    for h in h3_ids:
        lat, lon = _h3_to_geo(h)
        result[h] = (lon, lat)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Construccion de la red pandana
# ══════════════════════════════════════════════════════════════════════════════

_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "http://www.overpass-api.de/api/interpreter",
]

_OVERPASS_HEADERS = {
    "User-Agent": "pandana/0.6 (urbantrips; osmnet)",
    "Accept": "*/*",
}


def _patch_osmnet_overpass(verbose: bool = False) -> None:
    """
    Monkey-patchea osmnet para:
    1. Agregar headers correctos (User-Agent) — overpass-api.de devuelve 406
       cuando recibe el user-agent por defecto de python-requests.
    2. Hacer fallback automatico entre servidores Overpass alternativos.
    """
    try:
        import osmnet.load as _ol
        import requests as _req
        import time as _time
        import re as _re

        def _overpass_request_with_fallback(
            data, pause_duration=None, timeout=180, error_pause_duration=None
        ):
            last_exc = None
            for url in _OVERPASS_URLS:
                try:
                    t0 = _time.time()
                    if verbose:
                        print(f"[od_distances] Overpass POST -> {url}")
                    resp = _req.post(
                        url, data=data, headers=_OVERPASS_HEADERS, timeout=timeout
                    )
                    size_kb = len(resp.content) / 1000.0
                    domain = _re.findall(r"(?s)//(.*?)/", url)[0]
                    if verbose:
                        print(
                            f"[od_distances] {size_kb:,.1f}KB desde {domain} "
                            f"en {_time.time()-t0:.2f}s (HTTP {resp.status_code})"
                        )
                    if resp.status_code == 200:
                        rj = resp.json()
                        if "remark" in rj:
                            print(f"[od_distances] Overpass remark: {rj['remark']}")
                        return rj
                    if resp.status_code in [429, 504]:
                        pause = error_pause_duration or 60
                        print(
                            f"[od_distances] {url} -> {resp.status_code}, "
                            f"reintentando en {pause}s..."
                        )
                        _time.sleep(pause)
                        return _overpass_request_with_fallback(
                            data, pause_duration, timeout, error_pause_duration
                        )
                    last_exc = Exception(
                        f"Server returned no JSON data.\n{resp} {resp.reason}\n{resp.text}"
                    )
                    if verbose:
                        print(
                            f"[od_distances] {url} -> HTTP {resp.status_code}, "
                            f"probando siguiente..."
                        )
                except (_req.exceptions.ConnectionError,
                        _req.exceptions.Timeout) as exc:
                    last_exc = exc
                    if verbose:
                        print(f"[od_distances] {url} -> {exc}, probando siguiente...")
            raise last_exc or Exception("Todos los servidores Overpass fallaron.")

        _ol.overpass_request = _overpass_request_with_fallback

    except ImportError:
        pass  # osmnet no disponible; pandana usara su propio loader


def _build_network(
    bbox: tuple,
    network_type: str,
    verbose: bool,
) -> "pandana.Network":
    """
    Construye red pandana usando su loader interno de OSM.
    bbox = (ymin, xmin, ymax, xmax)  — convencion pandana.

    No llama a network.precompute(): pandana usa Dijkstra directo por par,
    lo que evita el alto consumo de RAM de las Contraction Hierarchies.
    Este es el mismo comportamiento que la version original en carto.py.
    """
    from pandana.loaders import osm as osm_pandana

    _patch_osmnet_overpass(verbose=verbose)

    ymin, xmin, ymax, xmax = bbox

    if verbose:
        print(
            f"[od_distances] Descargando red OSM ({network_type}) | "
            f"lat {ymin:.3f}-{ymax:.3f} lon {xmin:.3f}-{xmax:.3f}"
        )

    network = osm_pandana.pdna_network_from_bbox(
        ymin, xmin, ymax, xmax,
        network_type=network_type,
    )

    if verbose:
        print(
            f"[od_distances] Red: {len(network.nodes_df):,} nodos | "
            f"{len(network.edges_df):,} aristas"
        )

    return network


def _largest_component(
    nodes: "pd.DataFrame",
    edges: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Filtra nodos y aristas al componente debilmente conexo mas grande.

    La red OSM cruda contiene sub-grafos aislados. Nodos en esos islotes
    retornan el sentinel de pandana (4294967.295 m). Guardar solo el
    componente principal evita esos NaN en corridas posteriores.
    """
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(zip(edges["from"], edges["to"]))
    largest_cc = max(nx.connected_components(G), key=len)

    nodes_out = nodes[nodes.index.isin(largest_cc)]
    edges_out = edges[
        edges["from"].isin(largest_cc) & edges["to"].isin(largest_cc)
    ]
    return nodes_out, edges_out


def _save_network(network, nodes_path: str, edges_path: str) -> None:
    """
    Persiste nodes y edges en parquet para reutilizar entre sesiones.

    Filtra al componente conexo principal antes de guardar para evitar que
    nodos aislados de OSM produzcan sentinel (4294967 m) en corridas futuras.

    NOTA: edges_df de pandana tiene columnas ['from', 'to', 'distance'].
    Se guarda 'distance' (metros float) como 'weight'. No usar iloc[:,0]
    porque esa posicion es la columna 'from' (node IDs enteros), no la distancia.
    """
    Path(nodes_path).parent.mkdir(parents=True, exist_ok=True)

    edf = network.edges_df.reset_index() if network.edges_df.index.names != [None] else network.edges_df
    weight_col = (
        "distance" if "distance" in edf.columns
        else edf.select_dtypes(include="float").columns[0]
    )
    edges_df = pd.DataFrame({
        "from"  : edf["from"],
        "to"    : edf["to"],
        "weight": edf[weight_col],
    })

    nodes_df, edges_df = _largest_component(network.nodes_df[["x", "y"]], edges_df)

    nodes_df.to_parquet(nodes_path)
    edges_df.to_parquet(edges_path)


def _load_network(
    nodes_path: str,
    edges_path: str,
    verbose: bool,
) -> "pandana.Network":
    """Carga red pandana desde parquet. No llama precompute (ver _build_network)."""
    import pandana
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)

    if verbose:
        print(
            f"[od_distances] Red cargada desde cache: "
            f"{len(nodes):,} nodos | {len(edges):,} aristas"
        )

    network = pandana.Network(
        node_x       = nodes["x"],
        node_y       = nodes["y"],
        edge_from    = edges["from"],
        edge_to      = edges["to"],
        edge_weights = edges[["weight"]],
        twoway       = False,
    )
    return network


# ══════════════════════════════════════════════════════════════════════════════
# Motor de computo pandana: procesamiento por chunks
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pandana_chunked(
    missing: pd.DataFrame,
    network,
    h3_coords: dict[str, tuple[float, float]],
    chunk_size: int = 100_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calcula distancias para los pares faltantes en chunks para controlar
    el uso de memoria. Cada chunk se procesa con pandana.shortest_path_lengths()
    y se libera antes de procesar el siguiente.
    """
    INF = 4294967.0  # pandana sentinel: UINT32_MAX/1000 ≈ 4294967.295 m (nodo no conectado)
    n_total = len(missing)

    if n_total == 0:
        missing["distance_m"] = np.nan
        return missing

    o_vals = missing["o_norm"].values
    d_vals = missing["d_norm"].values

    # Lookup vectorizado: construir arrays de lon/lat de una vez
    all_h3 = list(h3_coords.keys())
    h3_to_idx = {h: i for i, h in enumerate(all_h3)}
    coords_arr = np.array([h3_coords[h] for h in all_h3])  # (n_h3, 2): lon, lat

    o_idx = np.array([h3_to_idx[h] for h in o_vals])
    d_idx = np.array([h3_to_idx[h] for h in d_vals])

    orig_lons = coords_arr[o_idx, 0]
    orig_lats = coords_arr[o_idx, 1]
    dest_lons = coords_arr[d_idx, 0]
    dest_lats = coords_arr[d_idx, 1]

    del o_idx, d_idx, o_vals, d_vals
    gc.collect()

    if verbose:
        print(f"[od_distances] Snapping {n_total:,} origenes y destinos a la red...")

    orig_nodes = network.get_node_ids(orig_lons, orig_lats)
    dest_nodes = network.get_node_ids(dest_lons, dest_lats)

    del orig_lons, orig_lats, dest_lons, dest_lats
    gc.collect()

    n_chunks = (n_total + chunk_size - 1) // chunk_size
    distances = np.empty(n_total, dtype=np.float64)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_total)

        if verbose and n_chunks > 1:
            print(
                f"[od_distances]   Chunk {i+1}/{n_chunks}: "
                f"pares {start:,}-{end:,}"
            )

        chunk_dist = network.shortest_path_lengths(
            orig_nodes.iloc[start:end].to_numpy(),
            dest_nodes.iloc[start:end].to_numpy(),
        )
        distances[start:end] = chunk_dist

        del chunk_dist
        gc.collect()

    del orig_nodes, dest_nodes
    gc.collect()

    distances = np.where(distances >= INF, np.nan, distances)
    missing = missing.copy()
    missing["distance_m"] = distances

    del distances
    gc.collect()

    return missing


# Version legacy sin chunks (para compatibilidad)
def _compute_pandana(
    missing: pd.DataFrame,
    network,
    h3_coords: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Wrapper que delega a la version chunked con defaults."""
    return _compute_pandana_chunked(missing, network, h3_coords, verbose=False)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: backends de cache
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_od(origins: np.ndarray, dests: np.ndarray):
    flipped = origins > dests
    return np.where(flipped, dests, origins), np.where(flipped, origins, dests)


# ── SQLite ────────────────────────────────────────────────────────────────────

def _init_sqlite(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-131072")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("""
        CREATE TABLE IF NOT EXISTS od_distances (
            o_norm     TEXT NOT NULL,
            d_norm     TEXT NOT NULL,
            distance_m REAL,
            PRIMARY KEY (o_norm, d_norm)
        ) WITHOUT ROWID
    """)
    con.commit()
    return con


def _query_sqlite(pairs: pd.DataFrame, con) -> pd.DataFrame:
    pairs[["o_norm", "d_norm"]].to_sql("_tmp_q", con, if_exists="replace", index=False)
    return pd.read_sql("""
        SELECT q.o_norm, q.d_norm, c.distance_m,
               CASE WHEN c.o_norm IS NOT NULL THEN 1 ELSE 0 END AS in_cache
        FROM   _tmp_q q
        LEFT JOIN od_distances c USING (o_norm, d_norm)
    """, con)


def _store_sqlite(rows: pd.DataFrame, con) -> None:
    rows[["o_norm", "d_norm", "distance_m"]].to_sql(
        "_tmp_ins", con, if_exists="replace", index=False
    )
    con.execute("""
        INSERT OR REPLACE INTO od_distances (o_norm, d_norm, distance_m)
        SELECT o_norm, d_norm, distance_m FROM _tmp_ins
    """)
    con.commit()


# ── DuckDB ────────────────────────────────────────────────────────────────────

def _init_duckdb(db_path: str):
    import duckdb
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS od_distances (
            o_norm     VARCHAR NOT NULL,
            d_norm     VARCHAR NOT NULL,
            distance_m DOUBLE,
            PRIMARY KEY (o_norm, d_norm)
        )
    """)
    return con


def _query_duckdb(pairs: pd.DataFrame, con) -> pd.DataFrame:
    con.register("_tmp_q", pairs[["o_norm", "d_norm"]])
    return con.execute("""
        SELECT q.o_norm, q.d_norm, c.distance_m,
               (c.o_norm IS NOT NULL) AS in_cache
        FROM   _tmp_q q
        LEFT JOIN od_distances c USING (o_norm, d_norm)
    """).df()


def _store_duckdb(rows: pd.DataFrame, con) -> None:
    con.register("_tmp_ins", rows[["o_norm", "d_norm", "distance_m"]])
    con.execute("""
        INSERT OR REPLACE INTO od_distances
        SELECT o_norm, d_norm, distance_m FROM _tmp_ins
    """)


# ── Cache de red: metadata ────────────────────────────────────────────────────

def _network_cache_meta(cache_dir: str) -> dict:
    """Genera paths de cache para una red unica (areas chicas)."""
    d = Path(cache_dir)
    return {
        "nodes": str(d / "network_nodes.parquet"),
        "edges": str(d / "network_edges.parquet"),
        "meta":  str(d / "network_meta.json"),
    }


def _network_cache_meta_tile(cache_dir: str, r: int, c: int) -> dict:
    """Genera paths de cache para el tile (r, c) del grid geografico."""
    d = Path(cache_dir)
    return {
        "nodes": str(d / f"network_nodes_t{r}_{c}.parquet"),
        "edges": str(d / f"network_edges_t{r}_{c}.parquet"),
        "meta":  str(d / f"network_meta_t{r}_{c}.json"),
    }


def _network_cache_valid(
    paths: dict, bbox: tuple, network_type: str, verbose: bool,
) -> bool:
    """
    Verifica que los archivos de cache existan y que los parametros
    coincidan con los de la red guardada. Si no coinciden, avisa y
    retorna False para que se reconstruya.
    """
    if not all(Path(paths[k]).exists() for k in ("nodes", "edges", "meta")):
        return False

    with open(paths["meta"], "r") as f:
        meta = json.load(f)

    cached_bbox = tuple(meta.get("bbox", []))
    covers = (
        len(cached_bbox) == 4
        and cached_bbox[0] <= bbox[0]   # ymin
        and cached_bbox[1] <= bbox[1]   # xmin
        and cached_bbox[2] >= bbox[2]   # ymax
        and cached_bbox[3] >= bbox[3]   # xmax
    )

    if not covers:
        if verbose:
            print(
                f"[od_distances] Cache de red no cubre el bbox actual. "
                f"Cached: {cached_bbox} | Actual: {bbox}. Reconstruyendo..."
            )
        return False

    if meta.get("network_type") != network_type:
        if verbose:
            print(
                f"[od_distances] Cache de red es tipo '{meta.get('network_type')}' "
                f"pero se pidio '{network_type}'. Reconstruyendo..."
            )
        return False

    return True


def _save_network_meta(
    paths: dict, bbox: tuple, network_type: str,
) -> None:
    """Guarda metadata de la red junto a los parquet."""
    meta = {
        "bbox": list(bbox),
        "network_type": network_type,
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Procesamiento en tiles geograficos (para areas grandes como AMBA)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pandana_tiled(
    missing: pd.DataFrame,
    full_bbox: tuple,
    h3_coords: dict[str, tuple[float, float]],
    network_type: str,
    precompute_dist: int,
    max_tile_deg: float,
    network_cache_dir: str | None,
    chunk_size: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Divide el area en tiles geograficos y procesa cada uno por separado.

    Estrategia:
      - Asigna cada par al tile que contiene su origen (o_norm).
      - Para cada tile construye una sub-red cuya bbox cubre los origenes
        del tile mas un buffer = precompute_dist * 1.3 / 111_000 grados.
        Esto garantiza que cualquier destino alcanzable (< precompute_dist m)
        quede dentro de la sub-red.
      - Los destinos fuera del buffer reciben NaN, que es correcto porque
        sus rutas superarian precompute_dist de todos modos.
      - Las redes se cargan y liberan de a una, controlando el pico de RAM.

    Cache por tile:
      Cada tile guarda su red en network_nodes_t{r}_{c}.parquet y
      network_edges_t{r}_{c}.parquet. En corridas posteriores se cargan
      desde disco en vez de re-descargar de OSM.
    """
    from math import ceil

    ymin, xmin, ymax, xmax = full_bbox

    n_rows = ceil((ymax - ymin) / max_tile_deg)
    n_cols = ceil((xmax - xmin) / max_tile_deg)

    if verbose:
        print(
            f"[od_distances] Tiling activado: bbox {ymax - ymin:.2f}deg x "
            f"{xmax - xmin:.2f}deg -> {n_rows}x{n_cols} tiles de ~{max_tile_deg:.2f}deg"
        )

    # Buffer en grados para que la sub-red cubra rutas de hasta precompute_dist m
    net_buffer_deg = precompute_dist / 111_000 * 1.3

    # Asignar cada par al tile de su origen
    o_lons = np.array([h3_coords[h][0] for h in missing["o_norm"].values])
    o_lats = np.array([h3_coords[h][1] for h in missing["o_norm"].values])
    tile_rows = np.clip(((o_lats - ymin) / max_tile_deg).astype(int), 0, n_rows - 1)
    tile_cols = np.clip(((o_lons - xmin) / max_tile_deg).astype(int), 0, n_cols - 1)
    tile_ids = tile_rows * n_cols + tile_cols

    del o_lons, o_lats, tile_rows, tile_cols
    gc.collect()

    results = []
    unique_tiles = np.unique(tile_ids)
    if verbose:
        print('tiles', unique_tiles)
        
    for tid in unique_tiles:
        mask = tile_ids == tid
        tile_miss = missing[mask].reset_index(drop=True)
        r, c = divmod(int(tid), n_cols)

        if verbose:
            print(f"[od_distances] Tile ({r},{c}): {len(tile_miss):,} pares")

        # H3s unicos en este tile (origenes y destinos)
        tile_h3s = np.union1d(
            tile_miss["o_norm"].values,
            tile_miss["d_norm"].values,
        )
        tile_h3_coords = {h: h3_coords[h] for h in tile_h3s if h in h3_coords}

        # Bbox de los ORIGENES del tile + buffer de red
        tile_o_h3s = np.unique(tile_miss["o_norm"].values)
        tile_o_h3s_in_dict = [h for h in tile_o_h3s if h in h3_coords]
        if not tile_o_h3s_in_dict:
            continue
        tile_bbox_origins = _bbox_from_h3(np.array(tile_o_h3s_in_dict), 0)
        tile_net_bbox = (
            tile_bbox_origins[0] - net_buffer_deg,
            tile_bbox_origins[1] - net_buffer_deg,
            tile_bbox_origins[2] + net_buffer_deg,
            tile_bbox_origins[3] + net_buffer_deg,
        )

        # Construir o cargar sub-red para este tile
        if network_cache_dir is not None:
            paths = _network_cache_meta_tile(network_cache_dir, r, c)
            if _network_cache_valid(paths, tile_net_bbox, network_type, verbose):
                network = _load_network(
                    paths["nodes"], paths["edges"],
                    verbose,
                )
            else:
                network = _build_network(
                    tile_net_bbox, network_type, verbose,
                )
                _save_network(network, paths["nodes"], paths["edges"])
                _save_network_meta(paths, tile_net_bbox, network_type)
                if verbose:
                    print(
                        f"[od_distances] Sub-red tile ({r},{c}) guardada en "
                        f"{network_cache_dir}"
                    )
        else:
            network = _build_network(
                tile_net_bbox, network_type, verbose,
            )

        tile_computed = _compute_pandana_chunked(
            tile_miss, network, tile_h3_coords,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        results.append(tile_computed)

        del network, tile_miss, tile_h3_coords, tile_h3s
        gc.collect()

    del tile_ids
    gc.collect()

    if not results:
        return pd.DataFrame(columns=["o_norm", "d_norm", "distance_m"])
    return pd.concat(results, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Funcion principal
# ══════════════════════════════════════════════════════════════════════════════

def compute_od_distances(
    od_df: pd.DataFrame,
    origin_col: str,
    dest_col: str,
    distance_col: str                    = "distance_m",
    unit: Literal["m", "km"]             = "m",
    db_path: str                         = "cache/od_distances.duckdb",
    backend: Literal["duckdb", "sqlite"] = "duckdb",
    network_type: str                    = "drive",
    symmetric: bool                      = True,
    bbox_buffer_deg: float               = 0.02,
    precompute_dist: int                 = 30_000,
    chunk_size: int                      = 100_000,
    network_cache_dir: str | None        = "cache/osm",
    verbose: bool                        = True,
    max_tile_deg: float                  = 0.3,
) -> pd.DataFrame:
    """
    Calcula distancias de red entre pares OD con cache persistente.
    Usa pandana.loaders.osm.pdna_network_from_bbox para construir la red,
    el mismo metodo que usa UrbanTrips internamente.

    Parametros
    ----------
    od_df           : DataFrame con columnas de origen y destino en H3
    origin_col      : nombre de la columna de origen H3
    dest_col        : nombre de la columna de destino H3
    db_path         : ruta al archivo de cache de distancias
    backend         : "duckdb" (recomendado) o "sqlite"
    network_type    : tipo de red ('drive', 'walk', 'bike')
    symmetric       : True -> normaliza (o,d) para ahorrar espacio.
                      Valido si dist(A->B) ~ dist(B->A).
                      False -> guarda ambas direcciones por separado.
    bbox_buffer_deg : buffer en grados alrededor del bbox de los H3. Default 0.02.
    precompute_dist : distancia maxima esperada de un viaje en metros.
                      NO llama a network.precompute() (eso consumia demasiada
                      RAM con la red del AMBA). Se usa solo como buffer
                      geografico en el tiling: la sub-red de cada tile se
                      expande precompute_dist/111km grados para que los
                      destinos alcanzables queden dentro de la red.
                      Default: 30_000 (30 km).
    chunk_size      : cantidad de pares OD a procesar por chunk en pandana.
                      Reducir si hay problemas de memoria. Default: 100_000.
    network_cache_dir : directorio donde guardar/cargar la red pandana en
                      parquet. La primera vez descarga y guarda; las siguientes
                      carga desde disco (mucho mas rapido). Default: "cache/osm".
    verbose         : imprime estadisticas de progreso y cache hit/miss.
    max_tile_deg    : si el bbox supera este valor en alguna dimension (grados),
                      activa el procesamiento en tiles geograficos para controlar
                      el uso de memoria. Cada tile construye una sub-red de
                      (max_tile_deg + 2 * precompute_dist/111km * 1.3) grados.
                      Con max_tile_deg=0.3 y precompute_dist=15_000, el AMBA
                      (~1.4x1.6 grados) se divide en ~5x6=30 tiles.
                      Default: 0.3 (aprox. 33 km).

    Retorna
    -------
    Copia de od_df con columna distance_col agregada.
    Pares sin ruta valida quedan como NaN.
    """

    # ── 1. H3 unicos ─────────────────────────────────────────────────────────

    h3_unicos = (
        pd.concat([od_df[origin_col], od_df[dest_col]])
        .dropna()
        .pipe(lambda s: s[s.str.strip() != ""])
        .unique()
    )


    if verbose:
        print(f"[od_distances] H3 unicos: {len(h3_unicos):,}")

    # ── 2. Coordenadas H3 ────────────────────────────────────────────────────
    h3_coords = _h3_to_coords(h3_unicos)

    # ── 3. Backend de cache ───────────────────────────────────────────────────
    if backend == "duckdb":
        con      = _init_duckdb(db_path)
        query_fn = _query_duckdb
        store_fn = _store_duckdb
    else:
        con      = _init_sqlite(db_path)
        query_fn = _query_sqlite
        store_fn = _store_sqlite

    try:
        # ── 4. Normalizar pares ───────────────────────────────────────────────
        o = od_df[origin_col].values.astype(str)
        d = od_df[dest_col].values.astype(str)

        if symmetric:
            o_norm, d_norm = _normalize_od(o, d)
        else:
            o_norm, d_norm = o, d

        # ── 5. Deduplicar ─────────────────────────────────────────────────────
        unique_pairs = pd.DataFrame({"o_norm": o_norm, "d_norm": d_norm})
        unique_pairs = unique_pairs[
                (unique_pairs["o_norm"].notna()) &
                (unique_pairs["d_norm"].notna()) &
                (unique_pairs["o_norm"] != "") &
                (unique_pairs["d_norm"] != "")
            ].drop_duplicates().reset_index(drop=True)

        if verbose:
            print(f"[od_distances] Pares unicos: {len(unique_pairs):,}")

        # ── 6. Consultar cache ────────────────────────────────────────────────
        cached  = query_fn(unique_pairs, con)
        in_cache_mask = cached["in_cache"].astype(bool)
        found   = cached[in_cache_mask][["o_norm", "d_norm", "distance_m"]]
        missing = cached.loc[~in_cache_mask, ["o_norm", "d_norm"]]

        del unique_pairs, cached
        gc.collect()

        if verbose:
            n_hit   = len(found)
            n_miss  = len(missing)
            n_total = n_hit + n_miss
            pct     = n_hit / n_total if n_total else 0
            n_nan_hit = found["distance_m"].isna().sum()
            nan_str = f" ({n_nan_hit:,} sin ruta)" if n_nan_hit else ""
            print(
                f"[od_distances] Cache hit: {n_hit:,} ({pct:.0%}){nan_str} | "
                f"A calcular: {n_miss:,}"
            )

        # ── 7. Calcular faltantes y persistir ─────────────────────────────────
        if not missing.empty:
            raw_bbox   = _bbox_from_h3(h3_unicos, 0)
            bbox       = _bbox_from_h3(h3_unicos, bbox_buffer_deg)
            bbox_h     = raw_bbox[2] - raw_bbox[0]
            bbox_w     = raw_bbox[3] - raw_bbox[1]
            use_tiling = (bbox_h > max_tile_deg or bbox_w > max_tile_deg)

            if use_tiling:
                if verbose:
                    print(
                        f"[od_distances] Bbox {bbox_h:.2f}deg x {bbox_w:.2f}deg "
                        f"supera max_tile_deg={max_tile_deg:.2f}deg. "
                        f"Activando tiling..."
                    )
                computed = _compute_pandana_tiled(
                    missing, raw_bbox, h3_coords,
                    network_type=network_type,
                    precompute_dist=precompute_dist,
                    max_tile_deg=max_tile_deg,
                    network_cache_dir=network_cache_dir,
                    chunk_size=chunk_size,
                    verbose=verbose,
                )
            else:
                if network_cache_dir is not None:
                    paths = _network_cache_meta(network_cache_dir)
                    if _network_cache_valid(paths, bbox, network_type, verbose):
                        network = _load_network(
                            paths["nodes"], paths["edges"],
                            verbose,
                        )
                    else:
                        network = _build_network(
                            bbox, network_type, verbose,
                        )
                        _save_network(network, paths["nodes"], paths["edges"])
                        _save_network_meta(paths, bbox, network_type)
                        if verbose:
                            print(
                                f"[od_distances] Red guardada en {network_cache_dir}"
                            )
                else:
                    network = _build_network(
                        bbox, network_type, verbose,
                    )

                computed = _compute_pandana_chunked(
                    missing, network, h3_coords,
                    chunk_size=chunk_size,
                    verbose=verbose,
                )
                del network
                gc.collect()

            # Persistir en chunks
            store_chunk = 500_000
            for i in range(0, len(computed), store_chunk):
                store_fn(computed.iloc[i:i+store_chunk], con)

            if verbose:
                n_ok  = computed["distance_m"].notna().sum()
                n_nan = len(computed) - n_ok
                print(
                    f"[od_distances] Calculados: {n_ok:,} | "
                    f"Sin ruta valida: {n_nan:,}"
                )

            all_dist = pd.concat([found, computed], ignore_index=True)
            del computed
            gc.collect()
        else:
            all_dist = found

        del found, missing
        gc.collect()

        # ── 8. Join al df original ────────────────────────────────────────────
        lookup     = all_dist.set_index(["o_norm", "d_norm"])["distance_m"]
        del all_dist
        gc.collect()

        merge_keys  = list(zip(o_norm, d_norm))
        dist_values = lookup.reindex(merge_keys).values

        del lookup, merge_keys, o_norm, d_norm
        gc.collect()

    finally:
        con.close()

    out = od_df.copy()
    out[distance_col] = dist_values / 1000.0 if unit == "km" else dist_values
    out[distance_col] = out[distance_col].round(2)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Utilidad: estadisticas de la cache
# ══════════════════════════════════════════════════════════════════════════════

def cache_stats(
    db_path: str,
    backend: Literal["duckdb", "sqlite"] = "duckdb",
) -> dict:
    """Devuelve estadisticas basicas de la cache de distancias."""
    sql = """
        SELECT
            COUNT(*)                     AS total_pairs,
            COUNT(distance_m)            AS con_distancia,
            COUNT(*) - COUNT(distance_m) AS sin_ruta,
            MIN(distance_m)              AS min_m,
            AVG(distance_m)              AS avg_m,
            MAX(distance_m)              AS max_m
        FROM od_distances
    """
    if backend == "duckdb":
        con   = _init_duckdb(db_path)
        stats = con.execute(sql).df().to_dict(orient="records")[0]
        con.close()
    else:
        con   = _init_sqlite(db_path)
        row   = con.execute(sql).fetchone()
        stats = dict(zip(
            ["total_pairs", "con_distancia", "sin_ruta", "min_m", "avg_m", "max_m"],
            row
        ))
        con.close()
    return stats
