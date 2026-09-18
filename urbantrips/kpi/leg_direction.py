"""
leg_direction.py
----------------
Identifies the direction (0 or 1) a leg was traveled by comparing it to route data.

Key concept:
- Routes have direction field (0 or 1) and section_id (order along route)
- When traveling in direction 0: section_id increases
- When traveling in direction 1: section_id also increases (but it's the opposite physical direction)
- We match leg origin/destination to route h3 cells and use section_id delta to vote
"""

from __future__ import annotations
import h3
import pandas as pd
import random
from typing import Optional
from urbantrips.geo import geo
from urbantrips.utils.utils import leer_configs_generales
from urbantrips.storage.context import StorageContext
import multiprocessing
import logging

logger = logging.getLogger(__name__)


def identify_leg_direction(
    leg_h3_o: str,
    leg_h3_d: str,
    id_linea: int,
    routes_df: pd.DataFrame,
    leg_id_ramal: Optional[int] = None,
    debug=False,
    ring_size=1,
) -> dict:
    """
    Identify the direction (0 or 1) a single leg was traveled.

    Parameters
    ----------
    leg_h3_o : str
        Origin h3 cell of the leg
    leg_h3_d : str
        Destination h3 cell of the leg
    id_linea : int
        Line ID to filter routes
    routes_df : pd.DataFrame
        DataFrame with columns: h3, id_linea, direction, section_id
        May include: id_ramal (if line has branches), has_branches
    leg_id_ramal : int, optional
        The id_ramal from the leg data.
        If provided and present in possible_branches, it will be selected.
    debug : bool, default False
        If True, include additional diagnostic information in results

    Returns
    -------
    tuple of (bool, dict)
        First element: True if successful, False if error occurred
        Second element: Dictionary with:
        On success:
        - direction: identified direction (0, 1, or None)
        - confidence: voting confidence (0-1)
        - selected_section_id_o: section_id at origin
        - selected_section_id_d: section_id at destination
        - selected_branch: chosen id_ramal (present for all lines)
        If line has branches, also includes:
        - possible_branches: list of id_ramal that match the direction
        On failure:
        - error: error type string
        - additional diagnostic fields depending on error type
    """
    # Filter routes for this line
    line_routes = routes_df[routes_df["id_linea"] == id_linea].copy()

    # Determine if this line has branches
    if "has_branches" in line_routes.columns:
        try:
            has_branches = bool(line_routes["has_branches"].iloc[0])
        except Exception:
            return False, {"error": "no_routes_for_line"}
    elif "id_ramal" in line_routes.columns:
        # Fallback: check if multiple unique id_ramal exist
        has_branches = line_routes["id_ramal"].nunique() > 1
    else:
        # No id_ramal column means no branches
        has_branches = False

    if line_routes.empty:
        return False, {"error": "no_routes_for_line"}

    # Match origin and destination to routes
    matches_o = _find_h3_matches(
        leg_h3_o,
        line_routes,
        role="origin",
        has_branches=has_branches,
        ring_size=ring_size,
    )
    matches_d = _find_h3_matches(
        leg_h3_d,
        line_routes,
        role="destination",
        has_branches=has_branches,
        ring_size=ring_size,
    )

    if matches_o.empty or matches_d.empty:
        return False, {
            "error": "no_matches",
            "matches_o_count": len(matches_o),
            "matches_d_count": len(matches_d),
        }

    # Find common routes/branches (same id_ramal and direction)
    common = _find_common_routes(matches_o, matches_d)

    if common.empty:
        # Build error info based on available columns
        error_cols = ["direction"]
        if "id_ramal" in matches_o.columns:
            error_cols.insert(0, "id_ramal")

        return False, {
            "error": "no_common_routes",
            "unique_o": matches_o[error_cols].drop_duplicates().values.tolist(),
            "unique_d": matches_d[error_cols].drop_duplicates().values.tolist(),
        }

    # Vote based on section_id delta
    votes = _calculate_direction_votes(common)

    if not votes:
        return False, {"error": "no_valid_votes", "common_count": len(common)}

    # Determine winner
    direction = _determine_direction_from_votes(votes)

    # Get possible branches and select one (only if line has branches)
    possible_branches = []
    selected_branch = None
    selected_section_id_o = None
    selected_section_id_d = None

    if has_branches and direction is not None:
        # Filter matches to only the identified direction
        matches_o_dir = matches_o[matches_o["direction"] == direction]
        matches_d_dir = matches_d[matches_d["direction"] == direction]

        # Find common id_ramal values
        ramales_o = set(matches_o_dir["id_ramal"].unique())
        ramales_d = set(matches_d_dir["id_ramal"].unique())
        possible_branches = sorted(ramales_o & ramales_d)

        # Select one branch from possible_branches
        if possible_branches:
            if leg_id_ramal is not None and leg_id_ramal in possible_branches:
                # Use the leg's id_ramal if it's in the possible branches
                selected_branch = leg_id_ramal
            else:
                # Otherwise, randomly select from possible branches
                selected_branch = random.choice(possible_branches)

            # Get section_id values for the selected branch and direction
            mask_o = (matches_o["id_ramal"] == selected_branch) & (
                matches_o["direction"] == direction
            )
            mask_d = (matches_d["id_ramal"] == selected_branch) & (
                matches_d["direction"] == direction
            )

            if mask_o.any():
                selected_section_id_o = int(matches_o[mask_o]["section_id"].iloc[0])
            if mask_d.any():
                selected_section_id_d = int(matches_d[mask_d]["section_id"].iloc[0])

    elif not has_branches and direction is not None:
        # No branches - just get section_id for the direction
        mask_o = matches_o["direction"] == direction
        mask_d = matches_d["direction"] == direction
        selected_branch = leg_id_ramal
        if mask_o.any():
            selected_section_id_o = int(matches_o[mask_o]["section_id"].iloc[0])

        if mask_d.any():
            selected_section_id_d = int(matches_d[mask_d]["section_id"].iloc[0])

    results = {
        "direction": direction,
        "confidence": (
            votes.count(direction) / len(votes) if direction is not None else 0
        ),
        "selected_section_id_o": selected_section_id_o,
        "selected_section_id_d": selected_section_id_d,
    }

    # Add branch-related fields
    # possible_branches only for lines with actual branches
    if has_branches:
        results["possible_branches"] = possible_branches
    # selected_branch for all lines (when available)
    if selected_branch is not None:
        results["selected_branch"] = selected_branch

    if debug:
        results["matches_o_count"] = len(matches_o)
        results["matches_o"] = matches_o
        results["matches_d_count"] = len(matches_d)
        results["matches_d"] = matches_d
        results["common_routes_count"] = len(common)
        results["common"] = common
        results["votes"] = votes

    return True, results


def identify_legs_direction(
    legs_df: pd.DataFrame,
    routes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify direction for multiple legs.

    Parameters
    ----------
    legs_df : pd.DataFrame
        DataFrame with columns: h3_o, h3_d, id_linea
        Optional columns: id_ramal (used for branch selection if line
            has branches)
    routes_df : pd.DataFrame
        DataFrame with columns: h3, id_linea, direction, section_id
        May include: id_ramal (if lines have branches), has_branches

    Returns
    -------
    pd.DataFrame
        DataFrame containing only successfully processed legs with added columns:
        - direction_inferred: identified direction (0, 1, or None)
        - confidence: voting confidence (0-1)
        - selected_section_id_o: section_id at origin
        - selected_section_id_d: section_id at destination
        - selected_branch: chosen id_ramal (present for all lines)
        When line has branches, also includes:
        - possible_branches: list of id_ramal that match the direction
        Legs that failed direction identification are excluded from the result.
    """
    configs = leer_configs_generales(autogenerado=False)
    # resolucion_h3 = configs["resolucion_h3"]
    # tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    # ring_size = geo.get_h3_buffer_ring_size(resolucion_h3, tolerancia_parada_destino)
    ring_size = 1

    successful_results = []
    successful_indices = []

    for idx, row in legs_df.iterrows():
        # Get leg_id_ramal if it exists in the row
        leg_id_ramal = row.get("id_ramal") if "id_ramal" in row.index else None

        success, result = identify_leg_direction(
            leg_h3_o=row["h3_o"],
            leg_h3_d=row["h3_d"],
            id_linea=row["id_linea"],
            routes_df=routes_df,
            leg_id_ramal=leg_id_ramal,
            ring_size=ring_size,
        )

        # Only process successful results
        if not success:
            continue

        try:
            # Build result dict with common fields
            result_dict = {
                "direction_inferred": result.get("direction"),
                "confidence": result.get("confidence", 0),
                "selected_section_id_o": result.get("selected_section_id_o"),
                "selected_section_id_d": result.get("selected_section_id_d"),
            }
        except Exception as e:
            print(f"Error processing leg {idx}: {e}")
            print("Leg data:", row.to_dict())
            print(result)
            raise e

        # Add branch fields if they exist in result
        if "possible_branches" in result:
            result_dict["possible_branches"] = result.get("possible_branches", [])
        if "selected_branch" in result:
            result_dict["selected_branch"] = result.get("selected_branch")

        successful_results.append(result_dict)
        successful_indices.append(idx)

    # Create result dataframe only for successful legs
    if successful_results:
        result_df = pd.DataFrame(successful_results, index=successful_indices)
        successful_legs = legs_df.loc[successful_indices]
        return pd.concat([successful_legs, result_df], axis=1)
    else:
        # No successful legs, return empty dataframe with expected columns
        return pd.DataFrame()


def _find_h3_matches(
    h3_cell: str,
    routes_df: pd.DataFrame,
    role: str = "origin",
    has_branches: bool = True,
    ring_size=1,
) -> pd.DataFrame:
    """
    Find route records matching the h3 cell and its k-ring neighbors (k=1).
    Returns one section_id per (id_linea, id_ramal, direction) when
    has_branches=True, or per (id_linea, direction) when has_branches=False:
    - role="origin": max section_id
    - role="destination": min section_id

    Parameters
    ----------
    h3_cell : str
        H3 cell to match
    routes_df : pd.DataFrame
        DataFrame with route data
    role : str
        Either "origin" or "destination" to determine section_id aggregation
    has_branches : bool
        If True, group by id_ramal; if False, ignore id_ramal
            (lines without branches)
    """
    # Always search in k-ring neighbors (including center)
    # This looks before knowing the direction
    neighbors = h3.grid_disk(h3_cell, ring_size)
    matches = routes_df[routes_df["h3"].isin(neighbors)].copy()

    if matches.empty:
        return matches

    # Keep only one section_id per group
    # Max for origin, min for destination
    agg_func = "max" if role == "origin" else "min"

    # Group by id_linea and direction (and id_ramal if has_branches)
    group_cols = ["id_linea", "direction"]
    if has_branches:
        group_cols.insert(1, "id_ramal")

    result = (
        matches.groupby(group_cols)
        .agg({"section_id": agg_func, "h3": "first"})
        .reset_index()
    )

    return result


def _find_common_routes(
    matches_o: pd.DataFrame, matches_d: pd.DataFrame
) -> pd.DataFrame:
    """
    Find routes/directions that appear in both origin and destination
    matches. Returns merged dataframe with section_id from both ends.
    Handles both cases: with id_ramal (branches) and without
    (direction only).
    """
    # Determine if we have branches based on presence of id_ramal column
    has_branches = "id_ramal" in matches_o.columns

    # Define merge keys
    merge_keys = ["direction"]
    if has_branches:
        merge_keys.insert(0, "id_ramal")

    # Get unique combinations from each
    o_keys = matches_o[merge_keys].drop_duplicates()
    d_keys = matches_d[merge_keys].drop_duplicates()

    # Find intersection
    common_keys = pd.merge(o_keys, d_keys, on=merge_keys)

    if common_keys.empty:
        return pd.DataFrame()

    # Get section_id for each common route/branch at origin/destination
    # Take max section_id at origin, min at destination (multiple h3 cells)
    o_sections = (
        matches_o.groupby(merge_keys)["section_id"]
        .max()
        .reset_index()
        .rename(columns={"section_id": "section_id_o"})
    )

    d_sections = (
        matches_d.groupby(merge_keys)["section_id"]
        .min()
        .reset_index()
        .rename(columns={"section_id": "section_id_d"})
    )

    # Merge to get both section_ids
    common = pd.merge(common_keys, o_sections, on=merge_keys)
    common = pd.merge(common, d_sections, on=merge_keys)

    # Calculate delta
    common["section_delta"] = common["section_id_d"] - common["section_id_o"]

    return common


def _calculate_direction_votes(common_df: pd.DataFrame) -> list:
    """
    Calculate direction votes based on section_id delta and direction field.

    Logic:
    - If section_delta > 0: we're moving forward along the route
      - If direction == 0: vote for 0
      - If direction == 1: vote for 1
    - If section_delta < 0: we're moving backward (shouldn't happen often)
      - If direction == 0: vote for 1 (going opposite)
      - If direction == 1: vote for 0 (going opposite)
    - If section_delta == 0: no vote (same section)
    """
    votes = []

    for _, row in common_df.iterrows():
        delta = row["section_delta"]
        direction = row["direction"]

        if delta > 0:
            # Moving forward along route
            votes.append(int(direction))
        elif delta < 0:
            # Moving backward (opposite direction)
            votes.append(1 - int(direction))
        # delta == 0: no vote

    return votes


def _determine_direction_from_votes(votes: list) -> Optional[int]:
    """
    Determine final direction from votes.
    Returns 0, 1, or None if unclear.

    Rules:
    - If all votes agree: return that direction
    - If majority (>50%): return majority direction
    - Otherwise: None
    """
    if not votes:
        return None

    # Count votes
    count_0 = votes.count(0)
    count_1 = votes.count(1)
    total = len(votes)

    # Unanimous or majority
    if count_0 == total:
        return 0
    elif count_1 == total:
        return 1
    elif count_0 > total / 2:
        return 0
    elif count_1 > total / 2:
        return 1
    else:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for batch processing
# ─────────────────────────────────────────────────────────────────────────────


def prepare_routes_from_h3(
    routes_h3: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    has_branches: bool = True,
) -> pd.DataFrame:
    """
    Prepare routes dataframe from routes_h3 format.

    Expected input columns: h3, direction, section_id
    When has_branches=True, also requires: id_ramal
    When has_branches=False, also requires: id_linea

    Parameters
    ----------
    routes_h3 : pd.DataFrame
        Route h3 data with h3, direction, section_id columns.
        If has_branches=True: must include id_ramal.
        If has_branches=False: must include id_linea.
    metadata : pd.DataFrame, optional
        Metadata table. Only used when has_branches=True.
        Must contain id_ramal and id_linea columns for mapping.
    has_branches : bool, default=True
        If True, the dataset contains routes with branches (id_ramal).
        If False, the dataset contains routes without branches (only
        id_linea and direction).

    Returns
    -------
    pd.DataFrame
        Prepared routes with required columns including has_branches flag
    """
    df = routes_h3.copy()

    if has_branches:
        # City/dataset has branches - process with id_ramal
        # Ensure id_ramal is int64
        if "id_ramal" not in df.columns:
            raise ValueError(
                "When has_branches=True, routes_h3 must include id_ramal column"
            )
        df["id_ramal"] = df["id_ramal"].astype("int64")

        # Get id_linea if not present
        if "id_linea" not in df.columns:
            if metadata is not None:
                # Use metadata table to get id_linea
                if (
                    "id_ramal" not in metadata.columns
                    or "id_linea" not in metadata.columns
                ):
                    raise ValueError(
                        "When has_branches=True and id_linea not in routes_h3, "
                        "metadata must contain both id_ramal and id_linea columns"
                    )
                # Ensure metadata has int64 types
                metadata_subset = (
                    metadata[["id_ramal", "id_linea"]].drop_duplicates().copy()
                )
                metadata_subset["id_ramal"] = metadata_subset["id_ramal"].astype(
                    "int64"
                )
                metadata_subset["id_linea"] = metadata_subset["id_linea"].astype(
                    "int64"
                )

                df = df.merge(metadata_subset, on="id_ramal", how="left")

                # Check for missing mappings

                missing_count = len(df.loc[df.id_linea.isna(), "id_ramal"].unique())

                if missing_count > 0:
                    print(
                        f"Warning: {missing_count} ramales could not be "
                        "mapped to id_linea. Dropping affected rows."
                    )
                    df = df[df["id_linea"].notna()].copy()
            else:
                # Fallback: extract from id_ramal string (first 10 chars)
                df["id_linea"] = df["id_ramal"].astype(str).str[:10].astype("int64")
        else:
            # Ensure id_linea is int64 if it already exists
            df["id_linea"] = df["id_linea"].astype("int64")

        # Ensure required columns
        required = ["h3", "id_linea", "id_ramal", "direction", "section_id"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure proper types for all columns
        result = df[required].copy()
        result["id_linea"] = result["id_linea"].astype("int64")
        result["id_ramal"] = result["id_ramal"].astype("int64")
        result["direction"] = result["direction"].astype("int64")
        result["section_id"] = result["section_id"].astype("int64")

        # Add has_branches column: True if line has multiple branches
        branches_per_line = result.groupby("id_linea")["id_ramal"].nunique().to_dict()
        result["has_branches"] = result["id_linea"].map(
            lambda x: branches_per_line[x] > 1
        )

    else:
        # City/dataset has no branches - process without id_ramal
        if "id_linea" not in df.columns:
            raise ValueError(
                "When has_branches=False, routes_h3 must include id_linea column"
            )

        # Ensure id_linea is int64
        df["id_linea"] = df["id_linea"].astype("int64")

        # For no-branches case, create a synthetic id_ramal = id_linea
        # This ensures compatibility with the rest of the code
        df["id_ramal"] = df["id_linea"]

        # Ensure required columns
        required = ["h3", "id_linea", "id_ramal", "direction", "section_id"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure proper types for all columns
        result = df[required].copy()
        result["id_linea"] = result["id_linea"].astype("int64")
        result["id_ramal"] = result["id_ramal"].astype("int64")
        result["direction"] = result["direction"].astype("int64")
        result["section_id"] = result["section_id"].astype("int64")

        # All lines have no branches in this dataset
        result["has_branches"] = False

    return result


def _identify_legs_direction_chunk_wrapper(chunk_data):
    """
    Wrapper for identify_legs_direction to work with multiprocessing.Pool.

    Parameters
    ----------
    chunk_data : tuple
        (legs_chunk, routes_prepared) - chunk of legs DataFrame and routes data

    Returns
    -------
    pandas.DataFrame
        DataFrame with direction identified for legs in the chunk
    """
    legs_chunk, routes_prepared = chunk_data
    try:
        result = identify_legs_direction(legs_df=legs_chunk, routes_df=routes_prepared)
        return result
    except Exception as e:
        logger.error("Error procesando chunk de legs: %s", str(e))
        # Return empty DataFrame with expected columns
        return pd.DataFrame()


def _process_legs_direction_parallel(legs_data, routes_prepared):
    """
    Process legs direction identification in parallel using multiprocessing.

    Parameters
    ----------
    legs_data : pandas.DataFrame
        DataFrame with legs data
    routes_prepared : pandas.DataFrame
        Prepared routes data from prepare_routes_from_h3

    Returns
    -------
    pandas.DataFrame
        DataFrame with direction identified for all legs
    """
    n_cores = max(int(multiprocessing.cpu_count() - 1), 1)
    n = len(legs_data)

    # Calculate chunk size - aim for reasonable chunks per core
    # Use larger chunks than routes processing since leg processing is lighter
    chunks_per_core = 4
    n_chunks = min(n_cores * chunks_per_core, n)
    chunk_size = max(1, n // n_chunks)

    logger.info(
        "Procesando %d legs en paralelo con %d cores (%d chunks de ~%d legs)",
        n,
        n_cores,
        n_chunks,
        chunk_size,
    )

    # Split legs into chunks
    chunks = []
    for i in range(0, n, chunk_size):
        chunk = legs_data.iloc[i : i + chunk_size]
        chunks.append((chunk, routes_prepared))

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(_identify_legs_direction_chunk_wrapper, chunks)

    # Filter out empty results and concatenate
    valid_results = [r for r in results if len(r) > 0]
    if valid_results:
        legs_with_direction = pd.concat(valid_results, ignore_index=True)
        return legs_with_direction
    else:
        return pd.DataFrame()


def assign_direction_for_line_and_hours(
    ctx: StorageContext,
    id_linea: int,
    hora_inicio: Optional[int] = None,
    hora_fin: Optional[int] = None,
    dia: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assigns direction and branch to legs for a specific line and hour range.

    Similar to assign_direction_and_branch_or_line (legs.py) but filters by:
    - Specific line (id_linea)
    - Hour range when the trip started (hora field) - optional
    - Optionally, a specific day

    This is useful for analyzing direction patterns for a specific line during
    specific hours, without processing the entire dataset.

    Parameters
    ----------
    ctx : StorageContext
        Storage context with database connections
    id_linea : int
        Line ID to filter legs
    hora_inicio : int, optional
        Start hour (inclusive), 0-23. If None, no lower bound on hours.
    hora_fin : int, optional
        End hour (inclusive), 0-23. If None, no upper bound on hours.
    dia : str, optional
        Specific day to process (format: 'YYYY-MM-DD'). If None,
        processes all days.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, dia, id_linea, id_ramal, h3_o, h3_d,
        direction_inferred, confidence, selected_section_id_o,
        selected_section_id_d, possible_branches (if line has branches),
        selected_branch

    Examples
    --------
    >>> # Analyze line 60 during morning rush hour
    >>> results = assign_direction_for_line_and_hours(ctx, 60, 7, 9)
    >>>
    >>> # Analyze line 152 on a specific day, all hours
    >>> results = assign_direction_for_line_and_hours(
    ...     ctx, 152, dia='2025-10-15'
    ... )
    >>>
    >>> # Analyze line 152, all days and hours
    >>> results = assign_direction_for_line_and_hours(ctx, 152)
    """
    configs = leer_configs_generales(autogenerado=False)
    has_branches = configs.get("lineas_contienen_ramales", False)
    h3_res = configs.get("resolucion_h3", None)

    if has_branches:
        source_table_h3 = "branches"
        source_table_metadata = "ramales"
        id_col = "id_ramal"
    else:
        source_table_h3 = "lines"
        source_table_metadata = "lineas"
        id_col = "id_linea"

    hora_msg = (
        f", horas {hora_inicio}-{hora_fin}"
        if hora_inicio is not None and hora_fin is not None
        else ""
    )
    dia_msg = f", día {dia}" if dia else ""
    logger.info(
        "[assign_direction_line_hours] Procesando línea %d%s%s",
        id_linea,
        hora_msg,
        dia_msg,
    )

    # Read metadata for this specific line
    metadata_query = f"""
    SELECT * FROM metadata_{source_table_metadata}
    WHERE modo = 'autobus' AND id_linea = {id_linea}
    """
    metadata = ctx.insumos.query(metadata_query)

    if len(metadata) == 0:
        logger.warning(
            "[assign_direction_line_hours] No se encontró metadata " "para línea %d",
            id_linea,
        )
        return pd.DataFrame()

    # Read h3 geoms for the given resolution, filtered by line
    if has_branches:
        # For branches, we need to get all ramales that belong to this line
        ramales_query = f"""
        SELECT DISTINCT {id_col} FROM metadata_{source_table_metadata}
        WHERE modo = 'autobus' AND id_linea = {id_linea}
        """
        ramales_df = ctx.insumos.query(ramales_query)
        if len(ramales_df) == 0:
            logger.warning(
                "[assign_direction_line_hours] No se encontraron ramales "
                "para línea %d",
                id_linea,
            )
            return pd.DataFrame()
        ramales_list = ramales_df[id_col].tolist()
        ramales_str = ",".join(map(str, ramales_list))
        route_filter = f"{id_col} IN ({ramales_str})"
    else:
        route_filter = f"id_linea = {id_linea}"

    if h3_res == 10:
        routes_h3_query = f"""
            SELECT {id_col}, direction, section_id, h3
            FROM official_{source_table_h3}_geoms_h3
            WHERE resolution = {h3_res} AND {route_filter}
        """
    else:
        routes_h3_query = f"""
            SELECT {id_col}, direction, section_id, h3
            FROM official_{source_table_h3}_geoms_h3_parent
            WHERE resolution = {h3_res} AND {route_filter}
        """

    routes_h3 = ctx.insumos.query(routes_h3_query)
    if len(routes_h3) == 0:
        logger.warning(
            "[assign_direction_line_hours] No se encontraron geometrías H3 "
            "para línea %d",
            id_linea,
        )
        return pd.DataFrame()

    routes_h3[id_col] = routes_h3[id_col].astype(int)

    # Prepare routes data for the algorithm
    routes_prepared = prepare_routes_from_h3(
        routes_h3, metadata=metadata, has_branches=has_branches
    )
    logger.info(
        "[assign_direction_line_hours] %d registros de ruta preparados",
        len(routes_prepared),
    )

    # Build the legs query with line and hour filters
    route_ids = routes_prepared[id_col].dropna().unique().tolist()
    if not route_ids:
        logger.warning(
            "[assign_direction_line_hours] No hay route_ids en " "routes_prepared"
        )
        return pd.DataFrame()

    route_ids_str = ",".join(map(str, route_ids))

    # Build WHERE clause with filters
    where_clauses = [
        "od_validado = 1",
        "modo = 'autobus'",
        f"{id_col} IN ({route_ids_str})",
    ]

    if hora_inicio is not None:
        where_clauses.append(f"hora >= {hora_inicio}")
    if hora_fin is not None:
        where_clauses.append(f"hora <= {hora_fin}")

    if dia is not None:
        where_clauses.append(f"dia = '{dia}'")

    where_str = " AND ".join(where_clauses)

    legs_query = f"""
    SELECT id, dia, id_linea, id_ramal, h3_o, h3_d, hora
    FROM etapas
    WHERE {where_str}
    ORDER BY dia, hora, id
    """

    hora_msg = (
        f", horas {hora_inicio}-{hora_fin}"
        if hora_inicio is not None and hora_fin is not None
        else ""
    )
    logger.info(
        "[assign_direction_line_hours] Leyendo etapas para línea %d%s",
        id_linea,
        hora_msg,
    )
    legs_data = ctx.data.query(legs_query)

    if len(legs_data) == 0:
        logger.info(
            "[assign_direction_line_hours] No se encontraron etapas para "
            "los filtros especificados"
        )
        return pd.DataFrame()

    logger.info("[assign_direction_line_hours] Procesando %d etapas", len(legs_data))

    # Delete existing records for this line and hour range from the table
    # Build the WHERE clause
    where_parts = [f"id_linea = {id_linea}"]

    if hora_inicio is not None:
        where_parts.append(f"hora >= {hora_inicio}")
    if hora_fin is not None:
        where_parts.append(f"hora <= {hora_fin}")
    if dia is not None:
        where_parts.append(f"dia = '{dia}'")

    where_clause = " AND ".join(where_parts)

    delete_query = f"""
    DELETE FROM legs_direction_branch_line
    WHERE {where_clause}
    """

    try:
        ctx.data.execute(delete_query)
        hora_msg = (
            f", horas {hora_inicio}-{hora_fin}"
            if hora_inicio is not None and hora_fin is not None
            else ""
        )
        logger.info(
            "[assign_direction_line_hours] Registros previos eliminados para "
            "línea %d%s",
            id_linea,
            hora_msg,
        )
    except Exception as e:
        logger.debug("[assign_direction_line_hours] DELETE omitido: %s", e)

    # Process legs using the existing parallel processing function
    legs_with_direction = _process_legs_direction_parallel(
        legs_data=legs_data, routes_prepared=routes_prepared
    )

    if len(legs_with_direction) > 0:
        # Ensure dia and id_linea columns are present
        cols_to_merge = []
        if "dia" not in legs_with_direction.columns:
            cols_to_merge.append("dia")
        if "id_linea" not in legs_with_direction.columns:
            cols_to_merge.append("id_linea")

        if cols_to_merge:
            # Map missing columns from the original legs_data using id
            merge_cols = ["id"] + cols_to_merge
            legs_with_direction = legs_with_direction.merge(
                legs_data[merge_cols], on="id", how="left"
            )

        # Select only columns that match the table schema
        table_cols = [
            "id",
            "dia",
            "id_linea",
            "id_ramal",
            "hora",
            "direction_inferred",
            "confidence",
            "selected_section_id_o",
            "selected_section_id_d",
            "possible_branches",
            "selected_branch",
        ]
        # Only keep columns that exist in the DataFrame
        cols_to_save = [c for c in table_cols if c in legs_with_direction.columns]
        legs_to_save = legs_with_direction[cols_to_save].copy()

        # Append results to the table
        ctx.data.append_raw(legs_to_save, "legs_direction_branch_line")

        pct = len(legs_with_direction) / len(legs_data) * 100
        logger.info(
            "[assign_direction_line_hours] Completado: %d/%d etapas con "
            "dirección asignada (%.1f%%) y guardadas en tabla",
            len(legs_with_direction),
            len(legs_data),
            pct,
        )
    else:
        logger.warning(
            "[assign_direction_line_hours] No se pudo asignar dirección a "
            "ninguna etapa"
        )

    return legs_with_direction


def compute_section_usage(
    legs_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    metadata: pd.DataFrame,
    h3_resolution: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute how many legs traversed each section for each branch/direction.

    Takes legs with identified direction and expands each leg to all
    section_ids traversed (from selected_section_id_o to selected_section_id_d,
    inclusive). Then aggregates to count how many legs used each section.

    Parameters
    ----------
    legs_df : pd.DataFrame
        DataFrame with legs containing at minimum:
        - selected_section_id_o: origin section ID (at default resolution)
        - selected_section_id_d: destination section ID (at default resolution)
        - direction_inferred: identified direction (0 or 1)
        - selected_branch or id_linea: route identifier
        Optional columns:
        - factor_expansion_linea or factor_expansion_etapa: expansion factors
        Only required when h3_resolution differs from default resolution:
        - h3_o: origin H3 cell (for resolution recomputation)
        - h3_d: destination H3 cell (for resolution recomputation)
    h3_resolution : int, optional
        Target H3 resolution for section computation. If None or equal to
        the default resolution from config, uses existing section_ids. If
        different from default, recomputes section_ids at this resolution
        by getting parent H3 cells and matching to routes (requires h3_o
        and h3_d columns).
    routes_df : pd.DataFrame
        Routes data at the target h3_resolution. Required.
        Should contain columns: h3, direction, section_id, and
        either id_ramal or id_linea depending on has_branches.
    metadata : pd.DataFrame
        Metadata for routes, used when preparing routes for matching.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - selected_branch or id_linea: route identifier
        - direction_inferred: direction (0 or 1)
        - section_id: section identifier
        - n_legs: count of legs using this section
        - n_legs_expanded: sum of expansion factors (if available)

    Examples
    --------
    >>> # Use existing section_ids from legs (at default resolution)
    >>> routes_h3 = ctx.insumos.query(
    ...     "SELECT * FROM official_branches_geoms_h3"
    ... )
    >>> section_usage = compute_section_usage(
    ...     legs_with_dir, routes_df=routes_h3, metadata=metadata
    ... )
    >>>
    >>> # Recompute at parent resolution 8 (requires h3_o and h3_d)
    >>> routes_h3_res8 = ctx.insumos.query(
    ...     "SELECT * FROM official_branches_geoms_h3_parent "
    ...     "WHERE resolution = 8"
    ... )
    >>> section_usage_res8 = compute_section_usage(
    ...     legs_with_dir,
    ...     routes_df=routes_h3_res8,
    ...     metadata=metadata,
    ...     h3_resolution=8
    ... )
    """
    # Read configuration to determine if branches are used
    configs = leer_configs_generales(autogenerado=False)
    has_branches = configs.get("lineas_contienen_ramales", False)
    if h3_resolution is None:
        h3_resolution = configs["resolucion_h3"]
        logger.info("Using default H3 resolution from config: %d", h3_resolution)

    # check h3_resolution is not none and not higher than configs['resolucion_h3']
    if h3_resolution > configs["resolucion_h3"]:
        raise ValueError(
            f"h3_resolution {h3_resolution} no puede ser mayor a la usada para georeferenciar etapas en configs: {configs['resolucion_h3']}"
        )
    if routes_df is None:
        raise ValueError("routes_df tiene que estar presente")

    # Determine route identifier column based on configuration
    if has_branches:
        route_col = "selected_branch"
    else:
        route_col = "id_linea"

    # Filter to legs with valid section IDs
    valid_legs = legs_df[
        legs_df["selected_section_id_o"].notna()
        & legs_df["selected_section_id_d"].notna()
        & legs_df["direction_inferred"].notna()
    ].copy()

    if len(valid_legs) == 0:
        logger.warning("No legs with valid section IDs found")
        return pd.DataFrame()

    # Verify required column exists
    if route_col not in valid_legs.columns:
        raise ValueError(
            f"DataFrame must contain '{route_col}' column "
            f"(has_branches={has_branches})"
        )

    # Check if we need to recompute section_ids at a different resolution
    default_resolution = configs["resolucion_h3"]
    needs_recomputation = h3_resolution != default_resolution

    if needs_recomputation:
        # Check required columns for recomputation
        if "h3_o" not in valid_legs.columns or "h3_d" not in valid_legs.columns:
            raise ValueError(
                "h3_o and h3_d columns required when recomputing "
                "section_ids at different resolution. "
                "If loading from legs_direction_branch_line table, "
                "JOIN with etapas to get these columns: "
                "SELECT ldbl.*, e.h3_o, e.h3_d FROM legs_direction_branch_line ldbl "
                "LEFT JOIN etapas e ON ldbl.id = e.id AND ldbl.dia = e.dia"
            )

        # Prepare routes for matching
        routes_prepared = prepare_routes_from_h3(
            routes_df, metadata=metadata, has_branches=has_branches
        )

        # Determine route filter column
        if route_col == "selected_branch":
            route_filter_col = "id_ramal"
        else:
            route_filter_col = "id_linea"

        # Vectorized computation of parent H3 cells
        valid_legs["h3_o_parent"] = valid_legs["h3_o"].apply(
            lambda x: h3.cell_to_parent(x, h3_resolution)
        )
        valid_legs["h3_d_parent"] = valid_legs["h3_d"].apply(
            lambda x: h3.cell_to_parent(x, h3_resolution)
        )

        # Prepare routes with just the columns we need for joining
        routes_for_join = routes_prepared[
            ["h3", route_filter_col, "direction", "section_id"]
        ].copy()

        # Join for origin: match h3_o_parent with routes h3
        # We need to match on h3, route_id, and direction
        legs_with_o = valid_legs.merge(
            routes_for_join,
            left_on=["h3_o_parent", route_col, "direction_inferred"],
            right_on=["h3", route_filter_col, "direction"],
            how="left",
            suffixes=("", "_route_o"),
        )

        # If there are duplicates (same h3 has multiple section_ids),
        # keep max section_id for origin
        legs_with_o = (
            legs_with_o.groupby(legs_with_o.index)
            .agg({"section_id": "max"})
            .rename(columns={"section_id": "new_section_id_o"})
        )

        # Join for destination: match h3_d_parent with routes h3
        legs_with_d = valid_legs.merge(
            routes_for_join,
            left_on=["h3_d_parent", route_col, "direction_inferred"],
            right_on=["h3", route_filter_col, "direction"],
            how="left",
            suffixes=("", "_route_d"),
        )

        # If there are duplicates, keep min section_id for destination
        legs_with_d = (
            legs_with_d.groupby(legs_with_d.index)
            .agg({"section_id": "min"})
            .rename(columns={"section_id": "new_section_id_d"})
        )

        # Update valid_legs with new section_ids
        valid_legs["selected_section_id_o"] = legs_with_o["new_section_id_o"]
        valid_legs["selected_section_id_d"] = legs_with_d["new_section_id_d"]

        # Clean up temporary columns
        valid_legs = valid_legs.drop(columns=["h3_o_parent", "h3_d_parent"])

        # Filter out legs where section_ids couldn't be recomputed
        valid_legs = valid_legs[
            valid_legs["selected_section_id_o"].notna()
            & valid_legs["selected_section_id_d"].notna()
        ].copy()

        logger.info(
            "Recomputed section_ids: %d legs have valid section_ids at resolution %d",
            len(valid_legs),
            h3_resolution,
        )
    else:
        # Use existing section_ids (already at the correct resolution)
        logger.info(
            "Using existing section_ids at default resolution %d: %d legs",
            h3_resolution,
            len(valid_legs),
        )

    if len(valid_legs) == 0:
        logger.warning("No legs with valid section_ids at target resolution")
        return pd.DataFrame()

    # Check for expansion factor
    if "factor_expansion_linea" in valid_legs.columns:
        expansion_col = "factor_expansion_linea"
    elif "factor_expansion_etapa" in valid_legs.columns:
        expansion_col = "factor_expansion_etapa"
    else:
        expansion_col = None

    # Expand each leg to all sections traversed
    expanded_rows = []

    for _, leg in valid_legs.iterrows():
        section_o = int(leg["selected_section_id_o"])
        section_d = int(leg["selected_section_id_d"])
        route_id = leg[route_col]
        direction = int(leg["direction_inferred"])

        # Generate range of sections (always from lower to higher)
        if section_o <= section_d:
            sections = range(section_o, section_d + 1)
        else:
            # Backward traversal (shouldn't happen often with correct direction)
            sections = range(section_d, section_o + 1)

        # Get expansion factor if available
        if expansion_col:
            expansion_factor = leg.get(expansion_col, 1.0)
            if pd.isna(expansion_factor):
                expansion_factor = 1.0
        else:
            expansion_factor = 1.0

        # Create a row for each section traversed
        for section_id in sections:
            expanded_rows.append(
                {
                    route_col: route_id,
                    "direction_inferred": direction,
                    "section_id": section_id,
                    "expansion_factor": expansion_factor,
                }
            )

    if not expanded_rows:
        logger.warning("No sections could be expanded from legs")
        return pd.DataFrame()

    # Create DataFrame from expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    # Aggregate by route, direction, and section
    group_cols = [route_col, "direction_inferred", "section_id"]

    section_usage = (
        expanded_df.groupby(group_cols, as_index=False)
        .agg(
            n_legs=("section_id", "count"),
            n_legs_expanded=("expansion_factor", "sum"),
        )
        .sort_values(group_cols)
    )

    # Round n_legs_expanded to avoid excessive decimal places
    section_usage["n_legs_expanded"] = section_usage["n_legs_expanded"].round(2)

    logger.info(
        "Computed section usage: %d unique (route, direction, section) "
        "combinations from %d legs",
        len(section_usage),
        len(valid_legs),
    )

    return section_usage
