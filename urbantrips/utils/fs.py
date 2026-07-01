import os
from pathlib import Path


def create_directories():
    """Creates the standard UrbanTrips directory structure under the active base dir."""
    from urbantrips.utils.paths import get_paths
    p = get_paths()
    dirs = [
        p.db_dir,
        p.input_dir,
        p.configs_dir,
        p.base / "docs",
        p.output_dir / "tablas",
        p.output_dir / "png",
        p.output_dir / "pdf",
        p.output_dir / "matrices",
        p.output_dir / "data",
        p.output_dir / "html",
        p.output_dir / "geojson",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
