import os


def create_directories():
    """Creates the standard UrbanTrips directory structure in the current working dir."""
    dirs = [
        os.path.join("data", "db"),
        os.path.join("data", "data_ciudad"),
        "configs",
        os.path.join("resultados", "tablas"),
        os.path.join("resultados", "png"),
        "docs",
        os.path.join("resultados", "pdf"),
        os.path.join("resultados", "matrices"),
        os.path.join("resultados", "data"),
        os.path.join("resultados", "html"),
        os.path.join("resultados", "geojson"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
