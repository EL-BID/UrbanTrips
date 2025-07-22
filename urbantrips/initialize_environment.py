import os
from urbantrips.utils.check_configs import check_config, replace_tabs_with_spaces
from urbantrips.carto.routes import process_routes_metadata, process_routes_geoms
from urbantrips.carto.stops import create_stops_table
from urbantrips.carto.carto import guardo_zonificaciones
from urbantrips.utils.utils import (
    create_insumos_general_dbs,
    create_data_dash_dbs,
    create_directories,
    leer_configs_generales,
)


def main():

    # Leer las corridas en el archivo de configuracion
    replace_tabs_with_spaces(os.path.join("configs", "configuraciones_generales.yaml"))

    configs_usuario = leer_configs_generales(autogenerado=False)
    print(configs_usuario)
    corridas = configs_usuario.get("corridas", None)
    if corridas is None or len(corridas) == 0:
        raise ValueError("No se han definido corridas en el archivo de configuracion.")

    # chequear consistencia de configuracion
    check_config(corridas[0])
    # Remove file created by check_config if it exists
    file_path = os.path.join("configs", "configuraciones_generales_autogenerado.yaml")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed config file: {file_path}")

    # Crear directorios basicos de trabajo:
    create_directories()

    # Crear una base de datos para insumos y general
    create_insumos_general_dbs()

    # Procesar metadata de rutas
    process_routes_metadata()

    # Procesar y subir geometrías de rutas
    process_routes_geoms()

    # Crear tabla de paradas
    create_stops_table()

    # Guarda zonificaciones
    guardo_zonificaciones()

    for alias_db in corridas:
        # Crear una tabla por corrida de data y dash:
        create_data_dash_dbs(alias_db)


if __name__ == "__main__":
    main()
