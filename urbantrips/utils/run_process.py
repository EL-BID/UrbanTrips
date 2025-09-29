import os
from pathlib import Path
from urbantrips.utils.check_configs import check_config
from urbantrips.carto.routes import process_routes_metadata, process_routes_geoms
from urbantrips.carto.stops import create_stops_table
from urbantrips.carto.carto import guardo_zonificaciones

from urbantrips.utils.utils import (
    create_insumos_general_dbs,
    create_data_dash_dbs,
    create_directories,
    levanto_tabla_sql,
)


from urbantrips.datamodel import legs, trips
from urbantrips.datamodel import transactions as trx
from urbantrips.datamodel import services
from urbantrips.destinations import destinations as dest
from urbantrips.geo import geo
from urbantrips.carto import carto, routes
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config
from urbantrips.kpi.kpi import compute_kpi
from urbantrips.datamodel.misc import persist_datamodel_tables
from urbantrips.preparo_dashboard.preparo_dashboard import preparo_indicadores_dash


def inicializo_ambiente():

    corridas_nuevas = []
    # Leer las corridas en el archivo de configuracion
    configs_usuario = utils.leer_configs_generales(autogenerado=False)
    corridas = configs_usuario.get("corridas", None)

    if corridas is None or len(corridas) == 0:
        print("No se han definido corridas en el archivo de configuracion.")
        raise ValueError("No se han definido corridas en el archivo de configuracion.")

    path_insumos = configs_usuario["alias_db"]
    path_insumos = Path() / "data" / "db" / f"{path_insumos}_insumos.sqlite"

    if not path_insumos.is_file():
        print("Inicializo ambiente por primera vez")

        # chequear consistencia de configuracion
        check_config(corridas[0])

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
        path_data = Path() / "data" / "db" / f"{alias_db}_data.sqlite"

        # corridas_nuevas = []
        if not path_data.is_file():
            # Crear una tabla por corrida de data y dash:
            create_data_dash_dbs(alias_db)
            corridas_nuevas += [alias_db]

    return corridas_nuevas


def procesar_transacciones(corrida):
    # Chequear consistencia y crear configuracion
    check_config(corrida)

    # Read config file
    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]

    # trx configs
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]

    # gps configs
    nombre_archivo_gps = configs["nombre_archivo_gps"]
    nombres_variables_gps = configs["nombres_variables_gps"]
    tiempos_viaje_estaciones = configs["tiempos_viaje_estaciones"]

    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    resolucion_h3 = configs["resolucion_h3"]
    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }

    # Compute tolerance in h3 ring
    ring_size = geo.get_h3_buffer_ring_size(resolucion_h3, tolerancia_parada_destino)

    # Produce transaction table
    trx.create_transactions(
        geolocalizar_trx_config,
        nombre_archivo_trx,
        nombres_variables_trx,
        formato_fecha,
        col_hora,
        tipo_trx_invalidas,
        nombre_archivo_gps,
        nombres_variables_gps,
    )

    # Turn transactions into legs
    legs.create_legs_from_transactions(trx_order_params)

    # Update destination validation matrix
    carto.update_stations_catchment_area(ring_size=ring_size)

    # Infer legs destinations
    dest.infer_destinations()

    # Create distances table
    carto.create_distances_table(use_parallel=False)

    if nombre_archivo_gps is not None:
        services.process_services(line_ids=None)

        # Assign a gps point id to legs' origins
        legs.assign_gps_origin()

        # Assign a gps point id to legs' destination
        legs.assign_gps_destination()

    if tiempos_viaje_estaciones is not None:
        # Assign stations to legs for travel times
        legs.assign_stations_od()

    # Add distances and travel times to legs
    legs.add_distance_and_travel_time()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

    # compute travel time for trips
    trips.compute_trips_travel_time()

    trips.add_distance_and_travel_time()

    # Inferir route geometries based on legs data
    routes.infer_routes_geoms()

    # Build final routes from official an inferred sources
    routes.build_routes_from_official_inferred()

    # write information about transactions in the database
    trx.write_transactions_to_db(corrida)

    # Compute KPI
    compute_kpi()

    persist_datamodel_tables()


def borrar_corridas(alias_db="all"):

    corridas_nuevas = []
    # Leer las corridas en el archivo de configuracion
    configs_usuario = utils.leer_configs_generales(autogenerado=False)
    corridas = configs_usuario.get("corridas", None)

    if corridas is None or len(corridas) == 0:
        raise ValueError("No se han definido corridas en el archivo de configuracion.")

    if len(alias_db) > 0:
        path_ = configs_usuario["alias_db"]
        path_insumos = Path() / "data" / "db" / f"{path_}_insumos.sqlite"
        path_general = Path() / "data" / "db" / f"{path_}_general.sqlite"

        if alias_db == "all":
            if path_insumos.exists():
                path_insumos.unlink()
                print(f"Se borró {path_insumos}")
            if path_general.exists():
                path_general.unlink()
                print(f"Se borró {path_general}")

            for i in corridas:
                path_data = Path() / "data" / "db" / f"{i}_data.sqlite"
                path_dash = Path() / "data" / "db" / f"{i}_dash.sqlite"
                if path_data.exists():
                    path_data.unlink()
                    print(f"Se borró {path_data}")
                if path_dash.exists():
                    path_dash.unlink()
                    print(f"Se borró {path_dash}")
        else:
            path_data = Path() / "data" / "db" / f"{alias_db}_data.sqlite"
            path_dash = Path() / "data" / "db" / f"{alias_db}_dash.sqlite"
            if path_data.exists():
                path_data.unlink()
                print(f"Se borró {path_data}")
            if path_dash.exists():
                path_dash.unlink()
                print(f"Se borró {path_dash}")

        corridas_anteriores = levanto_tabla_sql(
            "corridas",
            "general",
            query="SELECT DISTINCT corrida FROM corridas WHERE process = 'transactions_completed'",
        )

        if len(corridas_anteriores) > 0:
            corridas_anteriores = corridas_anteriores.corrida.values.tolist()
        else:
            corridas_anteriores = []

        corridas = [c for c in corridas if c not in corridas_anteriores]

        for i in corridas:
            path_data = Path() / "data" / "db" / f"{i}_data.sqlite"
            path_dash = Path() / "data" / "db" / f"{i}_dash.sqlite"
            if path_data.exists():
                path_data.unlink()
                print(f"Se borró {path_data}")
            if path_dash.exists():
                path_dash.unlink()
                print(f"Se borró {path_dash}")


def run_all(borrar_corrida="", crear_dashboard=True):

    print(f"[INFO] borrar_corrida = '{borrar_corrida}'")
    print(f"[INFO] crear_dashboard = {crear_dashboard}")

    borrar_corridas(borrar_corrida)

    corridas = inicializo_ambiente()
    print("Se procesarán estas corridas:", corridas)
    for corrida in corridas:
        print(f"Procesando corrida: {corrida}")
        procesar_transacciones(corrida)
        if crear_dashboard:
            preparo_indicadores_dash(corrida)
