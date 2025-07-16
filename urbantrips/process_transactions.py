import argparse
from urbantrips.datamodel import legs, trips
from urbantrips.datamodel import transactions as trx
from urbantrips.datamodel import services
from urbantrips.destinations import destinations as dest
from urbantrips.geo import geo
from urbantrips.carto import carto, routes
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config


def main(args):
    # Obtener el parametro de corrida
    corrida = args.corrida

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

    # TODO: remove legs with origin far away from any station
    # when stations or lines exists

    # Infer legs destinations
    dest.infer_destinations()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

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

    # compute travel time for trips
    trips.compute_trips_travel_time()

    # Inferir route geometries based on legs data
    routes.infer_routes_geoms()

    # Build final routes from official an inferred sources
    routes.build_routes_from_official_inferred()

    # write information about transactions in the database
    trx.write_transactions_to_db(corrida)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize UrbanTrips environment.")
    parser.add_argument("--corrida", type=str, required=True, help="Corrida identifier")
    args = parser.parse_args()
    main(args)
