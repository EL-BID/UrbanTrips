from urbantrips.datamodel import legs, trips
from urbantrips.datamodel import transactions as trx
from urbantrips.destinations import destinations as dest
from urbantrips.geo import geo
from urbantrips.carto import carto, routes
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config


def main():

    # Check config file consistency
    check_config()

    # Read config file
    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]

    # trx configs
    nombres_variables_trx = configs["nombres_variables_trx"]
    formato_fecha = configs["formato_fecha"]
    col_hora = configs["columna_hora"]
    tipo_trx_invalidas = configs["tipo_trx_invalidas"]
    nombre_archivo_trx = configs["nombre_archivo_trx"]

    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    resolucion_h3 = configs["resolucion_h3"]
    trx_order_params = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }

    # gps configs
    if geolocalizar_trx_config:
        nombre_archivo_gps = configs["nombre_archivo_gps"]
        nombres_variables_gps = configs["nombres_variables_gps"]
    else:
        nombre_archivo_gps = None
        nombres_variables_gps = None

    # Compute tolerance in h3 ring
    ring_size = geo.get_h3_buffer_ring_size(
        resolucion_h3, tolerancia_parada_destino
    )

    # Produce transaction table
    trx.create_transactions(geolocalizar_trx_config,
                            nombre_archivo_trx,
                            nombres_variables_trx,
                            formato_fecha,
                            col_hora,
                            tipo_trx_invalidas,
                            nombre_archivo_gps,
                            nombres_variables_gps)

    # Turn transactions into legs
    legs.create_legs_from_transactions(trx_order_params)

    # Update destination validation matrix
    carto.update_stations_catchment_area(ring_size=ring_size)

    # Infer legs destinations
    dest.infer_destinations()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

    # Inferir route geometries based on legs data
    routes.infer_routes_geoms(plotear_lineas=False)

    # Build final routes from official an inferred sources
    routes.build_routes_from_official_inferred()


if __name__ == "__main__":
    main()
