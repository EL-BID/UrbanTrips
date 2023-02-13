from urbantrips.datamodel import legs, misc, legs, trips
from urbantrips.datamodel import transactions as trx
from urbantrips.kpi import kpi
from urbantrips.destinations import destinations as dest
from urbantrips.viz import viz
from urbantrips.geo import geo
from urbantrips.carto import carto
from urbantrips.utils import utils


def main():
    # Check config file consistency
    utils.check_config()

    # Read config file
    configs = utils.leer_configs_generales()
    geolocalizar_trx_config = configs["geolocalizar_trx"]

    tolerancia_parada_destino = configs["tolerancia_parada_destino"]
    resolucion_h3 = configs["resolucion_h3"]
    criterio_orden_transacciones = {
        "criterio": configs["ordenamiento_transacciones"],
        "ventana_viajes": configs["ventana_viajes"],
        "ventana_duplicado": configs["ventana_duplicado"],
    }

    # Compute tolerance in h3 ring
    ring_size = geo.get_h3_buffer_ring_size(
        resolucion_h3, tolerancia_parada_destino
    )

    # Create basic dir structure:
    utils.crear_directorios()

    # Create DB:
    utils.crear_base()
    misc.create_line_and_branches_metadata()

    # Produce transaction table
    trx.create_transactions(geolocalizar_trx_config)

    # Turn transactions into legs
    legs.create_legs_from_transactions(criterio_orden_transacciones)

    # Update destination validation matrix
    carto.update_stations_catchment_area(ring_size=ring_size)

    # Infer legs destinations
    dest.infer_destinations()

    # Fix trips with same OD
    trips.rearrange_trip_id_same_od()

    # Produce trips and users tables from legs
    trips.create_trips_from_legs()

    # Upload route geometries
    carto.upload_routes_geoms()

    # Inferir route geometries based on legs data
    carto.infer_routes_geoms(plotear_lineas=False)

    # Compute and viz route section load by line
    kpi.compute_route_section_load(id_linea=False, rango_hrs=False)
    viz.visualize_route_section_load(id_linea=False, rango_hrs=False)

    # Create TAZs
    carto.create_zones_table()

    # Create voronoi TAZs
    carto.create_voronoi_zones()

    # Create distances table
    carto.create_distances_table(use_parallel=True)

    # Persist datamodel into csv tables
    misc.persist_datamodel_tables()

    # Poduce main viz
    viz.create_visualizations()

    # Compute KPI
    kpi.compute_kpi()


if __name__ == "__main__":
    main()
