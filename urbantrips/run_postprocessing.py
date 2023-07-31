from urbantrips.datamodel.misc import persist_datamodel_tables
from urbantrips.kpi import kpi
from urbantrips.viz import viz
from urbantrips.carto import carto
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config


def main():
    check_config()

    # Compute and viz route section load by line
    kpi.compute_route_section_load(id_linea=False, rango_hrs=False)
    viz.visualize_route_section_load(
        id_linea=False, rango_hrs=False,
        save_gdf=True, indicador='prop_etapas', factor=500,
        factor_min=50, )

    # Create TAZs
    carto.create_zones_table()

    # Create voronoi TAZs
    carto.create_voronoi_zones()

    # Create distances table
    carto.create_distances_table(use_parallel=True)

    # Persist datamodel into csv tables
    persist_datamodel_tables()

    # Compute KPI
    kpi.compute_kpi()


if __name__ == "__main__":
    main()
