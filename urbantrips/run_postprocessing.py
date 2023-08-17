from urbantrips.datamodel.misc import persist_datamodel_tables
from urbantrips.kpi import kpi
from urbantrips.viz import viz
from urbantrips.carto import carto
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config


def main():
    check_config()

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
