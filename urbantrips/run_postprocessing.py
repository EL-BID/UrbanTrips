from urbantrips.datamodel.misc import persist_datamodel_tables
from urbantrips.kpi import kpi
from urbantrips.carto import carto


def main():

    # Persist datamodel into csv tables
    persist_datamodel_tables()

    # Compute KPI
    kpi.compute_kpi()


if __name__ == "__main__":
    main()
