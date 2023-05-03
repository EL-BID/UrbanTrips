from urbantrips.datamodel.misc import persist_datamodel_tables

from urbantrips.viz import viz
from urbantrips.utils import utils


def main():
    utils.check_config()

    # Poduce main viz
    viz.create_visualizations()



if __name__ == "__main__":
    main()