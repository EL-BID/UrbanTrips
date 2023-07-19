from urbantrips.datamodel.misc import persist_datamodel_tables

from urbantrips.viz import viz
from urbantrips.viz_ppt_utils import viz_ppt_utils
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config

def main():
    check_config()

    # Prduce main viz
    viz.create_visualizations()

    # Produce ppt
    viz_ppt_utils.create_ppt()


if __name__ == "__main__":
    main()
