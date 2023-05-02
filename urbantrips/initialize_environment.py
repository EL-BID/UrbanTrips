from urbantrips.datamodel.misc import create_line_and_branches_metadata
from urbantrips.utils import utils


def main():
    # Check config file consistency
    utils.check_config()

    # Create basic dir structure:
    utils.create_directories()

    # Create DB:
    utils.create_db()
    create_line_and_branches_metadata()


if __name__ == "__main__":
    main()
