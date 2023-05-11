from urbantrips.utils import utils
from urbantrips.carto import routes


def main():
    # Check config file consistency
    utils.check_config()

    # Create basic dir structure:
    utils.create_directories()

    # Create DB:
    utils.create_db()

    # Process routes info and upload to db
    routes.process_routes_metadata()

    # Process and upload route geometries
    routes.process_routes_geoms()


if __name__ == "__main__":
    main()
