from urbantrips.utils.check_configs import check_config
from urbantrips.carto import routes, stops
from urbantrips.utils import utils


def main():
    # Create basic dir structure:
    utils.create_directories()

    # Check config file consistency
    check_config()

    # Create DB:
    utils.create_db()

    # Process routes info and upload to db
    routes.process_routes_metadata()

    # Process and upload route geometries
    routes.process_routes_geoms()

    # Create stops table
    stops.create_stops_table()


if __name__ == "__main__":
    main()
