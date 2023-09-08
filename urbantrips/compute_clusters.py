from urbantrips.cluster import dbscan
from urbantrips.geo.geo import get_epsg_m


def main():
    day_type = 'weekday'
    epsg_m = get_epsg_m()

    id_linea = 1
    rango_hrs = [7, 9]
    legs, route_geom = dbscan.get_legs_and_route_geoms(
        id_linea, rango_hrs, day_type)

    # 4D
    clustered_legs_d0, clustered_legs_d1 = dbscan.cluster_legs_4d(
        legs, route_geom)

    dbscan.plot_cluster_legs_4d(clustered_legs_d0=clustered_legs_d0,
                                clustered_legs_d1=clustered_legs_d1,
                                route_geom=route_geom,
                                epsg_m=epsg_m,
                                id_linea=id_linea,
                                rango_hrs=rango_hrs,
                                day_type=day_type,
                                factor=2)

    # LRS
    clustered_legs_d0, clustered_legs_d1 = dbscan.cluster_legs_lrs(
        legs, route_geom)
    dbscan.plot_cluster_legs_lrs(
        clustered_legs_d0, clustered_legs_d1, id_linea, rango_hrs, day_type)


if __name__ == "__main__":
    main()
