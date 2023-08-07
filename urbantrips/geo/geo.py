from shapely import line_interpolate_point
import geopandas as gpd
import numpy as np
import pandas as pd
from urbantrips.utils.utils import leer_configs_generales
from itertools import repeat
import h3
from math import ceil
from shapely.geometry import Polygon, Point, LineString,  LinearRing
import libpysal
import statsmodels.api as sm


def referenciar_h3(df, res, nombre_h3, lat='latitud', lon='longitud'):
    """
    Esta funcion toma un DF con latitud y longitud y georeferencia
    sus coordenadas en grillas hexagonales h3
    """
    out = list(map(h3.geo_to_h3, df[lat], df[lon], repeat(res)))
    df[nombre_h3] = out
    return df


def h3_from_row(row, res, lat, lng):
    """
    Esta funcion toma una fila, un nivel de resolucion de h3
    y los nombres que contienen las coordenadas xy
    y devuelve un id de indice h3
    """

    return h3.geo_to_h3(row[lat], row[lng], resolution=res)


def get_h3_buffer_ring_size(resolucion_h3, buffer_meters):
    """
    Esta funcion toma una resolucion h3 y una tolerancia en metros
    y calcula la cantidad de h3 tolerancia en para alcanzar esa tolerancia
    """
    lado = round(h3.edge_length(resolution=resolucion_h3, unit="m"))

    if buffer_meters < lado:
        ring_size = 0
        buff_max = lado
    else:
        ring_size = ceil(buffer_meters / lado / 2)
        buff_max = (lado*2*ring_size) + lado
    print(f"Se utilizarán hexágonos con un lado igual a {round(lado)} m. ")
    print(
        f"Para la matriz de validacion se usará un buffer de {ring_size}" +
        " hexágonos.")
    print(
        "Se utilizará para la matriz de validacion una distancia máxima de " +
        f"{buff_max} m entre el origen de la etapa siguiente y las " +
        "estaciones de la línea de la etapa a validar")
    print("Si desea mayor precisión utilice un número más grande de " +
          "resolucion h3")

    return ring_size


def get_stop_hex_ring(h, ring_size):
    """
    This functions takes a h3 index referencing a public transit stop
    a h3 ring size, and returns a DataFrame with that stops and all the
    hexs within that ring
    """
    rings = list(h3.k_ring(h, ring_size))
    df = pd.DataFrame({"parada": [h] * (len(rings)), "area_influencia": rings})
    return df


def h3togeo(x):
    try:
        result = str(h3.h3_to_geo(x)[0]) + ", " + str(h3.h3_to_geo(x)[1])            
    except (TypeError, ValueError):
        result = ''
    return result


def h3dist(x, distancia_entre_hex=1, h3_o='', h3_d=''):
    if len(h3_o) == 0:
        h3_o = 'h3_o'
    if len(h3_d) == 0:
        h3_d = 'h3_d'

    try:
        x = round(h3.h3_distance(x[h3_o], x[h3_d]) * distancia_entre_hex, 2)
    # except (H3CellError, TypeError) as e:
    except (TypeError) as e:
        print(e)
        x = np.nan
    return x


def add_geometry(row, bring='polygon'):
    '''
    Devuelve una tupla de pares lat/lng que describen el polígono de la celda.
    Llama a la función h3_to_geo_boundary de la librería h3.

    Parámetros:
    row = código del hexágono en formato h3
    bring = define si devuelve un polígono, latitud o longitud

    Salida: geometry resultado
    '''
    points = h3.h3_to_geo_boundary(row, True)

    points = Polygon(points)
    if bring == 'lat':
        points = points.representative_point().y
    if bring == 'lon':
        points = points.representative_point().x

    return points


def create_voronoi(centroids, var_zona='Zona'):
    xmin = centroids.geometry.x.min()-.1
    xmax = centroids.geometry.x.max()+.1
    ymin = centroids.geometry.y.min()-.1
    ymax = centroids.geometry.y.max()+.1

    poly = Polygon(LinearRing([Point(xmin, ymin),
                               Point(xmin, ymax),
                               Point(xmax, ymax),
                               Point(xmax, ymin)]))

    # Extract the coordinates into a numpy array
    x_coords = centroids.geometry.x
    y_coords = centroids.geometry.y
    coords = np.dstack((x_coords, y_coords))

    regions_df, _ = libpysal.cg.voronoi.voronoi_frames(coords[0], clip=poly)

    regions_df = regions_df.reset_index()
    regions_df.columns = [var_zona, 'geometry']
    regions_df[var_zona] = str(regions_df[var_zona]+1)

    regions_df = regions_df.set_crs("EPSG:4326")

    return regions_df


def bring_latlon(x, latlon='lat'):
    if latlon == 'lat':
        posi = 0
    if latlon == 'lon':
        posi = 1
    try:
        result = float(x.split(',')[posi])
    except (AttributeError, IndexError): 
        result = 0
    return result

def normalizo_lat_lon(df,
                      h3_o='h3_o',
                      h3_d='h3_d',
                      origen='',
                      destino=''):
    
    if len(origen) == 0:
        origen = h3_o
    if len(destino) == 0:
        destino = h3_d

    df["origin"] = df[h3_o].apply(h3togeo)
    df['lon_o_tmp'] = df["origin"].apply(bring_latlon, latlon='lon')
    df['lat_o_tmp'] = df["origin"].apply(bring_latlon, latlon='lat')
       
    df["destination"] = df[h3_d].apply(h3togeo)        
    df['lon_d_tmp'] = df["destination"].apply(bring_latlon, latlon='lon')
    df['lat_d_tmp'] = df["destination"].apply(bring_latlon, latlon='lat')

    if 'h3_' not in origen:
        cols = {destino: origen, 'lat_d_tmp': 'lat_o_tmp',
                'lon_d_tmp': 'lon_o_tmp'}
        zonif = pd.concat(
            [df[[origen, 'lat_o_tmp', 'lon_o_tmp']],
             df[[destino, 'lat_d_tmp', 'lon_d_tmp']].rename(columns=cols)],
            ignore_index=True)
        zonif = zonif.groupby(origen, as_index=False).agg(
            {'lat_o_tmp': 'mean', 'lon_o_tmp': 'mean'})

        df = df.drop(['lat_o_tmp', 'lon_o_tmp',
                     'lat_d_tmp', 'lon_d_tmp'], axis=1)

        df = df.merge(
            zonif,
            how='left',
            on=origen
        )
        cols = {origen: destino, 'lat_o_tmp': 'lat_d_tmp',
                'lon_o_tmp': 'lon_d_tmp'}
        df = df.merge(
            zonif.rename(
                columns=cols),
            how='left',
            on=destino
        )

    if f"{origen}_norm" in df.columns:
        df = df.drop([f"{origen}_norm", f"{destino}_norm"], axis=1)

    df["dist_y"] = (
        df[['lat_o_tmp', 'lat_d_tmp']].max(axis=1).values
        - df[['lat_o_tmp', 'lat_d_tmp']].min(axis=1).values
    )
    df["dist_x"] = (
        df[['lon_o_tmp', 'lon_d_tmp']].max(axis=1).values
        - df[['lon_o_tmp', 'lon_d_tmp']].min(axis=1).values
    )

    df["dif_y"] = df['lat_o_tmp'] - df['lat_d_tmp']
    df["dif_x"] = df['lon_o_tmp'] - df['lon_d_tmp']

    df[f"{origen}_norm"] = df[origen]
    df[f"{destino}_norm"] = df[destino]

    df.loc[(df.dist_x >= df.dist_y) & (df.dif_x < 0),
           f"{origen}_norm"] = df[destino]
    df.loc[(df.dist_x >= df.dist_y) & (df.dif_x < 0),
           f"{destino}_norm"] = df[origen]

    df.loc[(df.dist_x < df.dist_y) & (df.dif_y < 0),
           f"{origen}_norm"] = df[destino]
    df.loc[(df.dist_x < df.dist_y) & (df.dif_y < 0),
           f"{destino}_norm"] = df[origen]

    df = df.drop(
        [
            "dist_x",
            "dist_y",
            "dif_x",
            "dif_y",
            'lat_o_tmp',
            'lon_o_tmp',
            'lat_d_tmp',
            'lon_d_tmp',
            "origin",
            "destination",
        ],
        axis=1,
    )

    return df


def create_point_from_h3(h):
    return Point(h3.h3_to_geo(h)[::-1])


def crear_linestring(df,
                     lon_o,
                     lat_o,
                     lon_d,
                     lat_d):
    lineas = df.apply(crear_linea,
                      axis=1,
                      lon_o=lon_o,
                      lat_o=lat_o,
                      lon_d=lon_d,
                      lat_d=lat_d)
    df = gpd.GeoDataFrame(df,
                          geometry=lineas,
                          crs=4326)
    return df


def crear_linea(row, lon_o, lat_o, lon_d, lat_d):
    return (LineString([[row[lon_o], row[lat_o]], [row[lon_d], row[lat_d]]]))


def check_all_geoms_linestring(gdf):
    if not all(gdf.geometry.type == 'LineString'):
        raise ValueError(
            'Invalid geometry type. Only LineStrings are supported.')


def get_points_over_route(route_geom, distance):
    """
    Interpolates points over a projected route geom in meters
    every x meters set by distance
    """
    ranges = range(0, int(route_geom.length), distance)
    points = line_interpolate_point(route_geom, ranges).tolist()
    return points


def lowess_linea(df):
    """
    Takes a DataFrame with legs and lat long for a given line,
    and produces a LineString for that line route geom
    using lowes regression for that


    Parameters
    ----------
    df : opandas.DataFrame
        geoDataFrame legs with latlong

    Returns
    -------
    geopandas.geoDataFrame
        GeoDataFrame containing a single LineString for each line
    """

    id_linea = df.id_linea.unique()[0]
    epsg_m = get_epsg_m()

    print("Obteniendo lowess linea:", id_linea)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
        df['longitud'], df['latitud']), crs=4326).to_crs(epsg_m)
    y = gdf.geometry.y
    x = gdf.geometry.x
    lowess = sm.nonparametric.lowess
    lowess_points = lowess(x, y, frac=0.4, delta=500)
    lowess_points_df = pd.DataFrame(lowess_points.tolist(), columns=['y', 'x'])
    lowess_points_df = lowess_points_df.drop_duplicates()

    if len(lowess_points_df) > 1:
        geom = LineString([(x, y) for x, y in zip(
            lowess_points_df.x, lowess_points_df.y)])
        out = gpd.GeoDataFrame({'geometry': geom}, geometry='geometry',
                               crs=f'EPSG:{epsg_m}', index=[0]).to_crs(4326)
        return out

    else:
        print("Imposible de generar una linea lowess para id_linea = ",
              id_linea)


def get_epsg_m():
    '''
    Gets the epsg id for a coordinate reference system in meters from config
    '''
    configs = leer_configs_generales()
    epsg_m = configs['epsg_m']

    return epsg_m


def distancia_h3(row, *args, **kwargs):
    """
    Computes for a distance between a h3 point and its lag

    Parameters
    ----------
    row : dict
        row with a h3 coord and its lag

    Returns
    ----------
    int
        distance in h3

    """
    try:
        out = h3.h3_distance(row["h3"], row["h3_lag"])
    except ValueError as e:
        out = None
    return out
