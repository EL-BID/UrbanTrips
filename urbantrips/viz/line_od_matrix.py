from mycolorpy import colorlist as mcp
import folium
import mapclassify
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from requests.exceptions import ConnectionError as r_ConnectionError
from PIL import UnidentifiedImageError
from folium import Figure
import seaborn as sns
import matplotlib.pyplot as plt
import contextily as cx

from urbantrips.viz.viz import (
    create_squared_polygon,
    crear_linestring,
    get_branch_geoms_from_line
)


from urbantrips.kpi.line_od_matrix import delete_old_lines_od_matrix_by_section_data_q

from urbantrips.utils.utils import (
    iniciar_conexion_db,
    create_line_ids_sql_filter,
    leer_alias
)
from urbantrips.geo import geo


def visualize_lines_od_matrix(line_ids=None, hour_range=False,
                              day_type='weekday',
                              n_sections=10, section_meters=None,
                              indicador='cantidad_etapas'):
    sns.set_style("whitegrid")
    # Download line data
    od_lines = get_lines_od_matrix_data(
        line_ids, hour_range, day_type, n_sections, section_meters)

    # Viz data
    od_lines.groupby(['id_linea', 'yr_mo']).apply(
        viz_line_od_matrix,
        indicator=indicador
    )
    od_lines.groupby(['id_linea', 'yr_mo']).apply(
        map_desire_lines)


def get_lines_od_matrix_data(line_ids, hour_range=False,
                             day_type='weekday',
                             n_sections=10, section_meters=None):

    q = """
    select * from lines_od_matrix_by_section 
    """
    line_ids_where = create_line_ids_sql_filter(line_ids)
    q += line_ids_where

    # hour range filter
    if hour_range:
        hora_min_filter = f"= {hour_range[0]}"
        hora_max_filter = f"= {hour_range[1]}"
    else:
        hora_min_filter = "is NULL"
        hora_max_filter = "is NULL"

    hour_where = f"""
    and hour_min {hora_min_filter}
    and hour_max {hora_max_filter}
    """
    q += hour_where
    q += f"and day_type = '{day_type}'"

    conn_data = iniciar_conexion_db(tipo="data")
    od_lines = pd.read_sql(q, conn_data)
    conn_data.close()

    if section_meters is not None:
        line_sections = get_route_n_sections_from_sections_meters(
            line_ids, section_meters)
        od_lines = od_lines.merge(line_sections,
                                  on=['id_linea', 'n_sections'],
                                  how='inner')
    else:
        od_lines = od_lines.query(f"n_sections == {n_sections}")

    line_ids_to_check = pd.Series(line_ids)
    line_ids_in_data = od_lines.id_linea.drop_duplicates()
    check_line_ids_in_data = line_ids_to_check.isin(line_ids_in_data)

    if not check_line_ids_in_data.all():
        print("Las siguientes líneas no fueron procesadas con matriz od lineas")
        print(', '.join(line_ids_to_check[~check_line_ids_in_data].map(str)))
        print("para esos parámetros de n_sections o section_meters")
        print("Volver a correr compute_lines_od_matrix() con otro parámetros")

    if len(od_lines) == 0:
        raise Exception("La consulta para estos id_lineas con estos parametros"
                        "volvio vacía")
    else:
        return od_lines


def get_route_n_sections_from_sections_meters(line_ids, section_meters):
    """
    For a given section meters param, returns how many sections there is
    in that line route geom

    Parameters
    ----------
    line_ids : int, list of ints or bool
        route id or list of route ids present in the legs dataset. Route
        section load will be computed for that subset of lines. If False, it
        will run with all routes.

    section_meters: int
        section lenght in meters to split the route geom. If specified,
        this will be used instead of n_sections.

    Returns
    ----------
    pandas.DataFrame
        df with line id and n sections

    """
    conn_insumos = iniciar_conexion_db(tipo="insumos")
    line_ids_where = create_line_ids_sql_filter(line_ids)
    q_route_geoms = "select * from lines_geoms" + line_ids_where
    route_geoms = pd.read_sql(q_route_geoms, conn_insumos)
    conn_insumos.close()

    route_geoms["geometry"] = gpd.GeoSeries.from_wkt(route_geoms.wkt)
    epsg_m = geo.get_epsg_m()
    route_geoms = gpd.GeoDataFrame(
        route_geoms, geometry="geometry", crs="EPSG:4326"
    ).to_crs(epsg=epsg_m)

    new_n_sections = (
        route_geoms.geometry.length / section_meters).astype(int)
    route_geoms["n_sections"] = new_n_sections
    route_geoms = route_geoms.reindex(columns=['id_linea', 'n_sections'])

    return route_geoms


def viz_line_od_matrix(od_line, indicator='prop_etapas'):
    """
    Creates viz for line od matrix
    """
    line_id = od_line.id_linea.unique()[0]
    n_sections = od_line.n_sections.unique()[0]
    mes = od_line.yr_mo.unique()[0]
    total_legs = int(od_line.legs.sum())

    # get data
    sections_data_q = f"""
    select * from routes_section_id_coords 
    where id_linea = {line_id}
    and n_sections = {n_sections}
    """
    conn_insumos = iniciar_conexion_db(tipo="insumos")
    sections = pd.read_sql(sections_data_q, conn_insumos)

    s = "select nombre_linea from metadata_lineas" +\
        f" where id_linea = {line_id};"
    metadata = pd.read_sql(s, conn_insumos)
    conn_insumos.close()

    # set title params
    if len(metadata) > 0:
        line_str = metadata.nombre_linea.item()
    else:
        line_str = ''

    day = od_line['day_type'].unique().item()

    if day == 'weekend':
        day_str = 'Fin de semana'
    elif day == 'weekday':
        day_str = 'Dia habil'
    else:
        day_str = day

    font_dicc = {'fontsize': 18,
                 # 'fontweight': 'bold'
                 }

    title = 'Matriz OD por segmento del recorrido'
    if indicator == 'cantidad_etapas':
        title += ' - Cantidad de etapas'
        values = 'legs'

    elif indicator == 'prop_etapas':
        title += ' - Porcentaje de etapas totales'
        values = 'prop'
    else:
        raise Exception(
            "Indicador debe ser 'cantidad_etapas' o 'prop_etapas'")

    if not od_line.hour_min.isna().all():
        from_hr = od_line.hour_min.unique()[0]
        to_hr = od_line.hour_max.unique()[0]
        hr_str = f' {from_hr}-{to_hr} hrs'
        hour_range = [from_hr, to_hr]
    else:
        hr_str = ''
        hour_range = None

    title = title + hr_str + ' - ' + day_str + '-' + mes + \
        '-' + f" {line_str} (id_linea: {line_id})"

    # upload to dash db
    od_line_dash = od_line.copy()
    od_line_dash['nombre_linea'] = line_str
    od_line_dash = od_line_dash.rename(columns={
        'section_id_o': 'Origen',
        'section_id_d': 'Destino'
    })

    conn_dash = iniciar_conexion_db(tipo='dash')
    delete_df = od_line\
        .reindex(columns=['id_linea', 'n_sections'])\
        .drop_duplicates()

    delete_old_lines_od_matrix_by_section_data_q(
        delete_df, hour_range=hour_range,
        day_type=day, yr_mos=[mes],
        db_type='dash')

    od_line_dash.to_sql("matrices_linea", conn_dash,
                        if_exists="append", index=False)

    matrix = od_line.pivot_table(values=values,
                                 index='section_id_o',
                                 columns='section_id_d')

    # produce carto
    epsg = geo.get_epsg_m()
    section_id_start = 1
    section_id_end = sections.section_id.max()
    section_id_middle = int(section_id_end/2)

    geom = [LineString(
        [[sections.loc[i, 'x'], sections.loc[i, 'y']],
         [sections.loc[i+1, 'x'], sections.loc[i+1, 'y']]]
    ) for i in sections.index[:-1]]

    gdf = gpd.GeoDataFrame(pd.DataFrame(
        {'section_id': sections.section_id.iloc[:-1]}),
        geometry=geom, crs='epsg:4326')
    gdf = gdf.to_crs(epsg=epsg)

    gdf.geometry = gdf.geometry.buffer(250)

    # upload sections carto to dash
    gdf_dash = gdf.to_crs(epsg=4326).copy()
    gdf_dash['id_linea'] = line_id
    gdf_dash['n_sections'] = n_sections
    gdf_dash['wkt'] = gdf_dash.geometry.to_wkt()
    gdf_dash['x'] = gdf_dash.geometry.centroid.x
    gdf_dash['y'] = gdf_dash.geometry.centroid.y
    gdf_dash = gdf_dash.drop('geometry', axis=1)
    gdf_dash['nombre_linea'] = line_str

    q_delete = f"""
    delete from matrices_linea_carto
    where id_linea = {line_id}
    and n_sections = {n_sections}
    """
    cur = conn_dash.cursor()
    cur.execute(q_delete)
    conn_dash.commit()

    gdf_dash.to_sql("matrices_linea_carto", conn_dash,
                    if_exists="append", index=False)

    conn_dash.close()

    # set sections to show in map
    section_id_start_xy = gdf.loc[gdf.section_id ==
                                  section_id_start, 'geometry'].centroid.item().coords.xy
    section_id_start_xy = section_id_start_xy[0][0], section_id_start_xy[1][0]

    section_id_end_xy = gdf.loc[gdf.section_id ==
                                section_id_end, 'geometry'].centroid.item().coords.xy
    section_id_end_xy = section_id_end_xy[0][0], section_id_end_xy[1][0]

    section_id_middle_xy = gdf.loc[gdf.section_id ==
                                   section_id_middle, 'geometry'].centroid.item().coords.xy
    section_id_middle_xy = section_id_middle_xy[0][0], section_id_middle_xy[1][0]

    # create figure
    f, (ax1, ax2) = plt.subplots(tight_layout=True, figsize=(24, 10), ncols=2)

    minx, miny, maxx, maxy = gdf.total_bounds
    box = create_squared_polygon(minx, miny, maxx, maxy, epsg)

    gdf.plot(ax=ax1, alpha=0.5)
    box.plot(ax=ax1, color='#ffffff00')

    sns.heatmap(matrix, cmap='Blues', ax=ax2, annot=True, fmt='g')

    ax1.set_axis_off()
    ax2.grid(False)

    prov = cx.providers.CartoDB.Positron
    try:
        cx.add_basemap(ax1, crs=gdf.crs.to_string(), source=prov)
    except (UnidentifiedImageError, ValueError):
        cx.add_basemap(ax1, crs=gdf.crs.to_string())
    except r_ConnectionError:
        pass

    # Notes
    total_legs = '{:,.0f}'.format(total_legs).replace(',', '.')
    ax1.annotate(f'Total de etapas: {total_legs}',
                 xy=(0, -0.05), xycoords='axes fraction',
                 size=18)
    ax1.annotate(f'{section_id_start}', xy=section_id_start_xy,
                 xytext=(-100, 0),
                 size=18,
                 textcoords='offset points',
                 bbox=dict(boxstyle="round", fc="#ffffff", ec="#888888"),
                 arrowprops=dict(arrowstyle="-", ec="#888888")
                 )

    ax1.annotate(f'{section_id_middle}', xy=section_id_middle_xy,
                 xytext=(-100, 0),
                 size=18,
                 textcoords='offset points',
                 bbox=dict(boxstyle="round", fc="#ffffff", ec="#888888"),
                 arrowprops=dict(arrowstyle="-", ec="#888888")
                 )

    ax1.annotate(f'{section_id_end}', xy=section_id_end_xy,
                 xytext=(-100, 0),
                 size=18,
                 textcoords='offset points',
                 bbox=dict(boxstyle="round", fc="#ffffff", ec="#888888"),
                 arrowprops=dict(arrowstyle="-", ec="#888888")
                 )
    # Labels
    f.suptitle(title, fontsize=18)

    ax1.set_title('Cartografía segmentos', fontdict=font_dicc,
                  # y=1.0, pad=-20
                  )
    ax2.set_title('Matriz OD', fontdict=font_dicc,
                  # y=1.0, pad=-20
                  )

    ax2.set_ylabel('Origenes')
    ax2.set_xlabel('Destinos')

    minor_sections = sections[~sections.section_id.isin(
        [section_id_start, section_id_middle, section_id_end, -1])]
    major_sections = sections[sections.section_id.isin(
        [section_id_start, section_id_middle, section_id_end])]

    ax2.set_yticks(minor_sections.index + 0.5, minor=True)
    ax2.set_yticklabels(minor_sections.section_id.to_list(), minor=True)
    ax2.set_yticks(major_sections.index + 0.5, minor=False)
    ax2.set_yticklabels(major_sections.section_id.to_list(),
                        minor=False, size=16)

    ax2.set_xticks(minor_sections.index + 0.5, minor=True)
    ax2.set_xticklabels(minor_sections.section_id.to_list(), minor=True)
    ax2.set_xticks(major_sections.index + 0.5, minor=False)
    ax2.set_xticklabels(major_sections.section_id.to_list(),
                        minor=False, size=16)

    alias = leer_alias()

    for frm in ['png', 'pdf']:
        archivo = f"{alias}_{mes}({day_str})_matriz_od_id_linea_"
        archivo = archivo+f"{line_id}_{indicator}_{hr_str}.{frm}"
        db_path = os.path.join("resultados", frm, archivo)
        f.savefig(db_path, dpi=300)
    plt.close(f)


def map_desire_lines(od_line):

    line_id = od_line.id_linea.unique()[0]
    n_sections = od_line.n_sections.unique()[0]
    mes = od_line.yr_mo.unique()[0]
    day = od_line['day_type'].unique().item()

    if day == 'weekend':
        day_str = 'Fin de semana'
    elif day == 'weekday':
        day_str = 'Dia habil'
    else:
        day_str = day

    if not od_line.hour_min.isna().all():
        from_hr = od_line.hour_min.unique()[0]
        to_hr = od_line.hour_max.unique()[0]
        hr_str = f' {from_hr}-{to_hr} hrs'
    else:
        hr_str = ''

    # get data
    sections_data_q = f"""
    select id_linea,n_sections,section_id,x,y from routes_section_id_coords 
    where id_linea = {line_id}
    and n_sections = {n_sections}
    """
    conn_insumos = iniciar_conexion_db(tipo="insumos")
    sections = pd.read_sql(sections_data_q, conn_insumos)
    conn_insumos.close()

    od_line = od_line\
        .merge(
            sections.rename(
                columns={'x': 'lon_o', 'y': 'lat_o', 'section_id': 'section_id_o'}),
            on=['id_linea', 'n_sections', 'section_id_o'],
            how='left')\
        .merge(
            sections.rename(
                columns={'x': 'lon_d', 'y': 'lat_d', 'section_id': 'section_id_d'}),
            on=['id_linea', 'n_sections', 'section_id_d'],
            how='left')

    od_line = crear_linestring(od_line, 'lon_o', 'lat_o', 'lon_d', 'lat_d')

    alias = leer_alias()

    file_name = f"{alias}_{mes}({day_str})_matriz_od_id_linea_"
    file_name = file_name+f"{line_id}_{hr_str}_{n_sections}_secciones"
    print(file_name)
    create_folium_desire_lines(od_line,
                               cmap='Blues',
                               var_fex='legs',
                               savefile=f"{file_name}.html",
                               k_jenks=5)


def create_folium_desire_lines(od_line,
                               cmap,
                               var_fex,
                               savefile,
                               k_jenks=5):

    bins = [od_line[var_fex].min()-1] + \
        mapclassify.FisherJenks(od_line[var_fex], k=k_jenks).bins.tolist()

    range_bins = range(0, len(bins)-1)
    bins_labels = [
        f'{int(bins[n])} a {int(bins[n+1])} viajes' for n in range_bins]
    od_line['cuts'] = pd.cut(od_line[var_fex], bins=bins, labels=bins_labels)

    fig = folium.Figure(width=800, height=800)
    m = folium.Map(location=[od_line.lat_o.mean(
    ), od_line.lon_o.mean()], zoom_start=9, tiles='cartodbpositron')

    # map lines
    branch_geoms = get_branch_geoms_from_line(
        id_linea=od_line.id_linea.unique().item())
    branch_geoms.explore(m=m, name='Ramales',
                         style_kwds=dict(
                             color="black", opacity=0.4, dashArray='10'),
                         )
    line_w = 0.5

    colors = mcp.gen_color(cmap=cmap, n=k_jenks)

    n = 0
    for i in bins_labels:

        od_line[od_line.cuts == i].explore(
            m=m,
            color=colors[n],
            style_kwds={'fillOpacity': 0.3, 'weight': line_w},
            name=i,
            tooltip=False,
        )
        n += 1
        line_w += 3

    folium.LayerControl(name='Legs').add_to(m)

    fig.add_child(m)

    db_path = os.path.join("resultados", "html", savefile)
    m.save(db_path)
