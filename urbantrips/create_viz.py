import pandas as pd
from urbantrips.viz import viz
from urbantrips.kpi import kpi
from urbantrips.viz_ppt_utils import viz_ppt_utils
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config
from urbantrips.utils.utils import (leer_configs_generales)
from urbantrips.kpi.line_od_matrix import compute_lines_od_matrix
from urbantrips.viz.line_od_matrix import visualize_lines_od_matrix


def main():
    check_config()

    conn_data = utils.iniciar_conexion_db(tipo='data')

    configs = leer_configs_generales()
    lineas_principales = configs['imprimir_lineas_principales']

    # get top trx id_lines
    if lineas_principales != 'All':
        print("Leyendo datos de las lineas con más demanda")
        q = f"""
            select id_linea
            from etapas e
            where modo = 'autobus'
            group by id_linea
            order by sum(factor_expansion_linea) DESC
            limit {lineas_principales};
            """
    else:
        print("Leyendo datos de las lineas de transporte")
        q = """
            select id_linea
            from etapas e
            where modo = 'autobus'
            group by id_linea
            order by sum(factor_expansion_linea) DESC;
            """
    top_line_ids = pd.read_sql(q, conn_data)
    top_line_ids = top_line_ids.id_linea.to_list()

    print("Computar carga por tramo de estas lineas y visualizar")

    # Compute and viz route section load by line
    for rango in [[7, 10], [17, 19]]:
        # crate rout section load
        kpi.compute_route_section_load(
            id_linea=top_line_ids, rango_hrs=rango)
        viz.visualize_route_section_load(
            line_ids=top_line_ids, hour_range=rango,
            save_gdf=True, stat='proportion', factor=500,
            factor_min=50, )

        compute_lines_od_matrix(
            line_ids=top_line_ids, hour_range=rango,
            n_sections=10, day_type='weekday', save_csv=False
        )
        visualize_lines_od_matrix(
            line_ids=top_line_ids, hour_range=rango,
            day_type='weekday', n_sections=10, indicador='prop_etapas')

    # Prduce main viz
    viz.create_visualizations()

    # Produce ppt
    viz_ppt_utils.create_ppt()


if __name__ == "__main__":
    main()
