import pandas as pd
import sqlite3
import os
import yaml
import time
from pandas.io.sql import DatabaseError
from functools import wraps

from urbantrips.utils.utils import (leer_configs_generales,
                                    iniciar_conexion_db,
                                    )

import ast

def check_config_fecha(df, columns_with_date, date_format):
    """
    Esta funcion toma un dataframe, una columna donde se guardan fechas,
    un formato de fecha, intenta parsear las fechas y arroja un error
    si mas del 80% de las fechas no pueden parsearse
    """
    fechas = pd.to_datetime(
        df[columns_with_date], format=date_format, errors="coerce"
    )

    # Chequear si el formato funciona
    checkeo = fechas.isna().sum() / len(df)
    string = f"El formato de fecha {date_format} no es correcto. Actualmente se pierden el " +\
        f"{round((checkeo * 100),2)} por ciento de registros" +\
        f"\nVerifique que coincida con el formato de fecha del archivo según este ejemplo de la tabla {df[columns_with_date].sample(1).values[0]}"
    assert checkeo < 0.8, string

def check_config():
    """
    This function takes a configuration file in yaml format
    and read its content. Then check for any inconsistencies
    in the file, printing an error message if one is found.

    Args:
    None

    Returns:
    None

    """
    
    # Crea el archivo de configuración
    create_config()

    print("Chequeando archivo de configuracion")
    configs = leer_configs_generales()
    nombre_archivo_trx = configs["nombre_archivo_trx"]

    assert nombre_archivo_trx, f'No está declarado el archivo de transacciones en {os.path.join("data", "data_ciudad")}'
    ruta = os.path.join("data", "data_ciudad", nombre_archivo_trx)    
    assert os.path. isfile(ruta), f'No se encuentra el archivo de transacciones {ruta}'
    
    trx = pd.read_csv(ruta, nrows=1000)
    
        # chequear que esten los atributos obligatorios
    configs_obligatorios = [
        'geolocalizar_trx', 'resolucion_h3', 'tolerancia_parada_destino',
        'nombre_archivo_trx', 'nombres_variables_trx',
        'imputar_destinos_min_distancia', 'formato_fecha', 'columna_hora',
        'ordenamiento_transacciones', 'lineas_contienen_ramales']

    for param in configs_obligatorios:
        if param not in configs:
            raise KeyError(
                f'Error: El archivo de configuracion no especifica el parámetro {param}')
            
            
    # check date
    columns_with_date = configs['nombres_variables_trx']['fecha_trx']
    date_format = configs['formato_fecha']
    check_config_fecha(
        df=trx, columns_with_date=columns_with_date, date_format=date_format)

    if not configs['nombres_variables_trx']['orden_trx']:
        assert not len(trx[columns_with_date].sample(1).values[0]) <= 10, 'No está especificado el orden de las transacciones. ' +\
                                                                '\n                El orden puede estar especificado por el campo "fecha_trx" si tiene hora/minuto o' +\
                                                                '\n                por el campo "orden_trx" que no se encuentra en el config.yaml'
    
    if len(trx[columns_with_date].sample(1).values[0]) >= 10:
        assert (not configs['ordenamiento_transacciones'])| \
        (configs['ordenamiento_transacciones']=='fecha_completa')| \
        (configs['ordenamiento_transacciones']=='orden_trx'), '"ordenamiento_transacciones" debe tener valores "fecha_completa" o "orden_trx"'

    if configs['ordenamiento_transacciones'] == 'fecha_completa':
        assert configs['ventana_viajes'], '"ventana_viajes" debe tener una valor en minutos definido (ej. 60 minutos)'
        assert configs['ventana_duplicado'], '"ventana_duplicado" debe tener una valor en minutos definido (ej. 5 minutos)'

    # check branch param
    mensaje = "Debe especificarse `lineas_contienen_ramales`"
    assert 'lineas_contienen_ramales' in configs, mensaje
    assert configs['lineas_contienen_ramales'] is not None, mensaje
    assert isinstance(configs['lineas_contienen_ramales'],
                      bool), '`lineas_contienen_ramales` debe ser True o False'

    # Chequear que los parametros tengan valor correcto
    assert isinstance(configs['geolocalizar_trx'],
                      bool), "El parámetro geolocalizar_trx debe ser True o False"

    assert isinstance(configs['columna_hora'],
                      bool), "El parámetro columna_hora debe ser True o False"

    assert isinstance(configs['resolucion_h3'], int) and configs['resolucion_h3'] >= 0 and configs[
        'resolucion_h3'] <= 15, "El parámetro resolucion_h3 debe ser un entero entre 0 y 16"

    assert isinstance(configs['tolerancia_parada_destino'], int) and configs['tolerancia_parada_destino'] >= 0 and configs[
        'tolerancia_parada_destino'] <= 10000, "El parámetro tolerancia_parada_destino debe ser un entero entre 0 y 10000"

    assert not isinstance(configs['nombre_archivo_trx'], type(
        None)), "El parámetro nombre_archivo_trx no puede estar vacío"

    # chequear nombres de variables en archivo trx
    nombres_variables_trx = configs['nombres_variables_trx']
    assert isinstance(nombres_variables_trx,
                      dict), "El parámetro nombres_variables_trx debe especificarse como un diccionario"

    nombres_variables_trx = pd.DataFrame(
        {'trx_name': nombres_variables_trx.keys(), 'csv_name': nombres_variables_trx.values()})

    nombres_variables_trx_s = nombres_variables_trx.csv_name.dropna()
    nombres_var_config_en_trx = nombres_variables_trx_s.isin(trx.columns)

    if not nombres_var_config_en_trx.all():
        raise KeyError('Algunos nombres de atributos especificados en el archivo de configuración no están en el archivo csv de transacciones: ' +
                       ','.join(nombres_variables_trx_s[~nombres_var_config_en_trx]))

    # check mandatory attributes for trx
    atributos_trx_obligatorios = pd.Series(
        ['fecha_trx', 'id_tarjeta_trx', 'id_linea_trx'])

    if not configs['geolocalizar_trx']:
        trx_coords = pd.Series(['latitud_trx', 'longitud_trx'])
        atributos_trx_obligatorios = pd.concat(
            [atributos_trx_obligatorios, trx_coords])
    else:
        # if geocoding vehicle id but be present
        interno_col = pd.Series(['interno_trx'])
        atributos_trx_obligatorios = pd.concat(
            [atributos_trx_obligatorios, interno_col])

    if configs['lineas_contienen_ramales']:
        # if branches branch id must be present
        ramal_trx = pd.Series(['id_ramal_trx'])
        atributos_trx_obligatorios = pd.concat(
            [atributos_trx_obligatorios, ramal_trx])

    attr_obligatorios_en_csv = atributos_trx_obligatorios.isin(
        nombres_variables_trx.dropna().trx_name)

    assert attr_obligatorios_en_csv.all(), "Algunos atributos obligatorios no tienen un atributo correspondiente en el csv de transacionnes: " + \
        ','.join(atributos_trx_obligatorios[~attr_obligatorios_en_csv])


    # check consistency in params

    if configs['ordenamiento_transacciones'] == 'fecha_completa':

        assert isinstance(configs['ventana_viajes'], int) and configs['ventana_viajes'] >= 1 and configs[
            'ventana_viajes'] <= 1000, "Cuando el parametro ordenamiento_transacciones es 'fecha_completa', el parámetro 'ventana_viajes' debe ser un entero mayor a 0"

        assert isinstance(configs['ventana_duplicado'],
                          int) and configs['ventana_duplicado'] >= 1, "Cuando el parametro ordenamiento_transacciones es 'fecha_completa', el parámetro 'ventana_duplicado' debe ser un entero mayor a 0"

    # check consistency in geocoding

    if configs['geolocalizar_trx']:
        mensaje = "Si geolocalizar_trx = True entonces se debe especificar un archivo con informacion gps" + \
            " con los parámetros `nombre_archivo_gps` y `nombres_variables_gps`"
        assert 'nombre_archivo_gps' in configs, mensaje
        assert configs['nombre_archivo_gps'] is not None, mensaje

        assert 'nombres_variables_gps' in configs, mensaje
        nombres_variables_gps = configs['nombres_variables_gps']

        assert isinstance(nombres_variables_gps,
                          dict), "El parámetro nombres_variables_gps debe especificarse como un diccionario"

        ruta = os.path.join("data", "data_ciudad",
                            configs['nombre_archivo_gps'])
        gps = pd.read_csv(ruta, nrows=1000)

        nombres_variables_gps = pd.DataFrame(
            {
                'trx_name': nombres_variables_gps.keys(),
                'csv_name': nombres_variables_gps.values()
            }
        )

        nombres_variables_gps_s = nombres_variables_gps.csv_name.dropna()
        nombres_var_config_en_gps = nombres_variables_gps_s.isin(gps.columns)

        if not nombres_var_config_en_gps.all():
            raise KeyError('Algunos nombres de atributos especificados en el archivo de configuración no están en el archivo de transacciones',
                           nombres_variables_gps_s[~nombres_var_config_en_gps])

        # chequear que todos los atributos obligatorios de trx
        # tengan un atributo en el csv
        atributos_gps_obligatorios = pd.Series(
            ['id_linea_gps',
             'interno_gps',
             'fecha_gps',
             'latitud_gps',
             'longitud_gps'])

        if configs['lineas_contienen_ramales']:
            ramal_gps = pd.Series(['id_ramal_gps'])
            atributos_gps_obligatorios = pd.concat(
                [atributos_gps_obligatorios, ramal_gps])

        attr_obligatorios_en_csv = atributos_gps_obligatorios.isin(
            nombres_variables_gps.trx_name)

        assert attr_obligatorios_en_csv.all(), "Algunos atributos obligatorios no tienen un atributo correspondiente en el csv de transacionnes" + \
            ','.join(atributos_gps_obligatorios[~attr_obligatorios_en_csv])

        # chequear validez de fecha
        columns_with_date = configs['nombres_variables_gps']['fecha_gps']
        check_config_fecha(
            df=gps,
            columns_with_date=columns_with_date, date_format=date_format)

    # Checkear que existan los archivos de zonficación especificados config
    assert 'zonificaciones' in configs, "Debe haber un atributo " +\
        "`zonificaciones` en config aunque este vacío"

    if configs['zonificaciones']:
        for i in configs['zonificaciones']:
            if ('geo' in i) and (configs['zonificaciones'][i]):
                geo_file = os.path.join(
                    "data", "data_ciudad", configs['zonificaciones'][i])
                assert os.path.exists(
                    geo_file), f"File {geo_file} does not exist"

    # check epsg in meters
    assert 'epsg_m' in configs, "Debe haber un atributo `epsg_m` en config " +\
        "especificando un id de EPSG para una proyeccion en metros"

    assert isinstance(
        configs['epsg_m'], int), "Debe haber un id de EPSG en metros en" +\
        " configs['epsg_m'] "

    print("Proceso de chequeo de archivo de configuración concluido con éxito")
    return None



def check_if_list(string):
    result = string
    if len(str(string)) > 0:
        try:
            # Convert the string to a list using ast.literal_eval()
            result = ast.literal_eval(string)
        except:
            pass
        
    return result


def replace_tabs_with_spaces(file_path, num_spaces=4):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        content = file.read()

    # Check if the file contains tabs
    if '\t' in content:
        # Replace tabs with spaces
        content = content.replace('\t', ' ' * num_spaces)
        # Save the modified content to the same file
        with open(file_path, 'w') as file:
            file.write(content)
            
def create_config():
    """
    Actualiza el archivo config.yaml en base al template configuraciones.
    """


    path = os.path.join("configs", "configuraciones_generales.yaml")
    
    if os.path.isfile(path):
        replace_tabs_with_spaces(path)
        configs = leer_configs_generales()
        configuracion = pd.DataFrame([])
    else:
        configs = {}
        configuracion = pd.DataFrame([], 
                                     columns=['item', 
                                              'variable', 
                                              'subvar', 
                                              'subvar_param', 
                                              'default',
                                              'obligatorio', 
                                              'descripcion_campo', 
                                              'descripcion_general'])

    for i in configs:   

        if type(configs[i]) != dict:
            y = configs[i]
            configuracion = pd.concat([
                                    configuracion,
                                    pd.DataFrame([[i, '', y]], columns=['variable', 'subvar', 'valor'])])

        else:
            for x in configs[i]:

                y = configs[i][x]            

                configuracion = pd.concat([
                        configuracion,
                        pd.DataFrame([[i, x, y]], columns=['variable', 'subvar', 'valor']) ])

    conf_path = os.path.join("docs", 'configuraciones.xlsx')
    config_default = pd.read_excel(conf_path).fillna('')

    config_default['valor'] = '0'
    for _, i in config_default.iterrows():

        if (i.subvar_param):
            if len(configuracion[(configuracion.variable==i.variable)])>0:                    
                if configuracion[(configuracion.variable==i.variable)].valor.notna().values[0]:
                    config_default.loc[_, 'subvar'] = configuracion[(configuracion.variable==i.variable)].subvar.values[0]
                    config_default.loc[_, 'valor'] = str(configuracion[(configuracion.variable==i.variable)].valor.values[0])
                    config_default.loc[_, 'default'] = config_default.loc[_, 'valor']
        else:
            if len(configuracion[(configuracion.variable==i.variable)&(configuracion.subvar==i.subvar)]) > 0:
                if configuracion[(configuracion.variable==i.variable)&(configuracion.subvar==i.subvar)].valor.notna().values[0]:
                    config_default.loc[_, 'valor'] = str(configuracion[(configuracion.variable==i.variable)&(configuracion.subvar==i.subvar)].valor.values[0])            
                    config_default.loc[_, 'default'] = config_default.loc[_, 'valor']

    with open(path, 'w', encoding='utf8') as file:

        file.write('# Archivo de configuración para urbantrips\n\n' )

        for i in config_default.item.unique():
            tmp = config_default[config_default.item == i].reset_index(drop=True)

            for _, x in tmp.iterrows():
                x.default = check_if_list(x.default)
                if (_ == 0) and (len(x.descripcion_general)>0):
                    file.write(f'# {x.descripcion_general}\n' )

                if len(tmp.variable.unique()) == 1:
                    if _ == 0:
                        file.write(f'{x.variable}:\n' )
                    if len(x.subvar) > 0:

                        if type(x.default) != list:
                            if not x.default == '':
                                if type(x.default) == str:
                                    file.write(f'    {x.subvar}: "{x.default}"'.ljust(67) ) #subvars

                                else:
                                    file.write(f'    {x.subvar}: {x.default}'.ljust(67) ) #subvars

                                
                            else:
                                file.write(f'    {x.subvar}: '.ljust(67) ) #subvars
                            if len(x.descripcion_campo)>0:                                 
                                file.write(f'# {x.descripcion_campo}')                        

                        else:
                            file.write(f'    {x.subvar}: '.ljust(15))
                            file.write('['.ljust(48))                        
                            if len(x.descripcion_campo)>0:                            
                                file.write(f'# {x.descripcion_campo}'.ljust(15))                        
                            file.write('\n')

                            for z in x.default:
                                file.write(''.ljust(16))
                                file.write(f'"{z}",\n')

                            file.write(''.ljust(22)+']\n')

                        file.write('\n')
                else:
                    if len(x.variable) > 0:
                        
                        
                        if type(x.default) != list:     
                            if not x.default == '':
                                if type(x.default) == str:
                                    file.write(f'{x.variable}: "{x.default}"'.ljust(67) )

                                else:
                                    file.write(f'{x.variable}: {x.default}'.ljust(67) )

                            else:
                                file.write(f'{x.variable}: '.ljust(67) )
                            

                            if len(x.descripcion_campo)>0:                                 
                                file.write(f'# {x.descripcion_campo}\n')                        

                        else:
                            file.write(f'    {x.variable}: '.ljust(15))
                            file.write('['.ljust(48))
                            if len(x.descripcion_campo)>0:                            
                                file.write(f'# {x.descripcion_campo}'.ljust(15))                        
                            file.write('\n')

                            for z in x.default:
                                file.write(''.ljust(16))
                                file.write(f'"{z}",\n')

                            file.write(''.ljust(22)+']\n')


                        file.write('\n')

            file.write('\n')