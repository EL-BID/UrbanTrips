import pandas as pd
import sqlite3
import os
import yaml
import time
from pandas.io.sql import DatabaseError
from functools import wraps
import re
import ast
from urbantrips.utils.utils import (leer_configs_generales,
                                    iniciar_conexion_db,
                                    duracion
                                    )

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


def check_if_list(string):
    result = string
    if type(string) == str:    
        pattern = r'\[([^\[\]]+)\]'
        match = re.search(pattern, string)
        if match:                    
            try:
                # Convert the string to a list using ast.literal_eval()
                result = ast.literal_eval(string)
            except (RuntimeError, TypeError, NameError):
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

    result = None
    if checkeo >= 0.8:
        result = string
    return result
    


def replace_tabs_with_spaces(file_path, num_spaces=4):
    
    if os.path.isfile(file_path):
    
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

def create_configuracion(configs):

    configuracion = pd.DataFrame([], 
                             columns=['item', 
                                      'variable', 
                                      'subvar', 
                                      'subvar_param', 
                                      'default',
                                      'obligatorio', 
                                      'descripcion_campo', 
                                      'descripcion_general',
                                      'valor'])

    if configs:
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
    return configuracion.fillna('')
    
def revise_configs(configs):
    
    configuracion = create_configuracion(configs)

    conf_path = os.path.join("docs", 'configuraciones.xlsx')
    if os.path.isfile(conf_path):
        config_default = pd.read_excel(conf_path).fillna('')
    else:
        github_csv_url = 'https://raw.githubusercontent.com/EL-BID/UrbanTrips/dev/docs/configuraciones.xlsx'
        config_default = pd.read_excel(github_csv_url).fillna('')

    for _, i in config_default.iterrows():
        if not (i.subvar_param):
            try:
                valor = configuracion[(configuracion.variable==i.variable)&(configuracion.subvar==i.subvar)].valor.values[0]
            except (IndexError, ValueError):
                valor = ''
            try:
                subvar = configuracion[(configuracion.variable==i.variable)&(configuracion.subvar==i.subvar)].subvar.values[0]
            except (IndexError, ValueError):
                subvar = ''
        else:        
            try:
                subvar = configuracion[(configuracion.variable==i.variable)].subvar.values[0]
            except (IndexError, ValueError):
                subvar = ''
                
            try:
                valor = configuracion[(configuracion.variable==i.variable)].valor.values[0]
            except (IndexError, ValueError):
                valor = ''
        if subvar:  
            config_default.loc[_, 'subvar'] = subvar      
        if str(valor):
            if type(valor) == list:
                config_default.loc[_, 'default'] = config_default.loc[_, 'default'] = f'{valor}'
            else:
                config_default.loc[_, 'default'] = config_default.loc[_, 'default'] = valor
    # Chequea si existe hora
    # hora_trx
    hora_trx = config_default.loc[(config_default.variable == 'nombres_variables_trx')&
                                           (config_default.subvar=='hora_trx'), 'default'].values[0]
    if hora_trx:
        config_default.loc[(config_default.variable=='columna_hora'), 'default'] = True
    else:
        config_default.loc[(config_default.variable=='columna_hora'), 'default'] = False

    return config_default

def write_config(config_default):
    path = os.path.join("configs", "configuraciones_generales.yaml")

    with open(path, 'w', encoding='utf8') as file:

        file.write('# Archivo de configuración para urbantrips\n\n' )

        for i in config_default.item.unique():
            tmp = config_default[config_default.item == i].reset_index(drop=True)

            for _, x in tmp.iterrows():  
                x.default = check_if_list(x.default)
                
                if (_ == 0) and (len(x.descripcion_general)>0):
                    file.write(f'# {x.descripcion_general}\n' )

                if len(tmp.variable.unique()) == 1:  

                    if len(x.subvar) > 0:                        
                        if _ == 0:                        
                            file.write(f'{x.variable}: \n' )
                        if type(x.default) != list:
                            
                            if x.default != '':
                                
                                if (type(x.default) == str) & ~((x.default == 'True')|(x.default == 'False')):
                                    file.write(f'    {x.subvar}: "{x.default}"'.ljust(67) ) #subvars

                                else:
                                    file.write(f'    {x.subvar}: {x.default}'.ljust(67) ) #subvars

                                
                            else:
                                file.write(f'    {x.subvar}: '.ljust(67) ) #subvars
                            if len(x.descripcion_campo)>0:                                 
                                file.write(f'# {x.descripcion_campo}')                        

                        else:
                            
                            file.write(f'    {x.subvar}: '.ljust(15))
                            file.write('['.ljust(52))                        
                            if len(x.descripcion_campo)>0:                            
                                file.write(f'# {x.descripcion_campo}'.ljust(15))                        
                            file.write('\n')

                            for z in x.default:
                                file.write(''.ljust(16))
                                file.write(f'"{z}",\n')

                            file.write(''.ljust(22)+']\n')

                        file.write('\n')
                        
                    else:           
                        
                        if x.default != '':                            
                            if (type(x.default) == str) & ~((x.default == 'True')|(x.default == 'False')):
                                file.write(f'{x.variable}: "{x.default}"'.ljust(67) ) #subvars
                            else:
                                file.write(f'{x.variable}: {x.default}'.ljust(67) ) #subvars
                        else:
                            file.write(f'{x.variable}:'.ljust(67) ) #subvars

                        if len(x.descripcion_campo)>0:                            
                            file.write(f'# {x.descripcion_campo}'.ljust(15))                        
                            file.write('\n')

                elif len(x.variable) > 0:
                    
                    if type(x.default) != list:     
                        if x.default != '':
                            if (type(x.default) == str) & ~((x.default == 'True')|(x.default == 'False')):
                                file.write(f'{x.variable}: "{x.default}"'.ljust(67) )

                            else:
                                file.write(f'{x.variable}: {x.default}'.ljust(67) )

                        else:
                            file.write(f'{x.variable}: '.ljust(67) )
                        

                        if len(x.descripcion_campo)>0:                                 
                            file.write(f'# {x.descripcion_campo}')                        

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

            
def check_config_errors(config_default):

    conf_path = os.path.join("docs", 'configuraciones.xlsx')
    if os.path.isfile(conf_path):
        configuraciones  = pd.read_excel(conf_path).fillna('')
    else:
        github_csv_url = 'https://raw.githubusercontent.com/EL-BID/UrbanTrips/dev/docs/configuraciones.xlsx'
        configuraciones  = pd.read_excel(github_csv_url).fillna('')
    
    vars_boolean = []
    for _, i in configuraciones[(configuraciones.default.notna())&(configuraciones.default != '')].iterrows():
        if (i.default == 'True')|(i.default == 'False'):
            vars_boolean += [[i.variable, i.subvar]]
    vars_required = []
    for _, i in configuraciones[configuraciones.obligatorio==True].iterrows():
        vars_required += [[i.variable, i.subvar]]

    orden_trx = None
    
    errores = []
    nombre_archivo_trx = config_default.loc[config_default.variable == 'nombre_archivo_trx'].default.values[0]
    if not nombre_archivo_trx:            
        errores += [f'No está declarado el archivo de transacciones en {os.path.join("data", "data_ciudad")}']
    else:                
        ruta = os.path.join("data", "data_ciudad", nombre_archivo_trx)    
        if not os.path.isfile(ruta):
            errores += [f'No se encuentra el archivo de transacciones {ruta}']
        else:                    
            trx = pd.read_csv(ruta, nrows=1000)

            # check date
            columns_with_date = config_default.loc[(config_default.variable == 'nombres_variables_trx')&
                                                   (config_default.subvar=='fecha_trx'), 'default'].values[0]

            date_format = config_default.loc[(config_default.variable == 'formato_fecha'), 'default'].values[0]

            check_result = check_config_fecha(
                df=trx, columns_with_date=columns_with_date, date_format=date_format)
            if check_result:
                errores += [check_result]

            orden_trx = config_default.loc[(config_default.variable == 'nombres_variables_trx')&
                                           (config_default.subvar == 'orden_trx'), 'default'].values[0]

            if (not orden_trx) & (len(trx[columns_with_date].sample(1).values[0]) <= 10):
                errores += ['No está especificado el orden de las transacciones. ' +\
                            '\n                El orden puede estar especificado por el campo "fecha_trx" si tiene hora/minuto o' +\
                            '\n                por el campo "orden_trx" que no se encuentra en el config.yaml']
            else:
                if not config_default.loc[(config_default.variable == 'ventana_viajes'), 'default'].values[0]:
                    errores += ['"ventana_viajes" debe tener una valor en minutos definido (ej. 60 minutos)']
                if not config_default.loc[(config_default.variable == 'ventana_duplicado'), 'default'].values[0]:
                    errores += ['"ventana_duplicado" debe tener una valor en minutos definido (ej. 5 minutos)']

            # check factor_expansion
            factor_expansion = config_default.loc[(config_default.variable == 'nombres_variables_trx')&
                                                   (config_default.subvar=='factor_expansion'), 'default'].values[0]
            if factor_expansion:
                if factor_expansion not in trx.columns:
                    errores += [f'La variable {factor_expansion} no se encuentra en la tabla de transacciones']
                else:
                    if len( trx[(trx[factor_expansion].isna())|(trx[factor_expansion]==0)] ) > 0:
                        errores += [f'La variable {factor_expansion} no tiene valores o los valores son igual a cero']
            # hora_trx
            hora_trx = config_default.loc[(config_default.variable == 'nombres_variables_trx')&
                                                   (config_default.subvar=='hora_trx'), 'default'].values[0]
            if hora_trx:
                if hora_trx not in trx.columns:
                    errores += [f'La variable {hora_trx} no se encuentra en la tabla de transacciones']
                else:
                    if len( trx[(trx[hora_trx].isna())] ) > 0:
                        errores += [f'La variable {hora_trx} no tiene valores']
                
            # tipo_trx_invalidas
            tipo_trx_invalidas = config_default.loc[(config_default.variable == 'tipo_trx_invalidas'), 'subvar'].values[0]
            if tipo_trx_invalidas:
                if tipo_trx_invalidas not in trx.columns:
                    errores += [f'La variable {tipo_trx_invalidas} no se encuentra en la tabla de transacciones']

        

            # chequea modos

            var_modo = config_default[(config_default.subvar=='modo_trx')].default.values[0]
            if var_modo:             
                if var_modo in trx.columns:             
                    modos = config_default[(config_default.variable=='modos')&(config_default.default!='')].default.unique()
                    modos_faltantes = [i for i in trx[var_modo].unique() if i not in modos]
                    if modos_faltantes:
                        errores += [f'Faltan especificar los modos {modos_faltantes} en el archivo de configuración']
                else:
                    errores += [f'La columna {var_modo} no se encuentra en la tabla de transacciones']
                    
        config_default.loc[config_default.default=='True', 'default'] = True
        config_default.loc[config_default.default=='False', 'default'] = False
        for i in vars_boolean:
            if not ((config_default.loc[(config_default.variable == i[0])&(config_default.subvar == i[1]), 'default'].values[0] == True) | \
                    (config_default.loc[(config_default.variable == i[0])&(config_default.subvar == i[1]), 'default'].values[0] == False)):
                if len(i[1]) > 0:
                    errores += [f'"{i[1]}" debe ser True o False']
                else:
                    errores += [f'"{i[0]}" debe ser True o False']

        for i in vars_required:
            if (config_default.loc[(config_default.variable == i[0])&(config_default.subvar == i[1]), 'default'].values[0] == '') :
                if len(i[1]) > 0:
                    errores += [f'"{i[1]}" no puede tener un valor vacío']
                else:
                    errores += [f'"{i[0]}" no puede tener un valor vacío']

        try:
            resolucion_h3 = int(config_default.loc[(config_default.variable == 'resolucion_h3'), 'default'].values[0])
        except ValueError:
            resolucion_h3 = -99
        if (resolucion_h3 < 0) | (resolucion_h3 > 16):
            errores += ["El parámetro 'resolucion_h3' debe ser un entero entre 0 y 16"]

        try:
            tolerancia_parada_destino = int(config_default.loc[(config_default.variable == 'tolerancia_parada_destino'), 'default'].values[0])
        except ValueError:
            tolerancia_parada_destino = -99

        if (tolerancia_parada_destino < 0) | (tolerancia_parada_destino > 10000):
            errores += ["El parámetro 'tolerancia_parada_destino' debe ser un entero entre 0 y 10000"]

        # ordenamiento de transacciones
        if config_default[config_default.variable=='ordenamiento_transacciones'].default.values[0] == 'fecha_completa':
            if len(date_format) < 14:
                errores += ['La variable "fecha_trx" debe tener hora/minuto para ordenamiento']
        elif config_default[config_default.variable=='ordenamiento_transacciones'].default.values[0] == 'orden_trx':
            if not orden_trx:
                errores += ['La variable "orden_trx" debe estar especificada para ordenar transacciones']
        else:
            errores += ['"ordenamiento_transacciones" debe ser "fecha_completa" o "orden_trx"']
        
        # Chequea exista variable ramal si lineas contienen ramales
        lineas_contienen_ramales = config_default[config_default.variable=='lineas_contienen_ramales'].default.values[0]
        if (lineas_contienen_ramales)& \
           (not config_default[(config_default.variable=='nombres_variables_trx')& \
                      (config_default.subvar=='id_ramal_trx')].default.values[0]):

            errores += ['Debe especificarse el campo "id_ramal_trx" del archivo de transacciones']

        nombre_archivo_informacion_lineas = config_default.loc[
                                                config_default.variable == 'nombre_archivo_informacion_lineas'].default.values[0]     
        
        if nombre_archivo_informacion_lineas:
            ruta = os.path.join("data", "data_ciudad", nombre_archivo_informacion_lineas)
            if not os.path.isfile(ruta):
                errores += [f'No existe el archivo {nombre_archivo_informacion_lineas} que contiene la información de las líneas']
            else:
                # Check all columns are present
                if lineas_contienen_ramales:
                    cols = ['id_linea', 'nombre_linea',
                                        'id_ramal', 'nombre_ramal', 'modo']
                else:
                    cols = ['id_linea', 'nombre_linea', 'modo']

                info = pd.read_csv(ruta)
                
                if not pd.Series(cols).isin(info.columns).all():
                    errores += [f'Faltan columnas en el archivo "{nombre_archivo_informacion_lineas} - deben estar los campos {cols}"']
            
            
    geolocalizar_trx = config_default.loc[config_default.variable == 'geolocalizar_trx'].default.values[0]    
    nombre_archivo_gps = config_default.loc[config_default.variable == 'nombre_archivo_gps'].default.values[0]
    
    if (geolocalizar_trx) and (not nombre_archivo_gps):
        errores += ['Para gelocalizar transacciones debe estar especificado el archivo de transacciones gps']
    
    if not geolocalizar_trx:
        latitud_trx = config_default.loc[config_default.subvar == 'latitud_trx'].default.values[0]   
        longitud_trx = config_default.loc[config_default.subvar == 'longitud_trx'].default.values[0]   
        if (not latitud_trx) | (not longitud_trx):
            errores += ['Si geolocalizar_trx = False deben exister los campos latitud_trx y longitud_trx en la tabla de transacciones']
            
    
    if nombre_archivo_gps:
        ruta = os.path.join("data", "data_ciudad", nombre_archivo_gps)    
        if not os.path.isfile(ruta):
            errores += [f'No se encuentra el archivo de transacciones gps {ruta}']
            cols = ['id_linea_gps', 'interno_gps', 'fecha_gps', 'latitud_gps', 'longitud_gps']
            for i in cols:
                var_gps = config_default.loc[(config_default.variable == 'nombres_variables_gps')&
                                             (config_default.subvar == i)].default.values[0]
                if not var_gps:
                    errores += [f'Debe especificarse la variable {i} del archivo de transacciones gps']

        utilizar_servicios_gps = config_default.loc[config_default.variable == 'utilizar_servicios_gps'].default.values[0]   
        if utilizar_servicios_gps:
            servicios_gps = config_default.loc[config_default.subvar == 'servicios_gps'].default.values[0]   
            if not servicios_gps:
                errores += ['Si se van a utilizar los servicios gps se debe especificar la variable "servicios_gps"']
            valor_inicio_servicio = config_default.loc[config_default.variable == 'valor_inicio_servicio'].default.values[0]   
            if not valor_inicio_servicio:
                errores += ['Si se van a utilizar los servicios gps se debe especificar la variable "valor_inicio_servicio"']

    recorridos_geojson = config_default.loc[config_default.variable == 'recorridos_geojson'].default.values[0]    
    if recorridos_geojson:
        ruta = os.path.join("data", "data_ciudad", recorridos_geojson)
        if not os.path.isfile(ruta):
            errores += [f'No existe el archivo {recorridos_geojson} con los recorridos de las líneas de transporte público']

        

    
    error_txt = '\n'
    for i in errores:
        error_txt += 'ERROR: '+i + '\n'
    assert error_txt == '\n', error_txt
    print('Se concluyó el chequeo del archivo de configuración')

def check_configs_file():

    # Define the directory and file name
    directory = 'configs'
    file_name = 'configuraciones_generales.yaml'
    file_path = os.path.join(directory, file_name)
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Check if the YAML file exists, and if not, create it
    if not os.path.exists(file_path):
        # Create an empty YAML file
        with open(file_path, 'w') as file:
            yaml.dump({}, file)
    
            print(f"Se creo el archivo '{file_name}' en '{directory}'")


@ duracion
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
    check_configs_file()
    replace_tabs_with_spaces(os.path.join("configs", "configuraciones_generales.yaml"))
    configs = leer_configs_generales()
    config_default = revise_configs(configs)
    write_config(config_default)
    check_config_errors(config_default)
