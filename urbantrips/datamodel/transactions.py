import os
import pandas as pd
import warnings

from urbantrips.geo import geo
from urbantrips.utils.utils import (leer_configs_generales,
                                    duracion,
                                    iniciar_conexion_db,
                                    agrego_indicador,
                                    eliminar_tarjetas_trx_unica,
                                    crear_tablas_geolocalizacion)


@duracion
def create_transactions(geolocalizar_trx_config,
                        nombre_archivo_trx,
                        nombres_variables_trx,
                        formato_fecha,
                        col_hora,
                        tipo_trx_invalidas,
                        nombre_archivo_gps,
                        nombres_variables_gps
                        ):
    """
    Esta función toma las tablas originales y las convierte en el esquema
    que necesita el proceso
    """

    print("Estableciendo conexion con la db")
    conn = iniciar_conexion_db(tipo='data')

    print("Abriendo archivos de configuracion")
    configs = leer_configs_generales()

    try:
        modos_homologados = configs["modos"]
        zipped = zip(modos_homologados.values(), modos_homologados.keys())
        modos_homologados = {k: v for k, v in zipped}
        print('Utilizando los siguientes modos homologados')
        print(modos_homologados)
    except KeyError:
        pass

    print("Leyendo archivo de transacciones")
    if geolocalizar_trx_config:

        print("desde transacciones geolocalizadas")
        # Cargar las transacciones geolocalizadas
        trx, tmp_trx_inicial = geolocalizar_trx(
            nombre_archivo_trx_eco=nombre_archivo_trx,
            nombres_variables_trx=nombres_variables_trx,
            tipo_trx_invalidas=tipo_trx_invalidas,
            formato_fecha=formato_fecha,
            nombre_archivo_gps=nombre_archivo_gps,
            nombres_variables_gps=nombres_variables_gps,
        )

    else:
        print("desde archivo csv de transacciones")
        ruta = os.path.join("data", "data_ciudad", nombre_archivo_trx)
        trx = pd.read_csv(ruta)

        print("Filtrando transacciones invalidas:", tipo_trx_invalidas)
        # Filtrar transacciones invalidas
        if tipo_trx_invalidas is not None:
            trx = filtrar_transacciones_invalidas(trx, tipo_trx_invalidas)

        trx = renombrar_columnas_tablas(
            trx,
            nombres_variables_trx,
            postfijo="_trx",
        )
        trx = trx.rename(columns={"orden": "orden_trx"})

        # Convertir fechas en dia y hora
        if col_hora:
            crear_hora = False
        else:
            crear_hora = True

        trx = convertir_fechas(
            trx,
            formato_fecha=formato_fecha,
            crear_hora=crear_hora,
        )
        print(trx.shape)
        trx, tmp_trx_inicial = agrego_factor_expansion(trx)

        # Eliminar trx fuera del bbox
        trx = eliminar_trx_fuera_bbox(trx)
        print(trx.shape)
        agrego_indicador(
            trx,
            'Cantidad de transacciones latlon válidos', 'transacciones', 1)

        # chequear que no haya faltantes en id
        if trx["id"].isna().any():
            warnings.warn("Hay faltantes en el id que identifica a las trx")
        # crear un id original de las transacciones
        trx["id_original"] = trx["id"].copy()

        # Elminar trx con NA en variables fundamentales
        subset = ["id_tarjeta", "fecha", "id_linea", "latitud", "longitud"]
        trx = eliminar_NAs_variables_fundamentales(trx, subset)

        # crear un id interno de la transaccion
        n_rows_trx = len(trx)
        trx["id"] = crear_id_interno(
            conn, n_rows=n_rows_trx, tipo_tabla='transacciones')

    # Elminar transacciones unicas en el dia
    trx = eliminar_tarjetas_trx_unica(trx)

    # Chequea si modo está null en todos le pone autobus por default
    if trx.modo.isna().all():
        print('No existe información sobre el modo en transaciones')
        print('Se asume que se trata de autobus')
        trx['modo'] = 'autobus'
    else:
        # Estandariza los modos
        modos_ausentes_configs = ~trx['modo'].isin(
            modos_homologados.keys())
        prop_na = modos_ausentes_configs.sum() / len(trx)

        if prop_na > 0.15:
            w_str = f" {round(prop_na * 100,1)} por ciento las transacciones"
            w_str = w_str + " tienen un modo que no coincide con "
            w_str = w_str + " los modos estandarizados en el archivo de "
            w_str = w_str + "configuracion"
            warnings.warn(w_str)

        trx.loc[modos_ausentes_configs, 'modo'] = 'otros'
        trx['modo'] = trx['modo'].replace(modos_homologados)

    # Si la tarjeta venia con NaNs los numeros van a tener un .0
    # que se mantiene si se pasa a strs asi nomas
    # Si es float convierte a entero
    if trx.id_tarjeta.dtype == 'float':
        trx.id_tarjeta = pd.to_numeric(trx.id_tarjeta, downcast='integer')

    # Asignar un largo fijo a las tarjetas
    trx.id_tarjeta = trx.id_tarjeta.map(lambda s: str(s))
    tmp_trx_inicial.id_tarjeta = tmp_trx_inicial.id_tarjeta.map(
        lambda s: str(s))
    zfill = trx.id_tarjeta.map(lambda s: len(s)).max()

    trx['id_tarjeta'] = trx['id_tarjeta'].str.zfill(zfill)
    tmp_trx_inicial['id_tarjeta'] = tmp_trx_inicial['id_tarjeta'].str.zfill(
        zfill)

    print(f"Subiendo {len(trx)} registros a la db")

    lista_cols_db = [
        "id",
        "id_original",
        "id_tarjeta",
        "fecha",
        "dia",
        "tiempo",
        "hora",
        "modo",
        "id_linea",
        "id_ramal",
        "interno",
        "orden_trx",
        "latitud",
        "longitud",
        "factor_expansion"
    ]

    trx = trx.reindex(columns=lista_cols_db)

    # Borrar transacciones que tienen id_tarjetas no validos
    # Construir una tabla de las tarjetas dia con la cantidad de trx validas
    tmp_trx_limpio = trx\
        .groupby(['dia', 'id_tarjeta'], as_index=False)\
        .agg(cant_trx_limpias=('id', 'count'))

    # Comparar con las transacciones originales
    tmp_trx_limpio = tmp_trx_inicial.merge(
        tmp_trx_limpio, on=['dia', 'id_tarjeta'])

    tmp_trx_limpio = tmp_trx_limpio[tmp_trx_limpio.cant_trx ==
                                    tmp_trx_limpio.cant_trx_limpias]

    # Mantener solo las trx de tarjeta con todas las transacciones validas
    print('Borrar informacion de tarjetas con transacciones no validas')
    trx = trx.loc[trx.id_tarjeta.isin(tmp_trx_limpio.id_tarjeta), :]

    agrego_indicador(
        trx, 'Cantidad de transacciones limpias', 'transacciones', 1)

    trx.to_sql("transacciones", conn, if_exists="append", index=False)
    print("Fin subir base")

    conn.close()


@ duracion
def filtrar_transacciones_invalidas(trx, tipo_trx_invalidas):
    """
    Esta funcion toma un DF de transacciones y el dict de columnas
    y valores con transacciones no consideradas validas para elminar
    y las elimina
    """
    for columna in tipo_trx_invalidas.keys():
        valores = tipo_trx_invalidas[columna]
        trx = trx.loc[~trx[columna].isin(valores), :]
    return trx


def renombrar_columnas_tablas(df, nombres_variables, postfijo):
    """
    Esta funcion toma un df, un dict con nombres de variables a ser
    reemplazados y un postfijo que identifica las variables
    del modelo de datos de la app y cambia los nombres y reindexa
    con los atributos de interes de la app. Aquellos atributos que no
    tengan equivalente en nombres_variables apareceran con NULL
    """
    zipped = zip(nombres_variables.values(), nombres_variables.keys())
    renombrar_columnas = {k: v for k, v in zipped}

    print("Renombrando columnas:", nombres_variables)

    df = df.rename(columns=renombrar_columnas)
    df = df.reindex(columns=nombres_variables.keys())
    df.columns = df.columns.map(lambda s: s.replace(postfijo, ""))
    return df


@ duracion
def convertir_fechas(df, formato_fecha, crear_hora=False):
    """
    Esta funcion toma una DF de transacciones con el campo 'fecha'
    y un parametro para saber si la hora esta en una columna separada
    """
    print("Convirtiendo fechas")

    df["fecha"] = pd.to_datetime(
        df["fecha"], format=formato_fecha, errors="coerce"
    )
    # Chequear si el formato funciona
    checkeo = df["fecha"].isna().sum() / len(df)
    if checkeo > 0.8:
        warnings.warn(
            f"Eliminando {round((checkeo * 100),2)} por ciento de registros"
            + " por mala conversion de fechas de acuerdo"
            + " al formato provisto en configs"
            + " Verifique el formato de fecha en configuración"
            + " puede haber un error que no permite la conversión"
        )
        print("Convirtiendo fechas infiriendo el formato."
              + "Esto hará el proceso más lento")

        df["fecha"] = pd.to_datetime(
            df["fecha"], infer_datetime_format=True,
            errors="coerce"
        )
        checkeo = df["fecha"].isna().sum() / len(df)

        print(f"Infiriendo el formato se pierden {round((checkeo * 100),2)}"
              + "por ciento de registros")

    # Elminar errores en conversion de fechas
    df = df.dropna(subset=['fecha'], axis=0)

    df["dia"] = df.fecha.dt.strftime("%Y-%m-%d")

    # Si la hora esta en otra columna, usar esa
    if crear_hora:
        df["tiempo"] = df['fecha'].dt.strftime("%H:%M:%S")
        df['hora'] = df['fecha'].dt.hour
    else:
        df["tiempo"] = None

    print("Fin convertir fechas")
    return df


def agrego_factor_expansion(trx):
    # Traigo var_fex si existe
    configs = leer_configs_generales()
    try:
        var_fex = configs['nombres_variables_trx']['factor_expansion']
    except KeyError:
        var_fex = ''

    if not var_fex:
        trx['factor_expansion'] = 1

    agrego_indicador(
        trx, 'Cantidad de transacciones totales', 'transacciones', 0)

    agrego_indicador(trx[trx.id_tarjeta.notna()].groupby(
        ['dia', 'id_tarjeta'], as_index=False).factor_expansion.min(),
        'Cantidad de tarjetas únicas', 'tarjetas', 0)

    tmp_trx_inicial = trx.dropna(subset=['id_tarjeta'])

    # Si id_tarjeta tenía nan y eran float sacar el .0
    if tmp_trx_inicial.id_tarjeta.dtype == 'float':
        tmp_trx_inicial.id_tarjeta = pd.to_numeric(
            tmp_trx_inicial.id_tarjeta, downcast='integer')

    tmp_trx_inicial = tmp_trx_inicial\
        .groupby(['dia', 'id_tarjeta'], as_index=False)\
        .agg(cant_trx=('id', 'count'))

    return trx, tmp_trx_inicial


@ duracion
def eliminar_trx_fuera_bbox(trx):
    """
    Esta funcion toma una DF de transacciones, lee las coordenadas validas
    del archivo de configuracion y elimina aquellas transacciones fuera de
    las mismas
    """

    print("Eliminando trx con mal lat long")

    configs = leer_configs_generales()
    try:
        configs = configs["filtro_latlong_bbox"]
        print(configs)

        filtro = (
            (trx.longitud > configs["minx"])
            & (trx.latitud > configs["miny"])
            & (trx.longitud < configs["maxx"])
            & (trx.latitud < configs["maxy"])
        )

        pre = len(trx)
        trx = trx.loc[filtro, :]
        post = len(trx)
        print(pre - post, "casos elminados por latlong fuera del bbox")
    except KeyError:
        print("No se especificó una ventana para la bbox")
    return trx


@ duracion
def eliminar_NAs_variables_fundamentales(trx, subset):
    """
    Esta funcion toma un DF de trx y elmina los casos con NA en variables
    indispensables para el proceso
    """

    print("Eliminando NAs en variables fundamentales")
    pre = len(trx)
    trx = trx.dropna(
        subset=subset,
        axis=0,
        how="any",
    )
    post = len(trx)

    print(pre - post, "casos elminados por NA en variables fundamentales")
    return trx


def crear_id_interno(conn, n_rows, tipo_tabla):
    """
    Esta funcion toma una conexion, una cantidad de registros y
    un tipo de tabla (transacciones o gps) y obtiene el id maximo de la tabla
    correspondiente y devuelve un entero incremental unico para identificar
    cada registro de modo consistente en el proceso
    """
    # obtener el id maximo de la tabla de transaccion
    cur = conn.cursor()
    cur.execute(f"select max(id) as max_id from {tipo_tabla} t;")
    rows = cur.fetchall()
    max_id = rows[0][0]

    if max_id is None:
        new_max_id = 0
    else:
        new_max_id = max_id + 1

    # asignar nuevos ids
    new_ids = list(range(new_max_id, new_max_id + n_rows))

    return new_ids


@ duracion
def geolocalizar_trx(
    nombre_archivo_trx_eco,
    nombres_variables_trx,
    tipo_trx_invalidas,
    formato_fecha,
    nombre_archivo_gps,
    nombres_variables_gps,
):
    """
    Esta función lee de dos csv las transacciones y los datos de
    posicionamiento gps de las unidades y geolocaliza las transacciones
    con el latlong de la linea, ramal e interno con el timestamp anterior
    más cercano, sube dos tablas trx_eco y gps y actualiza la tabla
    transacciones con las trx_eco geolocalizadas
    """
    # crear tablas de trx_eco y gps
    conn = iniciar_conexion_db(tipo='data')
    print("Creando tablas de trx_eco y gps para geolocalizacion")
    crear_tablas_geolocalizacion()
    print("Fin crear tablas de trx_eco y gps para geolocalizacion")
    # Leer archivos de trx_eco
    id_tarjeta_trx = nombres_variables_trx['id_tarjeta_trx']

    ruta_trx_eco = os.path.join("data", "data_ciudad", nombre_archivo_trx_eco)
    trx_eco = pd.read_csv(ruta_trx_eco, dtype={id_tarjeta_trx: 'str'})

    print("Filtrando transacciones invalidas:", tipo_trx_invalidas)
    # Filtrar transacciones invalidas
    if tipo_trx_invalidas is not None:
        trx_eco = filtrar_transacciones_invalidas(trx_eco, tipo_trx_invalidas)

    # Formatear archivos trx
    trx_eco = renombrar_columnas_tablas(
        trx_eco,
        nombres_variables_trx,
        postfijo="_trx",
    )
    trx_eco = trx_eco.drop(columns=["latitud", "longitud"])

    # Parsear fechas. Crear hora, si tiene gps tiene hora completa
    trx_eco = convertir_fechas(trx_eco, formato_fecha, crear_hora=True)

    # Crear un id interno
    trx_eco["id_original"] = trx_eco["id"].copy()
    n_rows_trx = len(trx_eco)
    trx_eco["id"] = crear_id_interno(
        conn, n_rows=n_rows_trx, tipo_tabla='transacciones')

    # Agregar factor de expansion
    trx_eco, tmp_trx_inicial = agrego_factor_expansion(trx_eco)

    # Eliminar datos con faltantes en variables fundamentales
    subset = ["id_tarjeta", "fecha", "id_linea"]
    trx_eco = eliminar_NAs_variables_fundamentales(trx_eco, subset)

    # Convertir id tarjeta en int si son float y tienen .0
    if trx_eco.id_tarjeta.dtype == 'float':
        trx_eco.id_tarjeta = pd.to_numeric(
            trx_eco.id_tarjeta, downcast='integer')

    if tmp_trx_inicial.id_tarjeta.dtype == 'float':
        tmp_trx_inicial.id_tarjeta = pd.to_numeric(
            tmp_trx_inicial.id_tarjeta, downcast='integer')

    print("Parseando fechas trx_eco")

    trx_eco["fecha"] = trx_eco["fecha"].map(lambda s: s.timestamp())
    trx_eco = trx_eco.dropna(subset=["id_linea", "id_ramal", "interno"])

    # Eliminar trx unica en el dia
    trx_eco = eliminar_tarjetas_trx_unica(trx_eco)

    cols = ['id',
            'id_original',
            'id_tarjeta',
            'fecha',
            'dia',
            'tiempo',
            'hora',
            'modo',
            'id_linea',
            'id_ramal',
            'interno',
            'orden',
            'factor_expansion']
    trx_eco = trx_eco.reindex(columns=cols)

    print("Subiendo datos a tablas temporales")
    trx_eco.to_sql("trx_eco", conn, if_exists="append", index=False)
    print("Fin subida datos")

    # procesar y subir tabla gps
    process_and_upload_gps_table(
        nombre_archivo_gps=nombre_archivo_gps,
        nombres_variables_gps=nombres_variables_gps,
        formato_fecha=formato_fecha)

    # hacer el join por fecha
    print("Geolocalizando datos")
    query = """
        WITH trx AS (
        select t.id,t.id_original, t.id_tarjeta, datetime(t.fecha, 'unixepoch') as fecha,
                t.dia,t.tiempo,t.hora, t.modo, t.id_linea,
                t.id_ramal, t.interno, t.orden as orden, g.latitud, g.longitud,
                (t.fecha - g.fecha) / 60 as delta_trx_gps_min,
                t.factor_expansion,
            ROW_NUMBER() OVER(
                PARTITION BY t."id"
                ORDER BY g.fecha DESC) AS n_row
        from trx_eco t, gps g
        where  t."id_linea" = g."id_linea"
        and  t."id_ramal" = g."id_ramal"
        and  t."interno" = g."interno"
        and t.fecha > g.fecha
        )
        SELECT *
        FROM trx
        WHERE n_row = 1;
    """
    trx = pd.read_sql_query(
        query,
        conn,
        parse_dates={"fecha": "%Y-%m-%d %H:%M:%S"},
    )

    print(
        "Gelocalización terminada "
        + "Resumen diferencia entre las fechas de las trx "
        + "y las del gps en minutos:"
    )
    print(trx.delta_trx_gps_min.describe())
    trx = trx.drop("delta_trx_gps_min", axis=1)

    conn.execute("""DROP TABLE IF EXISTS trx_eco;""")
    conn.close()
    return trx, tmp_trx_inicial


def process_and_upload_gps_table(nombre_archivo_gps, nombres_variables_gps, formato_fecha):
    """
    Esta función lee el archivo csv de información de gps
    lo procesa y sube a la base de datos
    """
    print("Procesando tabla gps")
    conn = iniciar_conexion_db(tipo='data')

    # crear tabla gps en la db
    crear_tablas_geolocalizacion()

    ruta_gps = os.path.join("data", "data_ciudad", nombre_archivo_gps)
    gps = pd.read_csv(ruta_gps)

    # Formatear archivos gps
    gps = renombrar_columnas_tablas(
        gps,
        nombres_variables_gps,
        postfijo="_gps",
    )

    # parsear fechas
    gps = eliminar_trx_fuera_bbox(gps)

    # Parsear fechas y crear atributo dia
    # col_hora false para no crear tiempo y hora
    gps = convertir_fechas(gps, formato_fecha, crear_hora=False)

    subset = ["interno", "id_ramal", "id_linea", "latitud", "longitud"]
    gps = eliminar_NAs_variables_fundamentales(gps, subset)

    # Convertir fecha en segundos desde 1970
    gps["fecha"] = gps["fecha"].map(lambda s: s.timestamp())
    gps = gps.drop_duplicates(subset=['dia', 'id_linea', 'id_ramal', 'interno',
                                      'fecha', 'latitud', 'longitud'])

    # crear un id original del gps
    gps["id_original"] = gps["id"].copy()

    # crear un id interno de la transaccion
    n_rows_gps = len(gps)
    gps["id"] = crear_id_interno(
        conn, n_rows=n_rows_gps, tipo_tabla='gps')

    cols = ['id',
            'id_original',
            'dia',
            'id_linea',
            'id_ramal',
            'interno',
            'fecha',
            'latitud',
            'longitud']
    gps = gps.reindex(columns=cols)

    # subir datos a tablas temporales
    print("Subiendo tabla gps")
    gps.to_sql("gps", conn, if_exists="append", index=False)
