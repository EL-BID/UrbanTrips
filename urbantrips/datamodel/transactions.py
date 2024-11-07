import h3
import os
import pandas as pd
import geopandas as gpd
import warnings
from urbantrips.carto.carto import compute_distances_osm

from shapely.geometry import Point
from urbantrips.geo import geo
from urbantrips.utils.utils import (
    leer_configs_generales,
    duracion,
    iniciar_conexion_db,
    levanto_tabla_sql,
    agrego_indicador,
    crear_tablas_geolocalizacion,
)


@duracion
def create_transactions(
    geolocalizar_trx_config,
    nombre_archivo_trx,
    nombres_variables_trx,
    formato_fecha,
    col_hora,
    tipo_trx_invalidas,
    nombre_archivo_gps,
    nombres_variables_gps,
):
    """
    Esta función toma las tablas originales y las convierte en el esquema
    que necesita el proceso
    """
    conn = iniciar_conexion_db(tipo="data")

    print("Abriendo archivos de configuracion")
    configs = leer_configs_generales()

    try:
        modos_homologados = configs["modos"]
        zipped = zip(modos_homologados.values(), modos_homologados.keys())
        modos_homologados = {k: v for k, v in zipped}
        print("Utilizando los siguientes modos homologados")
        print(modos_homologados)
    except KeyError:
        pass

    if geolocalizar_trx_config:

        print("Transacciones geolocalizadas")
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
        ruta = os.path.join("data", "data_ciudad", nombre_archivo_trx)
        print("Levanta archivo de transacciones", ruta)
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

        trx, tmp_trx_inicial = agrego_factor_expansion(trx, conn)

        # Guardo los días que se están analizando en la corrida actual
        dias_ultima_corrida = pd.DataFrame(trx.dia.unique(), columns=["dia"])

        conn = iniciar_conexion_db(tipo="data")
        dias_ultima_corrida.to_sql(
            "dias_ultima_corrida", conn, if_exists="replace", index=False
        )

        # borro si ya existen transacciones de una corrida anterior
        values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
        query = f"DELETE FROM transacciones WHERE dia IN ({values})"
        conn.execute(query)
        conn.commit()

        # Eliminar trx fuera del bbox
        trx = eliminar_trx_fuera_bbox(trx)
        agrego_indicador(
            trx,
            "Cantidad de transacciones latlon válidos",
            "transacciones",
            1,
            var_fex="factor_expansion",
        )

        agrego_indicador(
            trx, "Registros válidas en transacciones", "transacciones", 1, var_fex=""
        )

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
            conn, n_rows=n_rows_trx, tipo_tabla="transacciones"
        )

        # process gps table when no geocoding
        if nombre_archivo_gps is not None:
            print("Uploading GPS table without geocoding")
            process_and_upload_gps_table(
                nombre_archivo_gps=nombre_archivo_gps,
                nombres_variables_gps=nombres_variables_gps,
                formato_fecha=formato_fecha,
            )

    # Chequea si modo está null en todos le pone autobus por default
    if trx.modo.isna().all():
        print("No existe información sobre el modo en transaciones")
        print("Se asume que se trata de autobus")
        trx["modo"] = "autobus"
    else:
        # Estandariza los modos
        modos_ausentes_configs = ~trx["modo"].isin(modos_homologados.keys())
        prop_na = modos_ausentes_configs.sum() / len(trx)

        if prop_na > 0.15:
            w_str = f" {round(prop_na * 100, 1)} por ciento las transacciones"
            w_str = w_str + " tienen un modo que no coincide con "
            w_str = w_str + " los modos estandarizados en el archivo de "
            w_str = w_str + "configuracion"
            warnings.warn(w_str)

        trx.loc[modos_ausentes_configs, "modo"] = "otros"
        trx["modo"] = trx["modo"].replace(modos_homologados)

    # Si la tarjeta venia con NaNs los numeros van a tener un .0
    # que se mantiene si se pasa a strs asi nomas
    # Si es float convierte a entero
    if trx.id_tarjeta.dtype == "float":
        trx.id_tarjeta = pd.to_numeric(trx.id_tarjeta, downcast="integer")

    # Asignar un largo fijo a las tarjetas
    trx.id_tarjeta = trx.id_tarjeta.map(lambda s: str(s))
    tmp_trx_inicial.id_tarjeta = tmp_trx_inicial.id_tarjeta.map(lambda s: str(s))
    zfill = trx.id_tarjeta.map(lambda s: len(s)).max()

    trx["id_tarjeta"] = trx["id_tarjeta"].str.zfill(zfill)
    tmp_trx_inicial["id_tarjeta"] = tmp_trx_inicial["id_tarjeta"].str.zfill(zfill)

    # parse date into timestamp
    trx["fecha"] = trx["fecha"].map(lambda s: s.timestamp())

    # if branches are not present, add branch id as the same as line
    if not configs["lineas_contienen_ramales"]:
        trx.loc[:, "id_ramal"] = trx["id_linea"].copy()

    print(f"Subiendo {len(trx)} registros a la db")

    if "genero" not in trx.columns:
        trx["genero"] = "-"
    if "tarifa" not in trx.columns:
        trx["tarifa"] = "-"
    trx["genero"] = trx["genero"].fillna("-")
    trx["tarifa"] = trx["tarifa"].fillna("-")

    lista_cols_db = [
        "id",
        "fecha",
        "id_original",
        "id_tarjeta",
        "dia",
        "tiempo",
        "hora",
        "modo",
        "id_linea",
        "id_ramal",
        "interno",
        "orden_trx",
        "genero",
        "tarifa",
        "latitud",
        "longitud",
        "factor_expansion",
    ]

    trx = trx.reindex(columns=lista_cols_db)

    # Borrar transacciones que tienen id_tarjetas no validos
    # Construir una tabla de las tarjetas dia con la cantidad de trx validas
    tmp_trx_limpio = trx.groupby(["dia", "id_tarjeta"], as_index=False).agg(
        cant_trx_limpias=("id", "count")
    )

    # Comparar con las transacciones originales
    tmp_trx_limpio = tmp_trx_inicial.merge(tmp_trx_limpio, on=["dia", "id_tarjeta"])

    tmp_trx_limpio = tmp_trx_limpio[
        tmp_trx_limpio.cant_trx == tmp_trx_limpio.cant_trx_limpias
    ]

    # Mantener solo las trx de tarjeta con todas las transacciones validas
    print("Borrar informacion de tarjetas con transacciones no validas")
    trx = trx.loc[trx.id_tarjeta.isin(tmp_trx_limpio.id_tarjeta), :]

    agrego_indicador(
        trx,
        "Cantidad de transacciones limpias",
        "transacciones",
        1,
        var_fex="factor_expansion",
    )

    trx = trx.sort_values("id")

    trx.to_sql("transacciones", conn, if_exists="append", index=False)
    print("Fin subir base")

    conn.close()


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

    # if service id column provided in gps table:
    if ("servicios_gps" in nombres_variables) and (
        nombres_variables["servicios_gps"] is not None
    ):

        # get the name in the original df holding service type data
        service_id_col_name = nombres_variables.pop("servicios_gps")

        # get the values for services start and finish
        gps_config = leer_configs_generales()
        start_service_value = gps_config["valor_inicio_servicio"]
        finish_service_value = gps_config["valor_fin_servicio"]

        # create a replace values dict
        service_id_values = {
            start_service_value: "start_service",
            finish_service_value: "finish_service",
        }

        df["service_type"] = df[service_id_col_name].replace(service_id_values)

        # add to the naming dict the new service type attr
        nombres_variables.update({"service_type": ""})

        # remove all values besides start and end of service
        not_service_id_values = ~df[service_id_col_name].isin(
            service_id_values.values()
        )

        df.loc[not_service_id_values, service_id_col_name] = None

    renombrar_columnas = {v: k for k, v in nombres_variables.items()}

    print("Renombrando columnas:", renombrar_columnas)

    df = df.rename(columns=renombrar_columnas)
    df = df.reindex(columns=renombrar_columnas.values())
    df.columns = df.columns.map(lambda s: s.replace(postfijo, ""))

    return df


def convertir_fechas(df, formato_fecha, crear_hora=False):
    """
    Esta funcion toma una DF de transacciones con el campo 'fecha'
    y un parametro para saber si la hora esta en una columna separada
    """
    print("Convirtiendo fechas")

    df["fecha"] = pd.to_datetime(df["fecha"], format=formato_fecha, errors="coerce")
    # Chequear si el formato funciona
    checkeo = df["fecha"].isna().sum() / len(df)
    if checkeo > 0.8:
        warnings.warn(
            f"Eliminando {round((checkeo * 100), 2)} por ciento de registros"
            + " por mala conversion de fechas de acuerdo"
            + " al formato provisto en configs"
            + " Verifique el formato de fecha en configuración"
            + " puede haber un error que no permite la conversión"
        )
        print(
            "Convirtiendo fechas infiriendo el formato."
            + "Esto hará el proceso más lento"
        )

        df["fecha"] = pd.to_datetime(
            df["fecha"], infer_datetime_format=True, errors="coerce"
        )
        checkeo = df["fecha"].isna().sum() / len(df)

        print(
            f"Infiriendo el formato se pierden {round((checkeo * 100), 2)}"
            + "por ciento de registros"
        )

    # Elminar errores en conversion de fechas
    df = df.dropna(subset=["fecha"], axis=0)

    df.loc[:, ["dia"]] = df.fecha.dt.strftime("%Y-%m-%d")

    # Si la hora esta en otra columna, usar esa
    if crear_hora:
        df.loc[:, ["tiempo"]] = df["fecha"].dt.strftime("%H:%M:%S")
        df.loc[:, ["hora"]] = df["fecha"].dt.hour
    else:
        df.loc[:, ["tiempo"]] = None

    print("Fin convertir fechas")
    return df


def agrego_factor_expansion(trx, conn):
    # Traigo var_fex si existe
    configs = leer_configs_generales()
    try:
        var_fex = configs["nombres_variables_trx"]["factor_expansion"]
    except KeyError:
        var_fex = ""

    if not var_fex:
        trx["factor_expansion"] = 1

    agrego_indicador(
        trx,
        "Cantidad de transacciones totales",
        "transacciones",
        0,
        var_fex="factor_expansion",
    )

    agrego_indicador(
        trx[trx.id_tarjeta.notna()]
        .groupby(["dia", "id_tarjeta"], as_index=False)
        .factor_expansion.min(),
        "Cantidad de tarjetas únicas",
        "tarjetas",
        0,
        var_fex="factor_expansion",
    )

    tmp_trx_inicial = trx.dropna(subset=["id_tarjeta"]).copy()

    # Si id_tarjeta tenía nan y eran float sacar el .0
    if tmp_trx_inicial.id_tarjeta.dtype == "float":
        tmp_trx_inicial.id_tarjeta = pd.to_numeric(
            tmp_trx_inicial.id_tarjeta, downcast="integer"
        )

    tmp_trx_inicial = tmp_trx_inicial.groupby(
        ["dia", "id_tarjeta"], as_index=False
    ).agg(cant_trx=("id", "count"))

    # Agrego viajes x id_linea para cálculo de factor de expansión
    transacciones_linea = (
        trx[trx.id_linea.notna()]
        .groupby(["dia", "id_linea"], as_index=False)
        .factor_expansion.sum()
        .rename(columns={"factor_expansion": "transacciones"})
    )

    # borro si ya existen transacciones_linea de una corrida anterior
    dias_ultima_corrida = pd.read_sql_query(
        """
                                SELECT *
                                FROM dias_ultima_corrida
                                """,
        conn,
    )

    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM transacciones_linea WHERE dia IN ({values})"
    conn.execute(query)
    conn.commit()

    transacciones_linea.to_sql(
        "transacciones_linea", conn, if_exists="append", index=False
    )

    return trx, tmp_trx_inicial


def eliminar_trx_fuera_bbox(trx):
    """
    Esta funcion toma una DF de transacciones, lee las coordenadas validas
    del archivo de configuracion y elimina aquellas transacciones fuera de
    las mismas
    """

    print("Eliminando trx con mal lat long")

    original = len(trx)

    zonificaciones = levanto_tabla_sql("zonificaciones", tabla_tipo="insumos")
    zonificaciones = zonificaciones[zonificaciones.zona != "Zona_voi"]
    if len(zonificaciones) > 0:
        trx["geometry"] = trx.apply(
            lambda row: Point(row["longitud"], row["latitud"]), axis=1
        )
        trx = gpd.GeoDataFrame(trx, geometry="geometry", crs=4326)

        zonificaciones["dissolve_column"] = 1
        zonificaciones = zonificaciones.dissolve(by="dissolve_column")

        trx = (
            gpd.sjoin(zonificaciones[["geometry"]], trx)
            .drop(["geometry", "index_right"], axis=1)
            .drop_duplicates()
            .sort_values("id")
            .reset_index(drop=True)
        )
    else:
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
    limpio = len(trx)
    print(f"--Se borraron {original-limpio} registros")

    assert len(trx) > 0, (
        "Se borraron todos los registros, verifique el 'filtro_latlong_bbox' en "
        + "configuraciones_generales.yaml"
    )

    return trx


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
    configs = leer_configs_generales()

    conn = iniciar_conexion_db(tipo="data")
    print("Creando tablas de trx_eco y gps para geolocalizacion")
    crear_tablas_geolocalizacion()
    print("Fin crear tablas de trx_eco y gps para geolocalizacion")
    # Leer archivos de trx_eco
    id_tarjeta_trx = nombres_variables_trx["id_tarjeta_trx"]

    ruta_trx_eco = os.path.join("data", "data_ciudad", nombre_archivo_trx_eco)
    print("Levanta archivo de transacciones", ruta_trx_eco)
    trx_eco = pd.read_csv(ruta_trx_eco, dtype={id_tarjeta_trx: "str"})

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

    # Parsear fechas. Crear hora, si tiene gps tiene hora completa
    trx_eco = convertir_fechas(trx_eco, formato_fecha, crear_hora=True)

    for col in ["latitud", "longitud"]:
        if col in trx_eco.columns:
            trx_eco = trx_eco.drop(columns=[col])

    # Crear un id interno
    trx_eco["id_original"] = trx_eco["id"].copy()
    n_rows_trx = len(trx_eco)
    trx_eco["id"] = crear_id_interno(
        conn, n_rows=n_rows_trx, tipo_tabla="transacciones"
    )

    # Agregar factor de expansion
    trx_eco, tmp_trx_inicial = agrego_factor_expansion(trx_eco, conn)

    # Guardo los días que se están analizando en la corrida actual
    dias_ultima_corrida = pd.DataFrame(trx_eco.dia.unique(), columns=["dia"])
    conn = iniciar_conexion_db(tipo="data")
    dias_ultima_corrida.to_sql(
        "dias_ultima_corrida", conn, if_exists="replace", index=False
    )

    # borro si ya existen transacciones de una corrida anterior
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM transacciones WHERE dia IN ({values})"
    conn.execute(query)
    conn.commit()

    # Eliminar datos con faltantes en variables fundamentales
    if configs["lineas_contienen_ramales"]:
        subset = ["id_tarjeta", "fecha", "id_linea", "id_ramal"]
    else:
        subset = ["id_tarjeta", "fecha", "id_linea"]

    trx_eco = eliminar_NAs_variables_fundamentales(trx_eco, subset)

    # Convertir id tarjeta en int si son float y tienen .0
    if trx_eco.id_tarjeta.dtype == "float":
        trx_eco.id_tarjeta = pd.to_numeric(trx_eco.id_tarjeta, downcast="integer")

    if tmp_trx_inicial.id_tarjeta.dtype == "float":
        tmp_trx_inicial.id_tarjeta = pd.to_numeric(
            tmp_trx_inicial.id_tarjeta, downcast="integer"
        )

    print("Parseando fechas trx_eco")

    trx_eco["fecha"] = trx_eco["fecha"].map(lambda s: s.timestamp())

    if configs["lineas_contienen_ramales"]:
        cols = ["id_linea", "id_ramal", "interno"]
    else:
        cols = ["id_linea", "interno"]

    trx_eco = trx_eco.dropna(subset=cols)
    if "genero" not in trx_eco.columns:
        trx_eco["genero"] = ""
    if "tarifa" not in trx_eco.columns:
        trx_eco["tarifa"] = ""

    cols = [
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
        "orden",
        "genero",
        "tarifa",
        "factor_expansion",
    ]
    trx_eco = trx_eco.reindex(columns=cols)

    print("Subiendo datos a tablas temporales")
    trx_eco.to_sql("trx_eco", conn, if_exists="append", index=False)
    print("Fin subida datos")

    # procesar y subir tabla gps
    process_and_upload_gps_table(
        nombre_archivo_gps=nombre_archivo_gps,
        nombres_variables_gps=nombres_variables_gps,
        formato_fecha=formato_fecha,
    )

    # hacer el join por fecha
    print("Geolocalizando datos")

    if configs["lineas_contienen_ramales"]:
        query = """
            WITH trx AS (
            select t.id,t.id_original, t.id_tarjeta,
                    datetime(t.fecha, 'unixepoch') as fecha,
                    t.dia,t.tiempo,t.hora, t.modo, t.id_linea,
                    t.id_ramal, t.interno, t.orden, t.genero, t.tarifa as tarifa,
                    g.latitud, g.longitud,
                    (t.fecha - g.fecha) / 60 as delta_trx_gps_min,
                    t.factor_expansion,
                ROW_NUMBER() OVER(
                    PARTITION BY t."id"
                    ORDER BY g.fecha DESC) AS n_row
            from trx_eco t, gps g
            where t."dia" = g."dia" 
            and t."id_linea" = g."id_linea"
            and t."id_ramal" = g."id_ramal"
            and t."interno" = g."interno"
            and t.fecha > g.fecha
            )
            SELECT *
            FROM trx
            WHERE n_row = 1;
        """
    else:
        query = """
            WITH trx AS (
            select t.id,t.id_original, t.id_tarjeta,
                    datetime(t.fecha, 'unixepoch') as fecha,
                    t.dia,t.tiempo,t.hora, t.modo, t.id_linea,
                    t.interno, t.orden, t.genero, t.tarifa as tarifa, g.latitud,
                    g.longitud, (t.fecha - g.fecha) / 60 as delta_trx_gps_min,
                    t.factor_expansion,
                ROW_NUMBER() OVER(
                    PARTITION BY t."id"
                    ORDER BY g.fecha DESC) AS n_row
            from trx_eco t, gps g
            where t."dia" = g."dia" 
            and t."id_linea" = g."id_linea"
            and t."interno" = g."interno"
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

    conn.execute("""DELETE FROM trx_eco;""")
    conn.close()
    return trx, tmp_trx_inicial


def process_and_upload_gps_table(
    nombre_archivo_gps, nombres_variables_gps, formato_fecha
):
    """
    Esta función lee el archivo csv de información de gps
    lo procesa y sube a la base de datos
    """
    configs = leer_configs_generales()

    print("Procesando tabla gps")
    conn = iniciar_conexion_db(tipo="data")

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
    # Parsear fechas y crear atributo dia
    # col_hora false para no crear tiempo y hora

    gps = convertir_fechas(gps, formato_fecha, crear_hora=False)

    # compute expansion factors for gps
    veh_exp = get_veh_expansion_from_gps(gps)
    veh_exp.to_sql("vehicle_expansion_factors", conn, if_exists="append", index=False)

    # parsear fechas
    gps = eliminar_trx_fuera_bbox(gps)

    if configs["lineas_contienen_ramales"]:
        subset = ["interno", "id_ramal", "id_linea", "latitud", "longitud"]
    else:
        subset = ["interno", "id_linea", "latitud", "longitud"]

    gps = eliminar_NAs_variables_fundamentales(gps, subset)

    # Convertir fecha en segundos desde 1970
    gps["fecha"] = gps["fecha"].map(lambda s: s.timestamp())

    if configs["lineas_contienen_ramales"]:
        subset = [
            "dia",
            "id_linea",
            "id_ramal",
            "interno",
            "fecha",
            "latitud",
            "longitud",
        ]
    else:
        subset = ["dia", "id_linea", "interno", "fecha", "latitud", "longitud"]

    gps = gps.drop_duplicates(subset=subset)

    # crear un id original del gps
    gps["id_original"] = gps["id"].copy()

    # crear un id interno de la transaccion
    n_rows_gps = len(gps)
    gps["id"] = crear_id_interno(conn, n_rows=n_rows_gps, tipo_tabla="gps")

    # si se informa un service type que el start_service exista
    if "service_type" in gps.columns:
        if not (gps.service_type == "start_service").any():
            raise Exception(
                "No hay valores que indiquen el inicio de un servicio. "
                "Revisar el configs para servicios_gps"
            )

    # compute distance between gps points if there is no distance provided
    if gps["distance"].isna().all():
        gps = compute_distance_km_gps(gps, use_pandana=True)
    else:
        gps = gps.rename(columns={"distance": "distance_km"})

    # if branches are not present, add branch id as the same as line
    if not configs["lineas_contienen_ramales"]:
        gps.loc[:, "id_ramal"] = gps["id_linea"].copy()

    cols = [
        "id",
        "id_original",
        "dia",
        "id_linea",
        "id_ramal",
        "interno",
        "fecha",
        "latitud",
        "longitud",
        "velocity",
        "service_type",
        "distance_km",
        "h3",
    ]

    gps = gps.reindex(columns=cols)

    # subir datos a tablas temporales
    print("Subiendo tabla gps")
    gps.to_sql("gps", conn, if_exists="append", index=False)


def count_unique_vehicles(s):
    return len(s.unique())


def all_gps_broken(s):
    if (s == 0).all() or s.isna().all():
        return 1
    else:
        return 0


def get_veh_expansion_from_gps(gps):
    """
    This function takes a gps table
    and computes average speed in kmr by
    vehicle line and day
    """
    vehicles_with_gps_broken = (
        gps.reindex(columns=["id_linea", "dia", "interno", "latitud", "longitud"])
        .groupby(["id_linea", "dia", "interno"], as_index=False)
        .agg(vehicles_no_gps_lon=("longitud", all_gps_broken))
    )
    vehicles_with_gps_broken = vehicles_with_gps_broken.groupby(
        ["id_linea", "dia"], as_index=False
    ).agg(
        unique_vehicles=("interno", "count"),
        broken_gps_veh=("vehicles_no_gps_lon", "sum"),
    )

    vehicles_with_gps_broken["veh_exp"] = vehicles_with_gps_broken.unique_vehicles / (
        vehicles_with_gps_broken.unique_vehicles
        - vehicles_with_gps_broken.broken_gps_veh
    )

    # cap  veh_exp
    vehicles_with_gps_broken.loc[vehicles_with_gps_broken.veh_exp > 2, "veh_exp"] = 2

    return vehicles_with_gps_broken


@duracion
def compute_distance_km_gps(gps, use_pandana=False):
    """
    Computes the distance in kilometers between GPS points using either H3 hexagons
    or Pandana.

    Parameters:
    gps (pd.DataFrame): A DataFrame containing GPS data with columns 'latitud',
    longitud', 'dia', 'id_linea', 'interno', and 'fecha'.
    use_pandana (bool): If True, computes distances using Pandana and OpenStreetMap
    data. If False, computes distances using H3 hexagons. Default is False.

    Returns:
    pd.DataFrame: The input DataFrame with an additional column 'distance_km'
    representing the computed distances between GPS points.
    """

    # Assign h3 to gps
    print("Computing distances for gps")
    res = 11
    configs = leer_configs_generales()
    if configs["lineas_contienen_ramales"]:
        order_cols = ["dia", "id_linea", "id_ramal", "interno", "fecha"]
        reindex_cols = ["dia", "id_linea", "id_ramal", "interno", "h3"]
    else:
        order_cols = ["dia", "id_linea", "interno", "fecha"]
        reindex_cols = ["dia", "id_linea", "interno", "h3"]

    gps["h3"] = gps.apply(geo.h3_from_row, axis=1, args=(res, "latitud", "longitud"))

    # order by day, line, vehicle and date
    gps = gps.sort_values(order_cols).reset_index(drop=True)

    # Assign to each gps point the h3 of the following gps point
    gps["h3_lag"] = (
        gps.reindex(columns=reindex_cols).groupby(reindex_cols[:-1]).shift(1)
    )
    gps = gps.dropna(subset=["h3", "h3_lag"])
    gps["delta_fecha"] = (
        gps.reindex(columns=order_cols).groupby(order_cols[:-1]).diff().fillna(0)
    )

    if use_pandana:
        # compute distances with pandana
        temp_df = gps.reindex(columns=["h3", "h3_lag"]).drop_duplicates().dropna()
        temp_df = temp_df[temp_df.h3 != temp_df.h3_lag].reset_index(drop=True)

        distances = compute_distances_osm(
            temp_df,
            h3_o="h3",
            h3_d="h3_lag",
            processing="pandana",
            modes=["drive"],
            use_parallel=True,
        )

        distances = distances.rename(
            columns={"distance_osm_drive": "distance_km"}
        ).reindex(columns=["h3", "h3_lag", "distance_km"])
        gps = gps.merge(distances, on=["h3", "h3_lag"], how="left")
        gps["distance_km"] = gps["distance_km"].fillna(0)

    else:
        # compute h3 distance
        distancia_entre_hex = h3.edge_length(resolution=res, unit="km")
        distancia_entre_hex = distancia_entre_hex * 2
        gps_dict = gps.to_dict("records")
        gps.loc[:, ["distance_km"]] = list(map(geo.distancia_h3, gps_dict))
        gps.loc[:, ["distance_km"]] = gps["distance_km"] * distancia_entre_hex

    percentile_99 = gps["delta_fecha"].quantile(0.995)
    gps.loc[gps.delta_fecha > percentile_99, "distance_km"] = 0

    # remove h3_lag
    gps = gps.drop(["h3_lag", "delta_fecha"], axis=1)

    return gps
