import datetime
import logging
import os
import warnings

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Point

from urbantrips.carto.compute_distances import compute_od_distances

# from urbantrips.carto.carto import compute_distances_osm
from urbantrips.geo import geo
from urbantrips.storage.context import StorageContext
from urbantrips.utils.utils import (
    agrego_indicador,
    duracion,
    leer_configs_generales,
)
from urbantrips.utils.paths import get_paths

warnings.filterwarnings(
    "ignore",
    message="Columns \\(.*\\) have mixed types",
    category=pd.errors.DtypeWarning,
)

logger = logging.getLogger(__name__)


@duracion
def create_transactions(
    ctx: StorageContext,
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

    configs = leer_configs_generales(autogenerado=False)

    try:
        modos_homologados = configs["modos"]
        zipped = zip(modos_homologados.values(), modos_homologados.keys())
        modos_homologados = {k: v for k, v in zipped}
        logger.debug(
            "Utilizando los siguientes modos homologados: %s", modos_homologados
        )
    except KeyError:
        pass

    if geolocalizar_trx_config:
        logger.info("Transacciones geolocalizadas")
        # Cargar las transacciones geolocalizadas
        trx, tmp_trx_inicial = geolocalizar_trx(
            ctx=ctx,
            nombre_archivo_trx_eco=nombre_archivo_trx,
            nombres_variables_trx=nombres_variables_trx,
            tipo_trx_invalidas=tipo_trx_invalidas,
            formato_fecha=formato_fecha,
            nombre_archivo_gps=nombre_archivo_gps,
            nombres_variables_gps=nombres_variables_gps,
        )

    else:
        ruta = str(get_paths().input_dir / nombre_archivo_trx)
        logger.info("Levanta archivo de transacciones %s", ruta)
        trx = pd.read_csv(ruta)

        logger.debug("Filtrando transacciones invalidas: %s", tipo_trx_invalidas)
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

        trx, tmp_trx_inicial = agrego_factor_expansion(trx, ctx)

        # Guardo los días que se están analizando en la corrida actual
        dias_ultima_corrida = pd.DataFrame(trx.dia.unique(), columns=["dia"])
        ctx.data.delete_run_days(dias_ultima_corrida["dia"].tolist())
        ctx.data.save_run_days(dias_ultima_corrida)

        # Eliminar trx fuera del bbox
        trx = eliminar_trx_fuera_bbox(trx, ctx=ctx)

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
        trx["id"] = crear_id_interno(ctx, n_rows=n_rows_trx, tipo_tabla="transacciones")

        # process gps table when no geocoding
        if nombre_archivo_gps is not None:
            process_and_upload_gps_table(
                ctx=ctx,
                nombre_archivo_gps=nombre_archivo_gps,
                nombres_variables_gps=nombres_variables_gps,
                formato_fecha=formato_fecha,
            )

    # Chequea si modo está null en todos le pone autobus por default
    if trx.modo.isna().all():
        logger.warning(
            "No existe información sobre el modo en transacciones; se asume autobus"
        )
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

    logger.info("Subiendo %d registros a la db", len(trx))

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
    trx = trx.loc[trx.id_tarjeta.isin(tmp_trx_limpio.id_tarjeta), :]

    trx = trx.sort_values("id")

    ctx.data.save_transactions(trx)


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
        gps_config = leer_configs_generales(autogenerado=False)
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

    df = df.rename(columns=renombrar_columnas)
    df = df.reindex(columns=renombrar_columnas.values())
    df.columns = df.columns.map(lambda s: s.replace(postfijo, ""))

    return df


def convertir_fechas(df, formato_fecha, crear_hora=False):
    """
    Esta funcion toma una DF de transacciones con el campo 'fecha'
    y un parametro para saber si la hora esta en una columna separada
    """

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
        logger.warning(
            "Convirtiendo fechas infiriendo el formato. Esto hará el proceso más lento"
        )

        df["fecha"] = pd.to_datetime(
            df["fecha"], infer_datetime_format=True, errors="coerce"
        )
        checkeo = df["fecha"].isna().sum() / len(df)

        logger.warning(
            "Infiriendo el formato se pierden %.2f%% de registros", checkeo * 100
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

    return df


def agrego_factor_expansion(trx, ctx: StorageContext):
    # Traigo var_fex si existe
    configs = leer_configs_generales(autogenerado=False)
    try:
        var_fex = configs["nombres_variables_trx"]["factor_expansion"]
    except KeyError:
        var_fex = ""

    if not var_fex:
        trx["factor_expansion"] = 1

    agrego_indicador(
        trx, "Registros en transacciones", "transacciones", 0, var_fex="", ctx=ctx
    )
    agrego_indicador(
        trx,
        "Cantidad de transacciones totales",
        "transacciones",
        0,
        var_fex="factor_expansion",
        ctx=ctx,
    )
    agrego_indicador(
        trx[trx.id_tarjeta.notna()]
        .groupby(["dia", "id_tarjeta"], as_index=False)
        .factor_expansion.min(),
        "Cantidad de tarjetas únicas",
        "tarjetas",
        0,
        var_fex="factor_expansion",
        ctx=ctx,
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

    dias_ultima_corrida = ctx.data.get_run_days()
    run_days = dias_ultima_corrida["dia"].tolist()
    transacciones_linea = transacciones_linea[transacciones_linea.dia.isin(run_days)]
    ctx.data.save_line_transactions(transacciones_linea)

    return trx, tmp_trx_inicial


def eliminar_trx_fuera_bbox(trx, ctx: StorageContext):
    """
    Marca transacciones con geo_valido = 1/0 según si caen
    dentro del área de estudio. El bbox se obtiene de la tabla
    zonificaciones (si existe) o del archivo de configuración.
    """
    zonificaciones = ctx.insumos.get_zones()

    if len(zonificaciones) > 0:
        minx, miny, maxx, maxy = zonificaciones.total_bounds
    else:
        configs = leer_configs_generales(autogenerado=False)
        try:
            bbox = configs["filtro_latlong_bbox"]
            minx, miny, maxx, maxy = (
                bbox["minx"],
                bbox["miny"],
                bbox["maxx"],
                bbox["maxy"],
            )
        except KeyError:
            logger.warning(
                "No se especificó bbox ni zonificaciones. No se puede marcar geo_valido."
            )
            trx["geo_valido"] = 1
            return trx

    # aplicar buffer
    buffer_grados = 0.009 * 30
    minx -= buffer_grados
    miny -= buffer_grados
    maxx += buffer_grados
    maxy += buffer_grados

    logger.info(
        "Eliminando transacciones fuera del bbox: xmin=%.5f, ymin=%.5f, xmax=%.5f, ymax=%.5f",
        minx,
        miny,
        maxx,
        maxy,
    )

    trx["geo_valido"] = (
        trx["longitud"].between(minx, maxx) & trx["latitud"].between(miny, maxy)
    ).astype(int)

    n_invalidos = (trx["geo_valido"] == 0).sum()
    logger.info(
        "Transacciones fuera del bbox: %d de %d (%.2f%%)",
        n_invalidos,
        len(trx),
        (n_invalidos / len(trx)) * 100,
    )
    if "id_tarjeta" in trx.columns:
        tarjetas_validas = round(
            len(trx[trx.geo_valido == 1].id_tarjeta.unique())
            / len(trx.id_tarjeta.unique())
            * 100,
            1,
        )
        logger.info("Tarjetas con alguna transacción válida: %s%%", tarjetas_validas)

        tarjetas_invalidas = round(
            len(trx[(trx.latitud == 0) | (trx.longitud == 0)].id_tarjeta.unique())
            / len(trx.id_tarjeta.unique())
            * 100,
            1,
        )
        logger.info(
            "Tarjetas con latitud / longitud igual a cero: %s%%", tarjetas_invalidas
        )

        tarjetas_invalidas = round(
            len(trx[trx.geo_valido == 0].id_tarjeta.unique())
            / len(trx.id_tarjeta.unique())
            * 100,
            1,
        )
        logger.info(
            "Tarjetas con al menos una transacción inválida: %s%%", tarjetas_invalidas
        )
        logger.debug(
            "Se borran las transacciones fuera del bounding box; se mantienen lat/lon == 0"
        )

        agrego_indicador(
            trx[trx.geo_valido == 1],
            "Cantidad de transacciones latlon válidos",
            "transacciones",
            1,
            var_fex="factor_expansion",
            ctx=ctx,
        )
        agrego_indicador(
            trx[(trx.geo_valido == 0) | ((trx.latitud == 0) & (trx.longitud == 0))],
            "Cantidad de transacciones fuera del bounding box",
            "transacciones",
            1,
            var_fex="factor_expansion",
            ctx=ctx,
        )

        trx = trx[
            (trx.geo_valido == 1) | ((trx.latitud == 0) & (trx.longitud == 0))
        ].drop(columns=["geo_valido"])

    else:
        trx = trx[(trx.geo_valido == 1)].drop(columns=["geo_valido"])

    return trx


def eliminar_NAs_variables_fundamentales(trx, subset):
    """
    Esta funcion toma un DF de trx y elmina los casos con NA en variables
    indispensables para el proceso
    """

    pre = len(trx)
    trx = trx.dropna(
        subset=subset,
        axis=0,
        how="any",
    )
    post = len(trx)

    logger.info(
        "Eliminando NAs en variables fundamentales: %d casos eliminados", pre - post
    )
    return trx


def crear_id_interno(ctx: StorageContext, n_rows: int, tipo_tabla: str) -> list:
    """Returns a list of n_rows sequential integer IDs starting after the current max."""
    new_max_id = ctx.data.get_max_id(tipo_tabla)
    return list(range(new_max_id, new_max_id + n_rows))


def geolocalizar_trx(
    ctx: StorageContext,
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
    configs = leer_configs_generales(autogenerado=False)
    # Leer archivos de trx_eco
    id_tarjeta_trx = nombres_variables_trx["id_tarjeta_trx"]

    ruta_trx_eco = str(get_paths().input_dir / nombre_archivo_trx_eco)
    logger.info("Levanta archivo de transacciones %s", ruta_trx_eco)
    _trx_needed_cols = {v for v in nombres_variables_trx.values() if v}
    if tipo_trx_invalidas:
        _trx_needed_cols |= set(tipo_trx_invalidas.keys())
    trx_eco = pd.read_csv(
        ruta_trx_eco,
        dtype={id_tarjeta_trx: "str"},
        usecols=lambda c: c in _trx_needed_cols,
    )

    logger.debug("Filtrando transacciones invalidas: %s", tipo_trx_invalidas)
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
    trx_eco["id"] = crear_id_interno(ctx, n_rows=n_rows_trx, tipo_tabla="transacciones")

    # Agregar factor de expansion
    trx_eco, tmp_trx_inicial = agrego_factor_expansion(trx_eco, ctx)

    # Guardo los días que se están analizando en la corrida actual
    dias_ultima_corrida = pd.DataFrame(trx_eco.dia.unique(), columns=["dia"])
    ctx.data.delete_run_days(dias_ultima_corrida["dia"].tolist())
    ctx.data.save_run_days(dias_ultima_corrida)

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

    logger.debug("Parseando fechas trx_eco")

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

    # trx_eco.to_sql(
    #     "trx_eco", conn, if_exists="append", index=False, method="multi", chunksize=40
    # )
    # print("Fin subida datos")
    dias_ultima_corrida = ctx.data.get_run_days()
    trx_eco_save = trx_eco[trx_eco.dia.isin(dias_ultima_corrida.dia)]
    ctx.data.save_raw(trx_eco_save, "trx_eco")

    # procesar y subir tabla gps
    process_and_upload_gps_table(
        ctx=ctx,
        nombre_archivo_gps=nombre_archivo_gps,
        nombres_variables_gps=nombres_variables_gps,
        formato_fecha=formato_fecha,
    )

    # hacer el join por fecha
    logger.info("Geolocalizando datos")

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

    trx = ctx.data.query(query)

    logger.info(
        "Geolocalización terminada. Resumen delta trx-gps (min):\n%s",
        trx.delta_trx_gps_min.describe().to_string(),
    )

    trx = trx.drop("delta_trx_gps_min", axis=1)

    ctx.data.execute("DELETE FROM trx_eco")
    return trx, tmp_trx_inicial


def process_and_upload_gps_table(
    ctx: StorageContext, nombre_archivo_gps, nombres_variables_gps, formato_fecha
):
    """
    Esta función lee el archivo csv de información de gps
    lo procesa y sube a la base de datos
    """
    configs = leer_configs_generales(autogenerado=False)

    ruta_gps = str(get_paths().input_dir / nombre_archivo_gps)
    if not os.path.exists(ruta_gps) and ruta_gps.endswith(".csv"):
        zip_path = ruta_gps + ".zip"
        if os.path.exists(zip_path):
            ruta_gps = zip_path
    _gps_needed_cols = {v for v in nombres_variables_gps.values() if v}
    compression = "zip" if ruta_gps.endswith(".zip") else "infer"
    gps = pd.read_csv(ruta_gps, usecols=lambda c: c in _gps_needed_cols, compression=compression)

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
    ctx.data.save_vehicle_expansion_factors(veh_exp)

    # parsear fechas
    gps = eliminar_trx_fuera_bbox(gps, ctx=ctx)

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
    gps["id"] = crear_id_interno(ctx, n_rows=n_rows_gps, tipo_tabla="gps")

    # si se informa un service type que el start_service exista
    if "service_type" in gps.columns:
        if not (gps.service_type == "start_service").any():
            raise Exception(
                "No hay valores que indiquen el inicio de un servicio. "
                "Revisar el configs para servicios_gps"
            )

    # if "distance" not in gps.columns:
    #     gps["distance"] = None

    # if gps["distance"].isna().all():
    #    gps = compute_distance_km_gps(gps)
    # else:
    #    gps = gps.rename(columns={"distance": "distance_km"})

    gps = compute_distance_km_gps(gps)

    # if branches are not present, add branch id as the same as line
    if not configs["lineas_contienen_ramales"]:
        gps.loc[:, "id_ramal"] = gps["id_linea"].copy()

    cols = ["id_linea", "id_ramal", "interno", "fecha"]

    gps = gps.sort_values(cols).copy()

    if (
        "distance_servicio_mts_agg" in gps.columns
        and gps["distance_servicio_mts_agg"].notna().any()
    ):
        gps["distance_servicio_mts"] = gps.groupby(["id_linea", "id_ramal", "interno"])[
            "distance_servicio_mts_agg"
        ].diff()

        # corregir
        gps["distance_servicio_mts"] = gps["distance_servicio_mts"].fillna(0)
        gps.loc[gps["distance_servicio_mts"] < 0, "distance_servicio_mts"] = 0

    if (
        "distance_servicio_mts" in gps.columns
        and gps["distance_servicio_mts"].notna().any()
        and "distance_servicio_mts_agg" not in gps.columns
    ):
        gps["distance_servicio_mts_agg"] = gps.groupby(
            ["id_linea", "id_ramal", "interno"]
        )["distance_servicio_mts"].cumsum()
        # corregir
        gps["distance_servicio_mts_agg"] = gps["distance_servicio_mts_agg"].fillna(0)
        gps.loc[gps["distance_servicio_mts_agg"] < 0, "distance_servicio_mts_agg"] = 0

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
        "id_servicio",
        "service_type",
        "distance_km",
        "distance_servicio_mts",
        "distance_servicio_mts_agg",
        "h3",
    ]

    gps = gps.reindex(columns=cols)

    logger.info("Subiendo tabla gps")

    configs = leer_configs_generales(autogenerado=False)
    res = configs["resolucion_h3"]

    gps["h3"] = gps.apply(geo.h3_from_row, axis=1, args=(res, "latitud", "longitud"))

    dias_ultima_corrida = ctx.data.get_run_days()
    gps_save = gps[gps.dia.isin(dias_ultima_corrida.dia)]
    ctx.data.save_gps(gps_save)


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
def compute_distance_km_gps(gps):
    """
    Computes the distance in kilometers between GPS points using H3 hexagons.

    Parameters:
    gps (pd.DataFrame): A DataFrame containing GPS data with columns 'latitud',
    longitud', 'dia', 'id_linea', 'interno', and 'fecha'.

    Returns:
    pd.DataFrame: The input DataFrame with an additional column 'distance_km'
    representing the computed distances between GPS points.
    """

    # Assign h3 to gps
    logger.info("Computing distances for GPS points...")
    res = 11  # resolution 11 has an average hexagon area of 0.74 km², which is a good balance for urban mobility analysis
    configs = leer_configs_generales(autogenerado=False)

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
    # fill h3_lag with h3 so distance are 0
    gps.h3_lag = gps.h3_lag.combine_first(gps.h3)
    gps = gps.dropna(subset=["h3", "h3_lag"])
    gps["delta_fecha"] = (
        gps.reindex(columns=order_cols).groupby(order_cols[:-1]).diff().fillna(0)
    )

    gps = compute_od_distances(
        od_df=gps,
        origin_col="h3",
        dest_col="h3_lag",
        distance_col="distance_km",
        unit="km",
        db_path="data/matriz_distancia/matriz_distancia.duckdb",
        network_cache_dir="data/matriz_distancia",
        symmetric=False,
        precompute_dist=50_000,
        max_tile_deg=99,
        verbose=True,
    )

    percentile_99 = gps["delta_fecha"].quantile(0.995)
    gps.loc[gps.delta_fecha > percentile_99, "distance_km"] = 0

    # remove h3_lag
    gps = gps.drop(["h3_lag", "delta_fecha"], axis=1)

    return gps


def write_transactions_to_db(ctx: StorageContext, corrida: str) -> None:
    """Registers completion of a run in the general DB."""
    ctx.general.register_run(alias=corrida, process="transactions_completed")
