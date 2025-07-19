import pandas as pd
import os
from pandas.io.sql import DatabaseError
from urbantrips.utils.utils import (
    iniciar_conexion_db,
    leer_alias,
    agrego_indicador,
    duracion,
    leer_configs_generales,
)


@duracion
def persist_datamodel_tables():
    """
    Esta funcion lee los datos de etapas, viajes y usuarios
    le suma informacion de distancias y de zonas
    y las guarda en csv
    """

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    conn_data = iniciar_conexion_db(tipo="data")

    q = """
        SELECT *
        from etapas e
        where e.od_validado==1
    """
    etapas = pd.read_sql_query(q, conn_data)

    agrego_indicador(etapas, "Cantidad total de etapas", "etapas_expandidas", 0)

    for i in etapas.modo.unique():
        agrego_indicador(
            etapas.loc[etapas.modo == i], f"Etapas {i}", "etapas_expandidas", 1
        )

    agrego_indicador(
        etapas.groupby(
            ["dia", "id_tarjeta"], as_index=False
        ).factor_expansion_linea.sum(),
        "Cantidad de tarjetas finales",
        "usuarios",
        0,
        var_fex="",
    )

    agrego_indicador(
        etapas.groupby(
            ["dia", "id_tarjeta"], as_index=False
        ).factor_expansion_linea.min(),
        "Cantidad total de usuarios",
        "usuarios expandidos",
        0,
    )

    # VIAJES
    viajes = pd.read_sql_query(
        """
                                select *
                                from viajes
                                where od_validado==1
                               """,
        conn_data,
    )

    agrego_indicador(viajes, "Cantidad de registros en viajes", "viajes", 0, var_fex="")

    agrego_indicador(
        viajes, "Cantidad total de viajes expandidos", "viajes expandidos", 0
    )
    agrego_indicador(
        viajes[(viajes.distancia <= 5)],
        "Cantidad de viajes cortos (<5kms)",
        "viajes expandidos",
        1,
    )
    agrego_indicador(
        viajes[(viajes.cant_etapas > 1)],
        "Cantidad de viajes con transferencia",
        "viajes expandidos",
        1,
    )

    agrego_indicador(viajes, "Cantidad total de viajes expandidos", "modos viajes", 0)

    for i in viajes.modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.od_validado == 1) & (viajes.modo == i)],
            f"Viajes {i}",
            "modos viajes",
            1,
        )

    agrego_indicador(
        viajes,
        "Distancia de los viajes (promedio en kms)",
        "avg",
        0,
        var="distance_osm_drive",
        aggfunc="mean",
    )

    agrego_indicador(
        viajes,
        "Distancia de los viajes (mediana en kms)",
        "avg",
        0,
        var="distance_osm_drive",
        aggfunc="median",
    )

    for i in viajes.modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.od_validado == 1) & (viajes.modo == i)],
            f"Distancia de los viajes (promedio en kms) - {i}",
            "avg",
            0,
            var="distance_osm_drive",
            aggfunc="mean",
        )

    for i in viajes.modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.modo == i)],
            f"Distancia de los viajes (mediana en kms) - {i}",
            "avg",
            0,
            var="distance_osm_drive",
            aggfunc="median",
        )

    agrego_indicador(
        viajes,
        "Etapas promedio de los viajes",
        "avg",
        0,
        var="cant_etapas",
        aggfunc="mean",
    )

    # USUARIOS
    print("Leyendo informacion de usuarios...")
    usuarios = pd.read_sql_query(
        """
                                SELECT *
                                from usuarios
                                where od_validado==1
                                """,
        conn_data,
    )

    agrego_indicador(
        usuarios,
        "Cantidad promedio de viajes por tarjeta",
        "avg",
        0,
        var="cant_viajes",
        aggfunc="mean",
    )

    conn_data.close()
    conn_insumos.close()
