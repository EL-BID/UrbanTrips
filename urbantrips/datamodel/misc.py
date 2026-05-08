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
def persist_indicators():
    """
    Esta funcion crea tabla de indicatores clave
    """

    alias_insumos = leer_configs_generales(autogenerado=False).get("alias_db", "")
    conn_insumos = iniciar_conexion_db(tipo="insumos", alias_db=alias_insumos)

    conn_data = iniciar_conexion_db(tipo="data")

    q = """
            SELECT e.*, tt.travel_time_min, tt.distance_od, tt.distance_route,
                tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
            FROM etapas e
            JOIN dias_ultima_corrida d
            ON e.dia = d.dia
            LEFT JOIN travel_times_legs tt
            ON e.id = tt.id
            WHERE e.od_validado = 1
        """
    etapas = pd.read_sql_query(q, conn_data)
    

    agrego_indicador(etapas, "Cantidad total de etapas", "etapas_expandidas", 0, var_fex="factor_expansion_linea")

    for i in etapas.modo.unique():
        agrego_indicador(
            etapas.loc[etapas.modo == i], f"Etapas {i}", "etapas_expandidas", 1, var_fex="factor_expansion_linea"
        )

    agrego_indicador(
        etapas.groupby(
            ["dia", "id_tarjeta"], as_index=False
        ).factor_expansion_tarjeta.max(),
        "Cantidad de tarjetas finales",
        "usuarios",
        0,
        var_fex="factor_expansion_tarjeta",
    )

    agrego_indicador(
        etapas.groupby(
            ["dia", "id_tarjeta"], as_index=False
        ).factor_expansion_linea.min(),
        "Cantidad total de tarjetas",
        "usuarios expandidos",
        0,
    )

    # VIAJES
    viajes = pd.read_sql_query(
            """
            SELECT v.*, tt.travel_time_min, tt.distance_od, tt.distance_route, 
                tt.distance_route_gps, tt.kmh_od, tt.kmh_route, tt.kmh_route_gps
            FROM viajes v
            JOIN dias_ultima_corrida d
            ON v.dia = d.dia
            LEFT JOIN travel_times_trips tt
            ON v.dia = tt.dia
            AND v.id_tarjeta = tt.id_tarjeta
            AND v.id_viaje = tt.id_viaje
            WHERE v.od_validado = 1
            """,
            conn_data,
        )

    agrego_indicador(viajes, "Cantidad de registros en viajes", "viajes", 0, var_fex="")

    agrego_indicador(
        viajes, "Cantidad total de viajes expandidos", "viajes expandidos", 0, var_fex="factor_expansion_linea"
    )
    agrego_indicador(
        viajes[(viajes.distance_od <= 5)],
        "Cantidad de viajes cortos (<5kms)",
        "viajes expandidos",
        1,
        var_fex="factor_expansion_linea"
    )
    agrego_indicador(
        viajes[(viajes.cant_etapas > 1)],
        "Cantidad de viajes con transferencia",
        "viajes expandidos",
        1,
        var_fex="factor_expansion_linea"
    )


    for i in viajes.modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.od_validado == 1) & (viajes.modo == i)],
            f"Viajes {i}",
            "modos viajes",
            0,
            var_fex="factor_expansion_linea"
        )
   
    agrego_indicador(
        viajes[viajes.od_validado == 1],
        "Distancia de los viajes (promedio en kms)",
        "avg",
        0,
        var="distance_od",
        aggfunc="mean",
        var_fex="factor_expansion_linea"
    )

    agrego_indicador(
        viajes[viajes.od_validado == 1],
        "Distancia de los viajes (mediana en kms)",
        "avg",
        0,
        var="distance_od",
        aggfunc="median",
        var_fex="factor_expansion_linea"
    )

    for i in viajes.modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.od_validado == 1) & (viajes.modo == i)],
            f"Distancia de los viajes (promedio en kms) - {i}",
            "avg",
            0,
            var="distance_od",
            aggfunc="mean",
            var_fex="factor_expansion_linea"
        )

    for i in viajes.modo.unique():
        agrego_indicador(
            viajes.loc[(viajes.modo == i)],
            f"Distancia de los viajes (mediana en kms) - {i}",
            "avg",
            0,
            var="distance_od",
            aggfunc="median",
            var_fex="factor_expansion_linea"
        )

    agrego_indicador(
        viajes,
        "Etapas promedio de los viajes",
        "avg",
        0,
        var="cant_etapas",
        aggfunc="mean",
        var_fex="factor_expansion_linea"
    )

    # USUARIOS
    
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
        var_fex="factor_expansion_linea"
    )

    conn_data.close()
    conn_insumos.close()
