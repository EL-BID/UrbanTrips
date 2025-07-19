import numpy as np
import multiprocessing
import pandas as pd
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    levanto_tabla_sql,
    guardar_tabla_sql,
)
from urbantrips.kpi.kpi import add_distances_to_legs


@duracion
def create_trips_from_legs():
    """
    Loads the legs table from db, updates expansion factores and produces
    trips and users table
    """

    # Leer etapas que no esten en ya viajes por id_tarjeta, id_viaje, dia
    conn = iniciar_conexion_db(tipo="data")

    print("Leyendo datos de etapas para producir viajes")
    dias_ultima_corrida = pd.read_sql_query(
        """
                            SELECT *
                            FROM dias_ultima_corrida
                            """,
        conn,
    )

    etapas = pd.read_sql_query(
        """
                                    SELECT e.*
                                    FROM etapas e
                                    JOIN dias_ultima_corrida d
                                    ON e.dia = d.dia
                                    """,
        conn,
    )

    indicadores = pd.read_sql_query(
        """
                                SELECT i.*
                                FROM indicadores i
                                JOIN dias_ultima_corrida d
                                ON i.dia = d.dia
                                """,
        conn,
    )

    # Calculo de factores de expansion por línea
    transacciones_linea = pd.read_sql_query(
        """
                            SELECT t.*
                            FROM transacciones_linea t
                            JOIN dias_ultima_corrida d
                            ON t.dia = d.dia
                            """,
        conn,
    )

    print("Creando factores de expansion")

    # Creo factores de expansión
    etapas = etapas.drop(["factor_expansion_tarjeta", "factor_expansion_linea"], axis=1)

    tot_tarjetas = (
        etapas.groupby(["dia", "id_tarjeta"], as_index=False)
        .factor_expansion_original.mean()
        .groupby("dia", as_index=False)
        .factor_expansion_original.sum()
        .round()
        .rename(columns={"factor_expansion_original": "tot_tarjetas"})
    )

    tarjetas = etapas.groupby(["dia", "id_tarjeta"], as_index=False).agg(
        {"factor_expansion_original": "mean", "od_validado": "min"}
    )

    tot_tarjetas = tot_tarjetas.merge(
        tarjetas[tarjetas.od_validado == 1]
        .groupby("dia", as_index=False)
        .agg({"factor_expansion_original": "sum", "id_tarjeta": "count"})
        .round()
        .rename(
            columns={
                "factor_expansion_original": "tarjetas_validas",
                "id_tarjeta": "len_tarjetas",
            }
        )
    )

    tot_tarjetas["diff_tarjetas"] = (
        tot_tarjetas["tot_tarjetas"] - tot_tarjetas["tarjetas_validas"]
    ) / tot_tarjetas["len_tarjetas"]

    tarjetas = tarjetas.merge(tot_tarjetas[["dia", "diff_tarjetas"]])

    tarjetas["factor_expansion_tarjeta"] = (
        tarjetas.factor_expansion_original + tarjetas.diff_tarjetas
    ) * tarjetas.od_validado

    etapas = etapas.merge(
        tarjetas[["dia", "id_tarjeta", "factor_expansion_tarjeta"]],
        on=["dia", "id_tarjeta"],
        how="left",
    )

    # Calibración de factor de expansión por línea

    print("Calibrando factores de expansion")

    etapas["od_validado_cadena"] = 1
    etapas.loc[etapas.factor_expansion_tarjeta == 0, "od_validado_cadena"] = 0

    factores_expansion_etapas = (
        transacciones_linea[["dia", "id_linea", "transacciones"]]
        .merge(
            etapas[(etapas.od_validado_cadena == 1)]
            .groupby(["dia", "id_linea"], as_index=False)
            .agg({"factor_expansion_tarjeta": "sum"})
        )
        .rename(columns={"factor_expansion_tarjeta": "transacciones_validas"})
    )

    factores_expansion_etapas["factor_expansion_linea"] = (
        factores_expansion_etapas["transacciones"]
        / factores_expansion_etapas["transacciones_validas"]
    )

    factores_expansion_etapas = etapas.loc[
        (etapas.od_validado_cadena == 1),
        [
            "id",
            "dia",
            "id_etapa",
            "id_viaje",
            "id_tarjeta",
            "id_linea",
            "factor_expansion_tarjeta",
        ],
    ].merge(
        factores_expansion_etapas[["dia", "id_linea", "factor_expansion_linea"]],
        how="left",
    )

    factores_expansion_etapas["factor_expansion_linea"] = (
        factores_expansion_etapas["factor_expansion_linea"]
        * factores_expansion_etapas["factor_expansion_tarjeta"]
    )

    etapas = etapas.merge(
        factores_expansion_etapas[["id", "factor_expansion_linea"]], on="id", how="left"
    )

    etapas.loc[etapas.factor_expansion_linea.isna(), "factor_expansion_linea"] = 0

    etapas = etapas.drop(["od_validado_cadena"], axis=1)

    # Si la cadena de viajes no está validada le fuera el od_validado a 0
    etapas.loc[etapas.factor_expansion_linea == 0, "od_validado"] = 0

    etapas = etapas.sort_values("id").reset_index(drop=True)

    print("Actualizando tabla de etapas en la db...")

    # borro si ya existen etapas de una corrida anterior
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM etapas WHERE dia IN ({values})"

    conn.execute(query)
    conn.commit()

    # etapas.to_sql("etapas", conn, if_exists="append", index=False)
    chunk_size = 400000  # Número de registros por chunk

    # Subir los datos por partes
    for i in range(0, len(etapas), chunk_size):
        etapas_chunk = etapas.iloc[i : i + chunk_size]
        etapas_chunk.to_sql("etapas", conn, if_exists="append", index=False)

    print(f"Creando tabla de viajes de {len(etapas)} etapas")
    # Crear tabla viajes
    etapas = pd.concat([etapas, pd.get_dummies(etapas.modo)], axis=1)

    etapas = etapas.sort_values(["dia", "id_tarjeta", "id_viaje", "id_etapa"])

    # Guarda viajes y usuarios en sqlite
    agg_func_dict = {
        "tiempo": "first",
        "hora": "first",
        "h3_o": "first",
        "h3_d": "last",
        "genero": "first",
        "tarifa": "first",
        "od_validado": "min",
        "factor_expansion_linea": "mean",
        "factor_expansion_tarjeta": "mean",
    }
    viajes = etapas.groupby(
        ["dia", "id_tarjeta", "id_viaje"],
        as_index=False,
    ).agg(agg_func_dict)

    print("Clasificando modalidad...")

    cols = pd.get_dummies(etapas.modo).columns.tolist()

    viajes = viajes.merge(
        etapas.groupby(
            ["dia", "id_tarjeta", "id_viaje"],
            as_index=False,
        )[cols].max()
    )
    # Sumar cantidad de etapas por modo
    viajes["tmp_cant_modos"] = viajes[cols].sum(axis=1)

    print(cols)
    viajes["modo"] = ""

    for i in cols:
        viajes.loc[viajes[i] == 1, "modo"] = i

    viajes.loc[viajes.tmp_cant_modos > 1, "modo"] = "Multimodal"
    viajes = viajes.drop(cols, axis=1)

    viajes = viajes.merge(
        etapas.groupby(
            ["dia", "id_tarjeta", "id_viaje"],
            as_index=False,
        )[cols].sum()
    )
    viajes["cant_etapas"] = viajes[cols].sum(axis=1)

    # Clasificar los viajes como Multimodal o Multietapa
    viajes.loc[(viajes.cant_etapas > 1) & (viajes.modo != "Multimodal"), "modo"] = (
        "Multietapa"
    )

    viajes_cols = [
        "id_tarjeta",
        "id_viaje",
        "dia",
        "tiempo",
        "hora",
        "cant_etapas",
        "modo",
        "autobus",
        "tren",
        "metro",
        "tranvia",
        "brt",
        "cable",
        "lancha",
        "otros",
        "h3_o",
        "h3_d",
        "genero",
        "tarifa",
        "od_validado",
        "factor_expansion_linea",
        "factor_expansion_tarjeta",
    ]

    viajes = viajes.reindex(columns=viajes_cols)

    print("Subiendo tabla de viajes a la db...")

    # borro si ya existen viajes de una corrida anterior
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM viajes WHERE dia IN ({values})"

    conn.execute(query)
    conn.commit()
    # viajes.to_sql("viajes", conn, if_exists="append", index=False)

    chunk_size = 400000  # Número de registros por chunk

    # Subir los datos por partes
    for i in range(0, len(viajes), chunk_size):
        viajes_chunk = viajes.iloc[i : i + chunk_size]
        viajes_chunk.to_sql("viajes", conn, if_exists="append", index=False)

    print("Creando tabla de usuarios...")
    usuarios = (
        viajes.groupby(["dia", "id_tarjeta"], as_index=False)
        .agg(
            {
                "od_validado": "min",
                "id_viaje": "count",
                "factor_expansion_linea": "mean",
                "factor_expansion_tarjeta": "mean",
            }
        )
        .rename(columns={"id_viaje": "cant_viajes"})
    )

    print("Subiendo tabla de usuarios a la db...")

    # borro si ya existen etapas de una corrida anterior
    values = ", ".join([f"'{val}'" for val in dias_ultima_corrida["dia"]])
    query = f"DELETE FROM usuarios WHERE dia IN ({values})"
    conn.execute(query)
    conn.commit()

    # Guarda viajes y usuarios en sqlite
    usuarios.to_sql("usuarios", conn, if_exists="append", index=False)
    print("Fin de creacion de tablas viajes y usuarios")

    conn.close()


@duracion
def rearrange_trip_id_same_od():
    """
    Takes a legs dataframe with legs and trips id and splits
    trips with same id into 2 trips with different ids and uploads
    new legs to the db

    Parameters
    ----------
    df : pandas DataFrame
        legs dataframe

    Returns
    ----------

    pandas DataFrame
        legs with new trips ids

    """
    conn_data = iniciar_conexion_db(tipo="data")

    print("Leer etapas")

    dias_ultima_corrida = pd.read_sql_query(
        """
                                SELECT *
                                FROM dias_ultima_corrida
                                """,
        conn_data,
    )

    df = pd.read_sql_query(
        """
                                SELECT e.*
                                FROM etapas e
                                JOIN dias_ultima_corrida d
                                ON e.dia = d.dia
                                """,
        conn_data,
    )

    print("Crear nuevos ids cuando un mismo viaje se hace en una misma línea")

    cols_df = df.columns.tolist()

    # Ordenar el DataFrame para procesarlo secuencialmente
    # 1) Ordenar
    df = df.sort_values(
        by=["dia", "id_tarjeta", "id_viaje", "tiempo", "hora", "id_etapa"]
    ).reset_index(drop=True)

    # 2) Calcular la línea anterior dentro de cada grupo
    df["id_linea_anterior"] = df.groupby(["dia", "id_tarjeta", "id_viaje"])[
        "id_linea"
    ].shift()

    # 3) Crear columna booleana: True si es la misma línea que la anterior, False en caso contrario
    df["es_igual"] = df["id_linea"] == df["id_linea_anterior"]

    # 4) Hacer la suma acumulada dentro de cada grupo
    df["sum_id_viaje"] = df.groupby(["dia", "id_tarjeta"])["es_igual"].cumsum()

    df["id_viaje"] = df.id_viaje + df.sum_id_viaje
    df["id_etapa"] = df.groupby(["dia", "id_tarjeta", "id_viaje"]).cumcount() + 1

    # Borrar columnas auxiliares
    df.drop(columns=["es_igual", "id_linea_anterior", "sum_id_viaje"], inplace=True)

    # Corrige viajes con origen y destino iguales
    print("Corrige viajes con origen y destino iguales")

    # Crear tabla temporal para detectar viajes con el mismo OD
    df_viajes = df.groupby(["dia", "id_tarjeta", "id_viaje"], as_index=False).agg(
        h3_o=("h3_o", "first"),  # Primer origen
        h3_d=("h3_d", "last"),  # Último destino
        od_validado=("od_validado", "min"),  # Validación mínima del viaje
        cant_etapas=("id_etapa", "count"),  # Número de etapas en el viaje
        id_etapa=("id_etapa", "last"),  # Última etapa del viaje
    )

    # Filtrar viajes con el mismo OD
    mask = (
        (df_viajes.h3_o == df_viajes.h3_d)
        & (df_viajes.od_validado == 1)
        & (df_viajes.cant_etapas > 1)
    )

    # Seleccionar solo las tarjetas y días con problemas
    df_viajes_problemas = df_viajes.loc[
        mask, ["dia", "id_tarjeta", "id_viaje", "id_etapa"]
    ]
    df_viajes_problemas["mismo_od"] = 1

    # Filtrar las etapas originales con problemas
    df = df.merge(
        df_viajes_problemas[["dia", "id_tarjeta", "id_viaje", "mismo_od"]],
        on=["dia", "id_tarjeta", "id_viaje"],
        how="left",
    )
    df["mismo_od"] = df["mismo_od"].fillna(0)

    df["con_problemas"] = df.groupby(["dia", "id_tarjeta"])["mismo_od"].transform("max")

    df_ok = df[df.con_problemas == 0].copy()
    df_viajes_problemas = df[df.con_problemas == 1].copy()

    # 3) Crear columna booleana: True si es la misma línea que la anterior, False en caso contrario
    df_viajes_problemas["es_igual"] = 0
    df_viajes_problemas.loc[
        (df_viajes_problemas.mismo_od == 1) & (df_viajes_problemas.id_etapa != 1),
        "es_igual",
    ] = 1

    # 4) Hacer la suma acumulada dentro de cada grupo
    df_viajes_problemas["sum_id_viaje"] = df_viajes_problemas.groupby(
        ["dia", "id_tarjeta"]
    )["es_igual"].cumsum()

    df_viajes_problemas["id_viaje"] = (
        df_viajes_problemas.id_viaje + df_viajes_problemas.sum_id_viaje
    )
    df_viajes_problemas["id_etapa"] = (
        df_viajes_problemas.groupby(["dia", "id_tarjeta", "id_viaje"]).cumcount() + 1
    )

    df = pd.concat([df_ok, df_viajes_problemas], ignore_index=True)
    df = df.sort_values(by=["dia", "id_tarjeta", "id_viaje", "id_etapa"]).reset_index(
        drop=True
    )

    # Borrar columnas auxiliares
    df = df[cols_df]

    guardar_tabla_sql(df, "etapas", "data", {"dia": df.dia.unique().tolist()})


@duracion
def compute_trips_travel_time():
    """
    This function reads from legs travel time in gps and stations
    and computes travel times for trips
    """

    conn_data = iniciar_conexion_db(tipo="data")

    print("Insertando tiempos de viaje a etapas en base a gps y estaciones")

    q = """
    INSERT INTO travel_times_legs (dia, id, id_tarjeta, id_etapa, id_viaje, travel_time_min)
    SELECT e.dia, e.id,e.id_tarjeta,  e.id_etapa,e.id_viaje,
    (ifnull(tg.travel_time_min,0) + ifnull(ts.travel_time_min,0)) tt
    FROM etapas e
    JOIN dias_ultima_corrida d
    ON e.dia = d.dia
    LEFT JOIN travel_times_gps tg
    ON e.id = tg.id
    LEFT JOIN travel_times_stations ts
    ON e.id = ts.id
    WHERE e.od_validado = 1
    AND (tg.travel_time_min IS NOT NULL OR ts.travel_time_min IS NOT NULL)
    """
    conn_data.execute(q)
    conn_data.commit()

    print("Insertando tiempos de viaje a viajes en base a etapas")
    q = """
    INSERT INTO travel_times_trips (dia, id_tarjeta, id_viaje, travel_time_min)
    SELECT tt.dia, tt.id_tarjeta,tt.id_viaje, sum(tt.travel_time_min) AS travel_time_min 
    FROM travel_times_legs tt
    JOIN dias_ultima_corrida d
    ON tt.dia = d.dia
    GROUP BY tt.dia, tt.id_tarjeta,tt.id_viaje ;
    """
    conn_data.execute(q)
    conn_data.commit()


def add_distance_and_travel_time():
    """
    This function reads trips data and adds distances and travel times
    from the distances table. It also computes the travel speed.
    """

    print("Agregando distancias y tiempos de viaje a los viajes")
    conn_data = iniciar_conexion_db(tipo="data")

    # read unprocessed data from legs

    q = """
        select v.id_tarjeta, v.id_viaje, v.dia, v.h3_d, v.h3_o
        from viajes v
        JOIN dias_ultima_corrida d
        ON v.dia = d.dia
        where od_validado = 1
        ;
    """
    print("Leyendo datos de demanda")
    trips = pd.read_sql(q, conn_data)
    trips = add_distances_to_legs(legs=trips)

    trips.to_sql(
        "temp_distancias",
        conn_data,
        if_exists="replace",
        index=False,
    )
    print("Actualizando distancias a etapas")

    q_update = """
    UPDATE viajes
    SET distancia = temp_distancias.distance
    FROM temp_distancias
    WHERE viajes.id_tarjeta = temp_distancias.id_tarjeta
    AND viajes.id_viaje = temp_distancias.id_viaje
    AND viajes.dia = temp_distancias.dia;
    """
    cur = conn_data.cursor()
    cur.execute(q_update)
    conn_data.commit()

    print("Actualizando tiempos de viaje a etapas")

    q_update = """
    UPDATE viajes
    SET travel_time_min = travel_times_legs.travel_time_min
    FROM travel_times_legs
    WHERE viajes.id_tarjeta = travel_times_legs.id_tarjeta
    AND viajes.id_viaje = travel_times_legs.id_viaje
    AND viajes.dia = travel_times_legs.dia;
    """
    cur = conn_data.cursor()
    cur.execute(q_update)
    conn_data.commit()

    q = """
    drop table temp_distancias;
    """
    cur.execute(q)
    conn_data.commit()
    conn_data.close()
