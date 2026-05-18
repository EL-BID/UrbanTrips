import numpy as np
import multiprocessing
import pandas as pd
from datetime import datetime
from urbantrips.utils.utils import (
    duracion,
    iniciar_conexion_db,
    levanto_tabla_sql,
    guardar_tabla_sql,
)
from urbantrips.carto.compute_distances import compute_od_distances

@duracion
def create_trips_from_legs_and_fex():
    """
    Loads the legs table from db, updates expansion factors and produces
    trips and users tables.

    Produces 3 expansion factors:
    1. factor_expansion_etapa: expands individually validated legs
       (etapa_validada==1) to match total weighted legs per line.
    2. factor_expansion_tarjeta: redistributes weight from cards with
       invalid OD chains to valid ones, preserving total cards.
    3. factor_expansion_linea: expands legs from fully validated trip
       chains (od_validado==1) to match total weighted legs per line,
       then calibrates against reported transactions per line.
    """

    conn = iniciar_conexion_db(tipo="data")

    # ------------------------------------------------------------------
    # 1. Lectura de insumos
    # ------------------------------------------------------------------
    # print("Leyendo datos de etapas para producir viajes")

    dias_ultima_corrida = pd.read_sql_query(
        "SELECT * FROM dias_ultima_corrida", conn
    )

    etapas = pd.read_sql_query(
        """
        SELECT e.*
        FROM etapas e
        JOIN dias_ultima_corrida d ON e.dia = d.dia
        """,
        conn,
    )

    transacciones_linea = pd.read_sql_query(
        """
        SELECT t.*
        FROM transacciones_linea t
        JOIN dias_ultima_corrida d ON t.dia = d.dia
        """,
        conn,
    )

    # Limpiar columnas de factores previos si existen
    cols_drop = [
        "factor_expansion_tarjeta",
        "factor_expansion_linea",
        "factor_expansion_etapa",
        "etapa_validada",
    ]
    etapas = etapas.drop(
        columns=[c for c in cols_drop if c in etapas.columns], errors="ignore"
    )

    # ------------------------------------------------------------------
    # 2. Validaciones previas
    # ------------------------------------------------------------------

    # Forzar od_validado = 0 si lat y lon son ambos 0
    etapas.loc[
        (etapas["latitud"] == 0) & (etapas["longitud"] == 0),
        "od_validado",
    ] = 0
    
    etapas.loc[
        (etapas.distance_od == 0) | (etapas["distance_od"].isna()),
        "od_validado",
    ] = 0

    # Guardar validación individual de la etapa antes de aplicar
    # la lógica de cadena que puede sobreescribir od_validado
    etapas["etapa_validada"] = etapas["od_validado"]

    # ------------------------------------------------------------------
    # 3. Factor de expansión por etapa validada
    #    Expande etapas con etapa_validada==1 para que por línea sumen
    #    el total de factor_expansion_original de TODAS las etapas de
    #    esa línea (validadas y no validadas).
    # ------------------------------------------------------------------
    print('Calculando factores de expansión por etapa, línea y tarjeta')
    # print("Calculando factor_expansion_etapa")

    peso_total_linea = (
        etapas.groupby(["dia", "id_linea"], as_index=False)
        .agg(peso_total=("factor_expansion_original", "sum"))
    )

    etapas_validas_linea = (
        etapas[etapas["etapa_validada"] == 1]
        .groupby(["dia", "id_linea"], as_index=False)
        .agg(peso_validas=("factor_expansion_original", "sum"))
    )

    factor_etapa = peso_total_linea.merge(
        etapas_validas_linea, on=["dia", "id_linea"]
    )
    factor_etapa["ratio_etapa"] = (
        factor_etapa["peso_total"] / factor_etapa["peso_validas"]
    )

    etapas = etapas.merge(
        factor_etapa[["dia", "id_linea", "ratio_etapa"]],
        on=["dia", "id_linea"],
        how="left",
    )

    etapas["factor_expansion_etapa"] = (
        etapas["factor_expansion_original"]
        * etapas["ratio_etapa"].fillna(0)
        * etapas["etapa_validada"]
    )
    etapas = etapas.drop(columns=["ratio_etapa"])

    # ------------------------------------------------------------------
    # 4. Factor de expansión por tarjeta
    #    Redistribuye el peso expandido de tarjetas con cadena inválida
    #    hacia las tarjetas con cadena válida. Si cualquier etapa de la
    #    tarjeta tiene od_validado==0, toda la tarjeta se invalida.
    # ------------------------------------------------------------------
    # print("Calculando factor_expansion_tarjeta")

    tarjetas = etapas.groupby(["dia", "id_tarjeta"], as_index=False).agg(
        factor_expansion_original=("factor_expansion_original", "mean"),
        od_validado=("od_validado", "min"),
    )

    peso_total = tarjetas.groupby("dia", as_index=False).agg(
        peso_total=("factor_expansion_original", "sum")
    )

    peso_valido = (
        tarjetas[tarjetas["od_validado"] == 1]
        .groupby("dia", as_index=False)
        .agg(peso_valido=("factor_expansion_original", "sum"))
    )

    ajuste_tarjeta = peso_total.merge(peso_valido, on="dia")
    ajuste_tarjeta["ratio_tarjeta"] = (
        ajuste_tarjeta["peso_total"] / ajuste_tarjeta["peso_valido"]
    )

    tarjetas = tarjetas.merge(ajuste_tarjeta[["dia", "ratio_tarjeta"]], on="dia")

    tarjetas["factor_expansion_tarjeta"] = (
        tarjetas["factor_expansion_original"]
        * tarjetas["ratio_tarjeta"]
        * tarjetas["od_validado"]
    )

    etapas = etapas.merge(
        tarjetas[["dia", "id_tarjeta", "factor_expansion_tarjeta"]],
        on=["dia", "id_tarjeta"],
        how="left",
    )
    etapas["factor_expansion_tarjeta"] = etapas["factor_expansion_tarjeta"].fillna(0)

    # Sobreescribir od_validado a nivel etapa: si la cadena es inválida,
    # od_validado pasa a 0 (etapa_validada conserva el valor individual)
    etapas.loc[etapas["factor_expansion_tarjeta"] == 0, "od_validado"] = 0

    # ------------------------------------------------------------------
    # 5. Factor de expansión por línea (cadena validada)
    #    Paso A: expande etapas con od_validado==1 para que por línea
    #    sumen el total de factor_expansion_original de todas las etapas.
    #    Paso B: calibra contra transacciones_linea.
    # ------------------------------------------------------------------
    # print("Calculando factor_expansion_linea")

    # Paso A: redistribuir peso de etapas con cadena inválida
    peso_total_linea_od = (
        etapas.groupby(["dia", "id_linea"], as_index=False)
        .agg(peso_total=("factor_expansion_original", "sum"))
    )

    peso_od_validas = (
        etapas[etapas["od_validado"] == 1]
        .groupby(["dia", "id_linea"], as_index=False)
        .agg(peso_validas=("factor_expansion_original", "sum"))
    )

    factor_linea = peso_total_linea_od.merge(
        peso_od_validas, on=["dia", "id_linea"]
    )
    factor_linea["ratio_od"] = (
        factor_linea["peso_total"] / factor_linea["peso_validas"]
    )

    # Paso B: calibrar contra transacciones por línea
    factor_linea = factor_linea.merge(
        transacciones_linea[["dia", "id_linea", "transacciones"]],
        on=["dia", "id_linea"],
        how="left",
    )
    factor_linea["ratio_linea"] = (
        factor_linea["transacciones"] / factor_linea["peso_total"]
    )

    # Ratio combinado
    factor_linea["ratio_final"] = (
        factor_linea["ratio_od"] * factor_linea["ratio_linea"].fillna(1)
    )

    etapas = etapas.merge(
        factor_linea[["dia", "id_linea", "ratio_final"]],
        on=["dia", "id_linea"],
        how="left",
    )

    etapas["factor_expansion_linea"] = (
        etapas["factor_expansion_original"]
        * etapas["ratio_final"].fillna(0)
        * etapas["od_validado"]
    )
    etapas = etapas.drop(columns=["ratio_final"])

    # ------------------------------------------------------------------
    # 6. Upload de etapas
    # ------------------------------------------------------------------
    etapas = etapas.sort_values("id").reset_index(drop=True)

    # print("Actualizando tabla de etapas en la db...")
    # _delete_dias(conn, "etapas", dias_ultima_corrida)
    # _upload_chunked(etapas, "etapas", conn)
    
    dias_ultima_corrida = levanto_tabla_sql("dias_ultima_corrida", "data")
    guardar_tabla_sql(
                etapas,
                "etapas",
                tabla_tipo="data",
                modo="append",
                filtros={"dia": dias_ultima_corrida["dia"].tolist()},
            )

    # ------------------------------------------------------------------
    # 7. Crear tabla de viajes
    # ------------------------------------------------------------------
    print(f"Creando tabla de viajes de {len(etapas)} etapas")

    modos = etapas["modo"].unique().tolist()
    modo_dummies = pd.get_dummies(etapas["modo"])

    etapas_con_modos = pd.concat([etapas, modo_dummies], axis=1)
    etapas_con_modos = etapas_con_modos.sort_values(
        ["dia", "id_tarjeta", "id_viaje", "id_etapa"]
    )

    agg_viajes = {
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

    grp = ["dia", "id_tarjeta", "id_viaje"]
    viajes = etapas_con_modos.groupby(grp, as_index=False).agg(agg_viajes)

    # Clasificación modal
    # print("Clasificando modalidad...")
    modo_max = etapas_con_modos.groupby(grp, as_index=False)[modos].max()
    modo_sum = etapas_con_modos.groupby(grp, as_index=False)[modos].sum()

    viajes = viajes.merge(modo_max, on=grp)

    viajes["tmp_cant_modos"] = viajes[modos].sum(axis=1)
    viajes["modo"] = ""
    for m in modos:
        viajes.loc[viajes[m] == 1, "modo"] = m
    viajes.loc[viajes["tmp_cant_modos"] > 1, "modo"] = "Multimodal"

    viajes = viajes.drop(columns=modos + ["tmp_cant_modos"])
    viajes = viajes.merge(modo_sum, on=grp)

    viajes["cant_etapas"] = viajes[modos].sum(axis=1)
    viajes.loc[
        (viajes["cant_etapas"] > 1) & (viajes["modo"] != "Multimodal"), "modo"
    ] = "Multietapa"

    # Columnas finales dinámicas
    viajes_cols = (
        ["id_tarjeta", "id_viaje", "dia", "tiempo", "hora", "cant_etapas", "modo"]
        + modos
        + [
            "h3_o", "h3_d", "genero", "tarifa", "od_validado",
            "factor_expansion_linea", "factor_expansion_tarjeta",
        ]
    )
    viajes = viajes.reindex(columns=[c for c in viajes_cols if c in viajes.columns])

    dias_ultima_corrida = levanto_tabla_sql("dias_ultima_corrida", "data")
    guardar_tabla_sql(
                viajes,
                "viajes",
                tabla_tipo="data",
                modo="append",
                filtros={"dia": dias_ultima_corrida["dia"].tolist()},
            )
    

    # ------------------------------------------------------------------
    # 8. Crear tabla de usuarios
    # ------------------------------------------------------------------
    # print("Creando tabla de usuarios...")
    usuarios = (
        viajes.groupby(["dia", "id_tarjeta"], as_index=False)
        .agg({
            "od_validado": "min",
            "id_viaje": "count",
            "factor_expansion_linea": "mean",
            "factor_expansion_tarjeta": "mean",
        })
        .rename(columns={"id_viaje": "cant_viajes"})
    )

    # _delete_dias(conn, "usuarios", dias_ultima_corrida)
    # _upload_chunked(usuarios, "usuarios", conn)
    dias_ultima_corrida = levanto_tabla_sql("dias_ultima_corrida", "data")
    guardar_tabla_sql(
                usuarios,
                "usuarios",
                tabla_tipo="data",
                modo="append",
                filtros={"dia": dias_ultima_corrida["dia"].tolist()},
            )

    # print("Fin de creacion de tablas viajes y usuarios")
    conn.close()
    print("========== Verificación de factores de expansión==========")
    print(f"factor_expansion_original total: {etapas.factor_expansion_original.sum():.0f}")
    print(f"factor_expansion_etapa total:    {etapas.factor_expansion_etapa.sum():.0f}")
    print(f"transacciones_linea total:       {transacciones_linea.transacciones.sum():.0f}")
    print(f"factor_expansion_linea total:    {etapas.factor_expansion_linea.sum():.0f}")
    print("========== Verificación de factores de expansión==========")


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
def add_distance_and_travel_time():
    """
    This function reads trips data and adds distances and travel times
    from the distances table. It also computes the travel speed.
    """

    # print("Agregando distancias y tiempos de viaje a los viajes")
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

    trips = pd.read_sql(q, conn_data)
    # trips = add_distances_to_legs(legs=trips)
    trips = compute_od_distances(
        od_df             = trips,
        origin_col        = "h3_o",
        dest_col          = "h3_d",
        distance_col      = 'distance',
        unit              = 'km',
        db_path           = "data/matriz_distancia/matriz_distancia.duckdb",
        network_cache_dir = "data/matriz_distancia",
        symmetric         = False,
        precompute_dist   = 50_000,   
        max_tile_deg      = 99,      
        verbose           = False
    )

    trips.to_sql(
        "temp_distancias",
        conn_data,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=10_000,
    )    

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

    # print("Actualizando tiempos de viaje a viajes")

    q_update = """
    UPDATE viajes
    SET travel_time_min = t.travel_time_min,
        distance_route = t.distance_route,
        distance_route_gps = t.distance_route_gps
    FROM (
        SELECT
            dia,
            id_tarjeta,
            id_viaje,
            SUM(COALESCE(travel_time_min, 0)) AS travel_time_min,
            SUM(distance_route) AS distance_route,
            SUM(distance_route_gps) AS distance_route_gps
        FROM travel_times_legs
        GROUP BY dia, id_tarjeta, id_viaje
    ) t
    WHERE viajes.dia = t.dia
    AND viajes.id_tarjeta = t.id_tarjeta
    AND viajes.id_viaje = t.id_viaje;
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
