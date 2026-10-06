# Importación de librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import folium
import os


from urbantrips.utils.utils import levanto_tabla_sql


def normalizar_id_linea(col):
    def convertir(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        # si es número entero o decimal → convertir a int y luego a str
        if s.replace(".", "", 1).isdigit():
            return str(int(float(s)))
        # si no es número → dejarlo como está
        return s

    return col.apply(convertir)


def correlation_analysis(
    data,
    vars,
    title="Matriz correlación",
    nombre_archivo="",
    fsize=(7, 4),
    output_path=Path() / "data" / "clusters" / "resultados",
):
    """
    Realiza el análisis de correlación entre las variables seleccionadas y muestra la matriz de correlación.

    Parámetros:
    - data: DataFrame con los datos
    - vars: Lista de variables para el análisis de correlación
    """
    corr_matrix = data[vars].corr()

    fig, ax = plt.subplots(figsize=fsize)

    # Generar el heatmap sobre el objeto ax
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)

    # Título del gráfico
    plt.title(title)

    # Crear el directorio si no existe

    output_path.mkdir(parents=True, exist_ok=True)

    # Guardar el gráfico como PNG
    if len(nombre_archivo) > 0:
        archivo_guardado = output_path / nombre_archivo
        print(f"Guardando el archivo en: {archivo_guardado}")
        plt.savefig(archivo_guardado, dpi=300, bbox_inches="tight", pad_inches=0.1)

    # Mostrar el gráfico
    plt.show()

    print("\nInterpretación de la Matriz de Correlación:")
    print("- Valores cercanos a 1 o -1 indican alta correlación positiva o negativa.")
    print("- Variables altamente correlacionadas pueden redundar información.")
    print(corr_matrix)

    if len(nombre_archivo) > 0:
        nombre_archivo = nombre_archivo.replace(".png", ".xlsx")
        print(nombre_archivo)
        corr_matrix.to_excel(output_path / nombre_archivo, index=False)
    return corr_matrix


def cluster_profile(
    data,
    cluster_label,
    eval_vars,
    n_cols=5,
    n=0,
    filepath=Path() / "data" / "clusters" / "resultados",
):
    """
    Muestra el perfil de cada cluster en términos de las variables de evaluación.

    Parámetros:
    - data: DataFrame con los datos y etiquetas de cluster
    - cluster_label: Nombre de la columna con las etiquetas de cluster
    - eval_vars: Lista de variables para evaluar los clusters
    """
    data["casos"] = 1
    cluster_sum = data.groupby(cluster_label, as_index=False).casos.sum()
    if f"{cluster_label}_original" in data.columns:
        cluster_summary = (
            data.groupby([cluster_label, f"{cluster_label}_original"], as_index=False)[
                eval_vars
            ]
            .mean()
            .round(2)
        )
    else:
        cluster_summary = (
            data.groupby(cluster_label, as_index=False)[eval_vars].mean().round(2)
        )

    cluster_summary = cluster_sum.merge(cluster_summary)

    print(f"\nPerfil de clusters basado en variables de evaluación:")

    guardar_tabla_como_png(
        cluster_summary,
        f"escenario{n+1}_2_tabla.png",
        f"Perfil de clusters basado en variables de evaluación (Escenario {n+1})",
        filepath=filepath,
    )

    # Boxplots de variables de evaluación por cluster
    n_vars = len(eval_vars)

    n_rows = math.ceil(n_vars / n_cols)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows)
    )
    axes = axes.flatten()
    for i, var in enumerate(eval_vars):
        sns.boxplot(x=cluster_label, y=var, data=data, ax=axes[i])
        axes[i].set_title(f"{var.capitalize()}")
    plt.tight_layout()
    plt.show()

    fig.savefig(
        filepath / f"escenario{n+1}_3_boxplot.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def ordernar_clusters(data_clustered, eval_vars, cluster_var):
    data_ordered = (
        data_clustered.groupby(cluster_var, as_index=False)[
            [eval_vars[0], eval_vars[1]]
        ]
        .mean()
        .sort_values(
            [
                eval_vars[0],
                eval_vars[1],
            ]
        )
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": f"{cluster_var}_ordered"})[
            [cluster_var, f"{cluster_var}_ordered"]
        ]
    )
    data_clustered = data_clustered.merge(data_ordered, on=cluster_var)
    data_clustered = data_clustered.rename(
        columns={cluster_var: f"{cluster_var}_original"}
    )
    data_clustered = data_clustered.rename(
        columns={f"{cluster_var}_ordered": cluster_var}
    )
    return data_clustered


import folium
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from urbantrips.viz import basemaps


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def plot_cluster_in_map(
    carto,
    data,
    cluster_var,
    archivo_salida=None,
    filepath=Path() / "data" / "clusters" / "resultados",
):
    """
    Genera un mapa de clusters y opcionalmente lo guarda como archivo HTML.

    Parámetros:
        carto (GeoDataFrame): Geometrías de las líneas.
        data (DataFrame): Datos con la variable de cluster.
        cluster_var (str): Nombre de la columna de cluster.
        archivo_salida (str, opcional): Ruta para guardar el archivo HTML.
    """
    # Merge
    # carto = carto.merge(data.reindex(columns=["id_linea", cluster_var]), on="id_linea")

    if "id_linea" in carto.columns:
        carto["id_linea"] = normalizar_id_linea(carto["id_linea"])
        carto["id_linea"] = carto["id_linea"].astype(str)
    if "id_linea" in data.columns:
        data["id_linea"] = normalizar_id_linea(data["id_linea"])
        data["id_linea"] = data["id_linea"].astype(str)

    carto = carto.merge(data.reindex(columns=["id_linea", cluster_var]), on="id_linea")

    # Inicializar mapa
    m = basemaps.folium_map(
        location=(-34.6, -58.5),
        zoom_start=12,
        width=1300,
        height=800,
    )

    # Obtener lista de clusters únicos (excluyendo ruido si fuera el caso)
    clusters_unicos = sorted(carto[cluster_var].dropna().unique())
    clusters_unicos = [c for c in clusters_unicos if c != -1]

    # Preparar colores dinámicamente según la cantidad de clusters
    n_clusters = len(clusters_unicos)
    base_colors = [
        rgb_to_hex(colormaps["tab20"](i / n_clusters)) for i in range(n_clusters)
    ]

    # Crear mapeo cluster -> color
    cluster_colors = {
        cluster: base_colors[idx] for idx, cluster in enumerate(clusters_unicos)
    }

    # Crear las capas por cluster
    for cluster, color in cluster_colors.items():
        carto.query(f"{cluster_var} == {cluster}").explore(
            name=str(cluster), m=m, color=color, legend=False
        )

    m.add_child(folium.LayerControl())

    # Guardar como HTML si se especifica
    if archivo_salida:
        m.save(filepath / archivo_salida)

    return m


def generar_datos(df, variables, cluster_col):
    save_data = (
        """

Los clusters fueron creados a partir de las siguientes variables operativas:
"""
        + ", ".join(variables)
        + """

---

Resultados del análisis"""
    )

    # Agrupar por cluster y calcular estadísticas
    resumen = df.groupby(cluster_col)[variables].agg(
        [
            ("count", "count"),
            ("mean", "mean"),
            ("median", "median"),
            ("std", "std"),
            ("min", "min"),
            ("Q1", lambda x: x.quantile(0.25)),
            ("Q3", lambda x: x.quantile(0.75)),
            ("IQR", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            ("max", "max"),
        ]
    )

    # Generar texto formateado
    for cluster in resumen.index:
        save_data += f"**Cluster {cluster}:**\n"
        save_data += (
            f"- Cantidad de casos: {resumen.loc[cluster, (variables[0], 'count')]}\n"
        )

        for var in variables:
            stats = resumen.loc[cluster, var]
            save_data += f"- **{var}**:\n"
            save_data += f"  - Media: {stats['mean']:.2f}\n"
            save_data += f"  - Mediana: {stats['median']:.2f}\n"
            save_data += f"  - Desviación estándar: {stats['std']:.2f}\n"
            save_data += f"  - Mínimo: {stats['min']:.2f}\n"
            save_data += f"  - Q1: {stats['Q1']:.2f}\n"
            save_data += f"  - Q3: {stats['Q3']:.2f}\n"
            save_data += f"  - IQR: {stats['IQR']:.2f}\n"

    return save_data


def guardar_datos_txt(
    save_data,
    nombre_archivo="resumen_clusters.txt",
    filepath=Path() / "data" / "clusters" / "resultados",
):
    with open(filepath / nombre_archivo, "w", encoding="utf-8") as file:
        file.write(save_data)


def hierarchical_clustering(
    df,
    variables,
    n_clusters=None,
    linkage="ward",
    plot_dendrogram=False,
    var="hcluster",
):
    """
    Aplica clustering jerárquico a un dataframe y devuelve el dataframe con una nueva columna de cluster.

    Parámetros:
    - df: DataFrame de pandas con los datos.
    - variables: Lista de columnas a utilizar para el clustering.
    - n_clusters: Número de clusters a generar (si None, se sugiere usar el dendrograma).
    - linkage: Método de enlace ('ward', 'complete', 'average', 'single').
    - plot_dendrogram: Booleano para visualizar el dendrograma antes de aplicar clustering.
    - Usa distancia euclidiana por defecto

    Retorna:
    - DataFrame original con una nueva columna 'cluster' asignando cada punto a un cluster.
    """
    # Escalar los datos para mejorar la separación de clusters
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[variables])

    # Visualización del dendrograma si se requiere
    if plot_dendrogram:
        plt.figure(figsize=(20, 10))
        sch.dendrogram(sch.linkage(data_scaled, method=linkage))
        plt.title("Dendrograma para determinar el número de clusters")
        plt.xlabel("Observaciones")
        plt.ylabel("Distancia")
        # plt.savefig('dendograma.png')
        plt.show()

    # Si no se especifica número de clusters, sugerimos inspeccionar el dendrograma
    if n_clusters is None:
        # raise ValueError("Debes definir 'n_clusters' o visualizar el dendrograma para seleccionar un valor adecuado.")
        print(
            "Hay que definir el número de clusters. Visualizar el dendrograma para seleccionar un valor adecuado."
        )
    else:
        # Aplicar clustering jerárquico
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        # df[var] = cluster_model.fit_predict(data_scaled)
        df.loc[:, var] = cluster_model.fit_predict(data_scaled)

    return df


def guardar_tabla_como_png(
    df, nombre_archivo, titulo, filepath=Path() / "data" / "clusters" / "resultados"
):
    """
    Genera y guarda una tabla en formato PNG a partir de un DataFrame, con un título cercano a la tabla y sin espacio excesivo.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene la tabla ya agrupada.
        nombre_archivo (str): Nombre del archivo de salida (ej. 'tabla.png').
        titulo (str): Título que se mostrará sobre la tabla.
    """
    # Estimar el tamaño de la figura según cantidad de filas y columnas
    filas, columnas = df.shape
    fig_height = 0.5 + filas * 0.3
    fig_width = 1 + columnas * 2

    # Crear figura ajustada
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")

    # Crear tabla
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )

    # Ajustar tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([i for i in range(len(df.columns))])

    # Añadir título muy pegado
    plt.title(titulo, fontsize=12, weight="bold", pad=3)

    # Guardar sin margen extra
    plt.savefig(filepath / nombre_archivo, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def correr_clusters(data):

    carto = levanto_tabla_sql("lines_geoms", "insumos")
    # usar solo los sentidos de ida
    carto = carto.loc[carto.direction == 0, :]

    escenarios_clusterizacion = levanto_tabla_sql(
        "escenarios_clusterizacion", "insumos"
    )
    data = data[data.modo == "Autobus"]

    cluster_vars = [
        [v.strip() for v in fila.split(",") if v.strip()]
        for fila in escenarios_clusterizacion["variables"]
    ]
    eval_vars = cluster_vars
    n_clusters = escenarios_clusterizacion.cant_clusters.tolist()
    max_clase = escenarios_clusterizacion.max_clusters_clase.tolist()
    n_clusters_explotados = escenarios_clusterizacion.cant_clusters_recluster.tolist()

    filepath = Path() / "data" / "clusters" / f"resultados"

    clusters_result = pd.DataFrame([])

    for n in range(0, len(n_clusters)):
        print("Escenario", n + 1, cluster_vars[n])

        lineas_new = data[
            ["dia", "mes", "id_linea", "nombre_linea", "empresa", "modo"]
            + cluster_vars[n]
        ].copy()
        lineas_new = lineas_new.dropna()

        corr_matrix = correlation_analysis(
            lineas_new,
            cluster_vars[n],
            nombre_archivo=f"escenario{n+1}_1_corr.png",
            fsize=(8, 4),
            output_path=filepath,
        )
        data_hierarchical = hierarchical_clustering(
            data.copy(),
            cluster_vars[n],
            n_clusters=n_clusters[n],
            linkage="ward",
            plot_dendrogram=False,
            var="hcluster",
        )

        data_hierarchical["hcluster"] = data_hierarchical["hcluster"].astype(str)
        eval_h = (
            data_hierarchical.groupby("hcluster")
            .size()
            .reset_index()
            .rename(columns={0: "cant"})
        )

        # Explota cluster
        for i in eval_h[eval_h.cant > max_clase[n]].hcluster:
            data_tmp = hierarchical_clustering(
                data_hierarchical[data_hierarchical.hcluster == i].copy(),
                cluster_vars[n],
                n_clusters=n_clusters_explotados[n],
                linkage="ward",
                plot_dendrogram=False,
                var=f"hcluster_{i}",
            )

            data_tmp[f"hcluster_{i}"] = data_tmp[f"hcluster_{i}"].astype(str)

            if "id_linea" in data_hierarchical.columns:
                data_hierarchical["id_linea"] = normalizar_id_linea(
                    data_hierarchical["id_linea"]
                )
                data_hierarchical["id_linea"] = data_hierarchical["id_linea"].astype(
                    str
                )
            if "id_linea" in data_tmp.columns:
                data_tmp["id_linea"] = normalizar_id_linea(data_tmp["id_linea"])
                data_tmp["id_linea"] = data_tmp["id_linea"].astype(str)

            data_hierarchical = data_hierarchical.merge(
                data_tmp[["dia", "id_linea", f"hcluster_{i}"]], how="left"
            )
            data_hierarchical.loc[
                data_hierarchical[f"hcluster_{i}"].notna(), "hcluster"
            ] = (
                data_hierarchical.loc[
                    data_hierarchical[f"hcluster_{i}"].notna(), "hcluster"
                ]
                + "_"
                + data_hierarchical.loc[
                    data_hierarchical[f"hcluster_{i}"].notna(), f"hcluster_{i}"
                ]
            )
            data_hierarchical = data_hierarchical.drop([f"hcluster_{i}"], axis=1)

        # Ordenar clusters
        data_hierarchical = ordernar_clusters(
            data_hierarchical, eval_vars[n], "hcluster"
        )

        if "id_linea" in lineas_new.columns:
            lineas_new["id_linea"] = normalizar_id_linea(lineas_new["id_linea"])
            lineas_new["id_linea"] = lineas_new["id_linea"].astype(str)
        if "id_linea" in data_hierarchical.columns:
            data_hierarchical["id_linea"] = normalizar_id_linea(
                data_hierarchical["id_linea"]
            )
            data_hierarchical["id_linea"] = data_hierarchical["id_linea"].astype(str)

        lineas_new = lineas_new.merge(
            data_hierarchical[
                ["dia", "id_linea", "hcluster", "hcluster_original"]
            ].rename(
                columns={
                    "hcluster": f"escenario{n+1}_hcluster",
                    "hcluster_original": f"escenario{n+1}_hcluster_original",
                }
            ),
            how="left",
        )

        if len(clusters_result) == 0:
            clusters_result = lineas_new.copy()
        else:

            if "id_linea" in lineas_new.columns:
                lineas_new["id_linea"] = normalizar_id_linea(lineas_new["id_linea"])
                lineas_new["id_linea"] = lineas_new["id_linea"].astype(str)
            if "id_linea" in clusters_result.columns:
                clusters_result["id_linea"] = normalizar_id_linea(
                    clusters_result["id_linea"]
                )
                clusters_result["id_linea"] = clusters_result["id_linea"].astype(str)

            clusters_result = clusters_result.merge(
                lineas_new[
                    [
                        "dia",
                        "id_linea",
                        f"escenario{n+1}_hcluster",
                        f"escenario{n+1}_hcluster_original",
                    ]
                ]
            )
        cluster_profile(
            lineas_new,
            f"escenario{n+1}_hcluster",
            eval_vars[n],
            n_cols=len(eval_vars[n]),
            n=n,
            filepath=filepath,
        )

        mapa = plot_cluster_in_map(
            carto=carto,
            data=lineas_new,
            cluster_var=f"escenario{n+1}_hcluster",
            archivo_salida=f"html_escenario{n+1}_map.html",
            filepath=filepath,
        )

        resumen = generar_datos(lineas_new, cluster_vars[n], f"escenario{n+1}_hcluster")
        guardar_datos_txt(resumen, f"escenario{n+1}_4_datos.txt", filepath)

    os.makedirs(filepath, exist_ok=True)

    if "id_linea" in data.columns:
        data["id_linea"] = normalizar_id_linea(data["id_linea"])
        data["id_linea"] = data["id_linea"].astype(str)
    if "id_linea" in clusters_result.columns:
        clusters_result["id_linea"] = normalizar_id_linea(clusters_result["id_linea"])
        clusters_result["id_linea"] = clusters_result["id_linea"].astype(str)

    data = data.merge(clusters_result, how="left")
    data.to_csv(filepath / "clusters_lineas.csv", index=False)
    return data
