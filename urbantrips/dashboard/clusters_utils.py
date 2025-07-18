# Importación de librerías necesarias
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr, shapiro, anderson, kstest
import scipy.cluster.hierarchy as sch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import folium
from palettable.colorbrewer.qualitative import Set3_12
import os








from urbantrips.utils.utils import levanto_tabla_sql, guardar_tabla_sql

def correlation_analysis(data, vars, title='Matriz correlación', nombre_archivo='', fsize=(7, 4), output_path=Path() / 'data' / 'clusters' / 'resultados'):
    """
    Realiza el análisis de correlación entre las variables seleccionadas y muestra la matriz de correlación.

    Parámetros:
    - data: DataFrame con los datos
    - vars: Lista de variables para el análisis de correlación
    """
    corr_matrix = data[vars].corr()

    fig, ax = plt.subplots(figsize=fsize)
    
    # Generar el heatmap sobre el objeto ax
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    
    # Título del gráfico
    plt.title(title)
    
    # Crear el directorio si no existe
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar el gráfico como PNG
    if len(nombre_archivo) > 0:
        archivo_guardado = output_path / nombre_archivo
        print(f"Guardando el archivo en: {archivo_guardado}")
        plt.savefig(archivo_guardado, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    # Mostrar el gráfico
    plt.show()


    print("\nInterpretación de la Matriz de Correlación:")
    print("- Valores cercanos a 1 o -1 indican alta correlación positiva o negativa.")
    print("- Variables altamente correlacionadas pueden redundar información.")
    print(corr_matrix)

    if len(nombre_archivo)>0:
        nombre_archivo = nombre_archivo.replace('.png', '.xlsx')
        print(nombre_archivo)
        corr_matrix.to_excel(output_path / nombre_archivo, index=False)
    return corr_matrix

def run_clustering(data, cluster_vars, n_clusters, model_type='kmeans', scale_data=True):
    """
    Ejecuta el modelo de clusterización seleccionado y retorna el DataFrame con las etiquetas de cluster.

    Parámetros:
    - data: DataFrame con los datos
    - cluster_vars: Lista de variables para la clusterización
    - n_clusters: Número de clusters a utilizar
    - model_type: Tipo de modelo ('kmeans', 'gmm')
    - scale_data: Booleano, si es True escala los datos antes de clusterizar

    Retorna:
    - data_clustered: DataFrame con las etiquetas de cluster asignadas
    """
    # Copia del DataFrame original
    data_clustered = data.copy()

    # Selección de variables para clusterizar
    X = data_clustered[cluster_vars].copy()

    # Escalado de datos si se especifica
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # Ejecutar modelo de clusterización
    if model_type == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif model_type == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError("Modelo no soportado. Use 'kmeans' o 'gmm'.")

    # Asignación de etiquetas al DataFrame
    cluster_label = f'cluster_{model_type}'
    data_clustered[cluster_label] = labels

    # Evaluación del modelo
    silhouette_avg = metrics.silhouette_score(X_scaled, labels)
    davies_bouldin = metrics.davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = metrics.calinski_harabasz_score(X_scaled, labels)

    print(f"\nMétricas de evaluación para {model_type.upper()} con {n_clusters} clusters:")
    print(f"Silhouette Score: {silhouette_avg:.4f} (Más cercano a 1 es mejor)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (Más cercano a 0 es mejor)")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f} (Más alto es mejor)")

    return data_clustered, model

def plot_cluster_results(data, cluster_vars, cluster_label):
    """
    Genera visualizaciones de los clusters en el espacio de variables o componentes principales.

    Parámetros:
    - data: DataFrame con los datos y etiquetas de cluster
    - cluster_vars: Lista de variables utilizadas para la clusterización
    - cluster_label: Nombre de la columna con las etiquetas de cluster
    """
    # Si hay dos variables, podemos graficar en 2D
    if len(cluster_vars) == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=cluster_vars[0], y=cluster_vars[1], hue=cluster_label, palette='Set1', s=50)
        plt.title(f'Clusters en el Espacio de Variables ({cluster_vars[0]} vs {cluster_vars[1]})')
        plt.show()
    else:
        # Aplicar PCA para reducir a 2 dimensiones
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[cluster_vars])
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        pca_df[cluster_label] = data[cluster_label].values

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=cluster_label, palette='Set1', s=50)
        plt.title('Clusters en el Espacio de Componentes Principales')
        plt.show()

def cluster_profile(data, cluster_label, eval_vars, n_cols = 5, n=0, filepath=Path() / 'data' / 'clusters' / 'resultados'):
    """
    Muestra el perfil de cada cluster en términos de las variables de evaluación.

    Parámetros:
    - data: DataFrame con los datos y etiquetas de cluster
    - cluster_label: Nombre de la columna con las etiquetas de cluster
    - eval_vars: Lista de variables para evaluar los clusters
    """
    data['casos'] = 1
    cluster_sum = data.groupby(cluster_label, as_index=False).casos.sum()
    if f'{cluster_label}_original' in data.columns:
        cluster_summary = data.groupby([cluster_label, f'{cluster_label}_original'], as_index=False)[eval_vars].mean().round(2)
    else:
        cluster_summary = data.groupby(cluster_label, as_index=False)[eval_vars].mean().round(2)

    cluster_summary = cluster_sum.merge(cluster_summary)
    
    print(f"\nPerfil de clusters basado en variables de evaluación:")

    guardar_tabla_como_png(cluster_summary, f'escenario{n+1}_2_tabla.png', f'Perfil de clusters basado en variables de evaluación (Escenario {n+1})', filepath=filepath)

    # Boxplots de variables de evaluación por cluster
    n_vars = len(eval_vars)
    
    n_rows = math.ceil(n_vars / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, var in enumerate(eval_vars):
        sns.boxplot(x=cluster_label, y=var, data=data, ax=axes[i])
        axes[i].set_title(f'{var.capitalize()}')
    plt.tight_layout()
    plt.show()
    
    fig.savefig(filepath / f'escenario{n+1}_3_boxplot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    


def plot_cluster_distributions(data, vars, cluster_var):
    """
    Genera gráficos de densidad para las variables especificadas, coloreando por cluster.

    Parámetros:
    - data: DataFrame con los datos.
    - vars: Lista de variables a graficar.
    - cluster_var: Nombre de la columna que indica el cluster.
    """
    nrows = len(vars)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 3 * nrows))
    
    for i, var in enumerate(vars):
        sns.kdeplot(data=data, x=var, hue=cluster_var, fill=True, common_norm=False, alpha=0.5, ax=axes[i])
        axes[i].set_title(f'Distribución de {var} por {cluster_var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Densidad')
    
    plt.tight_layout()
    plt.show()

def ordernar_clusters(data_clustered, eval_vars, cluster_var):
    data_ordered = data_clustered.groupby(cluster_var, as_index=False)[[eval_vars[0], 
                                                                        eval_vars[1]]].mean().sort_values([eval_vars[0], 
                                                                                                           eval_vars[1], 
                                                                                                           ]).reset_index(drop=True).reset_index().rename(columns={'index':f'{cluster_var}_ordered'})[[cluster_var, f'{cluster_var}_ordered']]
    data_clustered = data_clustered.merge(data_ordered, on=cluster_var)
    data_clustered = data_clustered.rename(columns={cluster_var: f'{cluster_var}_original'})    
    data_clustered = data_clustered.rename(columns={f'{cluster_var}_ordered':cluster_var})
    return data_clustered


def corr_cluster(col, varx, vary, varz='', cluster_var='', xtick='', ytick='', ztick='', 
                 cmap='RdYlBu', markersize=50, title = '', savefile='', figsize=(8,8), regline=False):

    '''
    Realiza un gráfico de correlación entre variables, puden ser dos o tres variables.
    varx, vary, varz = variables a correlacionaar. varz puede quedar vacía
    xtick, ytick, ztick = label de las variables a correlacionar. De estar vacías toma el nombre de la variable
    cluster_var = Variable de clusterización, muestra distintos markers por cluster
    '''
    sns.set_style("white")
    if len(varz) == 0:        
        fig, ax = plt.subplots(figsize=figsize, dpi=100)    
    else:    
        fig = plt.figure(figsize=figsize)            
        ax = fig.add_subplot(111, projection='3d')

    
    mark = ['*', 'o', 'v', 's', 'h', 'P', 'D']
    colores = ['#01665e', '#5ab4ac', '#c7eae5', '#f6e8c3', '#d8b365','#8c510a'  ]
    
    n=0
    for i in col[cluster_var].unique():
        
        if len(varz) == 0:        
            plt.scatter(x=col[(col[cluster_var]==i)][varx], y=col[(col[cluster_var]==i)][vary],  s=markersize, cmap=cmap, marker=mark[n], alpha=.5)
            
        else:
            ax.scatter(col.loc[col[cluster_var]==i, varx], col.loc[col[cluster_var]==i, vary], col.loc[col[cluster_var]==i, varz],  color=colores[n],  s=markersize, marker=mark[n])
        n += 1
        
    
    if len(xtick) == 0: ax.set_xlabel(varx)
    else: ax.set_xlabel(xtick)

    if len(ytick) == 0: ax.set_ylabel(vary)
    else: ax.set_ylabel(ytick)
    
    
    if len(varz) > 0:
        if len(ztick) == 0: ax.set_zlabel(varz)
        else: ax.set_zlabel(ztick)
    elif regline:
        sns.regplot(data=col, x=varx, y=vary, scatter=False)
    ax.set_title(title)
    
    if len(str(savefile))>0: fig.savefig(savefile)

def evaluate_clustering_models(X, max_clusters=10):
    """
    Evalúa diferentes modelos de clusterización para determinar el número óptimo de clusters.
    Utiliza el método del codo, Silhouette Score, BIC y AIC.

    Parámetros:
    - X: Datos escalados para la clusterización
    - max_clusters: Número máximo de clusters a evaluar

    Retorna:
    - None
    """
    # Listas para almacenar las métricas
    wcss = []  # Within-Cluster Sum of Squares para KMeans
    silhouette_scores_kmeans = []
    silhouette_scores_gmm = []
    bic_scores_gmm = []
    aic_scores_gmm = []
    range_n_clusters = range(2, max_clusters+1)

    print("\nEvaluando modelos de clusterización...")
    for n_clusters in range_n_clusters:
        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        labels_kmeans = kmeans.labels_
        wcss.append(kmeans.inertia_)
        silhouette_avg_kmeans = metrics.silhouette_score(X, labels_kmeans)
        silhouette_scores_kmeans.append(silhouette_avg_kmeans)

        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(X)
        labels_gmm = gmm.predict(X)
        silhouette_avg_gmm = metrics.silhouette_score(X, labels_gmm)
        silhouette_scores_gmm.append(silhouette_avg_gmm)
        bic_scores_gmm.append(gmm.bic(X))
        aic_scores_gmm.append(gmm.aic(X))

    # Método del Codo para KMeans
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, wcss, 'bo-')
    plt.xlabel('Número de Clusters K')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Método del Codo para KMeans')
    plt.show()
    print("\nInterpretación del Método del Codo:")
    print("- Buscamos el punto donde la disminución de WCSS se vuelve menos pronunciada (el codo).")
    print("- Este punto sugiere un número óptimo de clusters.")

    # Silhouette Score para KMeans y GMM
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, silhouette_scores_kmeans, 'ro-', label='KMeans')
    plt.plot(range_n_clusters, silhouette_scores_gmm, 'go-', label='GMM')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score para KMeans y GMM')
    plt.legend()
    plt.show()
    print("\nInterpretación del Silhouette Score:")
    print("- Valores cercanos a 1 indican que las muestras están bien agrupadas.")
    print("- Podemos seleccionar el número de clusters que maximice el Silhouette Score.")

    # BIC y AIC para GMM
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, bic_scores_gmm, 'bo-', label='BIC')
    plt.plot(range_n_clusters, aic_scores_gmm, 'ro-', label='AIC')
    plt.xlabel('Número de Clusters')
    plt.ylabel('BIC / AIC')
    plt.title('BIC y AIC para GMM')
    plt.legend()
    plt.show()
    print("\nInterpretación de BIC y AIC:")
    print("- Valores más bajos de BIC/AIC indican un mejor modelo.")
    print("- Podemos seleccionar el número de clusters que minimice el BIC/AIC.")

import folium
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

def plot_cluster_in_map(carto, data, cluster_var, archivo_salida=None, filepath=Path() / 'data' / 'clusters' / 'resultados'):
    """
    Genera un mapa de clusters y opcionalmente lo guarda como archivo HTML.

    Parámetros:
        carto (GeoDataFrame): Geometrías de las líneas.
        data (DataFrame): Datos con la variable de cluster.
        cluster_var (str): Nombre de la columna de cluster.
        archivo_salida (str, opcional): Ruta para guardar el archivo HTML.
    """
    # Merge
    carto = carto.merge(data.reindex(columns=['id_linea', cluster_var]), on='id_linea')

    # Inicializar mapa
    m = folium.Map(location=(-34.6, -58.5), zoom_start=12, tiles="cartodbpositron", width=1300, height=800)

    # Obtener lista de clusters únicos (excluyendo ruido si fuera el caso)
    clusters_unicos = sorted(carto[cluster_var].dropna().unique())
    clusters_unicos = [c for c in clusters_unicos if c != -1]

    # Preparar colores dinámicamente según la cantidad de clusters
    n_clusters = len(clusters_unicos)
    base_colors = [rgb_to_hex(colormaps['tab20'](i / n_clusters)) for i in range(n_clusters)]

    # Crear mapeo cluster -> color
    cluster_colors = {cluster: base_colors[idx] for idx, cluster in enumerate(clusters_unicos)}

    # Crear las capas por cluster
    for cluster, color in cluster_colors.items():
        carto.query(f"{cluster_var} == {cluster}").explore(
            name=str(cluster),
            m=m,
            color=color,
            legend=False
        )

    m.add_child(folium.LayerControl())

    # Guardar como HTML si se especifica
    if archivo_salida:
        m.save( filepath / archivo_salida)

    return m

def generar_datos(df, variables, cluster_col):
    save_data = """

Los clusters fueron creados a partir de las siguientes variables operativas:
""" + ", ".join(variables) + """

---

Resultados del análisis"""

    # Agrupar por cluster y calcular estadísticas
    resumen = df.groupby(cluster_col)[variables].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75)),
        ('IQR', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        ('max', 'max')
    ])

    # Generar texto formateado
    for cluster in resumen.index:
        save_data += f"**Cluster {cluster}:**\n"
        save_data += f"- Cantidad de casos: {resumen.loc[cluster, (variables[0], 'count')]}\n"

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

def guardar_datos_txt(save_data, nombre_archivo="resumen_clusters.txt", filepath=Path() / 'data' / 'clusters' / 'resultados'):
    with open(filepath / nombre_archivo, "w", encoding="utf-8") as file:
        file.write(save_data)




def check_clustering_suitability(df, variables, plot=True):
    """
    Evalúa si las variables de un DataFrame siguen una distribución normal (gaussiana).
    También evalúa si K-Means es adecuado y retorna recomendaciones de clustering.

    Parámetros:
    - df: DataFrame con los datos.
    - variables: Lista de columnas a evaluar.
    - plot: Booleano para mostrar histogramas y PCA (por defecto True).
    
    Retorna:
    - Un diccionario con los resultados de las pruebas de normalidad y una recomendación de clustering.
    """
    results = {}
    variables_gmm = []  # Variables recomendadas para GMM
    variables_hierarchical = []  # Variables recomendadas para Hierarchical Clustering
    variables_kmeans = []  # Variables recomendadas para K-Means
    
    # Crear histogramas
    if plot:
        plt.figure(figsize=(12, 6))
        for var in variables:
            sns.histplot(df[var].dropna(), kde=True, bins=30, label=var, alpha=0.6)
        plt.title("Distribución de Variables con KDE")
        plt.legend()
        plt.show()
    
    print("\n📌 **Evaluación de Normalidad para cada variable:**")
    
    for var in variables:
        stat_shapiro, p_shapiro = shapiro(df[var].dropna())  # Prueba de Shapiro-Wilk
        stat_ks, p_ks = kstest(df[var].dropna(), 'norm')  # Prueba de Kolmogorov-Smirnov
        result_anderson = anderson(df[var].dropna(), dist='norm')  # Prueba de Anderson-Darling
        
        results[var] = {
            "Shapiro-Wilk p-value": p_shapiro,
            "Kolmogorov-Smirnov p-value": p_ks,
            "Anderson-Darling stat": result_anderson.statistic,
            "Anderson Critical Values": result_anderson.critical_values
        }
        
        # Evaluar si la variable pasa todas las pruebas de normalidad
        if (p_shapiro > 0.05) and (p_ks > 0.05) and (result_anderson.statistic < result_anderson.critical_values[2]):
            variables_gmm.append(var)  # Recomendado para GMM
        else:
            variables_hierarchical.append(var)  # Recomendado para Hierarchical Clustering

    # Evaluar si K-Means es adecuado con análisis de varianza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[variables].dropna())
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    cluster_sizes = np.bincount(cluster_labels)

    # Si la silueta es buena (> 0.5) y los clusters son de tamaño similar, K-Means es recomendable
    if silhouette_avg > 0.5 and max(cluster_sizes) / min(cluster_sizes) < 2:
        variables_kmeans = variables.copy()
    
    # Análisis de PCA
    if plot:
        print("\n📌 **Análisis de Componentes Principales (PCA) - Evaluando si las componentes son gaussianas:**")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(12, 6))
        sns.histplot(X_pca[:, 0], kde=True, bins=30, label="PC1", alpha=0.6)
        sns.histplot(X_pca[:, 1], kde=True, bins=30, label="PC2", alpha=0.6)
        plt.title("Distribución de Componentes Principales")
        plt.legend()
        plt.show()

    # Mostrar la recomendación final
    print("\n📌 **Recomendación de Clustering:**")
    if len(variables_gmm) > 0:
        print(f"✅ **GMM es recomendable para:** {variables_gmm}")
    else:
        print("❌ No se recomienda GMM, los datos no son gaussianos.")

    if len(variables_hierarchical) > 0:
        print(f"✅ **Hierarchical Clustering es recomendable para:** {variables_hierarchical}")
    else:
        print("❌ No se recomienda Hierarchical Clustering.")

    if len(variables_kmeans) > 0:
        print(f"✅ **K-Means es recomendable para:** {variables_kmeans}")
    else:
        print("❌ No se recomienda K-Means, los clusters no parecen ser esféricos o tienen alta varianza.")

    return {"GMM": variables_gmm, "Hierarchical": variables_hierarchical, "K-Means": variables_kmeans, "Normality Results": results}




def hierarchical_clustering(df, variables, n_clusters=None, linkage='ward', plot_dendrogram=False, var='hcluster'):
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
        plt.title('Dendrograma para determinar el número de clusters')
        plt.xlabel('Observaciones')
        plt.ylabel('Distancia')
        # plt.savefig('dendograma.png')
        plt.show()
    
    # Si no se especifica número de clusters, sugerimos inspeccionar el dendrograma
    if n_clusters is None:
        # raise ValueError("Debes definir 'n_clusters' o visualizar el dendrograma para seleccionar un valor adecuado.")
        print('Hay que definir el número de clusters. Visualizar el dendrograma para seleccionar un valor adecuado.')
    else:
        # Aplicar clustering jerárquico
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        # df[var] = cluster_model.fit_predict(data_scaled)
        df.loc[:, var] = cluster_model.fit_predict(data_scaled)

    
    return df

def guardar_tabla_como_png(df, nombre_archivo, titulo, filepath = Path() / 'data' / 'clusters' / 'resultados'):
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
    ax.axis('tight')
    ax.axis('off')

    # Crear tabla
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')

    # Ajustar tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([i for i in range(len(df.columns))])

    # Añadir título muy pegado
    plt.title(titulo, fontsize=12, weight='bold', pad=3)

    # Guardar sin margen extra
    plt.savefig(filepath / nombre_archivo, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def correr_clusters():

    data = levanto_tabla_sql('kpis_lineas', 'dash')
    carto = levanto_tabla_sql("lines_geoms", "insumos")
    escenarios_clusterizacion = levanto_tabla_sql("escenarios_clusterizacion", "insumos")
    data = data[data.modo=='Autobus']
    
    cluster_vars = [
        [v.strip() for v in fila.split(",") if v.strip()]
        for fila in escenarios_clusterizacion["variables"]
    ]
    eval_vars = cluster_vars
    n_clusters = escenarios_clusterizacion.cant_clusters.tolist()
    max_clase = escenarios_clusterizacion.max_clusters_clase.tolist()
    n_clusters_explotados = escenarios_clusterizacion.cant_clusters_recluster.tolist()
    
    filepath = Path() / 'data' / 'clusters' / f'resultados'

    clusters_result = pd.DataFrame([])
    
    for n in range(0, len(n_clusters)):
        print('Escenario', n+1, cluster_vars[n])
    
        lineas_new = data[['dia', 'mes', 'id_linea', 'nombre_linea', 'empresa', 'modo']+cluster_vars[n]].copy()
        lineas_new = lineas_new.dropna()
       
        corr_matrix = correlation_analysis(lineas_new, cluster_vars[n], nombre_archivo=f"escenario{n+1}_1_corr.png", fsize=(8, 4), output_path=filepath)    
        data_hierarchical = hierarchical_clustering(data.copy(), cluster_vars[n], n_clusters=n_clusters[n], linkage='ward', plot_dendrogram=False, var='hcluster')
    
        data_hierarchical['hcluster'] = data_hierarchical['hcluster'].astype(str)
        eval_h = data_hierarchical.groupby('hcluster').size().reset_index().rename(columns={0:'cant'})
    
        # Explota cluster
        for i in eval_h[eval_h.cant>max_clase[n]].hcluster:
            data_tmp = hierarchical_clustering(data_hierarchical[data_hierarchical.hcluster==i].copy(), 
                                               cluster_vars[n], 
                                               n_clusters=n_clusters_explotados[n], 
                                               linkage='ward', 
                                               plot_dendrogram=False, 
                                               var=f'hcluster_{i}')
            
            data_tmp[f'hcluster_{i}'] = data_tmp[f'hcluster_{i}'].astype(str)
    
            data_hierarchical = data_hierarchical.merge(data_tmp[['dia', 'id_linea', f'hcluster_{i}' ]], how='left')
            data_hierarchical.loc[data_hierarchical[f'hcluster_{i}'].notna(), 
                                                    'hcluster'] = data_hierarchical.loc[data_hierarchical[f'hcluster_{i}'].notna(), 
                                                                                                                    'hcluster'] + '_' + data_hierarchical.loc[data_hierarchical[f'hcluster_{i}'].notna(), f'hcluster_{i}']
            data_hierarchical = data_hierarchical.drop([f'hcluster_{i}'], axis=1)
        
        # Ordenar clusters
        data_hierarchical = ordernar_clusters(data_hierarchical, eval_vars[n], 'hcluster')
    
        lineas_new = lineas_new.merge(data_hierarchical[['dia', 'id_linea', 'hcluster', 'hcluster_original']].rename(columns={'hcluster':f'escenario{n+1}_hcluster',
                                                                                                                             'hcluster_original':f'escenario{n+1}_hcluster_original'}), how='left')

        if len(clusters_result)==0:
            clusters_result = lineas_new.copy()
        else:
            clusters_result = clusters_result.merge(lineas_new[['dia', 'id_linea', f'escenario{n+1}_hcluster', f'escenario{n+1}_hcluster_original']])
        cluster_profile(lineas_new, f'escenario{n+1}_hcluster', eval_vars[n], n_cols = len(eval_vars[n]), n=n, filepath=filepath)
    
        mapa = plot_cluster_in_map(carto=carto,
                               data=lineas_new,
                               cluster_var=f'escenario{n+1}_hcluster',
                               archivo_salida=f'html_escenario{n+1}_map.html',
                               filepath=filepath)
    
        resumen = generar_datos(lineas_new, cluster_vars[n], f'escenario{n+1}_hcluster')    
        guardar_datos_txt(resumen, f"escenario{n+1}_4_datos.txt", filepath)
    
    os.makedirs(filepath, exist_ok=True)

    data = data.merge(clusters_result, how='left')
    data.to_csv(filepath / 'clusters_lineas.csv', index=False)
    return data