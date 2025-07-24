import pandas as pd
import streamlit as st
from pathlib import Path
from dash_utils import (
    levanto_tabla_sql,
    guardar_tabla_sql,
    get_logo, 
    configurar_selector_dia
)

st.set_page_config(page_title="Indicadores Operativos por Línea", layout="wide")

# Cabecera estándar
logo = get_logo()
st.image(logo)
alias_seleccionado = configurar_selector_dia()

# Cargar KPIs
kpis = levanto_tabla_sql("kpis_lineas", "general")
kpis_merge = levanto_tabla_sql("kpis_lineas_merge", "general")


# Inicializar sesión
if "tabla_mergeada" not in st.session_state:
    st.session_state["tabla_mergeada"] = None
    st.session_state["archivo_anterior"] = None
if len(kpis_merge) > 0:
    st.session_state["tabla_mergeada"] = kpis_merge.copy()

# 1. Mostrar tabla original
with st.expander("📄 Ver tabla original de KPIs", expanded=True):
    st.dataframe(kpis)

# 2. Merge con tabla externa
with st.expander("🔗 Subir tabla externa y hacer merge"):
    archivo = st.file_uploader("Elegí un archivo CSV o Excel", type=["csv", "xlsx"])

    if archivo is not None:
        try:
            if archivo.name.endswith(".csv"):
                tabla_externa = pd.read_csv(archivo)
            else:
                tabla_externa = pd.read_excel(archivo)

            st.success("Archivo cargado correctamente.")
            st.dataframe(tabla_externa)

            # Selección de claves de merge
            opcion_merge = st.radio(
                "Seleccioná la clave de combinación",
                options=["['dia', 'id_linea']", "['mes', 'id_linea']"]
            )
            merge_keys = eval(opcion_merge)

            # Verificación de columnas
            cols_faltantes_ext = [col for col in merge_keys if col not in tabla_externa.columns]
            cols_faltantes_kpis = [col for col in merge_keys if col not in kpis.columns]

            if cols_faltantes_ext:
                st.error(f"Faltan estas columnas en la tabla externa: {cols_faltantes_ext}")
            elif cols_faltantes_kpis:
                st.error(f"Faltan estas columnas en la tabla KPIs: {cols_faltantes_kpis}")
            else:
                mismo_archivo = st.session_state["archivo_anterior"] == archivo.name
                merge_previo = st.session_state["tabla_mergeada"] is not None and mismo_archivo

                if merge_previo:
                    st.warning("Ya existe un merge con este archivo.")
                    if st.button("Reemplazar merge anterior"):
                        st.session_state["tabla_mergeada"] = pd.merge(kpis, tabla_externa, on=merge_keys, how="left")
                        st.session_state["archivo_anterior"] = archivo.name
                        guardar_tabla_sql(st.session_state["tabla_mergeada"], "kpis_lineas_merge", 'general', modo='replace')
                        st.success("Merge reemplazado con éxito.")
                        st.info("✅ Tabla guardada en `kpis_lineas`.")
                        st.dataframe(st.session_state["tabla_mergeada"])
                else:
                    if st.button("Hacer merge"):
                        st.session_state["tabla_mergeada"] = pd.merge(kpis, tabla_externa, on=merge_keys, how="left")
                        st.session_state["archivo_anterior"] = archivo.name
                        guardar_tabla_sql(st.session_state["tabla_mergeada"], "kpis_lineas_merge", 'general', modo='replace')
                        st.success("Merge realizado con éxito.")
                        st.info("✅ Tabla guardada en `kpis_lineas`.")
                        st.dataframe(st.session_state["tabla_mergeada"])

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")


# ---------- Cargar KPIs y base ----------
if "kpis" not in st.session_state:
    st.session_state["kpis"] = levanto_tabla_sql("kpis_lineas", "general")

# Inicializar sesión
if "tabla_mergeada" not in st.session_state:
    st.session_state["tabla_mergeada"] = levanto_tabla_sql("kpis_lineas_merge", "general")


if "tabla_mergeada" in st.session_state and st.session_state["tabla_mergeada"] is not None:
    df_base = st.session_state["tabla_mergeada"]
else:
    df_base = st.session_state["kpis"]


def save_escenarios(lista: list[dict]):
    df_guardar = pd.DataFrame(lista)
    if df_guardar.empty:
        st.warning("No hay escenarios para guardar.")
        return
    columnas = ["nombre", "variables", "cant_clusters", "max_clusters_clase", "cant_clusters_recluster"]
    df_guardar = df_guardar[columnas]
    guardar_tabla_sql(df_guardar, "escenarios_clusterizacion", "insumos", modo="replace")

# UI
with st.expander("🗂️ Escenarios de clusterización", expanded=True):

    columnas_numericas = df_base.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # -------- crear nuevo escenario --------
    with st.form("formulario_escenario"):
        st.markdown("### Crear nuevo escenario")

        nombre_escenario = st.text_input("Nombre del escenario")
        variables_seleccionadas = st.multiselect(
            "Seleccioná variables",
            options=columnas_numericas
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            cant_clusters = st.number_input("Cantidad de clusters", min_value=1, step=1, value=6)
        with col2:
            max_clusters_clase = st.number_input("Máx. elementos por clase", min_value=1, step=1, value=80)
        with col3:
            cant_clusters_recluster = st.number_input("Clusters (reclasterización)", min_value=1, step=1, value=3)

        submit = st.form_submit_button("Guardar escenario")

    if submit:
        if not nombre_escenario:
            st.warning("Debés ingresar un nombre.")
        elif not variables_seleccionadas:
            st.warning("Debés seleccionar al menos una variable.")
        else:
            # Leer los existentes desde la base para chequear duplicados
            existentes = levanto_tabla_sql("escenarios_clusterizacion", "insumos")
            if not existentes.empty and nombre_escenario in existentes["nombre"].astype(str).tolist():
                st.error("Ya existe un escenario con ese nombre.")
            else:
                nuevo = {
                    "nombre": nombre_escenario,
                    "variables": ", ".join(variables_seleccionadas),
                    "cant_clusters": int(cant_clusters),
                    "max_clusters_clase": int(max_clusters_clase),
                    "cant_clusters_recluster": int(cant_clusters_recluster)
                }
                lista_actualizada = existentes.to_dict(orient="records") if not existentes.empty else []
                lista_actualizada.append(nuevo)
                save_escenarios(lista_actualizada)
                st.cache_data.clear() 
                st.success(f"Escenario '{nombre_escenario}' guardado.")
                st.rerun()

    # -------- mostrar y eliminar escenarios --------
    df_escenarios = levanto_tabla_sql("escenarios_clusterizacion", "insumos")
    if not df_escenarios.empty:
        df_escenarios["variables"] = df_escenarios["variables"].astype(str)
        for col, default in [
            ("cant_clusters", 6),
            ("max_clusters_clase", 80),
            ("cant_clusters_recluster", 3)
        ]:
            if col not in df_escenarios.columns:
                df_escenarios[col] = default

        st.markdown("### Escenarios guardados")
        st.dataframe(df_escenarios)

        a_borrar = st.selectbox(
            "Seleccioná un escenario para eliminar",
            options=df_escenarios["nombre"].astype(str).tolist(),
            key="select_borrar"
        )

        if st.button("Eliminar escenario seleccionado"):
            df_filtrado = df_escenarios[df_escenarios["nombre"].astype(str) != a_borrar]
            guardar_tabla_sql(df_filtrado, "escenarios_clusterizacion", "insumos", modo="replace")
            st.cache_data.clear()
            st.success(f"Escenario '{a_borrar}' eliminado.")
            st.rerun()
    else:
        st.info("Aún no hay escenarios guardados.")


# -------------------------------------------------------------------
# EXPANDER: Clusterizar
# -------------------------------------------------------------------

from clusters_utils import correr_clusters

with st.expander("🧩 Clusterizar", expanded=False):

    # 1) ¿Hay datos en kpis_lineas?
    df_kpis = st.session_state["kpis"]            # ya lo cargaste al inicio
    kpis_ok = not df_kpis.empty

    # 2) ¿Hay al menos un escenario?
    df_escenarios = levanto_tabla_sql("escenarios_clusterizacion", "insumos")
    escenarios_ok = not df_escenarios.empty

    # Mensajes de estado
    if not kpis_ok:
        st.error("La tabla **kpis_lineas** está vacía. Primero cargá datos.")
    if not escenarios_ok:
        st.error("No existe ningún escenario guardado. Creá al menos uno.")

    # Botón habilitado sólo si ambos requisitos se cumplen
    days = sorted(df_kpis["dia"].dropna().unique())
    days = [x for x in days if x not in ["Promedios"]]
    day_sel = st.selectbox("Día", ["Promedios"] + days, index=0)
    cluster_btn = st.button(
        "🚀 Clusterizar escenarios",
        disabled=not (kpis_ok and escenarios_ok)
    )

    if cluster_btn:
        # ------------------------------------------------------------------
        # Llamá aquí tu función real de clustering
        # ------------------------------------------------------------------
        st.info("Ejecutando clustering. Esto puede tardar unos segundos…")
        try:
            df_kpis = df_kpis[df_kpis.dia==day_sel]
            resultados = correr_clusters(df_kpis)  # <- tu función
            st.success("¡Clustering completado!")

            st.markdown("### 📁 Carpeta de resultados:")
            resultados_dir = Path.cwd() / "data" / "clusters" / "resultados"
            st.markdown("---")
                        
            # Enlace clicable (funciona en navegadores locales)
            if resultados_dir.exists():
                st.code(str(resultados_dir), language="bash")
                st.markdown(
                    f"[📂 Abrir carpeta de resultados]({resultados_dir.as_uri()})",
                    unsafe_allow_html=True
                )


            
            st.dataframe(resultados)
        except Exception as e:
            st.error(f"Error al clusterizar: {e}")
        

