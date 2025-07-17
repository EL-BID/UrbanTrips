import pandas as pd
import streamlit as st
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
kpis = levanto_tabla_sql("kpis_lineas", "dash")

# Inicializar sesión
if "tabla_mergeada" not in st.session_state:
    st.session_state["tabla_mergeada"] = None
    st.session_state["archivo_anterior"] = None

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
                        st.success("Merge reemplazado con éxito.")
                        st.dataframe(st.session_state["tabla_mergeada"])
                else:
                    if st.button("Hacer merge"):
                        st.session_state["tabla_mergeada"] = pd.merge(kpis, tabla_externa, on=merge_keys, how="left")
                        st.session_state["archivo_anterior"] = archivo.name
                        st.success("Merge realizado con éxito.")
                        st.dataframe(st.session_state["tabla_mergeada"])
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")




# from dash_utils import guarda_tabla_sql, levanto_tabla_sql
# import pandas as pd
# import streamlit as st
from dash_utils import levanto_tabla_sql_local
# ---------- helpers ----------
def load_escenarios() -> list[dict]:
    """
    Lee la tabla escenarios_clusterizacion de la BD.
    Devuelve una lista de diccionarios con llaves 'nombre' y 'variables'.
    Si la tabla no existe o está vacía, devuelve [].
    """
    try:
        df = levanto_tabla_sql_local("escenarios_clusterizacion", "insumos")
        if df.empty:
            return []
        # Nos aseguramos de que 'variables' quede como string (por si viene lista/None)
        df["variables"] = df["variables"].astype(str)
        return df.to_dict(orient="records")
    except Exception as e:
        st.error(f"Error al leer escenarios_clusterizacion: {e}")
        return []

def save_escenarios(lista_escenarios: list[dict]):
    """Sobrescribe la tabla en la BD con el contenido actual."""
    df_guardar = pd.DataFrame(lista_escenarios)

    if df_guardar.empty:
        st.warning("No hay escenarios para guardar.")
        return

    # Validación extra: columnas correctas
    df_guardar = df_guardar[["nombre", "variables"]].copy()

    guardar_tabla_sql(df_guardar, "escenarios_clusterizacion", "insumos", modo="replace")


# ---------- estado ----------
# Cargamos SIEMPRE desde la base en cada rerun; así reflejamos cambios externos.
st.session_state["escenarios"] = load_escenarios()

# ---------- UI ----------
with st.expander("🗂️ Escenarios de selección de variables", expanded=True):

    df_base = (
        st.session_state.get("tabla_mergeada")
        if st.session_state.get("tabla_mergeada") is not None
        else kpis
    )
    columnas_numericas = df_base.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # -------- crear nuevo escenario --------
    with st.form("formulario_escenario"):
        st.markdown("### Crear nuevo escenario")
        nombre_escenario = st.text_input("Nombre del escenario")
        variables_seleccionadas = st.multiselect(
            "Seleccioná variables numéricas para este escenario",
            options=columnas_numericas,
        )
        submit_escenario = st.form_submit_button("Guardar escenario")

    if submit_escenario:
        if not nombre_escenario:
            st.warning("Debés ingresar un nombre para el escenario.")
        elif not variables_seleccionadas:
            st.warning("Debés seleccionar al menos una variable.")
        elif nombre_escenario in [e["nombre"] for e in st.session_state["escenarios"]]:
            st.error("Ya existe un escenario con ese nombre.")
        else:
            nuevo = {
                "nombre": nombre_escenario,
                "variables": ", ".join(variables_seleccionadas),
            }
            st.session_state["escenarios"].append(nuevo)
            save_escenarios(st.session_state["escenarios"])
            st.success(f"Escenario '{nombre_escenario}' guardado.")

    # -------- mostrar escenarios --------
    if st.session_state["escenarios"]:
        st.markdown("### Escenarios guardados")
        st.dataframe(pd.DataFrame(st.session_state["escenarios"]))

        # -------- eliminar escenario --------
        a_borrar = st.selectbox(
            "Seleccioná un escenario para eliminar",
            options=[e["nombre"] for e in st.session_state["escenarios"]],
            key="select_borrar",
        )
        if st.button("Eliminar escenario seleccionado"):
            st.session_state["escenarios"] = [
                e for e in st.session_state["escenarios"] if e["nombre"] != a_borrar
            ]
            save_escenarios(st.session_state["escenarios"])
            st.success(f"Escenario '{a_borrar}' eliminado.")
            st.rerun()()
    else:
        st.info("Aún no hay escenarios guardados.")
