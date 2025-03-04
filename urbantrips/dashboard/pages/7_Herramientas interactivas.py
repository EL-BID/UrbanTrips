import pandas as pd
import streamlit as st
from dash_utils import (
    levanto_tabla_sql,
    get_logo,
)
from urbantrips.kpi.kpi import compute_route_section_load
from urbantrips.viz.viz import visualize_route_section_load
from urbantrips.kpi.line_od_matrix import compute_lines_od_matrix
from urbantrips.viz.line_od_matrix import visualize_lines_od_matrix
from urbantrips.kpi.supply_kpi import compute_route_section_supply
from urbantrips.viz.section_supply import visualize_route_section_supply_data

try:
    from urbantrips.utils.utils import iniciar_conexion_db
    from urbantrips.utils import utils
except ImportError as e:
    st.error(
        f"Falta una librería requerida: {e}. Algunas funcionalidades no estarán disponibles. \nSe requiere full acceso a Urbantrips para correr esta página"
    )
    st.stop()


# --- Función para levantar tablas SQL y almacenar en session_state ---
def cargar_tabla_sql(tabla_sql, tipo_conexion="dash", query=""):
    if f"{tabla_sql}_{tipo_conexion}" not in st.session_state:
        conn = iniciar_conexion_db(tipo=tipo_conexion)
        try:
            query = query or f"SELECT * FROM {tabla_sql}"
            tabla = pd.read_sql_query(query, conn)
            st.session_state[f"{tabla_sql}_{tipo_conexion}"] = tabla
        except Exception:
            st.error(f"{tabla_sql} no existe")
            st.session_state[f"{tabla_sql}_{tipo_conexion}"] = pd.DataFrame()
        finally:
            conn.close()
    return st.session_state[f"{tabla_sql}_{tipo_conexion}"]


def seleccionar_linea(key_input, key_select):
    texto_a_buscar = st.text_input(
        f"Ingrese el texto a buscar en líneas", key=key_input
    )
    if texto_a_buscar:
        if f"df_filtrado_{texto_a_buscar}" not in st.session_state:
            st.session_state[f"df_filtrado_{texto_a_buscar}"] = metadata_lineas[
                metadata_lineas.apply(
                    lambda row: row.astype(str)
                    .str.contains(texto_a_buscar, case=False, na=False)
                    .any(),
                    axis=1,
                )
            ]
        df_filtrado = st.session_state[f"df_filtrado_{texto_a_buscar}"]

        if not df_filtrado.empty:
            opciones = df_filtrado.apply(
                lambda row: f"{row['nombre_linea']}", axis=1
            ).tolist()
            seleccion_texto = st.selectbox(
                f"Seleccione una línea de colectivo",
                opciones,
                key=key_select,
            )
            df_seleccionado = df_filtrado.iloc[opciones.index(seleccion_texto)]

            st.session_state["nombre_linea_7"] = df_seleccionado.nombre_linea
            st.session_state["id_linea_7"] = df_seleccionado.id_linea

        else:
            st.warning("No se encontró ninguna coincidencia.")


st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)
try:
    # --- Cargar configuraciones y conexiones en session_state ---
    if "configs" not in st.session_state:
        st.session_state.configs = utils.leer_configs_generales()

    configs = st.session_state.configs
    h3_legs_res = configs["resolucion_h3"]
    alias = configs["alias_db_data"]
    use_branches = configs["lineas_contienen_ramales"]
    metadata_lineas = cargar_tabla_sql("metadata_lineas", "insumos")[
        ["id_linea", "nombre_linea"]
    ]
    conn_insumos = iniciar_conexion_db(tipo="insumos")
except ValueError as e:
    st.error(
        f"Falta una base de datos requerida: {e}. \nSe requiere full acceso a Urbantrips para correr esta página"
    )
    st.stop()

for var in [
    "id_linea_7",
    "nombre_linea_7",
    "day_type_7",
    "yr_mo_7",
    "n_sections_7",
    "section_meters_7",
    "hour_range_7",
]:
    if var not in st.session_state:
        st.session_state[var] = None


st.header("Herramientas")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.subheader("Periodo")

    kpi_lineas = levanto_tabla_sql("basic_kpi_by_line_hr")
    day_type = col1.selectbox("Tipo de dia  ", options=["weekday", "weekend"])
    st.session_state["day_type_7"] = day_type

    # add month and year
    yr_mo = col1.selectbox(
        "Periodo  ", options=kpi_lineas.yr_mo.unique(), key="year_month"
    )
    st.session_state["yr_mo_7"] = yr_mo

with col2:
    st.subheader("Línea")
    seleccionar_linea("base_input", "base_select")

with col3:
    st.subheader("Parámetros")
    n_sections = col3.number_input(
        "Numero Secciones", min_value=0, max_value=999, value=None
    )
    st.session_state["n_sections_7"] = n_sections

    section_meters = col3.number_input(
        "Metros de cada seccion", min_value=0, max_value=5000, value=None
    )
    st.session_state["section_meters_7"] = section_meters

    rango_desde = col3.selectbox(
        "Rango horario (desde) ", options=range(0, 24), key="rango_hora_desde", index=9
    )
    rango_hasta = col3.selectbox(
        "Rango horario (hasta)", options=range(0, 24), key="rango_hora_hasta", index=9
    )
    hour_range = [rango_desde, rango_hasta]
    st.session_state["hour_range_7"] = hour_range


line_ids = st.session_state["id_linea_7"]

geoms_check = (st.session_state["n_sections_7"] is not None) | (
    st.session_state["section_meters_7"] is not None
)


if (line_ids is not None) & (geoms_check):

    hour_range = st.session_state["hour_range_7"]
    n_sections = st.session_state["n_sections_7"]
    section_meters = st.session_state["section_meters_7"]
    day_type = st.session_state["day_type_7"]

    st.write("Calculando los estadisticos de carga de las secciones de las lineas")
    # Se calculan los estadisticos de carga de las secciones de las lineas
    compute_route_section_load(
        line_ids=[line_ids],
        hour_range=hour_range,
        n_sections=n_sections,
        section_meters=section_meters,
        day_type=day_type,
    )

    st.write("Visualizando los estadisticos de carga de las secciones de las lineas")
    # Se visualizan los estadisticos de carga de las secciones de las lineas
    visualize_route_section_load(
        line_ids=[line_ids],
        hour_range=hour_range,
        day_type=day_type,
        n_sections=n_sections,
        section_meters=section_meters,
        save_gdf=True,
        stat="totals",
        factor=500,
        factor_min=10,
    )

    st.write("Calculando la matriz OD de la linea")
    # Se computa la matriz OD de las lineas
    compute_lines_od_matrix(
        line_ids=[line_ids],
        hour_range=hour_range,
        n_sections=n_sections,
        section_meters=section_meters,
        day_type=day_type,
        save_csv=True,
    )
    st.write("Visualizando la matriz OD de la linea")
    # Se visualiza la matriz OD de las lineas
    visualize_lines_od_matrix(
        line_ids=[line_ids],
        hour_range=hour_range,
        day_type=day_type,
        n_sections=n_sections,
        section_meters=section_meters,
        stat="totals",
    )

    st.write("Calculando los estadisticos de oferta por secciones de las lineas")
    # Calcula los estadisticos de oferta por sección de las lineas
    route_section_supply = compute_route_section_supply(
        line_ids=[line_ids],
        hour_range=hour_range,
        n_sections=n_sections,
        section_meters=section_meters,
        day_type=day_type,
    )

    st.write("Visualizando los estadisticos de oferta por secciones de las lineas")
    # Visualiza los estadisticos de oferta por sección de las lineas
    visualize_route_section_supply_data(
        line_ids=[line_ids],
        hour_range=hour_range,
        day_type="weekday",
        n_sections=n_sections,
        section_meters=section_meters,
    )

    st.write(
        "Resultados pueden consultarse en el directorio UrbanTrips/resultados o en la pestaña Indicadores de oferta y demanda reiniciando el dashboard"
    )
else:
    st.write("No hay datos para mostrar")
