import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from urbantrips.utils.utils import iniciar_conexion_db
from urbantrips.utils import utils
from urbantrips.kpi import overlapping as ovl
from urbantrips.viz import overlapping as ovl_viz

st.set_page_config(layout="wide")


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


# --- Cargar configuraciones y conexiones en session_state ---
if "configs" not in st.session_state:
    st.session_state.configs = utils.leer_configs_generales()

configs = st.session_state.configs
alias = configs["alias_db_data"]
use_branches = configs["lineas_contienen_ramales"]
metadata_lineas = cargar_tabla_sql("metadata_lineas", "insumos")[
    ["id_linea", "nombre_linea"]
]
conn_insumos = iniciar_conexion_db(tipo="insumos")

# --- Inicializar variables en session_state ---
for var in [
    "id_linea_1",
    "nombre_linea_1",
    "branch_id_1",
    "branch_name_1",
    "id_linea_2",
    "nombre_linea_2",
    "branch_id_2",
    "branch_name_2",
]:
    if var not in st.session_state:
        st.session_state[var] = None


# --- Función para seleccionar líneas y ramales y almacenarlos en session_state ---
def seleccionar_linea(nombre_columna, key_input, key_select, branch_key, conn_insumos):
    texto_a_buscar = st.text_input(
        f"Ingrese el texto a buscar para {nombre_columna}", key=key_input
    )
    if texto_a_buscar:
        if f"df_filtrado_{texto_a_buscar}_{branch_key}" not in st.session_state:
            st.session_state[f"df_filtrado_{texto_a_buscar}_{branch_key}"] = (
                metadata_lineas[
                    metadata_lineas.apply(
                        lambda row: row.astype(str)
                        .str.contains(texto_a_buscar, case=False, na=False)
                        .any(),
                        axis=1,
                    )
                ]
            )
        df_filtrado = st.session_state[f"df_filtrado_{texto_a_buscar}_{branch_key}"]

        if not df_filtrado.empty:
            opciones = df_filtrado.apply(
                lambda row: f"{row['nombre_linea']}", axis=1
            ).tolist()
            seleccion_texto = st.selectbox(
                f"Seleccione una línea de colectivo para {nombre_columna}",
                opciones,
                key=key_select,
            )
            st.session_state[f"seleccion_{branch_key}"] = df_filtrado.iloc[
                opciones.index(seleccion_texto)
            ]

            if use_branches:
                if (
                    f"metadata_branches_{st.session_state[f'seleccion_{branch_key}']['id_linea']}"
                    not in st.session_state
                ):
                    st.session_state[
                        f"metadata_branches_{st.session_state[f'seleccion_{branch_key}']['id_linea']}"
                    ] = pd.read_sql(
                        f"SELECT * FROM metadata_ramales WHERE id_linea = {st.session_state[f'seleccion_{branch_key}']['id_linea']}",
                        conn_insumos,
                    )
                metadata_branches = st.session_state[
                    f"metadata_branches_{st.session_state[f'seleccion_{branch_key}']['id_linea']}"
                ]

                selected_branch = st.selectbox(
                    "Seleccione un ramal",
                    metadata_branches.nombre_ramal.unique(),
                    key=f"branch_{branch_key}",
                )
                st.session_state[f"seleccion_{branch_key}"]["branch_id"] = (
                    metadata_branches.loc[
                        metadata_branches.nombre_ramal == selected_branch, "id_ramal"
                    ].values[0]
                )
                st.session_state[f"seleccion_{branch_key}"][
                    "branch_name"
                ] = selected_branch
        else:
            st.warning("No se encontró ninguna coincidencia.")


# --- Selección de líneas y ramales con almacenamiento en session_state ---
with st.expander("Seleccionar líneas", expanded=True):
    col1, col2, col3 = st.columns([1, 3, 3])

    with col1:
        if st.button("Comparar líneas"):
            for i in [1, 2]:
                if f"seleccion_{i}" in st.session_state:
                    st.session_state[f"id_linea_{i}"] = st.session_state[
                        f"seleccion_{i}"
                    ]["id_linea"]
                    st.session_state[f"nombre_linea_{i}"] = st.session_state[
                        f"seleccion_{i}"
                    ]["nombre_linea"]
                    st.session_state[f"branch_id_{i}"] = st.session_state[
                        f"seleccion_{i}"
                    ].get("branch_id")
                    st.session_state[f"branch_name_{i}"] = st.session_state[
                        f"seleccion_{i}"
                    ].get("branch_name")
                    st.write(
                        f"Línea {i} guardada:",
                        st.session_state[f"nombre_linea_{i}"],
                        "ramal",
                        (
                            st.session_state[f"branch_name_{i}"]
                            if st.session_state[f"branch_name_{i}"]
                            else "N/A"
                        ),
                    )
                else:
                    st.write(
                        f"No hay ninguna línea seleccionada para guardar como Línea {i}."
                    )

    with col2:
        st.subheader("Línea base:")
        seleccionar_linea("Línea base", "base_input", "base_select", "1", conn_insumos)
    with col3:
        st.subheader("Línea comparación:")
        seleccionar_linea(
            "Línea comparación", "comp_input", "comp_select", "2", conn_insumos
        )

# --- Comparación de líneas ---
with st.expander("Comparación de líneas", expanded=True):
    col1, col2 = st.columns([2, 2])
    
    if st.session_state.id_linea_1 and st.session_state.id_linea_2:
        if use_branches:
            base_route_id, comp_route_id = int(st.session_state.branch_id_1), int(
                st.session_state.branch_id_2
            )
        else:
            base_route_id, comp_route_id = int(st.session_state.id_linea_1), int(
                st.session_state.id_linea_2
            )
        
        # Evita cálculos repetidos si ya se han realizado para las mismas líneas
        if f"overlapping_dict_{base_route_id}_{comp_route_id}" not in st.session_state:

            overlapping_dict = ovl.compute_supply_overlapping(
                "weekday",
                base_route_id,
                comp_route_id,
                "branches" if use_branches else "lines",
                8,
            )
            st.session_state[f"overlapping_dict_{base_route_id}_{comp_route_id}"] = (
                overlapping_dict
            )
            st.session_state[f"supply_overlapping_{base_route_id}_{comp_route_id}"] = (
                overlapping_dict["text_base_v_comp"]
            )
            st.session_state[f"supply_overlapping_{comp_route_id}_{base_route_id}"] = (
                overlapping_dict["text_comp_v_base"]
            )

        overlapping_dict = st.session_state[
            f"overlapping_dict_{base_route_id}_{comp_route_id}"
        ]

        # Renderiza el primer mapa
        f = ovl_viz.plot_interactive_supply_overlapping(overlapping_dict)
        # Muestra la salida solo en col1
        with col1:
            st_folium(f, width=800, height=600)
            st.write(
            st.session_state[f"supply_overlapping_{base_route_id}_{comp_route_id}"]
            )
            st.write(
            st.session_state[f"supply_overlapping_{comp_route_id}_{base_route_id}"]
            )

        # Cálculo y visualización de la demanda, si no se ha realizado previamente
        if (
            f"base_demand_comp_demand_{base_route_id}_{comp_route_id}"
            not in st.session_state
        ):
            demand_overlapping = ovl.compute_demand_overlapping(
                st.session_state.id_linea_1,
                st.session_state.id_linea_2,
                "weekday",
                base_route_id,
                comp_route_id,
                overlapping_dict["base"]["h3"],
                overlapping_dict["comp"]["h3"],
            )
            st.session_state[
                f"base_demand_comp_demand_{base_route_id}_{comp_route_id}"
            ] = demand_overlapping

            st.session_state[f"demand_overlapping_{base_route_id}_{comp_route_id}"] = (
                demand_overlapping["base"]["output_text"]
            )
            st.session_state[f"demand_overlapping_{comp_route_id}_{base_route_id}"] = (
                demand_overlapping["comp"]["output_text"]
            )

        demand_overlapping = st.session_state[
            f"base_demand_comp_demand_{base_route_id}_{comp_route_id}"
        ]
        base_demand = demand_overlapping["base"]["data"]
        comp_demand = demand_overlapping["comp"]["data"]

        # Renderiza el segundo mapa y muestra el texto justo después del mapa en col2
        fig = ovl_viz.plot_interactive_demand_overlapping(
            base_demand, comp_demand, overlapping_dict
        )
        with col2:
            st_folium(fig, width=800, height=600)
            st.write(
                st.session_state[f"demand_overlapping_{base_route_id}_{comp_route_id}"]
            )  # Muestra la segunda salida justo después del mapa
            st.write(
                st.session_state[f"demand_overlapping_{comp_route_id}_{base_route_id}"]
            )  # Muestra la segunda salida justo después del mapa
