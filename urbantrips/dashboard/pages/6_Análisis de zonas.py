import pandas as pd
import streamlit as st
import h3
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import json
from folium import plugins
from shapely import wkt
from dash_utils import (
    iniciar_conexion_db,
    get_logo,
    bring_latlon,
    configurar_selector_dia,
)
from urbantrips.carto.carto import get_h3_indices_in_geometry
from urbantrips.geo.geo import h3_to_polygon
from streamlit_folium import folium_static

pd.options.display.float_format = "{:,.0f}".format


def levanto_tabla_sql_local(tabla_sql, tabla_tipo="dash", query=""):

    conn = iniciar_conexion_db(tipo=tabla_tipo)

    try:
        if len(query) == 0:
            query = f"""
            SELECT *
            FROM {tabla_sql}
            """

        tabla = pd.read_sql_query(query, conn)
    except:
        print(f"{tabla_sql} no existe")
        tabla = pd.DataFrame([])

    conn.close()

    if len(tabla) > 0:
        if "wkt" in tabla.columns:
            tabla["geometry"] = tabla.wkt.apply(wkt.loads)
            tabla = gpd.GeoDataFrame(tabla, crs=4326)
            tabla = tabla.drop(["wkt"], axis=1)

    return tabla


@st.cache_data
def traigo_mes_dia():
    mes_dia = levanto_tabla_sql_local(
        "etapas_agregadas",
        "dash",
        "SELECT DISTINCT mes, tipo_dia FROM etapas_agregadas;",
    )
    mes = mes_dia.mes.values.tolist()
    tipo_dia = mes_dia.tipo_dia.values.tolist()
    return mes, tipo_dia


# Convert H3 indices to GeoDataFrame
def h3_indices_to_gdf(h3_indices):
    hex_geometries = [h3_to_polygon(h) for h in h3_indices]
    return gpd.GeoDataFrame(
        {"h3_index": h3_indices}, geometry=hex_geometries, crs="EPSG:4326"
    )


# Initialize session state for zones
if "zona_1" not in st.session_state:
    st.session_state["zona_1"] = []
if "zona_2" not in st.session_state:
    st.session_state["zona_2"] = []


def main():

    st.set_page_config(layout="wide")
    logo = get_logo()
    st.image(logo)

    alias_seleccionado = configurar_selector_dia()

    latlon = bring_latlon()
    mes_lst, tipo_dia_lst = traigo_mes_dia()

    with st.expander("Selecciono zonas", expanded=True):
        col1, col2 = st.columns([1, 4])

        # # Sidebar controls
        # resolution = col1.slider("Selecciona la Resolución H3", min_value=0, max_value=15, value=8, step=1)
        resolution = 8

        # Initialize Folium map
        m = folium.Map(location=latlon, zoom_start=10)
        draw = plugins.Draw(
            export=False,
            draw_options={"polygon": True, "rectangle": True},
            edit_options={"edit": True, "remove": True},
        )
        draw.add_to(m)

        # Display map with drawing tools
        with col2:
            output = st_folium(m, width=700, height=700, key="map")

        # Handle user drawing
        if output.get("last_active_drawing"):
            geometry_data = output["last_active_drawing"]["geometry"]
            geometry = shape(geometry_data)
            h3_indices = get_h3_indices_in_geometry(geometry, resolution)

            # Save hexagons to session state based on button clicks
            if col1.button("Guardar en Zona 1"):
                st.session_state["zona_1"] = []
                st.session_state["zona_1"].extend(h3_indices)
            if col1.button("Guardar en Zona 2"):
                st.session_state["zona_2"] = []
                st.session_state["zona_2"].extend(h3_indices)

        zona1 = st.session_state["zona_1"]
        zona2 = st.session_state["zona_2"]

        # Convertir la lista de índices H3 a una cadena en formato de lista de Python
        zona1_str = json.dumps(zona1)
        col2.code(zona1_str, language="python")

        zona2_str = json.dumps(zona2)
        col2.code(zona2_str, language="python")

    with st.expander("Zonas", expanded=False):
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
        zona1 = st.session_state["zona_1"]
        zona2 = st.session_state["zona_2"]

        # mes_lst = ['Todos'] + etapas_all.mes.unique().tolist()
        desc_mes = col1.selectbox("Mes", options=mes_lst)

        desc_tipo_dia = col1.selectbox("Tipo dia", options=tipo_dia_lst)

        if len(zona1) > 0:
            h3_values1 = ", ".join(f"'{item}'" for item in zona1)
            ## Etapas
            query1 = f"SELECT * FROM etapas_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values1}));"
            etapas1 = levanto_tabla_sql_local(
                "etapas_agregadas", tabla_tipo="dash", query=query1
            )

            if len(etapas1) > 0:
                etapas1["Zona_1"] = "Zona 1"

                ## Viajes
                query1 = f"SELECT * FROM viajes_agregados WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values1}) );"
                viajes1 = levanto_tabla_sql_local(
                    "viajes_agregados", tabla_tipo="dash", query=query1
                )
                viajes1["Zona_1"] = "Zona 1"

                modos_e1 = (
                    etapas1.groupby(["modo", "nombre_linea"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(
                        columns={
                            "factor_expansion_linea": "Etapas",
                            "nombre_linea": "Línea",
                            "modo": "Modo",
                        }
                    )
                )

                modos_v1 = (
                    viajes1.groupby(["modo"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(
                        columns={"factor_expansion_linea": "Viajes", "modo": "Modo"}
                    )
                )

                # Calculate the total and append as a new row
                total_row1e = pd.DataFrame(
                    {
                        "Modo": ["Total"],
                        "Línea": ["-"],
                        "Etapas": [modos_e1["Etapas"].sum()],
                    }
                )
                modos_e1 = pd.concat([modos_e1, total_row1e], ignore_index=True)

                # Calculate the total and append as a new row
                total_row1 = pd.DataFrame(
                    {"Modo": ["Total"], "Viajes": [modos_v1["Viajes"].sum()]}
                )
                modos_v1 = pd.concat([modos_v1, total_row1], ignore_index=True)

                col2.title("Zona 1")
                col2.write("Etapas")
                modos_e1["Etapas"] = modos_e1["Etapas"].round()
                col2.dataframe(modos_e1.set_index("Modo"), height=400, width=400)
                col3.title("")
                col3.write("Viajes")
                modos_v1["Viajes"] = modos_v1["Viajes"].round()
                col3.dataframe(modos_v1.set_index("Modo"), height=400, width=300)
        if len(zona2) > 0:
            h3_values2 = ", ".join(f"'{item}'" for item in zona2)
            ## Etapas
            query2 = f"SELECT * FROM etapas_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values2}));"
            etapas2 = levanto_tabla_sql_local(
                "etapas_agregadas", tabla_tipo="dash", query=query2
            )
            if len(etapas2) > 0:
                etapas2["Zona_2"] = "Zona 2"

                ## Viajes
                query2 = f"SELECT * FROM viajes_agregados WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values2}) );"
                viajes2 = levanto_tabla_sql_local(
                    "viajes_agregados", tabla_tipo="dash", query=query2
                )
                viajes2["Zona_2"] = "Zona 2"

                modos_e2 = (
                    etapas2.groupby(["modo", "nombre_linea"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(
                        columns={
                            "factor_expansion_linea": "Etapas",
                            "nombre_linea": "Línea",
                            "modo": "Modo",
                        }
                    )
                )

                modos_v2 = (
                    viajes2.groupby(["modo"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(
                        columns={"factor_expansion_linea": "Viajes", "modo": "Modo"}
                    )
                )
                # Calculate the total and append as a new row
                total_row2e = pd.DataFrame(
                    {
                        "Modo": ["Total"],
                        "Línea": ["-"],
                        "Etapas": [modos_e2["Etapas"].sum()],
                    }
                )
                modos_e2 = pd.concat([modos_e2, total_row2e], ignore_index=True)

                # Calculate the total and append as a new row
                total_row2 = pd.DataFrame(
                    {"Modo": ["Total"], "Viajes": [modos_v2["Viajes"].sum()]}
                )
                modos_v2 = pd.concat([modos_v2, total_row2], ignore_index=True)

                col4.title("Zona 2")
                col4.write("Etapas")
                modos_e2["Etapas"] = modos_e2["Etapas"].round()
                col4.dataframe(modos_e2.set_index("Modo"), height=400, width=400)

                modos_v2["Viajes"] = modos_v2["Viajes"].round()
                col5.title("")
                col5.write("Viajes")
                # col5.markdown(modos_v2.to_html(index=False), unsafe_allow_html=True)
                col5.dataframe(modos_v2.set_index("Modo"), height=400, width=300)

    with st.expander("Viajes entre zonas", expanded=True):
        col1, col2, col3, col4 = st.columns([1, 2, 2, 3])

        if len(zona1) > 0 and len(zona2) > 0:
            h3_values = ", ".join(f"'{item}'" for item in zona1 + zona2)
            ## Etapas
            query = f"SELECT * FROM etapas_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values}) OR h3_d IN ({h3_values}));"
            etapas = levanto_tabla_sql_local(
                "etapas_agregadas", tabla_tipo="dash", query=query
            )

            if len(etapas) > 0:

                etapas["Zona_1"] = ""
                etapas["Zona_2"] = ""
                etapas.loc[etapas.h3_o.isin(zona1), "Zona_1"] = "Zona 1"
                etapas.loc[etapas.h3_o.isin(zona2), "Zona_1"] = "Zona 2"
                etapas.loc[etapas.h3_d.isin(zona1), "Zona_2"] = "Zona 1"
                etapas.loc[etapas.h3_d.isin(zona2), "Zona_2"] = "Zona 2"
                etapas = etapas[
                    (etapas.Zona_1 != "")
                    & (etapas.Zona_2 != "")
                    & (etapas.Zona_1 != etapas.Zona_2)
                ]

                etapas = etapas.fillna("")

                zonasod_e = (
                    etapas.groupby(["Zona_1", "Zona_2"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(columns={"factor_expansion_linea": "Etapas"})
                    .round(0)
                )

                modos_e = (
                    etapas.groupby(["modo", "nombre_linea"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(
                        columns={
                            "factor_expansion_linea": "Etapas",
                            "nombre_linea": "Línea",
                            "modo": "Modo",
                        }
                    )
                    .round(0)
                )

                # Calculate the total and append as a new row
                total_rowe = pd.DataFrame(
                    {
                        "Modo": ["Total"],
                        "Línea": ["-"],
                        "Etapas": [modos_e["Etapas"].sum()],
                    }
                )
                modos_e = pd.concat([modos_e, total_rowe], ignore_index=True)

                modos_e["Etapas"] = modos_e["Etapas"].round()

                zonasod_e["Zonas"] = zonasod_e["Zona_1"] + " - " + zonasod_e["Zona_2"]
                zonasod_e = zonasod_e[["Zonas", "Etapas"]]
                zonasod_e["Etapas"] = zonasod_e["Etapas"].apply(lambda x: f"{int(x):,}")

                col2.write("Etapas")
                if len(zonasod_e) > 0:
                    col2.dataframe(zonasod_e.set_index("Zonas"), height=100, width=300)
                else:
                    col2.write("No hay datos para mostrar")

                ## Viajes
                h3_values = ", ".join(f"'{item}'" for item in zona1 + zona2)
                query = f"SELECT * FROM viajes_agregados WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values}) OR h3_d IN ({h3_values}));"
                viajes = levanto_tabla_sql_local(
                    "viajes_agregados", tabla_tipo="dash", query=query
                )

                viajes["Zona_1"] = ""
                viajes["Zona_2"] = ""
                viajes.loc[viajes.h3_o.isin(zona1), "Zona_1"] = "Zona 1"
                viajes.loc[viajes.h3_o.isin(zona2), "Zona_1"] = "Zona 2"
                viajes.loc[viajes.h3_d.isin(zona1), "Zona_2"] = "Zona 1"
                viajes.loc[viajes.h3_d.isin(zona2), "Zona_2"] = "Zona 2"
                viajes = viajes[
                    (viajes.Zona_1 != "")
                    & (viajes.Zona_2 != "")
                    & (viajes.Zona_1 != viajes.Zona_2)
                ]

                zonasod_v = (
                    viajes.groupby(["Zona_1", "Zona_2"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(columns={"factor_expansion_linea": "Viajes"})
                )
                zonasod_v["Zonas"] = zonasod_v["Zona_1"] + " - " + zonasod_v["Zona_2"]
                zonasod_v = zonasod_v[["Zonas", "Viajes"]]
                zonasod_v["Viajes"] = zonasod_v["Viajes"].apply(lambda x: f"{int(x):,}")

                modos_v = (
                    viajes.groupby(["modo"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(
                        columns={"factor_expansion_linea": "Viajes", "modo": "Modo"}
                    )
                )

                # Calculate the total and append as a new row
                total_row = pd.DataFrame(
                    {"Modo": ["Total"], "Viajes": [modos_v["Viajes"].sum()]}
                )
                modos_v = pd.concat([modos_v, total_row], ignore_index=True)

                col3.write("Viajes")
                if len(zonasod_v):
                    col3.dataframe(zonasod_v.set_index("Zonas"), height=100, width=300)
                else:
                    col3.write("No hay datos para mostrar")

                modos_v["Viajes"] = modos_v["Viajes"].round()
                col2.write("Modal")
                col2.dataframe(modos_v.set_index("Modo"), height=200, width=300)

                ## Mapa

                # Create unified geometry for each zone
                def zona_to_geometry(h3_list):
                    polygons = [h3_to_polygon(h3_index) for h3_index in h3_list]
                    return unary_union(polygons)

                geometry_zona1 = zona_to_geometry(st.session_state["zona_1"])
                geometry_zona2 = zona_to_geometry(st.session_state["zona_2"])
                gdf = gpd.GeoDataFrame(
                    {
                        "zona": ["Zona 1", "Zona 2"],
                        "geometry": [geometry_zona1, geometry_zona2],
                    },
                    crs="EPSG:4326",
                )

                # Plot the zones on a new Folium map
                m2 = folium.Map(
                    location=[
                        gdf.geometry.centroid.y.mean(),
                        gdf.geometry.centroid.x.mean(),
                    ],
                    zoom_start=10,
                )
                folium.GeoJson(gdf, name="GeoData").add_to(m2)

                with col4:
                    output2 = st_folium(m2, width=700, height=700)

            ## Transferencias
            h3_values = ", ".join(f"'{item}'" for item in zona1 + zona2)
            query = f"SELECT * FROM transferencias_agregadas WHERE mes = '{desc_mes}' AND tipo_dia = '{desc_tipo_dia}' AND (h3_o IN ({h3_values}) OR h3_d IN ({h3_values}));"
            transferencias = levanto_tabla_sql_local(
                "transferencias_agregadas", tabla_tipo="dash", query=query
            )

            if len(transferencias) > 0:

                transferencias["Zona_1"] = ""
                transferencias["Zona_2"] = ""
                transferencias.loc[transferencias.h3_o.isin(zona1), "Zona_1"] = "Zona 1"
                transferencias.loc[transferencias.h3_o.isin(zona2), "Zona_1"] = "Zona 2"
                transferencias.loc[transferencias.h3_d.isin(zona1), "Zona_2"] = "Zona 1"
                transferencias.loc[transferencias.h3_d.isin(zona2), "Zona_2"] = "Zona 2"
                transferencias = transferencias[
                    (transferencias.Zona_1 != "")
                    & (transferencias.Zona_2 != "")
                    & (transferencias.Zona_1 != transferencias.Zona_2)
                ]

                transferencias = transferencias.fillna("")

                transferencias = (
                    transferencias.groupby(["modo", "seq_lineas"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(
                        columns={
                            "factor_expansion_linea": "Viajes",
                            "modo": "Modo",
                            "seq_lineas": "Líneas",
                        }
                    )
                    .sort_values("Viajes", ascending=False)
                )
                transferencias["Viajes"] = transferencias["Viajes"].astype(int)

                col3.write("Viajes por líneas")
                col3.dataframe(transferencias.set_index("Modo"), height=500, width=500)


if __name__ == "__main__":
    main()
