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
from dash_storage import leer_configs_generales
from dash_utils import (
    get_logo,
    bring_latlon,
    configurar_selector_dia,
    traer_dias_chains,
    get_h3_indices_in_geometry,
    h3_to_polygon,
)
from urbantrips.utils import utils

# from urbantrips.carto.carto import get_h3_indices_in_geometry
# from urbantrips.geo.geo import h3_to_polygon
from streamlit_folium import folium_static

pd.options.display.float_format = "{:,.0f}".format

# Resolución de los h3 de chains_norm (las cadenas se guardan a esta resolución).
RES_CHAINS = 10


def levanto_tabla_sql_local(tabla_sql, tabla_tipo="dash", query=""):
    return utils.levanto_tabla_sql(
        tabla_sql,
        tabla_tipo=tabla_tipo,
        query=query,
    )


# Convert H3 indices to GeoDataFrame
def h3_indices_to_gdf(h3_indices):
    hex_geometries = [h3_to_polygon(h) for h in h3_indices]
    return gpd.GeoDataFrame(
        {"h3_index": h3_indices}, geometry=hex_geometries, crs="EPSG:4326"
    )


def _zona_a_res_chains(zona_cells):
    """Lleva las celdas de la zona (dibujadas a resolucion_h3) a la resolución
    de chains_norm (res 10), para poder comparar contra los nodos h3 de las
    cadenas. Cada celda de menor resolución se expande a sus hijos res 10;
    `nodo in zona_res10` equivale exactamente a `parent(nodo) in zona`."""
    out = set()
    for c in zona_cells:
        try:
            res = h3.get_resolution(c)
            if res < RES_CHAINS:
                out |= set(h3.cell_to_children(c, RES_CHAINS))
            elif res == RES_CHAINS:
                out.add(c)
            else:
                out.add(h3.cell_to_parent(c, RES_CHAINS))
        except Exception:
            out.add(c)
    return out


def _traer_chains_zonas(desc_dia, nodos):
    """Viajes (chains_norm) del día cuyos nodos (inicio/transfer1/transfer2/fin)
    tocan alguna de las celdas `nodos` (set de h3 res 10)."""
    cols = (
        "h3_inicio, h3_transfer1, h3_transfer2, h3_fin, "
        "seq_lineas, modo_agregado, transferencia, factor_expansion_linea"
    )
    where = f"WHERE dia = '{desc_dia}'"
    nodos = list(nodos)
    if nodos:
        vals = ", ".join(f"'{h}'" for h in nodos)
        where += (
            f" AND (h3_inicio IN ({vals}) OR h3_transfer1 IN ({vals}) "
            f"OR h3_transfer2 IN ({vals}) OR h3_fin IN ({vals}))"
        )
    return levanto_tabla_sql_local(
        "chains_norm", "dash", f"SELECT {cols} FROM chains_norm {where}"
    )


def _explotar_etapas(chains):
    """Una fila por etapa. Separa `seq_lineas` (cada línea = una etapa) y le
    asigna origen/destino h3 usando los nodos de la cadena en orden
    (inicio, transfer1, transfer2, fin). El modo es `modo_agregado` (a nivel
    viaje; chains_norm no guarda el modo por tramo)."""
    filas = []
    for r in chains.itertuples(index=False):
        lineas = [x.strip() for x in str(r.seq_lineas).split("--") if x.strip()]
        if not lineas:
            continue
        nodos = [
            n
            for n in (r.h3_inicio, r.h3_transfer1, r.h3_transfer2, r.h3_fin)
            if isinstance(n, str) and n
        ]
        n = len(lineas)
        if len(nodos) >= n + 1:
            origenes = nodos[:n]
            destinos = nodos[1 : n + 1]
        else:
            # datos inconsistentes: uso origen->destino del viaje para cada tramo
            origenes = [r.h3_inicio] * n
            destinos = [r.h3_fin] * n
        for i in range(n):
            filas.append(
                (
                    origenes[i],
                    destinos[i],
                    r.modo_agregado,
                    lineas[i],
                    r.factor_expansion_linea,
                )
            )
    return pd.DataFrame(
        filas,
        columns=["h3_o", "h3_d", "modo", "nombre_linea", "factor_expansion_linea"],
    )


def _viajes_desde_chains(chains):
    """Vista trip-level con los nombres de columna que espera el display:
    h3_o (origen), h3_d (destino), modo."""
    return chains.rename(
        columns={"h3_inicio": "h3_o", "h3_fin": "h3_d", "modo_agregado": "modo"}
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

    with st.expander("Selecciono zonas", expanded=True):
        col1, col2 = st.columns([1, 4])

        try:
            # --- Cargar configuraciones y conexiones en session_state ---
            if "configs" not in st.session_state:
                # autogenerado=False: leer el config base (configuraciones_generales.yaml),
                # consistente con la resolución de DB. No usamos el autogenerado en esta versión.
                st.session_state.configs = leer_configs_generales(autogenerado=False)

            configs = st.session_state.configs
            # defino la resolución en base a las configuraciones generales
            resolution = configs["resolucion_h3"]
            st.write("Resolución h3 para etapas:", resolution)

        except ValueError as e:
            st.error(f"No se pudo leer la resolucion del config general: {e}. \n")
            st.stop()

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

        # Selector de día específico (sin 'Todos')
        dias_disponibles = traer_dias_chains()
        if len(dias_disponibles) == 0:
            st.warning("No hay días disponibles en chains_norm.")
            st.stop()
        desc_dia = col1.selectbox("Día", options=dias_disponibles)

        # Zonas a la resolución de chains_norm (res 10) para comparar nodos
        zona1_r = _zona_a_res_chains(zona1)
        zona2_r = _zona_a_res_chains(zona2)

        if len(zona1) > 0:
            chains1 = _traer_chains_zonas(desc_dia, zona1_r)
            # Etapas con ORIGEN en zona 1
            etapas1 = _explotar_etapas(chains1)
            etapas1 = etapas1[etapas1.h3_o.isin(zona1_r)]
            if len(etapas1) > 0:
                # Viajes con ORIGEN en zona 1
                viajes1 = _viajes_desde_chains(chains1)
                viajes1 = viajes1[viajes1.h3_o.isin(zona1_r)]

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
            chains2 = _traer_chains_zonas(desc_dia, zona2_r)
            # Etapas con ORIGEN en zona 2
            etapas2 = _explotar_etapas(chains2)
            etapas2 = etapas2[etapas2.h3_o.isin(zona2_r)]
            if len(etapas2) > 0:
                # Viajes con ORIGEN en zona 2
                viajes2 = _viajes_desde_chains(chains2)
                viajes2 = viajes2[viajes2.h3_o.isin(zona2_r)]

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
            # Un solo pull de chains_norm para ambas zonas; se reutiliza para
            # etapas, viajes y transferencias.
            chains = _traer_chains_zonas(desc_dia, zona1_r | zona2_r)

            ## Etapas
            etapas = _explotar_etapas(chains)

            if len(etapas) > 0:

                etapas["Zona_1"] = ""
                etapas["Zona_2"] = ""
                etapas.loc[etapas.h3_o.isin(zona1_r), "Zona_1"] = "Zona 1"
                etapas.loc[etapas.h3_o.isin(zona2_r), "Zona_1"] = "Zona 2"
                etapas.loc[etapas.h3_d.isin(zona1_r), "Zona_2"] = "Zona 1"
                etapas.loc[etapas.h3_d.isin(zona2_r), "Zona_2"] = "Zona 2"
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
                viajes = _viajes_desde_chains(chains)

                viajes["Zona_1"] = ""
                viajes["Zona_2"] = ""
                viajes.loc[viajes.h3_o.isin(zona1_r), "Zona_1"] = "Zona 1"
                viajes.loc[viajes.h3_o.isin(zona2_r), "Zona_1"] = "Zona 2"
                viajes.loc[viajes.h3_d.isin(zona1_r), "Zona_2"] = "Zona 1"
                viajes.loc[viajes.h3_d.isin(zona2_r), "Zona_2"] = "Zona 2"
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
            transferencias = _viajes_desde_chains(chains)

            transferencias["Zona_1"] = ""
            transferencias["Zona_2"] = ""
            transferencias.loc[transferencias.h3_o.isin(zona1_r), "Zona_1"] = "Zona 1"
            transferencias.loc[transferencias.h3_o.isin(zona2_r), "Zona_1"] = "Zona 2"
            transferencias.loc[transferencias.h3_d.isin(zona1_r), "Zona_2"] = "Zona 1"
            transferencias.loc[transferencias.h3_d.isin(zona2_r), "Zona_2"] = "Zona 2"
            transferencias = transferencias[
                (transferencias.Zona_1 != "")
                & (transferencias.Zona_2 != "")
                & (transferencias.Zona_1 != transferencias.Zona_2)
            ]

            transferencias = transferencias.fillna("")

            if len(transferencias) > 0:

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
