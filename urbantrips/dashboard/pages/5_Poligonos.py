import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import mapclassify
import plotly.express as px
from folium import Figure
from shapely import wkt
import numpy as np
from dash_utils import (
    levanto_tabla_sql,
    levanto_tabla_sql_local,
    get_logo,
    create_data_folium,
    traigo_indicadores,
    extract_hex_colors_from_cmap,
    iniciar_conexion_db,
    normalize_vars,
    traigo_lista_zonas,
    configurar_selector_dia,
    calcular_bins,
)

from streamlit_folium import folium_static

import mapclassify

import mapclassify
import numpy as np
from collections import OrderedDict


def crear_mapa_poligonos(
    df_viajes,
    df_etapas,
    poly,
    zonif,
    origenes,
    destinos,
    var_fex,
    cmap_viajes="Blues",
    cmap_etapas="Greens",
    map_title="",
    savefile="",
    k_jenks=5,
    show_poly=False,
):

    m = ""
    if (
        (len(df_viajes) > 0)
        | (len(df_etapas) > 0)
        | (len(origenes) > 0)
        | (len(destinos) > 0)
    ):

        fig = Figure(width=800, height=600)
        m = folium.Map(
            location=[
                poly.geometry.representative_point().y.mean(),
                poly.geometry.representative_point().x.mean(),
            ],
            zoom_start=9,
            tiles="cartodbpositron",
        )

        colors_viajes = extract_hex_colors_from_cmap(cmap="viridis_r", n=k_jenks)
        colors_etapas = extract_hex_colors_from_cmap(cmap="magma_r", n=k_jenks)

        # Etapas
        line_w = 0.5
        if len(df_etapas) > 0:

            # bins = calcular_bins(df_etapas, var_fex, 5)

            # range_bins = range(0, len(bins) - 1)
            # bins_labels = [
            #     f"{int(bins[n])} a {int(bins[n+1])} etapas" for n in range_bins
            # ]
            # df_etapas["cuts"] = pd.cut(
            #     df_etapas[var_fex], bins=bins, labels=bins_labels
            # )

            df_etapas, bins_labels = calcular_bins(df_etapas, var_fex, 5)

            n = 0
            for i in bins_labels:

                df_etapas[df_etapas.cuts == i].explore(
                    m=m,
                    color=colors_etapas[n],
                    style_kwds={"fillOpacity": 0.1, "weight": line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 3

        # Viajes
        line_w = 0.5
        if len(df_viajes) > 0:

            # bins = calcular_bins(df_viajes, var_fex, 5)

            # range_bins = range(0, len(bins) - 1)

            # bins_limpios = sorted(set(
            #     0 if pd.isna(b) or np.isinf(b) else b
            #     for b in bins
            # ))

            # # 3. Crear etiquetas seguras
            # bins_labels = [
            #     f"{int(bins_limpios[n])} a {int(bins_limpios[n+1])} viajes"
            #     for n in range(len(bins_limpios) - 1)
            # ]

            # # 4. Cortar y asignar categorías
            # df_viajes["cuts"] = pd.cut(
            #     df_viajes[var_fex],
            #     bins=bins_limpios,
            #     labels=bins_labels,
            #     include_lowest=True
            # )
            df_viajes, bins_labels = calcular_bins(df_viajes, var_fex, 5)

            n = 0
            for i in bins_labels:

                df_viajes[df_viajes.cuts == i].explore(
                    m=m,
                    color=colors_viajes[n],
                    style_kwds={"fillOpacity": 0.1, "weight": line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 3

        if len(origenes) > 0:

            # bins = calcular_bins(origenes, var_fex, 5)

            # range_bins = range(0, len(bins) - 1)
            # bins_labels = [
            #     f"{int(bins[n])} a {int(bins[n+1])} origenes" for n in range_bins
            # ]

            # origenes["cuts"] = pd.cut(
            #     origenes[var_fex], bins=bins, labels=bins_labels
            # )
            origenes, bins_labels = calcular_bins(origenes, var_fex, 5)

            n = 0
            line_w = 10
            for i in bins_labels:

                origenes[origenes.cuts == i].explore(
                    m=m,
                    # color=colors_origenes[n],
                    color="#0173b299",
                    style_kwds={"fillOpacity": 0.1, "weight": line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 5

        if len(destinos) > 0:

            # bins = calcular_bins(destinos, var_fex, 5)

            # range_bins = range(0, len(bins) - 1)
            # bins_labels = [
            #     f"{int(bins[n])} a {int(bins[n+1])} destinos" for n in range_bins
            # ]

            # destinos["cuts"] = pd.cut(
            #     destinos[var_fex], bins=bins, labels=bins_labels
            # )
            destinos, bins_labels = calcular_bins(destinos, var_fex, 5)

            n = 0
            line_w = 10
            for i in bins_labels:

                destinos[destinos.cuts == i].explore(
                    m=m,
                    # color=colors_destinos[n],
                    color="#de8f0599",
                    style_kwds={"fillOpacity": 0.1, "weight": line_w},
                    name=i,
                    tooltip=False,
                )
                n += 1
                line_w += 5

        # Agrego polígono
        if (len(poly) > 0) & (show_poly):
            geojson = poly.to_json()
            # Add the GeoJSON to the map as a GeoJson Layer
            folium.GeoJson(
                geojson,
                name=poly.id.values[0],
                style_function=lambda feature: {
                    "fillColor": "grey",
                    "color": "white",
                    "weight": 2,
                    "fillOpacity": 0.5,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["id"], labels=False, sticky=False
                ),
            ).add_to(m)

        # Agrego polígono
        if len(zonif) > 0:
            geojson = zonif.to_json()
            # Add the GeoJSON to the map as a GeoJson Layer
            folium.GeoJson(
                geojson,
                name="Zonificación",
                style_function=lambda feature: {
                    "fillColor": "navy",
                    "color": "navy",
                    "weight": 0.5,
                    "fillOpacity": 0.2,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["id"], labels=False, sticky=False
                ),
            ).add_to(m)

        folium.LayerControl(name="xx").add_to(m)

    return m


def traigo_socio_indicadores(socio_indicadores):

    df = socio_indicadores[socio_indicadores.tabla == "viajes-genero-tarifa"].copy()
    totals = (
        pd.crosstab(
            values=df.factor_expansion_linea,
            columns=df.Genero,
            index=df.Tarifa,
            aggfunc="sum",
            margins=True,
            margins_name="Total",
            normalize=False,
        )
        .fillna(0)
        .round()
        .astype(int)
        .apply(lambda col: col.map(lambda x: f"{x:,.0f}".replace(",", ".")))
    )
    totals_porc = (
        pd.crosstab(
            values=df.factor_expansion_linea,
            columns=df.Genero,
            index=df.Tarifa,
            aggfunc="sum",
            margins=True,
            margins_name="Total",
            normalize=True,
        )
        * 100
    ).round(2)

    modos = socio_indicadores[socio_indicadores.tabla == "etapas-genero-modo"].copy()
    modos_genero_abs = (
        pd.crosstab(
            values=modos.factor_expansion_linea,
            index=[modos.Genero],
            columns=modos.Modo,
            aggfunc="sum",
            normalize=False,
            margins=True,
            margins_name="Total",
        )
        .fillna(0)
        .astype(int)
        .apply(lambda col: col.map(lambda x: f"{x:,.0f}".replace(",", ".")))
    )
    modos_genero_porc = (
        pd.crosstab(
            values=modos.factor_expansion_linea,
            index=modos.Genero,
            columns=modos.Modo,
            aggfunc="sum",
            normalize=True,
            margins=True,
            margins_name="Total",
        )
        * 100
    ).round(2)

    modos = socio_indicadores[socio_indicadores.tabla == "etapas-tarifa-modo"].copy()
    modos_tarifa_abs = (
        pd.crosstab(
            values=modos.factor_expansion_linea,
            index=[modos.Tarifa],
            columns=modos.Modo,
            aggfunc="sum",
            normalize=False,
            margins=True,
            margins_name="Total",
        )
        .fillna(0)
        .astype(int)
        .apply(lambda col: col.map(lambda x: f"{x:,.0f}".replace(",", ".")))
    )
    modos_tarifa_porc = (
        pd.crosstab(
            values=modos.factor_expansion_linea,
            index=modos.Tarifa,
            columns=modos.Modo,
            aggfunc="sum",
            normalize=True,
            margins=True,
            margins_name="Total",
        )
        * 100
    ).round(2)

    avg_distances = (
        pd.crosstab(
            values=df.Distancia,
            columns=df.Genero,
            index=df.Tarifa,
            margins=True,
            margins_name="Total",
            aggfunc=lambda x: (x * df.loc[x.index, "factor_expansion_linea"]).sum()
            / df.loc[x.index, "factor_expansion_linea"].sum(),
        )
        .fillna(0)
        .round(2)
    )
    avg_times = (
        pd.crosstab(
            values=df["Tiempo de viaje"],
            columns=df.Genero,
            index=df.Tarifa,
            margins=True,
            margins_name="Total",
            aggfunc=lambda x: (x * df.loc[x.index, "factor_expansion_linea"]).sum()
            / df.loc[x.index, "factor_expansion_linea"].sum(),
        )
        .fillna(0)
        .round(2)
    )
    avg_velocity = (
        pd.crosstab(
            values=df["Velocidad"],
            columns=df.Genero,
            index=df.Tarifa,
            margins=True,
            margins_name="Total",
            aggfunc=lambda x: (x * df.loc[x.index, "factor_expansion_linea"]).sum()
            / df.loc[x.index, "factor_expansion_linea"].sum(),
        )
        .fillna(0)
        .round(2)
    )
    avg_etapas = (
        pd.crosstab(
            values=df["Etapas promedio"],
            columns=df.Genero,
            index=df.Tarifa,
            margins=True,
            margins_name="Total",
            aggfunc=lambda x: (x * df.loc[x.index, "factor_expansion_linea"]).sum()
            / df.loc[x.index, "factor_expansion_linea"].sum(),
        )
        .round(2)
        .fillna("")
    )
    user = socio_indicadores[socio_indicadores.tabla == "usuario-genero-tarifa"].copy()
    avg_viajes = (
        pd.crosstab(
            values=user["Viajes promedio"],
            index=[user.Tarifa],
            columns=user.Genero,
            margins=True,
            margins_name="Total",
            aggfunc=lambda x: (x * user.loc[x.index, "factor_expansion_linea"]).sum()
            / user.loc[x.index, "factor_expansion_linea"].sum(),
        )
        .round(2)
        .fillna("")
    )

    avg_tiempo_entre_viajes = (
        pd.crosstab(
            values=df["Tiempo entre viajes"],
            columns=df.Genero,
            index=df.Tarifa,
            margins=True,
            margins_name="Total",
            aggfunc=lambda x: (x * df.loc[x.index, "factor_expansion_linea"]).sum()
            / df.loc[x.index, "factor_expansion_linea"].sum(),
        )
        .fillna(0)
        .round(2)
    )

    return (
        totals,
        totals_porc,
        avg_distances,
        avg_times,
        avg_velocity,
        modos_genero_abs,
        modos_genero_porc,
        modos_tarifa_abs,
        modos_tarifa_porc,
        avg_viajes,
        avg_etapas,
        avg_tiempo_entre_viajes,
    )


# Función para detectar cambios
def hay_cambios_en_filtros(current, last):
    return current != last


# ---
st.set_page_config(layout="wide")
logo = get_logo()
st.image(logo)
alias_seleccionado = configurar_selector_dia()

with st.expander("Líneas de Deseo", expanded=True):

    col1, col2, col3 = st.columns([1, 7, 1])

    variables = [
        "last_filters",
        "last_options",
        "data_cargada",
        "etapas_lst",
        "matrices_all",
        "etapas_all",
        "matrices_",
        "etapas_",
        "etapas",
        "viajes",
        "matriz",
        "origenes",
        "destinos",
        "general",
        "modal",
        "distancias",
        "dia",
        "zona",
        "transferencia",
        "modo_agregado",
        "rango_hora",
        "distancia_agregada",
        "socio_indicadores_",
        "general_",
        "modal_",
        "distancias_",
        "zonif",
        "desc_transfers",
        "desc_modos",
        "desc_horas",
        "desc_distancia",
        "agg_cols_etapas",
        "agg_cols_viajes",
    ]

    # Inicializar todas las variables con None si no existen en session_state
    for var in variables:
        if var not in st.session_state:
            st.session_state[var] = ""

    etapas_lst_ = levanto_tabla_sql(
        "poly_etapas", "dash", "SELECT DISTINCT dia FROM poly_etapas;"
    )
    poligonos = levanto_tabla_sql("poligonos", "insumos")

    # st.session_state.etapas = traigo()

    if (len(poligonos) > 0) & (len(etapas_lst_) > 0):

        zonificaciones = levanto_tabla_sql("zonificaciones", "insumos")
        socio_indicadores = levanto_tabla_sql("socio_indicadores")
        # desc_tipo_dia_ = levanto_tabla_sql(
        #     "poly_etapas", "dash", "SELECT DISTINCT tipo_dia FROM poly_etapas;"
        # )
        desc_zona_ = levanto_tabla_sql(
            "poly_etapas", "dash", "SELECT DISTINCT zona FROM poly_etapas;"
        ).sort_values("zona")
        modos_list_all_ = levanto_tabla_sql(
            "poly_etapas", "dash", "SELECT DISTINCT modo_agregado FROM poly_etapas;"
        )
        rango_hora_all_ = levanto_tabla_sql(
            "poly_etapas", "dash", "SELECT DISTINCT rango_hora FROM poly_etapas;"
        )
        distancia_all_ = levanto_tabla_sql(
            "poly_etapas", "dash", "SELECT DISTINCT distancia FROM poly_etapas;"
        )
        desc_poly_all_ = levanto_tabla_sql(
            "poly_etapas", "dash", "SELECT DISTINCT id_polygon FROM poly_etapas;"
        )
        zonas_values = traigo_lista_zonas("poligonos")

        # st.session_state.etapas_all = st.session_state.etapas_all[st.session_state.etapas_all.factor_expansion_linea > 0].copy()
        general, modal, distancias = traigo_indicadores("poligonos")

        # Inicializar valores de `st.session_state` solo si no existen
        if "last_filters" not in st.session_state:
            st.session_state.last_filters = {
                "dia": "Todos",
                "zona": None,
                "transferencia": "Todos",
                "modo_agregado": "Todos",
                "rango_hora": "Todos",
                "distancia_agregada": "Todas",
                "desc_zonas_values": "Todos",
            }

        if "data_cargada" not in st.session_state:
            st.session_state.data_cargada = False

        # Opciones de los filtros en Streamlit
        # st.session_state.etapas_lst = ['Todos'] + etapas_lst_.mes.unique().tolist()
        st.session_state.etapas_lst = etapas_lst_.dia.unique().tolist()
        desc_dia = col1.selectbox("Día", options=st.session_state.etapas_lst)

        # desc_tipo_dia = col1.selectbox(
        #     "Tipo día", options=desc_tipo_dia_.tipo_dia.unique()
        # )

        st.session_state.desc_poly = col1.selectbox(
            "Polígono", options=desc_poly_all_.id_polygon.unique()
        )

        desc_zona = col1.selectbox("Zonificación", options=desc_zona_.zona.unique())
        transf_list_all = ["Todos", "Con transferencia", "Sin transferencia"]
        transf_list = col1.selectbox("Transferencias", options=transf_list_all)

        modos_list_all = ["Todos"] + modos_list_all_[
            modos_list_all_.modo_agregado != "99"
        ].modo_agregado.unique().tolist()
        # modos_list = col1.selectbox('Modos', options=[text.capitalize() for text in modos_list_all])
        modos_list = col1.selectbox("Modos", options=[text for text in modos_list_all])

        rango_hora_all = ["Todos"] + rango_hora_all_[
            rango_hora_all_.rango_hora != "99"
        ].rango_hora.unique().tolist()
        # rango_hora = col1.selectbox('Rango hora', options=[text.capitalize() for text in rango_hora_all])
        rango_hora = col1.selectbox(
            "Rango hora", options=[text for text in rango_hora_all]
        )

        distancia_all = ["Todas"] + distancia_all_[
            distancia_all_.distancia != "99"
        ].distancia.unique().tolist()

        distancia_agregada = col1.selectbox("Distancia", options=distancia_all)

        desc_et_vi = col1.selectbox(
            "Datos de", options=["Etapas", "Viajes", "Ninguno"], index=1
        )
        if desc_et_vi == "Viajes":
            desc_viajes = True
            desc_etapas = False
        elif desc_et_vi == "Etapas":
            desc_viajes = False
            desc_etapas = True
        else:
            desc_viajes = False
            desc_etapas = False

        zonas_values_all = ["Todos"] + zonas_values[
            zonas_values.zona == desc_zona
        ].Nombre.unique().tolist()
        desc_zonas_values1 = col3.selectbox(
            "Filtro 1", options=zonas_values_all, key="filtro1"
        )
        desc_zonas_values2 = col3.selectbox(
            "Filtro 2", options=zonas_values_all, key="filtro2"
        )

        desc_origenes = col3.checkbox(":blue[Origenes]", value=False)

        desc_destinos = col3.checkbox(":orange[Destinos]", value=False)

        desc_zonif = col3.checkbox("Mostrar zonificación", value=True)
        if desc_zonif:
            st.session_state.zonif = zonificaciones[zonificaciones.zona == desc_zona]
        else:
            st.session_state.zonif = ""

        st.session_state.show_poly = col3.checkbox("Mostrar polígono", value=True)

        st.session_state.poly = poligonos[(poligonos.id == st.session_state.desc_poly)]

        if st.session_state.poly["tipo"].values[0] == "cuenca":
            desc_cuenca = col3.checkbox("OD en cuenca", value=False)
        else:
            desc_cuenca = False

        mtabla = col2.checkbox("Mostrar tabla", value=False)

        # Construye el diccionario de filtros actual
        current_filters = {
            "dia": None if desc_dia == "Todos" else desc_dia,
            "zona": None if desc_zona == "Todos" else desc_zona,
            "transferencia": (
                None
                if transf_list == "Todos"
                else (1 if transf_list == "Con transferencia" else 0)
            ),
            "modo_agregado": None if modos_list == "Todos" else modos_list,
            "rango_hora": None if rango_hora == "Todos" else rango_hora,
            "distancia_agregada": (
                None if distancia_agregada == "Todas" else distancia_agregada
            ),
            "coincidencias": None if desc_cuenca == False else True,
            "id_polygon": st.session_state.desc_poly,
            "desc_zonas_values1": (
                None if desc_zonas_values1 == "Todos" else desc_zonas_values1
            ),
            "desc_zonas_values2": (
                None if desc_zonas_values2 == "Todos" else desc_zonas_values2
            ),
        }
        current_options = {
            "desc_etapas": desc_etapas,
            "desc_viajes": desc_viajes,
            "desc_origenes": desc_origenes,
            "desc_destinos": desc_destinos,
            "desc_zonif": desc_zonif,
            "show_poly": st.session_state.show_poly,
            "desc_cuenca": desc_cuenca,
            "desc_et_vi": desc_et_vi,
            "mtabla": mtabla,
        }

        # Solo cargar datos si hay cambios en los filtros
        if hay_cambios_en_filtros(current_filters, st.session_state.last_filters):

            query = ""
            conditions = " AND ".join(
                f"{key} = '{value}'"
                for key, value in current_filters.items()
                if (value is not None)
                & (key != "desc_zonas_values1")
                & (key != "desc_zonas_values2")
            )
            if conditions:
                query += f" WHERE {conditions}"

            conditions_etapas1 = ""
            conditions_matrices1 = ""
            if desc_zonas_values1 != "Todos":
                conditions_etapas1 = f" AND (inicio_norm = '{desc_zonas_values1}' OR transfer1_norm = '{desc_zonas_values1}' OR transfer2_norm = '{desc_zonas_values1}' OR fin_norm = '{desc_zonas_values1}')"
                conditions_matrices1 = f" AND (inicio = '{desc_zonas_values1}' OR fin = '{desc_zonas_values1}')"

            conditions_etapas2 = ""
            conditions_matrices2 = ""
            if desc_zonas_values2 != "Todos":
                conditions_etapas2 = f" AND (inicio_norm = '{desc_zonas_values2}' OR transfer1_norm = '{desc_zonas_values2}' OR transfer2_norm = '{desc_zonas_values2}' OR fin_norm = '{desc_zonas_values2}')"
                conditions_matrices2 = f" AND (inicio = '{desc_zonas_values2}' OR fin = '{desc_zonas_values2}')"

            query_etapas = query + conditions_etapas1 + conditions_etapas2
            query_matrices = query + conditions_matrices1 + conditions_matrices2

            st.session_state.etapas_ = levanto_tabla_sql_local(
                "poly_etapas",
                tabla_tipo="dash",
                query=f"SELECT * FROM poly_etapas{query_etapas}",
            )

            st.session_state.matrices_ = levanto_tabla_sql_local(
                "poly_matrices",
                tabla_tipo="dash",
                query=f"SELECT * FROM poly_matrices{query_matrices}",
            )

            if len(st.session_state.etapas_) == 0:
                col2.write("No hay datos para mostrar")
            else:

                if desc_dia != "Todos":
                    st.session_state.socio_indicadores_ = socio_indicadores[
                        (socio_indicadores.dia == desc_dia)
                    ].copy()

                else:
                    st.session_state.socio_indicadores_ = socio_indicadores.copy()

                st.session_state.socio_indicadores_ = (
                    st.session_state.socio_indicadores_.groupby(
                        ["tabla", "genero_agregado", "tarifa_agregada", "Modo"],
                        as_index=False,
                    )[
                        [
                            "Distancia",
                            "Tiempo de viaje",
                            "Velocidad",
                            "Etapas promedio",
                            "Viajes promedio",
                            "Tiempo entre viajes",
                            "factor_expansion_linea",
                        ]
                    ]
                    .mean()
                    .round(2)
                )

                st.session_state.general_ = general.loc[
                    (general.id_polygon == st.session_state.desc_poly)
                    & (general.dia == desc_dia)
                ][["Tipo", "Indicador", "Valor"]].set_index("Tipo")

                st.session_state.modal_ = modal.loc[
                    (modal.id_polygon == st.session_state.desc_poly)
                    & (modal.dia == desc_dia)
                ][["Tipo", "Indicador", "Valor"]].set_index("Tipo")

                st.session_state.distancias_ = distancias.loc[
                    (distancias.id_polygon == st.session_state.desc_poly)
                    & (distancias.dia == desc_dia)
                ][["Tipo", "Indicador", "Valor"]].set_index("Tipo")

                if transf_list == "Todos":
                    st.session_state.desc_transfers = True
                else:
                    st.session_state.desc_transfers = False

                if modos_list == "Todos":
                    st.session_state.desc_modos = True
                else:
                    st.session_state.desc_modos = False

                if rango_hora == "Todos":
                    st.session_state.desc_horas = True
                else:
                    st.session_state.desc_horas = False

                if distancia_agregada == "Todas":
                    st.session_state.desc_distancia = True
                else:
                    st.session_state.desc_distancia = False

                st.session_state.agg_cols_etapas = [
                    "zona",
                    "inicio_norm",
                    "transfer1_norm",
                    "transfer2_norm",
                    "fin_norm",
                    "transferencia",
                    "modo_agregado",
                    "rango_hora",
                    "distancia_agregada",
                ]
                st.session_state.agg_cols_viajes = [
                    "zona",
                    "inicio_norm",
                    "fin_norm",
                    "transferencia",
                    "modo_agregado",
                    "rango_hora",
                    "distancia_agregada",
                ]

        if len(st.session_state.etapas_) == 0:
            col2.write("No hay datos para mostrar")
        else:
            if (
                not st.session_state.data_cargada
                or hay_cambios_en_filtros(
                    current_options, st.session_state.last_options
                )
                or hay_cambios_en_filtros(
                    current_filters, st.session_state.last_filters
                )
            ):

                # Actualiza los filtros en `session_state` para detectar cambios futuros
                st.session_state.last_filters = current_filters.copy()
                st.session_state.last_options = current_options.copy()
                st.session_state.data_cargada = True

                (
                    st.session_state.etapas,
                    st.session_state.viajes,
                    st.session_state.matriz,
                    st.session_state.origenes,
                    st.session_state.destinos,
                    st.session_state.transferencias,
                ) = create_data_folium(
                    st.session_state.etapas_,
                    st.session_state.matrices_,
                    agg_transferencias=st.session_state.desc_transfers,
                    agg_modo=st.session_state.desc_modos,
                    agg_hora=st.session_state.desc_horas,
                    agg_distancia=st.session_state.desc_distancia,
                    agg_cols_etapas=st.session_state.agg_cols_etapas,
                    agg_cols_viajes=st.session_state.agg_cols_viajes,
                    etapas_seleccionada=desc_etapas,
                    viajes_seleccionado=desc_viajes,
                    origenes_seleccionado=desc_origenes,
                    destinos_seleccionado=desc_destinos,
                    transferencias_seleccionado=False,
                )

                if (
                    (len(st.session_state.etapas) > 0)
                    | (len(st.session_state.viajes) > 0)
                    | (len(st.session_state.origenes) > 0)
                    | (len(st.session_state.destinos) > 0)
                    | (len(st.session_state.transferencias) > 0)
                    | (desc_zonif)
                ):

                    m = crear_mapa_poligonos(
                        df_viajes=st.session_state.viajes,
                        df_etapas=st.session_state.etapas,
                        poly=st.session_state.poly,
                        zonif=st.session_state.zonif,
                        origenes=st.session_state.origenes,
                        destinos=st.session_state.destinos,
                        var_fex="factor_expansion_linea",
                        cmap_viajes="Blues",
                        cmap_etapas="Greens",
                        map_title=st.session_state.desc_poly,
                        savefile="",
                        k_jenks=5,
                        show_poly=st.session_state.show_poly,
                    )
                    if m:
                        st.session_state.map = m

                    if st.session_state.map:
                        with col2:
                            folium_static(st.session_state.map, width=1000, height=800)
                            # output = st_folium(st.session_state.map, width=1000, height=800, key='m', returned_objects=["center"])

                        if mtabla:
                            col2.dataframe(
                                st.session_state.etapas_[
                                    [
                                        "inicio_norm",
                                        "transfer1_norm",
                                        "transfer2_norm",
                                        "fin_norm",
                                        "factor_expansion_linea",
                                    ]
                                ]
                            )

                    else:
                        col2.text("No hay datos suficientes para mostrar el mapa.")

with st.expander("Indicadores"):
    col1, col2, col3 = st.columns([2, 2, 2])

    if len(st.session_state.etapas_) > 0:
        col1.table(st.session_state.general_)
        col2.table(st.session_state.modal_)
        col3.table(st.session_state.distancias_)

with st.expander("Matrices"):

    col1, col2 = st.columns([1, 4])
    if len(st.session_state.matriz) > 0:

        # col2.table(st.session_state.matriz)

        tipo_matriz = col1.selectbox(
            "Variable",
            options=[
                "Viajes",
                "Distancia promedio (kms)",
                "Tiempo promedio (min)",
                "Velocidad promedio (km/h)",
            ],
        )

        normalize = False
        if tipo_matriz == "Viajes":
            var_matriz = "factor_expansion_linea"
            normalize = col1.checkbox("Normalizar", value=True)
        if tipo_matriz == "Distancia promedio (kms)":
            var_matriz = "distancia"
        if tipo_matriz == "Tiempo promedio (min)":
            var_matriz = "travel_time_min"
        if tipo_matriz == "Velocidad promedio (km/h)":
            var_matriz = "travel_speed"

        od_heatmap = pd.crosstab(
            index=st.session_state.matriz["Origen"],
            columns=st.session_state.matriz["Destino"],
            values=st.session_state.matriz[var_matriz],
            aggfunc="sum",
            normalize=normalize,
        )

        if normalize:
            od_heatmap = (od_heatmap * 100).round(2)
        else:
            od_heatmap = od_heatmap.round(0)

        od_heatmap = od_heatmap.reset_index()
        od_heatmap["Origen"] = od_heatmap["Origen"].str[4:]
        od_heatmap = od_heatmap.set_index("Origen")
        od_heatmap.columns = [i[4:] for i in od_heatmap.columns]

        fig = px.imshow(
            od_heatmap,
            text_auto=True,
            color_continuous_scale="Blues",
        )

        fig.update_coloraxes(showscale=False)

        if len(od_heatmap) <= 20:
            fig.update_layout(width=1000, height=1000)
        elif (len(od_heatmap) > 20) & (len(od_heatmap) <= 40):
            fig.update_layout(width=1000, height=1000)
        elif len(od_heatmap) > 40:
            fig.update_layout(width=1000, height=1000)

        col2.plotly_chart(fig)
    else:
        col2.text("No hay datos para mostrar")
