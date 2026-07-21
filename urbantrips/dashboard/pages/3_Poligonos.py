import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
import numpy as np
from dash_storage import normalize_vars
from dash_utils import (
    levanto_tabla_sql,
    levanto_tabla_sql_local,
    get_logo,
    create_data_folium,
    traigo_indicadores,
    traigo_lista_zonas,
    configurar_selector_dia,
    build_where_clauses,
    traer_dias_chains,
    traer_opciones_chains,
    condicion_zona_sql,
    condicion_poligono_sql,
    traer_etapas_matrices_sql,
    crear_mapa_lineas_deseo,
    bring_latlon,
)

from collections import OrderedDict


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
        "poly",
        "map_poligonos",
    ]

    variables_bool = ['resumen']

    for var in variables:
        if var not in st.session_state:
            st.session_state[var] = ""

    for var in variables_bool:
        if var not in st.session_state:
            st.session_state[var] = False

    dias_chains = traer_dias_chains()
    poligonos = levanto_tabla_sql_local("poligonos", "dash")

    if len(poligonos) == 0:
        st.info("No hay polígono cargado.")
        st.stop()

    if (len(poligonos) > 0) & (len(dias_chains) > 0):

        zonificaciones = levanto_tabla_sql("zonificaciones", "dash")
        socio_indicadores = levanto_tabla_sql("socio_indicadores")

        zonas_values = traigo_lista_zonas("poligonos")

        general, modal, distancias = traigo_indicadores("poligonos")

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

        st.session_state.etapas_lst = dias_chains
        desc_dia = col1.selectbox("Día", options=st.session_state.etapas_lst)

        st.session_state.desc_poly = col1.selectbox(
            "Polígono", options=poligonos.id.unique()
        )

        desc_zona = col1.selectbox(
            "Zonificación", options=zonas_values.zona.unique()
        )
        transf_list_all = ["Todos", "Con transferencia", "Sin transferencia"]
        transf_list = col1.selectbox("Transferencias", options=transf_list_all)

        modos_list_all = ["Todos"] + traer_opciones_chains("modo_agregado")
        modos_list = col1.selectbox("Modos", options=[text for text in modos_list_all])

        rango_hora_all = ["Todos"] + traer_opciones_chains("rango_hora")
        rango_hora = col1.selectbox(
            "Rango hora", options=[text for text in rango_hora_all]
        )

        distancia_all = ["Todas"] + traer_opciones_chains("distancia_agregada")
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

        tipo_visualizacion = col1.radio(
            "Tipo de visualización",
            options=["Líneas", "Arcos"],
            index=1,  # default explícito: arcos
            horizontal=True,
        )

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
        poly = st.session_state.poly

        tipo_poly = (
            poly["tipo"].iloc[0]
            if poly is not None and not poly.empty and "tipo" in poly.columns
            else "poligono"
        )

        if tipo_poly == "cuenca":
            opciones_filtro = ["Origen y Destino", "Origen o Destino", "OD y Transferencias"]
            default_idx = 0
        else:
            opciones_filtro = ["Origen o Destino", "OD y Transferencias"]
            default_idx = 0
        filtro_od = col3.selectbox(
            "Toca el polígono", options=opciones_filtro, index=default_idx
        )

        mtabla = col2.checkbox("Mostrar tabla", value=False)

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
            "filtro_od": filtro_od,
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
            "desc_et_vi": desc_et_vi,
            "tipo_visualizacion": tipo_visualizacion,
        }

        if hay_cambios_en_filtros(current_filters, st.session_state.last_filters):

            filters = {
                "modo_agregado": current_filters["modo_agregado"],
                "rango_hora": current_filters["rango_hora"],
                "transferencia": current_filters["transferencia"],
                "distancia_agregada": current_filters["distancia_agregada"],
            }
            where_extra = build_where_clauses(filters)

            condiciones = condicion_poligono_sql(
                st.session_state.desc_poly,
                filtro_od,
            )

            if desc_zonas_values1 != "Todos":
                condiciones += condicion_zona_sql(desc_zona, desc_zonas_values1)
            if desc_zonas_values2 != "Todos":
                condiciones += condicion_zona_sql(desc_zona, desc_zonas_values2)

            etapas_, matrices_ = traer_etapas_matrices_sql(
                desc_zona,
                zonificaciones,
                desc_dia,
                where_extra,
                condiciones,
                id_polygon=st.session_state.desc_poly,
                tipo_poligono=tipo_poly,
            )
            st.session_state.etapas_ = etapas_
            st.session_state.matrices_ = matrices_

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
                            "distance_od",
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

                st.session_state.desc_transfers = transf_list == "Todos"
                st.session_state.desc_modos = modos_list == "Todos"
                st.session_state.desc_horas = rango_hora == "Todos"
                st.session_state.desc_distancia = distancia_agregada == "Todas"

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
                    latlon = [
                        poly.geometry.representative_point().y.mean(),
                        poly.geometry.representative_point().x.mean(),
                    ]
                    zonif_para_mapa = (
                        st.session_state.zonif
                        if isinstance(st.session_state.zonif, pd.DataFrame)
                        else pd.DataFrame()
                    )
                    st.session_state.map_poligonos = crear_mapa_lineas_deseo(
                        df_viajes=st.session_state.viajes,
                        df_etapas=st.session_state.etapas,
                        zonif=zonif_para_mapa,
                        origenes=st.session_state.origenes,
                        destinos=st.session_state.destinos,
                        transferencias=st.session_state.transferencias,
                        var_fex="factor_expansion_linea",
                        k_jenks=5,
                        latlon=latlon,
                        tipo_visualizacion=tipo_visualizacion,
                        poly=st.session_state.poly,
                        show_poly=st.session_state.show_poly,
                    )
                else:
                    st.session_state.map_poligonos = None

            # Render siempre fuera del condicional de recarga
            _mapa = st.session_state.get("map_poligonos")
            if _mapa is not None and isinstance(_mapa, pdk.Deck):
                with col2:
                    st.pydeck_chart(
                        _mapa,
                        use_container_width=True,
                        height=800,
                    )

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
            var_matriz = "distance_od"
        if tipo_matriz == "Tiempo promedio (min)":
            var_matriz = "travel_time_min"
        if tipo_matriz == "Velocidad promedio (km/h)":
            var_matriz = "kmh_od"

        st.session_state.resumen = col1.checkbox("Principales OD", value=False)

        _poly_tipo = (
            st.session_state.poly["tipo"].values[0]
            if st.session_state.poly is not None
            and not st.session_state.poly.empty
            and "tipo" in st.session_state.poly.columns
            else "poligono"
        )
        if _poly_tipo == "cuenca":
            mat_cuenca = col1.checkbox("Matriz de la cuenca", value=False)
        else:
            mat_cuenca = False

        mmatriz_ = col1.checkbox("Mostrar tabla", value=False, key="mmatriz_")

        col1.write(
            'Cantidad total de viajes: ' +
            f"{int(st.session_state.matriz.factor_expansion_linea.sum()):,}"
        )

        matriz_tmp = st.session_state.matriz.copy()

        if mat_cuenca:
            matriz_tmp = matriz_tmp[
                (matriz_tmp.Origen.str.contains('cuenca'))
                & (matriz_tmp.Destino.str.contains('cuenca'))
            ]
            col1.write(
                'Cantidad total en cuenca: ' +
                f"{int(matriz_tmp.factor_expansion_linea.sum()):,}"
            )

        if not st.session_state.resumen:
            od_heatmap = pd.crosstab(
                index=matriz_tmp["Origen"],
                columns=matriz_tmp["Destino"],
                values=matriz_tmp[var_matriz],
                aggfunc="sum",
                normalize=normalize,
            )
        else:
            matriz_resumen = matriz_tmp.copy()
            matriz_resumen = matriz_resumen[matriz_resumen.resumen == 1]

            od_heatmap = pd.crosstab(
                index=matriz_resumen["Origen"],
                columns=matriz_resumen["Destino"],
                values=matriz_resumen[var_matriz],
                aggfunc="sum",
                normalize=normalize,
            )

            col1.write(
                f"Resumen: {matriz_resumen.porcentaje.sum().round(1)}% de viajes"
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

        if mmatriz_:
            col2.write(matriz_tmp)
    else:
        col2.text("No hay datos para mostrar")
