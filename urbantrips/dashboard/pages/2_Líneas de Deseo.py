import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import json
import mapclassify
import plotly.express as px
from shapely import wkt
from dash_utils import (
    levanto_tabla_sql,
    get_logo,
    create_data_folium,
    extract_hex_colors_from_cmap,
    normalize_vars,
    bring_latlon,
    traigo_lista_zonas,
    configurar_selector_dia,
    build_where_clauses,
    traer_dias_chains,
    traer_opciones_chains,
    traer_lineas_chains,
    condicion_zona_sql,
    traer_etapas_matrices_sql,
    traer_etapas_matrices_linea,
    viajes_con_origen_en_zona,
    etapas_por_linea_en_zona,
    viajes_entre_zonas_sql,
    etapas_entre_zonas_sql,
    crear_mapa_lineas_deseo,
)
from datetime import datetime


# Función para detectar cambios
def hay_cambios_en_filtros(current, last):
    return current != last


st.set_page_config(layout="wide")

alias_seleccionado = configurar_selector_dia()

logo = get_logo()
st.image(logo)

# Defaults for variables defined inside the data-gated block below, so the
# Matrices / Zonas expanders don't raise NameError when chains_norm is empty.
dia_seleccionado = "Todos"
filtro_seleccion1 = filtro_seleccion2 = "Todos"
zona_filtro_seleccion1 = zona_filtro_seleccion2 = None

with st.expander("Líneas de Deseo", expanded=True):

    col1, col2, col3 = st.columns([1, 7, 1])

    variables = [
        "last_filters",
        "last_options",
        "data_cargada",
        "lista_etapas",
        "matrices_all",
        "etapas_all",
        "where_extra",
        "etapas",
        "viajes",
        "matriz",
        "origenes",
        "destinos",
        "general",
        "modal",
        "distancia_seleccionada",
        "dia",
        "tipo_dia",
        "zona",
        "transferencia",
        "modo_agregado",
        "rango_hora_seleccionado",
        "distancia_agregada",
        "socio_indicadores_all",
        "alias_seleccionado",
        "zona_click",
        "map",
    ]

    variables_bool = [
        "etapas_seleccionada",
        "viajes_seleccionado",
        "origenes_seleccionado",
        "destinos_seleccionado",
        "transferencias_seleccionado",
        "resumen",
        "normalize",
        "mmatriz",
    ]
    # Inicializar todas las variables con None si no existen en session_state
    for var in variables:
        if var not in st.session_state:
            st.session_state[var] = ""

    for var in variables_bool:
        if var not in st.session_state:
            st.session_state[var] = False

    # Selector de días desde chains_norm
    st.session_state.lista_etapas = ["Todos"] + traer_dias_chains()

    if len(st.session_state.lista_etapas) > 1:
        zonificaciones = levanto_tabla_sql("zonificaciones", "dash")

        socio_indicadores = levanto_tabla_sql("socio_indicadores", "dash")
        if "Genero" not in socio_indicadores.columns:
            socio_indicadores["Genero"] = "-"
        if "Tarifa" not in socio_indicadores.columns:
            socio_indicadores["Tarifa"] = "-"

        # Selectores desde chains_norm + zonificaciones
        lista_zonas = traigo_lista_zonas("etapas")
        lista_modos = ["Todos"] + traer_opciones_chains("modo_agregado")
        lista_rango_hora = ["Todos"] + traer_opciones_chains("rango_hora")
        lista_distancia = ["Todas"] + traer_opciones_chains("distancia_agregada")
        lista_genero_agregado = ["Todos"] + traer_opciones_chains("genero_agregado")
        lista_tarifa_agregada = ["Todos"] + traer_opciones_chains("tarifa_agregada")
        lista_transfer = ["Todos", "Con transferencia", "Sin transferencia"]

        # Inicializar valores de `st.session_state` solo si no existen
        if "last_filters" not in st.session_state:
            st.session_state.last_filters = {
                "dia": "Todos",
                "zona": None,
                "transferencia": "Todos",
                "modo_agregado": "Todos",
                "rango_hora_seleccionado": "Todos",
                "distancia_seleccionada": "Todas",
                "genero_agregado_seleccionada": "Todas",
                "tarifa_agregada_seleccionada": "Todas",
                "filtro_seleccion1": "Todos",
                "filtro_seleccion2": "Todos",
                "zona_filtro_seleccion1": None,
                "zona_filtro_seleccion2": None,
                "alias_seleccionado": alias_seleccionado,
            }

        if "data_cargada" not in st.session_state:
            st.session_state.data_cargada = False

        valores_zonas = lista_zonas.zona.unique().tolist()

        # Opciones de los filtros en Streamlit
        dia_seleccionado = col1.selectbox(
            "Día", options=st.session_state.lista_etapas, index=1
        )
        zona_seleccionada = col1.selectbox("Zonificación", options=valores_zonas)

        transfer_seleccionado = col1.selectbox("Transferencias", options=lista_transfer)
        modo_seleccionado = col1.selectbox(
            "Modos", options=[text for text in lista_modos]
        )
        rango_hora_seleccionado = col1.selectbox(
            "Rango hora", options=[text for text in lista_rango_hora]
        )
        distancia_seleccionada = col1.selectbox("Distancia", options=lista_distancia)
        genero_agregado_seleccionado = col1.selectbox(
            "Género", options=[text for text in lista_genero_agregado]
        )
        tarifa_agregada_seleccionado = col1.selectbox(
            "Tarifa", options=[text for text in lista_tarifa_agregada]
        )
        vi_et_seleccion = col1.selectbox(
            "Datos de", options=["Etapas", "Viajes", "Ninguno"], index=1
        )
        st.session_state.viajes_seleccionado = vi_et_seleccion == "Viajes"
        st.session_state.etapas_seleccionada = vi_et_seleccion == "Etapas"

        tipo_visualizacion = col1.radio(
            "Tipo de visualización", options=["Líneas", "Arcos"], horizontal=True
        )

        col3.write("Agregar Filtros")
        index_zona = valores_zonas.index(zona_seleccionada)

        zona_filtro_seleccion1 = col3.selectbox(
            "Zona Filtro 1", options=valores_zonas, key="zon1", index=index_zona
        )
        lista_zonas_all = ["Todos"] + lista_zonas[
            lista_zonas.zona == zona_filtro_seleccion1
        ].Nombre.unique().tolist()
        filtro_seleccion1 = col3.selectbox(
            "Filtro 1", options=lista_zonas_all, key="filtro1"
        )
        zona_filtro_seleccion2 = col3.selectbox(
            "Zona Filtro 2", options=valores_zonas, key="zon2", index=index_zona
        )
        lista_zonas_all = ["Todos"] + lista_zonas[
            lista_zonas.zona == zona_filtro_seleccion2
        ].Nombre.unique().tolist()
        filtro_seleccion2 = col3.selectbox(
            "Filtro 2", options=lista_zonas_all, key="filtro2"
        )

        tipo_filtro = col3.selectbox(
            "Tipo de Filtro", options=["OD y Transferencias", "Solo OD"]
        )

        lineas_principales = col3.selectbox(
            "Mostrar líneas de deseo", options=["Solo principales", "Todas"]
        )

        nombre_linea_seleccionado = col3.selectbox(
            "Línea", options=["Todas"] + traer_lineas_chains()
        )

        col3.write("Mostrar:")
        st.session_state.origenes_seleccionado = col3.checkbox(
            ":blue[Origenes]", value=False
        )

        st.session_state.destinos_seleccionado = col3.checkbox(
            ":orange[Destinos]", value=False
        )

        st.session_state.transferencias_seleccionado = col3.checkbox(
            ":red[Transferencias]", value=False
        )

        zona_mostrar = col3.selectbox(
            "Mostrar zonificación",
            options=valores_zonas + ["Ninguna"],
            index=index_zona,
        )

        if zona_mostrar != "Ninguna":
            zonif = zonificaciones[zonificaciones.zona == zona_mostrar]
        else:
            zonif = ""

        mtabla = col2.checkbox("Mostrar tabla", value=False)

        # Leer zona clickeada desde el estado pydeck (disponible tras on_select="rerun")
        _pdk_state = st.session_state.get("map_pydeck", {})
        if isinstance(_pdk_state, dict):
            _sel_objs = _pdk_state.get("selection", {}).get("objects", {}).get("zonas", [])
            if _sel_objs:
                try:
                    _clicked_id = _sel_objs[0]["properties"]["id"]
                    if _clicked_id:
                        st.session_state.zona_click = _clicked_id
                except (KeyError, IndexError, TypeError):
                    pass

        # Construye el diccionario de filtros actual
        current_filters = {
            "dia": None if dia_seleccionado == "Todos" else dia_seleccionado,
            "zona": None if zona_seleccionada == "Todos" else zona_seleccionada,
            "transferencia": (
                None
                if transfer_seleccionado == "Todos"
                else (1 if transfer_seleccionado == "Con transferencia" else 0)
            ),
            "modo_agregado": (
                None if modo_seleccionado == "Todos" else modo_seleccionado
            ),
            "rango_hora": (
                None if rango_hora_seleccionado == "Todos" else rango_hora_seleccionado
            ),
            "distancia_agregada": (
                None if distancia_seleccionada == "Todas" else distancia_seleccionada
            ),
            "genero_agregado": (
                None
                if genero_agregado_seleccionado == "Todos"
                else genero_agregado_seleccionado
            ),
            "tarifa_agregada": (
                None
                if tarifa_agregada_seleccionado == "Todos"
                else tarifa_agregada_seleccionado
            ),
            "filtro_seleccion1": (
                None if filtro_seleccion1 == "Todos" else filtro_seleccion1
            ),
            "filtro_seleccion2": (
                None if filtro_seleccion2 == "Todos" else filtro_seleccion2
            ),
            "nombre_linea_seleccionado": (
                None
                if nombre_linea_seleccionado == "Todas"
                else nombre_linea_seleccionado
            ),
            "zona_filtro_seleccion1": zona_filtro_seleccion1,
            "zona_filtro_seleccion2": zona_filtro_seleccion2,
            "tipo_filtro": tipo_filtro,
            "lineas_principales": lineas_principales,
            "alias_seleccionado": alias_seleccionado,
        }

        current_options = {
            "etapas_seleccionada": st.session_state.etapas_seleccionada,
            "viajes_seleccionado": st.session_state.viajes_seleccionado,
            "origenes_seleccionado": st.session_state.origenes_seleccionado,
            "destinos_seleccionado": st.session_state.destinos_seleccionado,
            "vi_et_seleccion": vi_et_seleccion,
            "transferencias_seleccionado": st.session_state.transferencias_seleccionado,
            "zona_mostrar": zona_mostrar,
            "mtabla": mtabla,
            "resumen": st.session_state.resumen,
            "normalize": st.session_state.normalize,
            "mmatriz": st.session_state.mmatriz,
            "tipo_visualizacion": tipo_visualizacion,
            "zona_click": st.session_state.get("zona_click"),
        }

        # Solo cargar datos si hay cambios en los filtros
        if hay_cambios_en_filtros(current_filters, st.session_state.last_filters):

            # WHERE opcional sobre chains_norm con los filtros del usuario
            filters = {
                "modo_agregado": current_filters["modo_agregado"],
                "rango_hora": current_filters["rango_hora"],
                "transferencia": current_filters["transferencia"],
                "distancia_agregada": current_filters["distancia_agregada"],
                "genero_agregado": current_filters["genero_agregado"],
                "tarifa_agregada": current_filters["tarifa_agregada"],
            }
            where_extra = build_where_clauses(filters)
            st.session_state.where_extra = where_extra

            # Filtros adicionales por zona (sobre cualquier zonificación),
            # resueltos como condiciones SQL sobre chains_norm
            condiciones = ""
            if filtro_seleccion1 != "Todos":
                condiciones += condicion_zona_sql(
                    zona_filtro_seleccion1, filtro_seleccion1, tipo_filtro
                )
            if filtro_seleccion2 != "Todos":
                condiciones += condicion_zona_sql(
                    zona_filtro_seleccion2, filtro_seleccion2, tipo_filtro
                )

            if nombre_linea_seleccionado != "Todas":
                # nivel etapa de la línea seleccionada (semántica del viejo
                # traigo_viajes_linea: pares OD de las etapas de esa línea)
                etapas_all, matrices_all = traer_etapas_matrices_linea(
                    zona_seleccionada,
                    zonificaciones,
                    nombre_linea_seleccionado,
                    dia_seleccionado,
                    where_extra,
                    condiciones,
                )
            else:
                # join + GROUP BY dentro de la base dash (una sola conexión)
                etapas_all, matrices_all = traer_etapas_matrices_sql(
                    zona_seleccionada,
                    zonificaciones,
                    dia_seleccionado,
                    where_extra,
                    condiciones,
                )

            st.session_state.etapas_all = etapas_all
            st.session_state.matrices_all = matrices_all

            if len(st.session_state.matrices_all) != 0:

                if dia_seleccionado != "Todos":
                    st.session_state.socio_indicadores_all = socio_indicadores[
                        (socio_indicadores.dia == dia_seleccionado)
                    ].copy()

                else:
                    st.session_state.socio_indicadores_all = socio_indicadores.copy()

                st.session_state.socio_indicadores_all = (
                    st.session_state.socio_indicadores_all.groupby(
                        ["tabla", "Genero", "Tarifa", "Modo"], as_index=False
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

                st.session_state.desc_transfers = transfer_seleccionado == "Todos"
                st.session_state.desc_modos = modo_seleccionado == "Todos"
                st.session_state.desc_horas = rango_hora_seleccionado == "Todos"
                st.session_state.desc_distancia = distancia_seleccionada == "Todas"
                st.session_state.desc_genero_agregado = (
                    genero_agregado_seleccionado == "Todos"
                )
                st.session_state.desc_tarifa_agregada = (
                    tarifa_agregada_seleccionado == "Todos"
                )

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
                    "genero_agregado",
                    "tarifa_agregada",
                ]
                st.session_state.agg_cols_viajes = [
                    "zona",
                    "inicio_norm",
                    "fin_norm",
                    "transferencia",
                    "modo_agregado",
                    "rango_hora",
                    "genero_agregado",
                    "tarifa_agregada",
                    "distancia_agregada",
                ]

        if len(st.session_state.etapas_all) == 0:
            st.session_state.last_filters = current_filters.copy()
            col2.write("No hay datos para mostrar")
        else:

            if lineas_principales == 'Todas':
                mostrar_lineas_principales = False
            else:
                mostrar_lineas_principales = True

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
                    st.session_state.etapas_all.copy(),
                    st.session_state.matrices_all.copy(),
                    agg_transferencias=st.session_state.desc_transfers,
                    agg_modo=st.session_state.desc_modos,
                    agg_hora=st.session_state.desc_horas,
                    agg_distancia=st.session_state.desc_distancia,
                    agg_genero_agregado=st.session_state.desc_genero_agregado,
                    agg_tarifa_agregada=st.session_state.desc_tarifa_agregada,
                    agg_cols_etapas=st.session_state.agg_cols_etapas,
                    agg_cols_viajes=st.session_state.agg_cols_viajes,
                    etapas_seleccionada=st.session_state.etapas_seleccionada,
                    viajes_seleccionado=st.session_state.viajes_seleccionado,
                    origenes_seleccionado=st.session_state.origenes_seleccionado,
                    destinos_seleccionado=st.session_state.destinos_seleccionado,
                    transferencias_seleccionado=st.session_state.transferencias_seleccionado,
                    mostrar_lineas_principales=mostrar_lineas_principales,
                )

                if (
                    (len(st.session_state.etapas) > 0)
                    | (len(st.session_state.viajes) > 0)
                    | (len(st.session_state.origenes) > 0)
                    | (len(st.session_state.destinos) > 0)
                    | (len(st.session_state.transferencias) > 0)
                ) | (len(zona_mostrar) > 0):

                    latlon = bring_latlon()

                    # Filtrar por zona clickeada (solo para visualización, no modifica session_state)
                    zona_click = st.session_state.get("zona_click")
                    etapas_para_mapa = st.session_state.etapas
                    viajes_para_mapa = st.session_state.viajes
                    if zona_click:
                        if len(etapas_para_mapa) > 0 and "inicio_norm" in etapas_para_mapa.columns:
                            etapas_para_mapa = etapas_para_mapa[
                                (etapas_para_mapa["inicio_norm"] == zona_click)
                                | (etapas_para_mapa["fin_norm"] == zona_click)
                            ]
                        if len(viajes_para_mapa) > 0 and "inicio_norm" in viajes_para_mapa.columns:
                            viajes_para_mapa = viajes_para_mapa[
                                (viajes_para_mapa["inicio_norm"] == zona_click)
                                | (viajes_para_mapa["fin_norm"] == zona_click)
                            ]

                    st.session_state.map = crear_mapa_lineas_deseo(
                        df_viajes=viajes_para_mapa,
                        df_etapas=etapas_para_mapa,
                        zonif=zonif,
                        origenes=st.session_state.origenes,
                        destinos=st.session_state.destinos,
                        transferencias=st.session_state.transferencias,
                        var_fex="factor_expansion_linea",
                        cmap_viajes="viridis_r",
                        cmap_etapas="magma_r",
                        map_title="Líneas de Deseo",
                        savefile="",
                        k_jenks=5,
                        latlon=latlon,
                        tipo_visualizacion=tipo_visualizacion,
                    )
                else:
                    st.session_state.map = None

            # Renderizar siempre el mapa actual (fuera del condicional de datos)
            if st.session_state.get("map"):
                with col2:
                    zona_click = st.session_state.get("zona_click")
                    if zona_click:
                        col_info, col_btn = col2.columns([4, 1])
                        col_info.info(f"Filtro activo: zona **{zona_click}**")
                        if col_btn.button("✕ Limpiar", key="btn_clear_zona"):
                            st.session_state.zona_click = None
                            st.rerun()

                    st.pydeck_chart(
                        st.session_state.map,
                        key="map_pydeck",
                        on_select="rerun",
                        use_container_width=True,
                        height=800,
                    )

                if mtabla:
                    if len(st.session_state.etapas) > 0:
                        col2.write("Etapas")
                        col2.write(st.session_state.etapas)
                    if len(st.session_state.viajes) > 0:
                        col2.write("Viajes")
                        col2.write(st.session_state.viajes)
            else:
                col2.text("No hay datos suficientes para mostrar el mapa.")
    else:
        st.write("No hay datos en chains_norm para mostrar")


with st.expander("Matrices"):

    col1, col2 = st.columns([1, 4])

    if len(st.session_state.matriz) > 0:

        st.session_state.matriz["Origen"] = st.session_state.matriz["Origen"].astype(
            str
        )
        st.session_state.matriz["Destino"] = st.session_state.matriz["Destino"].astype(
            str
        )

        tipo_matriz = col1.selectbox(
            "Variable",
            options=[
                "Viajes",
                "Distancia promedio (kms)",
                "Tiempo promedio (min)",
                "Velocidad promedio (km/h)",
            ],
        )

        st.session_state.normalize = False
        if tipo_matriz == "Viajes":
            var_matriz = "factor_expansion_linea"
            st.session_state.normalize = col1.checkbox("Normalizar", value=True)
            if st.session_state.normalize:
                var_matriz = "porcentaje"

        st.session_state.resumen = col1.checkbox("Principales OD", value=False)

        mmatriz_ = col1.checkbox("Mostrar tabla", value=False, key="mmatriz_")
        st.session_state.mmatriz = mmatriz_
        col1.write(f"Día: {dia_seleccionado}")

        col1.write(
                'Cantidad total de viajes: ' +
                f"{int(st.session_state.matriz.factor_expansion_linea.sum()):,}"
            )

        col1.write(f"Transferencias: {transfer_seleccionado}")
        col1.write(f"Modos: {modo_seleccionado}")
        col1.write(f"Rango hora: {rango_hora_seleccionado}")
        col1.write(f"Distancias: {distancia_seleccionada}")
        col1.write(f"Genero: {genero_agregado_seleccionado}")
        col1.write(f"Tarifa: {tarifa_agregada_seleccionado}")

        if nombre_linea_seleccionado != "Todas":
            col1.write(f"Línea: {nombre_linea_seleccionado}")

        if tipo_matriz == "Distancia promedio (kms)":
            var_matriz = "distance_od"
        if tipo_matriz == "Tiempo promedio (min)":
            var_matriz = "travel_time_min"
        if tipo_matriz == "Velocidad promedio (km/h)":
            var_matriz = "kmh_od"

        if not st.session_state.resumen:
            od_heatmap = pd.crosstab(
                index=st.session_state.matriz["Origen"],
                columns=st.session_state.matriz["Destino"],
                values=st.session_state.matriz[var_matriz],
                aggfunc="sum",
                normalize=False,
            )
        else:
            matriz_resumen = st.session_state.matriz.copy()
            matriz_resumen = matriz_resumen[matriz_resumen.resumen == 1]

            od_heatmap = pd.crosstab(
                index=matriz_resumen["Origen"],
                columns=matriz_resumen["Destino"],
                values=matriz_resumen[var_matriz],
                aggfunc="sum",
                normalize=False,
            )

            col1.write(
                f"Resumen: {matriz_resumen.porcentaje.sum().round(1)}% de viajes"
            )

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

        if len(od_heatmap) <= 30:
            fig.update_layout(width=1100, height=1100, font=dict(size=10))
            fig.update_traces(textfont=dict(size=12))
        elif (len(od_heatmap) > 30) & (len(od_heatmap) <= 40):
            fig.update_layout(width=1000, height=1000)
        elif len(od_heatmap) > 40:
            fig.update_layout(width=1000, height=1000)

        col2.plotly_chart(fig)
        if st.session_state.mmatriz:
            col2.write(st.session_state.matriz)

    else:
        col2.text("No hay datos para mostrar")


with st.expander("Zonas", expanded=False):
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])

    where_extra = st.session_state.get("where_extra", "")

    # Igual que urbantrips_viejo: Etapas = etapas (por línea) con origen de
    # la etapa en la zona; Viajes = viajes (por modo) con origen del viaje
    # en la zona (ambos direccionales).
    for filtro_zona, zonif_zona, col_e, col_v in [
        (filtro_seleccion1, zona_filtro_seleccion1, col2, col3),
        (filtro_seleccion2, zona_filtro_seleccion2, col4, col5),
    ]:
        if filtro_zona == "Todos":
            continue

        etapas_z = etapas_por_linea_en_zona(
            dia_seleccionado, where_extra, zonif_zona, filtro_zona
        )
        viajes_z = viajes_con_origen_en_zona(
            dia_seleccionado, where_extra, zonif_zona, filtro_zona
        )

        col_e.markdown(
            f"""
            <h3 style='font-size:22px;'>{filtro_zona}</h3>
            """,
            unsafe_allow_html=True,
        )

        if len(etapas_z) > 0:
            etapas_z = etapas_z.rename(
                columns={"nombre_linea": "Línea", "factor_expansion_linea": "Etapas"}
            )
            total_e = pd.DataFrame(
                {"Línea": ["Total"], "Etapas": [etapas_z["Etapas"].sum()]}
            )
            etapas_z = pd.concat([etapas_z, total_e], ignore_index=True)
            etapas_z["Etapas"] = etapas_z["Etapas"].round()
            col_e.write("Etapas (origen de la etapa en la zona)")
            col_e.dataframe(etapas_z.set_index("Línea"), height=400, width=400)

        if len(viajes_z) > 0:
            viajes_z = viajes_z.rename(
                columns={"modo_agregado": "Modo", "factor_expansion_linea": "Viajes"}
            )
            total_v = pd.DataFrame(
                {"Modo": ["Total"], "Viajes": [viajes_z["Viajes"].sum()]}
            )
            viajes_z = pd.concat([viajes_z, total_v], ignore_index=True)
            viajes_z["Viajes"] = viajes_z["Viajes"].round()
            col_v.markdown(
                """
                <h3 style='font-size:22px;'></h3>
                """,
                unsafe_allow_html=True,
            )
            col_v.write("Viajes (origen del viaje en la zona)")
            col_v.dataframe(viajes_z.set_index("Modo"), height=400, width=300)


with st.expander("Viajes entre zonas", expanded=True):
    col1, col2, col3 = st.columns([1, 2, 4])

    where_extra = st.session_state.get("where_extra", "")

    if filtro_seleccion1 != "Todos" and filtro_seleccion2 != "Todos":

        col1.write(f"Día: {dia_seleccionado}")
        col1.write(f"Zona 1: {filtro_seleccion1}")
        col1.write(f"Zona 2: {filtro_seleccion2}")

        entre_zonas = viajes_entre_zonas_sql(
            dia_seleccionado,
            where_extra,
            zona_filtro_seleccion1,
            filtro_seleccion1,
            zona_filtro_seleccion2,
            filtro_seleccion2,
        )

        etapas_ez = etapas_entre_zonas_sql(
            dia_seleccionado,
            where_extra,
            zona_filtro_seleccion1,
            filtro_seleccion1,
            zona_filtro_seleccion2,
            filtro_seleccion2,
        )

        if len(entre_zonas) > 0:
            zonasod_v = (
                entre_zonas.groupby(["Zona_1", "Zona_2"], as_index=False)
                .factor_expansion_linea.sum()
                .rename(columns={"factor_expansion_linea": "Viajes"})
            )
            zonasod_v["Zonas"] = zonasod_v["Zona_1"] + " - " + zonasod_v["Zona_2"]
            zonasod_v = zonasod_v[["Zonas", "Viajes"]]
            zonasod_v["Viajes"] = zonasod_v["Viajes"].apply(lambda x: f"{int(x):,}")

            modos_v = (
                entre_zonas.groupby(["modo_agregado"], as_index=False)
                .factor_expansion_linea.sum()
                .rename(
                    columns={"factor_expansion_linea": "Viajes", "modo_agregado": "Modo"}
                )
            )
            total_row = pd.DataFrame(
                {"Modo": ["Total"], "Viajes": [modos_v["Viajes"].sum()]}
            )
            modos_v = pd.concat([modos_v, total_row], ignore_index=True)
            modos_v["Viajes"] = modos_v["Viajes"].apply(lambda x: f"{int(x):,}")

            transferencias = (
                entre_zonas[entre_zonas.seq_lineas != ""]
                .groupby(["modo_agregado", "seq_lineas"], as_index=False)
                .factor_expansion_linea.sum()
                .rename(
                    columns={
                        "factor_expansion_linea": "Viajes",
                        "modo_agregado": "Modo",
                        "seq_lineas": "Líneas",
                    }
                )
                .sort_values("Viajes", ascending=False)
            )
            if len(transferencias) > 0:
                total_rowt = pd.DataFrame(
                    {
                        "Modo": ["Total"],
                        "Líneas": ["-"],
                        "Viajes": [transferencias["Viajes"].sum()],
                    }
                )
                transferencias = pd.concat(
                    [transferencias, total_rowt], ignore_index=True
                )
                transferencias["Viajes"] = transferencias["Viajes"].apply(
                    lambda x: f"{int(x):,}"
                )

            col2.write("Etapas")
            if len(etapas_ez) > 0:
                zonasod_e = (
                    etapas_ez.groupby(["Zona_1", "Zona_2"], as_index=False)
                    .factor_expansion_linea.sum()
                    .rename(columns={"factor_expansion_linea": "Etapas"})
                )
                zonasod_e["Zonas"] = (
                    zonasod_e["Zona_1"] + " - " + zonasod_e["Zona_2"]
                )
                zonasod_e = zonasod_e[["Zonas", "Etapas"]]
                zonasod_e["Etapas"] = zonasod_e["Etapas"].apply(
                    lambda x: f"{int(x):,}"
                )
                col2.dataframe(zonasod_e.set_index("Zonas"), height=100, width=300)
            else:
                col2.write("No hay datos para mostrar")

            col2.write("Viajes")
            col2.dataframe(zonasod_v.set_index("Zonas"), height=100, width=300)

            col2.write("Modal")
            col2.dataframe(modos_v.set_index("Modo"), height=300, width=300)

            col3.write("Viajes por líneas")
            if len(transferencias) > 0:
                col3.dataframe(transferencias.set_index("Modo"), height=700, width=800)
            else:
                col3.write("No hay datos para mostrar")
        else:
            col2.write("No hay datos para mostrar")
