import numpy as np
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import mapclassify
import plotly.express as px
from folium import Figure
from dash_utils import (
    levanto_tabla_sql,
    get_logo,
    create_linestring_od,
    extract_hex_colors_from_cmap,
    configurar_selector_dia,
)

from itertools import combinations
import squarify
from matplotlib_venn import venn3
import matplotlib.pyplot as plt


# Function to create activity combinations as tuples
def get_activity_tuple(cols_dummies, selected_cols_dummies):
    return tuple(
        1 if activity in selected_cols_dummies else 0 for activity in cols_dummies
    )


# Function to calculate subset sizes
def get_activity_combination_number(df, cols_dummies, activity_combination):
    activity_str_filter = [
        f"{a} > 0" if a in activity_combination else f"{a} == 0" for a in cols_dummies
    ]
    activity_str_filter = " & ".join(activity_str_filter)
    return len(df.query(activity_str_filter))


# Generate example data
def generate_example_data(num_rows=100):
    np.random.seed(42)  # For reproducibility
    data = {
        "train": np.random.randint(
            0, 2, size=num_rows
        ),  # Binary: 0 (not used), 1 (used)
        "subway": np.random.randint(0, 2, size=num_rows),
        "bus": np.random.randint(0, 2, size=num_rows),
    }
    return pd.DataFrame(data)


# Generate subset sizes dictionary
def calculate_subset_sizes(df, cols_dummies):
    list_of_tuples = [
        item
        for sublist in [
            list(combinations(cols_dummies, i)) for i in range(1, len(cols_dummies) + 1)
        ]
        for item in sublist
    ]
    return {
        get_activity_tuple(cols_dummies, combo): get_activity_combination_number(
            df, cols_dummies, combo
        )
        for combo in list_of_tuples
    }


# Extract subset sizes for Venn diagram
def get_venn_subsets(subset_sizes):
    return (
        subset_sizes.get((1, 0, 0), 0),  # Only 'train'
        subset_sizes.get((0, 1, 0), 0),  # Only 'subway'
        subset_sizes.get((1, 1, 0), 0),  # 'train' and 'subway'
        subset_sizes.get((0, 0, 1), 0),  # Only 'bus'
        subset_sizes.get((1, 0, 1), 0),  # 'train' and 'bus'
        subset_sizes.get((0, 1, 1), 0),  # 'subway' and 'bus'
        subset_sizes.get((1, 1, 1), 0),  # 'train', 'subway', and 'bus'
    )


# Plot Venn diagram
def plot_venn_diagram(etapas_modos):

    cols_dummies = [
        x
        for x in etapas_modos.columns.tolist()
        if x
        not in [
            "dia",
            "mes",
            "tipo_dia",
            "genero_agregado",
            "Modos",
            "factor_expansion_linea",
        ]
    ]

    cols_tmp = []
    for i in cols_dummies:
        etapas_modos[f"{i}_tmp"] = (
            etapas_modos[i] * etapas_modos["factor_expansion_linea"]
        )
        cols_tmp += [f"{i}_tmp"]
    cols_tmp = (
        etapas_modos[cols_tmp]
        .sum()
        .reset_index()
        .rename(columns={"index": "modo", 0: "viajes"})
        .sort_values("viajes", ascending=False)
        .round()
        .head(3)
        .modo.values.tolist()
    )
    cols_dummies_first3 = [modo.replace("_tmp", "") for modo in cols_tmp]

    # Calcular porcentajes
    absolute_values = calculate_weighted_values(
        etapas_modos,
        cols_dummies,
        weight_column="factor_expansion_linea",
        as_percentage=False,
    )
    percentage_values = calculate_weighted_values(
        etapas_modos,
        cols_dummies,
        weight_column="factor_expansion_linea",
        as_percentage=True,
    )
    absolute_values_first3 = calculate_weighted_values(
        etapas_modos,
        cols_dummies_first3,
        weight_column="factor_expansion_linea",
        as_percentage=False,
    )
    percentage_values_first3 = calculate_weighted_values(
        etapas_modos,
        cols_dummies_first3,
        weight_column="factor_expansion_linea",
        as_percentage=True,
    )

    modal_etapas = pd.DataFrame(
        list(absolute_values.items()), columns=["Modes", "Cantidad"]
    ).round(0)

    modal_etapas[cols_dummies] = pd.DataFrame(
        modal_etapas["Modes"].tolist(), index=modal_etapas.index
    )
    modal_etapas["Modo"] = ""
    for i in cols_dummies:
        modal_etapas.loc[modal_etapas[i] >= 1, "Modo"] += i + "-"
    modal_etapas.loc[modal_etapas.Modo.str[-1:] == "-", "Modo"] = modal_etapas.loc[
        modal_etapas.Modo.str[-1:] == "-"
    ].Modo.str[:-1]
    modal_etapas = modal_etapas[["Modo", "Cantidad"]]
    modal_etapas["Cantidad"] = modal_etapas["Cantidad"].astype(int)
    modal_etapas["%"] = (
        modal_etapas["Cantidad"] / modal_etapas["Cantidad"].sum() * 100
    ).round(1)
    modal_etapas = modal_etapas.sort_values("Cantidad", ascending=False)

    venn_subsets = get_venn_subsets(percentage_values_first3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot: Venn3
    venn3(
        subsets=venn_subsets,
        set_labels=[activity.capitalize() for activity in cols_dummies_first3],
        ax=ax1,
    )
    ax1.set_title("Partición modal (%)")

    # Identificar multimodal (más de un modo utilizado)
    etapas_modos["Multimodal"] = (
        etapas_modos[cols_dummies].gt(0).sum(axis=1) > 1
    ).astype(int)

    # Identificar multietapa (más de una etapa en al menos un modo)
    etapas_modos["Multietapa"] = (etapas_modos[cols_dummies].gt(1).any(axis=1)).astype(
        int
    )

    for i in cols_dummies:
        etapas_modos.loc[etapas_modos.Multimodal == 1, i] = 0
        etapas_modos.loc[etapas_modos.Multietapa == 1, i] = 0
        # etapas_modos.loc[etapas_modos[i]>=1, i] = 1

    etapas_modos.loc[
        (etapas_modos.Multietapa > 0) & (etapas_modos.Multimodal > 0), "Multietapa"
    ] = 0

    cols_dummies = cols_dummies + ["Multimodal", "Multietapa"]

    etapas_modos["Modos"] = etapas_modos[cols_dummies].idxmax(axis=1)

    v = (
        etapas_modos.groupby("Modos", as_index=False)
        .factor_expansion_linea.sum()
        .round()
    )
    v["p"] = (v.factor_expansion_linea / v.factor_expansion_linea.sum() * 100).round(1)
    v["m"] = v.Modos + "\n(" + v.p.astype(str) + "%)"
    v.loc[v.p < 3, "m"] = v.loc[v.p < 3, "m"].str.replace("\n", " ")

    values_data = v.p.values.tolist()
    values_names = v.m.values.tolist()

    fixed_palette = [
        "#AED6F1",
        "#F9E79F",
        "#ABEBC6",
        "#F5B7B1",
        "#D2B4DE",
        "#FAD7A0",
        "#85C1E9",
        "#A3E4D7",
        "#F7DC6F",
        "#F0B27A",
        "#F8C471",
        "#D7BDE2",
        "#A2D9CE",
        "#FDEBD0",
        "#D5F5E3",
        "#F9E79F",
        "#82E0AA",
        "#BB8FCE",
        "#EDBB99",
        "#A9CCE3",
    ]

    # Right subplot: Squarify treemap

    # Filtrar valores cero
    filtered_values = []
    filtered_labels = []

    for value, label in zip(values_data, values_names):
        if value > 0:
            filtered_values.append(value)
            filtered_labels.append(label)

    # Verificar que haya datos antes de graficar
    if len(filtered_values) == 0:
        st.warning(
            "No hay datos suficientes para mostrar el treemap de combinaciones de modos."
        )
    else:

        total = sum(filtered_values)
        normalized_values = [(v / total) * 100 for v in filtered_values]

        squarify.plot(
            sizes=filtered_values,
            label=filtered_labels,
            color=fixed_palette[: len(filtered_values)],
            text_kwargs={"fontsize": 10, "color": "black"},
            ax=ax2,
        )
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

    modal_viajes = (
        etapas_modos.groupby("Modos", as_index=False)
        .factor_expansion_linea.sum()
        .round(0)
        .rename(columns={"factor_expansion_linea": "Cantidad"})
    )
    modal_viajes["%"] = (
        modal_viajes["Cantidad"] / modal_viajes["Cantidad"].sum() * 100
    ).round(1)
    modal_viajes = modal_viajes.sort_values("Cantidad", ascending=False)
    modal_viajes = modal_viajes.rename(columns={"Modos": "Modo"})

    return fig, modal_etapas, modal_viajes


# Función para calcular los porcentajes o valores absolutos ponderados
from itertools import combinations


def calculate_weighted_values(df, cols_dummies, weight_column, as_percentage=True):
    # Calcular el total ponderado
    total_weight = df[weight_column].sum()

    # Crear el diccionario para cada combinación de actividades
    subset_sizes = {}
    for combo in [
        list(c)
        for i in range(1, len(cols_dummies) + 1)
        for c in combinations(cols_dummies, i)
    ]:
        activity_str_filter = [
            f"{a} > 0" if a in combo else f"{a} == 0" for a in cols_dummies
        ]
        query_str = " & ".join(activity_str_filter)
        subset_weight = df.query(query_str)[weight_column].sum()

        # Solo guardar combinaciones con valores > 0
        if subset_weight > 0:
            subset_sizes[tuple(1 if a in combo else 0 for a in cols_dummies)] = (
                subset_weight
            )

    # Convertir a porcentajes si as_percentage es True
    if as_percentage:
        subset_sizes = {
            key: round((value / total_weight) * 100, 1)
            for key, value in subset_sizes.items()
        }

    return subset_sizes


def traigo_socio_indicadores(socio_indicadores):
    totals = None
    totals_porc = 0
    avg_distances = 0
    avg_times = 0
    avg_velocity = 0
    modos_genero_abs = 0
    modos_genero_porc = 0
    modos_tarifa_abs = 0
    modos_tarifa_porc = 0
    avg_viajes = 0
    avg_etapas = 0
    avg_tiempo_entre_viajes = 0

    if len(socio_indicadores) > 0:

        df = socio_indicadores[
            socio_indicadores.tabla == "viajes-genero_agregado-tarifa_agregada"
        ].copy()
        totals = (
            pd.crosstab(
                values=df.factor_expansion_linea,
                columns=df.genero_agregado,
                index=df.tarifa_agregada,
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
                columns=df.genero_agregado,
                index=df.tarifa_agregada,
                aggfunc="sum",
                margins=True,
                margins_name="Total",
                normalize=True,
            )
            * 100
        ).round(2)

        modos = socio_indicadores[
            socio_indicadores.tabla == "etapas-genero_agregado-modo"
        ].copy()
        modos_genero_abs = (
            pd.crosstab(
                values=modos.factor_expansion_linea,
                index=[modos.genero_agregado],
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
                index=modos.genero_agregado,
                columns=modos.Modo,
                aggfunc="sum",
                normalize=True,
                margins=True,
                margins_name="Total",
            )
            * 100
        ).round(2)

        modos = socio_indicadores[
            socio_indicadores.tabla == "etapas-tarifa_agregada-modo"
        ].copy()
        modos_tarifa_abs = (
            pd.crosstab(
                values=modos.factor_expansion_linea,
                index=[modos.tarifa_agregada],
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
                index=modos.tarifa_agregada,
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
                columns=df.genero_agregado,
                index=df.tarifa_agregada,
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
                columns=df.genero_agregado,
                index=df.tarifa_agregada,
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
                columns=df.genero_agregado,
                index=df.tarifa_agregada,
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
                columns=df.genero_agregado,
                index=df.tarifa_agregada,
                margins=True,
                margins_name="Total",
                aggfunc=lambda x: (x * df.loc[x.index, "factor_expansion_linea"]).sum()
                / df.loc[x.index, "factor_expansion_linea"].sum(),
            )
            .round(2)
            .fillna("")
        )
        user = socio_indicadores[
            socio_indicadores.tabla == "usuario-genero_agregado-tarifa_agregada"
        ].copy()
        avg_viajes = (
            pd.crosstab(
                values=user["Viajes promedio"],
                index=[user.tarifa_agregada],
                columns=user.genero_agregado,
                margins=True,
                margins_name="Total",
                aggfunc=lambda x: (
                    x * user.loc[x.index, "factor_expansion_linea"]
                ).sum()
                / user.loc[x.index, "factor_expansion_linea"].sum(),
            )
            .round(2)
            .fillna("")
        )

        avg_tiempo_entre_viajes = (
            pd.crosstab(
                values=df["Tiempo entre viajes"],
                columns=df.genero_agregado,
                index=df.tarifa_agregada,
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


def crear_mapa_folium(df_agg, cmap, var_fex, savefile="", k_jenks=5):

    bins = [df_agg[var_fex].min() - 1] + mapclassify.FisherJenks(
        df_agg[var_fex], k=k_jenks
    ).bins.tolist()
    range_bins = range(0, len(bins) - 1)
    bins_labels = [f"{int(bins[n])} a {int(bins[n+1])} viajes" for n in range_bins]
    df_agg["cuts"] = pd.cut(df_agg[var_fex], bins=bins, labels=bins_labels)

    fig = Figure(width=800, height=800)
    m = folium.Map(
        location=[df_agg.lat_o.mean(), df_agg.lon_o.mean()],
        zoom_start=9,
        tiles="cartodbpositron",
    )

    title_html = """
    <h3 align="center" style="font-size:20px"><b>Your map title</b></h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    line_w = 0.5

    colors = extract_hex_colors_from_cmap(cmap=cmap, n=k_jenks)

    n = 0
    for i in bins_labels:

        df_agg[df_agg.cuts == i].explore(
            m=m,
            color=colors[n],
            style_kwds={"fillOpacity": 0.1, "weight": line_w},
            name=i,
            tooltip=False,
        )
        n += 1
        line_w += 3

    folium.LayerControl(name="xx").add_to(m)

    fig.add_child(m)

    return fig


st.set_page_config(layout="wide")

logo = get_logo()
st.image(logo)

alias_seleccionado = configurar_selector_dia()

with st.expander("Partición modal", True):

    col1, col2, col3, col4 = st.columns([1, 5, 1.5, 1.5])
    particion_modal = levanto_tabla_sql("datos_particion_modal")

    desc_dia = col1.selectbox(
        "Día", options=particion_modal.dia.unique(), key="desc_dia"
    )
    #     desc_tipo_dia = col1.selectbox(
    #         "Tipo de día", options=particion_modal.tipo_dia.unique(), key="desc_tipo_dia"
    #     )

    list_genero = particion_modal.genero_agregado.unique()
    list_genero = ["Todos" if item == "-" else item for item in list_genero]

    desc_genero = col1.selectbox("Genero", options=list_genero, key="desc_genero")

    query = f'select * from datos_particion_modal where dia="{desc_dia}"'
    if desc_genero != "Todos":
        query += f'and genero_agregado = "{desc_genero}"'

    etapas_modos = levanto_tabla_sql("datos_particion_modal", query=query)

    fig, modal_etapas, modal_viajes = plot_venn_diagram(etapas_modos)
    col2.pyplot(fig)
    col3.write("Etapas")
    col3.dataframe(modal_etapas.set_index("Modo"), height=300, width=300)
    col4.write("Viajes")
    col4.dataframe(modal_viajes.set_index("Modo"), height=300, width=300)

# with st.expander("Distancias de viajes"):

#     col1, col2 = st.columns([1, 4])

#     # hist_values = levanto_tabla_sql("distribucion")

#     # if len(hist_values) > 0:
#     #     hist_values.columns = [
#     #         "desc_dia",
#     #         "tipo_dia",
#     #         "Distancia (kms)",
#     #         "Viajes",
#     #         "Modo",
#     #     ]
#     #     hist_values = hist_values[hist_values["Distancia (kms)"] <= 60]
#     #     hist_values = hist_values.sort_values(["Modo", "Distancia (kms)"])

#     #     if col2.checkbox("Ver datos: distribución de viajes"):
#     #         col2.write(hist_values)

#     #     dist = hist_values.Modo.unique().tolist()
#     #     dist.remove("Todos")
#     #     dist = ["Todos"] + dist
#     #     modo_d = col1.selectbox("Modo", options=dist)
#     #     col1.write(f"Dia: {desc_dia}")
#     #     # col1.write(f"Tipo de día: {desc_tipo_dia}")

#     #     hist_values = hist_values[
#     #         (hist_values.desc_dia == desc_dia)
#     #         & (hist_values.tipo_dia.str.lower() == desc_tipo_dia.lower())
#     #         & (hist_values.Modo == modo_d)
#     #     ]

#     #     fig = px.histogram(
#     #         hist_values, x="Distancia (kms)", y="Viajes", nbins=len(hist_values)
#     #     )
#     #     fig.update_xaxes(type="category")
#     #     fig.update_yaxes(title_text="Viajes")

#     #     fig.update_layout(
#     #         xaxis=dict(tickmode="linear", tickangle=0, tickfont=dict(size=9)),
#     #         yaxis=dict(tickfont=dict(size=9)),
#     #     )

#     #     col2.plotly_chart(fig)
#     # else:
#     #     # Usar HTML para personalizar el estilo del texto
#     #     texto_html = """
#     #         <style>
#     #         .big-font {
#     #             font-size:30px !important;
#     #             font-weight:bold;
#     #         }
#     #         </style>
#     #         <div class='big-font'>
#     #             No hay datos para mostrar
#     #         </div>
#     #         """
#     #     col2.markdown(texto_html, unsafe_allow_html=True)
#     #     texto_html = """
#     #         <style>
#     #         .big-font {
#     #             font-size:30px !important;
#     #             font-weight:bold;
#     #         }
#     #         </style>
#     #         <div class='big-font'>
#     #             Verifique que los procesos se corrieron correctamente
#     #         </div>
#     #         """
#     #     col2.markdown(texto_html, unsafe_allow_html=True)


# # with st.expander("Viajes por hora"):

# #     col1, col2 = st.columns([1, 4])

# #     viajes_hora = levanto_tabla_sql("viajes_hora")

# #     modo_h = col1.selectbox("Modo", options=["Todos", "Por modos"], key="modo_h")

# #     if modo_h == "Todos":
# #         viajes_hora = viajes_hora[
# #             (viajes_hora.desc_dia == desc_mes)
# #             & (viajes_hora.tipo_dia.str.lower() == desc_tipo_dia.lower())
# #             & (viajes_hora.Modo == "Todos")
# #         ]
# #     else:
# #         viajes_hora = viajes_hora[
# #             (viajes_hora.desc_dia == desc_mes)
# #             & (viajes_hora.tipo_dia.str.lower() == desc_tipo_dia.lower())
# #             & (viajes_hora.Modo != "Todos")
# #         ]

# #     col1.write(f"Mes: {desc_mes}")
# #     col1.write(f"Tipo de día: {desc_tipo_dia}")

# #     viajes_hora = viajes_hora.sort_values("Hora")
# #     if col2.checkbox("Ver datos: viajes por hora"):
# #         col2.write(viajes_hora)

# #     fig_horas = px.line(viajes_hora, x="Hora", y="Viajes", color="Modo", symbol="Modo")

# #     fig_horas.update_xaxes(type="category")
# #     # fig_horas.update_layout()

# #     col2.plotly_chart(fig_horas)


with st.expander("Género y tarifas"):
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    socio_indicadores = levanto_tabla_sql("socio_indicadores")

    col1.write(f"Día: {desc_dia}")
    # col1.write(f"Tipo de día: {desc_tipo_dia}")

    if desc_dia != "Todos":
        st.session_state.socio_indicadores_ = socio_indicadores[
            (socio_indicadores.dia == desc_dia)
        ].copy()

    else:
        st.session_state.socio_indicadores_ = socio_indicadores[
            (socio_indicadores.tipo_dia == desc_tipo_dia)
        ].copy()

    (
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
    ) = traigo_socio_indicadores(st.session_state.socio_indicadores_)

    if totals is not None:
        col2.markdown(
            "<h4 style='font-size:16px;'>Total de viajes por género y tarifa</h4>",
            unsafe_allow_html=True,
        )
        col2.table(totals)
        col3.markdown(
            "<h4 style='font-size:16px;'>Porcentaje de viajes por género y tarifa</h4>",
            unsafe_allow_html=True,
        )
        col3.table(totals_porc.round(2).astype(str))

        col2.markdown(
            "<h4 style='font-size:16px;'>Cantidad promedio de viajes por género y tarifa</h4>",
            unsafe_allow_html=True,
        )
        col2.table(avg_viajes.round(2).astype(str))
        col3.markdown(
            "<h4 style='font-size:16px;'>Cantidad promedio de etapas por género y tarifa</h4>",
            unsafe_allow_html=True,
        )
        col3.table(avg_etapas.round(2).astype(str))

        col2.markdown(
            "<h4 style='font-size:16px;'>Total de etapas por género y modo</h4>",
            unsafe_allow_html=True,
        )
        col2.table(modos_genero_abs)
        col3.markdown(
            "<h4 style='font-size:16px;'>Porcentaje de etapas por género y modo</h4>",
            unsafe_allow_html=True,
        )
        col3.table(modos_genero_porc.round(2).astype(str))

        col2.markdown(
            "<h4 style='font-size:16px;'>Total de etapas por tarifa y modo</h4>",
            unsafe_allow_html=True,
        )
        col2.table(modos_tarifa_abs)
        col3.markdown(
            "<h4 style='font-size:16px;'>Porcentaje de etapas por tarifa y modo</h4>",
            unsafe_allow_html=True,
        )
        col3.table(modos_tarifa_porc.round(2).astype(str))

        col2.markdown(
            "<h4 style='font-size:16px;'>Distancias promedio (kms)</h4>",
            unsafe_allow_html=True,
        )
        col2.table(avg_distances.round(2).astype(str))

        col3.markdown(
            "<h4 style='font-size:16px;'>Tiempos promedio (minutos)</h4>",
            unsafe_allow_html=True,
        )
        col3.table(avg_times.round(2).astype(str))

        col2.markdown(
            "<h4 style='font-size:16px;'>Velocidades promedio (kms/hora)</h4>",
            unsafe_allow_html=True,
        )
        col2.table(avg_velocity.round(2).astype(str))

        col3.markdown(
            "<h4 style='font-size:16px;'>Tiempos promedio entre viajes (minutos)</h4>",
            unsafe_allow_html=True,
        )
        col3.table(avg_tiempo_entre_viajes.round(2).astype(str))
    else:
        col2.write("No hay datos para mostrar")
