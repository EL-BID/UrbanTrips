import folium
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
from dash_storage import leer_configs_generales
from dash_utils import (
    levanto_tabla_sql,
    etiqueta_linea,
    get_logo,
    create_linestring_od,
    create_squared_polygon,
    get_epsg_m,
    extract_hex_colors_from_cmap,
    levanto_tabla_sql_local,
    configurar_selector_corrida,
)


# try:
from urbantrips.kpi.kpi import compute_route_section_load, run_basic_kpi
from urbantrips.viz.viz import visualize_route_section_load
from urbantrips.kpi.line_od_matrix import compute_lines_od_matrix
from urbantrips.viz.line_od_matrix import visualize_lines_od_matrix
from urbantrips.kpi.supply_kpi import compute_route_section_supply
from urbantrips.viz.section_supply import visualize_route_section_supply_data
from urbantrips.carto.routes import infer_route_geom_for_line
from urbantrips.carto.route_sections import (
    N_SECTIONS_DEFAULT,
    N_SECTIONS_MAX,
    N_SECTIONS_MIN,
    SECTION_METERS_MAX,
    SECTION_METERS_MIN,
    SECTION_METERS_OPTIONS,
    SectionParamsError,
    describe_route_sections,
    formatear_largo,
    formatear_numero,
)
from urbantrips.utils import utils
from urbantrips.utils.check_configs import check_config
# except ImportError as e:
#     st.error(
#         f"Falta una librería requerida: {e}. Algunas funcionalidades no estarán disponibles. \nSe requiere full acceso a Urbantrips para correr esta página"
#     )
#     st.stop()
from urbantrips.dashboard import dashboard_ctx
from urbantrips.storage.access import DatabaseBusyError, write_access

# El StorageContext ya NO se cachea con @st.cache_resource: una conexión DuckDB
# abierta toma el lock del archivo y bloquea tanto a otro dashboard como al
# pipeline. Se abre y cierra dentro de `with dashboard_ctx() as ctx:`.
# --- Función para levantar tablas SQL y almacenar en session_state ---
def cargar_tabla_sql(tabla_sql, tipo_conexion="dash", query=""):
    if f"{tabla_sql}_{tipo_conexion}" not in st.session_state:
        tabla = utils.levanto_tabla_sql(
            tabla_sql,
            tabla_tipo=tipo_conexion,
            query=query,
        )
        if tabla.empty:
            st.error(f"{tabla_sql} no existe")
        st.session_state[f"{tabla_sql}_{tipo_conexion}"] = tabla
    return st.session_state[f"{tabla_sql}_{tipo_conexion}"]


AYUDA_SECCIONES = f"""
El recorrido de la línea se parte en tramos consecutivos para medir cuánta gente
sube, baja o viaja en cada uno. Se define de **una sola** de estas dos formas:

- **Cantidad de secciones** ({N_SECTIONS_MIN} a {N_SECTIONS_MAX}): el recorrido
  se divide en esa cantidad de tramos iguales, midan lo que midan.
- **Metros por sección** ({formatear_numero(SECTION_METERS_MIN)} a
  {formatear_numero(SECTION_METERS_MAX)}): cada tramo mide esos metros, y la
  cantidad de tramos sale del largo del recorrido.

Elegís una y la otra se calcula sola. Los límites no son caprichosos: menos de
{N_SECTIONS_MIN} tramos no distingue nada dentro del recorrido, y más de
{N_SECTIONS_MAX} (o tramos de menos de {formatear_numero(SECTION_METERS_MIN)} m)
divide la demanda en pedazos tan chicos que el resultado es ruido.
"""


def cargar_dias_disponibles():
    """Días presentes en la base, cacheados (una sola consulta por sesión)."""
    if "dias_disponibles" not in st.session_state:
        dias = utils.levanto_tabla_sql(
            "etapas",
            tabla_tipo="data",
            query="SELECT DISTINCT dia FROM etapas ORDER BY dia",
        )
        st.session_state["dias_disponibles"] = (
            [] if len(dias) == 0 else dias.dia.astype(str).tolist()
        )
    return st.session_state["dias_disponibles"]


def dias_del_periodo(yr_mo, day_type):
    """Días del período elegido que corresponden al tipo de día pedido."""
    dias = [
        dia
        for dia in cargar_dias_disponibles()
        if yr_mo is None or dia.startswith(str(yr_mo))
    ]
    if not dias:
        return []

    habiles = pd.to_datetime(pd.Series(dias), format="%Y-%m-%d").dt.dayofweek < 5
    quiero_habiles = day_type == "weekday"
    return [dia for dia, habil in zip(dias, habiles) if habil == quiero_habiles]


def cargar_recorrido(id_linea):
    """
    Trae el recorrido de una línea desde insumos, cacheado por línea.

    Devuelve un GeoDataFrame de una fila (o vacío si la línea no tiene ningún
    recorrido, ni oficial ni inferido) junto con el dato de si ese recorrido es
    el oficial o el inferido a partir de los datos.
    """
    cache_key = f"recorrido_{id_linea}"
    if cache_key not in st.session_state:
        recorrido = utils.levanto_tabla_sql(
            "lines_geoms",
            tabla_tipo="insumos",
            query=(
                "SELECT * FROM lines_geoms "
                f"WHERE direction = 0 AND id_linea = {int(id_linea)}"
            ),
        )
        oficiales = utils.levanto_tabla_sql(
            "official_lines_geoms",
            tabla_tipo="insumos",
            query=(
                "SELECT id_linea FROM official_lines_geoms "
                f"WHERE id_linea = {int(id_linea)} LIMIT 1"
            ),
        )
        st.session_state[cache_key] = (recorrido, len(oficiales) > 0)
    return st.session_state[cache_key]


def olvidar_recorrido(id_linea):
    """Saca el recorrido del cache, para releerlo después de inferirlo."""
    st.session_state.pop(f"recorrido_{id_linea}", None)


def inferir_recorrido(id_linea):
    """
    Infiere el recorrido de la línea en el momento y lo deja guardado.

    La inferencia de todas las líneas es parte del pipeline y es cara, así que
    no se rehace acá: se calcula sólo esta línea, que va de instantáneo a unos
    segundos según cuántas etapas tenga.
    """
    with write_access("inferir el recorrido de una línea"), dashboard_ctx() as ctx:
        return infer_route_geom_for_line(ctx, id_linea)


def describir_secciones(id_linea, n_sections, section_meters):
    """
    Resuelve en cuántos tramos queda dividida la línea y de qué largo, para
    mostrarlo antes de procesar. Devuelve None si no hay recorrido cargado.
    """
    recorrido, es_oficial = cargar_recorrido(id_linea)
    if len(recorrido) == 0:
        return None

    detalle = describe_route_sections(recorrido, n_sections, section_meters)
    fila = detalle.iloc[0].to_dict()
    fila["oficial"] = es_oficial
    fila["geometry"] = recorrido.geometry.iloc[0]
    return fila


def boton_inferir(id_linea, etiqueta, key):
    """
    Botón que infiere el recorrido de la línea en el momento y lo guarda.

    Se ofrece en los dos casos sin salida: cuando no hay ningún recorrido, y
    cuando el que hay no se puede usar (típicamente un inferido viejo que quedó
    guardado antes de que la inferencia filtrara las etapas sin coordenada).
    """
    if not st.button(etiqueta, key=key):
        return

    with st.spinner("Ajustando el recorrido sobre las etapas de la línea..."):
        try:
            recorrido = inferir_recorrido(id_linea)
        except DatabaseBusyError as e:
            st.error(
                f"{e} Cerrá el otro dashboard o esperá a que termine la "
                "corrida y volvé a intentar."
            )
            return

    if recorrido is None:
        st.error(
            "No se pudo inferir el recorrido de esta línea: o no tiene "
            "suficientes etapas con coordenada válida, o el ajuste no "
            "produjo una curva utilizable. El motivo queda en el log."
        )
        return

    olvidar_recorrido(id_linea)
    st.success(
        "Recorrido inferido y guardado. Es una aproximación ajustada sobre las "
        "etapas de la línea, no un recorrido oficial: conviene mirarlo antes de "
        "usar los resultados."
    )
    st.rerun()


def etiqueta_linea_id(id_linea):
    """Nombre e id de la línea, buscando el nombre en la metadata."""
    nombres = metadata_lineas.loc[
        metadata_lineas.id_linea == id_linea, "nombre_linea"
    ]
    nombre = nombres.iloc[0] if len(nombres) else "sin nombre"
    return etiqueta_linea(nombre, id_linea)


def mostrar_mapa_recorrido(estado):
    """
    Dibuja el recorrido de la línea, en el color que corresponde a su origen.

    Ver el trazado es la forma más rápida de saber si un recorrido inferido
    representa a la línea o no: los que estaban mal se veían como una recta que
    cruzaba el Atlántico.
    """
    geom = estado.get("geometry")
    if geom is None or geom.is_empty:
        return

    color = "#0B5C8A" if estado["oficial"] else "#B4581F"
    minx, miny, maxx, maxy = geom.bounds

    mapa = folium.Map(tiles="cartodbpositron")
    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in geom.coords],
        color=color,
        weight=4,
        opacity=0.85,
        tooltip=etiqueta_linea_id(estado["id_linea"]),
    ).add_to(mapa)

    # Las puntas, para ver de una si el trazado tiene sentido.
    for punto, nombre in ((geom.coords[0], "Inicio"), (geom.coords[-1], "Fin")):
        folium.CircleMarker(
            location=(punto[1], punto[0]),
            radius=5,
            color=color,
            fill=True,
            fill_opacity=1,
            popup=nombre,
        ).add_to(mapa)

    mapa.fit_bounds([(miny, minx), (maxy, maxx)])
    folium_static(mapa, width=700, height=420)


def mostrar_estado_secciones(estado, id_linea):
    """
    Renderiza la línea de estado del recorrido debajo de los parámetros.

    Cuando la línea no tiene recorrido usable —porque no hay ninguno, o porque
    el que hay no representa una línea real— ofrece inferirlo en el momento en
    vez de dejar al usuario sin salida.
    """
    if estado is None:
        st.warning(
            f"La línea {etiqueta_linea_id(id_linea)} no tiene recorrido "
            "cargado: no hay uno oficial y tampoco se infirió uno a partir de "
            "los datos. Sin recorrido no se puede analizar por secciones."
        )
        boton_inferir(
            id_linea, "Inferir el recorrido a partir de los datos", "inferir_nuevo"
        )
        return

    if estado["oficial"]:
        origen = "recorrido oficial"
    else:
        origen = "recorrido inferido a partir de los datos"

    if not estado["valido"]:
        st.error(
            f"⚠️ {etiqueta_linea_id(estado['id_linea'])} — {origen}. "
            f"{estado['motivo']}"
        )
        mostrar_mapa_recorrido(estado)
        if not estado["oficial"]:
            # El recorrido inferido que está guardado no sirve. Se puede
            # recalcular sin rehacer toda la corrida: la inferencia masiva es
            # incremental y no vuelve sobre las líneas que ya tienen geometría.
            boton_inferir(
                id_linea, "Recalcular el recorrido inferido", "inferir_recalcular"
            )
        return

    st.info(
        f"{etiqueta_linea_id(estado['id_linea'])} — {origen}, de "
        f"{formatear_largo(estado['largo_m'])}. Con {estado['n_sections']} "
        f"secciones, cada tramo mide "
        f"**{formatear_numero(estado['section_meters'])} m**."
    )
    if estado["motivo"]:
        st.warning(estado["motivo"])

    if not estado["oficial"]:
        st.caption(
            "Este recorrido no es oficial: es una aproximación ajustada sobre "
            "las etapas de la línea. Conviene mirarlo en el mapa antes de usar "
            "los resultados."
        )

    mostrar_mapa_recorrido(estado)


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
            # El texto lleva el id: el filtro busca en las dos columnas, así
            # que sin el id no se entiende por qué aparece una opción, y los
            # nombres repetidos eran indistinguibles (con `.index()` sobre el
            # texto, elegir la segunda `FFCC MITRE` devolvía la primera).
            opciones = df_filtrado.apply(
                lambda row: etiqueta_linea(row["nombre_linea"], row["id_linea"]),
                axis=1,
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
alias_seleccionado = configurar_selector_corrida()
# check_config(corrida=alias_seleccionado)
# st.text(f"Alias seleccionado: {alias_seleccionado}")

try:

    # --- Cargar configuraciones y conexiones en session_state ---
    if "configs" not in st.session_state:
        # autogenerado=False: leer el config base (configuraciones_generales.yaml),
        # consistente con la resolución de DB en dash_utils (get_db_path también usa
        # autogenerado=False). El autogenerado puede quedar viejo de otra corrida y
        # desincronizar flags como lineas_contienen_ramales respecto de la base cargada.
        st.session_state.configs = leer_configs_generales(autogenerado=False)

    configs = st.session_state.configs
    h3_legs_res = configs["resolucion_h3"]
    # El autogenerado quedó obsoleto: todo se guarda bajo un único alias =
    # alias_db_insumos (alias_db_data/alias_db_dashboard ya no aplican).
    alias = configs.get("alias_db_insumos", "")
    st.text(
        f"Base de datos seleccionada: {alias}. Si no es la correcta, cambiar el archivo configuraciones_generales.yaml"
    )
    use_branches = configs["lineas_contienen_ramales"]

    metadata_lineas = cargar_tabla_sql("metadata_lineas", "insumos")[
        ["id_linea", "nombre_linea"]
    ]

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
    "dias_7",
]:
    if var not in st.session_state:
        st.session_state[var] = None


st.header("Herramientas")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.subheader("Periodo")

    kpi_lineas = levanto_tabla_sql("agg_indicadores")
    if len(kpi_lineas) == 0:
        months = None
    else:
        months = kpi_lineas.mes.unique()

    day_type = col1.selectbox("Tipo de dia  ", options=["weekday", "weekend"])
    st.session_state["day_type_7"] = day_type

    # add month and year
    yr_mo = col1.selectbox("Periodo  ", options=months, key="year_month")
    st.session_state["yr_mo_7"] = yr_mo

    # Hasta ahora el período se guardaba y no lo usaba nadie: las cuatro
    # funciones barrían todos los días de la base. Con un mes cargado eso son
    # decenas de días por línea, y si la corrida cruza dos meses calendario se
    # procesan y grafican los dos. Acá se elige explícitamente qué días entran.
    dias_periodo = dias_del_periodo(yr_mo, day_type)
    dias_elegidos = col1.multiselect(
        "Días a procesar",
        options=dias_periodo,
        default=dias_periodo,
        help=(
            "Los resultados son un promedio de los días elegidos. Procesar "
            "menos días es más rápido y usa menos memoria."
        ),
    )
    st.session_state["dias_7"] = dias_elegidos

    if dias_periodo and not dias_elegidos:
        col1.warning("Elegí al menos un día.")

with col2:
    st.subheader("Línea")
    seleccionar_linea("base_input", "base_select")

with col3:
    st.subheader("Parámetros")

    # El recorrido se divide por cantidad de secciones o por metros por sección,
    # nunca por las dos cosas: hasta ahora convivían los dos campos y, si el
    # usuario llenaba ambos, los metros ganaban en silencio. El radio deja
    # habilitado uno solo y el otro valor aparece calculado más abajo.
    modo_division = col3.radio(
        "Dividir el recorrido por",
        options=["Cantidad de secciones", "Metros por sección"],
        key="modo_division_7",
    )

    if modo_division == "Cantidad de secciones":
        n_sections = col3.slider(
            "Cantidad de secciones",
            min_value=N_SECTIONS_MIN,
            max_value=N_SECTIONS_MAX,
            value=N_SECTIONS_DEFAULT,
        )
        section_meters = None
    else:
        n_sections = None
        # select_slider y no slider: el paso no es uniforme (de a 100 m hasta
        # los 5 km, de a 1 km después), así que las posiciones se enumeran.
        section_meters = col3.select_slider(
            "Metros por sección",
            options=SECTION_METERS_OPTIONS,
            value=1000,
        )

    st.session_state["n_sections_7"] = n_sections
    st.session_state["section_meters_7"] = section_meters

    (
        col3a,
        col3b,
    ) = st.columns([1, 1])

    rango_desde = col3a.selectbox(
        "Rango horario (desde) ", options=range(0, 24), key="rango_hora_desde", index=9
    )
    rango_hasta = col3b.selectbox(
        "Rango horario (hasta)", options=range(0, 24), key="rango_hora_hasta", index=9
    )
    hour_range = [rango_desde, rango_hasta]
    st.session_state["hour_range_7"] = hour_range


line_ids = st.session_state["id_linea_7"]

with st.expander("¿Cómo se divide el recorrido en secciones?"):
    st.markdown(AYUDA_SECCIONES)

# Se resuelve el recorrido antes de procesar y se muestra en qué queda dividido.
# Así el usuario ve el resultado de sus parámetros (y el caso de las líneas sin
# recorrido oficial, cuyo recorrido inferido no se puede seccionar) sin tener
# que esperar a que corra todo el bloque de cálculo.
estado_secciones = None
if line_ids is not None:
    estado_secciones = describir_secciones(
        line_ids,
        st.session_state["n_sections_7"],
        st.session_state["section_meters_7"],
    )
    mostrar_estado_secciones(estado_secciones, line_ids)
else:
    st.info("Elegí una línea para empezar.")

puede_procesar = (
    estado_secciones is not None
    and estado_secciones["valido"]
    and bool(st.session_state["dias_7"])
)

if st.button("Comenzar a procesar", disabled=not puede_procesar):
    if puede_procesar:

        hour_range = st.session_state["hour_range_7"]
        day_type = st.session_state["day_type_7"]
        dias = st.session_state["dias_7"]

        # Se pasa el parámetro que eligió el usuario (uno de los dos en None) y
        # la librería deriva el otro.
        n_sections_pedido = st.session_state["n_sections_7"]
        section_meters_pedido = st.session_state["section_meters_7"]

        # Para buscar los resultados guardados alcanza con la cantidad de
        # secciones ya resuelta al validar el recorrido: las funciones de
        # visualización no tienen por qué volver a derivarla del largo.
        n_sections = int(estado_secciones["n_sections"])

        # Las visualizaciones leen las tablas de resultados, que acumulan un
        # promedio por período: se acotan a los meses que se acaban de calcular
        # para no graficar corridas anteriores.
        yr_mos = sorted({dia[:7] for dia in dias})

        # Este bloque escribe en las bases (KPIs, matriz OD, oferta y carga por
        # sección), así que corre dentro de una ventana de escritura explícita.
        # Si otro dashboard o una corrida tiene la base tomada, se avisa en vez
        # de romper con un traceback de DuckDB.
        try:
            with write_access("procesar indicadores de línea"), dashboard_ctx() as ctx:
                st.write("Calculando indicadores basicos...")
                # Se llamaba con dia=None, que hace SELECT * (24 columnas) de
                # todas las etapas de la línea en toda la base. Acotado a los
                # días elegidos, en una sola pasada: el loop día por día sería
                # más liviano todavía, pero reconstruye la red de OSM en cada
                # llamada y eso se paga una vez por día.
                run_basic_kpi(ctx, id_linea=[line_ids], dias=dias)

                st.write("Calculando la matriz OD de la linea...")
                # Se computa la matriz OD de las lineas
                compute_lines_od_matrix(
                    ctx,
                    line_ids=[line_ids],
                    hour_range=hour_range,
                    n_sections=n_sections_pedido,
                    section_meters=section_meters_pedido,
                    day_type=day_type,
                    save_csv=True,
                    dias=dias,
                )
                st.write("Visualizando la matriz OD de la linea...")
                # Se visualiza la matriz OD de las lineas
                visualize_lines_od_matrix(
                    ctx,
                    line_ids=[line_ids],
                    hour_range=hour_range,
                    day_type=day_type,
                    n_sections=n_sections,
                    section_meters=None,
                    yr_mos=yr_mos,
                    stat="totals",
                )

                st.write(
                    "Calculando los estadisticos de oferta por secciones de las lineas..."
                )
                # Calcula los estadisticos de oferta por sección de las lineas
                route_section_supply = compute_route_section_supply(
                    ctx,
                    line_ids=[line_ids],
                    hour_range=hour_range,
                    n_sections=n_sections_pedido,
                    section_meters=section_meters_pedido,
                    day_type=day_type,
                    dias=dias,
                )

                st.write(
                    "Visualizando los estadisticos de oferta por secciones de las lineas..."
                )
                # Visualiza los estadisticos de oferta por sección de las lineas
                visualize_route_section_supply_data(
                    ctx,
                    line_ids=[line_ids],
                    hour_range=hour_range,
                    day_type=day_type,
                    n_sections=n_sections,
                    section_meters=None,
                    yr_mos=yr_mos,
                )

                st.write(
                    "Calculando los estadisticos de carga de las secciones de las lineas..."
                )
                # Se calculan los estadisticos de carga de las secciones de las lineas
                compute_route_section_load(
                    ctx,
                    line_ids=[line_ids],
                    hour_range=hour_range,
                    n_sections=n_sections_pedido,
                    section_meters=section_meters_pedido,
                    day_type=day_type,
                    dias=dias,
                )

                st.write(
                    "Visualizando los estadisticos de carga de las secciones de las lineas..."
                )
                # Se visualizan los estadisticos de carga de las secciones de las lineas
                visualize_route_section_load(
                    ctx,
                    line_ids=[line_ids],
                    hour_range=hour_range,
                    day_type=day_type,
                    n_sections=n_sections,
                    section_meters=None,
                    yr_mos=yr_mos,
                    save_gdf=True,
                    stat="totals",
                    factor=500,
                    factor_min=10,
                )
        except DatabaseBusyError as e:
            st.error(
                f"{e} Cerrá el otro dashboard o esperá a que termine la corrida "
                "y volvé a intentar."
            )
            st.stop()
        except SectionParamsError as e:
            # La pantalla ya valida el recorrido antes de habilitar el botón,
            # así que llegar acá significa que la línea cambió entre medio.
            # Igual se muestra el mensaje en vez de un traceback.
            st.error(str(e))
            st.stop()

        st.write(
            "Resultados pueden consultarse en el directorio UrbanTrips/"
            "resultados o en la pestaña Indicadores de oferta y demanda"
        )

    else:
        st.write("No hay datos para mostrar")
