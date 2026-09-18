"""Inferir el recorrido de una línea: filtrar las etapas sin coordenada, y
poder pedirlo para una sola línea desde el dashboard.

Las transacciones sin coordenada se conservan a propósito con latitud y longitud
en 0 y factor de expansión 0 (`eliminar_trx_fuera_bbox`), pero el lowess las
comía: son 3,2 millones de etapas en el mes del AMBA, y la línea 1056 tenía ahí
3.264 de sus 4.140. El ajuste terminaba siendo una recta entre el punto (0, 0) y
Buenos Aires, de 10.737 km. Filtrándolas, esa misma línea da 45,3 km.

importorskip: routes.py arrastra el stack pesado (geo/statsmodels lowess) al
importar, así que el módulo se skipea limpio donde esas deps no estén.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

routes = pytest.importorskip("urbantrips.carto.routes")


# ---------------------------------------------------------------------------
# el filtro de coordenadas nulas
# ---------------------------------------------------------------------------


def _ctx_que_captura_la_query():
    ctx = MagicMock()
    ctx.insumos.get_raw.return_value = pd.DataFrame()
    capturado = {}

    def fake_query(q):
        capturado["q"] = q
        return pd.DataFrame(columns=["id_linea", "longitud", "latitud"])

    ctx.data.query.side_effect = fake_query
    return ctx, capturado


def test_la_inferencia_masiva_excluye_las_etapas_sin_coordenada():
    ctx, capturado = _ctx_que_captura_la_query()

    routes.infer_routes_geoms(ctx)

    q = capturado["q"].lower()
    assert "latitud != 0" in q
    assert "longitud != 0" in q


def test_la_inferencia_de_una_linea_excluye_las_etapas_sin_coordenada():
    ctx, capturado = _ctx_que_captura_la_query()

    routes.infer_route_geom_for_line(ctx, 1056)

    q = capturado["q"].lower()
    assert "latitud != 0" in q
    assert "longitud != 0" in q
    assert "id_linea = 1056" in q


# ---------------------------------------------------------------------------
# inferencia por línea
# ---------------------------------------------------------------------------


def _etapas_de_una_linea(id_linea=1056, n=200):
    """Etapas repartidas a lo largo de un corredor de sudoeste a noreste."""
    return pd.DataFrame(
        {
            "id_linea": [id_linea] * n,
            "longitud": [-58.50 + i * 0.0004 for i in range(n)],
            "latitud": [-34.60 + i * 0.0002 for i in range(n)],
        }
    )


def test_infiere_guarda_las_dos_direcciones_y_devuelve_el_recorrido():
    ctx = MagicMock()
    ctx.data.query.return_value = _etapas_de_una_linea()

    recorrido = routes.infer_route_geom_for_line(ctx, 1056)

    assert recorrido is not None
    assert sorted(recorrido.direction) == [0, 1]
    assert set(recorrido.id_linea) == {1056}
    assert recorrido.wkt.str.startswith("LINESTRING").all()

    # se guarda en inferred_lines_geoms...
    guardadas = [c.args[1] for c in ctx.insumos.append_raw.call_args_list]
    assert guardadas == ["inferred_lines_geoms"]

    # ...y se refresca lines_geoms respetando el recorrido oficial
    sql = " ".join(str(c.args[0]) for c in ctx.insumos.execute.call_args_list)
    assert "delete from inferred_lines_geoms where id_linea = 1056" in sql.lower()
    assert "delete from lines_geoms where id_linea = 1056" in sql.lower()
    assert "coalesce(o.wkt, i.wkt)" in sql.lower()


def test_la_direccion_1_es_la_0_al_reves():
    ctx = MagicMock()
    ctx.data.query.return_value = _etapas_de_una_linea()

    recorrido = routes.infer_route_geom_for_line(ctx, 1056)

    from shapely import wkt as shapely_wkt

    ida = shapely_wkt.loads(recorrido.loc[recorrido.direction == 0, "wkt"].item())
    vuelta = shapely_wkt.loads(recorrido.loc[recorrido.direction == 1, "wkt"].item())

    assert list(ida.coords) == list(vuelta.coords)[::-1]


def test_sin_etapas_con_coordenada_no_infiere_ni_escribe():
    ctx = MagicMock()
    ctx.data.query.return_value = pd.DataFrame(
        columns=["id_linea", "longitud", "latitud"]
    )

    assert routes.infer_route_geom_for_line(ctx, 1056) is None
    ctx.insumos.append_raw.assert_not_called()


def test_si_el_ajuste_falla_no_escribe(monkeypatch):
    # Pasa de verdad: en la línea 427 (8,1 M de puntos) el lowess devuelve
    # coordenadas infinitas y la geometría no es válida.
    ctx = MagicMock()
    ctx.data.query.return_value = _etapas_de_una_linea()
    monkeypatch.setattr(routes.geo, "lowess_linea", lambda df: None)

    assert routes.infer_route_geom_for_line(ctx, 1056) is None
    ctx.insumos.append_raw.assert_not_called()


# ---------------------------------------------------------------------------
# el inferido es sólo para las líneas que NO tienen recorrido oficial
# ---------------------------------------------------------------------------


def test_no_infiere_las_lineas_que_ya_tienen_recorrido_oficial():
    # `build_routes_from_official_inferred` se queda con el oficial cuando
    # existe, así que inferir esas líneas es trabajo que se tira: en el mes del
    # AMBA eran 330 de 406.
    ctx = MagicMock()
    capturado = {}

    def fake_get_raw(tabla):
        if tabla == "official_lines_geoms":
            return pd.DataFrame(
                {"id_linea": [10, 10, 20, 20], "direction": [0, 1, 0, 1],
                 "wkt": ["LINESTRING(0 0,1 1)"] * 4}
            )
        return pd.DataFrame()

    def fake_query(q):
        capturado["q"] = q
        return pd.DataFrame(columns=["id_linea", "longitud", "latitud"])

    ctx.insumos.get_raw.side_effect = fake_get_raw
    ctx.data.query.side_effect = fake_query

    routes.infer_routes_geoms(ctx)

    q = capturado["q"].lower()
    assert "not in (10, 20)" in q


def test_el_recorrido_oficial_no_depende_de_que_haya_inferido():
    """Una línea con oficial pero sin inferido tiene que llegar a lines_geoms.

    Pasaba de verdad: la línea 407 (24 etapas) tenía recorrido oficial, el
    lowess no ajustaba con tan pocos puntos y el INSERT, que salía del inferido
    con un LEFT JOIN, la dejaba afuera.
    """
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    for tabla in ("lines_geoms", "branches_geoms", "official_branches_geoms"):
        conn.execute(
            f"create table {tabla} (id_linea BIGINT, direction INT, wkt TEXT)"
        )
    conn.execute("""
        create table official_lines_geoms as select * from (values
            (407, 0, 'OFICIAL_407'), (407, 1, 'OFICIAL_407_INV'),
            (100, 0, 'OFICIAL_100'), (100, 1, 'OFICIAL_100_INV')
        ) as t(id_linea, direction, wkt)
    """)
    conn.execute("""
        create table inferred_lines_geoms as select * from (values
            (100, 0, 'INFERIDO_100'), (100, 1, 'INFERIDO_100_INV'),
            (200, 0, 'INFERIDO_200'), (200, 1, 'INFERIDO_200_INV')
        ) as t(id_linea, direction, wkt)
    """)

    ctx = MagicMock()
    ctx.insumos.execute.side_effect = conn.execute

    routes.build_routes_from_official_inferred(ctx)

    salida = conn.execute(
        "select id_linea, direction, wkt from lines_geoms order by 1, 2"
    ).fetchdf()
    por_linea = dict(
        zip(salida.loc[salida.direction == 0, "id_linea"],
            salida.loc[salida.direction == 0, "wkt"])
    )

    # la que sólo tiene oficial ya no se pierde
    assert por_linea[407] == "OFICIAL_407"
    # con los dos, gana el oficial
    assert por_linea[100] == "OFICIAL_100"
    # la que sólo tiene inferido sigue estando
    assert por_linea[200] == "INFERIDO_200"
    assert len(salida) == 6


# ---------------------------------------------------------------------------
# el salteo incremental no conserva geometrías inservibles
# ---------------------------------------------------------------------------


def _wkt_de_largo(km):
    """LineString horizontal de aproximadamente `km` sobre el AMBA."""
    grados = km / 111.32
    return f"LINESTRING(-58.5 -34.6, {-58.5 + grados} -34.6)"


def _inferidas(pares):
    """pares: [(id_linea, km)] -> tabla inferred_lines_geoms con dos direcciones."""
    filas = []
    for id_linea, km in pares:
        for direction in (0, 1):
            filas.append(
                {"id_linea": id_linea, "direction": direction,
                 "wkt": _wkt_de_largo(km)}
            )
    return pd.DataFrame(filas)


def test_recalcula_los_recorridos_guardados_que_no_son_una_linea_real():
    """Las bases procesadas antes del filtro tienen recorridos de 10.600 km
    guardados. El salteo incremental los conservaba para siempre."""
    ctx = MagicMock()
    capturado = {}

    def fake_get_raw(tabla):
        if tabla == "inferred_lines_geoms":
            # la 10 está bien; la 20 es la recta a Null Island
            return _inferidas([(10, 22.0), (20, 10600.0)])
        return pd.DataFrame()

    def fake_query(q):
        capturado["q"] = q
        return pd.DataFrame(columns=["id_linea", "longitud", "latitud"])

    ctx.insumos.get_raw.side_effect = fake_get_raw
    ctx.data.query.side_effect = fake_query

    routes.infer_routes_geoms(ctx)

    q = capturado["q"].lower()
    assert "10" in q, "la línea sana tendría que saltearse"
    assert "not in (10)" in q, f"la línea rota tendría que recalcularse; query: {q}"


def test_la_geometria_recalculada_reemplaza_a_la_vieja(monkeypatch):
    import geopandas as gpd
    from shapely.geometry import LineString

    ctx = MagicMock()

    def fake_get_raw(tabla):
        if tabla == "inferred_lines_geoms":
            return _inferidas([(20, 10600.0)])
        return pd.DataFrame()

    ctx.insumos.get_raw.side_effect = fake_get_raw
    ctx.data.query.return_value = pd.DataFrame(
        {"id_linea": [20] * 5, "longitud": [-58.5] * 5, "latitud": [-34.6] * 5}
    )
    monkeypatch.setattr(
        routes.geo, "lowess_linea",
        lambda df: gpd.GeoDataFrame(
            {"geometry": [LineString([(-58.5, -34.6), (-58.3, -34.6)])]},
            geometry="geometry", crs=4326, index=[0],
        ),
    )

    routes.infer_routes_geoms(ctx)

    guardado = ctx.insumos.save_raw.call_args.args[0]
    # una sola geometría por dirección: la vieja no quedó duplicada
    assert len(guardado) == 2
    assert set(guardado.id_linea) == {20}
    assert not guardado.wkt.str.contains("36.7").any()  # el extremo de los 10.600 km


def test_no_conserva_una_geometria_inservible_que_no_se_pudo_rehacer(monkeypatch):
    """El caso de la línea 427 (FFCC Sarmiento, 8,1 M de puntos).

    Se la marca para rehacer porque lo guardado son 10.604 km, el ajuste vuelve
    a fallar, y antes se conservaba la geometría vieja: la línea quedaba con un
    recorrido inservible y encima se recalculaba en cada corrida.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    ctx = MagicMock()

    def fake_get_raw(tabla):
        if tabla == "inferred_lines_geoms":
            # la 10 sirve; la 427 es la que hay que rehacer y va a fallar
            return _inferidas([(10, 22.0), (427, 10604.0)])
        return pd.DataFrame()

    ctx.insumos.get_raw.side_effect = fake_get_raw
    ctx.data.query.return_value = pd.DataFrame(
        {"id_linea": [427] * 5, "longitud": [-58.5] * 5, "latitud": [-34.6] * 5}
    )
    # el ajuste devuelve algo válido pero absurdo, como pasa en esa línea
    monkeypatch.setattr(
        routes.geo, "lowess_linea",
        lambda df: gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (-58.5, -34.6)])]},
            geometry="geometry", crs=4326, index=[0],
        ),
    )

    routes.infer_routes_geoms(ctx)

    guardado = ctx.insumos.save_raw.call_args.args[0]
    ids = set(guardado.id_linea)
    assert 427 not in ids, "no se guarda un recorrido que no representa una línea"
    assert ids == {10}, "la que sí sirve se conserva"


def test_no_conserva_los_inferidos_de_lineas_con_recorrido_oficial(monkeypatch):
    """Si hay oficial, el inferido no hace falta: no se arrastra."""
    import geopandas as gpd
    from shapely.geometry import LineString

    ctx = MagicMock()

    def fake_get_raw(tabla):
        if tabla == "inferred_lines_geoms":
            return _inferidas([(10, 22.0), (50, 18.0)])
        if tabla == "official_lines_geoms":
            return pd.DataFrame(
                {"id_linea": [50, 50], "direction": [0, 1],
                 "wkt": ["LINESTRING(0 0,1 1)"] * 2}
            )
        return pd.DataFrame()

    ctx.insumos.get_raw.side_effect = fake_get_raw
    ctx.data.query.return_value = pd.DataFrame(
        {"id_linea": [99] * 5, "longitud": [-58.5] * 5, "latitud": [-34.6] * 5}
    )
    monkeypatch.setattr(
        routes.geo, "lowess_linea",
        lambda df: gpd.GeoDataFrame(
            {"geometry": [LineString([(-58.5, -34.6), (-58.3, -34.6)])]},
            geometry="geometry", crs=4326, index=[0],
        ),
    )

    routes.infer_routes_geoms(ctx)

    ids = set(ctx.insumos.save_raw.call_args.args[0].id_linea)
    assert 50 not in ids, "la 50 tiene oficial: su inferido no se conserva"
    assert ids == {10, 99}
