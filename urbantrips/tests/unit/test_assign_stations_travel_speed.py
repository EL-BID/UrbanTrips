# urbantrips/tests/unit/test_assign_stations_travel_speed.py
"""`travel_times_stations.travel_speed` se escribía siempre NULL (bug 2026-07-27).

Dos causas encadenadas en `assign_stations_od`:

1. La query de etapas no traía `distance_od` — la tabla `etapas` nunca la tuvo
   (su columna legacy se llamaba `distancia`, y hoy la métrica vive en
   `travel_times_legs`). El `reindex(columns=[..., "distance_od"])` posterior la
   creaba toda-NaN en silencio, así que `kmh_od` salía NaN.
2. Había DOS `reindex` encadenados sobre el frame final: el segundo tiraba el
   `kmh_od` recién calculado y creaba `travel_speed` toda-NaN.

El fix trae `distance_od` de `travel_times_legs` y renombra `kmh_od` →
`travel_speed` en un solo `reindex`.
"""
import h3
import numpy as np
import pandas as pd
import pytest


# Dos puntos en el AMBA, ~1,3 km entre sí. Las estaciones van cerca del centro
# de la celda H3 de cada etapa pero NO encima: classify_leg_into_station usa
# sjoin_nearest con exclusive=True, que descarta las geometrías idénticas.
_H3_RES = 9
_LATLNG_O = (-34.6037, -58.3816)
_LATLNG_D = (-34.6120, -58.3700)
_OFFSET_GRADOS = 0.001  # ~110 m, muy por debajo de tolerancia_parada_destino

_CONFIG = {
    "tiempos_viaje_estaciones": "travel_time_stations.csv",
    "tolerancia_parada_destino": 2200,
    "epsg_m": 5347,
    "resolucion_h3": _H3_RES,
}


class _FakeData:
    """Captura lo que assign_stations_od consulta y escribe."""

    def __init__(self, legs_df):
        self._legs = legs_df
        self.queries = []
        self.written = {}
        self.executed = []

    def get_run_days(self):
        return pd.DataFrame({"dia": ["2024-01-01"]})

    def query(self, sql):
        self.queries.append(sql)
        return self._legs.copy()

    def execute(self, sql):
        self.executed.append(sql)

    def append_raw(self, df, table_name):
        self.written.setdefault(table_name, []).append(df.copy())


class _FakeInsumos:
    def __init__(self, tts):
        self._tts = tts

    def get_travel_times_stations(self):
        return self._tts.copy()


@pytest.fixture
def escenario(monkeypatch):
    from urbantrips.datamodel import legs as legs_module
    from urbantrips.geo import geo as geo_module

    monkeypatch.setattr(legs_module, "leer_configs_generales", lambda *a, **k: _CONFIG)
    monkeypatch.setattr(geo_module, "leer_configs_generales", lambda *a, **k: _CONFIG)

    h3_o = h3.latlng_to_cell(*_LATLNG_O, _H3_RES)
    h3_d = h3.latlng_to_cell(*_LATLNG_D, _H3_RES)
    lat_o, lon_o = h3.cell_to_latlng(h3_o)
    lat_d, lon_d = h3.cell_to_latlng(h3_d)
    lat_o, lon_o = lat_o + _OFFSET_GRADOS, lon_o + _OFFSET_GRADOS
    lat_d, lon_d = lat_d + _OFFSET_GRADOS, lon_d + _OFFSET_GRADOS

    # matriz O→D de estaciones (insumos): 30 min entre la estación 1 y la 2
    tts = pd.DataFrame({
        "id_o": [1], "id_d": [2],
        "id_linea_o": [7], "id_ramal_o": [None],
        "lat_o": [lat_o], "lon_o": [lon_o],
        "id_linea_d": [7], "id_ramal_d": [None],
        "lat_d": [lat_d], "lon_d": [lon_d],
        "travel_time_min": [30.0],
    })

    # lo que devuelve la query de etapas ⋈ travel_times_legs: distance_od = 10 km
    legs_df = pd.DataFrame({
        "dia": ["2024-01-01"],
        "id": [101],
        "id_linea": [7],
        "id_ramal": [None],
        "h3_o": [h3_o],
        "h3_d": [h3_d],
        "distance_od": [10.0],
    })

    ctx = type("Ctx", (), {})()
    ctx.data = _FakeData(legs_df)
    ctx.insumos = _FakeInsumos(tts)
    return ctx


def test_travel_speed_se_escribe_con_valor(escenario):
    """La causa 2: el segundo reindex tiraba kmh_od y dejaba travel_speed NULL."""
    from urbantrips.datamodel.legs import assign_stations_od

    assign_stations_od(escenario)

    escrito = escenario.data.written.get("travel_times_stations")
    assert escrito, "no se escribió nada en travel_times_stations"
    df = pd.concat(escrito, ignore_index=True)

    assert list(df.columns) == ["dia", "id", "travel_time_min", "travel_speed"]
    assert len(df) == 1
    assert df["travel_speed"].notna().all(), (
        "travel_speed quedó NULL: el kmh_od calculado se está perdiendo"
    )
    # 10 km en 30 min = 20 km/h
    assert df["travel_speed"].iloc[0] == pytest.approx(20.0)
    assert df["travel_time_min"].iloc[0] == pytest.approx(30.0)


def _sin_comentarios(sql: str) -> str:
    """Saca los comentarios `--` para no medir el SQL por lo que dicen las notas."""
    return "\n".join(
        linea.split("--")[0] for linea in sql.splitlines()
    )


def test_query_trae_distance_od_de_travel_times_legs(escenario):
    """La causa 1: distance_od no se seleccionaba y el reindex la creaba NaN."""
    from urbantrips.datamodel.legs import assign_stations_od

    assign_stations_od(escenario)

    sql_etapas = [q for q in escenario.data.queries if "FROM etapas" in q]
    assert sql_etapas, "no se consultó etapas"
    sql = _sin_comentarios(sql_etapas[0])
    assert "travel_times_legs" in sql, (
        "distance_od tiene que venir de travel_times_legs, que es donde vive"
    )
    assert "distance_od" in sql


def test_velocidad_imposible_se_anula(escenario, monkeypatch):
    """El cap de VELOCIDAD_MAXIMA_KMH se sigue aplicando sobre el valor real.

    Antes era imposible de verificar: travel_speed era NULL pasara lo que pasara.
    """
    from urbantrips.datamodel import legs as legs_module
    from urbantrips.datamodel.legs import assign_stations_od

    # 10 km en 30 min = 20 km/h; con el máximo en 10 km/h tiene que anularse
    monkeypatch.setattr(legs_module, "VELOCIDAD_MAXIMA_KMH", 10)
    assign_stations_od(escenario)

    df = pd.concat(escenario.data.written["travel_times_stations"], ignore_index=True)
    assert df["travel_speed"].isna().all()
    # el tiempo de viaje sobrevive aunque la velocidad se anule
    assert df["travel_time_min"].iloc[0] == pytest.approx(30.0)


def test_sin_distance_od_la_velocidad_queda_nula_pero_no_rompe(escenario):
    """Etapa sin fila en travel_times_legs → distance_od NULL por el LEFT JOIN."""
    from urbantrips.datamodel.legs import assign_stations_od

    escenario.data._legs["distance_od"] = np.nan
    assign_stations_od(escenario)

    df = pd.concat(escenario.data.written["travel_times_stations"], ignore_index=True)
    assert df["travel_speed"].isna().all()
    assert df["travel_time_min"].iloc[0] == pytest.approx(30.0)
