"""Que vehiculo es cada interno del GPS (`datamodel/gps_vehiculos.py`).

En una formacion (tranvia, tren) cada equipo reporta como un interno distinto:
se agrupan los que se mueven juntos y se cuenta uno solo. Un interno con dos
vehiculos mezclados alterna entre posiciones lejanas y su traza no mide oferta.
"""

import numpy as np
import pandas as pd

from urbantrips.datamodel import services
from urbantrips.datamodel.gps_vehiculos import (
    detectar_vehiculos_gps,
    guardar_vehiculos_gps,
    leer_gps_no_validos,
    mapear_interno_a_vehiculo,
    quitar_gps_no_validos,
)

DIA = "2025-10-29"
LAT0, LON0 = -32.90, -68.85
T0 = 1_761_696_000  # 2025-10-29 00:00 UTC


def _traza(interno, inicio=0, minutos=90, cada=60, desfase=0, lon=LON0, id_linea=100):
    """Un vehiculo que avanza a 30 km/h hacia el sur sobre el meridiano `lon`."""
    t = np.arange(inicio * 60 + desfase, (inicio + minutos) * 60, cada)
    # 30 km/h = 0.5 km por minuto; un grado de latitud son 111.195 km
    lat = LAT0 - (t - inicio * 60) / 60 * 0.5 / 111.195
    return pd.DataFrame(
        {
            "dia": DIA, "id_linea": id_linea, "interno": interno, "fecha": T0 + t,
            "latitud": lat, "longitud": lon, "distance_km": 0.5 * cada / 60,
        }
    )


def _detectar(*trazas):
    out = detectar_vehiculos_gps(pd.concat(trazas, ignore_index=True), 100.0)
    return out.set_index("interno")


def test_equipos_de_una_formacion_son_un_vehiculo():
    # tres equipos del mismo tranvia, que reportan en segundos distintos
    out = _detectar(
        _traza(20011), _traza(20012, desfase=17), _traza(20013, desfase=41, cada=240)
    )
    assert out["interno_vehiculo"].nunique() == 1
    assert (out["n_equipos"] == 3).all()
    assert int(out["es_principal"].sum()) == 1
    assert int(out["gps_valido"].sum()) == 1
    assert set(out.loc[~out["gps_valido"], "motivo"]) == {"equipo_secundario"}


def test_el_principal_es_el_de_mas_km():
    # el segundo equipo estuvo apagado la primera media hora
    segundo = _traza(2, desfase=10)
    out = _detectar(_traza(1), segundo[segundo["fecha"] >= T0 + 30 * 60])
    assert out.loc[1, "es_principal"] and not out.loc[2, "es_principal"]
    assert out.loc[2, "interno_vehiculo"] == 1


def test_dos_colectivos_de_la_misma_linea_no_se_agrupan():
    # mismo recorrido, uno sale 10 minutos despues: van a 5 km de distancia
    out = _detectar(_traza(1), _traza(2, inicio=10))
    assert (out["n_equipos"] == 1).all()
    assert out["gps_valido"].all()
    assert out["motivo"].isna().all()


def test_colectivos_que_van_juntos_solo_un_rato_no_se_agrupan():
    # comparten 30 minutos de recorrido y despues cada uno sigue por su lado
    juntos = _traza(2, minutos=30, desfase=20)
    separado = _traza(2, inicio=30, minutos=60, desfase=20, lon=LON0 + 0.1)
    out = _detectar(_traza(1), pd.concat([juntos, separado]))
    assert (out["n_equipos"] == 1).all()


def test_parados_en_la_misma_terminal_no_se_agrupan():
    parado = _traza(1).assign(latitud=LAT0, distance_km=0.0)
    out = _detectar(parado, parado.assign(interno=2, fecha=parado["fecha"] + 15))
    assert (out["n_equipos"] == 1).all()


def test_solo_se_agrupa_en_las_lineas_con_formaciones():
    # mismas trazas en un tranvia (linea 100) y en un colectivo (linea 200): dos
    # colectivos que van pegados todo el recorrido son dos vehiculos
    gps = pd.concat(
        [_traza(1), _traza(2, desfase=20),
         _traza(1, id_linea=200), _traza(2, desfase=20, id_linea=200)]
    )
    out = detectar_vehiculos_gps(gps, 100.0, lineas_con_formaciones={100})
    assert (out.loc[out.id_linea == 100, "n_equipos"] == 2).all()
    colectivos = out[out.id_linea == 200]
    assert (colectivos["n_equipos"] == 1).all() and colectivos["gps_valido"].all()


def test_los_saltos_se_marcan_en_todos_los_modos():
    mezclado = pd.concat([_traza(7), _traza(7, desfase=30, lon=LON0 + 0.1)])
    out = detectar_vehiculos_gps(mezclado, 100.0, lineas_con_formaciones=set())
    assert out["motivo"].tolist() == ["gps_con_saltos"]


def test_guardar_agrupa_segun_el_modo_de_la_linea():
    guardado = {}

    class _Ctx:
        class data:
            @staticmethod
            def execute(q):
                pass

            @staticmethod
            def append_raw(df, tabla):
                guardado[tabla] = df

        class insumos:
            @staticmethod
            def get_metadata_lineas():
                return pd.DataFrame({"id_linea": [100, 200], "modo": ["tranvia", "autobus"]})

    gps = pd.concat(
        [_traza(1), _traza(2, desfase=20),
         _traza(1, id_linea=200), _traza(2, desfase=20, id_linea=200)]
    )
    guardar_vehiculos_gps(_Ctx, gps, DIA, 100.0)
    out = guardado["gps_vehiculos"]
    assert out.groupby("id_linea")["gps_valido"].sum().to_dict() == {100: 1, 200: 2}


def test_internos_iguales_de_lineas_distintas_no_se_agrupan():
    out = detectar_vehiculos_gps(
        pd.concat([_traza(1, id_linea=100), _traza(1, id_linea=200)]), 100.0
    )
    assert (out["n_equipos"] == 1).all() and len(out) == 2


def test_interno_con_dos_vehiculos_mezclados_no_es_valido():
    # dos vehiculos a 10 km reportan con el mismo interno
    mezclado = pd.concat([_traza(7), _traza(7, desfase=30, lon=LON0 + 0.1)])
    out = _detectar(mezclado, _traza(8, inicio=200))
    assert out.loc[7, "prop_saltos"] > 0.9
    assert not out.loc[7, "gps_valido"] and out.loc[7, "motivo"] == "gps_con_saltos"
    assert out.loc[8, "gps_valido"]


def test_interno_con_saltos_no_se_agrupa_con_la_formacion():
    mezclado = pd.concat([_traza(3, desfase=5), _traza(3, desfase=35, lon=LON0 + 0.1)])
    out = _detectar(_traza(1), _traza(2, desfase=20), mezclado)
    assert out.loc[2, "interno_vehiculo"] == out.loc[1, "interno_vehiculo"]
    assert out.loc[3, "n_equipos"] == 1 and out.loc[3, "motivo"] == "gps_con_saltos"


def test_sin_puntos_devuelve_vacio():
    vacio = pd.DataFrame(
        columns=["dia", "id_linea", "interno", "fecha", "latitud", "longitud", "distance_km"]
    )
    assert len(detectar_vehiculos_gps(vacio, 100.0)) == 0


def _vehiculos():
    return _detectar(_traza(1), _traza(2, desfase=20), _traza(9, inicio=300)).reset_index()


def test_mapear_interno_a_vehiculo():
    veh = _vehiculos()
    principal = int(veh.loc[veh.interno == 1, "interno_vehiculo"].iloc[0])
    etapas = pd.DataFrame(
        {"dia": [DIA] * 4 + ["2025-10-30"], "id_linea": [100, 100, 100, 200, 100],
         "interno": [1, 2, 9, 2, 2]}
    )
    out = mapear_interno_a_vehiculo(etapas, veh[~veh.gps_valido])
    # los dos equipos van al principal; otro vehiculo, otra linea y otro dia no cambian
    assert out["interno"].tolist() == [principal, principal, 9, 2, 2]
    assert out["interno"].dtype == etapas["interno"].dtype
    assert etapas["interno"].tolist() == [1, 2, 9, 2, 2]


def test_quitar_gps_no_validos():
    veh = _vehiculos()
    secundario = int(veh.loc[~veh.es_principal, "interno"].iloc[0])
    gps = pd.DataFrame({"dia": DIA, "id_linea": 100, "interno": [1, 2, 9, 9]})
    out = quitar_gps_no_validos(gps, veh[~veh.gps_valido])
    assert secundario not in out["interno"].tolist() and len(out) == 3


def test_servicios_de_un_equipo_secundario_no_son_validos():
    veh = _vehiculos()
    secundario = int(veh.loc[~veh.es_principal, "interno"].iloc[0])
    principal = int(veh.loc[veh.interno == 1, "interno_vehiculo"].iloc[0])
    servicios = pd.DataFrame(
        {"dia": DIA, "id_linea": 100, "interno": [principal, secundario, 9],
         "valid": [True, True, False],
         "motivo_no_valido": [None, None, "detenido"]}
    )
    out = services.invalidar_servicios_sin_vehiculo(servicios, veh[~veh.gps_valido])
    assert out["valid"].tolist() == [True, False, False]
    assert out["motivo_no_valido"].tolist() == [None, "equipo_secundario", "detenido"]
    assert list(out.columns) == list(servicios.columns)
    # sin tabla de vehiculos no cambia nada
    assert services.invalidar_servicios_sin_vehiculo(servicios, None) is servicios


class _CtxSinTabla:
    class data:
        @staticmethod
        def query(q):
            raise NotImplementedError("el adapter no resuelve consultas con filtro")


def test_leer_gps_no_validos_sin_tabla_devuelve_vacio():
    out = leer_gps_no_validos(_CtxSinTabla())
    assert len(out) == 0 and "interno_vehiculo" in out.columns


def test_tiempos_de_viaje_no_usan_el_gps_de_internos_con_saltos():
    """Las etapas de un interno con dos vehiculos mezclados quedan sin ancla GPS
    de destino (como las de un vehiculo sin GPS); los equipos secundarios no."""
    from urbantrips.datamodel import legs

    gps = pd.DataFrame(
        {"id": [1, 2, 3, 4], "dia": DIA, "id_linea": 100, "interno": [1, 2, 7, 7]}
    )
    no_validos = pd.DataFrame(
        {"dia": DIA, "id_linea": 100, "interno": [2, 7], "interno_vehiculo": [1, 7],
         "motivo": ["equipo_secundario", "gps_con_saltos"]}
    )
    consultas = []

    class _Ctx:
        class data:
            @staticmethod
            def query(q):
                consultas.append(q)
                if "gps_vehiculos" in q:
                    return no_validos
                if "legs_to_gps_origin" in q:
                    return pd.DataFrame({"id_legs": [], "id_gps_o": []})
                return gps

    out, _ = legs._fetch_time_distance_inputs_dia(_Ctx, DIA, "2025-10-30")
    assert out["interno"].tolist() == [1, 2]
    assert any("'2025-10-29', '2025-10-30'" in q for q in consultas if "gps_vehiculos" in q)
