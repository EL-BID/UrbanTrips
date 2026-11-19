"""Transacciones de un equipo con dos vehiculos mezclados (`gps_con_saltos`).

En un tranvia, bajo el interno de un equipo con la traza mezclada reportan
validadoras que van en formaciones distintas. Cada transaccion se asigna a la
formacion que en ese instante esta mas cerca (`asignar_transacciones_mezcladas`)
y el equipo no cuenta como vehiculo.
"""

import numpy as np
import pandas as pd

from urbantrips.datamodel.gps_vehiculos import (
    COLUMNAS,
    COLUMNAS_TRX,
    MOTIVO_SALTOS,
    asignar_transacciones_mezcladas,
    mapear_interno_a_vehiculo,
)
from urbantrips.kpi.kpi_lineas import vehiculos_de_transacciones

DIA = "2025-10-29"
TRANVIA, BUS = 100, 7
LAT0, LON0 = -32.90, -68.85
LON_B = LON0 + 0.05  # otra formacion, a unos 4,7 km
T0 = 1_761_696_000
A, B, M = 20013, 20172, 20014  # dos formaciones y el equipo mezclado


def _traza(interno, lon, id_linea=TRANVIA, minutos=90):
    """Formacion que avanza a 30 km/h hacia el sur sobre el meridiano `lon`."""
    t = np.arange(0, minutos * 60, 60)
    return pd.DataFrame({
        "id_linea": id_linea, "interno": interno, "fecha": T0 + t,
        "latitud": LAT0 - t / 60 * 0.5 / 111.195, "longitud": lon,
    })


def _lat(minuto):
    return LAT0 - minuto * 0.5 / 111.195


def _vehiculos(id_linea=TRANVIA):
    filas = [
        (A, A, True, True, None),
        (B, B, True, True, None),
        (M, M, True, False, MOTIVO_SALTOS),
    ]
    return pd.DataFrame(
        [{"dia": DIA, "id_linea": id_linea, "interno": i, "interno_vehiculo": v,
          "es_principal": p, "gps_valido": g, "motivo": m}
         for i, v, p, g, m in filas],
        columns=COLUMNAS,
    )


def _trx(filas, id_linea=TRANVIA):
    """filas: (id, minuto, longitud) de transacciones del equipo mezclado."""
    return pd.DataFrame([
        {"dia": DIA, "id": i, "id_linea": id_linea, "id_ramal": 1, "interno": M,
         "fecha": T0 + minuto * 60 + 20, "latitud": _lat(minuto + 1 / 3),
         "longitud": lon}
        for i, minuto, lon in filas
    ])


def _gps(id_linea=TRANVIA):
    return pd.concat(
        [_traza(A, LON0, id_linea), _traza(B, LON_B, id_linea)], ignore_index=True
    )


def test_cada_transaccion_va_a_la_formacion_que_estaba_cerca():
    trx = _trx([(1, 10, LON0), (2, 20, LON0), (3, 30, LON_B), (4, 40, LON0 + 0.02)])

    out = asignar_transacciones_mezcladas(trx, _gps(), _vehiculos())

    assert list(out.columns) == COLUMNAS_TRX
    # la 4 esta a ~1,9 km de las dos formaciones: no se asigna
    assert dict(zip(out["id"], out["interno_vehiculo"])) == {1: A, 2: A, 3: B}


def test_solo_candidatos_en_servicio_en_ese_momento():
    trx = _trx([(1, 10, LON0), (2, 60, LON_B)])
    # B estacionada / sin servicio valido hasta el minuto 50
    servicios = pd.DataFrame({
        "id_linea": [TRANVIA, TRANVIA], "interno": [A, B],
        "min_ts": [T0, T0 + 50 * 60], "max_ts": [T0 + 90 * 60, T0 + 90 * 60],
    })
    solo_a = servicios[servicios["interno"] == A]

    out = asignar_transacciones_mezcladas(trx, _gps(), _vehiculos(), servicios=servicios)
    assert dict(zip(out["id"], out["interno_vehiculo"])) == {1: A, 2: B}

    out = asignar_transacciones_mezcladas(trx, _gps(), _vehiculos(), servicios=solo_a)
    assert dict(zip(out["id"], out["interno_vehiculo"])) == {1: A}


def test_lineas_sin_formaciones_no_se_asignan():
    trx = _trx([(1, 10, LON0)], id_linea=BUS)

    out = asignar_transacciones_mezcladas(
        trx, _gps(BUS), _vehiculos(BUS), lineas_con_formaciones={TRANVIA}
    )

    assert out.empty


def test_la_asignacion_por_transaccion_tiene_prioridad():
    legs = pd.DataFrame({
        "dia": DIA, "id": [1, 2, 3], "id_linea": TRANVIA, "interno": [M, M, 20012],
    })
    no_validos = pd.DataFrame(
        [{"dia": DIA, "id_linea": TRANVIA, "interno": 20012, "interno_vehiculo": A,
          "gps_valido": False, "motivo": "equipo_secundario"},
         {"dia": DIA, "id_linea": TRANVIA, "interno": M, "interno_vehiculo": M,
          "gps_valido": False, "motivo": MOTIVO_SALTOS}],
        columns=COLUMNAS,
    )
    trx_vehiculos = pd.DataFrame(
        [{"dia": DIA, "id": 1, "id_linea": TRANVIA, "interno": M, "interno_vehiculo": B}],
        columns=COLUMNAS_TRX,
    )

    out = mapear_interno_a_vehiculo(legs, no_validos, trx_vehiculos)

    # 1 reasignada, 2 sin asignar sigue en el equipo, 3 secundario de A
    assert out["interno"].tolist() == [B, M, A]
    assert out["interno"].dtype == legs["interno"].dtype


def test_el_equipo_mezclado_no_cuenta_como_vehiculo_en_modos_guiados():
    trx = pd.DataFrame({
        "dia": DIA, "id": [1, 2, 3, 4], "id_linea": [TRANVIA] * 3 + [BUS],
        "id_ramal": 1, "interno": [M, M, A, 55],
    })
    con_gps = pd.DataFrame({
        "dia": DIA, "id_linea": [TRANVIA] * 3 + [BUS], "interno": [M, A, B, 55],
    })
    no_validos = pd.DataFrame(
        [{"dia": DIA, "id_linea": l, "interno": i, "interno_vehiculo": i,
          "gps_valido": False, "motivo": MOTIVO_SALTOS}
         for l, i in [(TRANVIA, M), (BUS, 55)]],
        columns=COLUMNAS,
    )
    trx_vehiculos = pd.DataFrame(
        [{"dia": DIA, "id": 1, "id_linea": TRANVIA, "interno": M, "interno_vehiculo": B}],
        columns=COLUMNAS_TRX,
    )

    out = vehiculos_de_transacciones(trx, con_gps, [TRANVIA], no_validos, trx_vehiculos)

    # tranvia: la 1 pasa a B, la 2 (sin asignar) se descarta, la 3 es A.
    # colectivo: un interno mezclado sigue siendo un vehiculo.
    por_linea = out.groupby("id_linea")["interno"].agg(set).to_dict()
    assert por_linea == {TRANVIA: {A, B}, BUS: {55}}
