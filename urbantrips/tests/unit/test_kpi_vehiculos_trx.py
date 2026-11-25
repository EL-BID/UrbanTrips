"""Conteo de vehiculos con transacciones (`cant_internos_en_trx`)."""

import pandas as pd

from urbantrips.datamodel.gps_vehiculos import COLUMNAS
from urbantrips.kpi.kpi_lineas import vehiculos_de_transacciones

DIA = "2025-10-29"
TRANVIA, BUS = 100, 7


def _trx(filas):
    return pd.DataFrame(filas, columns=["dia", "id_linea", "id_ramal", "interno"])


def _no_validos(filas):
    """Filas de gps_vehiculos: (id_linea, interno, interno_vehiculo)."""
    df = pd.DataFrame(
        [
            {"dia": DIA, "id_linea": l, "interno": i, "interno_vehiculo": v,
             "gps_valido": False, "motivo": "equipo_secundario"}
            for l, i, v in filas
        ],
        columns=COLUMNAS,
    )
    return df


def _contar(trx):
    return trx.groupby("id_linea")["interno"].nunique().to_dict()


def test_validadoras_de_una_formacion_cuentan_como_un_vehiculo():
    trx = _trx([(DIA, TRANVIA, 1, i) for i in (20101, 20102, 20103, 20111)])
    con_gps = trx[["dia", "id_linea", "interno"]]
    no_validos = _no_validos([(TRANVIA, 20102, 20101), (TRANVIA, 20103, 20101)])

    out = vehiculos_de_transacciones(trx, con_gps, [TRANVIA], no_validos)

    assert _contar(out) == {TRANVIA: 2}


def test_modo_guiado_sin_gps_no_es_vehiculo_y_colectivo_si():
    trx = _trx([
        (DIA, TRANVIA, 1, 20101),   # a bordo, con GPS
        (DIA, TRANVIA, 1, 90808),   # validadora de estacion, sin GPS
        (DIA, BUS, 1, 55),          # colectivo con GPS
        (DIA, BUS, 1, 56),          # colectivo sin GPS: sigue siendo vehiculo
    ])
    con_gps = pd.DataFrame(
        {"dia": [DIA, DIA], "id_linea": [TRANVIA, BUS], "interno": [20101, 55]}
    )

    out = vehiculos_de_transacciones(trx, con_gps, [TRANVIA], _no_validos([]))

    assert _contar(out) == {TRANVIA: 1, BUS: 2}


def test_gps_de_otro_dia_no_cuenta():
    trx = _trx([(DIA, TRANVIA, 1, 20101)])
    con_gps = pd.DataFrame(
        {"dia": ["2025-10-30"], "id_linea": [TRANVIA], "interno": [20101]}
    )

    out = vehiculos_de_transacciones(trx, con_gps, [TRANVIA], _no_validos([]))

    assert out.empty


def test_tipos_distintos_entre_tablas_se_cruzan_igual():
    trx = _trx([(DIA, TRANVIA, 1, 20101)])
    con_gps = pd.DataFrame(
        {"dia": [DIA], "id_linea": ["100"], "interno": [20101.0]}
    )

    out = vehiculos_de_transacciones(trx, con_gps, [TRANVIA], _no_validos([]))

    assert len(out) == 1
