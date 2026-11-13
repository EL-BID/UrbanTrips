# -*- coding: utf-8 -*-
"""Tests de la agregación de indicadores línea-día -> total de sistema.

Cubren los defectos encontrados el 2026-08-20 en la página de indicadores
operativos, que hasta entonces no tenían regresión porque la lógica vivía dentro
del script de Streamlit.
"""

import numpy as np
import pandas as pd
import pytest

from urbantrips.kpi.agregacion_indicadores import (
    filas_promedios,
    NOMBRE_DIAS,
    PERIOD_SUM_COLS,
    RATIO_AGG,
    _harmonic,
    _wmean,
    agregar_sistema,
    etiqueta_dia,
    preparar,
    tipo_de_dia,
)


def _linea(dia, id_linea, *, veh=10, trx=1000, km=100.0, ipk=None, vel=20.0,
           internos_trx=None, **extra):
    fila = {
        "dia": dia,
        "id_linea": id_linea,
        "vehiculos_operativos": veh,
        "cant_internos_en_trx": internos_trx if internos_trx is not None else veh,
        "transacciones": trx,
        "tot_km_route": km,
        "ipk_route": (trx / km) if ipk is None and km else ipk,
        "velocidad_comercial_route": vel,
        "distancia_media_veh_route": (km / veh) if veh else np.nan,
    }
    for c in ("Masculino", "Femenino", "No informado",
              "sin_descuento", "tarifa_social", "educacion_jubilacion"):
        fila[c] = trx / 6
    fila.update(extra)
    return fila


# ---------------------------------------------------------------------------
# extensivas: promedio de totales diarios vs suma del período
# ---------------------------------------------------------------------------


def test_promedio_diario_es_el_promedio_de_los_totales_no_de_las_filas():
    """El bug original: promediar las filas línea-día da el promedio POR LÍNEA,
    no el total diario del sistema."""
    df = pd.DataFrame([
        _linea("2026-03-09", 1, veh=10, trx=100),
        _linea("2026-03-09", 2, veh=20, trx=200),
        _linea("2026-03-10", 1, veh=30, trx=300),
        _linea("2026-03-10", 2, veh=40, trx=400),
    ])
    agg = agregar_sistema(df, promedio_diario=True)
    # totales diarios: 30 y 70 -> promedio 50 (la media de las filas daría 25)
    assert agg["vehiculos_operativos"].iloc[0] == 50
    assert agg["transacciones"].iloc[0] == 500


def test_suma_del_periodo_es_la_suma_de_todos_los_dias():
    df = pd.DataFrame([
        _linea("2026-03-09", 1, veh=10, trx=100),
        _linea("2026-03-10", 1, veh=30, trx=300),
    ])
    agg = agregar_sistema(df, promedio_diario=False)
    assert agg["vehiculos_operativos"].iloc[0] == 40
    assert agg["transacciones"].iloc[0] == 400


def test_acumulado_sobre_promedio_da_la_cantidad_de_dias():
    dias = [f"2026-03-{d:02d}" for d in range(9, 16)]
    df = pd.DataFrame([_linea(d, 1, trx=100) for d in dias])
    prom = agregar_sistema(df, promedio_diario=True)["transacciones"].iloc[0]
    acum = agregar_sistema(df, promedio_diario=False)["transacciones"].iloc[0]
    assert acum / prom == pytest.approx(len(dias))


def test_los_vehiculos_no_se_acumulan_en_el_periodo():
    """Sumar vehículos entre días da vehículo-día, no vehículos."""
    assert "vehiculos_operativos" not in PERIOD_SUM_COLS
    assert "cant_internos_en_trx" not in PERIOD_SUM_COLS
    assert "transacciones" in PERIOD_SUM_COLS
    assert "tot_km_route" in PERIOD_SUM_COLS


def test_columna_toda_nula_da_nan_y_no_cero():
    """Un modo sin GPS no 'operó cero vehículos': no hay dato."""
    df = pd.DataFrame([
        _linea("2026-03-09", 1, veh=np.nan, km=np.nan),
        _linea("2026-03-09", 2, veh=np.nan, km=np.nan),
    ])
    agg = agregar_sistema(df, promedio_diario=True)
    assert pd.isna(agg["vehiculos_operativos"].iloc[0])
    assert pd.isna(agg["tot_km_route"].iloc[0])
    assert pd.isna(agg["cobertura_gps"].iloc[0])
    # pero la demanda sí es un dato real
    assert agg["transacciones"].iloc[0] > 0


# ---------------------------------------------------------------------------
# intensivas: ponderar por el denominador
# ---------------------------------------------------------------------------


def test_ratio_ponderado_por_denominador_es_el_ratio_agregado():
    """Σ(R·B)/ΣB == ΣA/ΣB. Es la identidad que justifica la regla."""
    df = pd.DataFrame([
        _linea("2026-03-09", 1, trx=1000, km=100.0),   # ipk 10
        _linea("2026-03-09", 2, trx=300, km=300.0),    # ipk 1
    ])
    agg = agregar_sistema(df, promedio_diario=True)
    esperado = (1000 + 300) / (100.0 + 300.0)          # 3.25
    assert agg["ipk_route"].iloc[0] == pytest.approx(esperado)


def test_lineas_sin_el_denominador_no_contaminan_el_ratio():
    """Una línea sin km (subte/tren) entra con peso 0, no arrastra su IPK."""
    df = pd.DataFrame([
        _linea("2026-03-09", 1, trx=1000, km=100.0),                 # ipk 10
        _linea("2026-03-09", 2, trx=5000, km=np.nan, ipk=np.inf),    # sin GPS
    ])
    agg = agregar_sistema(df, promedio_diario=True)
    assert agg["ipk_route"].iloc[0] == pytest.approx(10.0)


def test_ratio_no_se_promedia_sin_ponderar():
    """La media simple de los ratios difiere del ratio agregado: el test fija
    que se usa el ponderado."""
    df = pd.DataFrame([
        _linea("2026-03-09", 1, trx=1000, km=100.0),   # ipk 10
        _linea("2026-03-09", 2, trx=300, km=300.0),    # ipk 1
    ])
    agg = agregar_sistema(df, promedio_diario=True)
    media_simple = (10 + 1) / 2
    assert agg["ipk_route"].iloc[0] != pytest.approx(media_simple)


def test_velocidad_comercial_usa_media_armonica():
    """Σkm/Σhoras, no el promedio ponderado por km (que sesga hacia las rápidas)."""
    df = pd.DataFrame([
        _linea("2026-03-09", 1, km=100.0, vel=10.0),   # 10 h
        _linea("2026-03-09", 2, km=100.0, vel=50.0),   # 2 h
    ])
    agg = agregar_sistema(df, promedio_diario=True)
    esperado = 200.0 / (100 / 10 + 100 / 50)           # 200/12 = 16.67
    assert agg["velocidad_comercial_route"].iloc[0] == pytest.approx(esperado)
    lineal = (100 * 10 + 100 * 50) / 200               # 30.0
    assert agg["velocidad_comercial_route"].iloc[0] < lineal


def test_cobertura_es_cociente_de_sumas_no_suma_de_cocientes():
    df = pd.DataFrame([
        _linea("2026-03-09", 1, veh=10, internos_trx=10),    # 100 %
        _linea("2026-03-09", 2, veh=10, internos_trx=90),    # 11 %
    ])
    agg = agregar_sistema(df, promedio_diario=True)
    assert agg["cobertura_gps"].iloc[0] == pytest.approx(20 / 100)


def test_los_indicadores_de_oferta_se_ponderan_por_oferta():
    """Decisión explícita del usuario: nunca por transacciones."""
    for col in ("velocidad_comercial_route", "ipk_route", "fo_mean_route",
                "kvd_route", "pvd", "distancia_media_veh_route"):
        peso, _ = RATIO_AGG[col]
        assert peso in ("tot_km_route", "tot_km_route_gps", "vehiculos_operativos"), (
            f"{col} se pondera por {peso}, que no es una magnitud de oferta"
        )


def test_los_indicadores_del_pasajero_se_ponderan_por_transacciones():
    for col in ("dmt_mean_od", "dmt_median_od", "travel_time_min", "kmh_od"):
        assert RATIO_AGG[col][0] == "transacciones"


def test_wmean_y_harmonic_devuelven_nan_sin_datos():
    vacio = pd.DataFrame({"a": [np.nan], "w": [0.0]})
    assert pd.isna(_wmean(vacio, "a", "w"))
    assert pd.isna(_harmonic(vacio, "a", "w"))
    assert pd.isna(_wmean(vacio, "no_existe", "w"))


# ---------------------------------------------------------------------------
# tipo de día
# ---------------------------------------------------------------------------


def test_tipo_de_dia_clasifica_la_semana():
    dias = pd.Series([
        "2026-03-09", "2026-03-13",   # lunes, viernes
        "2026-03-14", "2026-03-15",   # sábado, domingo
        "Promedios",
    ])
    assert tipo_de_dia(dias).tolist() == [
        "Día hábil", "Día hábil", "Sábado", "Domingo", "",
    ]


def test_etiqueta_dia_agrega_el_dia_de_la_semana():
    assert etiqueta_dia("2026-03-14") == "2026-03-14 (sábado)"
    assert etiqueta_dia("2026-03-09") == "2026-03-09 (lunes)"
    assert etiqueta_dia("Promedios") == "Promedios"


def test_nombres_de_dias_tienen_singular_y_plural():
    for tipo, (sing, plur) in NOMBRE_DIAS.items():
        assert sing and plur and sing != plur


# ---------------------------------------------------------------------------
# preparar()
# ---------------------------------------------------------------------------


def test_preparar_deriva_cobertura_y_tipo_de_dia():
    df = pd.DataFrame([
        _linea("2026-03-14", 1, veh=8, internos_trx=10),
    ])
    out = preparar(df)
    assert out["cobertura_gps"].iloc[0] == pytest.approx(0.8)
    assert out["tipo_dia"].iloc[0] == "Sábado"


def test_preparar_reconstruye_km_gps_solo_si_falta():
    base = _linea("2026-03-09", 1, veh=10)
    base["distancia_media_veh_route_gps"] = 25.0

    out = preparar(pd.DataFrame([base]))
    assert out["tot_km_route_gps"].iloc[0] == pytest.approx(250.0)

    conservar = dict(base, tot_km_route_gps=999.0)
    out2 = preparar(pd.DataFrame([conservar]))
    assert out2["tot_km_route_gps"].iloc[0] == 999.0


def test_preparar_tolera_df_vacio():
    assert preparar(pd.DataFrame()).empty
    assert preparar(None).empty


# --- fila "Promedios" por entidad y promedio por tipo de día -----------------


def _dias_de_una_linea(**extra):
    """Una linea en dos dias: el lunes con 1000 pasajeros en 100 km y el sabado
    con 100 pasajeros en 100 km (ipk 10 y 1)."""
    base = {
        "dia": ["2024-01-01", "2024-01-06"], "mes": "2024-01",
        "id_linea": "1", "nombre_linea": "L1", "empresa": "E", "modo": "autobus",
        "transacciones": [1000, 100], "vehiculos_operativos": [5.0, 5.0],
        "cant_internos_en_trx": [5.0, 5.0], "tot_km_route": [100.0, 100.0],
        "ipk_route": [10.0, 1.0], "velocidad_comercial_am_route": [20.0, 10.0],
    }
    base.update(extra)
    return pd.DataFrame(base)


CLAVES = ["id_linea", "nombre_linea", "empresa", "modo"]


def test_filas_promedios_calcula_el_cociente_de_los_totales_y_no_el_promedio_de_cocientes():
    # dias de 1000 y 100 pasajeros sobre 100 km cada uno: 1100 / 200 = 5,5. El
    # promedio de los cocientes (10 y 1) daria lo mismo solo por casualidad de
    # km iguales; con km distintos se separan
    df = _dias_de_una_linea(tot_km_route=[100.0, 400.0], ipk_route=[10.0, 0.25])
    out = filas_promedios(df, CLAVES).iloc[0]
    assert out["dia"] == "Promedios" and out["mes"] == ""
    assert out["transacciones"] == pytest.approx(550)
    assert out["tot_km_route"] == pytest.approx(250)
    assert out["ipk_route"] == pytest.approx(1100 / 500)       # y no (10 + 0,25) / 2
    # lo que no se agrega queda como media simple
    assert out["velocidad_comercial_am_route"] == pytest.approx(15.0)


def test_filas_promedios_no_mezcla_dias_con_y_sin_oferta():
    # el sabado la linea quedo sin oferta (cobertura GPS bajo el umbral)
    df = _dias_de_una_linea(
        tot_km_route=[100.0, np.nan], ipk_route=[10.0, np.nan],
        vehiculos_operativos=[5.0, np.nan],
    )
    out = filas_promedios(df, CLAVES).iloc[0]
    # la demanda promedia los mismos dias que la oferta: solo el lunes
    assert out["transacciones"] == pytest.approx(1000)
    assert out["tot_km_route"] == pytest.approx(100)
    assert out["ipk_route"] == pytest.approx(10.0)


def test_filas_promedios_una_entidad_que_nunca_tiene_oferta_promedia_todos_sus_dias():
    df = _dias_de_una_linea(
        tot_km_route=[np.nan, np.nan], ipk_route=[np.nan, np.nan],
        vehiculos_operativos=[np.nan, np.nan],
    )
    out = filas_promedios(df, CLAVES).iloc[0]
    assert out["transacciones"] == pytest.approx(550)
    assert np.isnan(out["tot_km_route"])


def test_filas_promedios_ignora_una_fila_promedios_anterior():
    df = _dias_de_una_linea()
    previa = df.iloc[[0]].assign(dia="Promedios", transacciones=99999)
    out = filas_promedios(pd.concat([df, previa], ignore_index=True), CLAVES)
    assert len(out) == 1 and out.iloc[0]["transacciones"] == pytest.approx(550)


def test_el_total_diario_con_todas_las_lineas_sin_dato_no_cuenta_como_cero():
    # el martes ninguna linea tiene km: es "sin dato", no un dia de 0 km
    df = pd.DataFrame({
        "dia": ["2024-01-01", "2024-01-02"], "id_linea": ["1", "1"],
        "tot_km_route": [100.0, np.nan], "transacciones": [10, 10],
    })
    agg = agregar_sistema(df, promedio_diario=True)
    assert agg["tot_km_route"].iloc[0] == pytest.approx(100.0)
    assert agg["transacciones"].iloc[0] == pytest.approx(10)


def test_filas_promedios_da_lo_mismo_que_agregar_sistema_entidad_por_entidad():
    rng = np.random.default_rng(7)
    dias = ["2024-01-01", "2024-01-02", "2024-01-06"]
    filas = []
    for linea in ("1", "2", "3"):
        for dia in dias:
            km = rng.uniform(50, 500)
            filas.append({
                "dia": dia, "mes": "2024-01", "id_linea": linea, "nombre_linea": linea,
                "empresa": "E", "modo": "autobus",
                "transacciones": rng.integers(100, 5000),
                "vehiculos_operativos": float(rng.integers(2, 20)),
                "cant_internos_en_trx": float(rng.integers(5, 25)),
                "tot_km_route": km, "ipk_route": rng.uniform(1, 8),
                "velocidad_comercial_route": rng.uniform(8, 30),
                "pvd": rng.uniform(100, 900),
                "cobertura_gps": rng.uniform(0.5, 1.0),
            })
    df = pd.DataFrame(filas)
    df.loc[(df.id_linea == "2") & (df.dia == "2024-01-06"), ["tot_km_route", "ipk_route"]] = np.nan
    df.loc[df.id_linea == "3", ["tot_km_route", "ipk_route", "velocidad_comercial_route"]] = np.nan

    out = filas_promedios(df, CLAVES).set_index("id_linea")
    for linea, g in df.groupby("id_linea"):
        g = g[g.tot_km_route.notna()] if g.tot_km_route.notna().any() else g
        esperado = agregar_sistema(preparar(g), promedio_diario=True).iloc[0]
        for col in ("transacciones", "tot_km_route", "ipk_route",
                    "velocidad_comercial_route", "pvd", "cobertura_gps"):
            if pd.isna(esperado[col]):
                assert pd.isna(out.loc[linea, col]), (linea, col)
            else:
                assert out.loc[linea, col] == pytest.approx(esperado[col]), (linea, col)


def test_filas_promedios_con_claves_vacias_y_con_tabla_vacia():
    # una linea sin empresa (clave vacia): no rompe el filtro de dias con oferta
    df = _dias_de_una_linea(empresa=[None, None])
    out = filas_promedios(df, CLAVES)
    assert len(out) == 0 or out["transacciones"].notna().all()
    # solo la fila Promedios: nada que promediar
    solo_promedios = _dias_de_una_linea().assign(dia="Promedios")
    vacio = filas_promedios(solo_promedios, CLAVES)
    assert vacio.empty and {"dia", "mes", *CLAVES} <= set(vacio.columns)


def test_filas_promedios_redondea_tambien_las_columnas_anulables():
    df = _dias_de_una_linea(transacciones=pd.array([1000, 101], dtype="Int64"))
    out = filas_promedios(df, CLAVES, decimales=1)
    assert out["transacciones"].iloc[0] == pytest.approx(550.5)
