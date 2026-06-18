import pandas as pd

from urbantrips.datamodel.transactions import renombrar_columnas_tablas


def test_renombrar_columnas_tablas_keeps_all_unset_columns_as_null():
    """Multiple unset (None) source columns must each survive as their own
    NaN column. The old dict comprehension `{v: k for k, v in ...}` collided
    on repeated `v=None`, so only the *last* unset variable ended up in the
    output and all earlier ones (e.g. latitud/longitud) vanished entirely
    instead of appearing as NULL columns.
    """
    nombres_variables = {
        "id_trx": "ID_TRX",
        "fecha_trx": "FECHATRX",
        "id_tarjeta_trx": "NROTARJETA",
        "modo_trx": "modo",
        "hora_trx": None,
        "id_linea_trx": "IDLINEA",
        "id_ramal_trx": None,
        "interno_trx": "INTERNO",
        "orden_trx": None,
        "genero": None,
        "tarifa": None,
        "latitud_trx": None,
        "longitud_trx": None,
        "factor_expansion": None,
    }
    df = pd.DataFrame({
        "ID_TRX": [1],
        "FECHATRX": ["2022-01-01"],
        "NROTARJETA": [1],
        "modo": ["a"],
        "IDLINEA": [1],
        "INTERNO": [1],
    })

    result = renombrar_columnas_tablas(df, nombres_variables, postfijo="_trx")

    for col in [
        "hora", "id_ramal", "orden", "genero", "tarifa",
        "latitud", "longitud", "factor_expansion",
    ]:
        assert col in result.columns, f"{col} missing from result"
        assert result[col].isna().all()
