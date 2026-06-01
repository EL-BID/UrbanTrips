import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def _make_gps_csv() -> str:
    """GPS CSV with canonical columns + one junk column."""
    return "\n".join([
        "id_gps,lat,lon,id_linea_gps,id_ramal_gps,interno_gps,fecha_gps,vel,JUNK_COL",
        "1,-34.6,-58.5,10,1,101,2024-01-01 08:00:00,30,garbage",
        "2,-34.7,-58.6,10,1,102,2024-01-01 08:01:00,25,garbage",
    ])


_NOMBRES_VARIABLES_GPS = {
    "id": "id_gps",
    "latitud": "lat",
    "longitud": "lon",
    "id_linea": "id_linea_gps",
    "id_ramal": "id_ramal_gps",
    "interno": "interno_gps",
    "fecha": "fecha_gps",
    "velocity": "vel",
}

_CONFIGS = {
    "lineas_contienen_ramales": True,
    "resolucion_h3": 8,
    "valor_inicio_servicio": "start_service",
    "valor_fin_servicio": "finish_service",
}


def test_gps_read_passes_usecols_to_read_csv(tmp_path):
    """process_and_upload_gps_table must pass usecols to read_csv."""
    from urbantrips.datamodel.transactions import process_and_upload_gps_table

    csv_path = tmp_path / "gps.csv"
    csv_path.write_text(_make_gps_csv())

    ctx = MagicMock()
    ctx.data.get_run_days.return_value = pd.DataFrame({"dia": ["2024-01-01"]})

    read_csv_kwargs = {}
    original_read_csv = pd.read_csv

    def spy_read_csv(path, **kwargs):
        read_csv_kwargs.update(kwargs)
        return original_read_csv(path, **kwargs)

    with patch("urbantrips.datamodel.transactions.pd.read_csv", side_effect=spy_read_csv), \
         patch("urbantrips.datamodel.transactions.leer_configs_generales", return_value=_CONFIGS), \
         patch("urbantrips.datamodel.transactions.eliminar_trx_fuera_bbox", side_effect=lambda df, **kw: df), \
         patch("urbantrips.datamodel.transactions.geo"):
        try:
            process_and_upload_gps_table(
                ctx,
                nombre_archivo_gps=str(csv_path),
                nombres_variables_gps=dict(_NOMBRES_VARIABLES_GPS),
                formato_fecha="%Y-%m-%d %H:%M:%S",
            )
        except Exception:
            pass  # downstream steps may fail; we only care about the CSV read

    assert "usecols" in read_csv_kwargs, "GPS read_csv is missing usecols parameter"


def test_gps_read_usecols_excludes_junk_columns(tmp_path):
    """usecols must accept needed cols and reject columns not in nombres_variables_gps."""
    from urbantrips.datamodel.transactions import process_and_upload_gps_table

    csv_path = tmp_path / "gps.csv"
    csv_path.write_text(_make_gps_csv())

    ctx = MagicMock()
    ctx.data.get_run_days.return_value = pd.DataFrame({"dia": ["2024-01-01"]})

    read_csv_kwargs = {}
    original_read_csv = pd.read_csv

    def spy_read_csv(path, **kwargs):
        read_csv_kwargs.update(kwargs)
        return original_read_csv(path, **kwargs)

    with patch("urbantrips.datamodel.transactions.pd.read_csv", side_effect=spy_read_csv), \
         patch("urbantrips.datamodel.transactions.leer_configs_generales", return_value=_CONFIGS), \
         patch("urbantrips.datamodel.transactions.eliminar_trx_fuera_bbox", side_effect=lambda df, **kw: df), \
         patch("urbantrips.datamodel.transactions.geo"):
        try:
            process_and_upload_gps_table(
                ctx,
                nombre_archivo_gps=str(csv_path),
                nombres_variables_gps=dict(_NOMBRES_VARIABLES_GPS),
                formato_fecha="%Y-%m-%d %H:%M:%S",
            )
        except Exception:
            pass

    usecols = read_csv_kwargs.get("usecols")
    assert usecols is not None

    needed = set(_NOMBRES_VARIABLES_GPS.values())
    if callable(usecols):
        assert all(usecols(c) for c in needed), "usecols rejected a needed column"
        assert not usecols("JUNK_COL"), "usecols accepted a junk column"
    else:
        assert set(usecols) >= needed
        assert "JUNK_COL" not in usecols


# ---------------------------------------------------------------------------
# Transaction CSV usecols
# ---------------------------------------------------------------------------

def _make_trx_csv() -> str:
    return "\n".join([
        "id_trx,tarjeta,fecha_trx,linea_trx,ramal_trx,interno_trx,tipo,JUNK1,JUNK2",
        "1,T001,2024-01-01 08:00:00,10,1,101,normal,x,y",
        "2,T002,2024-01-01 08:05:00,10,1,102,invalido,x,y",
        "3,T003,2024-01-01 08:10:00,20,2,201,normal,x,y",
    ])


_NOMBRES_VARIABLES_TRX = {
    "id_tarjeta_trx": "tarjeta",
    "id_trx":         "id_trx",
    "fecha_trx":      "fecha_trx",
    "id_linea_trx":   "linea_trx",
    "id_ramal_trx":   "ramal_trx",
    "interno_trx":    "interno_trx",
}

_TIPO_TRX_INVALIDAS = {"tipo": ["invalido"]}

_TRX_CONFIGS = {
    "lineas_contienen_ramales": True,
    "resolucion_h3": 8,
    "valor_inicio_servicio": "start_service",
    "valor_fin_servicio": "finish_service",
    "nombres_variables_trx": _NOMBRES_VARIABLES_TRX,
    "nombres_variables_gps": {},
}


def _spy_trx_read(tmp_path):
    """Returns (trx_path, spy context manager, captured_calls list)."""
    trx_path = tmp_path / "trx.csv"
    trx_path.write_text(_make_trx_csv())
    captured = []
    original = pd.read_csv

    def spy(path, **kwargs):
        captured.append({"path": str(path), **kwargs})
        return original(path, **kwargs)

    return trx_path, spy, captured


def test_trx_eco_read_passes_usecols(tmp_path):
    """geolocalizar_trx must pass usecols so unreferenced CSV columns are never parsed."""
    from urbantrips.datamodel.transactions import geolocalizar_trx

    trx_path, spy, captured = _spy_trx_read(tmp_path)

    with patch("urbantrips.datamodel.transactions.pd.read_csv", side_effect=spy), \
         patch("urbantrips.datamodel.transactions.leer_configs_generales", return_value=_TRX_CONFIGS):
        try:
            geolocalizar_trx(
                ctx=MagicMock(),
                nombre_archivo_trx_eco=str(trx_path),
                nombres_variables_trx=dict(_NOMBRES_VARIABLES_TRX),
                tipo_trx_invalidas=_TIPO_TRX_INVALIDAS,
                formato_fecha="%Y-%m-%d %H:%M:%S",
                nombre_archivo_gps="gps.csv",
                nombres_variables_gps=dict(_NOMBRES_VARIABLES_GPS),
            )
        except Exception:
            pass

    trx_calls = [c for c in captured if "trx" in c["path"]]
    assert trx_calls, "trx read_csv was not called"
    assert "usecols" in trx_calls[0], "trx read_csv missing usecols parameter"


def test_trx_eco_usecols_includes_filter_columns(tmp_path):
    """usecols must include columns used by filtrar_transacciones_invalidas."""
    from urbantrips.datamodel.transactions import geolocalizar_trx

    trx_path, spy, captured = _spy_trx_read(tmp_path)

    with patch("urbantrips.datamodel.transactions.pd.read_csv", side_effect=spy), \
         patch("urbantrips.datamodel.transactions.leer_configs_generales", return_value=_TRX_CONFIGS):
        try:
            geolocalizar_trx(
                ctx=MagicMock(),
                nombre_archivo_trx_eco=str(trx_path),
                nombres_variables_trx=dict(_NOMBRES_VARIABLES_TRX),
                tipo_trx_invalidas=_TIPO_TRX_INVALIDAS,
                formato_fecha="%Y-%m-%d %H:%M:%S",
                nombre_archivo_gps="gps.csv",
                nombres_variables_gps=dict(_NOMBRES_VARIABLES_GPS),
            )
        except Exception:
            pass

    trx_calls = [c for c in captured if "trx" in c["path"]]
    usecols = trx_calls[0].get("usecols")
    assert usecols is not None

    if callable(usecols):
        # "tipo" is in tipo_trx_invalidas — must be kept for pre-rename filtering
        assert usecols("tipo"), "usecols rejected 'tipo' needed by filtrar_transacciones_invalidas"
        assert not usecols("JUNK1"), "usecols accepted junk column"
    else:
        assert "tipo" in usecols
        assert "JUNK1" not in usecols
