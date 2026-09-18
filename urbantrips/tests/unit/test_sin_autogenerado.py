# urbantrips/tests/unit/test_sin_autogenerado.py
"""El config autogenerado se eliminó del proceso (2026-07-27).

Venía del diseño viejo de una base por corrida. Aportaba 4 claves: dos dañinas
(`alias_db_data`, `alias_db_dashboard`, que hacen que unas páginas del dashboard
abran otra base que el resto) y dos que la ingesta ya deriva por convención.

Estos tests fijan que no vuelva, y que `write_config` mantenga consistente el
yaml EN USO en vez de uno con nombre fijo.
"""
import yaml
import pytest

from urbantrips.utils import paths as paths_mod


_CONFIG_MINIMO = {
    "alias_db_insumos": "mi_corrida",
    "resolucion_h3": 9,
    "epsg_m": 5347,
    "lineas_contienen_ramales": False,
    "corridas": ["lunes"],
}


@pytest.fixture
def proyecto(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (tmp_path / "data" / "db").mkdir(parents=True)

    config_file = configs / "mi_config.yaml"
    config_file.write_text(yaml.safe_dump(_CONFIG_MINIMO), encoding="utf-8")

    monkeypatch.setenv("URBANTRIPS_CONFIG", str(config_file))
    paths_mod.reset_paths()
    yield tmp_path
    paths_mod.reset_paths()


# ── que el autogenerado no vuelva ────────────────────────────────────────────

def test_las_funciones_del_autogenerado_ya_no_existen():
    from urbantrips.utils import check_configs
    from urbantrips.dashboard import dash_storage

    assert not hasattr(check_configs, "check_configs_file")
    assert not hasattr(check_configs, "add_dash_and_data_dbs")
    assert not hasattr(dash_storage, "_find_first_valid_yaml")


def test_leer_configs_ignora_el_parametro_autogenerado(proyecto):
    """El parámetro quedó aceptado-e-ignorado: lo pasan tests y notebooks."""
    from urbantrips.dashboard.dash_storage import leer_configs_generales

    con_true = leer_configs_generales(autogenerado=True)
    con_false = leer_configs_generales(autogenerado=False)
    sin_arg = leer_configs_generales()

    assert con_true == con_false == sin_arg == _CONFIG_MINIMO


def test_no_se_crea_ningun_autogenerado_al_leer(proyecto):
    """Antes, leer con autogenerado=True materializaba el archivo copiándolo."""
    from urbantrips.dashboard.dash_storage import leer_configs_generales

    leer_configs_generales(autogenerado=True)

    configs = proyecto / "configs"
    assert not (configs / "configuraciones_generales_autogenerado.yaml").exists()
    assert not (configs / "autogenerados").exists()


def test_leer_sin_el_directorio_autogenerados_no_rompe(proyecto):
    """El caso que antes lanzaba FileNotFoundError desde _find_first_valid_yaml.

    Sin `configs/autogenerados/`, la versión vieja no podía reconstruir el
    autogenerado y explotaba en vez de degradar.
    """
    from urbantrips.dashboard.dash_storage import leer_configs_generales

    assert not (proyecto / "configs" / "autogenerados").exists()
    assert leer_configs_generales() == _CONFIG_MINIMO


def test_yaml_corrupto_degrada_a_vacio(proyecto):
    from urbantrips.dashboard.dash_storage import leer_configs_generales

    (proyecto / "configs" / "mi_config.yaml").write_text(
        "esto: [no cierra\n  - ni ahi\n:", encoding="utf-8"
    )
    assert leer_configs_generales() == {}


# ── write_config escribe al yaml EN USO ──────────────────────────────────────

def _config_df():
    """DataFrame con la forma que consume write_config."""
    import pandas as pd

    return pd.DataFrame({
        "item": ["general", "general"],
        "variable": ["alias_db_insumos", "resolucion_h3"],
        "subvar": ["", ""],
        "default": ["mi_corrida", 9],
        "descripcion_campo": ["", ""],
        "descripcion_general": ["", ""],
    })


def test_write_config_escribe_al_yaml_en_uso(proyecto):
    """No al nombre hardcodeado `configuraciones_generales.yaml`.

    check_config normaliza codificación y tabs de `get_paths().config_file`, así
    que persistir en otro archivo dejaba la consistencia aplicada al yaml
    equivocado — y de paso pisaba el único config versionado.
    """
    from urbantrips.utils.check_configs import write_config

    write_config(_config_df())

    en_uso = proyecto / "configs" / "mi_config.yaml"
    assert "alias_db_insumos" in en_uso.read_text(encoding="utf-8")
    assert not (proyecto / "configs" / "configuraciones_generales.yaml").exists()


def test_write_config_no_pisa_el_config_por_defecto(proyecto):
    """Corriendo con --config otro.yaml, configuraciones_generales.yaml no se toca."""
    from urbantrips.utils.check_configs import write_config

    por_defecto = proyecto / "configs" / "configuraciones_generales.yaml"
    contenido = yaml.safe_dump({"alias_db_insumos": "no_me_toques"})
    por_defecto.write_text(contenido, encoding="utf-8")

    write_config(_config_df())

    assert por_defecto.read_text(encoding="utf-8") == contenido


def test_write_config_ya_no_acepta_el_flag(proyecto):
    from urbantrips.utils.check_configs import write_config

    with pytest.raises(TypeError):
        write_config(_config_df(), autogenerado=True)


# ── write_config no borra las claves que el Excel no conoce ──────────────────

def test_write_config_preserva_claves_ajenas_al_excel(proyecto):
    """`write_config` emite SOLO las filas de docs/configuraciones.xlsx.

    Sin esto, toda clave fuera de la planilla (tmp_dir, input_dir, db_dir,
    output_dir, alias_db…) se borraba en cada corrida: duraba una y desaparecía.
    """
    from urbantrips.utils.check_configs import extra_config_keys, write_config
    from urbantrips.dashboard.dash_storage import leer_configs_generales

    en_uso = proyecto / "configs" / "mi_config.yaml"
    usuario = dict(_CONFIG_MINIMO, tmp_dir="D:/urbantrips_tmp", alias_db="corrida_x")
    en_uso.write_text(yaml.safe_dump(usuario), encoding="utf-8")

    # Dos pasadas: la clave tiene que seguir ahí después de la segunda corrida.
    for _ in range(2):
        actuales = leer_configs_generales()
        write_config(_config_df(), extra_config_keys(actuales, _config_df()))

    final = yaml.safe_load(en_uso.read_text(encoding="utf-8"))
    assert final["tmp_dir"] == "D:/urbantrips_tmp"
    assert final["alias_db"] == "corrida_x"
    # y lo que sí está en la planilla se sigue consolidando desde ella
    assert final["alias_db_insumos"] == "mi_corrida"


def test_extra_config_keys_excluye_lo_que_esta_en_el_excel():
    from urbantrips.utils.check_configs import extra_config_keys

    extras = extra_config_keys(
        {"alias_db_insumos": "x", "resolucion_h3": 9, "tmp_dir": "/scratch"},
        _config_df(),
    )
    assert extras == {"tmp_dir": "/scratch"}
