# urbantrips/tests/unit/test_selector_corridas.py
"""Registro de corridas y resolución alias → config del selector del dashboard.

La lógica vive en `dash_storage`, que no importa streamlit a propósito, así que
se puede testear entera sin levantar el dashboard.
"""
import duckdb
import pytest
import yaml

from urbantrips.dashboard import dash_storage
from urbantrips.utils import paths as paths_mod


_YAML_BASE = {
    "alias_db_insumos": "corrida_a",
    "resolucion_h3": 9,
    "epsg_m": 5347,
    "lineas_contienen_ramales": True,
}


@pytest.fixture
def proyecto(tmp_path, monkeypatch):
    """Un proyecto de mentira: configs/ + data/db/, con paths apuntando ahí."""
    configs = tmp_path / "configs"
    configs.mkdir()
    db_dir = tmp_path / "data" / "db"
    db_dir.mkdir(parents=True)

    config_file = configs / "configuraciones_generales.yaml"
    config_file.write_text(yaml.safe_dump(_YAML_BASE), encoding="utf-8")

    monkeypatch.setenv("URBANTRIPS_CONFIG", str(config_file))
    paths_mod.reset_paths()
    yield tmp_path
    paths_mod.reset_paths()


def _general_con_snapshot(db_dir, alias, contenido, archivo="origen.yaml"):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter

    a = DuckDBGeneralAdapter(db_dir / f"{alias}_general.duckdb")
    a.save_config_snapshot(alias, "c1", archivo, contenido)
    a.close()


def _general_legacy(db_dir, alias):
    """Base con el esquema viejo de `corridas`: sin config_yaml ni snapshot."""
    con = duckdb.connect(str(db_dir / f"{alias}_general.duckdb"))
    con.execute("CREATE TABLE corridas (corrida TEXT, process TEXT, date TEXT)")
    con.execute("INSERT INTO corridas VALUES ('c1', 'done', '2026-01-01')")
    con.close()


# ── registro corridas.yaml ───────────────────────────────────────────────────

def test_sin_registro_devuelve_vacio(proyecto):
    """Sin corridas.yaml el dashboard queda como antes: sin selector."""
    assert dash_storage.leer_corridas_registradas() == []


def test_registro_lista_simple(proyecto):
    (proyecto / "configs" / "corridas.yaml").write_text(
        "corridas:\n  - corrida_a\n  - corrida_b\n", encoding="utf-8"
    )
    assert dash_storage.leer_corridas_registradas() == ["corrida_a", "corrida_b"]


def test_registro_acepta_dicts_y_deduplica(proyecto):
    """Formato con dicts, para poder agregarle campos sin romper lo escrito."""
    (proyecto / "configs" / "corridas.yaml").write_text(
        "corridas:\n"
        "  - alias: corrida_a\n"
        "    nota: la buena\n"
        "  - corrida_b\n"
        "  - corrida_a\n",
        encoding="utf-8",
    )
    assert dash_storage.leer_corridas_registradas() == ["corrida_a", "corrida_b"]


def test_alias_con_guiones_bajos_entre_digitos(proyecto):
    """`2024_9_17` sin comillas es un NÚMERO para YAML (guión bajo = separador
    de miles) y llegaba convertido en 2024917. El registro son nombres de base,
    nunca números, así que se lee sin inferencia de tipos."""
    (proyecto / "configs" / "corridas.yaml").write_text(
        "corridas:\n  - 2024_9_17\n  - 2026\n", encoding="utf-8"
    )
    assert dash_storage.leer_corridas_registradas() == ["2024_9_17", "2026"]


def test_registro_malformado_no_rompe(proyecto):
    (proyecto / "configs" / "corridas.yaml").write_text(
        "esto: no es\n  - una lista valida\n:", encoding="utf-8"
    )
    assert dash_storage.leer_corridas_registradas() == []


def test_registro_vacio_no_rompe(proyecto):
    (proyecto / "configs" / "corridas.yaml").write_text("corridas:\n", encoding="utf-8")
    assert dash_storage.leer_corridas_registradas() == []


# ── resolución alias → config ────────────────────────────────────────────────

def test_resuelve_por_snapshot(proyecto):
    """Camino preferido: la config que produjo los datos, guardada en la base."""
    contenido = yaml.safe_dump({**_YAML_BASE, "alias_db_insumos": "corrida_a",
                                "epsg_m": 22185})
    _general_con_snapshot(proyecto / "data" / "db", "corrida_a", contenido)

    r = dash_storage.resolver_config_de_alias("corrida_a")

    assert r["ok"] and r["fuente"] == "snapshot"
    # se materializa dentro de configs/ (get_paths deriva la raíz de ahí)
    assert r["config"].parent == proyecto / "configs"
    assert r["config"].read_text(encoding="utf-8") == contenido
    # y el contenido es el del snapshot, no el del yaml del proyecto
    assert yaml.safe_load(r["config"].read_text(encoding="utf-8"))["epsg_m"] == 22185


def test_el_snapshot_gana_al_yaml_de_configs(proyecto):
    """Si el yaml de configs/ cambió después de la corrida, manda el snapshot."""
    (proyecto / "configs" / "otra.yaml").write_text(
        yaml.safe_dump({**_YAML_BASE, "alias_db_insumos": "corrida_a",
                        "epsg_m": 99999}),
        encoding="utf-8",
    )
    contenido = yaml.safe_dump({**_YAML_BASE, "epsg_m": 5347})
    _general_con_snapshot(proyecto / "data" / "db", "corrida_a", contenido)

    r = dash_storage.resolver_config_de_alias("corrida_a")

    assert r["fuente"] == "snapshot"
    assert yaml.safe_load(r["config"].read_text(encoding="utf-8"))["epsg_m"] == 5347


def test_resuelve_por_configs_si_la_base_es_legacy(proyecto):
    """Bases anteriores al snapshot: se busca el yaml que declare el alias."""
    _general_legacy(proyecto / "data" / "db", "corrida_legacy")
    (proyecto / "configs" / "la_de_legacy.yaml").write_text(
        yaml.safe_dump({**_YAML_BASE, "alias_db_insumos": "corrida_legacy"}),
        encoding="utf-8",
    )

    r = dash_storage.resolver_config_de_alias("corrida_legacy")

    assert r["ok"] and r["fuente"] == "configs"
    assert r["config"].name == "la_de_legacy.yaml"


def test_prefiere_el_yaml_sin_alias_obsoletos(proyecto):
    """Caso real: dos yamls declaran el alias y uno partiría el dashboard.

    Un yaml con alias_db_data/alias_db_dashboard hace que las páginas 4-8 abran
    otra base que el resto. Entre candidatos, gana el sano — aunque ordene después
    alfabéticamente.
    """
    _general_legacy(proyecto / "data" / "db", "corrida_x")
    (proyecto / "configs" / "aaa_autogenerado.yaml").write_text(
        yaml.safe_dump({**_YAML_BASE, "alias_db_insumos": "corrida_x",
                        "alias_db_data": "otra_base",
                        "alias_db_dashboard": "otra_base"}),
        encoding="utf-8",
    )
    (proyecto / "configs" / "zzz_sano.yaml").write_text(
        yaml.safe_dump({**_YAML_BASE, "alias_db_insumos": "corrida_x"}),
        encoding="utf-8",
    )

    r = dash_storage.resolver_config_de_alias("corrida_x")

    assert r["config"].name == "zzz_sano.yaml"


def test_prefiere_un_yaml_estable_sobre_el_config_base(proyecto):
    """`configuraciones_generales.yaml` no identifica una corrida.

    `check_config` lo reescribe en CADA corrida con la config del último run
    (check_configs.py:1013), así que siempre declara el alias de lo último que se
    corrió. Elegirlo daría una respuesta distinta según qué corriste último.
    """
    _general_legacy(proyecto / "data" / "db", "corrida_a")
    # el fixture ya dejó configuraciones_generales.yaml declarando corrida_a
    (proyecto / "configs" / "zzz_estable.yaml").write_text(
        yaml.safe_dump({**_YAML_BASE, "alias_db_insumos": "corrida_a"}),
        encoding="utf-8",
    )

    r = dash_storage.resolver_config_de_alias("corrida_a")

    assert r["config"].name == "zzz_estable.yaml"


def test_usa_el_config_base_si_no_hay_otro(proyecto):
    """Pero sigue sirviendo como último recurso."""
    _general_legacy(proyecto / "data" / "db", "corrida_a")

    r = dash_storage.resolver_config_de_alias("corrida_a")

    assert r["ok"]
    assert r["config"].name == "configuraciones_generales.yaml"


def test_alias_sin_nada_no_explota(proyecto):
    """Devuelve ok=False y un motivo legible, para mostrarlo deshabilitado."""
    r = dash_storage.resolver_config_de_alias("no_existe")

    assert r["ok"] is False
    assert r["config"] is None
    assert "no_existe" in r["motivo"]


def test_no_se_encuentra_a_si_mismo(proyecto):
    """El snapshot materializado no debe contarse como candidato del scan.

    Vive en configs/ (lo necesita get_paths para derivar la raíz del proyecto),
    así que sin el filtro por prefijo el scan se encontraría a sí mismo.
    """
    contenido = yaml.safe_dump({**_YAML_BASE, "alias_db_insumos": "corrida_a"})
    _general_con_snapshot(proyecto / "data" / "db", "corrida_a", contenido)
    materializado = dash_storage.resolver_config_de_alias("corrida_a")["config"]
    assert materializado.name.startswith(dash_storage.PREFIJO_SNAPSHOT)
    assert materializado.parent == proyecto / "configs"

    encontrado = dash_storage._buscar_config_por_alias("corrida_a")
    assert encontrado != materializado
    assert not encontrado.name.startswith(dash_storage.PREFIJO_SNAPSHOT)


# ── metadata para la etiqueta ────────────────────────────────────────────────

def test_describir_corrida_esquema_nuevo(proyecto):
    from urbantrips.storage.adapters.duckdb.general import DuckDBGeneralAdapter

    db_dir = proyecto / "data" / "db"
    a = DuckDBGeneralAdapter(db_dir / "corrida_a_general.duckdb")
    a.register_step("corrida_a", "c1", ["2026-03-09", "2026-03-10"], "ingest")
    a.close()

    info = dash_storage.describir_corrida("corrida_a")

    assert info["dias"] == 2
    assert info["desde"] == "2026-03-09"
    assert info["hasta"] == "2026-03-10"


def test_describir_corrida_legacy_no_rompe(proyecto):
    _general_legacy(proyecto / "data" / "db", "vieja")
    info = dash_storage.describir_corrida("vieja")
    assert info["dias"] == 1          # cuenta corridas, no días
    assert info["desde"] is None


def test_describir_corrida_inexistente(proyecto):
    assert dash_storage.describir_corrida("fantasma")["dias"] is None


def test_bases_faltantes(proyecto):
    db_dir = proyecto / "data" / "db"
    (db_dir / "parcial_data.duckdb").write_bytes(b"")
    (db_dir / "parcial_dash.duckdb").write_bytes(b"")

    assert dash_storage.bases_faltantes("parcial") == ["insumos", "general"]
    assert set(dash_storage.bases_faltantes("ninguna")) == {
        "data", "insumos", "dash", "general"
    }


def test_declara_alias_obsoletos(proyecto):
    sano = proyecto / "configs" / "sano.yaml"
    sano.write_text(yaml.safe_dump(_YAML_BASE), encoding="utf-8")
    roto = proyecto / "configs" / "roto.yaml"
    roto.write_text(
        yaml.safe_dump({**_YAML_BASE, "alias_db_data": "x"}), encoding="utf-8"
    )

    assert dash_storage.declara_alias_obsoletos(sano) == []
    assert dash_storage.declara_alias_obsoletos(roto) == ["alias_db_data"]
