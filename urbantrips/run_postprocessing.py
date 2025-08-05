from urbantrips.preparo_dashboard.preparo_dashboard import preparo_indicadores_dash
from urbantrips.utils.utils import leer_configs_generales


def main():
    configs_usuario = leer_configs_generales(autogenerado=False)
    corridas = configs_usuario.get("corridas", None)
    for corrida in corridas:
        preparo_indicadores_dash(corrida=corrida, resoluciones=[6, 7])


if __name__ == "__main__":
    main()
