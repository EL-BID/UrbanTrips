from urbantrips.datamodel import services
from urbantrips.datamodel import transactions as trx
from urbantrips.utils import utils
from datetime import datetime
from urbantrips.utils.check_configs import check_config

def main():
    # Check config file consistency
    check_config()

    # Read config file
    configs = utils.leer_configs_generales()

    # trx configs
    formato_fecha = configs["formato_fecha"]
    nombre_archivo_gps = configs["nombre_archivo_gps"]
    nombres_variables_gps = configs["nombres_variables_gps"]

    print("INICIO GPS")
    print(datetime.now())

    # SUBO LOS DATOS DE GPS , TRX YA GEOLOCALIZADO
    trx.process_and_upload_gps_table(
        nombre_archivo_gps=nombre_archivo_gps,
        nombres_variables_gps=nombres_variables_gps,
        formato_fecha=formato_fecha)

    print("FIN GPS")
    print(datetime.now())

    # proceso servicios
    print("INICIO SERVICIOS")
    print(datetime.now())
    services.process_services()
    print("FIN SERVICIOS")
    print(datetime.now())


if __name__ == "__main__":
    main()
