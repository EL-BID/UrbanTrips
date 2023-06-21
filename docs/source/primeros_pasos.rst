Primeros pasos
==============

Una vez creado el ambiente e instalada UrbanTrips es necesario organizar los datos que funcionarán como insumos del proceso y el archivo de configuración. 

**Insumos necesarios y opcionales**

Urbantrips requiere sólo 2 insumos indispensables:

* Un archivo ``csv`` con las transacciones del sistema de pago con tarjeta
* Un archivo de configuración: ``configuraciones_generales.yaml``



UrbanTrips tomará el archivo de configuración que le dirá (entre otras cosas) qué archivo csv contiene la información con los datos de las transacciones a utilizar en esta corrida. En el directorio de trabajo (ver `Estructura de directorios`_.) podrá haber diversos archivos con datos de diferentes días o periodos de tiempo (``lunes.csv``, ``martes.csv`` o ``enero.csv``, ``febrero.csv``). Cada uno será procesado en una corrida por vez. Qué archivo utilizar se configura en ``configuraciones_generales.yaml`` en el parámetro ``nombre_archivo_trx:``. El archivo csv con las transacciones debe tener una serie de campos obligatorios (para más detalles ver :doc:`inputs`). Los nombres de estos campos en el archivo pueden ser diferentes y la equivalencia se configura en el archivo ``configuraciones_generales.yaml`` en el parámetro ``nombres_variables_trx``. Para más detalles sobre cómo utilizar este archivo de configuración consulte el apartado :doc:`configuracion`). 

Con UrbanTrips se pueden procesar en una corrida la información de transacciones correspondientes a más de un día. Sin embargo, no se puede dividir un mismo día en dos corridas. Toda la información respecto de un día debe procesarse en la misma corrida. Si es demasiada información, conviene separarla en diversos archivos donde cada uno siempre tenga la totalidad de la información de los días a analizar (por ej. ``lunes.csv``, ``martes.csv`` o ``semana1.csv``, ``semana2.csv`` pero no ``lunes_a.csv``, ``lunes_b.csv``). Luego en otras corridas pueden procesarse otros días y la información se irá actualizando en las bases correspondientes.

Con sólo esos archivos podrá correr el proceso de imputación de destinos, construcción de matrices OD y elaboración de algunos KPIs. Dicho eso, se obtendrán más resultados y con mayor precisión si se suman estos archivos opcionales:

* Tabla con información de las líneas y/o ramales de transporte público (nombre de fantasía, etc).
* Tabla de GPS con el posicionamiento de las unidades.
* Cartografía de los recorridos de las líneas y/o ramales de transporte público.
* Cartografía de las zonificaciones con las unidades espaciales utilizadas para agregar datos para la matriz OD.
* Cartografía de las paradas y/o estaciones. 


A modo de ejemplo se puede descargar el `dataset abierto de transacciones SUBE de AMBA <https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/transacciones.csv>`_ , guardarlo en ``data/data_ciudad/transacciones.csv``. Este dataset no cuenta con un campo ``fecha`` con el formato ``dd/mm/aaaa``, deberá agregar con una fecha cualquiera y utilizar las configuraciones especificadas más abajo. A su vez, se debe especificar un ``id_linea`` con el criterio de UrbanTrips (:doc:`lineas_ramales`). Para eso se puede tomar la información de lineas de `este archivo <https://github.com/EL-BID/Matriz-Origen-Destino-Transporte-Publico/blob/main/data/lineas_ramales.csv>`_ (que se puede utilizar para el parámetro ``nombre_archivo_informacion_lineas``). En este archivo, cada ``id_ramal`` tiene un ``id_linea`` asignado, con esa información pueden construir el ``id_linea`` de la tabla transacciones.  


**Archivo de configuración para dataset SUBE AMBA**

.. code:: yaml

   geolocalizar_trx: False
   resolucion_h3: 8
   
   #tolerancia parada destino en metros
   tolerancia_parada_destino: 2200

   nombre_archivo_trx: transacciones.csv

   alias_db_data: amba

   alias_db_insumos: amba

   lineas_contienen_ramales: True
   nombre_archivo_informacion_lineas: lineas_amba.csv

   imputar_destinos_min_distancia: True

   #ingresar el nombre de las variables
   nombres_variables_trx:
      id_trx: id
      fecha_trx: fecha 
      id_tarjeta_trx: id_tarjeta
      modo_trx: modo
      hora_trx: hora
      id_linea_trx: id_linea
      id_ramal_trx:  id_ramal
      interno_trx: interno_bus
      orden_trx: etapa_red_sube
      latitud_trx: lat 
      longitud_trx: lon
      factor_expansion:   
	
   modos:
      autobus: COL
      tren: TRE
      metro: SUB
      tranvia:
      brt:
	 
   recorridos_geojson:

   # Filtro de coordenadas en formato minx, miny, maxx, maxy 
   filtro_latlong_bbox:
      minx: -59.3
      miny: -35.5
      maxx: -57.5
      maxy: -34.0 

	
   #Especificar el formato fecha
   formato_fecha: "%d/%m/%Y"

   columna_hora: True 
   ordenamiento_transacciones: orden_trx 


   tipo_trx_invalidas:
      tipo_trx_tren:
         - 'CHECK OUT SIN CHECKIN'
         - 'CHECK OUT'


Estructura de directorios
-------------------------
.. _Estructura de directorios:

Esta es la estructura de directorios de UrbanTrips. ``configs/`` guarda el archivo de configuraciones principal. ``data/`` tendrá por un lado los archivo de insumo para la ciudad (transacciones, gps, etc) y los resultados producto de la corrida de UrbanTrips que se guardarán en ``data/db/``. Para más información del modelo de datos de los resultados finales consulte :doc:`resultados`. Por último en el directorio ``resultados/`` se guardarán algunos resultados agregados en tablas, mapas, gráficos y en formatos más amigables como ``csv``, ``html``, ``png``.  

.. code:: 

   urbantrips
   │   README.md
   │
   └─── urbantrips
   │   ...
   └─── configs
   │   │   configuraciones_generales.yaml
   │   │   
   └─── data 
   │   └─── db
   │       │  amba_2023_semana1_data
   │       │  amba_2023_semana2_data
   │       │  amba_2023_insumos
   │       
   │   └─── data_ciudad
   │       │   semana1.csv
   │       │   semana2.csv
   │       │   lineas_amba.csv
   │       │   hexs_amba.geojson
   │       │   ...
   └─── resultados 
   │   └─── data
   │       │   amba_2023_semana1_etapas.csv
   │       │   amba_2023_semana1_viajes.csv
   │       │   amba_2023_semana1_usuarios.csv
   │       │   amba_2023_semana2_etapas.csv
   │       │   amba_2023_semana2_viajes.csv
   │       │   amba_2023_semana2_usuarios.csv
   │   └─── html
   │       │   ...
   │   └─── matrices
   │       │   ...
   │   └─── pdf
   │       │   ...
   │   └─── png
   │       │   ...
   │   └─── tablas



Correr Urbantrips
-----------------

Una vez que se dispone del archivo de transacciones y el de información de las líneas, es posible comenzar a utilizar UrbanTrips. En primer lugar es necesario inicializar los directorios y la base de datos necesarios. Este paso solo se corre una vez.

.. code:: sh

   $ python urbantrips/initialize_environment.py

Luego, se puede procesar la información de transacciones. Este archivo de transacciones puede tener la información de un día, una semana o un mes (siempre que no sea demasiada información). Este paso procesa las transacciones en etapas y viajes, imputando destinos. Luego pueden correr este paso por cada nuevo dataset que quieran procesar (``semana_1.csv``, ``semana_2.csv``, etc) ajustando lo necesario en el archivo ``configuraciones_generales.yaml`` previo a cada corrida.

.. code:: sh

   $ python urbantrips/process_transactions.py

Por último, una vez procesadas todas las transacciones que sean de interés y cargadas en la base de datos de la libería, es posible correr los pasos de post procesamiento sobre esa información, como los KPI, visualizaciones y exportación de resultados. 

.. code:: sh

   $ python urbantrips/run_postprocessing.py


Resultados finales
------------------

Una vez procesados los datos, los resultados de urbantrips se guardarán en una base de datos ``SQLite`` en ``data/db/``. Para más información del modelo de datos de los resultados finales consulte :doc:`resultados`.
