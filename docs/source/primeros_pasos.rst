Primeros pasos
==============

Una vez creado el ambiente e instalada UrbanTrips es necesario organizar los datos que funcionarán como insumos del proceso y el archivo de configuración. 

**Insumos necesarios y opcionales**

Urbantrips requiere sólo 3 insumos indispensables:

* Un archivo de configuración: ``configuraciones_generales.yaml``
* Un archivo ``csv`` con las transacciones del sistema de pago con tarjeta (las transacciones deben estar georeferenciadas y tener una serie de campos mínimos)
* Un archivo ``csv`` con la información de las líneas y/o ramales que conforman el sistema de transporte público.

El archivo de configuración tendrá especificados todos los parámetros requeridos para la corrida del proceso UrbanTrips. Entre otros parámetros, deben especificarse las corridas con los diferentes periodos de tiempo que se procesaran. Estos nombres de corridas determinan los nombres que tendran los archivos ``csv``. En el directorio de trabajo (ver `Estructura de directorios`_.) podrá haber diversos archivos con datos de diferentes días o periodos de tiempo (``lunes_trx.csv``, ``martes_trx.csv`` y ``lunes_gps.csv``, ``martes_gps.csv`` o ``enero_trx.csv``, ``febrero_trx.csv`` y ``enero_gps.csv``, ``febrero_gps.csv``). Cada uno será procesado en una corrida por vez. 

El archivo csv con las transacciones debe tener una serie de campos obligatorios (para más detalles ver :doc:`inputs`). Los nombres de estos campos en el archivo pueden ser diferentes y la equivalencia se configura en el archivo ``configuraciones_generales.yaml`` en el parámetro ``nombres_variables_trx``. Para más detalles sobre cómo utilizar este archivo de configuración consulte el apartado :doc:`configuracion`). 

También es necesario un archivo csv que contenga información de las líneas (y ramales en caso de existir). Fundamentalmente debe incluir un nombre de fantasía o de cartel para cada linea y/o ramal con su id correspondiente y el modo, que ser'a estandarizado luego utilizando los parámetros seteados en ``configuraciones_generales.yaml``. Adicionalmente se puede sumar información de empresa y algún campo descriptivo. Para más detalles de los campos que debe incluir puede ver el apartado :doc:`inputs`. La forma de tratar a las líneas y ramales en UrbanTrips es muy específica, por lo tanto se aconseja leer el apartado  :doc:`lineas_ramales`.

Con solo estos archivos se podrá correr el proceso que resultará en la imputación de destinos, construcción de matrices OD y elaboración de algunos KPIs, mapas y gráficos. 

De cualquier forma, se obtienen resultados adicionales y con mayor precisión si se incluyen los siguientes archivos opcionales:

* Tabla con información de las líneas y/o ramales de transporte público (nombre de fantasía, etc).
* Tabla de GPS con el posicionamiento de las unidades.
* Cartografía de los recorridos de las líneas y/o ramales de transporte público.
* Cartografía de las zonificaciones con las unidades espaciales utilizadas para agregar datos para la matriz OD.
* Cartografía de las paradas y/o estaciones. 


A modo de ejemplo se puede descargar el `dataset abierto de transacciones SUBE de AMBA <https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/transacciones.csv>`_ , guardarlo en ``data/data_ciudad/transacciones.csv``. Este dataset no cuenta con un campo ``fecha`` con el formato ``dd/mm/aaaa``, deberá agregar con una fecha cualquiera y utilizar las configuraciones especificadas más abajo. A su vez, se debe especificar un ``id_linea`` con el criterio de UrbanTrips (:doc:`lineas_ramales`). Para eso se puede tomar la información de lineas de `este archivo <https://github.com/EL-BID/Matriz-Origen-Destino-Transporte-Publico/blob/main/data/lineas_ramales.csv>`_ (que se puede utilizar para el parámetro ``nombre_archivo_informacion_lineas``). En este archivo, cada ``id_ramal`` tiene un ``id_linea`` asignado, con esa información pueden construir el ``id_linea`` de la tabla transacciones.  


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
   │       │   semana1_trx.csv
   │       │   semana2_trx.csv
   │       │   lineas_amba.csv
   │       │   hexs_amba.geojson
   │       │   ...



Correr Urbantrips
-----------------

Una vez que se dispone del archivo de transacciones y el de información de las líneas (junto con los opcionales como gps, recorridos, etc), es posible comenzar a utilizar UrbanTrips. Para una corrida del conjunto del proceso puede utilizar el archivo de configuración que viene por defecto y tendrá una corrida para una muestra del 1% de los datos de área urbana de Buenos Aires para 2019.

El proceso de UrbanTrips se ejecuta en 4 pasos secuenciales:

1. **ingest**: Carga y procesa los datos de entrada (transacciones, GPS, líneas, etc.)
2. **legs**: Construye los viajes individuales a partir de las transacciones
3. **outputs**: Genera matrices OD y KPIs
4. **dashboard**: Prepara los datos para visualización interactiva

Ejemplos de uso desde consola (Windows o Linux), siempre con el ambiente activado:

**Ejecutar el proceso completo**

Corre todos los pasos desde ingest hasta dashboard:

.. code:: sh
   
   $ python urbantrips/run_all_urbantrips.py

**Ejecutar con borrado previo**

Borra todos los datos de corridas anteriores y ejecuta el proceso completo desde cero:
    
.. code:: sh
   
   $ python urbantrips/run_all_urbantrips.py --borrar_corrida all

**Ejecutar sin dashboard**

Corre ingest → legs → outputs, pero omite la creación del dashboard:

.. code:: sh
   
   $ python urbantrips/run_all_urbantrips.py --no_dashboard

**Ejecutar hasta un paso específico**

Usa ``--through`` para ejecutar desde ingest hasta el paso indicado (inclusive):

.. code:: sh
   
   # Corre solo ingest + legs
   $ python urbantrips/run_all_urbantrips.py --through legs
   
   # Corre ingest + legs + outputs (sin dashboard)
   $ python urbantrips/run_all_urbantrips.py --through outputs

**Ejecutar solo un paso**

Usa ``--step`` para ejecutar únicamente un paso específico. El sistema validará que los pasos previos ya hayan sido ejecutados:

.. code:: sh
   
   # Corre solo el paso de ingest
   $ python urbantrips/run_all_urbantrips.py --step ingest
   
   # Corre solo el paso de legs (requiere que ingest ya se haya ejecutado)
   $ python urbantrips/run_all_urbantrips.py --step legs
   
   # Corre solo el paso de outputs (requiere que legs ya se haya ejecutado)
   $ python urbantrips/run_all_urbantrips.py --step outputs
   
   # Corre solo el dashboard (requiere que outputs ya se haya ejecutado)
   $ python urbantrips/run_all_urbantrips.py --step dashboard

**Usar un archivo de configuración alternativo**

Especifica un archivo de configuración diferente al predeterminado:

.. code:: sh
   
   $ python urbantrips/run_all_urbantrips.py \
       --config configs/otra_ciudad.yaml

**Especificar el directorio base del proyecto**

Útil cuando el proyecto no está en el directorio actual:

.. code:: sh
   
   $ python urbantrips/run_all_urbantrips.py \
       --base-dir /ruta/a/mi/proyecto

**Combinación de opciones**

Puedes combinar varios parámetros. Por ejemplo, usar una configuración alternativa y ejecutar hasta un paso específico:

.. code:: sh
   
   $ python urbantrips/run_all_urbantrips.py \
       --config configs/configuraciones_generales.yaml \
       --through outputs \
       --base-dir /home/usuario/urbantrips

.. note::
   No se puede usar ``--step`` junto con ``--borrar_corrida``, ya que no tiene sentido borrar datos y ejecutar solo un paso intermedio en aislamiento.


Resultados finales
------------------

Una vez procesados los datos, los resultados de urbantrips se guardarán en una base de datos ``duckdb`` en ``data/db/``. Los principales resultados pueden accederse mediante el  dashboard interactivo. 

.. code:: sh

   $ streamlit run urbantrips/dashboard/dashboard.py


