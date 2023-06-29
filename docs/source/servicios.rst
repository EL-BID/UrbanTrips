Clasificación de servicios
==========================

Normalmente en el modelo de datos de los Sistemas de Recaudación Electrónicos para transporte público existe una tabla de datos con la localización de los vehículos de las diferentes líneas en el tiempo (en UrbanTrips la tabla ``gps``). En algunos de estos modelos, existe en esa tabla un atributo que indica cuando un vehículo comienza un servicio, es decir que se presta a subir pasajeros mientras recorre el recorrido de la línea y ramal para el que presta ese servicio. En algunas ocasiones el conductor marca un inicio de servicio para todo su turno, realizando en realidad más de un servicio y posiblemente ramales diferentes en esos servicios.

Como los servicios son una unidad de información vital para obtener ciertos indicadores estadísticos del sistema, UrbanTrips procede a crear servicios en base a trazas de puntos gps cuando esta declaración de los servicios por parte del conductor no es del todo confiable. Previamente a correr los servicios es necesario que exista una tabla de gps cargada y procesada en la base de datos. Esto se puede hacer en UrbanTrips con la siguiente función, extrayendo los parámetros del archivo de configuración. 

.. code:: 

   trx.process_and_upload_gps_table(
    	nombre_archivo_gps=nombre_archivo_gps,
    	nombres_variables_gps=nombres_variables_gps,
    	formato_fecha=formato_fecha)

Dicha tabla debe tener un atributo donde se especifique el inicio de un servivcio. Tambien puede especificarse el final del mismo. Esto debera cargarse en el archivo de configuracion del especificando el [ATTR] y los valores correspondientes del siguiente modo:

.. code:: yaml

   service_type_gps:
        		[ATTR]:
        			start_service: [VAL]
        			finish_service: [VAL]


A su vez en el archivo de configuración se debe setear el parámetro correspondiente. Si ese atributo es confiable o si UrbanTrips debe, dentro de cada servicio tal como es declarado por el conductor, clasificar nuevos servicios.  

.. code:: yaml

   trust_service_type_gps: False

Por ultimo solo queda correr

.. code:: yaml

   services.process_services()




Como se clasifican nuevos servicios?
------------------------------------


UrbanTrips tomará los puntos gps que pertenezcan a un servicio tal cual fue declarado por el conductor (con el registro de apertura y cierre en la tabla gps) y procederá a clasificarlos en uno o más servicios, con su id nuevo, en base a un algoritmo que toma como elemento fundamental el orden de paso por las paradas.

En ese sentido, para que el proceso funcione debe estar cargada la tabla ``stops`` donde se define para cada linea y ramal una serie de paradas indicadas con el orden de paso de cada ramal. Esto se crea con la siguiente función en base a un archivo csv que debe estar creado (para la creación de paradas puede seguir el notebook ``Stops and nodes creation helper.ipynb`` que permitirá crear paradas en base a recorridos).

	stops.create_stops_table()

El proceso se puede resumir del siguiente modo: cada punto gps se asignará a la parada más cercana de esa linea, con su correspondiente orden de paso. Cuando se registra una inversión en el orden de paso por parada, es decir pase de un orden ascendente a descendente o viceversa, se abrira un nuevo servicio. 


.. image:: ../img/servicios_caso_simple.png
  :width: 800
  :alt: Alternative text
  

Si para esa linea existe más de un ramal, se evaluará el punto gps en todos los ramales de esa linea, juzgando si existe una inversión en el orden de paso solo en aquellos ramales cercanos.


GRAFICO CON LOS RAMALES EN Y


Posibles problemas
------------------------------------

Puede existir una configuración de ramales en una linea donde exista una inversión de sentido legítima que no implica un cambio de servicio. Un ramal puede ir y venir sobre sus propios pasos, teniendo paradas a lo largo de ese recorrido. Esto puede inducir un problema en este algoritmo de clasificación de servicios. Tomemos el siguiente ejemplo:

DIBUJO CON EJEMPLO DE PARADAS CON PROBLEMA

Para resolverlo, dichas paradas pueden agregarse en un único nodo mediante el campo ``node_id``. El proceso de clasificación de paradas en realidad utilizará los nodos. Con lo cual, si todas las paradas que puedan implicar una legitima inversión del sentido de paso quedan agrupadas en un único nodo, el algoritmo no registrará ese cambio. 




