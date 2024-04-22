Seteo del archivo de configuración 
==================================



Cada corrida leerá la información que hay en el archivo de configuración (``configuraciones_generales.yaml``). Su contenido puede editarse entre corrida y corrida para, por ejemplo, procesar días diferentes. Hay un único archivo de configuración en UrbanTrips. Se divide en diferentes categorías de parámetros.

Parámetros generales
--------------------

En este primer grupo encontramos parámetros generales que utliza UrbanTrips en diferentes momentos. El primer parámetro `resolucion_h3` establece el nivel de resolución del esquema e grillas hexagonales `H3 <https://h3geo.org/>`_	 con el que se va a trabajar. UrbanTrips realizar la mayoría de operaciones espaciales utilizando este esquema. Mientras mayor sea la resolución, más granulares serán las celdas haxagonales que UrbanTrips utilice, más preciso el resultado pero más costoso computacionalmente. La resolucion 8 tiene hexágonos de 460 metros de lado. En la resolucion 9 tienen 174 metros y en la 10 tienen 65 metros.

Luego vienen las configuraciǫnes que nombram las dos bases de datos con las que trabajará UrbanTips. ``alias_db_data`` guardará todo lo relativo a etapas, viajes y toda información que se actualiza con cada corrida. Así, puede haber una base de ``data`` diferente para cada semana o cada mes a medida que alcance un volumen determinado y utilizar un nombre específico para este propósito (``ciudad_2023_semana1``, ``ciudad_2023_semana2``,etc). Por su lado, ``alias_db_insumos`` es una base de datos que guardará información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de haxágonos H3 en una ciudad determinada, etc). 

También es necesario especificar una proyección de coordenadas en metros, pasando un id de `EPSG <https://epsg.io/>`_, para ciertos procesos espaciales que trabajan con distancias. Para Argentina puede usarse por defecto `9265 (POSGAR 2007 / UTM zone 19S) <https://epsg.io/9265>`_.

Es posible establecer un ``filtro_latlong_bbox`` que crea un box para eliminar rápidamente las transacciones que esten geolocalizadas fuera de una área lógica de cobertura del sistema de transporte público.

Por último el ``formato_fecha`` especifica el formato en el que se encuentra el campo ``fecha_trx`` (por ej. ``"%d/%m/%Y"``, ``"%d/%m/%Y %H:%M:%S"``) y las fechas en el archivo de posicionamiento GPS (si se utiliza). Todas las fechas a utilizar deben estar en el mismo formato.

.. code:: 

   resolucion_h3: 8

   # Alias a utilizar en las db de datos y de insumos
   alias_db_data: amba_test
   alias_db_insumos: amba_test

   # Proyeccion de coordenadas en metros a utilizar  
   epsg_m: 9265

   # Filtro de coordenadas en formato minx, miny, maxx, maxy 
   filtro_latlong_bbox:
    minx: -59.3
    miny: -35.5
    maxx: -57.5
    maxy: -34.0

   #Especificar el formato fecha presente en todos los dataset
   formato_fecha: "%d/%m/%Y"


Parámetros de imputación de destinos
------------------------------------

Este otro grupo de parámetros controla el método de imputación de destinos. Urbantrips utiliza como referencia para imputar para el destino de la transacción `t` la localización de la transacción `t + 1`. Asume que una persona que se tomo la Línea 1 se baja en una parada de esa linea "cercana" a su siguiente transacción. Si la siguiente transacción se encuentra muy lejos, se asume que hubo algun modo de transporte no registrado en la tarjeta en el medio y por ende no se puede imputar destino con cierta confiabilidad. El parámetro ``tolerancia_parada_destino`` establece este criterio de tolerancia en terminios de distancia (en metros). Si la distancia es mayor a esta tolerancia, no se imputará destino. 

Por otro lado, Urbantrips puede imputar el destino de la transacción `t` en la misma ubicación que `t + 1` (haya allí una parada de la Línea 1 o no) o puede escoger la parada de la Línea 1 más cercana a la localización de `t + 1`. El parametro  ``imputar_destinos_min_distancia`` establece si se imputará la localización de la siguiente transacción como destino o la localización de la parada de la linea utilizada en esa etapa que minimice la distancia con respecto a la siguiente transacción.

 .. code:: 
 
   # Distancia maxima tolerable entre destino imputado y siguiente transaccion (en metros)
   tolerancia_parada_destino: 2200

   # Imputar utilizando la parada de la linea de orige que minimice la distancia con respecto a la siguiente transaccion o solo la siguiente transaccion
   imputar_destinos_min_distancia: False



Parámetros de transacciones
---------------------------

En primer lugar se encuentra el nombre del archivo que contiene la información de las transacciones a utilizarse en la corrida. El mismo deberá localizarse en ``/data/data_ciudad/``. Esta parte del archivo de configuración permite especificar el nombre del archivo a utilizar como así también los nombres de los atributos tal cual aparecen en el csv para que puedan ser guardados en el esquema de datos de UrbanTrips.

El siguiente conjunto de parámetros de configuración definen el procesamiento de las transacciones.

* ``columna_hora``: Indica con ``True`` o ``False`` si la información sobre la hora está en una columna separada (``hora_trx``). Este debe ser un entero de 0 a 23.
* ``ordenamiento_transacciones``: El criterio para ordenar las transacciones en el tiempo. Si se cuenta con un timestamp completo con horas y minutos, entonces especificar ``fecha_completa``. Si solo se cuenta con la información del día y la hora, se puede usar ``orden_trx``. Este campo debe tener un entero secuencial que ordena las transacciones. Debe comenzar en cero cuando se comienza un nuevo viaje e incrementear con cada nueva etapa en ese viaje, dentro de una tarjeta para ese día.  
* ``ventana_viajes``: Cuando se tiene un timestamp completo, indica la ventana de tiempo en minutos para considerar que dos o más etapas que sucedan en esa ventana de tiempo con respecto a la primera etapa se agrupan en un mismo viaje.  
* ``ventana_duplicado``: Cuando se tiene un timestamp completo, indica la ventana de tiempo en minutos para considerar que dos transacciones son simultaneas, por lo se creará un ``id_tarjeta`` ad hoc a cada una, asumiendo que se trata de usuarios diferentes que utilizan una misma tarjeta.
* ``tipo_trx_invalidas``: Especifica primero el nombre del atributo tal cual aparece en el csv y luego los valores que deben eliminarse al no representar transacciones vinculadas a viajes (por ej. carga de salgo en la tarjeta, errores del sistema, check outs de modos que los posean). Se pueden especificar varios atributos y varios valores por cada atributo.
* ``modos``: UrbanTrips estandariza los modos en 5 categorias (``autobus``, ``tren``, ``metro``, ``tranvia`` y ``brt``). Debe pasarse el equivalente a cómo aparece categorizado en el csv cada modo.  


.. code:: 

   nombre_archivo_trx: semana1.csv

   nombres_variables_trx:
      id_trx: id
      fecha_trx: fecha
      id_tarjeta_trx: id_tarjeta
      modo_trx: modo
      hora_trx: hora
      id_linea_trx: id_linea
      id_ramal_trx:  
      interno_trx: interno_bus
      orden_trx: etapa_red_sube
      latitud_trx: lat
      longitud_trx: lon
      factor_expansion:  

   #Indicar si la informacion sobre la hora está en una columna separada. En nombres_variables debe indicarse el nombre. Dejar vacío en caso contrario 
   columna_hora: True 

   # Criterio para ordenar las transacciones en el tiempo. 'fecha_completa' utiliza el campo dado en fecha_trx mientras que `orden_trx` utiliza un entero incremental que se reinicia con cada viaje   
   ordenamiento_transacciones: orden_trx 

   # Cantidad de minutos de la ventana de tiempo para considerar diferentes etapas dentro de un mismo viaje
   ventana_viajes: 

   # Cantidad de minutos de la ventana de tiempo para considerar diferentes transacciones como una sola
   ventana_duplicado: 

   # Tipo de transacciones a elminar por no considerare usos en transporte publico. Indicar la columna y los valores para cada columna
   tipo_trx_invalidas:
      tipo_trx_tren:
         - 'CHECK OUT SIN CHECKIN'
         - 'CHECK OUT'

   # Especificar como se nombra a los modos en los archivos  
   modos:
      autobus: COL
      tren: TRE
      metro: SUB
      tranvia:
      brt:


Parámetros de posicionamiento GPS
---------------------------------

Este parámetro se utiliza para cuando existe una tabla separada con GPS que contenga el posicionamiento de los vehículos o internos. En ese caso, se gelocalizará cada transacción en base a la tabla GPS, uniendo por `id_linea` e `interno` (haciendo a este campo obligatorio) y minimizando el tiempo de la transacción con respecto a la transacción gps del interno de esa linea. Para eso el campo ``fecha`` debe estar completo con dia, hora y minutos. Esto hace obligatoria la existencia de un csv con la información de posicionamiento de los gps. Su nombre y atributos se especifican de modo similar a lo hecho en transacciones.


En ocasiones en la tabla de GPS puede haber información sobre los servicios prestados por cada vehículo. Para más detalles sobre esta configuración y cómo lo trabaja UrbanTrips ver el apartado **Servicios**.

.. code:: 

   geolocalizar_trx: True

   nombre_archivo_gps: gps_semana1.csv

   nombres_variables_gps:
      id_gps: 
      id_linea_gps: idlinea
      id_ramal_gps: c_ld_Id
      interno_gps: interno
      fecha_gps: date_time
      latitud_gps: latitude
      longitud_gps: longitude
      servicios_gps: TYPE
      velocity_gps: VELOCITY

   trust_service_type_gps: False
   valor_inicio_servicio: 7
   valor_fin_servicio: 9


Parámetro de lineas, ramales y paradas
--------------------------------------

Es necesario que se especifique si en el sistema de transporte existen lineas con ramales, tal como los entiende UrbanTrips (:doc:`lineas_ramales`). Esto debe indicarse en el parámetro ``lineas_contienen_ramales``.

Se puede agregar metadata para las lineas, como por ejemplo su nombre de fantasía ademas del id correspondiente, o a qué empresa pertenece. La misma puede identificar una linea o una linea-ramal. En este último caso UrbanTrips creara dos tablas diferentes, una para la metadata de las lineas y otra para la de ramales. 

Tambien permite agregar cartografías como los recorridos de las lineas y ramales, que deben ser una única Linestring en 2d (no permite multilineas). Si existe una tabla de recorridos, entonces debe proveerse un archivo con información de las lineas y ramales. Esta tabla puede identificar recorridos de lineas o tambien de lineas y ramales.

Por úlitmo, se puede especificar un archivo con la localización de las paradas o estaciones para cada linea y ramal, indicando un orden de paso o sucesión y también un ``node_id``, donde deben aparecer con un mismo id las paradas de diferentes ramales de una misma linea donde se pueda realizar un transbordo entre ramales. Para más información sobre estos datasets puede consultar :doc:`inputs`.

.. code:: 

   # Las lineas a ser utilizadas se subdividen en ramales?
   lineas_contienen_ramales: True

   # Nombre del archivo con la metadada de lineas y/o ramales
   nombre_archivo_informacion_lineas: lineas_amba_test.csv

   # Nombre del archivo con las rutas de las lineas y/o ramales
   recorridos_geojson: recorridos_amba.geojson

   # Nombre del archivo con las paradas para todas las lineas y/o ramales con orden y node_id 
   nombre_archivo_paradas: 



Parámetros de zonificaciones y polígonos de interés
---------------------------------------------------

Se pueden suministrar diferentes archivos con unidades espaciales o zonas de análisis de tránsito para las que se quiere agregar datos. Para cada archivo debe indicarse el nombre del archivo geojson a consumir, el nombre del atributo que contiene la información y, de ser necesario, un orden en el que se quiera producir las matrices OD que genera UrbanTrips. 

Puede haber tantos archivos como lo desee. Si existen estructuras anidadas (por ejemplo unidades censales de diferente nivel de agregación) se puede usar el mismo archivo, con diferentes atributos o columnas indicando los diferentes ids o valores para cada nivel de agregación. Luego se para el mismo archivo indicando en `var` qué atributo o columna tomar.

Estos archivos deben estar ubicados con el resto de los insumos de la ciudad en ``data/data_ciudad/``.

.. code:: 

   zonificaciones:
      geo1: hexs_amba.geojson
      var1: Corona
      orden1: ['CABA', 'Primer cordón', 'Segundo cordón', 'Tercer cordón', 'RMBA']
      geo2: hexs_amba.geojson
      var2: Partido
      
      
Al mismo tiempo, si se quiere realizar un análisis de patrones de orígenes y destinos para una determinada zona de interés, se puede suministrar otro archivo geojson donde se especifique una capa geográfica de polígonos en formato con las siguientes columnas `'id', 'tipo', 'geometry'`. `id` debe ser un texto con el nombre del polígono  de interés y `tipo` puede ser `poligono` o `cuenca`. El primero hace referencia a una zona particular de interés como un centro de transbordo o un barrio-localidad. El segundo hace referencia a una zona de mayor tamaño que puede agrupar el recorrido de un grupo de líneas o ser el área de influencia de una línea en particular.

El archivo puede contener la cantidad de polígonos (o cuencas) que se desee (ya sena polígnos o multi-polígonos), el proceso corre en forma independiente para cada poligoo o cuenca. Estos archivos deben estar ubicados con el resto de los insumos de la ciudad en ``data/data_ciudad/``. Estos archivos deben informarse en el archivo de configuraciones del siguiente modo. Los resultados se verán en el Dashboard.


.. code:: 

   poligonos: "[NOMBRE_DEL_ARCHIVO].geojson"  

