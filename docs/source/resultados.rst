Resultados finales
==================

Los resultados se guardarán en bases de ``SQLite``, donde cada base con sus tablas es un archivo en el disco. Los nombres de estos archivos están dados por el archivo de configuración (ver :doc:`configuracion`) y se ubican en ``data/db/``. Existen dos tipos de bases fundamentales ``data`` e ``insumos``. ``data`` guardará todo lo relativo a transacciónes, etapas, viajes, usuarios y toda información que se actualiza con cada corrida de Urbantrips que procesa una nueva tanda de datos. Utilizando los alias del archivo de configuracion, pueden crearse diferentes bases de tipo data, una para cada periodo agregado de tiempo. Así, puede haber una base de data diferente para cada semana o cada mes a medida que alcance un volumen determinado y utilizar un alias específico para este propósito configurable en el archivo de configuración (``ciudad_2023_semana1``, ``ciudad_2023_semana2``,etc). Por su lado, ``insumos`` es una base de datos que guardará información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de haxágonos H3 en una ciudad determinada, etc).


Modelo de datos de base ``data``
--------------------------------

Un grupo de tablas guardan los microdatos de las transacciónes, etapas, viajes, usuarios, servicios, gps, etc. Mientras que otro grupo de tablas guardarán los estadísticos que los analistas utilizando UrbanTrips vayan calculando para diferentes lineas.

**Tablas de microdatos**
           
.. list-table:: transacciónes
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id*
     - int
     - id unico que identifica cada transacción en esta base de datos.
   * - *id_original*
     - text
     - id de la transacción original en el csv usado en la corrida.
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta.
   * - *fecha*
     - datetime
     - fecha de la transacción.
   * - *dia*
     - text
     - dia de la transacción.
   * - *tiempo*
     - text
     - Hora minutos y segundos de la transacción en formato HH::MM::SS.
   * - *hora*
     - int
     - Hora de la transacción de 0 a 23.
   * - *modo*
     - text
     - Modo estandarizado de la transacción.
   * - *id_linea*
     - int
     - id de la linea utilizada en la transacción.
   * - *id_ramal*
     - int
     - id del ramal utilizado en la transacción.
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la transacción.
   * - *orden_trx*
     - int
     - entero incremental que indica la etapa dentro de una cadena de viajes.
   * - *latitud*
     - float
     - latitud.
   * - *longitud*
     - float
     - longitud.
   * - *factor_expansion*
     - float
     - factor de expansión original.

                        
            
.. list-table:: etapas
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id*
     - int
     - id unico que identifica cada transacción en esta base de datos.
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta.
   * - *dia*
     - text
     - dia de la etapa.
   * - *id_viaje*
     - text
     - id del viaje de esa tarjeta para ese dia.
   * - *id_etapa*
     - text
     - id de la etapa de esa tarjeta para ese dia y ese viaje.
   * - *tiempo*
     - text
     - Hora minutos y segundos del inicio de la etapa en formato HH::MM::SS.
   * - *hora*
     - int
     - Hora de la transacción de 0 a 23.
   * - *modo*
     - text
     - Modo estandarizado de la etapa.
   * - *id_linea*
     - int
     - id de la linea utilizada en la etapa.
   * - *id_ramal*
     - int
     - id del ramal utilizado en la etapa.
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la etapa.
   * - *latitud*
     - float
     - latitud.
   * - *longitud*
     - float
     - longitud.
   * - *h3_o*
     - text
     - índice H3 de las coordenadas de origen.
   * - *h3_d*
     - text
     - índice H3 de las coordenadas de destino.
   * - *od_validado*
     - int
     - indica si la etapa es válida (1) o no puede imputarse un destino (0). 
   * - *factor_expansion_original*
     - float
     - factor de expansión para las etapas.
   * - *factor_expansion_linea*
     - float
     - factor de expansión para las etapas que expande cada etapa para que el agregado por línea coincida con los totales por línea previo a filtrar datos inválidos.
   * - *factor_expansion_tarjeta*
     - float
     - factor de expansión para las etapas que expande cada etapa para que el agregado por tarjeta coincida con los totales por tarjeta previo a filtrar datos inválidos.



.. list-table:: viajes
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta.
   * - *id_viaje*
     - text
     - id del viaje de esa tarjeta para ese dia.
   * - *dia*
     - text
     - dia del viaje.
   * - *tiempo*
     - text
     - Hora minutos y segundos del inicio del viaje en formato HH::MM::SS.
   * - *hora*
     - int
     - Hora de la transacción de 0 a 23.
   * - *modo*
     - text
     - Modo estandarizado del viaje. Si es etapa simple toma el de la etapa, si no puede tomar valores Multietapa (mas de una etapa mismo modo) o Multimodal.
   * - *autobus*
     - int
     - cantidad de etapas hechas en este viaje en autobus.
   * - *tren*
     - int
     - cantidad de etapas hechas en este viaje en tren.
   * - *metro*
     - int
     - cantidad de etapas hechas en este viaje en metro.
   * - *tranvia*
     - int
     - cantidad de etapas hechas en este viaje en tranvia.
   * - *brt*
     - int
     - cantidad de etapas hechas en este viaje en brt.
   * - *otros*
     - int
     - cantidad de etapas hechas en este viaje en otros modos.
   * - *h3_o*
     - text
     - índice H3 de las coordenadas de origen.
   * - *h3_d*
     - text
     - índice H3 de las coordenadas de destino.  
   * - *od_validado*
     - int
     - indica si todas las etapas del viaje son válidas (1) o no puede imputarse un destino en alguna (0).
   * - *factor_expansion_linea*
     - float
     - factor de expansión para las etapas que expande cada etapa para que el agregado por línea coincida con los totales por línea previo a filtrar datos inválidos.
   * - *factor_expansion_tarjeta*
     - float
     - factor de expansión para las etapas que expande cada etapa para que el agregado por tarjeta coincida con los totales por tarjeta previo a filtrar datos inválidos.



.. list-table:: usuarios
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta.
   * - *dia*
     - text
     - dia del viaje.
   * - *od_validado*
     - int
     - indica si todas las etapas de todos los viajes son válidas (1) o no puede imputarse un destino en alguna (0).  
   * - *cant_viajes*
     - int
     - cantidad de viajes hechos por esa tarjeta en ese dia.
   * - *factor_expansion_linea*
     - float
     - factor de expansión para las etapas que expande cada etapa para que el agregado por línea coincida con los totales por línea previo a filtrar datos inválidos.
   * - *factor_expansion_tarjeta*
     - float
     - factor de expansión para las etapas que expande cada etapa para que el agregado por tarjeta coincida con los totales por tarjeta previo a filtrar datos inválidos.



.. list-table:: gps
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id*
     - int
     - id unico que identifica cada punto gps en esta base de datos.
   * - *id_original*
     - text
     - id del punto gps original en el csv usado en la corrida.
   * - *dia*
     - text
     - dia de la transacción.
   * - *id_linea*
     - int
     - id de la linea utilizada en la transacción.
   * - *id_ramal*
     - int
     - id del ramal utilizado en la transacción.
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la transacción.
   * - *fecha*
     - datetime
     - fecha de la transacción.
   * - *latitud*
     - float
     - latitud.
   * - *longitud*
     - float
     - longitud.
   * - *velocity*
     - float
     - velocidad instantanea del punto gps tal cual la registra el gps (campo opcional).
   * - *service_type*
     - text
     - indica el inicio y cierra del servicio tal cual lo declara el conductor del vehículo.
   * - *distance_km*
     - text
     - distancia en km entre puntos gps sucesivos en el tiempo para el mismo interno.
   * - *h3*
     - text
     - índice H3 de las coordenadas de destino.   


La tabla ``services`` agrupa los servicios ofertados por las diferentes lineas, sin clasificarlos por ramal. Cada servicio tiene un id tal cual fue identificado por el conductor del vehículo y otro tal como fue identificado por UrbanTrips. Para cada servicio se agregan algunos datos como la hora de inicio y de fin, la cantidad de puntos gps, el porcentaje de puntos donde el vehículo estuvo detenido, etc. Existe otra tabla relacionada a esta (``services_gps_points``) donde cada punto gps  de la tabla ``gps`` queda registrado en un nuevo servicio indicando el ``node_id`` más cercano y el ramal al que pertenece.

.. list-table:: services
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea del vehiculo.
   * - *dia*
     - text
     - dia de la transacción.
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la transacción.
   * - *original_service_id*
     - int
     - id del servicio tal cual lo declara el conductor del vehículo.
   * - *service_id*
     - int
     - id del servicio en función del criterio que sigue UrbanTrips.
   * - *total_points*
     - int
     - cantidad de puntos gps dentro del servicio.
   * - *distance_km*
     - text
     - distancia total en km recorrida en el servicio.
   * - *min_ts*
     - text
     - fecha de inicio del servicio en segundos Unix epoch.
   * - *max_ts*
     - text
     - fecha de fin del servicio en segundos Unix epoch.     
   * - *min_datetime*
     - text
     - fecha de inicio del servicio en formato YYYY-MM-DD HH:MM:SS.
   * - *max_datetime*
     - text
     - fecha de fin del servicio en formato YYYY-MM-DD HH:MM:SS.    
   * - *prop_idling*
     - float
     - proporción de puntos detenidos (distancia entre puntos menor a 100m) sobre el total de puntos gps.
   * - *valid*
     - int
     - indica si un servicio es considerado valido (1) o no (0) de acuerdo a si tiene mas de 5 puntos gps y una proporción de detención inferior a .5.
     
     
     
.. list-table:: services_gps_points
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id*
     - int
     - id unico que identifica cada punto gps en esta base de datos.
   * - *original_service_id*
     - int
     - id del servicio tal cual lo declara el conductor del vehículo.
   * - *new_service_id*
     - int
     - incremental que indica un nuevo servicio dentro del original_service_id.     
   * - *service_id*
     - int
     - id del servicio en función del criterio que sigue UrbanTrips.
   * - *id_ramal_gps_point*
     - int
     - id del ramal con el node_id más cercano.
   * - *node_od*
     - int
     - node_id del ramal más cercano.
     




**Tablas de estadisticos**

.. list-table:: indicadores
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *a*
     - int
     - 


.. list-table:: services_stats
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *a*
     - int
     - 



.. list-table:: ocupacion_por_linea_tramo
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *a*
     - int
     - 



.. list-table:: indicadores_operativos_linea
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id identificando la linea
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *tot_veh*
     - int
     - Total de vehículos ofertados en el día.
   * - *tot_km*
     - float
     - Total de kilómetros ofertados en el día.
   * - *tot_pax*
     - float
     - Total de pasajeros en el día.
   * - *dmt_mean*
     - float
     - Distancia media recorrida por pasajero.
   * - *dmt_median*
     - float
     - Distancia mediana recorrida por pasajero.
   * - *pvd*
     - float
     - Pasajeros promedio transportados por vehículo por día.
   * - *kvd*
     - float
     - Kilómetros promedio recorridos por vehículo por día.
   * - *ipk*
     - float
     - Índice Pasajero Kilómetro.
   * - *fo*
     - float
     - Factor de ocupación tomando 60 ubicaciónes por vehículo.



 
            

Modelo de datos de base ``insumos``
-----------------------------------

El siguiente grupo de tablas almacena información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de haxágonos H3 en una ciudad determinada, etc). 


.. list-table:: metadata_lineas 
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 
     
.. list-table:: metadata_ramales
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 

.. list-table:: matriz_validacion
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 

.. list-table:: lines_geoms
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 

.. list-table:: branches_geoms
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 

.. list-table:: stops
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 

.. list-table:: zonas
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 

.. list-table:: distancias
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *s*
     - int
     - 
    


