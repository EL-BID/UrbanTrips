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
     - factor de expansión original tal cual viene en transacciones. Si no es una muestra es 1.

                        
            
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
   * - *factor_expansion_linea*
     - float
     - factor de expansión para las etapas que expande cada etapa de modo que que el agregado por línea coincida con los totales por línea previo a filtrar datos inválidos.
   * - *factor_expansion_tarjeta*
     - float
     - factor de expansión para las etapas que expande cada etapa modo que el agregado por tarjeta coincida con los totales por tarjeta previo a filtrar datos inválidos.



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
     - dia del inicio del servicio.
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
    
Las tablas a continuación indican el origen y el destino de cada etapa con respecto a cada gps o estación.

.. list-table:: legs_to_gps_origin
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_legs*
     - int
     - id de la tabla etapas
   * - *id_gps*
     - int
     - id de la tabla gps 


.. list-table:: legs_to_gps_destination
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_legs*
     - int
     - id de la tabla etapas
   * - *id_gps*
     - int
     - id de la tabla gps 


.. list-table:: legs_to_station_origin
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_legs*
     - int
     - id de la tabla etapas
   * - *id_station*
     - int
     - id de la tabla travel_times_stations 

.. list-table:: legs_to_station_destination
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_legs*
     - int
     - id de la tabla etapas
   * - *id_station*
     - int
     - id de la tabla travel_times_stations 


**Tablas de estadisticos**

Estas tablas contienen estadísticos calculados por UrbanTrips. Algunos estádisticos serán calculados por defecto pero otros solo serán calculados luego de que cada analista los haya obtenido utilizando las diferentes herramientas de UrbanTrips. Estos no se corren de modo automático para todas las lineas, cada una debe ser procesada individualmente y con los parámetros necesarios que la función que calcula cada estadístico requiera.


.. list-table:: indicadores
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - dia de lastransacciones procesadas en esta corrida.
   * - *detalle*
     - text
     - tipo de indicador a calcular.
   * - *indicador*
     - int
     - cantidad de observaciones registradas para ese indicador en ese dia.          
   * - *tabla*
     - text
     - tabla donde	se almacenan esas observaciones.
   * - *nivel*
     - int
     - nivel del indicador que expresa la relación con un indicador previo más global.
   * - *porcentaje*
     - float
     - la cantidad de observaciones expresadas como porcentaje de un indicador previo más global.     
     


.. list-table:: services_stats
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea de los servicios.
   * - *dia*
     - text
     - dia de los datos sobre los cuales se calculan los estadísticos.
   * - *cant_servicios_originales*
     - int
     - cantidad de servicios tal como fueron declarados por el conductor del vehículo y declarado en la tabla gps.
   * - *cant_servicios_nuevos*
     - int
     - cantidad de servicios tal como fueron inferidos por UrbanTrips.
   * - *cant_servicios_nuevos_validos*
     - int
     - cantidad de servicios inferidos por UrbanTrips considerados validos.
   * - *n_servicios_nuevos_cortos*
     - int
     - cantidad de servicios inferidos por UrbanTrips con menos de 5 puntos gps.
   * - *prop_servicios_cortos_nuevos_idling*
     - float
     - proporcion de servicios nuevos cortos que se encuentran detenidos.
   * - *distancia_recorrida_original*
     - int
     - distancia acumulada en km por servicios tal como fueron declarados por el conductor del vehículo y declarado en la tabla gps.     
   * - *prop_distancia_recuperada*
     - int
     - proporción de la distancia recorrida original recuperada en los servicios validos inferidos por UrbanTrips.      
   * - *servicios_originales_sin_dividir*
     - float
     - proporción de servicios originales dentro de los cuales hay uno y solo un servicio valido inferio por UrbanTrips.
      

.. list-table:: services_by_line_hour
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea de los servicios.
   * - *dia*
     - text
     - dia de los datos sobre los cuales se calculan los estadísticos.
   * - *hora*
     - int
     - hora del dia sobre la cual se calculan los estadísticos.
   * - *servicios*
     - int
     - cantidad de servicios por dia y hora.


.. list-table:: ocupacion_por_linea_tramo
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea sobre el que se calculan los estadísticos.
   * - *yr_mo*
     - text
     - mes y anio sobre el que se calculan los estadísticos.
   * - *day_type*
     - text
     - dia o tipo de día sobre el cual se calcula los estadísticos.
   * - *n_sections*
     - int
     - cantidad de secciones en las que se segmentó el recorrido representativo de la linea.
   * - *section_meters*
     - int
     - largo en metros de la sección sobre la que se computa los estadísticos de ocupación. 
   * - *sentido*
     - text
     - sentido de las etapas utilizadas para calcular los estadísticos de ocupación.
   * - *section_id*
     - float
     - id de la sección para la que se calcula los estadísticos. Se une con routes_section_id_coords.
   * - *hour_min*
     - int
     - hora mínima de las etapas a utilizar para calcular los estadísticos.
   * - *hour_max*
     - int
     - hora máxima de las etapas a utilizar para calcular los estadísticos.
   * - *legs*
     - int
     - cantidad de etapas en esa sección para ese día y horas.
   * - *prop*
     - float
     - proporción de etapas de esa sección sobre el total de etapas para ese día y horas.



.. list-table:: kpi_by_day_line
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
   * - *fo_mean*
     - float
     - Factor de ocupación tomando 60 ubicaciónes por vehículo tomando la DMT promedio.
   * - *fo_median*
     - float
     - Factor de ocupación tomando 60 ubicaciónes por vehículo tomando la DMT mediana.



 
.. list-table:: kpi_by_day_line_service
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
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la transacción.
   * - *service_id*
     - int
     - numero de service dentro del vehiculo o interno para esa linea y dia.
   * - *hora_inicio*
     - float
     - hora de inicio del servicio.
   * - *hora_fin*
     - float
     - hora de cierre del servicio.
   * - *tot_km*
     - float
     - Total de kilómetros ofertados por el servicio.
   * - *tot_pax*
     - float
     - Total de pasajeros transportados por el servicio.
   * - *dmt_mean*
     - float
     - Distancia media recorrida por pasajero del servicio.
   * - *dmt_median*
     - float
     - Distancia mediana recorrida por pasajero del servicio.
   * - *ipk*
     - float
     - Índice Pasajero Kilómetro.
   * - *fo_mean*
     - float
     - Factor de ocupación tomando 60 ubicaciónes por vehículo tomando la DMT promedio.
   * - *fo_median*
     - float
     - Factor de ocupación tomando 60 ubicaciónes por vehículo tomando la DMT mediana.            
            


.. list-table:: basic_kpi_by_vehicle_hr
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_linea*
     - int
     - id identificando la linea
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la transacción.
   * - *hora*
     - float
     - hora en la que se encuentra circulando el vehiculo.
   * - *tot_pax*
     - float
     - Total de pasajeros transportados por el vehiculo para esa hora.
   * - *eq_pax*
     - float
     - Total de pasajeros equivalentes transportados por ese vehículo durante esa hora.
   * - *dmt*
     - float
     - Distancia media recorrida por pasajero del vehiculo para esa hora.
   * - *of*
     - float
     - Factor de ocupación calculado como la relación entre la DMT y la velocidad comercial.
   * - *speed_kmh*
     - float
     - Velocidad comercial promedio de ese vehiculo a esa hora.



.. list-table:: basic_kpi_by_line_hr
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_linea*
     - int
     - id identificando la linea
   * - *hora*
     - float
     - hora del día.
   * - *veh*
     - float
     - Total de vehículos únicos circulando a esa hora para esa linea y día.
   * - *pax*
     - float
     - Total de pasajeros que iniciaron una etapa en esa linea a esa hora y día.
   * - *dmt*
     - float
     - Distancia media recorrida por pasajero del vehiculo para esa hora y día.
   * - *of*
     - float
     - Factor de ocupación promedio calculado como la relación entre la DMT y la velocidad comercial.
   * - *speed_kmh*
     - float
     - Velocidad comercial promedio de esa línea a esa hora para ese día.


.. list-table:: basic_kpi_by_line_day
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_linea*
     - int
     - id identificando la linea
   * - *veh*
     - float
     - Total de vehículos únicos circulando para esa linea y día.
   * - *pax*
     - float
     - Total de pasajeros que utilizaron esa linea ese día.
   * - *dmt*
     - float
     - Distancia media recorrida por pasajero en esa línea ese día.
   * - *of*
     - float
     - Factor de ocupación promedio calculado como la relación entre la DMT y la velocidad comercial.
   * - *speed_kmh*
     - float
     - Velocidad comercial promedio de esa línea para ese día.



.. list-table:: overlapping
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *base_line_id*
     - int
     - id identificando la linea de base
   * - *base_branch_id*
     - int
     - id identificando el ramal de base
   * - *comp_line_id*
     - int
     - id identificando la linea de comparacion	
   * - *comp_branch_id*
     - int
     - id identificando el ramal de comparacion
   * - *res_h3*
     - int
     - resolucion h3 utilizada en el computo del estadistico
   * - *overlap*
     - float
     - estadistico de solapamiento
   * - *type_overlap*
     - text
     - tipo de estadistico de solapamiento (oferta o demanda)    




.. list-table:: lines_od_matrix_by_section
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea sobre el que se calculan los estadísticos.
   * - *yr_mo*
     - text
     - mes y anio sobre el que se calculan los estadísticos.
   * - *day_type*
     - text
     - dia o tipo de día sobre el cual se calcula los estadísticos.
   * - *n_sections*
     - int
     - cantidad de secciones en las que se segmentó el recorrido representativo de la linea.
   * - *section_id_o*
     - float
     - id de la sección de origen para la que se calcula los estadísticos. Se une con routes_section_id_coords.
   * - *section_id_d*
     - float
     - id de la sección de destino para la que se calcula los estadísticos. Se une con routes_section_id_coords.
   * - *hour_min*
     - int
     - hora mínima de las etapas a utilizar para calcular los estadísticos.
   * - *hour_max*
     - int
     - hora máxima de las etapas a utilizar para calcular los estadísticos.
   * - *legs*
     - int
     - cantidad de etapas en esa sección para ese día y horas.
   * - *prop*
     - float
     - proporción de etapas de esa sección sobre el total de etapas para ese día y horas.


Modelo de datos de base ``insumos``
-----------------------------------

El siguiente grupo de tablas almacena información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de haxágonos H3 en una ciudad determinada, etc). 


.. list-table:: metadata_lineas 
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea
   * - *nombre_linea*
     - text
     - nombre de fantasía de la linea o el que figura en el cartel 
   * - *modo*
     - text
     - Modo estandarizado de la linea.
   * - *empresa*
     - text
     - empresa a la que pertenece la linea.     
   * - *descripcion*
     - text
     - campo que almacena algun texto descriptivo de la linea.        
     
.. list-table:: metadata_ramales
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_ramal*
     - int
     - id del ramal
   * - *id_linea*
     - int
     - id de la linea a la que pertenece el ramal
   * - *nombre_ramal*
     - text
     - nombre de fantasía del ramal o el que figura en el cartel 
   * - *modo*
     - text
     - Modo estandarizado del ramal.
   * - *empresa*
     - text
     - empresa a la que pertenece la linea.     
   * - *descripcion*
     - text
     - campo que almacena algun texto descriptivo del ramal	.        

.. list-table:: matriz_validacion
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea a la que pertenece la parada.
   * - *parada*
     - text
     - coordenada h3 a la que pertenece la parada.
   * - *area_influencia*
     - text
     - coordenada h3 de una celda adyacente parte del area de influencia de la parada.
     
     

.. list-table:: lines_geoms
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea al que pertenece el recorrido.
   * - *wkt*
     - text
     - recorrido de la linea en formato WKT.
     
     

.. list-table:: branches_geoms
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_ramal*
     - int
     - id del ramal al que pertenece el recorrido.
   * - *wkt*
     - text
     - recorrido del ramal en formato WKT.

.. list-table:: stops
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea a la que pertenece la parada.
   * - *id_ramal*
     - int
     - id del ramal al que pertenece la parada.
   * - *node_id*
     - int
     - id del nodo que unifica paradas de la misma linea en un nodo unico.
   * - *branch_stop_order*
     - int
     - orden de paso de la parada en el ramal.
   * - stop_x
     - float
     - coordenada de la parada.     
   * - stop_y
     - float
     - coordenada de la parada.
   * - node_x
     - float
     - coordenada del nodo.
   * - node_y
     - float
     - coordenada del nodo.


.. list-table:: zonas
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *h3*
     - text
     - indice h3.
   * - *fex*
     - float
     - cantidad de etapas con origen en el indice. 
   * - *latitud*
     - float
     - latitud del centroide del indice h3.
   * - *longitud*
     - float
     - longitud del centroide del indice h3.
   * - *Zona_voi*
     - int
     - id de la zona voronoi al que pertenece el indice h3.     
     
     

.. list-table:: distancias
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *h3_o*
     - text
     - indice h3 de origen.
   * - *h3_d*
     - text
     - indice h3 de destino.    
   * - *h3_o_norm*
     - text
     - indice h3 de origen con el sentido normalizado.
   * - *h3_d_norm*
     - text
     - indice h3 de destino con el sentido normalizado.
   * - *distance_osm_drive*
     - float
     - distancia calculada sobre red de callejero en km - manejando.    
   * - *distance_osm_walk*
     - float
     - distancia calculada sobre red de callejero en km- caminando.
   * - *distance_h3*
     - float
     - distancia euclidiana en km.



.. list-table:: routes_section_id_coords
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - int
     - id de la linea del tramo del recorrido.
   * - *n_sections*
     - int
     - cantidad de secciones en el que se fragmento el recorrido de la linea.
   * - *section*
     - int
     - id del segmento del recorrido
   * - *section_lrs*
     - float
     - proporcion del recorrido en el que se encuentra el inicio del tramo
   * - *x*
     - float
     - longitud del inicio del tramo     
   * - *y*
     - float
     - latitud del inicio del tramo
