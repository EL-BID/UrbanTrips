Resultados finales
==================

Los resultados se guardarán en bases de datos ``DuckDB``, donde cada base con sus tablas es un archivo en el disco. Los nombres de estos archivos están dados por el archivo de configuración (ver :doc:`configuracion`) y se ubican en ``data/db/``. Existen tres tipos de bases fundamentales: ``data``, ``insumos`` y ``dashboard``. 

La base ``data`` guardará todo lo relativo a transacciones, etapas, viajes, usuarios y toda información que se actualiza con cada corrida de Urbantrips que procesa una nueva tanda de datos. Utilizando los alias del archivo de configuracion, pueden crearse diferentes bases de tipo data, una para cada periodo agregado de tiempo. Así, puede haber una base de data diferente para cada semana o cada mes a medida que alcance un volumen determinado y utilizar un alias específico para este propósito configurable en el archivo de configuración (``ciudad_2023_semana1``, ``ciudad_2023_semana2``, etc). 

Por su lado, ``insumos`` es una base de datos que guardará información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de hexágonos H3 en una ciudad determinada, etc).

Finalmente, ``dashboard`` contiene tablas agregadas y procesadas específicamente para la visualización interactiva en el dashboard de UrbanTrips.


Modelo de datos de base ``data``
--------------------------------

Un grupo de tablas guardan los microdatos de las transacciónes, etapas, viajes, usuarios, servicios, gps, etc. Mientras que otro grupo de tablas guardarán los estadísticos que los analistas utilizando UrbanTrips vayan calculando para diferentes lineas.

**Tablas de microdatos**
           
.. list-table:: transacciones
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id*
     - int
     - id unico que identifica cada transacción en esta base de datos (PRIMARY KEY).
   * - *batch_id*
     - int
     - identificador del lote de procesamiento.
   * - *fecha*
     - int
     - fecha de la transacción en formato Unix epoch.
   * - *id_original*
     - text
     - id de la transacción original en el csv usado en la corrida.
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta.
   * - *dia*
     - text
     - dia de la transacción.
   * - *tiempo*
     - text
     - Hora minutos y segundos de la transacción en formato HH:MM:SS.
   * - *hora*
     - int
     - Hora de la transacción de 0 a 23.
   * - *modo*
     - text
     - Modo estandarizado de la transacción.
   * - *id_linea*
     - bigint
     - id de la linea utilizada en la transacción.
   * - *id_ramal*
     - bigint
     - id del ramal utilizado en la transacción.
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la transacción.
   * - *orden_trx*
     - int
     - entero incremental que indica la etapa dentro de una cadena de viajes.
   * - *genero*
     - text
     - género del usuario (campo opcional).
   * - *tarifa*
     - text
     - tipo de tarifa aplicada (campo opcional).
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
     - id unico que identifica cada etapa en esta base de datos (PRIMARY KEY).
   * - *batch_id*
     - int
     - identificador del lote de procesamiento.
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta.
   * - *dia*
     - text
     - dia de la etapa.
   * - *id_viaje*
     - int
     - id del viaje de esa tarjeta para ese dia.
   * - *id_etapa*
     - int
     - id de la etapa de esa tarjeta para ese dia y ese viaje.
   * - *tiempo*
     - text
     - Hora minutos y segundos del inicio de la etapa en formato HH:MM:SS.
   * - *hora*
     - int
     - Hora de la transacción de 0 a 23.
   * - *modo*
     - text
     - Modo estandarizado de la etapa.
   * - *id_linea*
     - bigint
     - id de la linea utilizada en la etapa.
   * - *id_ramal*
     - bigint
     - id del ramal utilizado en la etapa.
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la etapa.
   * - *genero*
     - text
     - género del usuario (campo opcional).
   * - *tarifa*
     - text
     - tipo de tarifa aplicada (campo opcional).
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
   * - *etapa_validada*
     - int
     - indica si la etapa cumple con criterios de validación.
   * - *factor_expansion_original*
     - float
     - factor de expansión original de la transacción.
   * - *factor_expansion_linea*
     - float
     - factor de expansión para las etapas que expande cada etapa de modo que el agregado por línea coincida con los totales por línea previo a filtrar datos inválidos.
   * - *factor_expansion_tarjeta*
     - float
     - factor de expansión para las etapas que expande cada etapa de modo que el agregado por tarjeta coincida con los totales por tarjeta previo a filtrar datos inválidos.
   * - *factor_expansion_etapa*
     - float
     - factor de expansión específico de la etapa.
   * - *distancia*
     - float
     - distancia recorrida en la etapa en kilómetros.
   * - *travel_time_min*
     - float
     - tiempo de viaje en minutos.



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
     - int
     - id del viaje de esa tarjeta para ese dia.
   * - *dia*
     - text
     - dia del viaje.
   * - *tiempo*
     - text
     - Hora minutos y segundos del inicio del viaje en formato HH:MM:SS.
   * - *hora*
     - int
     - Hora de inicio del viaje de 0 a 23.
   * - *cant_etapas*
     - int
     - cantidad total de etapas en el viaje.
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
   * - *cable*
     - int
     - cantidad de etapas hechas en este viaje en cable.
   * - *lancha*
     - int
     - cantidad de etapas hechas en este viaje en lancha.
   * - *otros*
     - int
     - cantidad de etapas hechas en este viaje en otros modos.
   * - *h3_o*
     - text
     - índice H3 de las coordenadas de origen.
   * - *h3_d*
     - text
     - índice H3 de las coordenadas de destino.
   * - *genero*
     - text
     - género del usuario (campo opcional).
   * - *tarifa*
     - text
     - tipo de tarifa aplicada (campo opcional).
   * - *od_validado*
     - int
     - indica si todas las etapas del viaje son válidas (1) o no puede imputarse un destino en alguna (0).
   * - *factor_expansion_linea*
     - float
     - factor de expansión para que el agregado por línea coincida con los totales por línea previo a filtrar datos inválidos.
   * - *factor_expansion_tarjeta*
     - float
     - factor de expansión para que el agregado por tarjeta coincida con los totales por tarjeta previo a filtrar datos inválidos.
   * - *distancia*
     - float
     - distancia total recorrida en el viaje en kilómetros.
   * - *travel_time_min*
     - float
     - tiempo total de viaje en minutos.



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
     - id unico que identifica cada punto gps en esta base de datos (PRIMARY KEY).
   * - *id_original*
     - text
     - id del punto gps original en el csv usado en la corrida.
   * - *dia*
     - text
     - dia del registro GPS.
   * - *id_linea*
     - bigint
     - id de la linea del vehículo.
   * - *id_ramal*
     - bigint
     - id del ramal del vehículo.
   * - *interno*
     - int
     - numero de interno o vehículo.
   * - *fecha*
     - int
     - fecha del registro en formato Unix epoch.
   * - *latitud*
     - float
     - latitud.
   * - *longitud*
     - float
     - longitud.
   * - *velocity*
     - float
     - velocidad instantanea del punto gps tal cual la registra el gps (campo opcional).
   * - *id_servicio*
     - text
     - identificador del servicio.
   * - *service_type*
     - text
     - indica el inicio y cierre del servicio tal cual lo declara el conductor del vehículo.
   * - *distance_km*
     - float
     - distancia en km entre puntos gps sucesivos en el tiempo para el mismo interno.
   * - *distance_servicio_mts*
     - float
     - distancia acumulada del servicio en metros.
   * - *distance_servicio_mts_agg*
     - float
     - distancia acumulada agregada del servicio en metros.
   * - *h3*
     - text
     - índice H3 de las coordenadas del punto GPS.   


La tabla ``services`` agrupa los servicios ofertados por las diferentes lineas, sin clasificarlos por ramal. Cada servicio tiene un id tal cual fue identificado por el conductor del vehículo y otro tal como fue identificado por UrbanTrips. Para cada servicio se agregan algunos datos como la hora de inicio y de fin, la cantidad de puntos gps, el porcentaje de puntos donde el vehículo estuvo detenido, etc. Existe otra tabla relacionada a esta (``services_gps_points``) donde cada punto gps  de la tabla ``gps`` queda registrado en un nuevo servicio indicando el ``node_id`` más cercano y el ramal al que pertenece.

.. list-table:: services
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea del vehiculo.
   * - *id_ramal*
     - bigint
     - id del ramal del servicio.
   * - *dia*
     - text
     - dia del inicio del servicio.
   * - *interno*
     - int
     - numero de interno o vehículo utilizado.
   * - *original_service_id*
     - int
     - id del servicio tal cual lo declara el conductor del vehículo.
   * - *service_id*
     - int
     - id del servicio en función del criterio que sigue UrbanTrips.
   * - *total_points*
     - int
     - cantidad de puntos gps dentro del servicio.
   * - *distance_route*
     - float
     - distancia total en km recorrida en el servicio según la ruta.
   * - *distance_route_gps*
     - float
     - distancia total en km recorrida en el servicio según puntos GPS.
   * - *min_ts*
     - int
     - fecha de inicio del servicio en segundos Unix epoch.
   * - *max_ts*
     - int
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
     - id unico que identifica cada punto gps en esta base de datos (PRIMARY KEY).
   * - *id_linea*
     - bigint
     - id de la linea del punto GPS.
   * - *id_ramal*
     - bigint
     - id del ramal del punto GPS.
   * - *interno*
     - int
     - numero de interno o vehículo.
   * - *dia*
     - text
     - dia del registro.
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
     - bigint
     - id del ramal con el node_id más cercano.
   * - *node_id*
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


.. list-table:: travel_times_gps
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id*
     - int
     - id del registro GPS
   * - *travel_time_min*
     - float
     - tiempo de viaje en minutos
   * - *travel_speed*
     - float
     - velocidad de viaje en km/h


.. list-table:: travel_times_legs
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día
   * - *id*
     - int
     - id de la etapa
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta
   * - *id_viaje*
     - int
     - id del viaje
   * - *id_etapa*
     - int
     - id de la etapa
   * - *travel_time_min*
     - float
     - tiempo de viaje en minutos
   * - *distance_od*
     - float
     - distancia origen-destino en km
   * - *distance_route*
     - float
     - distancia según ruta en km
   * - *distance_route_gps*
     - float
     - distancia según ruta GPS en km
   * - *kmh_od*
     - float
     - velocidad según origen-destino en km/h
   * - *kmh_route*
     - float
     - velocidad según ruta en km/h
   * - *kmh_route_gps*
     - float
     - velocidad según ruta GPS en km/h
   * - *id_gps_o*
     - int
     - id del punto GPS de origen
   * - *id_gps_d*
     - int
     - id del punto GPS de destino


.. list-table:: travel_times_trips
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta
   * - *id_viaje*
     - int
     - id del viaje
   * - *travel_time_min*
     - float
     - tiempo de viaje total en minutos
   * - *distance_od*
     - float
     - distancia origen-destino en km
   * - *distance_route*
     - float
     - distancia según ruta en km
   * - *distance_route_gps*
     - float
     - distancia según ruta GPS en km
   * - *kmh_od*
     - float
     - velocidad según origen-destino en km/h
   * - *kmh_route*
     - float
     - velocidad según ruta en km/h
   * - *kmh_route_gps*
     - float
     - velocidad según ruta GPS en km/h


.. list-table:: vehicle_expansion_factors
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea
   * - *dia*
     - text
     - día del cálculo
   * - *unique_vehicles*
     - int
     - cantidad de vehículos únicos
   * - *broken_gps_veh*
     - int
     - cantidad de vehículos con GPS no funcional
   * - *veh_exp*
     - float
     - factor de expansión de vehículos


.. list-table:: transacciones_linea
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - día de las transacciones
   * - *id_linea*
     - bigint
     - id de la linea
   * - *transacciones*
     - float
     - cantidad total de transacciones


.. list-table:: tarjetas_duplicadas
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - día del registro
   * - *id_tarjeta_original*
     - text
     - id original de la tarjeta
   * - *id_tarjeta_nuevo*
     - text
     - nuevo id de la tarjeta


**Tablas de estadisticos**

Estas tablas contienen estadísticos calculados por UrbanTrips. Algunos estádisticos serán calculados por defecto pero otros solo serán calculados luego de que cada analista los haya obtenido utilizando las diferentes herramientas de UrbanTrips. Estos no se corren de modo automático para todas las lineas, cada una debe ser procesada individualmente y con los parámetros necesarios que la función que calcula cada estadístico requiera.


.. list-table:: services_stats
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea de los servicios.
   * - *id_ramal*
     - bigint
     - id del ramal de los servicios.
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
   * - *prop_servicos_cortos_nuevos_idling*
     - float
     - proporcion de servicios nuevos cortos que se encuentran detenidos.
   * - *distancia_recorrida_original*
     - float
     - distancia acumulada en km por servicios tal como fueron declarados por el conductor del vehículo y declarado en la tabla gps.
   * - *prop_distancia_recuperada*
     - float
     - proporción de la distancia recorrida original recuperada en los servicios validos inferidos por UrbanTrips.
   * - *servicios_originales_sin_dividir*
     - float
     - proporción de servicios originales dentro de los cuales hay uno y solo un servicio valido inferido por UrbanTrips.
      

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
     - bigint
     - id identificando la linea
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *tot_veh*
     - int
     - Total de vehículos ofertados en el día.
   * - *tot_km*
     - float
     - Total de kilómetros ofertados en el día según la ruta.
   * - *tot_km_gps*
     - float
     - Total de kilómetros ofertados en el día según GPS.
   * - *tot_pax*
     - float
     - Total de pasajeros en el día.
   * - *dmt_mean_od*
     - float
     - Distancia media recorrida por pasajero según origen-destino.
   * - *dmt_mean_route*
     - float
     - Distancia media recorrida por pasajero según ruta.
   * - *dmt_mean_route_gps*
     - float
     - Distancia media recorrida por pasajero según ruta GPS.
   * - *dmt_median_od*
     - float
     - Distancia mediana recorrida por pasajero según origen-destino.
   * - *dmt_median_route*
     - float
     - Distancia mediana recorrida por pasajero según ruta.
   * - *dmt_median_route_gps*
     - float
     - Distancia mediana recorrida por pasajero según ruta GPS.
   * - *pvd*
     - float
     - Pasajeros promedio transportados por vehículo por día.
   * - *kvd*
     - float
     - Kilómetros promedio recorridos por vehículo por día según ruta.
   * - *kvd_gps*
     - float
     - Kilómetros promedio recorridos por vehículo por día según GPS.
   * - *ipk_route*
     - float
     - Índice Pasajero Kilómetro según ruta.
   * - *ipk_route_gps*
     - float
     - Índice Pasajero Kilómetro según ruta GPS.
   * - *fo_mean_od*
     - float
     - Factor de ocupación promedio tomando 60 ubicaciones por vehículo según origen-destino.
   * - *fo_mean_route*
     - float
     - Factor de ocupación promedio tomando 60 ubicaciones por vehículo según ruta.
   * - *fo_mean_route_gps*
     - float
     - Factor de ocupación promedio tomando 60 ubicaciones por vehículo según ruta GPS.
   * - *fo_median_od*
     - float
     - Factor de ocupación mediano tomando 60 ubicaciones por vehículo según origen-destino.
   * - *fo_median_route*
     - float
     - Factor de ocupación mediano tomando 60 ubicaciones por vehículo según ruta.
   * - *fo_median_route_gps*
     - float
     - Factor de ocupación mediano tomando 60 ubicaciones por vehículo según ruta GPS.



 
.. list-table:: kpi_by_day_line_service
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id identificando la linea
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *id_ramal*
     - bigint
     - id del ramal del servicio.
   * - *interno*
     - text
     - numero de interno o vehículo utilizado.
   * - *service_id*
     - int
     - numero de servicio dentro del vehiculo o interno para esa linea y dia.
   * - *hora_inicio*
     - text
     - hora de inicio del servicio.
   * - *hora_fin*
     - text
     - hora de cierre del servicio.
   * - *tot_km*
     - float
     - Total de kilómetros ofertados por el servicio según ruta.
   * - *tot_km_gps*
     - float
     - Total de kilómetros ofertados por el servicio según GPS.
   * - *tot_pax*
     - float
     - Total de pasajeros transportados por el servicio.
   * - *dmt_mean_od*
     - float
     - Distancia media recorrida por pasajero según origen-destino.
   * - *dmt_mean_route*
     - float
     - Distancia media recorrida por pasajero según ruta.
   * - *dmt_mean_route_gps*
     - float
     - Distancia media recorrida por pasajero según ruta GPS.
   * - *dmt_median_od*
     - float
     - Distancia mediana recorrida por pasajero según origen-destino.
   * - *dmt_median_route*
     - float
     - Distancia mediana recorrida por pasajero según ruta.
   * - *dmt_median_route_gps*
     - float
     - Distancia mediana recorrida por pasajero según ruta GPS.
   * - *ipk_route*
     - float
     - Índice Pasajero Kilómetro según ruta.
   * - *ipk_route_gps*
     - float
     - Índice Pasajero Kilómetro según ruta GPS.
   * - *fo_mean_od*
     - float
     - Factor de ocupación promedio tomando 60 ubicaciones por vehículo según origen-destino.
   * - *fo_mean_route*
     - float
     - Factor de ocupación promedio tomando 60 ubicaciones por vehículo según ruta.
   * - *fo_mean_route_gps*
     - float
     - Factor de ocupación promedio tomando 60 ubicaciones por vehículo según ruta GPS.
   * - *fo_median_od*
     - float
     - Factor de ocupación mediano tomando 60 ubicaciones por vehículo según origen-destino.
   * - *fo_median_route*
     - float
     - Factor de ocupación mediano tomando 60 ubicaciones por vehículo según ruta.
   * - *fo_median_route_gps*
     - float
     - Factor de ocupación mediano tomando 60 ubicaciones por vehículo según ruta GPS.            
            


.. list-table:: overlapping_by_route
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - Fecha del día para el cual fue computado el estadístico
   * - *base_line_id*
     - bigint
     - id identificando la linea de base
   * - *base_branch_id*
     - bigint
     - id identificando el ramal de base
   * - *comp_line_id*
     - bigint
     - id identificando la linea de comparacion
   * - *comp_branch_id*
     - bigint
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




Modelo de datos de base ``insumos``
-----------------------------------

El siguiente grupo de tablas almacena información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de hexágonos H3 en una ciudad determinada, etc). 


.. list-table:: metadata_lineas 
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea (PRIMARY KEY)
   * - *nombre_linea*
     - text
     - nombre de fantasía de la linea o el que figura en el cartel 
   * - *id_linea_agg*
     - bigint
     - id de la linea agregada (para agrupación de líneas).
   * - *nombre_linea_agg*
     - text
     - nombre de la linea agregada.
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
     - bigint
     - id del ramal (PRIMARY KEY)
   * - *id_linea*
     - bigint
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
     - campo que almacena algun texto descriptivo del ramal.        

.. list-table:: matriz_validacion
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea_agg*
     - bigint
     - id de la linea agregada a la que pertenece la parada.
   * - *id_ramal*
     - bigint
     - id del ramal al que pertenece la parada.
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
     - bigint
     - id de la linea al que pertenece el recorrido (PRIMARY KEY).
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
     - bigint
     - id del ramal al que pertenece el recorrido (PRIMARY KEY).
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
     - bigint
     - id de la linea a la que pertenece la parada.
   * - *id_ramal*
     - bigint
     - id del ramal al que pertenece la parada.
   * - *node_id*
     - int
     - id del nodo que unifica paradas de la misma linea en un nodo unico.
   * - *branch_stop_order*
     - int
     - orden de paso de la parada en el ramal.
   * - *stop_x*
     - float
     - coordenada x (longitud) de la parada.
   * - *stop_y*
     - float
     - coordenada y (latitud) de la parada.
   * - *node_x*
     - float
     - coordenada x (longitud) del nodo.
   * - *node_y*
     - float
     - coordenada y (latitud) del nodo.


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
     - distancia calculada sobre red de callejero en km - caminando.
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
     - bigint
     - id de la linea del tramo del recorrido.
   * - *n_sections*
     - int
     - cantidad de secciones en el que se fragmento el recorrido de la linea.
   * - *section_id*
     - int
     - id del segmento del recorrido
   * - *section_lrs*
     - float
     - proporcion del recorrido en el que se encuentra el inicio del tramo
   * - *x*
     - float
     - longitud (coordenada x) del inicio del tramo
   * - *y*
     - float
     - latitud (coordenada y) del inicio del tramo


.. list-table:: poligonos
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id*
     - text
     - identificador único del polígono (PRIMARY KEY)
   * - *tipo*
     - text
     - tipo de polígono (zona, región, etc.)
   * - *wkt*
     - text
     - geometría del polígono en formato WKT


.. list-table:: official_branches_geoms
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_ramal*
     - bigint
     - id del ramal (PRIMARY KEY)
   * - *wkt*
     - text
     - geometría oficial del recorrido del ramal en formato WKT


.. list-table:: official_branches_geoms_h3
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_ramal*
     - bigint
     - id del ramal (PRIMARY KEY)
   * - *section_id*
     - int
     - id de la sección del recorrido
   * - *h3*
     - text
     - índice H3 de la sección
   * - *wkt*
     - text
     - geometría de la sección en formato WKT


.. list-table:: inferred_lines_geoms
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea (PRIMARY KEY)
   * - *wkt*
     - text
     - geometría inferida del recorrido de la linea en formato WKT


.. list-table:: travel_times_stations (insumos)
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_o*
     - int
     - id de la estación de origen
   * - *id_d*
     - int
     - id de la estación de destino
   * - *id_linea_o*
     - bigint
     - id de la linea de origen
   * - *id_ramal_o*
     - bigint
     - id del ramal de origen
   * - *lat_o*
     - float
     - latitud de la estación de origen
   * - *lon_o*
     - float
     - longitud de la estación de origen
   * - *id_linea_d*
     - bigint
     - id de la linea de destino
   * - *id_ramal_d*
     - bigint
     - id del ramal de destino
   * - *lat_d*
     - float
     - latitud de la estación de destino
   * - *lon_d*
     - float
     - longitud de la estación de destino
   * - *travel_time_min*
     - float
     - tiempo de viaje en minutos entre estaciones


Modelo de datos de base ``dashboard``
--------------------------------------

Las tablas de la base de datos ``dashboard`` contienen datos agregados y procesados para la visualización interactiva en el dashboard de UrbanTrips.


.. list-table:: matrices
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *desc_dia*
     - text
     - descripción del día
   * - *tipo_dia*
     - text
     - tipo de día (laboral, fin de semana, etc.)
   * - *var_zona*
     - text
     - variable de zonificación utilizada
   * - *filtro1*
     - text
     - filtro aplicado
   * - *Origen*
     - text
     - zona de origen
   * - *Destino*
     - text
     - zona de destino
   * - *Viajes*
     - int
     - cantidad de viajes


.. list-table:: lineas_deseo
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *desc_dia*
     - text
     - descripción del día
   * - *tipo_dia*
     - text
     - tipo de día
   * - *var_zona*
     - text
     - variable de zonificación utilizada
   * - *filtro1*
     - text
     - filtro aplicado
   * - *Origen*
     - text
     - zona de origen
   * - *Destino*
     - text
     - zona de destino
   * - *Viajes*
     - int
     - cantidad de viajes
   * - *lon_o*
     - float
     - longitud del origen
   * - *lat_o*
     - float
     - latitud del origen
   * - *lon_d*
     - float
     - longitud del destino
   * - *lat_d*
     - float
     - latitud del destino


.. list-table:: viajes_hora
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *desc_dia*
     - text
     - descripción del día
   * - *tipo_dia*
     - text
     - tipo de día
   * - *Hora*
     - int
     - hora del día
   * - *Viajes*
     - int
     - cantidad de viajes
   * - *Modo*
     - text
     - modo de transporte


.. list-table:: distribucion
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *desc_dia*
     - text
     - descripción del día
   * - *tipo_dia*
     - text
     - tipo de día
   * - *Distancia*
     - int
     - distancia en km
   * - *Viajes*
     - int
     - cantidad de viajes
   * - *Modo*
     - text
     - modo de transporte


.. list-table:: indicadores (dashboard)
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *desc_dia*
     - text
     - descripción del día
   * - *tipo_dia*
     - text
     - tipo de día
   * - *Titulo*
     - text
     - título del indicador
   * - *orden*
     - int
     - orden de presentación
   * - *Indicador*
     - text
     - nombre del indicador
   * - *Valor*
     - text
     - valor del indicador


.. list-table:: particion_modal
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *desc_dia*
     - text
     - descripción del día
   * - *tipo_dia*
     - text
     - tipo de día
   * - *tipo*
     - text
     - tipo de partición
   * - *modo*
     - text
     - modo de transporte
   * - *modal*
     - float
     - porcentaje de participación modal


.. list-table:: ocupacion_por_linea_tramo (dashboard)
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea
   * - *yr_mo*
     - text
     - año y mes
   * - *nombre_linea*
     - text
     - nombre de la linea
   * - *day_type*
     - text
     - tipo de día
   * - *n_sections*
     - int
     - cantidad de secciones
   * - *section_meters*
     - int
     - largo de la sección en metros
   * - *sentido*
     - text
     - sentido del recorrido
   * - *section_id*
     - int
     - id de la sección
   * - *hour_min*
     - int
     - hora mínima
   * - *hour_max*
     - int
     - hora máxima
   * - *legs*
     - int
     - cantidad de etapas
   * - *prop*
     - float
     - proporción de etapas
   * - *buff_factor*
     - float
     - factor de buffer para visualización
   * - *wkt*
     - text
     - geometría de la sección en formato WKT


.. list-table:: lines_od_matrix_by_section
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea
   * - *yr_mo*
     - text
     - año y mes
   * - *day_type*
     - text
     - tipo de día
   * - *n_sections*
     - int
     - cantidad de secciones
   * - *hour_min*
     - int
     - hora mínima
   * - *hour_max*
     - int
     - hora máxima
   * - *Origen*
     - int
     - sección de origen
   * - *Destino*
     - int
     - sección de destino
   * - *legs*
     - int
     - cantidad de etapas
   * - *prop*
     - float
     - proporción de etapas
   * - *nombre_linea*
     - text
     - nombre de la linea


.. list-table:: matrices_linea_carto
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_linea*
     - bigint
     - id de la linea
   * - *n_sections*
     - int
     - cantidad de secciones
   * - *section_id*
     - int
     - id de la sección
   * - *wkt*
     - text
     - geometría de la sección en formato WKT
   * - *x*
     - float
     - coordenada x (longitud)
   * - *y*
     - float
     - coordenada y (latitud)
   * - *nombre_linea*
     - text
     - nombre de la linea


.. list-table:: chains_norm
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *dia*
     - text
     - día del viaje
   * - *id_tarjeta*
     - text
     - id de la tarjeta
   * - *id_viaje*
     - bigint
     - id del viaje
   * - *h3_inicio*
     - text
     - índice H3 del inicio del viaje
   * - *h3_transfer1*
     - text
     - índice H3 del primer transbordo
   * - *h3_transfer2*
     - text
     - índice H3 del segundo transbordo
   * - *h3_fin*
     - text
     - índice H3 del fin del viaje
   * - *h3_inicio_norm*
     - text
     - índice H3 normalizado del inicio
   * - *h3_transfer1_norm*
     - text
     - índice H3 normalizado del primer transbordo
   * - *h3_transfer2_norm*
     - text
     - índice H3 normalizado del segundo transbordo
   * - *h3_fin_norm*
     - text
     - índice H3 normalizado del fin
   * - *modo_agregado*
     - text
     - modo de transporte agregado
   * - *rango_hora*
     - text
     - rango horario del viaje
   * - *genero_agregado*
     - text
     - género agregado
   * - *tarifa_agregada*
     - text
     - tarifa agregada
   * - *transferencia*
     - integer
     - indica si hubo transferencia
   * - *distancia_agregada*
     - text
     - rango de distancia agregada
   * - *travel_speed*
     - double
     - velocidad de viaje
   * - *factor_expansion_linea*
     - double
     - factor de expansión por linea
   * - *tipo_dia*
     - text
     - tipo de día
   * - *mes*
     - text
     - mes del viaje
   * - *seq_lineas*
     - text
     - secuencia de lineas utilizadas
   * - *distance_od*
     - double
     - distancia origen-destino
   * - *travel_time_min*
     - double
     - tiempo de viaje en minutos
