KPI
==============

En este apartado se describen los diferentes tipos de KPI que UrbanTrips puede producir en función de los inputs con los que cada ciudad cuente. Se comienza por el caso donde solamente existen datos de demanda. En segundo lugar se abordan los KPI que puede producir cuando existen datos en de la oferta, como puede ser la tabla de GPS como así también información de servicios ofertados. 

En base a demanda
-----------------

Cuando no existen datos de oferta, expresada fundamentalmente en la tabla GPS de posicionamiento de vehículos, UrbanTrips puede calcular algunos elementos de oferta y demanda en base a la tabla de transacciones en base al concepto de *pasajero equivalente*. Tomando la tabla de etapas se calcula por hora y dia para el interno de cada linea el total de pasajeros que iniciaron una etapa a esa hora en ese interno, la distancia media viajada (*DMT*) y la velocidad comercial promedio a la que circula ese interno en esa hora (siempre tomando solamente las coordenadas del interno cuando recoge pasajeros). 

Tomando estos datos se construye para cada pasajero un *pasajero equivalente* poniendo en relación cuántas posiciones disponibles en ese interno utilizó y por cuánto tiempo. Es decir, si un pasajero recorre 5 km en un interno que circula a 10 kmh, equivaldrá a 0.5 posiciones o pasajeros equivalentes. Para calcular un factor de ocupación se considera que cada interno oferta 60 ubicaciones y se compara el total de pasajeros equivalentes en esa hora en ese interno contra ese stock fijo. Luego se procede a agregar tdos estos estadísticos para diferentes niveles de agregación (interno y linea) así como también para un día de la semana tipo o dia de fin de semana tipo (siempre que hayan días procesados que pertenezcan a uno de esos tipos).  

Los resultados quedan almacenados en las siguientes tablas  (para más detalles vea el apartado :doc:`resultados`).  

* ``basic_kpi_by_vehicle_hr``: arroja la batería de estadísticos por vehículo para cada linea, día y hora.
* ``basic_kpi_by_line_hr``: arroja la batería de estadísticos promediando para cada linea, día y hora (incluyendo día de semana y de fín de semana típico).
* ``basic_kpi_by_line_day``: arroja la batería de estadísticos promediando para cada linea y día (incluyendo día de semana y de fín de semana típico).


En base a GPS
-------------

Cuando existe una tabla de GPS se puedem elaborar estadísicos más elaborados y precisos. En primer lugar se procura obtener un factor de ocupación comparando los Espacio Kilómetro Demandados (EKD) como proporción de los Espacio Kilómetro Ofertados (EKO). Para los primeros (EKD)  se toma la cantidad de pasajeros transportados multiplicados por una DMT que puede ser utilizando la distancia media o la mediana, para evitar la influencia de medidas extremas que puedan incidir en el indicador. Para los segundos (EKO) se toma el total de kilómetros recorridos en base a la información disponible en la tabla GPS y se los multiplca por las 60 ubicaciones que se considera posee cada interno. 

En segundo lugar, se obtiene un Índice Pasajero Kilómetro (IPK) poniendo en relación el total de pasajeros transportados sobre el total de kilómetros recorridos por la línea. Para obtener estos indicadores principales se obtiene otros insumos que quedan en la base de datos, como el total de pasajeros, el total de kilómetros recorridos, la distancias medias y medianas viajadas, los vehículos totales, los pasajeros por vehículo por día, y los kilómetros por vehículo por día. Esto se calcula agregado para cada linea y día procesado, así como también para un día de la semana tipo o dia de fin de semana tipo.

Los resultados quedan almacenados en la tabla ``kpi_by_day_line`` (para más detalles vea el apartado :doc:`resultados`).  

En base a servicios
-------------------

UrbanTrips permite obtener datos a nivel de servicios para cada vehículo de cada línea (para más información pueden leer el aparatdo de :doc:`servicios`). Una vez que esta clafisicación de los datos de GPS en servicios ha tenido lugar, pueden obtenerse diferentes KPI al nivel de cada servicio identificado. Los resultados quedan almacenados en la tabla ``kpi_by_day_line_service`` (para más detalles vea el apartado :doc:`resultados`)  




