KPI
==============

En este apartado se describen los diferentes tipos de KPI que UrbanTrips puede producir en función de los inputs con los que cada ciudad cuente. Se comienza por el caso donde solamente existen datos de demanda. En segundo lugar se abordan los KPI que puede producir cuando existen datos en de la oferta, como puede ser la tabla de GPS como así también información de servicios ofertados. 

En base a demanda
-----------------

Cuando no existen datos de oferta, expresada fundamentalmente en la tabla GPS de posicionamiento de vehículos, UrbanTrips puede calcular algunos elementos de oferta y demanda en base a la tabla de transacciones en base al concepto de *pasajero equivalente*. Tomando la tabla de etapas se calcula por hora y dia para el interno de cada linea el total de pasajeros que iniciaron una etapa a esa hora en ese interno, la distancia media viajada (*DMT*) y la velocidad comercial promedio a la que circula ese interno en esa hora (siempre tomando solamente las coordenadas del interno cuando recoge pasajeros). 

Tomando estos datos se construye para cada pasajero un *pasajero equivalente* poniendo en relación cuántas posiciones disponibles en ese interno utilizó y por cuánto tiempo. Es decir, si un pasajero recorre 5 km en un interno que circula a 10 kmh, equivaldrá a 0.5 posiciones o pasajeros equivalentes. Para calcular un factor de ocupación se considera que cada interno oferta 60 ubicaciones y se compara el total de pasajeros equivalentes en esa hora en ese interno contra ese stock fijo. Luego se procede a agregar todos estos estadísticos para diferentes niveles de agregación (interno y linea) así como también para un día de la semana tipo o dia de fin de semana tipo (siempre que hayan días procesados que pertenezcan a uno de esos tipos).  

Los resultados quedan almacenados en las siguientes tablas  (para más detalles vea el apartado :doc:`resultados`).  

* ``basic_kpi_by_vehicle_hr``: arroja la batería de estadísticos por vehículo para cada linea, día y hora.
* ``basic_kpi_by_line_hr``: arroja la batería de estadísticos promediando para cada linea, día y hora (incluyendo día de semana y de fín de semana típico).
* ``basic_kpi_by_line_day``: arroja la batería de estadísticos promediando para cada linea y día (incluyendo día de semana y de fín de semana típico).


**Tramo más cargado**

UrbanTrips también puede computar, sólo con datos de la demanda, para una línea, día y rango horario la carga o demanda para los tramos del recorrido de esa línea. El recorrido de una línea puede segmentarse en *n* tramos o en tramos cada *m* cantidad de metros. A partir de este parámetro se computan *x* cantidad de puntos sobre el recorrido de la línea. Si existe una cartografía provista por el usuario para las líneas, se utilizará ese recorrido. En caso contrario, se inferirá un recorrido a partir de las coordenadas de las transacciones de dicha línea.

Para cada etapa de una línea en un rango horario se toman las coordenadas de origen y de destinos de la misma, se proyectan sobre el recorrido de la línea mediante un `Linear Referencing System LRS <https://en.wikipedia.org/wiki/Linear_referencing>`_ y se asume que dicha etapa hizo uso del recorrido de la línea entre los puntos comprendidos por el tramo de origen y el de destino, inclusive. Luego se calcula para cada punto de la línea la demanda en base a las etapas. Dado el sentido del desplazamiento de la etapa, se puede inferir el sentido de la misma, por lo cual se calcula la demanda separada por sentido. La misma se calcula tanto en cantidad de etapas como proporcional con respecto al total de etapas de esa línea para ese rango horario en ese sentido.

Por ejemplo, un usuario que inicia una etapa en el punto equiparable al 10% del recorrido y desciende en el punto equiparable al 30% del recorrido en una línea segmentada en 10 tramos, habrá utilizado los puntos equivalentes al 10, 20 y 30 %. Dado el sentido ascendente del LRS, se asume un sentido de ida. Luego se calcula la cantidad de etapas que hayan atravesado el punto 10%, ya sea como origen, destino o como punto de paso. Una vez calculado para cada día, se puede calcular un promedio para los días de semana o para los fines de semana en base a los datos ya procesados para cada día.

Para computar esta demanda por tramo se puede utilizar la siguiente función:

* ``id_linea``: id de línea o lista de ids de líneas a procesar
* ``rango_hrs``: define el rango horario del origen de las etapas a utilizar
* ``n_sections``: cantidad de segmentos a dividir el recorrido
* ``section_meters``: cantidad de metros de largo del segmento. Si se especifica este parámetro no se considerará ``n_sections``.
* ``day_type``: fecha de los datos a procesar o tipo de día (``weekday`` o ``weekend``).

.. code:: python

   kpi.compute_route_section_load(
    	id_linea=False,
   		rango_hrs=False,
   		n_sections=10,
   		section_meters=None,
   		day_type="weekday"
   		)



Urbantrips también permite construir una visualización exploratoria en base a los datos computados previamente. La misma conformará a partir de los *x* puntos *x-1* segmentos donde el tamaño y color del mismo indicará la demanda, ya sea en cantidad total de etapas (``indicador='cantidad_etapas'``) o proporcional (``indicador = ‘prop_etapas’``). La función toma un parámetro de ancho mínimo en metros que dichos segmentos van a tomar (para los segmentos con mínima o nula demanda) y un factor de expansión también en metros. Por ejemplo, si se utiliza el indicador ``prop_etapas`` y ``factor = 500`` aquel segmento que tenga una demanda igual al máximo de la demanda de esa línea en ese sentido para ese rango horario, el buffer del recorrido a visualizar tendrá un ancho de 500 m (``1 * 500``) y aquel que tenga una demanda equivalente a la mitad de esa demanda total, tendrá un ancho de 250 m en la visualización (``0,5 * 500``). Por otro lado, si se toma como indicador de la visualización la cantidad total de etapas y el mismo parámetro ``factor``, un segmento con demanda equivalente a 100 etapas tendrá un ancho de 50.000 m  (``100 * 500``). Esta visualización permite guardar el geojson producto de la misma ``save_gdf=True``.


.. code:: python

   viz.visualize_route_section_load(
       id_linea=False,
       rango_hrs=False,
       day_type='weekday',
       n_sections=10,
       section_meters=None,
       indicador='cantidad_etapas',
       factor=1,
       factor_min=50,
       save_gdf=False
   )


* ``id_linea``: id de línea o lista de ids de líneas a procesar
* ``rango_hrs``: define el rango horario del origen de las etapas a utilizar
* ``day_type``: fecha de los datos a procesar o tipo de día (``weekday`` o ``weekend``).
* ``n_sections``: cantidad de segmentos a dividir el recorrido
* ``section_meters``: cantidad de metros de largo del segmento. Si se especifica este parámetro no se considerará ``n_sections``.
* ``indicador``: indicador a utilizar ``‘prop_etapas’`` o ``'cantidad_etapas'``.
* ``factor``: factor de escalado que el el tramo tendrá en la visualización en base al indicador utilizado. 
* ``factor_min``: ancho mínimo en metros que el tramo tendrá en la visualización.
* ``save_gdf``: guardar los resultados de la visualización en un archivo geojson.


En base a GPS
-------------

Cuando existe una tabla de GPS se pueden elaborar estadísicos más elaborados y precisos. En primer lugar se procura obtener un factor de ocupación comparando los Espacio Kilómetro Demandados (EKD) como proporción de los Espacio Kilómetro Ofertados (EKO). Para los primeros (EKD)  se toma la cantidad de pasajeros transportados multiplicados por una DMT que puede ser utilizando la distancia media o la mediana, para evitar la influencia de medidas extremas que puedan incidir en el indicador. Para los segundos (EKO) se toma el total de kilómetros recorridos en base a la información disponible en la tabla GPS y se los multiplca por las 60 ubicaciones que se considera posee cada interno. 

En segundo lugar, se obtiene un Índice Pasajero Kilómetro (IPK) poniendo en relación el total de pasajeros transportados sobre el total de kilómetros recorridos por la línea. Para obtener estos indicadores principales se obtiene otros insumos que quedan en la base de datos, como el total de pasajeros, el total de kilómetros recorridos, la distancias medias y medianas viajadas, los vehículos totales, los pasajeros por vehículo por día, y los kilómetros por vehículo por día. Esto se calcula agregado para cada linea y día procesado, así como también para un día de la semana tipo o dia de fin de semana tipo.

Los resultados quedan almacenados en la tabla ``kpi_by_day_line`` (para más detalles vea el apartado :doc:`resultados`).  

En base a servicios
-------------------

UrbanTrips permite obtener datos a nivel de servicios para cada vehículo de cada línea (para más información pueden leer el aparatdo de :doc:`servicios`). Una vez que esta clafisicación de los datos de GPS en servicios ha tenido lugar, pueden obtenerse diferentes KPI al nivel de cada servicio identificado. Los resultados quedan almacenados en la tabla ``kpi_by_day_line_service`` (para más detalles vea el apartado :doc:`resultados`)  




