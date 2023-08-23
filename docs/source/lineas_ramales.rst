Sobre el concepto de lineas y ramales en UrbanTrips
===================================================

Una linea de transporte público puede tener un recorrido principal en torno al cual hay pequeñas variantes. Estas son consideradas ramales dentro de una misma linea. En muchas ciudades no existen estas diferencias y cada recorrido tiene un nombre y id únicos. Pero en otras no es así. A su vez, puede darse una situación donde, por ejemplo, una persona utiliza el metro, subiendo a la estación del recorrido A y descendiendo en una estación del recorrido B, sin que ese transbordo sea identificado como transacción en la tarjeta. Por lo tanto, para imputar el destino UrbanTrips considera como puntos de descenso posible todas las estaciones del metro. En este caso, el metro funcionará como una única línea y cada recorrido un ramal dentro del mismo. La diferencia fundamental es que el proceso de imputación de destinos considerará como posible punto de destino todas las estaciones de la linea y no del ramal.

También puede suceder que una linea de autobuses tenga varios ramales, pero no siempre se identifica en los datos el ramal que realmente dicho vehículo o interno recorrió. Con lo cual podría ser cualquier recorrido de cualquiera de los ramales y al imputar el destino debería considerarse todas las estaciones potenciales de toda esa linea de autobus. Esta forma de tratar a las líneas y ramales permite que UrbanTrips se acomode a estas situaciones particulares que cada ciudad presenta. 
 
Si la ciudad en la que se va a correr UrbanTrips presente un caso de estás características con líneas y ramales debe indicarse en el archivo de configuración en el parámetro ``lineas_contienen_ramales`` ( ver :doc:`configuracion`). A su vez debe indicarse en la tabla de **Información de lineas y ramales** (ver :doc:`lineas_ramales`) un id de linea que unifique diferentes ramales y los trate como una línea única utilizando el campo ``id_linea_agg`` de dicha tabla.
  
UrbanTrips no modifica los datos originales de la linea o el ramal, con lo cual si la información del ramal es correcta, esta forma de imputar destinos no afectará el ramal de la etapa según fue declarado en el interno o vehículo por el conductor. A su vez si en una ciudad no existen ramales, simplemente se utiliza la linea para identificar cada recorrido. 


