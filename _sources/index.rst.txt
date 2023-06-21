.. urbantrips documentation master file, created by
   sphinx-quickstart on Mon Jun 19 13:43:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bienvenidos a la documentación de Urbantrips!
=============================================


UrbanTrips es una biblioteca de código abierto que toma información de un sistema de pago con tarjeta inteligente de transporte público, infiere destinos de las etapas, construye cadenas de viaje para cada usuario y produce matrices de origen-destino y otros indicadores operativos. El principal objetivo de la librería es producir insumos útiles para la gestión del transporte público a partir de requerimientos mínimos de información y pre-procesamiento. Con sólo una tabla geolocalizada de transacciones proveniente de un sistema de pago electrónico, se podrán generar resultados, que serán más precisos cuanta más información adicional se incorpore al proceso a través de los archivos opcionales. El proceso elabora las matrices, los indicadores y construye una serie de gráficos y mapas útiles para la planificación y fiscalización del transporte público.

Para una discusión metodológica de cómo se imputan destinos y se construye la matriz de origen y destino se puede consultar este `documento metodológico <https://github.com/EL-BID/UrbanTrips/blob/dev/Metodologia_UrbanTrips.pdf>`_ 

Esta documentación guiará al usuario a través del proceso de instalación y configuración del ambiente para correr UrbanTrips. Luego, la sección :doc:`primeros_pasos`, ofrecerá un tutorial básico de cómo utilizar la librería para correr un set de datos abiertos de ejemplo. También ofrecerá un definición del modelo de datos de los archivos que UrbanTrips toma como insumos (:doc:`inputs`), en particular dará detalles de cómo setear el archivo de configuración (:doc:`configuracion`). Por último ofrecerá detalles de la concepción que UrbanTrips tiene de las líneas y ramales (:doc:`lineas_ramales`) y una descripción del modelo de datos final con los resultados de la librería (:doc:`resultados`).  


Cotenido
========

.. toctree::

   instalacion
   primeros_pasos
   configuracion
   inputs
   lineas_ramales
   resultados
   
   
.. note::

   This project is under active development.
