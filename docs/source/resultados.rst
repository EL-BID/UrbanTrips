Resultados finales
==================

Los resultados se guardarán en bases de ``SQLite``, donde cada base con sus tablas es un archivo en el disco. Los nombres de estos archivos están dados por el archivo de configuración (ver :doc:`configuracion`) y se ubican en ``data/db/``. Existen dos tipos de bases fundamentales ``data`` e ``insumos``. ``data`` guardará todo lo relativo a etapas, viajes y toda información que se actualiza con cada corrida. Así, puede haber una base de data diferente para cada semana o cada mes a medida que alcance un volumen determinado y utilizar un alias específico para este propósito configurable en el archivo de configuración (``ciudad_2023_semana1``, ``ciudad_2023_semana2``,etc). Por su lado, ``insumos`` es una base de datos que guardará información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de haxágonos H3 en una ciudad determinada, etc).


Modelo de datos de base ``data``
--------------------------------


            
.. list-table:: transacciones
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id*
     - int
     - id unico que identifica cada transaccion en esta base de datos
   * - *id_original*
     - text
     - id de la transaccion original en el csv usado en la corrida
   * - *id_tarjeta*
     - text
     - id que identifica la tarjeta
   * - *fecha*
     - datetime
     - fecha de la transaccion
   * - *dia*
     - text
     - dia de la transaccion
   * - *tiempo*
     - text
     - Hora minutos y segundos de la transaccion en formato HH::MM::SS
   * - *hora*
     - int
     - Hora de la transaccion de 0 a 23	
   * - *modo*
     - text
     - Modo estandarizado de la transaccion
   * - *id_linea*
     - int
     - id de la linea utilizada en la transsacion
   * - *id_ramal*
     - int
     - id del ramal utilizado en la transsacion
   * - *interno*
     - int
     - numero de interno o vehículo utilizado en la transaccion
   * - *orden_trx*
     - int
     - entero incremental que indica la etapa dentro de una cadena de viajes
   * - *latitud*
     - float
     - latitud
   * - *longitud*
     - float
     - longitud
   * - *factor_expansion*
     - float
     - factor de expansión 




            
            
            
.. list-table:: etapas
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 
     
     

.. list-table:: destinos
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 


.. list-table:: viajes
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 

.. list-table:: usuarios
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 
     
     
.. list-table:: factores_expansion
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 


.. list-table:: indicadores
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 

.. list-table:: gps
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 

.. list-table:: services
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - **
     - int
     - 













Modelo de datos de base ``insumos``
-----------------------------------
