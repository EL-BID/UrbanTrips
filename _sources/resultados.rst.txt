Resultados finales
==================

Los resultados se guardarán en bases de ``SQLite``, donde cada base con sus tablas es un archivo en el disco. Los nombres de estos archivos están dados por el archivo de configuración (ver :doc:`configuracion`) y se ubican en ``data/db/``. Existen dos tipos de bases fundamentales ``data`` e ``insumos``. ``data`` guardará todo lo relativo a etapas, viajes y toda información que se actualiza con cada corrida. Así, puede haber una base de data diferente para cada semana o cada mes a medida que alcance un volumen determinado y utilizar un alias específico para este propósito configurable en el archivo de configuración (``ciudad_2023_semana1``, ``ciudad_2023_semana2``,etc). Por su lado, ``insumos`` es una base de datos que guardará información que no se actualiza periódicamente y servirá tanto para los datos de la semana 1 como los de la semana 2 (cartografía de recorridos, paradas, distancias entre pares de haxágonos H3 en una ciudad determinada, etc).


Modelo de datos de base ``data``
--------------------------------





Modelo de datos de base ``insumos``
-----------------------------------
