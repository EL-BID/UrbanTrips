Input de datos
==============

Este es el esquema de datos que deben seguir los archivos suministrados como insumos a `urbantrips`.

Transacciones
-------------

Tabla con las transacciones de la tarjeta.

.. list-table:: 
   :widths: 25 25 50
   :header-rows: 1

   * - Campo
     - Tipo de dato
     - Descripción
   * - *id_trx*
     - int
     - Opcional. Id único que identifique cada registro.
   * - *fecha_trx*
     - strftime
     - **Obligatorio**. Timestamp de la transaccion. Puede ser solo el día o el dia, hora y minuto.
   * - *id_trx*
     - int
     - Opcional. Id único que identifique cada registro.
   * - *fecha_trx*
     - strftime
     - **Obligatorio**. Timestamp de la transaccion. Puede ser solo el día o el dia, hora y minuto.
   * - *id_tarjeta_trx*
     - int/str
     - **Obligatorio**. Un id que identifique a cada tarjeta.
   * - *modo_trx*
     - str
     - Opcional. Se estandarizará con lo especificado en `modos` en el archivo de configuración. Si no hay información en la tabla, se imputará todo como `autobus`.
   * - *hora_trx*
     - int
     - Opcional a menos que `fecha_trx` no tenga información de la hora y minutos. Entero de 0 a 23 indicando la hora de la transacción.
   * - *id_linea_trx*
     - int
     - **Obligatorio**. Entero que identifique a la linea. 
   * - *id_ramal_trx*
     - int
     - Opcional. Entero que identifique al ramal.
   * - *interno_trx*
     - int
     - **Obligatorio**. Entero que identifique al interno 
   * - *orden_trx*
     - int
     - Opcional a menos que `fecha_trx` no tenga información de la hora y minutos. Entero comenzando en 0 que esatblezca el orden de transacciones para una misma tarjeta en un mismo día.
   * - *latitud_trx*
     - float
     - **Obligatorio**. Latitud de la transacción.
   * - *longitud_trx*
     - float
     - **Obligatorio**. Longitud de la transacción. 
   * - *factor_expansion*
     - float
     - Opcional. Factor de expansión en caso de tratarse de una muestra. 
    
     

