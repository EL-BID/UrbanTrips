Factores de expansión
==========================

El factor de expansión es una medida estadística que se utiliza para ajustar los datos recopilados en una muestra y hacer estimaciones más precisas sobre una población total. Se calcula mediante la relación entre el tamaño real de la población objetivo y el tamaño de la muestra utilizada. Este factor se aplica a cada respuesta individual para compensar posibles desviaciones en la representatividad de la muestra. Al utilizar el factor de expansión, se busca extrapolar los resultados de la muestra a toda la población objetivo, teniendo en cuenta características de la población objetivos u otros criterios relevantes, permitiendo obtener estimaciones más confiables y representativas de la realidad.

El proceso UrbanTrips puede correrse a partir de una muestra del archivo un archivo de transacciones de uno o varios días completos. En el caso de realizarse una muestra, deberá ser una muestra de tarjetas teniendo en cuenta para las tarjetas muestreadas se incluyan todas las transacciones para el/los días del proceso.

Esta muestra debe contener un campo factor de expansión que será especificado en el archivo de configuración en el campo factor_expansion en nombres_variables_trx del archivo de transacciones nombre_archivo_trx. En el caso de no especificarse el factor de expansión el proceso le asignará valor 1 a todas las transacciones.

Ejecución del proceso

Durante la ejecución del proceso, se realizarán validaciones a las transacciones y tarjetas que puede resultar en la eliminación de una cierta cantidad de registros. Para que los resultados se ajusten a la población total, el proceso va a construir una serie de factores de expansión teniendo en cuenta el total de transacciones, tarjetas únicas y transacciones por línea en el archivo de transacciones antes de la validación. Los campos relacionados al factor de expansión son los siguientes:

Factor_expansion_original: se encuentra en la tabla de transacciones, es el factor de expansión original que venía especificado en el archivo de transacciones (en caso de ser una muestra). En caso de no ser una muestra, el valor va a ser 1 para todos los registros.

Factor_expansion_tarjeta: Este factor expande a la cantidad total de tarjetas que se encuentra en el archivo de transacciones original antes de realizarse la depuración por la validación de datos. Este campo se encuentra en la tabla de etapas, viajes y tarjetas.

Factor_expansion_linea: Este factor calibra el factor_expansion_tarjeta teniendo en cuenta la cantidad de transacciones por línea que se encuentran en el archivo de transacciones original antes de realizarse la depuración por la validación de datos. Este campo se encuentra en la tabla de etapas, viajes y tarjetas.

Se recomienda usar factor_expansion_linea para el análisis de etapas o viajes y el factor_expansion_tarjeta si se están analizando usuarios.
