[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EL-BID_UrbanTrips&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EL-BID_UrbanTrips)

![analytics image (flat)](https://raw.githubusercontent.com/vitr/google-analytics-beacon/master/static/badge-flat.gif)

![analytics](https://www.google-analytics.com/collect?v=1&cid=555&t=pageview&ec=repo&ea=open&dp=/urbantrips/readme&dt=&tid=UA-4677001-16)


![logo](https://github.com/EL-BID/UrbanTrips/blob/dev/docs/logo_readme.png)

# README
`urbantrips` es una biblioteca de código abierto que toma información de un sistema de pago con tarjeta inteligente de transporte público y, a través de un procesamiento de la información que infiere destinos de los viajes y construye las cadenas de viaje para cada usuario, produce matrices de origen-destino y otros indicadores (KPI) para rutas de autobús. El principal objetivo de la librería es producir insumos útiles para la gestión del transporte público a partir de requerimientos mínimos de información y pre-procesamiento. Con sólo una tabla geolocalizada de transacciones económicas proveniente de un sistema de pago electrónico, se podrán generar resultados, que serán más precisos cuanto más información adicional se incorpore al proceso a través de los archivos opcionales. El proceso elabora las matrices, los indicadores y construye una serie de gráficos y mapas de transporte.

Para una discusión metodológica de cómo se imputan destinos y se construye la matriz de origen y destino se puede consultar este ![documento metodológico](https://github.com/EL-BID/UrbanTrips/blob/dev/Metodologia_UrbanTrips.pdf "Documento metodológico")

Con `urbantrips` se pueden procesar en una corrida la información de transacciones correspondientes a más de un día. Sin embargo, no se puede dividir un mismo día en dos corridas. Toda la información respecto de un día debe procesarse en la misma corrida. Si es demasiada información, conviene separarla en diversos archivos donde cada uno siempre tenga la totalidad de la información de los días a analizar (por ej. `lunes.csv`, `martes.csv` o `semana1.csv`, `semana2.csv` pero no `lunes_a.csv`, `lunes_b.csv`). Luego en otras corridas pueden procesarse otros días y la información se irá actualizando en las bases correspondientes.

Los resultados se guardarán en dos bases de `SQLite`, una para los datos de las etapas, viajes y otros que se van actualizando a medida que se ingresan nuevos datos, y otra base de insumos para información que no se actualiza con tanta periodicidad (como por ejemplo la matriz de las distancias entre puntos fijos de una ciudad). En el siguiente apartado se muestra como configurar el proceso, qué archivos tomará `urbantrips` para producir la información y en qué bases se guardarán los resultados.

## Insumos necesarios y opcionales
Urbantrips requiere sólo 2 insumos indispensables:
- Un archivo csv con las transacciones del sistema de pago con tarjeta
- Un archivo de configuración: `configuraciones_generales.yaml`

El archivo csv con las transacciones debe tener los siguientes campos obligatorios (los nombres pueden ser diferentes y esto se configura en el archivo configuraciones_generales.yaml):
-	`id_tarjeta_trx`: un id único para cada tarjeta para cada día.
-	`id_linea_trx`: un id único para cada linea de transporte
- 	`fecha_trx`: campo que indica la fecha de la transacción.
- 	`hora_trx`: solo es obligatorio cuando el campo fecha incluye solo el día, sin información de la hora y minutos.
-	`orden_trx`: un entero secuencial que establezca el orden de transacciones para una misma tarjeta en el día y que se reinicie con cada viaje. Es obligatorio unicamente cuando el campo fecha incluye solo el día o el día y hora, sin información a nivel de minutos.
-	`latitud_trx`: Latitud de la transacción.
-	`longitud_trx`: Longitud de la transacción.

Al correr el proceso general de `urbantrips`, éste tomará el archivo de configuración que le dirá qué archivo csv contiene la información con los datos de las transacciones a utilizar en esta corrida. En el directorio de trabajo podrá haber diversos archivos con datos de diferentes días o periodos de tiempo (`lunes.csv`, `martes.csv` o `enero.csv`, `febrero.csv`). Cada uno será procesado en una corrida por vez. Qué archivo utilizar se configura en `configuraciones_generales.yaml` en el parámetro `nombre_archivo_trx:`. En ese mismo archivo se configuran otros parámetros, como así también las bases donde los resultados se irán guardando. Para más detalles sobre cómo utilizar este archivo de configuración consulte el apartado [Seteo del archivo de configuración](#seteo-del-archivo-de-configuracion). 

Con sólo esos archivos podrá correr el proceso de imputación de destinos, construcción de matrices OD y elaboración de KPIs. Dicho eso, se obtendrán más resultados y con mayor precisión si se suman estos archivos opcionales:

- Tabla con información de las líneas y/o ramales de transporte público (nombre de fantasía, etc).
- Cartografía de los recorridos de las líneasy/o ramales de transporte público
- Cartografía de las zonificaciones con las unidades espaciales utilizadas para agregar datos para la matriz OD
- Tabla de GPS con el posicionamiento de las unidades


El esquema de datos de estos archivos se especifica en el apartado [Esquema de datos](#Esquema-de-datos).

## Sobre el concepto de lineas y ramales en urbantrips
Una linea de transporte público puede tener un recorrido principal en torno al cual hay pequeñas variantes. Estas son consideradas ramales dentro de una misma linea. En muchas ciudades no existen estas diferencias y cada recorrido tiene un nombre y id únicos. Pero en otras no es así. A su vez, puede darse una situación donde una persona utiliza por ej el metro, subiendo a la estación del recorrido A y bajarse en el recorrido B, sin que ese transbordo sea identificado como transacción en la tarjeta. Por lo tanto, para imputar el destino consideramos como puntos de descenso posible todas las estaciones del metro. En este caso, el metro funcionará como una única línea y cada recorrido un ramal dentro del mismo. También puede suceder que una linea de autobuses tenga varios ramales, pero no siempre se identifica en los datos el ramal que realmente dicho interno está recorriendo. Con lo cual podría ser cualquier recorrido de cualquiera de los ramales y al imputar el destino deberiamos considerar todas las estaciones potenciales de toda esa linea de autobus. Esta forma de tratar a las líneas y ramales permite que `urbantrips` se acomode a estas situaciones. 

Si en una ciudad no existen estas situaciones, simplemente se utiliza la linea para identificar cada recorrido. Si alguna de las situaciones que se identificaron aquí se presenta en una ciudad, se puede utilizar ese criterio de linea y ramal que debe estar de ese modo en la tabla de transacciones a utilizar. La diferencia fundamental es que el proceso de imputación de destinos considerará como posible punto de destino todas las estaciones de la linea y no del ramal.

## Seteo del archivo de configuración 

El archivo de configuración (`configuraciones_generales.yaml`) es único. Cada corrida leerá la información que hay en este archivo. Su contenido puede editarse entre corrida y corrida para, por ejemplo, procesar días diferentes. Se divide en diferentes categorías de parámetros.

### Parámetros generales
En este primer grupo encontramos parámetros generales que utliza `urbantrips` en diferentes momentos. El primer parámetro `resolucion_h3` establece el nivel de resolución del esquema [H3](https://h3geo.org/) con el que se va a trabajar. La resolucion 8 tiene hexágonos de 460 metros de lado. En la resolucion 9 tienen 174 metros y en la 10 tienen 65 metros.

Luego vienen las configuraciǫnes que nombram las dos bases de datos con las que trabaja `urbantrips`. `alias_db_data` guardará todo lo realtivo a etapas, viajes y toda información que se actualiza con cada corrida. Así, puede haber una base de `data` para cada semana o cada mes a medida que alcance un volumen determinado (`amba_2023_semana1`, `amba_2023_semana2`,etc). Por su lado, `alias_db_insumos` es una base de datos que guardará información de manera constante y servirá tanto para los datos de la semana 1 como los de la semana 2. 

También es necesario especificar una proyección de coordenadas en metros, pasando un id de [EPSG](https://epsg.io/).

Es posible establecer un `filtro_latlong_bbox` que crea un box para eliminar rápidamente las transacciones que esten geolocalizadas fuera de una área lógica de cobertura del sistema de transporte público.

Por último el `formato_fecha` especifica el formato en el que se encuentra el campo `fecha_trx` (por ej. `"%d/%m/%Y"`, `"%d/%m/%Y %H:%M:%S"`) y las fechas en el archivo de posicionamiento GPS (si se utiliza). Todas las fechas a utilizar deben estar en el mismo formato.

```
resolucion_h3: 8

# Alias a utilizar en las db de datos y de insumos
alias_db_data: amba_test
alias_db_insumos: amba_test

# Proyeccion de coordenadas en metros a utilizar  
epsg_m: 9265

# Filtro de coordenadas en formato minx, miny, maxx, maxy 
filtro_latlong_bbox:
    minx: -59.3
    miny: -35.5
    maxx: -57.5
    maxy: -34.0

#Especificar el formato fecha presente en todos los dataset
formato_fecha: "%d/%m/%Y"
```

### Parámetros de imputación de destinos

Este otro grupo de parámetros controla el método de imputación de destinos. Por un lado establece el criterio de tolerancia de la distancia entre la siguiente transaccion de esa tarjeta y alguna parada de la linea utilizada en la etapa a la que se está imputando el destino. Si la distancia es mayor a esta tolerancia, no se imputará destino. El parametro  `imputar_destinos_min_distancia` establece si se imputará la siguiente transacción como destino o la parada de la linea utilizada en la etapa que minimice la distancia con respecto a la siguiente transacción.

```
# Distancia maxima tolerable entre destino imputado y siguiente transaccion (en metros)
tolerancia_parada_destino: 2200

# Imputar utilizando la parada de la linea de orige que minimice la distancia con respecto a la siguiente transaccion o solo la siguiente transaccion
imputar_destinos_min_distancia: False
```


### Parámetros de transacciones
Este es el grupo de parámetros principal. Por un lado está el nombre del archivo que contiene la información de las transacciones. El mismo deberá localizarse en `/data/data_ciudad/` (más información sobre la [estructura de directorios](#estructura-de-directorios)). Esta parte del archivo de configuración permite especificar el nombre del archivo a utilizar como así también los nombres de los atributos tal cual aparecen en el csv para que puedan ser guardados en el esquema de datos de `urbantrips`.

El siguiente conjunto de parámetros de configuración definen el procesamiento de las transacciones.
- `columna_hora`: Indica si la información sobre la hora está en una columna separada (`hora_trx`). Este debe ser un entero de 0 a 23.
- `ordenamiento_transacciones`: El criterio para ordenar las transacciones en el tiempo. Si se cuenta con un timestamp completo con horas y minutos, entonces usar `fecha_completa`. Si solo se cuenta con la información del día y la hora, se puede usar `orden_trx`. Este campo debe tener un entero secuencial que ordena las transacciones. Debe comenzar en cero cuando se comienza un nuevo viaje e incrementear con cada nueva etapa en ese viaje.  
- `ventana_viajes`: Cuando se tiene un timestamp completo, indica la ventana de tiempo en minutos para considerar que las etapas se agrupan en un mismo viaje.  
- `ventana_duplicado`: Cuando se tiene un timestamp completo, indica la ventana de tiempo en minutos para considerar que dos transacciones son simultaneas, por lo se creará un `id_tarjeta` ad hoc a cada una.
- `tipo_trx_invalidas`: Especifica primero el nombre del atributo tal cual aparece en el csv y luego los valores que deben eliminarse al no representar transacciones vinculadas a viajes (por ej. carga de salgo, errores del sistema). Se pueden especificar varios atributos y varios valores por atributo.
- `modos`: urbantrips estandariza en 5 categorias (`autobus`,`tren`,`metro`,`tranvia` y `brt`) los modos. Debe pasarse el equivalente a cómo aparece categorizado en el csv cada modo.  


```
nombre_archivo_trx: semana1.csv

nombres_variables_trx:
	id_trx: id
	fecha_trx: fecha
	id_tarjeta_trx: id_tarjeta
	modo_trx: modo
	hora_trx: hora
	id_linea_trx: id_linea
	id_ramal_trx:  
	interno_trx: interno_bus
	orden_trx: etapa_red_sube
	latitud_trx: lat
	longitud_trx: lon
	factor_expansion:  

#Indicar si la informacion sobre la hora está en una columna separada. En nombres_variables debe indicarse el nombre. Dejar vacío en caso contrario 
columna_hora: True 

# Criterio para ordenar las transacciones en el tiempo. 'fecha_completa' utiliza el campo dado en fecha_trx mientras que `orden_trx` utiliza un entero incremental que se reinicia con cada viaje   
ordenamiento_transacciones: orden_trx 

# Cantidad de minutos de la ventana de tiempo para considerar diferentes etapas dentro de un mismo viaje
ventana_viajes: 

# Cantidad de minutos de la ventana de tiempo para considerar diferentes transacciones como una sola
ventana_duplicado: 

# Tipo de transacciones a elminar por no considerare usos en transporte publico. Indicar la columna y los valores para cada columna
tipo_trx_invalidas:
    tipo_trx_tren:
        - 'CHECK OUT SIN CHECKIN'
        - 'CHECK OUT'

# Especificar como se nombra a los modos en los archivos  
modos:
    autobus: COL
    tren: TRE
    metro: SUB
    tranvia:
    brt:
```

### Parámetros de posicionamiento GPS
Este parámetro se utiliza para cuando existe una tabla separada con GPS que contenga el posicionamiento de los internos. En ese caso, se gelocalizará cada transacción en base a la tabla GPS, uniendo por `id_linea` e `interno` (haciendo a este campo obligatorio) y minimizando el tiempo de la transacción con respecto a la transacción gps del interno de esa linea. Para eso el campo `fecha` debe estar completo con dia, hora y minutos. Esto hace obligatoria la existencia de un csv con la información de posicionamiento de los gps. Su nombre y atributos se especifican de modo similar a lo hecho en transacciones. La existencia de la tabla GPS permitira calcular KPI adicionales como el IPK.

```
geolocalizar_trx: True

nombre_archivo_gps: gps_semana1.csv

nombres_variables_gps:
    id_gps: 
    id_linea_gps: idlinea
    id_ramal_gps: c_ld_Id
    interno_gps: interno
    fecha_gps: date_time
    latitud_gps: latitude   
    longitud_gps: longitude

```

### Parámetro de lineas, ramales y paradas

Es necesario que se especifique si en el sistema de transporte existen lineas con ramales, tal como los entiende `urbantrips` y se especifica en [Sobre el concepto de lineas y ramales en urbantrips](#Sobre-el-concepto-de-lineas-y-ramales-en-urbantrips). Esto debe indicarse en el parámetro `lineas_contienen_ramales`.

Se puede agregar metadata para las lineas, como por ejemplo su nombre de fantasía ademas del id correspondiente, o a qué empresa pertenece.  La misma puede identificar una linea o una linea-ramal (siendo los ramales pequeñas desviaciones con respecto a un recorrido principal). En este último caso `urbantrips` creara dos tablas diferentes, una para la metadata de las lineas y otra para la de ramales. 

Tambien permite agregar cartografías como los recorridos de las lineas y ramales, que deben ser una única Linestring en 2d (no permite multilineas). Si existe una tabla de recorridos, entonces debe proveerse un archivo con información de las lineas y ramales. Esta tabla puede identificar recorridos de lineas o tambien de lineas y ramales. No es necesario indicar el sentido.

Por úlitmo, se puede especificar un archivo con la ubicación de paradas o estaciones para cada linea y ramal, indicando un orden de paso o sucesión y también un `node_id`, donde deben aparecer con un mismo id las paradas de diferentes ramales de una misma linea donde se pueda realizar un transbordo entre ramales. El `node_id` puede repetirse en otras lineas. 

```
# Las lineas a ser utilizadas se subdividen en ramales?
lineas_contienen_ramales: True

# Nombre del archivo con la metadada de lineas y/o ramales
nombre_archivo_informacion_lineas: lineas_amba_test.csv

# Nombre del archivo con las rutas de las lineas y/o ramales
recorridos_geojson: recorridos_amba.geojson

# Nombre del archivo con las paradas para todas las lineas y/o ramales con orden y node_id 
nombre_archivo_paradas: 
```

Finalmente se pueden suministrar diferentes archivos con unidades espaciales para las que se quiere agregar datos. Para cada archivo debe indicarse el nombre del atributo que contiene la información y, de ser necesario, un orden en el que se quiera producir las matrices OD que genera `urbantrips`. 

```
zonificaciones:
	geo1: hexs_amba.geojson
	var1: Corona
	orden1: ['CABA', 'Primer cordón', 'Segundo cordón', 'Tercer cordón', 'RMBA']
	geo2: hexs_amba.geojson
	var2: Partido
```

## Esquema de datos

Este es el esquema de datos que deben seguir los archivos suministrados como insumos a `urbantrips`.

### transacciones

Tabla con las transacciones de la tarjeta.

| Campo | Tipo de dato | Descripción |
| -- | -- | -- |
| `id_trx` | int | Opcional. Id único que identifique cada registro. |
| `fecha_trx` | strftime | **Obligatorio**. Timestamp de la transaccion. Puede ser solo el día o el dia, hora y minuto.|
| `id_tarjeta_trx` | int/str | **Obligatorio**. Un id que identifique a cada tarjeta. |
| `modo_trx` | str | Opcional. Se estandarizará con lo especificado en `modos` en el archivo de configuración. Si no hay información en la tabla, se imputará todo como `autobus`. |
| `hora_trx` | int | Opcional a menos que `fecha_trx` no tenga información de la hora y minutos. Entero de 0 a 23 indicando la hora de la transacción. |
| `id_linea_trx` | int | **Obligatorio**. Entero que identifique a la linea.  |
| `id_ramal_trx` | int | Opcional. Entero que identifique al ramal. |
| `interno_trx` | int | **Obligatorio**. Entero que identifique al interno |
| `orden_trx` | int | Opcional a menos que `fecha_trx` no tenga información de la hora y minutos. Entero comenzando en 0 que esatblezca el orden de transacciones para una misma tarjeta en un mismo día. |
| `latitud_trx` | float | **Obligatorio**. Latitud de la transacción. |
| `longitud_trx` | float | **Obligatorio**. Longitud de la transacción. |
| `factor_expansion` | float | Opcional. Factor de expansión en caso de tratarse de una muestra. |

### GPS

Tabla con el posicionamiento de cada interno con información de linea y ramal.

| Campo | Tipo de dato | Descripción |
| -- | -- | -- |
|`id_gps`|int|  **Obligatorio**. Id único que identifique cada registro. |
|`id_linea_gps`|int|**Obligatorio**. Id único que identifique la linea.|
|`id_ramal_gps`|int|**Obligatorio si hay ramales**. Id único que identifique cada ramal.|
|`interno_gps`|int|**Obligatorio**. Id único que identifique cada interno.|
|`fecha_gps`|strftime|**Obligatorio**. Dia, hora y minuto de la posición GPS del interno.|
|`latitud_gps`|float|**Obligatorio**. Latitud.| 
|`longitud_gps`|float|**Obligatorio**. Longitud.|
    
    
### Información de lineas y ramales

Tabla con metadata descriptiva de las lineas y ramales. 

| Campo | Tipo de dato | Descripción |
| -- | -- | -- |
|`id_linea`|int|**Obligatorio**. Entero que identifique a la linea.|
|`nombre_linea`|str|**Obligatorio**. Nombre de la línea.|
|`modo`|str|**Obligatorio**. Modo de la linea.|
|`id_ramal`|int|**Obligatorio si hay ramales**.Entero que identifique al ramal.|
|`nombre_ramal`|str|**Obligatorio si hay ramales**. Nombre del ramal.|
|`empresa`|str|Opcional. Nombre de la empresa.|
|`descripcion`|str|Opcional. Descripción adicional de la linea o ramal.|

### Recorridos lineas

Archivo `geojson` con la información de la linea. Debe haber un único LineString por cada ramal, sin importar el sentido. Si los recorridos difieren mucho, puede volver a dibujarse para que pase por puntos medios en esas zonas y aún así ser un recorrido representativo del ramal.

| Campo | Tipo de dato | Descripción |
| -- | -- | -- |
|`id_linea`|int|**Obligatorio**. Entero que identifique a la linea.|
|`id_ramal`|str|**Obligatorio si hay ramales**. Entero que identifique al ramal.|
|`stops_distance`|int|Opcional. Distancia en metros a aplicarse al interpolar paradas sobre el recorrido.|
|`line_stops_buffer`|int|Opcional. Distancia en metros entre paradas para que se puedan agregar en una sola.|
| `geometry`|2DLineString|Polilinea del recorrido. No puede ser multilinea.|


### Paradas

Tabla que contenga las paradas de cada linea y ramal (si hay ramales). El campo `node_id` se utiliza para identificar en qué paradas puede haber transbordo entre dos ramales de la misma linea. Para esas paradas el `node_id` debe ser el mismo, para las demas único dentro de la línea. 
| Campo | Tipo de dato | Descripción |
| -- | -- | -- |
|`id_linea`|int|**Obligatorio**. Entero que identifique a la linea.|
|`id_ramal`|int|**Obligatorio si hay ramales**. Entero que identifique a al ramal.|
|`order`|int| **Obligatorio**. Entero único que siga un recorrido de la linea o ramal de manera incremental. No importa el sentido|
|`y`|float|**Obligatorio**. Latitud.| 
|`x`|float|**Obligatorio**. Longitud.|
|`node_id`|int|**Obligatorio**. Identifica con el mismo id estaciones donde puede haber transbordo entre ramales de una misma linea. Único para los otros casos dentro de la misma línea.|

## Estructura de directorios
Al clonar `urbantrips` y correrlo, dejará la siguiente estructura de directorios:
```
urbantrips
│   README.md
│
└─── urbantrips
│   ...
└─── configs
│   │   configuraciones_generales.yaml
│   │   
└─── data 
│   └─── db
│       │  amba_2023_semana1_data
│       │  amba_2023_semana2_data
│       │  amba_2023_insumos
│       
│   └─── data_ciudad
│       │   semana1.csv
│       │   semana2.csv
│       │   lineas_amba.csv
│       │   hexs_amba.geojson
│       │   ...
└─── resultados 
│   └─── data
│       │   amba_2023_semana1_etapas.csv
│       │   amba_2023_semana1_viajes.csv
│       │   amba_2023_semana1_usuarios.csv
│       │   amba_2023_semana2_etapas.csv
│       │   amba_2023_semana2_viajes.csv
│       │   amba_2023_semana2_usuarios.csv
│   └─── html
│       │   ...
│   └─── matrices
│       │   ...
│   └─── pdf
│       │   ...
│   └─── png
│       │   ...
│   └─── tablas
```

## Configuración del ambiente

Para poder instalar la librería se aconseja crear un ambiente y luego instalar la librería con `pip`. Si desea hacerlo con `virtualenv` puede ejecutar los siguientes pasos:

```
virtualenv venv --python=python3.10
source venv/bin/activate
pip install urbantrips
```

Si desea hacerlo con `conda` entonces:

```
conda create -n env_urbantrips -c conda-forge python=3.10 rvlib
conda activate env_urbantrips
pip install urbantrips
```

## Primeros pasos
Una vez creado el ambiente, puede descargar el [dataset de transacciones SUBE de AMBA](https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/transacciones.csv), guardarlo en `data/data_ciudad/transacciones.csv`. Este dataset no cuenta con un campo `fecha` con el formato `dd/mm/aaaa`, deberá agregar con una fecha cualquiera y utilizar las configuraciones especificadas más abajo. A su vez, se debe especificar un `id_linea` con el criterio ya explicado previamente. Para eso se puede tomar la información de lineas de [este archivo](https://github.com/EL-BID/Matriz-Origen-Destino-Transporte-Publico/blob/main/data/lineas_ramales.csv) (que se puede utilizar para el parámetro `nombre_archivo_informacion_lineas`). En este archivo, cada `id_ramal` tiene un `id_linea` asignado, con esa información pueden construir el `id_linea` de la tabla transacciones.  


Una vez que se dispone del archivo de transacciones y el de información de las líneas, es posible comenzar a utilizar `urbantrips`. En primer lugar es necesario inicializar los directorios y la base de datos que la librería necesita. Este paso solo se corre una vez.

```
python urbantrips/initialize_environment.py
```

Luego, se puede procesar la información de transacciones. Este archivo de transacciones puede tener la información de un día, una semana o un mes (siempre que no sea demasiada información). Este paso procesa las transacciones en etapas y viajes, imputando destinos. Luego pueden correr este paso por cada nuevo dataset que quieran procesar (`semana_1.csv`,`semana_2.csv`, etc) ajustando lo necesario en el archivo `configuraciones_generales.yaml`.

```
python urbantrips/process_transactions.py
```

Por último, una vez procesadas todas las transacciones que sean de interés y cargadas en la base de datos de la libería, es posible correr los pasos de post procesamiento sobre esa información, como los KPI, visualizaciones y exportación de resultados. 

```
python urbantrips/run_postprocessing.py
```


### Configuraciones para el dataset de transacciones SUBE de AMBA

```yaml
geolocalizar_trx: False
resolucion_h3: 8
#tolerancia parada destino en metros
tolerancia_parada_destino: 2200

nombre_archivo_trx: transacciones.csv

alias_db_data: amba

alias_db_insumos: amba

lineas_contienen_ramales: True
nombre_archivo_informacion_lineas: lineas_amba.csv

imputar_destinos_min_distancia: True

#ingresar el nombre de las variables
nombres_variables_trx:
    id_trx: id
    fecha_trx: fecha 
    id_tarjeta_trx: id_tarjeta
    modo_trx: modo
    hora_trx: hora
    id_linea_trx: id_linea
    id_ramal_trx:  id_ramal
    interno_trx: interno_bus
    orden_trx: etapa_red_sube 
    latitud_trx: lat 
    longitud_trx: lon
    factor_expansion:   
    
modos:
    autobus: COL
    tren: TRE
    metro: SUB
    tranvia:
    brt:
     
recorridos_geojson:


# Filtro de coordenadas en formato minx, miny, maxx, maxy 
filtro_latlong_bbox:
    minx: -59.3
    miny: -35.5
    maxx: -57.5
    maxy: -34.0 

    
#Especificar el formato fecha
formato_fecha: "%d/%m/%Y"

columna_hora: True 
ordenamiento_transacciones: orden_trx 


tipo_trx_invalidas:
    tipo_trx_tren:
        - 'CHECK OUT SIN CHECKIN'
        - 'CHECK OUT'
```  
