[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EL-BID_UrbanTrips&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EL-BID_UrbanTrips)

![analytics image (flat)](https://raw.githubusercontent.com/vitr/google-analytics-beacon/master/static/badge-flat.gif)

![analytics](https://www.google-analytics.com/collect?v=1&cid=555&t=pageview&ec=repo&ea=open&dp=/urbantrips/readme&dt=&tid=UA-4677001-16)



# README
`urbantrips` es una biblioteca de código abierto que toma información de un sistema de pago con tarjeta inteligente de transporte público y, a través de un procesamiento de la información que infiere destinos de los viajes y construye las cadenas de viaje para cada usuario, produce matrices de origen-destino y otros indicadores (KPI) para rutas de autobús. El principal objetivo de la librería es producir insumos útiles para la gestión del transporte público a partir de requerimientos mínimos de información y pre-procesamiento. Con sólo una tabla geolocalizada de transacciones económicas proveniente de un sistema de pago electrónico, se podrán generar resultados, que serán más precisos cuanto más información adicional se incorpore al proceso a través de los archivos opcionales. El proceso elabora las matrices, los indicadores y construye una serie de gráficos y mapas de transporte.

Para una discusión metodológica de cómo se imputan destinos y se construye la matriz de origen y destino se puede consultar el documento metodológico:

[Link al Documento Metodológico](https://github.com/EL-BID/UrbanTrips/blob/eed8dec61089fb46269e392600d641444440820b/Metodologia_UrbanTrips.pdf)

Con `urbantrips` se pueden procesar en una corrida la información de transacciones correspondientes a más de un día. Sin embargo, no se puede dividir un mismo día en dos corridas. Toda la información respecto de un día debe procesarse en la misma corrida. Si es demasiada información, conviene separarla en diversos archivos donde cada uno siempre tenga la totalidad de la información de los días a analizar (por ej. `lunes.csv`, `martes.csv` o `semana1.csv`, `semana2.csv` pero no `lunes_a.csv`, `lunes_b.csv`). Luego en otras corridas pueden procesarse otros días y la información se irá actualizando en las bases correspondientes.

Los resultados se guardarán en dos bases de `SQLite`, una para los datos de las etapas, viajes y otros que se van actualizando a medida que se ingresan nuevos datos, y otra base de insumos para información que no se actualiza con tanta periodicidad (como por ejemplo la matriz de las distancias entre puntos fijos de una ciudad). En el siguiente apartado se muestra como configurar el proceso, qué archivos tomará `urbantrips` para producir la información y en qué bases se guardarán los resultados.

## Insumos necesarios y opcionales
Urbantrips requiere sólo 2 insumos indispensables:
- Un archivo csv con las transacciones del sistema de pago con tarjeta
- Un archivo de configuración: `configuraciones_generales.yaml`

El archivo csv con las transacciones debe tener los siguientes campos obligatorios (los nombres pueden ser diferentes y esto se configura en el archivo configuraciones_generales.yaml):
- 	`fecha_trx`: campo que indica la fecha de la transacción.
- 	`hora_trx`: solo es obligatorio cuando el campo fecha incluye solo el día, sin información de la hora y minutos.
-	`id_tarjeta_trx`: un id único para cada tarjeta para cada día.
-	`id_linea_trx`: un id único para cada linea de transporte
-	`orden_trx`: un entero secuencial que establezca el orden de transacciones para una misma tarjeta en el día. Solo el obligatorio cuando el campo fecha incluye solo el día o el día y hora, sin información a nivel de minutos.
-	`latitud_trx`: Latitud de la transacción.
-	`longitud_trx`: Longitud de la transacción.

Al correr el proceso general de `urbantrips`, éste tomará el archivo de configuración que le dirá qué archivo csv contiene la información con los datos de las transacciones a utilizar en esta corrida. En el directorio de trabajo podrá haber diversos archivos con datos de diferentes días o periodos de tiempo (`lunes.csv`, `martes.csv` o `enero.csv`, `febrero.csv`). Cada uno será procesado en una corrida por vez. Qué archivo utilizar se configura en `configuraciones_generales.yaml` en el parámetro `nombre_archivo_trx:`. En ese mismo archivo se configuran otros parámetros, como así también las bases donde los resultados se irán guardando. Para más detalles sobre cómo utilizar este archivo de configuración consulte el apartado [Seteo del archivo de configuración](#seteo-del-archivo-de-configuracion). 

Con sólo esos archivos podrá correr el proceso de imputación de destinos, construcción de matrices OD y elaboración de KPIs. Dicho eso, se obtendrán más resultados y con mayor precisión si se suman estos archivos opcionales:

- Tabla con información de las líneas y/o ramales de transporte público (nombre de fantasía, etc).
- Cartografía de los recorridos de las líneas de transporte público
- Cartografía de las zonificaciones con las unidades espaciales utilizadas para agregar datos para la matriz OD
- Tabla de GPS con el posicionamiento de las unidades


El esquema de datos de estos archivos se especifica en el apartado [Esquema de datos](#Esquema-de-datos).

## Aclaración sobre el concepto de lineas y ramales en urbantrips
Una linea de transporte público puede tener un recorrido principal en torno al cual hay pequeñas variantes. Estas son consideradas ramales dentro de una misma linea. En muchas ciudades no existen estas diferencias y cada recorrido tiene un nombre y id únicos. Pero en otras no es así. A su vez, puede darse una situación donde una persona utiliza por ej el metro, subiendo a la estación del recorrido A y bajarse en el recorrido B, sin que ese transbordo sea identificado como transacción en la tarjeta. Por lo tanto, para imputar el destino consideramos como puntos de descenso posible todas las estaciones del metro. En este caso, el metro funcionará como una única línea y cada recorrido un ramal dentro del mismo. También puede suceder que una linea de autobuses tenga varios ramales, pero no siempre se identifica en los datos el ramal que realmente dicho interno está recorriendo. Con lo cual podría ser cualquier recorrido de cualquiera de los ramales y al imputar el destino deberiamos considerar todas las estaciones potenciales de toda esa linea de autobus. Esta forma de tratar a las líneas y ramales permite que `urbantrips` se acomode a estas situaciones. 

Si en una ciudad no existen estas situaciones, simplemente se utiliza la linea para identificar cada recorrido. Si alguna de las situaciones que se identificaron aquí se presenta en una ciudad, se puede utilizar ese criterio de linea y ramal que debe estar de ese modo en la tabla de transacciones a utilizar. La diferencia fundamental es que el proceso de imputación de destinos considerará como posible punto de destino todas las estaciones de la linea y no del ramal.

## Seteo del archivo de configuración 

El archivo de configuración (`configuraciones_generales.yaml`) es único. Cada corrida leerá la información que hay en este archivo. Su contenido puede editarse entre corrida y corrida para, por ejemplo, procesar dos días diferentes. 

El primer parámetro `resolucion_h3` establece el nivel de resolución del esquema [H3](https://h3geo.org/) con el que se va a trabajar. La resolucion 8 tiene hexágonos de 460 metros de lado. En la resolucion 9 tienen 174 metros y en la 10 tienen 65 metros.

El segundo es el prámetro principal, el nombre del archivo que contiene la información de las transacciones. El mismo deberá localizarse en `/data/data_ciudad/` (más información sobre la [estructura de directorios](#estructura-de-directorios)). Esta parte del archivo de configuración permite especificar el nombre del archivo a utilizar como así también los nombres de los atributos tal cual aparecen en el csv para que puedan ser guardados en el esquema de datos de `urbantrips`.
```
resolucion_h3: 8
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
```
El siguiente conjunto de parámetros de configuración definen el procesamiento de las transacciones.
- `formato_fecha`: especifica el formato en el que se encuentra el campo `fecha_trx` (por ej. `"%d/%m/%Y"`, `"%d/%m/%Y %H:%M:%S"`)
- `columna_hora`: Indica si la información sobre la hora está en una columna separada (`hora_trx`). Este debe ser un entero de 0 a 23.
- `ordenamiento_transacciones`: El criterio para ordenar las transacciones en el tiempo. Si se cuenta con un timestamp completo con horas y minutos, entonces usar `fecha_completa`. Si solo se cuenta con la información del día y la hora, se puede usar `orden_trx`.   
- `ventana_viajes`: Cuando se tiene un timestamp completo, indica la ventana de tiempo en minutos para considerar que las etapas se agrupan en un mismo viaje.  
- `ventana_duplicado`: Cuando se tiene un timestamp completo, indica la ventana de tiempo en minutos para considerar que dos transacciones son simultaneas, por lo se creará un `id_tarjeta` ad hoc a cada una.
- `tipo_trx_invalidas`: Especifica primero el nombre del atributo tal cual aparece en el csv y luego los valores que deben eliminarse al no representar transacciones vinculadas a viajes (por ej. carga de salgo, errores del sistema). Se pueden especificar varios atributos y varios valores por atributo.
- `modos`: urbantrips estandariza en 5 categorias (`autobus`,`tren`,`metro`,`tranvia` y `brt`) los modos. Debe pasarse el equivalente a cómo aparece categorizado en el csv cada modo.  
- `filtro_latlong_bbox`: Establece un box para eliminar rápidamente las transacciones que esten geolocalizadas fuera de una área lógica de cobertura del sistema de transporte público.

```
formato_fecha: "%d/%m/%Y"
columna_hora: True
ordenamiento_transacciones: orden_trx #fecha_completa u orden_trx
ventana_viajes: 120
ventana_duplicado: 5

tipo_trx_invalidas:
	tipo_trx_tren:
    	- 'CHECK OUT SIN CHECKIN'
    	- 'CHECK OUT'

modos:
	autobus: COL
	tren: TRE
	metro: SUB
	tranvia:
	brt:

filtro_latlong_bbox:
	minx: -59.3
	miny: -35.5
	maxx: -57.5
	maxy: -34.0

```
El siguiente grupo de configuraciones nombra las dos bases de datos con las que trabaja `urbantrips`. `alias_db_data` guardará todo lo realtivo a etapas, viajes y toda información que se actualiza con cada corrida. Así, puede haber una base de `data` para cada semana o cada mes a medida que alcance un volumen determinado (`amba_2023_semana1`, `amba_2023_semana2`,etc). Por su lado, `alias_db_insumos` es una base de datos que guardará información de manera constante y servirá tanto para los datos de la semana 1 como los de la semana 2. 

```
alias_db_data: amba_2023_semana1
alias_db_insumos: amba_2023
```
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
Este otro grupo de parámetros controla el método de imputación de destinos. Por un lado establece el criterio de tolerancia de la distancia entre la siguiente transaccion de esa tarjeta y alguna parada de la linea utilizada en la etapa a la que se está imputando el destino. Si la distancia es mayor a esta tolerancia, no se imputará destino. El parametro  `imputar_destinos_min_distancia` establece si se imputará la siguiente transacción como destino o la parada de la linea utilizada en la etapa que minimice la distancia con respecto a la siguiente transacción.

```
tolerancia_parada_destino: 2200
imputar_destinos_min_distancia: True
```
Por último, se pueden especificar tablas adicionales de utilidad para el proceso. Por un lado se puede agregar metadata para las lineas, como por ejemplo su nombre de fantasía ademas del id correspondiente, o a qué empresa pertenece.  La misma puede identificar una linea o una linea-ramal (siendo los ramales pequeñas desviaciones con respecto a un recorrido principal). En este último caso `urbantrips` creara dos tablas diferentes, una para la metadata de las lineas y otra para la de ramales. 

Tambien permite agregar cartografías como los recorridos, que deben ser una única Linestring en 2d (no permite multilineas), o diferentes archivos con unidades espaciales para las que se quiere agregar datos. Para cada archivo debe indicarse el nombre del atributo que contiene la información y, de ser necesario, un orden en el que se quiera producir las matrices OD que genera `urbantrips`. 
```
nombre_archivo_informacion_lineas: lineas_amba.csv
informacion_lineas_contiene_ramales: True
recorridos_geojson: recorridos_amba.geojson

zonificaciones:
	geo1: hexs_amba.geojson
	var1: Corona
	orden1: ['CABA', 'Primer cordón', 'Segundo cordón', 'Tercer cordón', 'RMBA']
	geo2: hexs_amba.geojson
	var2: Partido
```

## Esquema de datos

Este es el esquema de datos que deben seguir los archivos `csv` suministrados como insumos a `urbantrips`.

### transacciones
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
| Campo | Tipo de dato | Descripción |
| -- | -- | -- |
|`id_linea`|int|**Obligatorio**. Entero que identifique a la linea.|
|`nombre_linea`|str|**Obligatorio**. Nombre de la línea.|
|`modo`|str|**Obligatorio**. Modo de la linea.|
|`id_ramal`|int|Opcional.Entero que identifique al ramal.|
|`nombre_ramal`|str|Opcional. Nombre del ramal.|
|`empresa`|str|Opcional. Nombre de la empresa.|
|`descripcion`|str|Opcional. Descripción adicional de la linea o ramal.|

### Recorridos lineas
| Campo | Tipo de dato | Descripción |
| -- | -- | -- |
|`id_linea`|int|**Obligatorio**. Entero que identifique a la linea.|
|`nombre_linea`|str|**Obligatorio**. Nombre de la línea.|
| `geometry`|2DLineString|Polilinea del recorrido. No puede ser multilinea|


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
Una vez creado el ambiente, puede descargar el [dataset de transacciones SUBE de AMBA](https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/transacciones.csv), guardarlo en `data/data_ciudad/transacciones.csv`. Este dataset no cuenta con un campo `fecha` con el formato `dd/mm/aaaa`, deberá agregar con una fecha cualquiera y utilizar las configuraciones especificadas más abajo. Por último ejecutar:
```
python urbantrips/run_urbantrips.py
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

nombre_archivo_informacion_lineas:
informacion_lineas_contiene_ramales: False

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
## Agradecimientos
Queremos agradecer la colaboración de los gobiernos de Ciudad de Buenos Aires, Córdoba, Mendoza y Bariloche que proveyeron datos y participaron de valiosas conversaciones para mejorar esta librería.

## Licencia

Copyright© 2023. Banco Interamericano de Desarrollo ("BID"). Uso autorizado. [AM-331-A3](/LICENSE.md)


## Autores

Felipe González ([@alephcero](https://github.com/alephcero/)) 
Sebastián Anaposlky([@sanapolsky](https://github.com/sanapolsky/))

