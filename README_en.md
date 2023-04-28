[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EL-BID_UrbanTrips&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EL-BID_UrbanTrips)

![analytics image (flat)](https://raw.githubusercontent.com/vitr/google-analytics-beacon/master/static/badge-flat.gif)

![analytics](https://www.google-analytics.com/collect?v=1&cid=555&t=pageview&ec=repo&ea=open&dp=/urbantrips/readme&dt=&tid=UA-4677001-16)


![logo](https://github.com/EL-BID/UrbanTrips/blob/feature/issue_4/docs/logo_readme.png)

# README
`urbantrips` is an open-source library that takes information from a smart card payment system for public transportation and, through information processing that infers trip destinations and constructs travel chains for each user, produces origin-destination matrices and other indicators (KPI) for bus routes. The main objective of the library is to produce useful inputs for public transport management from minimal information and pre-processing requirements. With only a geolocalized table of economic transactions from an electronic payment system, results can be generated, which will be more accurate the more additional information is incorporated into the process through optional files. The process elaborates the matrices, indicators, and constructs a series of transport graphs and maps.

For a methodological discussion of how destinations are imputed and the origin-destination matrix is constructed, refer to this ![document (in spanish)](https://github.com/EL-BID/UrbanTrips/blob/dev/Metodologia_UrbanTrips.pdf "Documento metodológico")

With `urbantrips`, the transaction information corresponding to more than one day can be processed in a single run. However, the same day cannot be split into two runs. All information regarding a day must be processed in the same run. If there is too much information, it is advisable to separate it into different files where each one always has all the information for the days to be analyzed (e.g., `monday.csv`, `tuesday.csv` or `week1.csv`, `week2.csv`, but not `monday_a.csv`, `monday_b.csv`). Then, other days can be processed in other runs, and the information will be updated in the corresponding databases.

The results will be saved in two `SQLite` databases, one for the data of the stages, trips, and others that are updated as new data is entered, and another database for inputs for information that is not updated as frequently (such as the matrix of distances between fixed points in a city). The following section shows how to configure the process, which files `urbantrips` will take to produce the information, and in which databases the results will be saved.

## Necessary and Optional Inputs

Urbantrips requires only 2 indispensable inputs:

- A CSV file with transactions from the smart card payment system
- A configuration file: `configuraciones_generales.yaml`

The CSV file with transactions must have the following mandatory fields (the names can be different, and this is configured in the configuraciones_generales.yaml file):

- `fecha_trx`: field indicating the transaction date.
- `hora_trx`: only mandatory when the date field includes only the day, without information on the hour and minutes.
- `id_tarjeta_trx`: a unique id for each card for each day.
- `id_linea_trx`: a unique id for each transport line
- `orden_trx`: a sequential integer that establishes the order of transactions for the same card on the day. Only mandatory when the date field includes only the day or the day and hour, without information at the minute level.
- `latitud_trx`: Transaction latitude.
- `longitud_trx`: Transaction longitude.

When running the general `urbantrips` process, it will take the configuration file that will tell it which CSV file contains the information with the transaction data to be used in this run. In the working directory, there may be various files with data from different days or periods of time (`monday.csv`, `tuesday.csv`, or `january.csv`, `february.csv`). Each one will be processed one run at a time. Which file to use is configured in `configuraciones_generales.yaml` in the `nombre_archivo_trx:` parameter. Other parameters, as well as the databases where the results will be saved, are also configured in that same file. For more details on how to use this configuration file, see the section [Setting up the configuration file](#setting-up-the-configuration-file). 


With only those files, you can run the destination imputation process, build the OD matrices, and create KPIs. Having said that, more results and greater accuracy can be obtained by adding these optional files:

- Table with information about public transportation lines and/or branches (fantasy name, etc.).
- Maps of the routes of public transportation lines.
- Maps of zoning with the spatial units used to aggregate data for the OD matrix.
- GPS table with the positioning of the units.

The data schema for these files is specified in the section [Data schema](#data-schema).

## Clarification about the concept of lines and branches in urbantrips

A public transportation line may have a main route around which there are small variations. These are considered branches within the same line. In many cities, these differences do not exist, and each route has a unique name and ID. But in others, this is not the case. It may also happen that a person uses the subway, getting on at station A and getting off at station B, without that transfer being identified as a transaction on the card. Therefore, to impute the destination, we consider all subway stations as potential drop-off points. In this case, the subway will function as a single line and each route a branch within it. It may also happen that a bus line has several branches, but the data does not always identify the branch that the internal route is actually traversing. Therefore, it could be any route of any of the branches, and when imputing the destination, we should consider all potential stations of that entire bus line. This way of handling lines and branches allows `urbantrips` to accommodate these situations.

If these situations do not exist in a city, simply use the line to identify each route. If any of the situations identified here occur in a city, that line and branch criterion can be used, which must be set up in the transaction table to be used. The fundamental difference is that the destination imputation process will consider all stations on the line as possible destination points, rather than just the branch.

## Setting up the configuration file 

The configuration file (`configuraciones_generales.yaml`) is unique. Each run will read the information in this file. Its content can be edited between runs to, for example, process two different days.

The first parameter resolucion_h3 establishes the resolution level of the [H3](https://h3geo.org/) schema that will be used. Resolution 8 has hexagons with sides of 460 meters. In resolution 9 they have 174 meters and in 10 they have 65 meters.

The second is the main parameter, the name of the file containing the transaction information. It must be located in /data/data_ciudad/ (more information about the [directory structure](#directory-structure)). This part of the configuration file allows specifying the name of the file to be used as well as the attribute names exactly as they appear in the CSV so that they can be saved in the data scheme of `urbantrips`.

```
resolucion_h3: 8
nombre_archivo_trx: week1.csv

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

- `fecha_formato`: specifies the format in which the fecha_trx field is in (e.g. `"%d/%m/%Y"`, `"%d/%m/%Y %H:%M:%S"`)
- `hora_columna`: indicates whether the information about the hour is in a separate column (`hora_trx`). This should be an integer from 0 to 23.
- `transacciones_ordenamiento`: the criterion for sorting transactions in time. If you have a complete timestamp with hours and minutes, then use `fecha_completa`. If you only have information about the day and hour, you can use `orden_trx`.
- `viajes_ventana`: when you have a complete timestamp, indicates the time window in minutes to consider that stages are grouped into the same trip.
- `duplicado_ventana`: when you have a complete timestamp, indicates the time window in minutes to consider that two transactions are simultaneous, so an ad hoc `id_tarjeta` will be created for each one.
- `tipo_trx_invalidas`: specifies the name of the attribute as it appears in the csv and then the values that should be eliminated as they do not represent transactions related to trips (e.g. balance recharge, system errors). You can specify several attributes and several values per attribute.
- `modos`: urbantrips standardizes into 5 categories (`autobus`,`tren`,`metro`,`tranvia`, and `brt`) the modes. You must pass the equivalent of how each mode is categorized in the csv.
- `filtro_latlong_bbox`: sets a box to quickly eliminate transactions that are geolocated outside of a logical area of public transportation coverage.
    
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

The following group of configurations names the two databases used by `urbantrips`. `alias_db_data` will store all the information related to stages, trips, and any other information that is updated with each run. Thus, there can be a data database for each week or month as it reaches a certain volume (`amba_2023_week1`, `amba_2023_week2`, etc.). On the other hand, `alias_db_insumos` is a database that will store information constantly and will serve both for `week 1` and `week 2` data.

```
alias_db_data: amba_2023_semana1
alias_db_insumos: amba_2023
```
This parameter is used when there is a separate GPS table containing the positioning of the vehicles. In this case, each transaction will be geolocated based on the GPS table, joining by `id_linea` and `interno` (making this field mandatory) and minimizing the time difference between the transaction and the GPS transaction of the vehicle on that line. For this, the `fecha` field must be complete with day, hour, and minutes. This makes it mandatory to have a CSV file with the GPS information. Its name and attributes are specified in a similar way to what was done for transactions. The existence of the GPS table will allow calculating additional KPIs such as the IPK.

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


This other group of parameters controls the destination imputation method. On one hand, it establishes the tolerance distance criterion between the next transaction of that card and any stop of the line used in the stage to which the destination is being imputed. If the distance is greater than this tolerance, the destination will not be imputed. The parameter `imputar_destinos_min_distancia` establishes whether the next transaction will be imputed as the destination or the stop of the line used in the stage that minimizes the distance with respect to the next transaction.

```
tolerancia_parada_destino: 2200
imputar_destinos_min_distancia: True
```

Finally, additional useful tables can be specified for the process. On one hand, metadata can be added for the lines, such as their fantasy name in addition to the corresponding id, or to which company they belong. It can identify a line or a line-branch (with branches being small deviations from a main route). In the latter case, `urbantrips` will create two different tables, one for the metadata of the lines and another for the branches.

It also allows the addition of cartographies such as routes, which must be a single 2D Linestring (it does not allow multiline), or different files with spatial units for which data is to be added. For each file, the name of the attribute containing the information must be indicated and, if necessary, an order in which to produce the OD matrices generated by `urbantrips`.

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

## Data schema
This is the data schema that `csv` files supplied as inputs to `urbantrips` must follow.

### transacciones
| Field | Type | Description |
| -- | -- | -- |
| `id_trx` | int | Optional. Unique identifier for each record. |
| `fecha_trx` | strftime | **Mandatory**. Timestamp of the transaction. It can be only the date or date, hour and minute.|
| `id_tarjeta_trx` | int/str | **Mandatory**. An identifier that identifies each card. |
| `modo_trx` | str | Optional. It will be standardized with what is specified in `modos` in the configuration file. If there is no information in the table, it will be entered as `autobus`. |
| `hora_trx` | int | Optional unless `fecha_trx` has no information about the hour and minutes. Integer from 0 to 23 indicating the hour of the transaction. |
| `id_linea_trx` | int | **Mandatory**. Integer that identifies the line.  |
| `id_ramal_trx` | int | Optional. Integer that identifies the branch. |
| `interno_trx` | int | **Mandatory**. Integer that identifies the internal number. |
| `orden_trx` | int | Optional unless `fecha_trx` has no information about the hour and minutes. Integer starting at 0 that establishes the order of transactions for the same card on the same day. |
| `latitud_trx` | float | **Mandatory**. Latitude of the transaction. |
| `longitud_trx` | float | **Mandatory**. Longitude of the transaction. |
| `factor_expansion` | float | Optional. Expansion factor in case of being a sample. |

### gps
| Field | Type | Description |
| -- | -- | -- |
| id_gps | int | **Mandatory**. Unique ID that identifies each record. |
| id_linea_gps | int | **Mandatory**. Unique ID that identifies the line. |
| id_ramal_gps | int | **Mandatory if lines have branches**. Unique ID that identifies each branch. |
| interno_gps | int | **Mandatory**. Unique ID that identifies each bus. |
| fecha_gps | strftime | **Mandatory**. Day, hour, and minute of the GPS position of the bus. |
| latitud_gps | float | **Mandatory**. Latitude. |
| longitud_gps | float | **Mandatory**. Longitude. |
    
    
### Lines and branches information 
| Field | Type | Description |
| -- | -- | -- |
| id_linea | int | **Mandatory**. Integer that identifies the line. |
| nombre_linea | str | **Mandatory**. Name of the line. |
| modo | str | **Mandatory**. Mode of the line. |
| id_ramal | int | Optional. Integer that identifies the branch. |
| nombre_ramal | str | Optional. Name of the branch. |
| empresa | str | Optional. Name of the company. |
| descripcion | str | Optional. Additional description of the line or branch. |


### Lines geoms
| Field | Type | Description |
| -- | -- | -- |
|id_linea|int|**Mandatory**. Integer that identifies the line.|
|nombre_linea|str|**Mandatory**. Name of the line.|
| geometry|2DLineString|Poliline representing the route. Cannot be a multiline.|


## Directory structure
After cloning the repo and run `urbantrips`, this will be an example of the resulting directory structure:
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
│       │  amba_2023_week1_data
│       │  amba_2023_week2_data
│       │  amba_2023_inputs
│       
│   └─── data_ciudad
│       │   semana1.csv
│       │   semana2.csv
│       │   lineas_amba.csv
│       │   hexs_amba.geojson
│       │   ...
└─── resultados 
│   └─── data
│       │   amba_2023_week1_etapas.csv
│       │   amba_2023_week1_viajes.csv
│       │   amba_2023_week1_usuarios.csv
│       │   amba_2023_week2_etapas.csv
│       │   amba_2023_week2_viajes.csv
│       │   amba_2023_week2_usuarios.csv
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

In order to install the library it is advisable to create an environment and then install the library with `pip`. If you want to do it with `virtualenv` you can execute the following steps:

```
virtualenv venv --python=python3.10
source venv/bin/activate
pip install urbantrips
```

For a `conda` enviroment then:

```
conda create -n env_urbantrips -c conda-forge python=3.10 rvlib
conda activate env_urbantrips
pip install urbantrips
```

## First steps
After creating the environment, you can download the [Buenos Aires SUBE transactions dataset](https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/transacciones.csv), save it in `data/data_ciudad/transacciones.csv`. This dataset does not have a `fecha` field with the format `dd/mm/yyyy`, so you will need to add one with any date and use the configurations specified below. Also, an `id_linea` must be specified with the criteria already explained previously. For that, you can take the line information from [this file](https://github.com/EL-BID/Matriz-Origen-Destino-Transporte-Publico/blob/main/data/lineas_ramales.csv) (which can be used for the parameter `nombre_archivo_informacion_lineas`). In this file, each `id_ramal` has an `id_linea` assigned, with that information an `id_linea` can be added to transactions table.


Once the transaction file and line information is available, it is possible to start using `urbantrips`. First of all it is necessary to initialize the directories and the database that the library needs. This step is only run once.

```
python urbantrips/initialize_environment.py
```

Then, the transaction information can be processed. This transaction file can have the information of a day, a week or a month (as long as there is not too much information). This step processes the transactions in stages and trips, imputing destinations. Then it can be run for each new dataset they want to process (`week_1.csv`, `week_2.csv`, etc) adjusting what is necessary in the `configuraciones_generales.yaml` file.

```
python urbantrips/process_transactions.py
```

Finally, once all the transactions that are of interest have been processed and loaded into the liberty database, it is possible to run the subsequent processing steps on that information, such as KPIs, visualizations and export of results.

```
python urbantrips/run_postprocessing.py
```


### Configurations for the Buenos Aires SUBE transactions dataset 
```yaml
geolocalizar_trx: False
resolucion_h3: 8
#tolerancia parada destino en metros
tolerancia_parada_destino: 2200

nombre_archivo_trx: transacciones.csv

alias_db_data: amba

alias_db_insumos: amba

nombre_archivo_informacion_lineas: lineas_amba.csv
informacion_lineas_contiene_ramales: True

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
