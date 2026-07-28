@echo off
REM Levanta el dashboard.
REM
REM   dashboard.bat                       usa configs\configuraciones_generales.yaml
REM   dashboard.bat mi_config.yaml        busca el archivo en configs\
REM   dashboard.bat ruta\a\config.yaml    usa esa ruta (relativa o absoluta)
REM
REM Se encarga del separador "--" que streamlit necesita para pasarle argumentos
REM al script en vez de interpretarlos como propios.

setlocal

if "%~1"=="" goto :default

set "CONFIG=%~1"

REM Aceptar tanto una ruta como el nombre suelto del archivo dentro de configs\
if exist "%CONFIG%" goto :run
if exist "configs\%CONFIG%" set "CONFIG=configs\%CONFIG%" & goto :run
goto :notfound

:default
echo Configuracion: configs\configuraciones_generales.yaml ^(por defecto^)
streamlit run urbantrips\dashboard\dashboard.py
goto :eof

:run
echo Configuracion: %CONFIG%
streamlit run urbantrips\dashboard\dashboard.py -- --config "%CONFIG%"
goto :eof

:notfound
echo.
echo ERROR: no se encontro el archivo de configuracion "%~1"
echo        (se busco tal cual y tambien dentro de configs\)
echo.
echo Uso:
echo   dashboard.bat                       configuracion por defecto
echo   dashboard.bat mi_config.yaml        busca en configs\
echo   dashboard.bat ruta\a\config.yaml    ruta relativa o absoluta
echo.
echo Configuraciones disponibles en configs\:
for /f "delims=" %%f in ('dir /b configs\*.yaml 2^>nul ^| findstr /v /b ".snapshot_"') do echo   %%f
echo.
exit /b 1
