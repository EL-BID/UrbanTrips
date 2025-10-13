Instalación
===========


Para poder instalar la librería se aconseja crear un ambiente y luego instalar la librería con `pip`. Si desea hacerlo con `virtualenv` puede ejecutar los siguientes pasos:

.. code:: sh

    $ virtualenv venv --python=python3.10
    $ source venv/bin/activate
    $ pip install --upgrade pip setuptools wheel
    (venv) $ pip install -e .
    
Si desea hacerlo con `conda` entonces:

.. code:: sh

    $ conda create -n env_urbantrips -c conda-forge python=3.10 rvlib
    $ conda activate env_urbantrips
    $ pip install urbantrips
    $ conda install anaconda::git
    
    


**Opción A: Instalación directa desde GitHub (sin clonar el código)**

.. code:: sh

    $ pip install git+https://github.com/EL-BID/UrbanTrips.git@main


**Opción B: Clonar el repositorio para desarrollo o personalización**

.. code:: sh

    $ git clone --branch alpha https://github.com/EL-BID/UrbanTrips.git
    $ cd UrbanTrips
    $ pip install -e .
