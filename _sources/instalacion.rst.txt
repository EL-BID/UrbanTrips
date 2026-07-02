Instalación
===========


Para poder instalar la librería se aconseja crear un ambiente y luego instalar la librería con `pip`. Tambien clonar el repositorio. Si desea hacerlo con `uv` puede ejecutar los siguientes pasos:

.. code:: sh

    $ uv venv .venv --python=3.12
    $ source .venv/bin/activate
    $ git clone --branch main https://github.com/EL-BID/UrbanTrips.git
    $ cd UrbanTrips
    (.venv) $ uv pip install -e .
    
Si desea hacerlo con `conda` entonces:

.. code:: sh

    $ conda create -n env_urbantrips -c conda-forge python=3.10 rvlib
    $ conda activate env_urbantrips
    $ git clone --branch main https://github.com/EL-BID/UrbanTrips.git
    $ cd UrbanTrips
    $ pip install urbantrips
    $ conda install anaconda::git
    $ pip install -e .
 

