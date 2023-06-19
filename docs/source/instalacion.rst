Instalación
===========


Para poder instalar la librería se aconseja crear un ambiente y luego instalar la librería con `pip`. Si desea hacerlo con `virtualenv` puede ejecutar los siguientes pasos:

.. code:: sh

    $ virtualenv venv --python=python3.10
    (.venv) $ source venv/bin/activate
    (.venv) $ pip install urbantrips
    
Si desea hacerlo con `conda` entonces:

.. code:: sh

    $ conda create -n env_urbantrips -c conda-forge python=3.10 rvlib
    $ conda activate env_urbantrips
    $ pip install urbantrips

