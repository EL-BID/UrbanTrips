import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().split('\n')[:-1]

setuptools.setup(
    name='urbantrips',
    version='0.2.1',
    author="Felipe Gonzalez & Sebastian Anapolsky",
    author_email="",
    description="A library to process public transit smart card data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EL-BID/UrbanTrips",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'openpyxl',
        'osmnx>=1.3.0',
        'geopandas>=0.12.2',
        'shapely>=2.0.1',
        'pandas>=2.0.2',
        'contextily',
        'h3 < 4',
        'mapclassify',
        'weightedstats',
        'pyyaml',
        'libpysal>=4.7.0',
        'statsmodels',
        'patsy',
        'folium',
        'seaborn',
        'IPython',
        'matplotlib-scalebar',
        'numba',
        'mycolorpy',
        'python-pptx',
        'streamlit',
        'streamlit_folium',
        'Pillow',
        'plotly',
        'notebook',
        'jupyterlab']
)
