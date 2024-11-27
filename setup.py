import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().split('\n')[:-1]

setuptools.setup(
    name='urbantrips',
    version='0.3.0',
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
        'anyio',   
        'contextily',
        'folium',
        'fiona',
        'geopandas',
        'fiona',
        'h3<4',
        'ipython',
        'jupyterlab<4.4',
        'jupyter',
        'libpysal',
        'mapclassify',
        'matplotlib',
        'matplotlib-scalebar',
        'notebook',
        'numba',
        'numpy<2',
        'openpyxl',
        'osmnx',
        'pandas',
        'pandana',
        'patsy',
        'Pillow<11',
        'platformdirs',
        'plotly',
        'python-pptx',
        'PyYAML',
        'seaborn',
        'shapely',
        'statsmodels',
        'streamlit',
        'streamlit-folium',
        'typing_extensions',
        'weightedstats',
        'OSMnet',
]
)
