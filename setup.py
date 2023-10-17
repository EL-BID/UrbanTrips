import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().split('\n')[:-1]

setuptools.setup(
    name='urbantrips',
    version='0.2.4',
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
        'contextily==1.4.0',
        'folium==0.14.0',
        'geopandas==0.14.0',
        'h3==3.7.6',
        'ipython==8.16.1',
        'jupyterlab==4.0.6',
        'libpysal==4.8.0',
        'mapclassify==2.6.1',
        'matplotlib-scalebar==0.8.1',
        'mycolorpy==1.5.1',
        'notebook==7.0.4',
        'numba==0.58.0',
        'numpy==1.25.2',
        'openpyxl==3.1.2',
        'osmnx==1.6.0',
        'pandas==2.1.1',
        'patsy==0.5.3',
        'Pillow==9.5.0',
        'plotly==5.17.0',
        'python-pptx==0.6.22',
        'PyYAML==6.0.1',
        'seaborn==0.13.0',
        'shapely==2.0.1',
        'statsmodels==0.14.0',
        'streamlit==1.27.2',
        'streamlit-folium==0.15.0',
        'weightedstats==0.4.1']
)
