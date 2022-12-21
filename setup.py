import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().split('\n')[:-1]

setuptools.setup(
     name='urbantrips',
     version='__version__', # This will be replaced with the commit tag
     author="Felipe Gonzalez & Sebastian Anapolsky",
     author_email="",
     description="A library to process public transit smart card data.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/EL-BID/urbanpy",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
     install_requires=install_requires,
     python_requires='>=3.9'
 )
