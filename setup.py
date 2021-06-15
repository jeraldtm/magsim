from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='magsim',
    version='0.1',
    packages=['magsim'],
    description='LLGS solver simulation package',
    long_description=open('README.rst').read(),
    include_dirs=[numpy.get_include()]
    # install_requires = [
    #     "xarray >= 0.10.8",
    #     "pandas >= 0.20.3",
    #     "scipy >= 0.19.1"
    # ]
)
