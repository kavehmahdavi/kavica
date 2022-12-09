'''
To build, run python setup.py build_ext --inplace
Then simply start a Python session and do from hello import say_hello_to and use the imported function as you see fit.
'''

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("prvparse.pyx"),
    include_dirs=[numpy.get_include()]
)
