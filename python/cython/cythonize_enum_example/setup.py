from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(
    "pymyenum.pyx",                 # our Cython source
    sources=["myenum.h"],  # additional source file(s)
    language="c",             # generate C code
    ))

