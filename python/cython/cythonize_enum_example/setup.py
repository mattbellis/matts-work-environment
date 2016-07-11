from distutils.core import setup,Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(
    Extension("pymyenum",
    sources=["pymyenum.pyx"],  # additional source file(s)
    language="c++",             # generate C code
    )))

