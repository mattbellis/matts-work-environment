#cython hello_world_0.pyx
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o hello_world_0.so hello_world_0.c
python setup.py build_ext --inplace
