# distutils: language = c++
cdef extern from "myenum.h":
    cpdef enum strategy:
        slow,
        medium,
        fast

