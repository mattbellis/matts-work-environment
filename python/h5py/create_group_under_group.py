import h5py
import numpy as np


f = h5py.File('foo.hdf5','w')
print(f.name)

grp = f.create_group("bar")
subgrp = grp.create_group("baz")

print(subgrp.name)
