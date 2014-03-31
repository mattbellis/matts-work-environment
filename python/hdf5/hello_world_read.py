import numpy as np
import h5py

f = h5py.File('filename.hdf5','r') 

tot = 0.0
for v in f.values():

    #print v
    muons = v.attrs['muons']

    #print sum(muons[0])
    tot += sum(muons[0])

print tot

