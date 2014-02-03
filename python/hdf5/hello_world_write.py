import numpy as np
import h5py

f = h5py.File('filename.hdf5','w')

for i in range(0,10000):
    name = "mydataset%05d" % (i)
    dset = f.create_dataset(name, (1,), dtype='f')
    #dset = f.create_dataset(name)

    dset.attrs['muons']  = np.array([[1.0,2.0,3.0,4.0], 
                            [1.0,2.0,3.0,4.0],
                            [1.0,2.0,3.0,4.0],
                            [1.0,2.0,3.0,4.0] ])

    #print dset.attrs['muons']
    #print dset.name

f.close()
