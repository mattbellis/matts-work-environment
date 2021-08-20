############# WORKS ###############################
import h5py


'''
h5File=h5py.File('outfile.h5','w')

sentence=['this','is','a','sentence']  
data = []

for i in range(10000):
    data += sentence

print(len(data))

dt = h5py.special_dtype(vlen=str)

dset = h5File.create_dataset('words',(len(data),1),dtype=dt)
for i,word in enumerate(data):
    dset[i] = word

h5File.flush() 
h5File.close()  
'''

############# DOESN'T WORK ###############################
'''
h5File=h5py.File('outfile.h5','w')

data_numbers = [0, 1, 2, 3, 4]
data = ['this','is','a','sentence']  

dt = h5py.special_dtype(vlen=str)

dset_num = h5File.create_dataset('numbers',(len(data_numbers),1),dtype=int,data=data_numbers)
print("Created the dataset with numbers!\n")

dset_str = h5File.create_dataset('words',(len(data),1),dtype=dt,data=data)
print("Created the dataset with strings!\n")

h5File.flush() 
h5File.close()  
'''

import h5py
import numpy as np

#h5File=h5py.File('outfile.h5','w')

sentence=['this','is','a','sentence']
data = []

for i in range(10000):
    data += sentence
print(len(data))
longest_word=len(max(data, key=len))
print('longest_word=',longest_word)

dt = h5py.special_dtype(vlen=str)

arr = np.array(data,dtype='S'+str(longest_word))
with h5py.File('outfile.h5','w') as h5File:
    dset = h5File.create_dataset('words',data=arr,dtype=dt, compression='gzip',compression_opts=9)
    print(dset.shape, dset.dtype)

    h5File.flush() 
    h5File.close()  
