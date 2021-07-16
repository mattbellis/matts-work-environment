import h5py

h5File=h5py.File('xxx.h5','w')

strList=['asas','asas','asas']  

#dt = h5py.special_dtype(vlen=str)
dt = h5py.string_dtype(encoding='utf-8')

dset = h5File.create_dataset('strings',(len(strList),1),dtype=dt)
for i,s in enumerate(strList):
    dset[i] = s

h5File.flush() 
h5File.close()  



#################

f = h5py.File('xxx.h5', 'r')

print(list(f.keys()))

print(f['strings'])
x = f['strings']

for a in x:
    print(a)
