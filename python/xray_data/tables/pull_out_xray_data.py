import numpy as np
import sys

f = open('tmp3.tmp')

vals = np.array(f.read().split())

nentries = len(vals)
print len(vals)

index = np.arange(0,nentries,5)

print vals[index]
print vals[index+1]
print vals[index+2]
print vals[index+3]
print vals[index+4]

for x,v in zip(vals[index],vals[index+3]):
    print x,v.replace('\xc3\x8e\xc2\xb1','alpha').replace('\xc3\x8e\xc2\xb2','beta').replace('\xc3\x8e\xc2\xb2','beta').replace('\xc3\x8e\xc2\xb3','gamma')
