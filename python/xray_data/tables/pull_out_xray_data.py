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

for v in vals[index+3]:
    print v.replace('\xc3\x8e\xc2\xb11','alpha').replace('\xc3\x8e\xc2\xb2','beta')
