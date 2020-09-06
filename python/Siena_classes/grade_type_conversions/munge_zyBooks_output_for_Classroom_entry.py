import sys
import numpy as np

vals = np.loadtxt(sys.argv[1],unpack=True,skiprows=1,dtype=str,delimiter=',')


header = open(sys.argv[1]).readline()

print(header)
print(vals)

col = 3

assignment = header[col]

print(assignment)
print()
for i in range(len(vals[0])):
    a = vals[0][i]
    b = vals[1][i]
    c = vals[2][i]
    d = vals[col][i]
    print('{0:16} {1:16} {2:24} {3:16}'.format(a,b,c,d)) 


