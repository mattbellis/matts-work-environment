import sys
import numpy as np

vals = np.loadtxt(sys.argv[1],unpack=True,skiprows=1,dtype=str,delimiter=',')


header = open(sys.argv[1]).readline()

print(header)
print(vals)

for i,h in enumerate(header.split(',')):
    print(i,h)

ncols = len(vals)
print("There are {0} columns".format(ncols))

col = None
if len(sys.argv)>2:
    col = int(sys.argv[2])
    assignment = header.split(',')[col]
    print("Outputting grades for: ")
    print(assignment)


print()
for i in range(len(vals[0])):
    a = vals[0][i]
    b = vals[1][i]
    c = vals[2][i]
    output = '{0:16} {1:16} {2:24}'.format(a,b,c)
    if col is not None:
        output += '{0:8} '.format(vals[col][i]) 
    else:
        for j in range(3,ncols):
            output += '{0:8} '.format(vals[j][i]) 
    #print("----")
    #print(a,b)
    ncols = len(vals.transpose()[i])
    #print(vals.transpose()[i])
    #print(ncols)
    grades = vals.transpose()[i][3:].astype(float)
    #print(type(grades[0]))
    #print(grades.shape)
    #print(grades)
    #print(np.mean(grades.astype(float)))
    output += '{0:16.2f} '.format(np.mean(grades))
    print(output)


