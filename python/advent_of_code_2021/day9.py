import numpy as np
import sys

################################################################################

data = np.loadtxt(sys.argv[1],dtype=str)

print(data)

temp = []
for d in data:
    row = []
    print(d)
    for i in d:
        row.append(int(i))
    temp.append(row)
print(temp)

data = np.array(temp)

width = len(data[0])
height = len(data)

print(width,height)

lowest = np.zeros(shape=data.shape,dtype=bool)
print(lowest)

print(data.size)

for i in range(0,height):
    for j in range(0,width):
        print(i,j)
        val = data[i][j]
        if i==0 and j==0:
            if val<data[i+1][j] and val<data[i][j+1]:
                lowest[i][j] = True
        if i==0 and j==width-1:
            if val<data[i+1][j] and val<data[i][j-1]:
                lowest[i][j] = True

print(lowest)
            

