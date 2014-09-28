import sys
import numpy as np
import matplotlib.pylab as plt

infile0 = open(sys.argv[1])
infile1 = open(sys.argv[2])

xpts = [np.array([]),np.array([])]
ypts = [np.array([]),np.array([])]

for line in infile0:
    vals = line.split()    
    x = float(vals[0])
    y = float(vals[1])
    xpts[0] = np.append(xpts[0],x)
    ypts[0] = np.append(ypts[0],y)

for line in infile1:
    vals = line.split()    
    x = float(vals[0])
    y = float(vals[1])
    xpts[1] = np.append(xpts[1],x)
    ypts[1] = np.append(ypts[1],y)

distances = np.zeros(len(xpts[0]))
min_distances = np.zeros(len(xpts[0]))

for i in range(0,len(xpts[0])):
    min_dist = 100000;
    for j in range(0,len(xpts[1])):
        deltax = xpts[0][i]-xpts[1][j]
        deltay = ypts[0][i]-ypts[1][j]
        dist = np.sqrt(deltax*deltax+(deltay*deltay))
        if dist<=min_dist and dist==dist:
            distances[i] = j
            min_distances[i] = dist
            min_dist = dist
            #print xpts[0][i],xpts[1][j],ypts[0][i],ypts[1][j]
            #print dist,min_dist


fig = plt.figure()
plt.plot(xpts[0],ypts[0],"ro",markersize=3)
plt.plot(xpts[1],ypts[1],"bo",markersize=3)
plt.xlim(0,900)
plt.ylim(0,700)
fig.savefig('fig2.png')

#print distances[0:10]
#print min_distances[0:10]
fig = plt.figure()
for i in range(0,len(xpts[0])):
    if min_distances[i]<50:
        newxpts = [xpts[0][i], xpts[1][distances[i]]]
        newypts = [ypts[0][i], ypts[1][distances[i]]]
        print newxpts,newypts
        plt.plot(newxpts,newypts,'g')
plt.xlim(0,900)
plt.ylim(0,700)
fig.savefig('velocities.png')


plt.show()
