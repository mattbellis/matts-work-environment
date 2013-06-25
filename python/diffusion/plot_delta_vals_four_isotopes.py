import numpy as np
import matplotlib.pylab as plt

import sys

ncolumns = 6

y0 = [None,None,None,None,None]
y1 = [None,None,None,None,None]
x = None

################################################################################
# First file
################################################################################
infile0 = open(sys.argv[1])
content0 = np.array(infile0.read().split()).astype('float')
nvals = len(content0)
index = np.arange(0,nvals,ncolumns)
x = content0[index]
y0[0] = content0[index+1]
y0[1] = content0[index+2]
y0[2] = content0[index+3]
y0[3] = content0[index+4]
y0[4] = content0[index+5]

################################################################################
# Second file
################################################################################
infile1 = open(sys.argv[2])
content1 = np.array(infile1.read().split()).astype('float')
nvals = len(content1)
index = np.arange(0,nvals,ncolumns)
x = content1[index]
y1[0] = content1[index+1]
y1[1] = content1[index+2]
y1[2] = content1[index+3]
y1[3] = content1[index+4]
y1[4] = content1[index+5]

fig0 = plt.figure()
axes0 = []
vals = [None,None,None,None,None]
for i in range(0,4):
    axes0.append(fig0.add_subplot(2,2,i+1))
    vals[i+1] = (y0[i+1]/y1[i+1] - 1)*1000.0
    plt.plot(x,vals[i+1],'o')
    #plt.ylim(0,7.0)

fig_bulk0 = plt.figure()
axes_bulk0 = fig_bulk0.add_subplot(1,1,1)
vals[0] = (y0[0]/y1[0] - 1)*1000.0
axes_bulk0.plot(x,vals[0],'ro')

name = "deltas_default.dat"
outfile = open(name,'w+')

output = ""
for xpt,v0,v1,v2,v3,v4 in zip(x,vals[0],vals[1],vals[2],vals[3],vals[4]):
    output += "%f %f %f %f %f %f\n" % (xpt,v0,v1,v2,v3,v4)
outfile.write(output)
outfile.close()

plt.show()
