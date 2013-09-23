import sys
import numpy as np
import matplotlib.pylab as plt

infile0 = open(sys.argv[1])

xpts = np.array([])
ypts = np.array([])
xerr = np.array([])
yerr = np.array([])


for line in infile0:
    vals = line.split()    
    x = float(vals[0])
    y = float(vals[1])
    xpts = np.append(xpts,x)
    ypts = np.append(ypts,y)

yerr = np.sqrt(ypts)
xerr = np.zeros(len(yerr))

fig = plt.figure()
plt.errorbar(xpts,ypts,yerr=yerr,fmt='o',markersize=2)
plt.yscale('log')
#plt.xlim(0,900)
#plt.ylim(0,700)
#fig.savefig('fig2.png')

plt.show()
