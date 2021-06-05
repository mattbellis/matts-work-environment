import numpy as np
import matplotlib.pylab as plt

import sys

infilename = sys.argv[1]

# ns is noise-or-signal
n,nstmp,xtmp,ytmp,ztmp,ttmp,dedxtmp = np.loadtxt(infilename,dtype=float,delimiter=',',unpack=True)

n = n.astype(int)

nvals = np.unique(n)
print(nvals)

##############################################################
def distance(v0,v1):

    dx = v0[0] - v1[0]
    dy = v0[1] - v1[1]
    dz = v0[2] - v1[2]

    d2 = dx*dx + dy*dy + dz*dz

    return np.sqrt(d2)
##############################################################


for nval in nvals:
    mask = n == nval
    ns = nstmp[mask]
    x = xtmp[mask]
    y = ytmp[mask]
    z = ztmp[mask]
    t = ttmp[mask]
    dedx = dedxtmp[mask]

    mask2 = dedx>30
    mask3 = z<0

    mask_seeds = mask2 & mask3

    ns = ns[mask_seeds]
    x = x[mask_seeds]
    y = y[mask_seeds]
    z = z[mask_seeds]
    t = t[mask_seeds]
    dedx = dedx[mask_seeds]

    print(t)

    nseeds = len(z)

    for i in range(nseeds):
        xpt0,ypt0,zpt0 = x[i],y[i],z[i]
        for j in range(nseeds):
            '''
            if i==j:
                continue
            '''

            if z[j]<z[i]:
                continue
            xpt1,ypt1,zpt1 = x[j],y[j],z[j]
            r = distance([xpt0,ypt0,zpt0],[xpt1,ypt1,zpt1])
            dt = t[j] - t[i]
            if ns[j]==1:
                plt.plot(dt,r,'ro')
            else:
                plt.plot(dt,r,'ko')

        plt.plot([0,50e-9],[0,50e-9*3e8],'k--')
        plt.show()






plt.hist(ztmp,bins=20)
plt.show()
