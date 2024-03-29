import numpy as np
import matplotlib.pylab as plt

import sys

from scipy.spatial.distance import pdist,squareform

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
    #nseeds = 1

    # Minkowski distance
    print("Minkowski")
    u = np.array([t,x,y,z]).T
    print(u)
    d = pdist(u,"Minkowski")
    print(d)
    print(np.sort(d))
    sfd = squareform(d)
    for r in sfd:
        output = ""
        for c in r:
            if c<0.3:
                output += f"\033[31m{c:0.2f}\033[0m "
            else:
                output += f"{c:0.2f} "
        output += "\n"
        print(output)
    exit()


    #'''
    #for i in range(nseeds):
    for i in range(1):
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
                #plt.plot(dt,r,'ro')
                plt.errorbar(dt,r,fmt='ro',xerr=2e-9)
            else:
                plt.plot(dt,r,'ko')

        plt.plot([0,50e-9],[0,50e-9*3e8],'k--')
        plt.xlabel(r'$\Delta t$ (s)',fontsize=18)
        plt.ylabel(r'Distance (m)',fontsize=18)
        #break

        plt.show()
    #'''

    '''
    # Find the line
    for i in range(nseeds):
        xpt0,ypt0,zpt0 = x[i],y[i],z[i]
        for j in range(nseeds):

            if z[j]<z[i]:
                continue

            xpt1,ypt1,zpt1 = x[j],y[j],z[j]
            r = distance([xpt0,ypt0,zpt0],[xpt1,ypt1,zpt1])
            dt = t[j] - t[i]
            if ns[j]==1:
                #plt.plot(dt,r,'ro')
                plt.errorbar(dt,r,fmt='ro',xerr=2e-9)
            else:
                plt.plot(dt,r,'ko')
    '''







plt.figure()
plt.hist(ztmp,bins=20)

plt.show()
