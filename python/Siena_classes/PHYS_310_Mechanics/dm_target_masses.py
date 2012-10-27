import matplotlib.pylab as plt
import numpy as np

m1 = 1.0

m2 = m1

v1 = 230

for j in range(1,11):
    m2 = (5.5*j)*m1
    x = []
    y = []
    npts = 1000
    for i in range(0,npts):
        theta = i*(np.pi/npts)
        A = 4*(m1)*(v1**2)*np.cos(theta)**2 
        B = m1 + m2*4*(np.cos(theta)**2)
        v2sq = (A/B)
        #print v2sq

        x.append(theta)
        y.append(np.sqrt(v2sq))

    plt.plot(x,y)

plt.show()
