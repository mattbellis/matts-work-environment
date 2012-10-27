import matplotlib.pylab as plt
import numpy as np

m1 = 1.0

m2 = m1*10

v1 = 230

x = []
y = []
npts = 1000
for i in range(0,npts):
    theta = i*(np.pi/npts)
    A = 16*(m2**2)*np.cos(theta)**4 
    B = 16*m1*(v1**2)*np.cos(theta)**2 
    C = -4*m2*np.cos(theta)**2 
    print A,np.sqrt(A+B)
    v2sq = (C + np.sqrt(A+B))/(2*m1)
    #print v2sq

    x.append(theta)
    y.append(np.sqrt(v2sq))

plt.plot(x,y)
plt.show()
