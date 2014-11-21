import matplotlib.pylab as plt
import numpy as np

npts = 1000

xpos = np.linspace(0,10,npts)

concentration = np.zeros(npts)
concentration[0:npts/2] = 1.0

plt.figure()
plt.plot(xpos,concentration,'o')

################################################################################
# Move in time
################################################################################

dt   = 1
t0   = 0
tmax = 10000

t = t0
while t<tmax:

    if t%100==0:
        print t

    for i in range(1,npts-1):
        c0 = concentration[i-1]
        c1 = concentration[i+1]

        concentration[i] = (c0+c1)/2.0

    #print concentration

    t += dt

plt.figure()
plt.plot(xpos,concentration,'o')
#plt.show()

