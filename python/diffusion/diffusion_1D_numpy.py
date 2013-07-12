import matplotlib.pylab as plt
import numpy as np

npts = 1000

xpos = np.linspace(0,10,npts)

concentration = np.zeros(npts)
c0 = np.zeros(npts)
c1 = np.zeros(npts)
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

    c00 = np.roll(concentration,2)
    c00[0] = 1.0
    c00[1] = 1.0
    c00[-1] = 0.0
    c00[-2] = 0.0

    c01 = np.roll(concentration,1)
    c01[0] = 1.0
    c01[1] = 1.0
    c01[-1] = 0.0
    c01[-2] = 0.0

    c10 = np.roll(concentration,-1)
    c10[0] = 1.0
    c10[1] = 1.0
    c10[-1] = 0.0
    c10[-2] = 0.0

    c11 = np.roll(concentration,-2)
    c11[0] = 1.0
    c11[1] = 1.0
    c11[-1] = 0.0
    c11[-2] = 0.0


    #concentration = (c0+c1)/2.0
    concentration = (c00+c01+c10+c11)/4.0

    concentration[0] = 1.0
    concentration[1] = 1.0
    concentration[-1] = 0.0
    concentration[-2] = 0.0

    #print concentration

    t += dt

plt.figure()
plt.plot(xpos,concentration,'o')
#plt.show()

