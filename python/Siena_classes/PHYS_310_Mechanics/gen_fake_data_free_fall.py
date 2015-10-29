import numpy as np
import matplotlib.pylab as plt

#g = -8.1
#g = -5.0
g = -9.8

#x0 = 15.0
x0 = 0.0

tpts = []
xpts = []
xerrpts = []

for t in np.arange(0,100,0.10):

    x = x0 + 0.5*g*t*t

    xerr = t*0.2 + 0.25
    x = np.random.normal(x,xerr)

    print "%4.2f %4.2f %4.2f" % (t,x,xerr)
    tpts.append(t)
    xpts.append(x)
    xerrpts.append(xerr)

    if x<-10:
        break

#plt.errorbar(tpts,xpts,fmt='o',yerr=xerrpts)
#plt.xlabel('Time (s)',fontsize=16)
#plt.ylabel('Height above ground (m)',fontsize=16)
#plt.show()
