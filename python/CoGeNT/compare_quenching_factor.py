import numpy as np
import matplotlib.pylab as plt
import chris_kelso_code as dmm

x = np.linspace(0.01,15,1000)

y = dmm.quench_keVr_to_keVee(x)

y0 = 0.2*(x**(1.12))

print "Compare"
print x[-1],y[-1]
print 1.0,dmm.quench_keVr_to_keVee(1.0)

plt.figure()
plt.plot(x,y0,label='Chris')
plt.plot(x,y,label='Phil')
plt.ylabel('E ionize')
plt.xlabel('E recoil')
plt.legend()

frac_diff = (y-y0)/y

plt.figure()
plt.plot(x,frac_diff,label='Frac diff')
plt.xlabel('(Phil-Chris)/Phil')
plt.xlabel('E recoil')
plt.legend()

plt.show()

