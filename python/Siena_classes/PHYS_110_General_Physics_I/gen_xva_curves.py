import numpy as np
import matplotlib.pylab as plt
import matplotlib

import matplotlib as mpl
mpl.rcParams['grid.linewidth'] = 3

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 


#'''
t_intervals = [0,2,5,6,8]
a_intervals = [0,-20,-5,15]
v0 = 20.0
x0 = 0
tag = "A"
#'''

'''
t_intervals = [0,2,5,6,8]
a_intervals = [10,20,-5,-15]
v0 = 0.0
x0 = 10
tag = "B"
'''

'''
t_intervals = [0,2,5,6,8]
a_intervals = [10,-10,0,5]
v0 = -10.0
x0 = 0
tag = "C"
'''

dt = 0.01
t = np.arange(0,t_intervals[-1],dt)
a = np.ones(len(t))
v = np.ones(len(t))
x = np.ones(len(t))

v[0] = v0
x[0] = x0

for i in range(0,len(t_intervals)-1):
    idx0 = t>t_intervals[i]
    idx1 = t<=t_intervals[i+1]
    a[idx0*idx1] = a_intervals[i]

for i in range(1,len(t)):
    acc = a[i]
    dv = acc*dt
    v[i] = v[i-1] + dv

    dx = dt*v[i]
    x[i] = x[i-1] + dx

plt.figure()
plt.plot(t,a,linewidth=5)
plt.ylim(min(a)-1,max(a)+1)
plt.xlabel("Time (s)",fontsize=24)
plt.ylabel(r"Acceleration (m/s$^2$)",fontsize=24)
plt.grid()
plt.tight_layout()
name = "acceleration_%s.png" % (tag)
plt.savefig(name)

plt.figure()
plt.plot(t,v,linewidth=5)
plt.ylim(min(v)-1,max(v)+1)
plt.xlabel("Time (s)",fontsize=24)
plt.ylabel("Velocity (m/s)",fontsize=24)
plt.grid()
plt.tight_layout()
name = "velocity_%s.png" % (tag)
plt.savefig(name)

plt.figure()
plt.plot(t,x,linewidth=5)
plt.ylim(min(x)-1,max(x)+1)
plt.xlabel("Time (s)",fontsize=24)
plt.ylabel("Position (m)",fontsize=24)
plt.grid()
plt.tight_layout()
name = "position_%s.png" % (tag)
plt.savefig(name)

plt.show()


