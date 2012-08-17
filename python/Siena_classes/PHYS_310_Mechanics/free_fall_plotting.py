from numpy import arange
from matplotlib.pylab import figure,plot,show,xlabel,ylabel

a =  9.8 # m/s^2
x0 = 0.0 # m
v0 = 0.0 # m/s

t0 = 0 # sec
tfinal = 3.0 # sec

dt = 0.1

t = arange(t0,tfinal,dt)

v = a*t + v0
x = (1/2.0) * a * (t**2) + v0*t + x0

print t
print x

figure()
plot(t,x)
xlabel("t (s)")
ylabel("x (m)")

figure()
plot(t,v)
xlabel("t (s)")
ylabel("v (m/s)")

show()



