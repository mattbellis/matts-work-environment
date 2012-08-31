a =  9.8 # m/s^2
x0 = 0.0 # m
v0 = 0.0 # m/s

t = 0 # s

v = v0
x = x0

dt = 0.0001

t += dt

while t <= 3.0:

    dv = dt*a

    v += dv

    dx = dt*v

    x += dx

    print "%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f" % (t,x,dx,v,dv,a)

    t += dt

