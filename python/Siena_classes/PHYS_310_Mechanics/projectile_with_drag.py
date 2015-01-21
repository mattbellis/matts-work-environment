from numpy import arange,ones,log
from matplotlib.pylab import figure,plot,show,xlabel,ylabel,ylim

g = -9.8 # m/s

def proj_drag(vx0,vy0,vterm,tau,x):

    y = ((vy0+vterm)/(vx0))*x + vterm*tau*log(1-(x/(vx0*tau)))

    return y

def proj_no_drag(vx0,vy0,x):

    y = vy0*x/vx0 + 0.5*g*(x/vx0)**2

    return y


vx0 = 80.0
vy0 = 120.0

R = -2*vx0*vy0/g
x = arange(0,R,0.1)

print R


y = proj_no_drag(vx0,vy0,x)

figure()
plot(x,y)
xlabel("x (m)")
ylabel("y (m)")

for tau in [10,50,100,200]:
    vterm = -tau*g
    x = arange(0,R,0.1)
    y = proj_drag(vx0,vy0,vterm,tau,x)
    x = x[y>=0]
    y = y[y>=0]
    plot(x,y)

show()



