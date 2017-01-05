from numpy import arange,ones,log
from matplotlib.pylab import figure,plot,show,xlabel,ylabel,ylim,legend
import numpy as np


g = 9.8 # m/s

name = ["Steel ball","Styrofoam ball","Baseball"]
radii = np.array([0.025,  0.10, 0.035  ]) # m
masses = np.array([0.5375, 50*(4/3.)*(radii[1]**3), 0.15 ]) # kg

D = 2*radii

beta = 1.6e-4
gamma = 0.25

b = beta*D
c = gamma*(D**2)

v = np.sqrt(2*(g)*2)

fquad = c*(v**2)
flin  = b*v

print b
print c
print fquad,flin

for i in range(0,2):
    print "%-15s %f %f %f %f" % (name[i], radii[i],masses[i],fquad[i],flin[i])


def proj_drag(h,t,vterm):

    y = h - ((vterm**2)/g) * np.log(np.cosh(g*t/vterm))

    return y

def proj_no_drag(h,t):

    y = h - (0.5*g*(t**2))

    return y


vterm = np.sqrt(masses*g/c)

print vterm

h = 2.0 # meters

t = np.arange(0,np.sqrt(2*h/g)+0.05,0.01)
#print t

figure()
for i,vt in enumerate(vterm):
    y = proj_drag(h,t,vt)
    #print y
    t = t[y==y]
    y = y[y==y]
    plot(t,y,label=name[i])

xlabel("x (m)")
ylabel("y (m)")
ylim(0.0)
legend()
show()



