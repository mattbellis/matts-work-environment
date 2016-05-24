import numpy as np
import matplotlib.pylab as plt

import scipy.constants as constants
import scipy.constants as constants
from scipy.special import erf
from scipy.integrate import quad

pi = constants.pi
kb = constants.k
mp = constants.m_p # Mass of proton

################################################################################
def mb_dist(v,m,T):

    k1 = np.sqrt(2/pi)
    sigma = np.sqrt(kb*T/m)
    sigma2 = sigma*sigma
    sigma3 = sigma*sigma*sigma

    y = k1*(v*v)*np.exp(-v*v/(2*sigma2))/(sigma3)

    return y

################################################################################

################################################################################
def mb_cdist(v,m,T):

    k1 = np.sqrt(2/pi)
    sigma = np.sqrt(kb*T/m)
    sigma2 = sigma*sigma

    y = erf(v/(np.sqrt(2)*sigma)) - (k1*v*np.exp(-v*v/(2*sigma2)))/sigma

    return y

################################################################################

# Earth, moon, titan
#bodies = ['Earth','Moon','Titan']
#masses = [6e24, 7.3e22, 1.3e23]
#radii = [6e6, 1.7e6, 2.6e6]
#temps = [300.0,400.0,90.0]
#bodies = ['Earth']
#masses = [6e24]
#radii = [6e6]
#temps = [300.0]
bodies = ['Moon']
masses = [ 7.3e22]
radii = [ 1.7e6]
temps = [400.0]

mols = ['H2','O2','N2']
amasses = [2*mp,32*mp,28*mp]


G = constants.G

print G

vesc = []
for m,r in zip(masses,radii):

    v = np.sqrt(2*G*m/r)

    vesc.append(v)

    print "vesc: ",v


# MB distribution
# http://www.wolframalpha.com/input/?i=Maxwell-boltzman+distribution

x = np.linspace(0,4000,10000)

plt.figure()
for i,T in enumerate(temps):
    for j,am in enumerate(amasses):
        y = mb_dist(x,am,T)
        label = "m=%e, T=%f" % (am,T)
        # if i==2:
        if 1:
            plt.plot(x,y,label=label,linewidth=3)
            plt.plot((vesc[i],vesc[i]),[0,.014],'k-')

for i,T in enumerate(temps):
    for j,am in enumerate(amasses):
        y = mb_cdist(vesc[i],am,T)
        y1 = quad(mb_dist,0,vesc[i],(am,T))[0]
        #y = mb_cdist(1000.0,am,T)
        print "%s %s %f %f %e %e %e" % (bodies[i],mols[j],T,vesc[i],am,1.0-y,1.0-y1)


print "HEREJHRKJE"
plt.xlim(0,4000)
#mb = lambda x,a,t: mb_dist(x,a,t)
#vmax = vesc[0]
vmax = 2583.0
#print quad(mb_dist,0,vmax,(amasses[2],temps[2]))
#print quad(mb,0,vmax,(amasses[0],temps[0]))
#print mb_cdist(vmax,amasses[2],temps[2])

plt.legend()
plt.show()
        

