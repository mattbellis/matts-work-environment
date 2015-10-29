import cleo_tools as cleo

import sys

import numpy as np
import matplotlib.pylab as plt

def energy(p3,mass):

    return np.sqrt(mass*mass + pmag([p3])**2)

def pmag(p3s):
    px,py,pz = 0,0,0
    for p3 in p3s:
        px += p3[0]
        py += p3[1]
        pz += p3[2]
    return np.sqrt(px*px + py*py + pz*pz)



def add4vecs(p4s):
    e,px,py,pz = 0,0,0,0
    for p4 in p4s:
        e += p4[0]
        px += p4[1]
        py += p4[2]
        pz += p4[3]
    return [e,px,py,pz]


def mass(p4s):
    e,px,py,pz = 0,0,0,0
    for p4 in p4s:
        e += p4[0]
        px += p4[1]
        py += p4[2]
        pz += p4[3]
    return np.sqrt(e*e - px*px - py*py - pz*pz)




infile = open(sys.argv[1],'r')

collisions = cleo.get_collisions(infile,True)

Jpsimasses = []
Bmasses = []
mJ = []
mJp = []
mkp = []
mJcon = []
mJconp = []

for c in collisions:

    # Pions are really protons
    pions,kaons,muons = c[0:3]
    npions = len(pions)
    nkaons = len(kaons)
    nmuons = len(muons)

    # Proton
    p40 = pions[0][0:4]
    p41 = kaons[0][0:4]
    p42 = muons[0][0:4]
    p43 = muons[1][0:4]

    m = mass([p40,p41,p42,p43])
    Bmasses.append(m)

    m = mass([p42,p43])
    mJ.append(m)

    J = add4vecs([p42,p43])
    J[0] = energy(J[1:4],3.096)
    mJcon.append(mass([J]))

    m = mass([p40,p42,p43])
    mJp.append(m)

    m = mass([p40,J])
    mJconp.append(m)

    m = mass([p40,p41])
    mkp.append(m)

mJp = np.array(mJp)
mJconp = np.array(mJconp)
mkp = np.array(mkp)
#print len(Dmasses)
#plt.figure()
#plt.hist(Dmasses,bins=100,range=(0,5))

#plt.figure()
#plt.hist(Dsmasses,bins=100,range=(0,5))

plt.figure()
plt.subplot(2,2,1)
plt.hist(Bmasses,bins=100,range=(0,9))
plt.subplot(2,2,2)
plt.hist(mJ,bins=100,range=(2,4))
plt.subplot(2,2,4)
plt.hist(mJcon,bins=100,range=(2,4))

plt.figure()

plt.subplot(2,2,1)
plt.plot(mkp*mkp,mJp*mJp,'o',markersize=0.5)
plt.ylim(15,27)
plt.xlim(2,7.5)

plt.subplot(2,2,2)
plt.plot(mkp*mkp,mJconp*mJconp,'o',markersize=0.5)
plt.ylim(15,27)
plt.xlim(2,7.5)

plt.subplot(2,2,3)
index =  np.abs(mJconp**2-19.8)<0.2
print len(index[index==True])
plt.plot(mkp[index]**2,mJconp[index]**2,'o',markersize=0.5)
plt.ylim(15,27)
plt.xlim(2,7.5)

plt.subplot(2,2,4)
plt.plot(mkp[index]**2,mJp[index]**2,'o',markersize=0.5)
plt.ylim(15,27)
plt.xlim(2,7.5)

plt.show()
