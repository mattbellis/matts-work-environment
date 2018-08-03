import numpy as np
import matplotlib.pylab as plt

import ROOT

import sys

import zipfile

import myPIDselector
from myPIDselector import *

eps = PIDselector("e")
pps = PIDselector("p")
pips = PIDselector("pi")
Kps = PIDselector("K")
mups = PIDselector("mu")

#particles = ["mu","e","pi","K","p"]
particle_masses = [0.000511, 0.105, 0.139, 0.494, 0.938, 0]
particle_lunds = [11, 13, 211, 321, 2212, 22]

################################################################################
'''
def selectPID(eps,mups,pips,Kps,pps,verbose=False):
    #verbose = True
    max_pid = -1
    max_particle = -1
    for i,ps in enumerate([eps,mups,pips,Kps,pps]):
        if verbose:
            ps.PrintSelectors()
        pid = ps.HighestBitFraction()
        #print pid
        if pid>max_pid:
            max_pid = pid
            max_particle = i
    #print max_particle,particles[max_particle]
    return max_particle,max_pid
'''

def selectPID(eps,mups,pips,Kps,pps,verbose=False):
    #verbose = True
    max_pid = 2 # Pion
    max_particle = -1

    s = mups.selectors
    #print(s)
    for i in s:
        #print(i)
        if i.find("Tight")>=0:
            if mups.IsSelectorSet(i):
                return(1,1.0) # Muon
    
    s = eps.selectors
    #print(s)
    for i in s:
        #print(i)
        if i.find("Tight")>=0:
            if eps.IsSelectorSet(i):
                return(0,1.0) # Electron
    
    s = Kps.selectors
    #print(s)
    for i in s:
        #print(i)
        if i.find("Tight")>=0:
            if Kps.IsSelectorSet(i):
                return(3,1.0) # Kaon
    
    s = pps.selectors
    #print(s)
    for i in s:
        #print(i)
        if i.find("SuperTightKM")>=0 or i.find("SuperTightKM")>=0:
            if pps.IsSelectorSet(i):
                return(4,1.0) # proton

     # Otherwise it is a pion
    
    return max_particle,max_pid
################################################################################

################################################################################
# Invariant Mass Function
################################################################################
def invmass(p4):
    if type(p4[0]) != float:
        p4 = list(p4)

    totp4 = np.array([0., 0., 0., 0.])
    for p in p4:
        totp4[0] += p[0]
        totp4[1] += p[1]
        totp4[2] += p[2]
        totp4[3] += p[3]

    m2 = totp4[0]**2 - totp4[1]**2 - totp4[2]**2 - totp4[3]**2

    m = -999
    if m2 >= 0:
        m = np.sqrt(m2)
    else:
        m = -np.sqrt(np.abs(m2))
    return m
################################################################################


################################################################################
def recalc_energy(mass,p3):
    energy = np.sqrt(mass*mass + p3[0]*p3[0] + p3[1]*p3[1] + p3[2]*p3[2] )
    return energy
################################################################################

################################################################################
def sph2cart(pmag,costh,phi):
    theta = np.arccos(costh)
    x = pmag*np.sin(theta)*np.cos(phi)
    y = pmag*np.sin(theta)*np.sin(phi)
    z = pmag*costh
    return x,y,z
################################################################################

f = ROOT.TFile(sys.argv[1])

f.ls()

tree = f.Get("ntp1")

tree.Print()

#exit()

nentries = tree.GetEntries()

#outfilename = "%s.dat" % (sys.argv[1].split('/')[-1].split('.root')[0])
#outfilename = "%s.dat" % (sys.argv[1].split('.root')[0])
#outfile = open(outfilename,'w')

invmasses = []
missmasses = []
nprotons = []

for i in range(nentries):

    #myparticles = {"electrons":[], "muons":[], "pions":[], "kaons":[], "protons":[], "gammas":[]}
    #myparticles = [[], [], [], [], [], []]
    myparticles = []


    if i%1000==0:
        print(i)

    if i>10000:
        break

    output = "Event: %d\n" % (i)
    #output = ""
    tree.GetEntry(i)

    nvals = 0

    beam = np.array([tree.eeE, tree.eePx, tree.eePy, tree.eePz, 0, 0])

    ntrks = tree.nTRK
    #print("----{0}----".format(ntrks))
    for j in range(ntrks):
        #print("track", j)
        ebit,mubit,pibit,Kbit,pbit = tree.eSelectorsMap[j],tree.muSelectorsMap[j],tree.piSelectorsMap[j],tree.KSelectorsMap[j],tree.pSelectorsMap[j]
        #print(ebit,mubit,pibit,Kbit,pbit)
        eps.SetBits(ebit); mups.SetBits(mubit); pips.SetBits(pibit); Kps.SetBits(Kbit); pps.SetBits(pbit);
        max_particle,max_pid = selectPID(eps,mups,pips,Kps,pps,verbose=True)
        #print(max_particle,max_pid)
        e = tree.TRKenergy[j]
        p3 = tree.TRKp3[j]
        phi = tree.TRKphi[j]
        costh = tree.TRKcosth[j]
        px,py,pz = sph2cart(p3,costh,phi)
        lund = tree.TRKLund[j]
        q = int(lund/np.abs(lund))

        new_energy = recalc_energy(particle_masses[max_particle],[px,py,pz])
        particle = [new_energy,px,py,pz,q,particle_lunds[max_particle]]
        myparticles.append(particle)

    ############################################################################
    # Print out the photons
    ############################################################################
    nphotons = tree.ngamma
    output += "%d\n" % (nphotons)
    for j in range(nphotons):
        e = tree.gammaenergy[j]
        p3 = tree.gammap3[j]
        phi = tree.gammaphi[j]
        costh = tree.gammacosth[j]
        px,py,pz = sph2cart(p3,costh,phi)

        #new_energy = recalc_energy(particle_masses[max_particle],[px,py,pz])
        particle = [e,px,py,pz,0,22]
        myparticles.append(particle)

    myparticles = np.array(myparticles)

    #print(myparticles)
    tot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tot += beam

    tpart = myparticles.transpose()
    pids = tpart[-1]
    qs = tpart[-2]
    '''
    print("pids: ",pids)
    print("qs:   ",qs)
    print(len(pids[pids==2212]))
    print(qs.sum())
    '''

    for p in myparticles:
        #print(p)
        #print("{0:-5.3} {1:-5.3} {2:-5.3} {3:-5.3} {4} {5}".format(p[0], p[1], p[2], p[3], int(p[4]), int(p[5])))
        tot -= p
    #print("mass:         ",invmass(myparticles))
    #print(tot)
    #print("missing mass: ",invmass([tot]))
    #if 1:
    if len(pids[pids==2212])>0:
        invmasses.append(invmass(myparticles))
        missmasses.append(invmass([tot]))
        nprotons.append(len(pids[pids==2212]))
        #break

        

plt.figure()
plt.hist(invmasses,bins=200,range=(0,20))
    
plt.figure()
plt.hist(missmasses,bins=200,range=(-10,10))

plt.figure()
plt.hist(nprotons,bins=11,range=(0,10))

plt.show()
