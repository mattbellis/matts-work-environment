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

totx = []
toty = []
totz = []


#particles = ["mu","e","pi","K","p"]
particle_masses = [0.000511, 0.105, 0.139, 0.494, 0.938, 0]
particle_lunds = [11, 13, 211, 321, 2212, 22]

allparts = [{}, {}, {}]

for pl in particle_lunds:
    allparts[0][pl] = []
    allparts[1][pl] = []
    allparts[2][pl] = []

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
                return 1,1.0 # Muon
    
    s = eps.selectors
    #print(s)
    for i in s:
        #print(i)
        if i.find("Tight")>=0:
            if eps.IsSelectorSet(i):
                return 0,1.0 # Electron
    
    s = Kps.selectors
    #print(s)
    for i in s:
        #print(i)
        if i.find("Tight")>=0:
            if Kps.IsSelectorSet(i):
                return 3,1.0 # Kaon
    
    s = pps.selectors
    #print(s)
    for i in s:
        #print(i)
        if i.find("SuperTightKM")>=0 or i.find("SuperTightKM")>=0:
            if pps.IsSelectorSet(i):
                return 4,1.0 # proton

     # Otherwise it is a pion
    
    return max_pid,max_particle
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
totq = []

bcand = []
bcandMES = []
bcandDeltaE = []
bcandMM = []

for i in range(nentries):

    #myparticles = {"electrons":[], "muons":[], "pions":[], "kaons":[], "protons":[], "gammas":[]}
    #myparticles = [[], [], [], [], [], []]
    myparticles = []


    if i%1000==0:
        print(i)

    if i>1000:
        break

    output = "Event: %d\n" % (i)
    #output = ""
    tree.GetEntry(i)

    nvals = 0

    #beam = np.array([tree.eeE, tree.eePx, tree.eePy, tree.eePz, 0, 0])
    beamp4 = np.array([tree.eeE, tree.eePx, tree.eePy, tree.eePz])
    beammass = invmass([beamp4])
    beam = np.array([beammass, 0.0, 0.0, 0.0, 0, 0])

    ntrks = tree.nTRK
    #print("----{0}----".format(ntrks))
    for j in range(ntrks):
        #print("track", j)
        ebit,mubit,pibit,Kbit,pbit = tree.eSelectorsMap[j],tree.muSelectorsMap[j],tree.piSelectorsMap[j],tree.KSelectorsMap[j],tree.pSelectorsMap[j]
        #print(ebit,mubit,pibit,Kbit,pbit)
        eps.SetBits(ebit); mups.SetBits(mubit); pips.SetBits(pibit); Kps.SetBits(Kbit); pps.SetBits(pbit);
        max_particle,max_pid = selectPID(eps,mups,pips,Kps,pps,verbose=True)
        #print(max_particle,max_pid)
        #e = tree.TRKenergy[j]
        #p3 = tree.TRKp3[j]
        #phi = tree.TRKphi[j]
        #costh = tree.TRKcosth[j]
        e = tree.TRKenergyCM[j]
        p3 = tree.TRKp3CM[j]
        phi = tree.TRKphiCM[j]
        costh = tree.TRKcosthCM[j]
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
        #e = tree.gammaenergy[j]
        #p3 = tree.gammap3[j]
        #phi = tree.gammaphi[j]
        #costh = tree.gammacosth[j]
        e = tree.gammaenergyCM[j]
        p3 = tree.gammap3CM[j]
        phi = tree.gammaphiCM[j]
        costh = tree.gammacosthCM[j]
        px,py,pz = sph2cart(p3,costh,phi)

        new_energy = recalc_energy(0,[px,py,pz])
        #particle = [e,px,py,pz,0,22]
        particle = [new_energy,px,py,pz,0,22]
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

        key = p[-1]
        allparts[0][key].append(p[1])
        allparts[1][key].append(p[2])
        allparts[2][key].append(p[3])

    #print("mass:         ",invmass(myparticles))
    #print(tot)
    #print("missing mass: ",invmass([tot]))
    #print(qs,qs.sum())
    #if qs.sum()==0:
    if 1:
    #if qs.sum()==0 and len(pids[pids==2212])>0:
        invmasses.append(invmass(myparticles))
        missmasses.append(invmass([tot]))
        nprotons.append(len(pids[pids==2212]))
        totq.append(qs.sum())
        #break


    #solo = 2212
    solo = 13
    if qs.sum()==0 and len(pids[pids==solo])==1:
        # B candidates
        bc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        proton = None
        for p in myparticles:
            totx.append(p[1])
            toty.append(p[2])
            totz.append(p[3])
            if p[-1] != solo:
                bc += p
            else:
                proton = p
        bcand.append(invmass([bc]))
        dE = bc[0] - beam[0]/2.0
        bc[0] = beam[0]/2.0
        mes = invmass([bc])
        bcandMES.append(mes)
        bcandDeltaE.append(dE)
        bcandMM.append(invmass([beam-bc-proton]))

        

plt.figure()
plt.subplot(3,3,1)
plt.hist(invmasses,bins=200,range=(0,20))
    
plt.subplot(3,3,2)
plt.hist(missmasses,bins=200,range=(-10,10))

plt.subplot(3,3,3)
plt.hist(nprotons,bins=11,range=(0,10))

plt.subplot(3,3,4)
plt.hist(totq,bins=52,range=(-5,5))

plt.tight_layout()


###################
plt.figure()
plt.subplot(3,3,1)
plt.hist(bcand,bins=200,range=(0,15))
plt.xlabel(r'B-cand mass [GeV/c$^{2}$]',fontsize=10)
    
plt.subplot(3,3,2)
#plt.hist(bcandMES,bins=200,range=(2,7))
plt.hist(bcandMES,bins=200,range=(5,5.3))
plt.xlabel(r'M$_{\rm ES}$ [GeV/c$^{2}$]',fontsize=10)

plt.subplot(3,3,3)
plt.hist(bcandDeltaE,bins=200,range=(-10,10))
plt.xlabel(r'$\Delta$E [GeV]',fontsize=10)

plt.subplot(3,3,4)
#plt.hist(bcandMM,bins=200,range=(2,7))
plt.hist(bcandMM,bins=200,range=(-5,7))
plt.xlabel(r'Missing mass [GeV/c$^{2}$]',fontsize=10)

plt.subplot(3,3,5)
plt.plot(bcandMES,bcandDeltaE,'.',alpha=0.8,markersize=1.0)
plt.xlabel(r'M$_{\rm ES}$ [GeV/c$^{2}$]',fontsize=10)
plt.ylabel(r'$\Delta$E [GeV]',fontsize=10)

plt.subplot(3,3,7)
plt.plot(totx,toty,'.',alpha=0.5,markersize=0.5)
plt.xlabel(r'p$_{x}$ [GeV/c]',fontsize=10)
plt.ylabel(r'p$_{y}$ [GeV/c]',fontsize=10)

plt.subplot(3,3,8)
plt.plot(totx,totz,'.',alpha=0.5,markersize=0.5)
plt.xlabel(r'p$_{x}$ [GeV/c]',fontsize=10)
plt.ylabel(r'p$_{z}$ [GeV/c]',fontsize=10)

plt.subplot(3,3,9)
plt.plot(toty,totz,'.',alpha=0.5,markersize=0.5)
plt.xlabel(r'p$_{y}$ [GeV/c]',fontsize=10)
plt.ylabel(r'p$_{z}$ [GeV/c]',fontsize=10)

plt.tight_layout()

# Momentum

for i,id in enumerate(particle_lunds):
    plt.figure(figsize=(12,4))

    x = allparts[0][id]
    y = allparts[1][id]
    z = allparts[2][id]

    print(id)
    #print(x)

    plt.subplot(1,3,1)
    plt.title(str(id))
    plt.plot(x,y,'.',alpha=0.5,markersize=0.5)
    plt.xlabel(r'p$_{x}$ [GeV/c]',fontsize=10)
    plt.ylabel(r'p$_{y}$ [GeV/c]',fontsize=10)
    plt.xlim(-7,7)
    plt.ylim(-7,7)

    plt.subplot(1,3,2)
    plt.plot(x,z,'.',alpha=0.5,markersize=0.5)
    plt.xlabel(r'p$_{x}$ [GeV/c]',fontsize=10)
    plt.ylabel(r'p$_{z}$ [GeV/c]',fontsize=10)
    plt.xlim(-7,7)
    plt.ylim(-7,7)

    plt.subplot(1,3,3)
    plt.plot(y,z,'.',alpha=0.5,markersize=0.5)
    plt.xlabel(r'p$_{y}$ [GeV/c]',fontsize=10)
    plt.ylabel(r'p$_{z}$ [GeV/c]',fontsize=10)
    plt.xlim(-7,7)
    plt.ylim(-7,7)

    plt.tight_layout()





plt.show()
