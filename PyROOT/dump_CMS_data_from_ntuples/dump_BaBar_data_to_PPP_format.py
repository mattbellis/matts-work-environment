import numpy as np
import matplotlib as plt


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

particles = ["mu","e","pi","K","p"]

################################################################################
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
outfilename = "%s.dat" % (sys.argv[1].split('.root')[0])
outfile = open(outfilename,'w')

for i in range(nentries):

    output = "Event: %d\n" % (i)
    #output = ""
    tree.GetEntry(i)

    nvals = 0
    
    ############################################################################
    # Print out the pions
    ############################################################################
    npions = tree.npi
    output += "%d\n" % (npions)
    for j in range(npions):
        e = tree.pienergy[j]
        p3 = tree.pip3[j]
        phi = tree.piphi[j]
        costh = tree.picosth[j]
        px,py,pz = sph2cart(p3,costh,phi)
        lund = tree.piLund[j]
        q = int(lund/np.abs(lund))
        idx = tree.piTrkIdx[j]
        dedx = tree.TRKdedxdch[idx]
        drc = tree.TRKDrcTh[idx]
        beta = 1.0/np.cos(drc)/1.474
        #print "pions"
        ebit,mubit,pibit,Kbit,pbit = tree.eSelectorsMap[j],tree.muSelectorsMap[j],tree.piSelectorsMap[j],tree.KSelectorsMap[j],tree.pSelectorsMap[j]
        eps.SetBits(ebit); mups.SetBits(mubit); pips.SetBits(pibit); Kps.SetBits(Kbit); pps.SetBits(pbit);
        max_particle,max_pid = selectPID(eps,mups,pips,Kps,pps,verbose=False)
        output += "%.4f %.4f %.4f %.4f %.4f %.4f %d\n" % (e,px,py,pz,q,beta,dedx)
        nvals += 5

    ############################################################################
    # Print out the Kaons
    ############################################################################
    nkaons = tree.nK
    output += "%d\n" % (nkaons)
    for j in range(nkaons):
        e = tree.Kenergy[j]
        p3 = tree.Kp3[j]
        phi = tree.Kphi[j]
        costh = tree.Kcosth[j]
        px,py,pz = sph2cart(p3,costh,phi)
        lund = tree.KLund[j]
        q = int(lund/np.abs(lund))
        idx = tree.KTrkIdx[j]
        dedx = tree.TRKdedxdch[idx]
        drc = tree.TRKDrcTh[idx]
        beta = 1.0/np.cos(drc)/1.474
        #print "Kaons"
        ebit,mubit,pibit,Kbit,pbit = tree.eSelectorsMap[j],tree.muSelectorsMap[j],tree.piSelectorsMap[j],tree.KSelectorsMap[j],tree.pSelectorsMap[j]
        eps.SetBits(ebit); mups.SetBits(mubit); pips.SetBits(pibit); Kps.SetBits(Kbit); pps.SetBits(pbit);
        max_particle,max_pid = selectPID(eps,mups,pips,Kps,pps,verbose=False)
        output += "%.4f %.4f %.4f %.4f %.4f %.4f %d\n" % (e,px,py,pz,q,beta,dedx)
        nvals += 5

    ############################################################################
    # Print out the protons
    ############################################################################
    nprotons = tree.np
    output += "%d\n" % (nprotons)
    for j in range(nprotons):
        e = tree.penergy[j]
        p3 = tree.pp3[j]
        phi = tree.pphi[j]
        costh = tree.pcosth[j]
        px,py,pz = sph2cart(p3,costh,phi)
        lund = tree.pLund[j]
        q = int(lund/np.abs(lund))
        idx = tree.pTrkIdx[j]
        dedx = tree.TRKdedxdch[idx]
        drc = tree.TRKDrcTh[idx]
        beta = 1.0/np.cos(drc)/1.474
        #print "protons"
        ebit,mubit,pibit,Kbit,pbit = tree.eSelectorsMap[j],tree.muSelectorsMap[j],tree.piSelectorsMap[j],tree.KSelectorsMap[j],tree.pSelectorsMap[j]
        eps.SetBits(ebit); mups.SetBits(mubit); pips.SetBits(pibit); Kps.SetBits(Kbit); pps.SetBits(pbit);
        max_particle,max_pid = selectPID(eps,mups,pips,Kps,pps,verbose=False)
        output += "%.4f %.4f %.4f %.4f %.4f %.4f %d\n" % (e,px,py,pz,q,beta,dedx)
        nvals += 5

    #'''
    ############################################################################
    # Print out the muons
    ############################################################################
    nmuons = tree.nmu
    output += "%d\n" % (nmuons)
    for j in range(nmuons):
        e = tree.muenergy[j]
        p3 = tree.mup3[j]
        phi = tree.muphi[j]
        costh = tree.mucosth[j]
        px,py,pz = sph2cart(p3,costh,phi)
        lund = tree.muLund[j]
        q = int(lund/np.abs(lund))
        idx = tree.muTrkIdx[j]
        dedx = tree.TRKdedxdch[idx]
        drc = tree.TRKDrcTh[idx]
        beta = 1.0/np.cos(drc)/1.474
        #print "muons"
        ebit,mubit,pibit,Kbit,pbit = tree.eSelectorsMap[j],tree.muSelectorsMap[j],tree.piSelectorsMap[j],tree.KSelectorsMap[j],tree.pSelectorsMap[j]
        eps.SetBits(ebit); mups.SetBits(mubit); pips.SetBits(pibit); Kps.SetBits(Kbit); pps.SetBits(pbit);
        max_particle,max_pid = selectPID(eps,mups,pips,Kps,pps,verbose=False)
        output += "%.4f %.4f %.4f %.4f %.4f %.4f %d\n" % (e,px,py,pz,q,beta,dedx)
        nvals += 5
    #'''

    #'''

    #'''
    ############################################################################
    # Print out the electrons
    ############################################################################
    nelectrons = tree.ne
    output += "%d\n" % (nelectrons)
    for j in range(nelectrons):
        e = tree.eenergy[j]
        p3 = tree.ep3[j]
        phi = tree.ephi[j]
        costh = tree.ecosth[j]
        px,py,pz = sph2cart(p3,costh,phi)
        lund = tree.eLund[j]
        q = int(lund/np.abs(lund))
        idx = tree.eTrkIdx[j]
        dedx = tree.TRKdedxdch[idx]
        drc = tree.TRKDrcTh[idx]
        beta = 1.0/np.cos(drc)/1.474
        #print "electrons"
        ebit,mubit,pibit,Kbit,pbit = tree.eSelectorsMap[j],tree.muSelectorsMap[j],tree.piSelectorsMap[j],tree.KSelectorsMap[j],tree.pSelectorsMap[j]
        eps.SetBits(ebit); mups.SetBits(mubit); pips.SetBits(pibit); Kps.SetBits(Kbit); pps.SetBits(pbit);
        max_particle,max_pid = selectPID(eps,mups,pips,Kps,pps,verbose=False)
        output += "%.4f %.4f %.4f %.4f %.4f %.4f %d\n" % (e,px,py,pz,q,beta,dedx)
        nvals += 5

    #''' 
    ############################################################################
    # Print out the photons
    ############################################################################
    nphotons = tree.ngamma
    output += "%d\n" % (nphotons)
    for j in range(nphotons):
        e = tree.eenergy[j]
        p3 = tree.ep3[j]
        phi = tree.ephi[j]
        costh = tree.ecosth[j]
        px,py,pz = sph2cart(p3,costh,phi)
        output += "%.4f %.4f %.4f %.4f\n" % (e,px,py,pz)
        nvals += 4

    ''' 
    #print "MET!!!!!!!!!!!!!!!!!!!"
    px = tree.MET_px
    py = tree.MET_py
    pt = np.sqrt(px*px + py*py)
    phi = np.arccos(px/pt)
    if py<0:
        phi += 3.14149
    output += "%.4f %.4f\n" % (px,py)
    #print output
    #output += "%d\n" % (1)
    #output += "%.4f %.4f\n" % (pt,phi)
    nvals += 2

    #output = "%d\n%s" % (nvals,output)
    '''

    #print output
    outfile.write(output)

outfile.close()


#zipfilename = "%s.zip" % (sys.argv[1].split('/')[-1].split('.root')[0])
zipfilename = "%s.zip" % (sys.argv[1].split('.root')[0])
zf = zipfile.ZipFile(zipfilename,'w')
zf.write(outfilename,compress_type=zipfile.ZIP_DEFLATED)
zf.close()
#outfile = open(outfilename,'w')
