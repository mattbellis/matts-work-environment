import numpy as np
import matplotlib.pylab as plt
import sys

import zipfile

import hep_tools_real_analysis as hep

#f = open(sys.argv[1],'r')
f = zipfile.ZipFile(sys.argv[1])
print f.namelist()
f  = f.open("shyft_ultraslim_100_1_SK7_for_analysis.txt")
#f = zipfile.ZipFile(sys.argv[1],'r')
#f = open('temp.txt','r')

print "Reading in the data...."
collisions = hep.get_collisions(f)

print len(collisions)

#count = 0
for count,collision in enumerate(collisions):

    #print collision
    jets,topjets,muons,electrons,met = collision


    #print gen_particles
    #print ak5jets
    #print ca8jets

    #exit()

    print "-------------------------- %d" % (count)
    print "------- jets"
    for p in jets:
        mass,px,py,pz,csv = p
        print "%8.5f %8.5f %12.5f %12.5f %12.5f" % (mass,px,py,pz,csv)
    print "------- top jets"
    for p in topjets:
        mass,px,py,pz,nsub,minmass = p
        print "%8.5f %8.5f %12.5f %12.5f %12.5f %12.5f" % (mass,px,py,pz,nsub,minmass)
    print "------- muons"
    for p in muons:
        mass,px,py,pz = p
        print "%8.5f %8.5f %12.5f %12.5f" % (mass,px,py,pz)
    print "------- electrons"
    for p in electrons:
        mass,px,py,pz = p
        print "%8.5f %8.5f %12.5f %12.5f" % (mass,px,py,pz)
    print "------- met"
    for p in met:
        pt,phi = p
        print "%8.5f %8.5f" % (pt,phi)
