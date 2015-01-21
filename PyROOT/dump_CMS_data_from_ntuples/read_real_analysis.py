import numpy as np
import matplotlib.pylab as plt
import sys

import zipfile

#import hep_tools_real_analysis as hep
from siena_cms_tools import get_collisions,pretty_print

#f = open(sys.argv[1],'r')
f = zipfile.ZipFile(sys.argv[1])
print f.namelist()
f  = f.open("shyft_ultraslim_100_1_SK7_for_analysis.txt")
#f = zipfile.ZipFile(sys.argv[1],'r')
#f = open('temp.txt','r')

print "Reading in the data...."
collisions = get_collisions(f)

print len(collisions)

#count = 0
for count,collision in enumerate(collisions):

    #print collision
    jets,topjets,muons,electrons,met = collision

    print "\n-------------------------- %d" % (count)
    pretty_print(collision)

