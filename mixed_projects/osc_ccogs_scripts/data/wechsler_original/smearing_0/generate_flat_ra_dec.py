#!/usr/bin/env python

################################################################################
# This program will generate intitial conditions for an n-body simulation.
#
# It will randomly generate the mass, initial position and intitial velocity
# for the particles, using an (x,y,z) coordinate system.
# 
################################################################################

import sys
import numpy as np
from optparse import OptionParser


################################################################################
# main
################################################################################
def main():

    nparticles = int(sys.argv[1])

    outfile_name = "flat_default.dat"
    if len(sys.argv)>2:
        outfile_name = sys.argv[2]
        
    outfile = open(outfile_name,'w+')
    ############################################################################
    ############################################################################
    #print "Right_Ascension , Declination"
    # Generate the config file
    #print nparticles

    ra_range = 90.0
    #dec_range = np.pi/4.0
    dec_range = 1.0

    #for i in range(nparticles):
    i=0
    output = "%d\n" % (nparticles)
    ra = 0.0
    dec = 0.0
    deg2arcsec = 3600.0
    for i in xrange(nparticles):
        
        ra = ra_range*np.random.random() 
        output += "%-7.4f " % (deg2arcsec*ra)

        dec = np.arccos(dec_range*np.random.random())
        output += "%-7.4f\n" % (deg2arcsec*(90.0-np.rad2deg(dec)))

    outfile.write(output)
    outfile.close()
    
################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
