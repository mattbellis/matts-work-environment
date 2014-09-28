#!/usr/bin/env python

###############################################################
# intro3.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro3.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from optparse import OptionParser

import shutil
from shutil import *

doFit = False

#### Command line variables ####
parser = OptionParser()
parser.add_option("-f", "--fit", dest="do_fit", action = "store_true", default = False, help="Run the fit")
parser.add_option("--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("-b", "--num-bkg-events", dest="numbkg", default=600, help="Number of background events in fit")
parser.add_option("-s", "--num-sig-events", dest="numsig", default=60, help="Number of signal events in fit")
parser.add_option("-N", "--num-studies", dest="num_studies", default=10, help="Number of toy studies to run")
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag to determine where the files live.")
parser.add_option("--fixed-num", dest="fixed_num", action="store_true", default=False, \
        help="Use a fixed number of both background and signal.")
parser.add_option("--sig-file", dest="sig_file_name", help="File from which to grab the signal events.")
parser.add_option("--pass", dest="my_pass", default=0, help="From which pass to grab fit ranges")
parser.add_option("--ntp", dest="ntp", default="ntp1", help="Baryon [LambdaC, Lambda0]")
parser.add_option("--baryon", dest="baryon", default="LambdaC", help="Ntuple over which we are running")


(options, args) = parser.parse_args()

#####################################################################
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

rnd = TRandom3()

if options.sig_file_name==None:
    print "Need to pass in a file with the signal events!"
    exit(-1)

# Grab the data ranges
from file_map import *

pass_info = fit_pass_info(options.baryon, options.ntp, int(options.my_pass))

mes_lo = pass_info[4][0]
mes_hi = pass_info[4][1]

deltae_lo = pass_info[5][0]
deltae_hi = pass_info[5][1]

nn_lo = pass_info[6][0]
nn_hi = pass_info[6][1]


################################################
################################################

sig_events = [[], [] , []]
#signal_file = open("textFiles/textLambdaC_ntp1_SP9446_newTMVA_cut2.txt", "r")
signal_file = open(options.sig_file_name, "r")

tot_sig = 0
for line in signal_file:
  sig_events[0].append( float(line.split()[0]) )
  sig_events[1].append( float(line.split()[1]) )
  sig_events[2].append( float(line.split()[2]) )
  tot_sig += 1


print "Total number of signal events available: %d" % (tot_sig)

################################################################################
########################################
# Embed the signal events
########################################
################################################################################

rnd = TRandom3()

for i in range(0, int(options.num_studies)):
    
    fixed_tag = ""
    nev = rnd.Poisson( int(options.numsig) ) # Draw from a Poisson distribution
    if options.fixed_num: # or don't draw from a Poisson
        fixed_tag = "_fixedSig"
        nev = int(options.numsig)

    ############################################################################
    # Make a copy of the file with no signal events and rename it
    ############################################################################
    old_file_name = "%s/mcstudies_bkg%d_sig0%s_%04d.dat" % ( options.tag, int(options.numbkg), fixed_tag, i )
    new_file_name = "%s/mcstudies_bkg%d_embed_sig%d%s_%04d.dat" % ( options.tag, int(options.numbkg), int(options.numsig), fixed_tag, i )

    print old_file_name
    print new_file_name
    shutil.copyfile( old_file_name, new_file_name)

    print old_file_name
    print new_file_name
    toyfile = open(new_file_name, 'a')
    ############################################################################

    # Keep track of which signal events we've already used.
    numbers_chosen = []
    j = 0

    while j < nev:

        k = int( (tot_sig - 1) * rnd.Rndm() )

        # Don't double up on signal events
        if not (k in numbers_chosen):
            
            x = float(sig_events[0][k])
            y = float(sig_events[1][k])
            z = float(sig_events[2][k])

            # If the signal event is within the range, we append it to the file.
            if z>nn_lo and z<nn_hi-0.00001 and x>mes_lo and x<mes_hi and y>deltae_lo and y<deltae_hi:

                output = "%12.9f %12.9f %12.9f\n" %( x, y, z )
                #print output
                toyfile.write(output)
                numbers_chosen.append(k)
                j += 1

    toyfile.close()


