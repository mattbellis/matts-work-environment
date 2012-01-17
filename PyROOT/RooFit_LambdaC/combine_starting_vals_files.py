#!/usr/bin/env python  

import sys
from optparse import OptionParser

from file_map import *


parser = OptionParser()

parser.add_option("--ntp", dest="ntp", default=1, help="Baryon [LambdaC, Lambda0]")
parser.add_option("--pass", dest="my_pass", default=0, help="From which pass to grab fit ranges")
parser.add_option("--baryon", dest="baryon", default="LambdaC", help="Ntuple over which we are running")

(options, args) = parser.parse_args()


baryon = options.baryon
ntp = "ntp%d" % (int(options.ntp))
tag = "pass%d" % (int(options.my_pass))

# Open the sig/bkg files with the determined paramters
file_bkg_name = "startingValuesForFits/determinedValues_%s_%s_bkg_%s.txt" % (baryon, ntp, tag)
file_sig_name = "startingValuesForFits/determinedValues_%s_%s_sig_%s.txt" % (baryon, ntp, tag)

file_bkg = open(file_bkg_name, "r")
file_sig = open(file_sig_name, "r")

# Make the file for generating the toy samples
file_gen_name = "startingValuesForFits/values_for_gen_pure_%s_%s_%s.txt" % (baryon, ntp, tag)
file_gen = open(file_gen_name, "w")

# Make the file for general fits
# Certain parameters will be fixed/free
file_fit_name = "startingValuesForFits/values_for_fits_%s_%s_%s.txt" % (baryon, ntp, tag)
file_fit = open(file_fit_name, "w")

inputfiles = [file_sig, file_bkg]

for files in inputfiles:
    for line in files:
        if line.split()[0][0]=='#':
            #print line
            file_gen.write(line)
            file_fit.write(line)
        else:
            vals = line.split()
            # For toy samples
            is_const = 1
            output = "%20s%20.5f%20.5f%20s\n" % (vals[0], float(vals[1]), float(vals[2]), is_const)
            file_gen.write(output)
            # For fits
            is_const = 1
            if vals[0]=='nsig' or \
               vals[0]=='alphaCB_NN' or \
               vals[0]=='argpar' or \
               vals[0]=='meanCB_NN' or \
               vals[0]=='nbkg' or \
               vals[0]=='poly1' or \
               vals[0]=='sigmaCB_NN':
                   is_const = 0
            output = "%20s%20.5f%20.5f%20s\n" % (vals[0], float(vals[1]), float(vals[2]), is_const)
            file_fit.write(output)


file_gen.close()
file_fit.close()

print "New starting files are:"
print "\t%s" % (file_gen_name)
print "\t%s" % (file_fit_name)
                

