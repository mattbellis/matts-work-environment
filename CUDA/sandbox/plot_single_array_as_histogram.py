#!/usr/bin/env python

import sys
from ROOT import *

infile_name = sys.argv[1]
infile = open(infile_name)

# Grab the entries from the input file
lo = 0.0
hi = 1.0
bin_entries = []
for line in infile:
    vals = line.split()
    if len(vals)==2:
        lo = float(vals[0])
        hi = float(vals[1])
    else:
        bin_entries = vals

# Assume first and last entry are underflow/overflow
nentries = len(bin_entries)-2

h = TH1F("h","h",nentries,lo,hi)
for i in range(0,nentries+2):
    h.SetBinContent(i,float(bin_entries[i]))

gStyle.SetOptStat(111111)
h.Draw()
gPad.Update()

print "integral: %f" % (h.Integral())

if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

