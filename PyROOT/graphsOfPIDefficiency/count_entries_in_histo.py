#!/usr/bin/env python 

import os
import sys

from ROOT import *

max = len(sys.argv)
files = sys.argv[1:max]

print files

for file in files:
    f = TFile(file)
    h = f.Get("hmass0_1_3")
    #print h
    n = int(h.Integral())

    print "%-15s %8d" % (file,n)
