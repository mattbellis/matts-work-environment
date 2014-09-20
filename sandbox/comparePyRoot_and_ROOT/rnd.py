#!/usr/bin/env python

import sys
from ROOT import * 

max = int(sys.argv[1])

h = TH1F("h", "h", 100, 0, 1.0)
rnd = TRandom3()

for i in range(0,max):
  dum = rnd.Rndm()
  h.Fill(dum)
