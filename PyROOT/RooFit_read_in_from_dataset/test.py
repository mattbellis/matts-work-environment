#!/usr/bin/env python
import numpy as np

import sys
from optparse import OptionParser
parser = OptionParser()
(options, args) = parser.parse_args()

import ROOT 

def main():
    mass= ROOT.RooRealVar("mass","mass",0,1)
    aset= ROOT.RooArgSet(mass,"aset")
    data= ROOT.RooDataSet("data","data",aset)

    nentries = 10000
    x = np.random.random(nentries)

    for n in range (nentries):
        mass.setVal(x[n])
        data.add(aset)

    frame= mass.frame()
    data.plotOn(frame)
    frame.Draw()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
  main()
  rep = ''
  while not rep in [ 'q', 'Q' ]:
    rep = input( 'enter "q" to quit: ' )
    if 1 < len(rep):
      rep = rep[0]

