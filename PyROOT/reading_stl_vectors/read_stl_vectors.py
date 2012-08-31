#!/usr/bin/env python

from ROOT import *
import sys

def readTree():

  gROOT.LoadMacro("load_stl.h+")

  # Create the TChain
  t = TChain("egamma")

  filename = None
  if len(sys.argv)<2:
      print "Must pass in file on command line!"
      exit(-1)
  else:
      filename = sys.argv[1]
      t.Add(filename)
      #t.Print()

  # Turn on only the one branch branch
  t.SetBranchStatus( '*', 0 )
  t.SetBranchStatus( 'trig_L2_emcl_energyInSample', 1 )

  # Event loop
  nev = t.GetEntries()
  for n in xrange (t.GetEntries()):
    t.GetEntry(n)

    print t.trig_L2_emcl_energyInSample
    test_vec = t.trig_L2_emcl_energyInSample
    length = len(t.trig_L2_emcl_energyInSample)
    if length>0:
        length_0 = len(t.trig_L2_emcl_energyInSample[0])
        print "%d %d" % (length, length)



#### Run read function
if __name__ == '__main__':
     readTree()
