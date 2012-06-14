#!/usr/bin/env python

from ROOT import *
import sys

def readTree():
  # Create the TChain
  t = TChain("T")
  t.Add("test.root")
  t.Print()

  # Turn on a branch
  #t.SetBranchStatus( '*', 0 )
  t.SetBranchStatus( 'my4vec', 1 )
  t.SetBranchStatus( 'tf', 1 )

  # Event loop
  nev = t.GetEntries()
  for n in xrange (t.GetEntries()):
    t.GetEntry(n)

    print t.tf
    print t.my4vec.E()

#### Run read function
if __name__ == '__main__':
     readTree()
