#!/usr/bin/env python

from ROOT import *
import sys

def readTree():
  f = TFile("small.root")
  t = f.Get("emc")
  t.Print()

  # Turn on a branch
  t.SetBranchStatus( '*', 0 )
  #t._1 = TLorentzVector()
  #t.SetBranchAddress( 'candP4', t._1 )
  t.SetBranchStatus( 'candP4', 1 )
  t._l  = TLorentzVector()
  t.SetBranchAddress( 'candP4', t._l ) 
  t.GetBranch('candP4').GetAddress()

  # Event loop
  nev = t.GetEntries()
  for n in xrange (nev):
    t.GetEntry(n)

    print t.candP4.E()

#### Run read function
if __name__ == '__main__':
     readTree()
