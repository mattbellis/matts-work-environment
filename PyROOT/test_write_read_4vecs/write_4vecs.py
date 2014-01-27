#!/usr/bin/env python

from ROOT import *
import sys
from array import array 
### Fill a root tree
def fillTree():

  p4 = TLorentzVector()
  test_float = array('f', [0.0])
  
  f = TFile( 'test.root', 'RECREATE' )
  tree = TTree( 'T', 'My tree' )

  tree.Branch( "my4vec", "TLorentzVector", p4 )
  tree.Branch( "tf", test_float, "tf/F", 32000)
  
  ######################################################
  # Fill the tree.
  ######################################################
  count = 0
  for i in range(0,10):
    test_float[0] = -999.0
    p4.SetXYZM(1.0, 1.0, 1.0, 2.0)
    
    tree.Fill()

  tree.Write()

#### Run fill function
if __name__ == '__main__':
   fillTree()
