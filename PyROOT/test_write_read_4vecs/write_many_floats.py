#!/usr/bin/env python

from ROOT import *
import sys
from array import array 
### Fill a root tree
def fillTree():

    #test_float = array('f', [0.0])
    test_float = []

    for i in range(0,10):
        test_float.append(array('f', [0.0]))

  
    f = TFile( 'test.root', 'RECREATE' )
    tree = TTree( 'T', 'My tree' )

    for i in range(0,10):
        name = "float_%d" % (i)
        type = "float_%d/F" % (i)
        tree.Branch(name, test_float[i], type, 32000)
  
  ######################################################
  # Fill the tree.
  ######################################################
    count = 0
    for n in range(0,10):
        for i in range(0,10):
            test_float[i][0] = i;
    
        tree.Fill()

    tree.Write()

#### Run fill function
if __name__ == '__main__':
    fillTree()
