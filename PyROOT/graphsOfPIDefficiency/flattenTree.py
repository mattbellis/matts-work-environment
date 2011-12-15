#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText, TH2F
from ROOT import TBranch
from ROOT import gROOT, gStyle

from numpy import *
from array import array


gROOT.Reset()
#
# Check the command line options
#

if len(sys.argv) < 3:
  print "\nUsage: " + sys.argv[0] + " <cand name> <original root file>\n"
  sys.exit(-1)
  
indexstring = sys.argv[1]
filename = sys.argv[2]

oldfile = TFile(filename, "READ")
###################################################
# HARD CODED!!!!!!!!!!!!!!!!!!!!!!
# I've hard coded the naming convention for the output file.
# Old file: xxx.root
# New file: xxx_flat.root
###################################################
newfilename = filename.split('.root')[0] + "_flat.root"
newfile = TFile(newfilename, "recreate")

###################################################
# HARD CODED!!!!!!!!!!!!!!!!!!!!!!
# I've hardcoded the possible ntuple names
###################################################
ntuples = ["ntp1", "ntp2", "ntp3", "ntp4" ]

for ntp in ntuples:
  oldtree = oldfile.Get(ntp)

  if(oldtree):
    print "Copying ntuple: " + ntp

    nentries = oldtree.GetEntries()
    print "Entries in oldtree: " + str(nentries)
    #####################################################
    # Define some variables for placeholders for the floats,
    # ints and the new index.
    #####################################################
    old_index = array('i', [ 0 ] )
    new_index = array('i', [ 0 ] )

    #####################################################
    # Even though the tree is flattened with respect to some 
    # candidate, I would still like to know which candidate it was
    # from the original tree, so I create a new branch to hold
    # this information.
    #####################################################
    cand_index = array('i', [ 0 ] )
    ncand_org =  array('i', [ 0 ] )

    oldval_i = []
    oldval_f = []
    newval_i = []
    newval_f = []
    ########################################
    # HARD CODED!!!!!!!!!!!!!!!!!!!!!!
    # If there are more than 16 candidates or more than 1000 branches
    # to copy, this will have to be changed.
    ########################################
    for i in range(0,1000):
      dumi = array('i')
      for j in range(0,100):
        dumi.append(0)
      oldval_i.append(dumi)

      dumf = array('f')
      for j in range(0,100):
        dumf.append(0)
      oldval_f.append(dumf)

      #oldval_i.append(array( 'i', [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] ) )
      #oldval_f.append(array( 'f', [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] ) )
      newval_i.append(array( 'i', [ 0 ] ) )
      newval_f.append(array( 'f', [ 0 ] ) )

    ######################################################
    # Get the branches associated with our candidate
    ######################################################
    branch_list = oldtree.GetListOfBranches();
    nbranches = branch_list.GetEntries();

    #print "nbranches: " + str(nbranches)

    #######################################################
    # Grab the 'float' branches and the 'int' branches
    #######################################################
    savebranches_f = []
    savebranches_i = []
    ni = 0
    nf = 0
    #######################################################
    # Get 'em
    #######################################################
    for br in range(0, nbranches):
      br_name = branch_list[br].GetName()
      br_title = branch_list[br].GetTitle()
      #print "br_title: " + br_title
      if br_title.find("[") >= 0:
        br_index = br_title.split("[")[1].split("]")[0]
        class_type = br_title.split("/")[1][0]
        # Floats
        if class_type=="F":
          if br_index == "n" + indexstring:
            savebranches_f.append(br_name)
            nf += 1
        # Ints
        elif class_type=="I":
          if br_index == "n" + indexstring:
            savebranches_i.append(br_name)
            ni += 1





    #######################################################
    #Create a new file + a clone of old tree in new file
    #######################################################
    newfile.cd()
    newtree = oldtree.CloneTree(0)

    #######################################################
    # Set the branch addresses for both the new index, cand
    # index and the old branches.
    #######################################################
    newtree.SetBranchAddress("n" + indexstring, new_index)
    oldtree.SetBranchAddress("n" + indexstring, old_index)

    newtree.Branch(indexstring + "cand_index", cand_index, indexstring + "cand_index/I")
    newtree.Branch("n" + indexstring + "_org", cand_index, "n" + indexstring + "_org/I")

    for i in range(0,nf):
      newtree.SetBranchAddress(savebranches_f[i], newval_f[i])
      oldtree.SetBranchAddress(savebranches_f[i], oldval_f[i])


    ###############################################################
    # Loop over the events, then loop over the candidates,
    # then loop over the branches to reset
    ###############################################################
    newcount = 0
    # Events loop
    for i in range(0, nentries):
      if i%1000 == 0:
        sys.stderr.write(str(i) + "\r")

      oldtree.GetEntry(i)
      numcand = old_index[0]
      # Candidates loop
      for j in range(0, numcand):
        ncand_org[0] =  numcand
        new_index[0] = 1
        cand_index[0] = j
        # Float branches loop
        for k in range(0,nf):
          newval_f[k][0] = oldval_f[k][j]
        # Int branches loop
        for k in range(0,ni):
          newval_i[k][0] = oldval_i[k][j]

        newtree.Fill()
        newcount += 1

    print "Entries in new file: " + str(newcount) + "\n"

    newtree.Write()

newfile.Close()
newfile.Write()

##############################################


