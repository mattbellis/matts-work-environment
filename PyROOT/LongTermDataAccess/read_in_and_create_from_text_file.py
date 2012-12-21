#!/usr/bin/env python
# example of macro to read data from an ascii file and
# create a root file with a Tree.
#
# NOTE: comparing the results of this macro with those of staff.C, you'll
# notice that the resultant file is a couple of bytes smaller, because the
# code below strips all white-spaces, whereas the .C version does not.

import re, array
from array import array
from ROOT import *
from datetime import *

import sys


### function to read in data from ASCII file and fill the ROOT tree
def fillTree(datafilename, readmefilename):

  datafile   = open(datafilename)
  readmefile = open(readmefilename)

  branch_names = []
  branch_types = []
  
  ##############################################
  # Read in the entry information
  ##############################################
  entry_vals = []
  for line in readmefile:
    vals = line.split("'")
    if len(vals) > 1:
      temp = []
      branch_names.append(vals[1])
      branch_types.append(vals[3])
      for i in range( 1, len(vals), 2):
        print vals[i]
        temp.append(vals[i]) 
      entry_vals.append(temp)

  ######################################################
  # Let's create the tree.
  ######################################################

  xf = [] # Floats
  xi = [] # Ints

  nbranches = len( branch_names )

  for n in range(0, nbranches ):
    xf.append(array('f', [0.]) )
    xi.append(array('i', [0]) )

  print xf
  print xi

  f = TFile( 'test_exercise.root', 'RECREATE' )
  tree = TTree( 'T', 'Here\'s some data to analyze' )

  f_or_i = [] # 1 if a float, 0 if an int
  numi = 0
  numf = 0
  for n in range(0, nbranches ):
    #print "THIS: %s" % ( branch_types[n] )
    if branch_types[n] == 'F':
      #print "%s %s %s" % ( branch_names[n], xf[numf], branch_names[n]+"/"+branch_types[n] )
      tree.Branch( branch_names[n], xf[numf], branch_names[n]+"/"+branch_types[n], 32000 )
      f_or_i.append(1)
      numf += 1
    elif branch_types[n] == 'I':
      tree.Branch( branch_names[n], xi[numi], branch_names[n]+"/"+branch_types[n], 32000 )
      f_or_i.append(0)
      numi += 1
  
  ######################################################
  # Fill some header info about the code used to create
  # this file.
  ######################################################
  today = datetime.now().ctime()
  list = TList()
  s_gen_info = []
  s_gen_info.append(TObjString("Date of creation: " + str(today)))
  s_gen_info.append(TObjString("Code: " + str(sys.argv[0])))
  s_gen_info.append(TObjString("Software version: V0.0"))
  s_gen_info.append(TObjString("More descriptions about how this file was created or how it is intended to be used."))

  for s in s_gen_info:
    list.Add(s)
  list.SetName("Information about how this file was generated")

  tree.GetUserInfo().Add(list)

  ######################################################
  # Fill some header info about the data that will be
  # stored in this file.
  ######################################################

  names = []
  name  = "Name" ; names.append( name )
  name  = "Float or Integer" ; names.append( name )
  name  = "Units" ; names.append( name )
  name  = "Latex units" ; names.append( name )
  name  = "Root units" ; names.append( name )
  name  = "Description" ; names.append( name )

  name_list = TList()
  for n in names: 
    temp_string = TObjString( n )
    name_list.Add(temp_string)

  name_list.SetName("Description of entries in header")
  tree.GetUserInfo().Add( name_list )

  ##############################################
  # Individual entry values
  ##############################################
  lists = []
  for e in entry_vals:
    list = TList()

    s1 = []

    for v in e:
      temp_string = TObjString( v )
      s1.append(temp_string)

    for s in s1:
      s.Print("v")
      #tree.GetUserInfo().Add(s)
      list.Add(s)

    name = "Information about %s" % ( str(s1[0]) )
    list.SetName(name)
    lists.append(list)

  for l in lists:
    tree.GetUserInfo().Add(l)



  ######################################################
  # Fill the tree.
  ######################################################
  count = 0
  for line in datafile:
    if count%100==0:
      print count
    count += 1
    #print line
    vals = line.split()
    nf = 0
    ni = 0
    for i,v in enumerate(vals):
      #print i
      if f_or_i[i] == 1:
        xf[nf][0] = float(v)
        #print xf[nf]
        nf += 1
      elif f_or_i[i] == 0:
        xi[ni][0] = int(v)
        #print xi[ni]
        ni += 1

    #print xf
    #print xi
    tree.Fill()

  #for i in range(0,10):
  #xf[0] = float(i)
  #tree.Fill()

  #tree.Print()
  tree.Write()

#### run fill function if invoked on CLI
if __name__ == '__main__':
   fillTree(sys.argv[1], sys.argv[2])
