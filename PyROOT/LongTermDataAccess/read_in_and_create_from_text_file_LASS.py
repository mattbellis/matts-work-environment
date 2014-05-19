#!/usr/bin/env python
# example of macro to read data from an ascii file and
# create a root file with a Tree.
#
# NOTE: comparing the results of this macro with those of staff.C, you'll
# notice that the resultant file is a couple of bytes smaller, because the
# code below strips all white-spaces, whereas the .C version does not.

import re, array
import numpy
from numpy import zeros
from array import array
from ROOT import *
from datetime import *

import sys

################################################################################
################################################################################
xf = [] # Floats
xi = [] # Ints

xaf = [] # Floats array
xai = [] # Ints array

#nbranches = len( branch_names )
nbranches = 100
num_tracks   = 4
num_beams    = 4
num_vertices = 4

for n in range(0, nbranches ):
  xf.append(array('f', [0.]) )
  xi.append(array('i', [0]) )

for n in range(0, nbranches ):
  #xaf.append( array('f', range(10)) )
  ##xaf.append( zeros(10))
  xaf.append( array('f',32*[0.]) )
  xai.append( array('i',32*[0]) )
  #xaf.append( None )
  #xai.append( None )

################################################################################
################################################################################
run_num = -999
# Global variable
f = TFile( 'test.root', 'RECREATE' )
tree = TTree( 'T', 'Here\'s some data to analyze' )
#tree.Branch( branch_names[n], xf[numf], branch_names[n]+"/"+branch_types[n], 32000 )
tree.Branch( 'run_num', xi[0], 'run_num/I' )
tree.Branch( 'event_num', xi[1], 'event_num/I' )
tree.Branch( 'hw_trigger', xi[2], 'hw_trigger/I' )
tree.Branch( 'sw_trigger', xi[3], 'sw_trigger/I' )

tree.Branch( 'num_beams', xi[4], 'num_beams/I' )
tree.Branch( 'num_vertices', xi[5], 'num_vertices/I' )
tree.Branch( 'num_cands', xi[6], 'num_cands/I' )

tree.Branch( 'tvt_map', xai[0], 'tvt_map[num_vertices]/I' )

# Vertices
tree.Branch( 'vertex_tv_map', xai[1], 'tv_map[num_vertices]/I' )

tree.Branch( 'vertex_X', xaf[0], 'vertex_X[num_vertices]/F' )
tree.Branch( 'vertex_Y', xaf[1], 'vertex_Y[num_vertices]/F' )
tree.Branch( 'vertex_Z', xaf[2], 'vertex_Z[num_vertices]/F' )
tree.Branch( 'vertex_chi2', xaf[9], 'vertex_chi2[num_vertices]/F' )

# Beam
tree.Branch( 'beam_tv_map', xai[2], 'beam_tv_map[num_beams]/I' )
tree.Branch( 'beam_charge', xai[3], 'beam_charge[num_beams]/I' )

tree.Branch( 'beam_X', xaf[3], 'beam_X[num_beams]/F' )
tree.Branch( 'beam_Y', xaf[4], 'beam_Y[num_beams]/F' )
tree.Branch( 'beam_Z', xaf[5], 'beam_Z[num_beams]/F' )

# cands
tree.Branch( 'cand_tv_map', xai[4], 'cand_tv_map[num_cands]/I' )
tree.Branch( 'cand_id', xai[5], 'cand_id[num_cands]/I' )

tree.Branch( 'cand_X', xaf[6], 'cand_X[num_cands]/F' )
tree.Branch( 'cand_Y', xaf[7], 'cand_Y[num_cands]/F' )
tree.Branch( 'cand_Z', xaf[8], 'cand_Z[num_cands]/F' )

################################################################################
# Open the XML output file.
outxmlfile = open('test.xml', 'w+')
outxmlfile.write("<?xml version='1.0'?>\n")
outxmlfile.write("<file>\n")
################################################################################


################################################################################
################################################################################
### function to read in data from ASCII file in LASS format
def parse_event_block(event_block):
  run_num = -999
  event_num = -999
  hw_trigger = -999
  sw_trigger = -999
  topology = -999
  num_vertices = 0
  num_cands = 0
  num_ks_cand = 0

  vertices = [[], [], [], []] # topology and is primary (0) or secondary (1) x, y, z
  cands = [[], [], [], [], []] # topology and is primary(0) or secondary (1), ID, px, py, pz
  beam = [[], [], [], [], []] # topology and is primary(0) or secondary, charge, px, py, pz
  vertexchi2 = []

  ntrks_per_vertex = [[], [], []]
  num_vertices = 0 # Total over all topolgies

  #print '-------------'
  lines = []
  if event_block != '':
    lines = event_block.split('\n')
    #print lines
  else:
    return

  getchi2 = False
  for i,l in enumerate(lines):
    #print l.split()
    if l.find('new EVENT')>-1:
      vals = l.split()
      #print vals
      run_num = int(vals[8].strip())
      event_num = int(vals[9].strip())
      hw_trigger = int(vals[10].strip(), 16)
      sw_trigger = int(vals[11].strip(), 16)

    elif l.find('Topology')>-1:
      vals = l.split()
      #print vals
      topology = int(vals[2].strip())

    # Vertex
    elif l.find('vtx')>-1:
      vals = l.split()
      #print vals
      vertex_type = 0
      if vals[0].strip() == "Primary":
        vertex_type = 0
      elif vals[0].strip() == "Secondary":
        vertex_type = 1

      ntrks_per_vertex[0].append(topology)
      ntrks_per_vertex[1].append(vertex_type)
      ntrks_per_vertex[2].append(0)
      num_vertices += 1

      #print "%d" % (1000*topology + 100*vertex_type)
      vertices[0].append(1000*topology + 100*vertex_type)
      vertices[1].append(float(vals[6].strip()))
      vertices[2].append(float(vals[7].strip()))
      vertices[3].append(float(vals[8].strip()))
      if len(vals) >= 10:
        vertexchi2.append(float(vals[9].strip()))
        getchi2 = False
      else:
        getchi2 = True

    # Vertex chi2 on next line
    elif getchi2:
      vals = l.split()
      #print vals
      vertexchi2.append(float(vals[0].strip()))
      getchi2 = False

    # Beam
    elif l.find('Beam')>-1:
      vals = l.split()
      #print vals
      beam[0].append(1000*topology + 100+vertex_type)
      beam[1].append(int(vals[6].strip()))
      beam[2].append(float(vals[7].strip()))
      beam[3].append(float(vals[8].strip()))
      beam[4].append(float(vals[9].strip()))

    # cands
    #'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
    #elif len(l.split())==4 and re.search('[\-]?\d+\s+[\-]?(\d+\.\d+)\s+[\-]?(\d+\.\d+)\s+[\-]?(\d+\.\d+)', l)!=None: # Reading in the cands
    elif len(l.split())==4 and re.search('[\-]?\d+\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+)+', l)!=None: # Reading in the cands
      ntrks_per_vertex[2][num_vertices-1] += 1
      vals = l.split()
      #print vals
      cands[0].append(1000*topology + 100*vertex_type)
      cands[1].append(int(vals[0].strip()))
      cands[2].append(float(vals[1].strip()))
      cands[3].append(float(vals[2].strip()))
      cands[4].append(float(vals[3].strip()))


  '''
  print "run num: %d" % (run_num)
  print "event num: %d" % (event_num)
  print "hw trigger: %d" % (hw_trigger)
  print "sw trigger: %d" % (sw_trigger)
  print "topology: %d" % (topology)
  print "beam: "
  for i,b in enumerate(beam[0]):
    print "\t%d %3d %14.10f %14.10f %14.10f" % (b, beam[1][i], beam[2][i], beam[3][i], beam[4][i])
  print "vertices: "
  for i,v in enumerate(vertices[0]):
    print "\t%d %14.10f %14.10f %14.10f" % (v, vertices[1][i], vertices[2][i], vertices[3][i])
    print "\t%14.10f" % ( vertexchi2[i] )
  print "ntrks per vertex:"
  for i,n in enumerate(ntrks_per_vertex[0]):
    print "\t%d %3d %d" % (n, ntrks_per_vertex[1][i], ntrks_per_vertex[2][i])
  print "cands: "
  for i,t in enumerate(cands[0]):
    print "\t%d %3d %14.10f %14.10f %14.10f" % (t, cands[1][i], cands[2][i], cands[3][i], cands[4][i])
  '''

  #'''
  output = "<event>\n"
  output += "<header "
  output += "run_num='%d' " % (run_num)
  output += "event_num='%d' " % (event_num)
  output += "hw_trigger='%d' " % (hw_trigger)
  output += "sw_trigger='%d' />\n" % (sw_trigger)
  output += "<beams>\n"
  for i,b in enumerate(beam[0]):
    output += "  <beam beam_tv_map='%d' beam_charge='%d' beam_X='%.8f' beam_Y='%.8f' beam_Z='%.8f' />\n" % (b, beam[1][i], beam[2][i], beam[3][i], beam[4][i])
  output += "</beams>\n"
  output += "<vertices>\n"
  for i,v in enumerate(vertices[0]):
    output += "  <vertex vertex_tv_map='%d' vertex_chi2='%.8f' vertex_X='%.8f' vertex_Y='%.8f' vertex_Z='%.8f' />\n" % (v, vertexchi2[i], vertices[1][i], vertices[2][i], vertices[3][i])
  output += "</vertices>\n"
  output += "<vertex_map_info>\n"
  for i,n in enumerate(ntrks_per_vertex[0]):
    output += "  <v_map topology='%d' vertex_type='%d' num_cands_with_vertex='%d' />" % (n, ntrks_per_vertex[1][i], ntrks_per_vertex[2][i])
  output += "</vertex_map_info>\n"
  output += "<cands>\n"
  for i,t in enumerate(cands[0]):
    output += "  <cand cand_info='%d' cand_id='%d' cand_X='%.8f' cand_Y='%.8f' cand_Z='%.8f' />\n" % (t, cands[1][i], cands[2][i], cands[3][i], cands[4][i])
  output += "</cands>\n"
  output += "</event>\n\n"

  outxmlfile.write(output)

  # Fill the tree
  xi[0][0] = run_num
  xi[1][0] = event_num
  xi[2][0] = hw_trigger
  xi[3][0] = sw_trigger

  xi[4][0] = num_beams = len(beam[0])
  xi[5][0] = num_vertices = len(vertices[0])
  xi[6][0] = num_cands = len(cands[0])

  #xaf.append( array('f', [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0. ] ) )
  #xai.append( array('i', [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0] ) )

  #print xaf[0]
  ##############################################################################
  # Fill the vertices
  ##############################################################################
  for i in range(0, num_vertices):
    # Make a mapping: 1000*topology + 100*vertex_type + num_cands_per_vertex
    # tvt_map: Topology Vertex numcands
    tv_map = 1000*ntrks_per_vertex[0][i] + 100*ntrks_per_vertex[1][i] 
    tvt_map = tv_map + ntrks_per_vertex[2][i] 
    xai[0][i] = tvt_map

    xai[1][i] = vertices[0][i]

    xaf[0][i] = vertices[1][i]
    xaf[1][i] = vertices[2][i]
    xaf[2][i] = vertices[3][i]
    xaf[9][i] = vertexchi2[i]

  ##############################################################################
  # Fill the beams
  ##############################################################################
  for i in range(0, num_beams):
    xaf[3][i] = beam[2][i]
    xaf[4][i] = beam[3][i]
    xaf[5][i] = beam[4][i]

    xai[2][i] = beam[0][i]
    xai[3][i] = beam[1][i] # charge


  ##############################################################################
  # Fill the cands
  ##############################################################################
  for i in range(0, num_cands):
    xaf[6][i] = cands[2][i]
    xaf[7][i] = cands[3][i]
    xaf[8][i] = cands[4][i]

    xai[4][i] = cands[0][i]
    xai[5][i] = cands[1][i] # ID


  ##############################################################################
  # Fill the tree with everything we added above.
  ##############################################################################
  tree.Fill()

  return


    

################################################################################
################################################################################
### function to open file and pass info to read_event
def read_input_text_file(datafilename, readmefilename):

  num_events = 0

  # Make the header for the XML file
  fill_header_info_XML(readmefilename)

  # Open the input file
  datafile = open(datafilename)

  # Prepare XML file for the events
  outxmlfile.write("<events>\n")

  # Empty event block to pass to other function
  event_block = ''

  for line in datafile:

    """
    if num_events >= 2000:
      break
    """


    # Look for beginning of event
    #print line
    if line.find('new EVENT')>-1:
      parse_event_block(event_block)
      event_block = ''
      event_block += line

      # Keep a counter
      if num_events%1000==0:
        print num_events
      num_events += 1

    else:
      event_block += line

  # Make sure to grab the last event
  parse_event_block(event_block)

  # Fill the header info
  fill_header_info(readmefilename)

  # Write the tree to the file.
  tree.Write()
  f.Close()
  outxmlfile.write("\n</events>\n")
  outxmlfile.write("</file>\n")
  outxmlfile.close()

################################################################################
################################################################################
### function to read in data from ASCII file and fill the ROOT tree
def fill_header_info_XML(readmefilename):

  readmefile = open(readmefilename)

  branch_names = []
  #branch_types = []
  
  ##############################################
  # Read in the entry information
  ##############################################
  entry_vals = []
  for line in readmefile:
    vals = line.split("'")
    if len(vals) > 1:
      temp = []
      branch_names.append(vals[1])
      #branch_types.append(vals[3])
      for i in range( 1, len(vals), 2):
        print vals[i]
        temp.append(vals[i]) 
      entry_vals.append(temp)

  ######################################################
  # Let's read the tree.
  ######################################################

  branch_list = tree.GetListOfBranches()
  nbranches = branch_list.GetEntries()

  print "nbranches: %d " % (nbranches)

  for b in branch_list:
    print b.GetName()

  
  ######################################################
  # Fill some header info about the code used to create
  # this file.
  ######################################################
  today = datetime.now().ctime()
  output = ""
  output += "<file_header>\n"
  output += "\t<file_info name='Date of creation: %s' />\n" % (str(today))
  output += "\t<file_info name='Code: %s' />\n" % (str(sys.argv[0]))
  output += "\t<file_info name='Software version: V0.0' />\n"
  output += "\t<file_info name='More descriptions about how this file was created or how it is intended to be used.' />\n"
  output += "</file_header>\n\n"

  ######################################################
  # Fill some header info about the data that will be
  # stored in this file.
  ######################################################

  output += "<entry_information>\n"
  names = []
  name  = "Name" ; names.append( name )
  name  = "Units" ; names.append( name )
  name  = "Short_description" ; names.append( name )
  name  = "Long_description" ; names.append( name )

  ##############################################
  # Individual entry values
  ##############################################
  for e in entry_vals:
    s1 = []

    output += "\t<entry name='%s'>\n" % (e[0])
    for i,v in enumerate(e):
      output += "\t\t<%s>%s</%s>\n" % (names[i], v, names[i])
      print v
    output += "\t</entry>\n" 

  output += "</entry_information>\n"
  print output
  outxmlfile.write(output)

################################################################################
################################################################################
### function to read in data from ASCII file and fill the ROOT tree
def fill_header_info(readmefilename):

  readmefile = open(readmefilename)

  branch_names = []
  #branch_types = []
  
  ##############################################
  # Read in the entry information
  ##############################################
  entry_vals = []
  for line in readmefile:
    vals = line.split("'")
    if len(vals) > 1:
      temp = []
      branch_names.append(vals[1])
      #branch_types.append(vals[3])
      for i in range( 1, len(vals), 2):
        print vals[i]
        temp.append(vals[i]) 
      entry_vals.append(temp)

  ######################################################
  # Let's read the tree.
  ######################################################

  branch_list = tree.GetListOfBranches()
  nbranches = branch_list.GetEntries()

  print "nbranches: %d " % (nbranches)

  for b in branch_list:
    print b.GetName()

  
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
  name  = "Units" ; names.append( name )
  name  = "Short description" ; names.append( name )
  name  = "Long description" ; names.append( name )

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

    #name = "Information about %s" % ( str(s1[0]) )
    name = "%s" % ( str(s1[0]) )
    list.SetName(name)
    lists.append(list)

  for l in lists:
    tree.GetUserInfo().Add(l)




################################################################################
################################################################################
#### run fill function if invoked on CLI
if __name__ == '__main__':
  #fillTree(sys.argv[1], sys.argv[2])
  read_input_text_file(sys.argv[1], sys.argv[2])
