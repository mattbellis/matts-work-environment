#!/usr/bin/env python

# import some modules
import sys
from optparse import OptionParser
from math import *

################################################################################
################################################################################
myusage = "\nusage: %prog [options] <file1.root> <file2.root> ..."
parser = OptionParser(usage = myusage)
parser.add_option("-n", "--ntuplename", dest="ntuplename", default="ntp", help="Name of the ntuple to grab (ntp1)")
parser.add_option("-r", "--output-rootfile",  dest="rootfilename", help="Root output file name")
parser.add_option("-m", "--max",  dest="max", default=1e9, help="Max number of events over which to run.")
parser.add_option("-p", "--plot-extenstion",  dest="plot_extension", help="Extension to define plot format (i.e. .eps, .png, .ps, etc.)")
parser.add_option("-t", "--tag",  dest="tag", default="default", help="Tag to apply to saved plots")
parser.add_option("-b", "--batch", action="store_true", dest="batch", default=False, help="Run in batch mode and exit")


# Parse the options
(options, args) = parser.parse_args()

if len(args) < 1:
  parser.error("Incorrect number of arguments")
  parser.print_help()


################################################################################
# Import the root libraries
################################################################################
from ROOT import *
################################################################################
# Make a canvas on which to display things.
################################################################################
can = []
for i in range(0, 2):
  name = "can%d" % (i)
  can.append(TCanvas(name, name, 10+10*i, 10+10*i, 500, 500))
  can[i].SetFillColor(0)
  can[i].Divide(2,2)

################################################################################
# Make some histograms
################################################################################
histos = []
for i in range(0, 2):
  histos.append([])
  for j in range(0, 4):
    name = "histos%d_%d" % (i, j)
    histos[i].append(TH1F(name, name, 100, 0, 3))

################################################################################
# Create a chain based on the ntuple names
################################################################################
numntuples = 1
print "ntuplename: %s" % ( options.ntuplename )
t = []
for i in range(0, numntuples):
  t.append(TChain(options.ntuplename))
  for j in args:
    filename = j
    print("Adding file: " + filename)
    t[i].Add(filename)


################################################################################
################################################################################


################################################################################
# Fill the histograms
################################################################################
for i in range(0, numntuples):
  # Disable/enable certain branches to increase the speed
  # In this case, we've enabled all the branches.
  t[i].SetBranchStatus("*",1)

  # event loop
  nentries = t[i].GetEntries()
  print "nentries: %d" % ( nentries )

  # Make sure this is not an empty file.
  if nentries==0:
    break

  if nentries > max:
    nentries = int(max)
    print "nentries over which to run: %d " % (nentries)
  for nev in xrange (nentries):
    if nev % 100 == 0:
      print "Event number", nev

    output = ""
    t[i].GetEntry( nev )

    # Dump out the values.
    num = t[i].ngamma
    for n in range(0, num):
      histos[0][0].Fill(t[i].gammap3[n])
      if n == 0:
        histos[1][0].Fill(t[i].gammap3[n])
   
    num = t[i].np
    for n in range(0, num):
      histos[0][1].Fill(t[i].pp3[n])
      if n == 0:
        histos[1][1].Fill(t[i].pp3[n])
   
################################################################################
# Plot the histograms
################################################################################
for i in range(0, 2):
  for j in range(0, 4):
    can[i].cd(j+1)
    histos[i][j].Draw()
    gPad.Update()


################################################################################
# Save the plots 
################################################################################
if options.plot_extension != None:
  for i in range(0, 2):
    name = "%s_%d.%s" % ( tag, i, options.plot_extension )
    can[i].SaveAs(name)

############################
# Save the histos
############################
if options.rootfilename!=None:
  rfile=TFile(options.rootfilename,"recreate")
  print "Saving to file %s" % (options.rootfilename)
  for i in range(0, 2):
    for j in range(0, 4):
      histos[i][j][k].Write()
  rfile.Write()
  rfile.Close()
  print "Wrote and closed %s" % (options.rootfilename)


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]





