#!/usr/bin/env python

from numpy import array
import ROOT
from ROOT import *

from color_palette import *

import sys
from optparse import OptionParser

import re

import random as rnd

################################################################################
################################################################################

################################################################################
################################################################################
def main(argv):

    parser = OptionParser()
    parser.add_option("-m", "--max", dest="max", default=1e9, 
            help="Max games to read in.")
    parser.add_option("--tag", dest="tag", default=None, 
            help="Tag for output files.")
    parser.add_option("--batch", dest="batch", default=False, 
            action="store_true", help="Run in batch mode.")

    (options, args) = parser.parse_args()

    infile = open(args[0])

    # Style options
    gStyle.SetOptStat(0)
    set_palette("palette",50)

    ############################################################################
    # Declare the canvases
    ############################################################################
    num_can = 1
    can = []
    for i in range(0,num_can):
        name = "can%d" % (i)
        if i<2:
            can.append(TCanvas(name,"",10+10*i,10+10*i,1350,700))
        else:
            can.append(TCanvas(name,"",10+10*i,10+10*i,700,700))
        can[i].SetFillColor(0)
        can[i].Divide(1,1)

    ############################################################################
    # Declare some histograms 
    ############################################################################
    lo = 0.0
    hi = 1.0
    color = 2
    nbins = 100
    h = []
    for i in range(0,5):
        name = "h%d" % (i)
        h.append(TH1F(name,"",nbins, lo, hi))
        h[i].SetFillStyle(1000)
        h[i].SetFillColor(color)
        h[i].SetTitle("")

        h[i].SetNdivisions(8)
        h[i].GetYaxis().SetTitle("# occurances")
        h[i].GetYaxis().SetTitleSize(0.09)
        h[i].GetYaxis().SetTitleFont(42)
        h[i].GetYaxis().SetTitleOffset(0.7)
        h[i].GetYaxis().CenterTitle()

        h[i].GetXaxis().SetTitle("Arbitrary measurements")
        h[i].GetXaxis().SetLabelSize(0.12)
        h[i].GetXaxis().SetTitleSize(0.10)
        h[i].GetXaxis().SetTitleFont(42)
        h[i].GetXaxis().SetTitleOffset(1.0)
        h[i].GetXaxis().CenterTitle()

        h[i].SetMinimum(0)

    h2D = []
    for i in range(0,5):
        name = "h2D%d" % (i)
        h2D.append(TH2F(name,"",nbins, lo, hi, nbins, lo, hi))
        h2D[i].SetFillStyle(1000)
        h2D[i].SetFillColor(color)
        h2D[i].SetTitle("")

        h2D[i].SetNdivisions(8)
        h2D[i].GetYaxis().SetTitleSize(0.09)
        h2D[i].GetYaxis().SetTitleFont(42)
        h2D[i].GetYaxis().SetTitleOffset(0.7)
        h2D[i].GetYaxis().CenterTitle()
        h2D[i].GetYaxis().SetTitle("Visitng team")
        h2D[i].GetXaxis().SetTitle("Home team")
        #h2D[i].GetXaxis().SetLabelSize(0.09)
        h2D[i].GetXaxis().SetTitleSize(0.09)
        h2D[i].GetXaxis().SetTitleFont(42)
        h2D[i].GetXaxis().SetTitleOffset(0.7)
        h2D[i].GetXaxis().CenterTitle()

        h2D[i].SetMinimum(0)

    ############################################################################
    # Set some physics quantitites
    ############################################################################

    recorded_values = [array('d'),array('d'),array('d')]

    for line in infile:
        vals = line.split()
        num_vals = len(vals)
        for i in range(0,num_vals):
            recorded_values[i].append(float(vals[i]))


    npts = len(recorded_values[0])

    gr = TGraph(npts,recorded_values[1],recorded_values[2])
    
    can[0].cd(1)
    gr.Draw("ap*")
    gPad.Update()
            



    '''
    if options.tag != None:
        name = "Plots/sportsplots%s_%d.eps" % (options.tag,i)
        can[0].SaveAs(name)
    '''

    ################################################################################
    ## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
    if not options.batch:
        rep = ''
        while not rep in [ 'q', 'Q' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

################################################################################
################################################################################
if __name__ == '__main__':
    main(sys.argv)



