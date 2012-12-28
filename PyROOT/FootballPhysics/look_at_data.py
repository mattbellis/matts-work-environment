#!/usr/bin/env python

import ROOT
from ROOT import *

from color_palette import *

import sys
from optparse import OptionParser

import re

import random as rnd

################################################################################
################################################################################
def main(argv):

    parser = OptionParser()
    parser.add_option("-m", "--max", dest="max", default=1e9, 
            help="Max games to read in.")
    parser.add_option("--lo", dest="lo", default=0, help="Lo edge of histogram")
    parser.add_option("--hi", dest="hi", default=50, help="Hi edge of histogram")
    parser.add_option("--nbins", dest="nbins", default=None, help="Number of bins to use")
    parser.add_option("--no-X-title", dest="no_x_title", default=False, 
            action="store_true", help="Don't plot the X axis title")
    parser.add_option("--tag", dest="tag", default=None, 
            help="Tag for output files.")
    parser.add_option("--batch", dest="batch", default=False, 
            action="store_true", help="Run in batch mode.")


    (options, args) = parser.parse_args()

    #void lookAtData(char *filename, int max=100000, int LO=0, int HI=50, bool plotXtitle=true, char *tag="")
    filename = args[0]
    IN = open(filename)

    #print filename 

    gStyle.SetOptStat(0)
    set_palette("palette",50)

    color = 32
    nbins = 10
    if options.nbins == None:
        nbins = int(options.hi)-int(options.lo)+1
    else:
        nbins = int(options.nbins)
    lo = float(options.lo)-0.5
    hi = float(options.hi)+0.5

    can = []
    for i in range(0,3):
        name = "can%d" % (i)
        if i<2:
            can.append(TCanvas(name,"",10+10*i,10+10*i,1350,700))
        else:
            can.append(TCanvas(name,"",10+10*i,10+10*i,700,700))
        can[i].SetFillColor(0)
        can[i].Divide(1,1)

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
        if options.no_x_title:
            h[i].GetXaxis().SetTitle("Arbitrary measurements")
        else:
            h[i].GetXaxis().SetTitle("Points scored by a team")
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

    i=0
    count=0

    for line in IN:

        if count >= float(options.max):
            break

        vals = line.split(',')
        #print vals
    
        win = None
        lose = None
        #print re.search('\d+',vals[8])
        if re.search('\d+',vals[8]):
            win = float(vals[8])
        if re.search('\d+',vals[9]):
            lose = float(vals[9])

        if win and lose:
            #print "%d %d" % (win, lose)
            h[0].Fill(win)
            h[0].Fill(lose)
            h[1].Fill(win)
            h[1].Fill(lose)

            test = vals[6]
            if vals[6] == '@':
                h2D[0].Fill(lose,win)
            else:
                h2D[0].Fill(win,lose)
            count += 1

    can[0].cd(1)
    gPad.SetLeftMargin(0.18)
    gPad.SetBottomMargin(0.24)
    h[0].Draw("")
    gPad.Update()

    can[1].cd(1)
    gPad.SetLeftMargin(0.18)
    gPad.SetBottomMargin(0.24)
    h[1].SetMaximum(250)
    h[1].Draw("")
    gPad.Update()

    can[2].cd(1)
    gPad.SetLeftMargin(0.18)
    gPad.SetBottomMargin(0.18)
    h2D[0].Draw("colz")
    line = TLine(0.0,0.0,hi,hi)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw()
    gPad.Update()

    if options.tag != None:
        name = "plots/sportsplots%s_%d.eps" % (options.tag,i)
        can[0].SaveAs(name)
        name = "plots/sportsplots%s_scaled_%d.eps" % (options.tag,i)
        can[1].SaveAs(name)
        name = "plots/sportsplots%s_2D_%d.eps" % (options.tag,i)
        can[2].SaveAs(name)

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



