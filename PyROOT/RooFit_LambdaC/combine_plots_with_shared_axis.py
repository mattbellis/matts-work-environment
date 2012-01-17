#!/usr/bin/env python

################################################################################
import sys
from ROOT import *
from array import *

from my_roofit_utilities import *

from optparse import OptionParser
from math import *

################################################
def main():

    #### Command line variables ####
    doFit = False

    parser = OptionParser()
    parser.add_option("-c", "--can", dest="can", default=4, help="Which canvas to grab")
    parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
    parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")

    (options, args) = parser.parse_args()

    xyz_string = "y,z"
    if options.can=="4":
        xyz_string = "y,z"
    elif options.can=="5":
        xyz_string = "x,z"
    elif options.can=="6":
        xyz_string = "x,y"

    infile_names = None
    infiles = []

    if len(args)==2:
        infile_names = [args[0],args[1]]
    else:
        print "Need to pass in two rootfiles."
        exit(-1)

    gROOT.Reset()
    # Some global style settings
    #gStyle.SetOptStat(11)
    #gStyle.SetPadRightMargin(0.05)
    gStyle.SetPadBottomMargin(0.20)
    gStyle.SetFrameFillColor(0)

    #gStyle.SetFillColor(0)
    #gStyle.SetPadLeftMargin(0.18)
    #gStyle.SetTitleYOffset(2.00)


    my_cans = []
    frames = []
    rhist = []
    rcurve0 = []
    rcurve1 = []
    lines = []
    for i,name in enumerate(infile_names):
        infiles.append(TFile(name))
        infiles[i].ls()
        which_can = "can%s_1" % (options.can)
        my_cans.append(infiles[i].Get(which_can))
        name = "can%d" % (i)
        my_cans[i].SetName(name)
        my_cans[i].ls()
        rhist.append(my_cans[i].FindObject("h_dataset_0"))
        name = "total_Int[%s]_Norm[x,y,z]_Comp[total]_Range[FULL]_NormRange[FULL]" % (xyz_string)
        rcurve0.append(my_cans[i].FindObject(name))
        name = "total_Int[%s]_Norm[x,y,z]_Comp[bkg_pdf]_Range[FULL]_NormRange[FULL]" % (xyz_string)
        rcurve1.append(my_cans[i].FindObject(name))
        tlist = my_cans[i].GetListOfPrimitives()
        name = None

        found = False
        for t in tlist:
            name = t.GetName()
            if name.find("frame")>=0:
                found = True
                break
        if found:
            frames.append(my_cans[i].FindObject(name))

        found = False
        for t in tlist:
            name = t.GetName()
            if name.find("TLine")>=0:
                found = True
                break
        if found:
            lines.append(my_cans[i].FindObject(name))

        print frames[i]


        #frames.append(my_cans[i].Get

    ############################################################################
    # Plot the graphs
    ############################################################################

    can = []
    pads = []
    for i in range(0,1):
        name = "canvas%d" % (i)
        can.append(TCanvas(name,name,10+10*i,10+10*i,1200,500))
        can[i].SetFillColor(0)
        #can[i].Divide(2, 1, 0.0001,0.0001, 0)
        #can[i].Divide(2, 1, 0,0, 0)
        can[i].cd()
        pads.append([])
        name = "pad%d_%d" % (i,0)
        pads[i].append(TPad(name,name,0.0,0.0,0.5,1.0))
        can[i].cd()
        name = "pad%d_%d" % (i,1)
        pads[i].append(TPad(name,name,0.5,0.0,1.0,1.0))


    for i in range(0,2):
        can[0].cd()
        pads[0][i].Draw()
        pads[0][i].SetFillColor(0)
        pads[0][i].cd()
        if i==1:
            pads[0][i].SetLeftMargin(0.0)
            pads[0][i].SetRightMargin(0.18)
        elif i==0:
            pads[0][i].SetLeftMargin(0.18)
            pads[0][i].SetRightMargin(0.0)

        frames[i].GetXaxis().SetNdivisions(3,5,0,True)

        frames[i].SetMinimum(0)
        frames[i].SetFillColor(0)
        frames[i].Draw()
        rhist[i].Draw("e")
        rcurve0[i].Draw()
        rcurve1[i].Draw()

        if len(lines)>i:
            print "DRAWING"
            lines[i].Draw()

        gPad.Update()



    ############################################################################
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]



################################################################################
################################################
if __name__ == "__main__":
  main()

################################################################################
