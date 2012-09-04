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
    parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
    parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")

    (options, args) = parser.parse_args()


    infile_names = None
    infiles = None

    if len(args)==2:
        infile_names = [args[0],args[1]]
    else:
        print "Need to pass in two textfiles of scan points."
        exit(-1)

    xpts = [array('f'),array('f'),array('f')]
    ypts = [array('f'),array('f'),array('f')]
    expypts = [array('f'),array('f'),array('f')]
    xpts90 = [array('f'),array('f'),array('f')]
    expypts90 = [array('f'),array('f'),array('f')]

    bf = [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]
    ul = [[0.0],[0.0]]
    sigma = [[0.0],[0.0]]

    for i,name in enumerate(infile_names):
        infile = open(name,"r")

        count = 0
        for line in infile:
            vals = line.split()
            if count==0:
                bf[i][0] = float(vals[1])
                bf[i][1] = float(vals[3])
                bf[i][2] = float(vals[6])
                bf[i][3] = float(vals[7])
            elif count==1:
                ul[i][0] = float(vals[1])
            elif count==2:
                sigma[i][0] = float(vals[1])
            else:
                xpts[i].append(float(vals[0]))
                ypts[i].append(float(vals[2]))

            count += 1


    print xpts
    print ypts

    # Check to see who goes out the least far in the scan.
    npts = len(xpts[0])

    for i in range(0,npts):

        xpts[2].append(xpts[0][i])
        y0 = ypts[0][i]
        y1 = ypts[1][i]
        y2 = y0 + y1
        ypts[2].append(y2)

        expypts[0].append(exp(-ypts[0][i]))
        expypts[1].append(exp(-ypts[1][i]))
        expypts[2].append(exp(-ypts[2][i]))

    step_width = xpts[0][4]-xpts[0][3]

    area_greater_than_0 = [0.0,0.0,0.0]
    area_greater_than_90 = [0.0,0.0,0.0]
    ul_90 = [0.0,0.0,0.0]

    # Calc area above 0
    for i in range(0,3):
        for j in range(0,npts):
            test_num = xpts[i][j]
            print test_num
            #if (test_num)%100!=0.0 and xpts[i][j]>0.0:
            if xpts[i][j]>0.0:
                area_greater_than_0[i] += expypts[i][j]*step_width


    # Calc 90% area above 0 
    npts90 = [0,0,0]
    for i in range(0,3):
        for j in range(0,npts):
            test_num = xpts[i][j]
            print test_num
            #if (test_num)%100!=0.0 and xpts[i][j]>0.0:
            if xpts[i][j]>0.0:
                area_greater_than_90[i] += expypts[i][j]*step_width
                test_val = area_greater_than_90[i]/area_greater_than_0[i]
                xpts90[i].append(xpts[i][j])
                expypts90[i].append(expypts[i][j])
                npts90[i] += 1
                if test_val >= 0.90:
                    ul_90[i] = xpts[i][j]
                    break


    print area_greater_than_0
    print area_greater_than_90
    print ul_90
        
        

    ############################################################################
    # Plot the graphs
    ############################################################################

    graphs = []

    for i in range(0,3):
        graphs.append([])
        for j in range(0,3):
            if i==0:
                graphs[i].append(TGraph(npts,xpts[j],ypts[j]))
            elif i==1:
                graphs[i].append(TGraph(npts,xpts[j],expypts[j]))
            elif i==2:
                graphs[i].append(TGraph(npts90[j],xpts90[j],expypts90[j]))
            
    
    can = []
    for i in range(0,2):
        name = "canvas%d" % (i)
        can.append(TCanvas(name,name,10+10*i,10+10*i,1200,800))
        can[i].SetFillColor(0)
        can[i].Divide(1,3)


    for i in range(0,3):
        for j in range(0,3):
            if i<2:
                can[i].cd(j+1)
                graphs[i][j].SetLineColor(2)
                graphs[i][j].SetLineWidth(2)
                graphs[i][j].Draw("apl")
            else:
                can[1].cd(j+1)
                graphs[i][j].SetLineColor(2)
                graphs[i][j].SetLineWidth(8)
                graphs[i][j].SetFillColor(4)
                graphs[i][j].Draw("b")

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
