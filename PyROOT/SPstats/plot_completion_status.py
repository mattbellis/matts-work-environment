#!/usr/bin/env python

# Import the needed modules
import sys
import os

import array

import numpy as np

from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile, TPaveText
from ROOT import gROOT, gStyle, gPad, TLegend, TLine

################################################################################
################################################################################
def main(argv):

    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option("--batch", dest="batch", action="store_true", \
            default=False,help="Run in batch mode")


    (options, args) = parser.parse_args()

    filename = args[0]

    modenum = "1237"

    gStyle.SetOptStat(0);

    modenums = []
    request_amounts = []

    pcts = []
    nevents = []

    nmodes = 0

    ############################################################################
    #Open the input file 
    if os.path.isfile(filename):
        file = open(filename,"r")

        line = file.readline()
        while line:
            vals = line.split()

            ####################################################################
            # Grab the modenums and request amount
            ####################################################################
            if vals[0]=="SP" and vals[1]=="mode" and vals[2]=="Request":
                file.readline() # Move past the "------" line
                line = file.readline()
                print line
                while line.find("------")<0:
                    modenums.append(line.split()[0])
                    print line
                    request_amounts.append(int(line.split()[1]))

                    line = file.readline()

            nmodes = len(modenums)
            for i,m in enumerate(modenums):
                pcts.append([])
                nevents.append([])
                for j in range(0,3):
                    pcts[i].append([]) # All, OnPeak, OffPeak
                    nevents[i].append([]) # All, OnPeak, OffPeak

            if len(vals)>1 and vals[1].find("Peak")>=0:
                if int(vals[0])>200003 and int(vals[0])<200712: # Check that is is when data taking started

                    lumi = float(vals[3])

                    for k in range(0,nmodes):
                        pcts[k][0].append(float(vals[4+k]))
                        nevents[k][0].append(lumi*request_amounts[k])
                        if vals[1].find("OnPeak")>=0:
                            pcts[k][1].append(float(vals[4+k]))
                            nevents[k][1].append(lumi*request_amounts[k])
                        elif vals[1].find("OffPeak")>=0:
                            pcts[k][2].append(float(vals[4+k]))
                            nevents[k][2].append(lumi*request_amounts[k])

                        

            # Move on to the next line. This should break when the end of file
            # is reached. 
            line = file.readline()

    ############################################################################
    # Fill the histograms
    ############################################################################
    h = []
    for i in range(0,nmodes):
        h.append([])
        for j in range(0,3):
            h[i].append([])
            for k in range(0,2): 

                # 0 - pcts
                # 1 - num events
                
                mean = 0
                std_dev = 0
                mean = np.mean(pcts[i][j])
                std_dev = np.std(pcts[i][j])
                '''
                if k==0:
                    mean = np.mean(pcts[i][j])
                    std_dev = np.std(pcts[i][j])
                elif k==1:
                    mean = np.mean(nevents[i][j])
                    std_dev = np.std(nevents[i][j])
                    print "mean: %f" % (mean)
                '''

                if std_dev<0.1:
                    std_dev = 1.0

                lorange = mean - 2.0*std_dev
                hirange = mean + 2.0*std_dev

                name = "h%d_%d_%d" % (i,j,k)

                h[i][j].append(TH1F(name,name,25,lorange,hirange))

                h[i][j][k].GetYaxis().SetTitle("Generated events")
                h[i][j][k].GetYaxis().SetTitle("% req events")
                h[i][j][k].SetTitle("")

                h[i][j][k].GetYaxis().SetTitleSize(0.09)
                h[i][j][k].GetYaxis().SetTitleFont(42)
                h[i][j][k].GetYaxis().SetTitleOffset(0.7)
                h[i][j][k].GetYaxis().CenterTitle()
                h[i][j][k].GetYaxis().SetNdivisions(6)

                h[i][j][k].GetXaxis().SetNdivisions(4)
                h[i][j][k].GetXaxis().SetNoExponent(False)

                color = 36
                h[i][j][k].SetLineWidth(2)
                h[i][j][k].SetLineColor(color)
                h[i][j][k].SetFillColor(color)
                h[i][j][k].SetMarkerColor(color)

                if k==0:
                    for p in pcts[i][j]:
                        h[i][j][k].Fill(p)
                elif k==1:
                    for l,p in enumerate(nevents[i][j]):
                        if i==0 and j==2:
                            print p
                        #print pcts[i][j][l]
                        h[i][j][k].Fill(pcts[i][j][l],p)


    print modenums
    print request_amounts
    can = []
    for i in range(0,nmodes):
        name = "can%d" % (i)
        title = "Mode %s" % (modenums[i])
        can.append(TCanvas(name, title, 10+10*i, 10+10*i, 900, 600))
        can[i].SetFillColor(0)
        can[i].Divide(3,2)


    text = []
    for i in range(0,nmodes):
        text.append([])
        for j in range(0,3):
            text[i].append([])
            for k in range(0,2):
                can[i].cd(j+1 + 3*k)
                gPad.SetLeftMargin(0.30);
                gPad.SetTopMargin(0.12);
                gPad.SetBottomMargin(0.15);
                h[i][j][k].Draw()

                text[i][j].append(TPaveText(0.00, 0.95, 0.45, 0.99,"NDC"))
                name = "mode: %d" % int(modenums[i])
                text[i][j][k].AddText(name);
                text[i][j][k].SetFillColor(1)
                text[i][j][k].SetTextColor(0)
                text[i][j][k].Draw()

                gPad.Update()
            

    ## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
    if (not options.batch):
      print "a"
      if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q' ]:
          rep = raw_input( 'enter "q" to quit: ' )
          if 1 < len(rep):
            rep = rep[0]



################################################################################
################################################################################

if __name__ == "__main__":
        main(sys.argv)


