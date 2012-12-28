#!/usr/bin/env python

###############################################################
# Matt Bellis
# bellis@slac.stanford.edu
###############################################################

import sys
from optparse import OptionParser
from math import *

#### Command line variables ####
doFit = False

parser = OptionParser()
parser.add_option("-f", "--fit", dest="do_fit", action = "store_true", default = False, help="Run the fit")
parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("-r", "--fit-range", dest="fit_range", help="String to represent the range over which to fit")
parser.add_option("--num-bins", dest="num_bins", default=25, help="Number of bins in 1-D plots")
parser.add_option("-m", "--max", dest="max", help="Maximum events to read in")
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")
parser.add_option("-s", "--scale-up", dest="scale_up", default=0, help="Scale factor for MC to calculate chi2")
parser.add_option("-d", "--dimensionality", dest="dimensionality", default=3, help="Dimensionality of fit [2,3]")
parser.add_option("--dir", dest="dir", help="Directory from which to read the pure and embedded study files.")
parser.add_option("--pure", dest="pure", action="store_true", default=False, help="Do pure toy MC studies.")
parser.add_option("--embedded", dest="embedded", action="store_true", default=False, help="Do embedded toy MC studies.")
parser.add_option("--num-sig", dest="num_sig", default=0, help="Number of signal events, embedded or otherwise.")
parser.add_option("--num-bkg", dest="num_bkg", default=650, help="Number of background events")
parser.add_option("--num-fits", dest="num_fits", default=1, help="Number of fits to run")
parser.add_option("--fit-only-sig", dest="fit_only_sig", action = "store_true", default = False, help="Fit only to the signal.")
parser.add_option("--fit-only-bkg", dest="fit_only_bkg", action = "store_true", default = False, help="Fit only to the background.")
parser.add_option("--sideband-first", dest="sideband_first", action = "store_true", default = False, help="Keep stuff free and then fix it")
parser.add_option("--use-single-cb", dest="use_single_cb", action = "store_true", default = False, help="Use the single CB in Delta E")
parser.add_option("--only-data", dest="only_data", action = "store_true", default=False, help="Plot only the data and not the fit results")
parser.add_option("--workspace", dest="workspace_file", default="default_workspace_file.root", help="File from which to grab the RooWorkspace")

(options, args) = parser.parse_args()

#############################################################
# Root stuff
#############################################################
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

from array import *

gROOT.Reset()
# Some global style settings
gStyle.SetOptStat(1111111)
gStyle.SetPadRightMargin(0.08)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)

#gStyle.SetFillColor(0)
gStyle.SetTitleYOffset(2.00)


############################################
# Grab the infile if there is one.
nfiles = len(args)
if nfiles==0 or args[0] == None:
    print "Must pass in a file!"
    exit(-1) 

fitnsigs = array('f')
nfits = array('f')
gtsigs = [array('f'),array('f'),array('f'),array('f'),array('f'),array('f')]

infilename = None
xpts = []
ypts = []
p_or_e = []
which_nsig_to_print = 12.0
nsig_to_print = -1
for f in range(0,nfiles):
    infilename = args[f]
    xpts.append([])
    ypts.append([])
    ############################################
    # Parse the infile name to get information.
    ############################################
    # nsig
    fitnsig = 0
    p0 = infilename.find("sig") + 3
    p1 = infilename.find("_",p0)
    fitnsig = float(infilename[p0:p1])
    print fitnsig
    if fitnsig==which_nsig_to_print:
        nsig_to_print = f
    
    # nbkg
    fitnbkg = 0
    p0 = infilename.find("bkg") + 3
    p1 = infilename.find("_",p0)
    fitnbkg = float(infilename[p0:p1])
    print fitnbkg
    
    # nfits
    nf = 0
    p0 = infilename.find("nfits") + 5
    p1 = infilename.find(".",p0)
    nf = float(infilename[p0:p1])
    print nf
    
    # p_or_e
    pe = None
    pe = infilename.find("pure")>=0 # pe is True for pure, false for embed
    print pe
    
    fitnsigs.append(fitnsig)
    nfits.append(nf)
    p_or_e.append(pe)

    ############################################
    ############################################
    std_dev_limits = [3.0,4.0,5.0]
    num_above = [0,0,0]
    ngraphs = 3
    for i in range(0,3):
        xpts[f].append(array('f'))
        ypts[f].append(array('f'))

    count = 0
    infile = open(infilename)
    for line in infile:
        if line[0]!='#':
            vals = line.split()
            nsig = float(vals[0])
            ul = float(vals[1])
            dll = float(vals[2])

            if dll<0.0 and nsig>0.0:
                print "%d %f %f %f" % (count,nsig,ul,dll)

            if dll==-0.0001 and nsig>0.0:
                dll = 0.0
            
            #print dll
            #if 1:
            if dll>=-0.0:
                std_dev = sqrt(2.0*dll)
                #std_dev = dll
            else:
                std_dev = -1.0

            if nsig<0.0:
                std_dev = 0.0

            for i,limits in enumerate(std_dev_limits):
                if std_dev>limits:
                    num_above[i] += 1
                

            xpts[f][0].append(nsig)
            xpts[f][1].append(nsig)
            xpts[f][2].append(std_dev)

            ypts[f][0].append(ul)
            ypts[f][1].append(std_dev)
            ypts[f][2].append(ul)

            count += 1

    for i,n in enumerate(num_above):
        print "Num greater than %3.1f sigma: %d" % (std_dev_limits[i],n)
        gtsigs[i].append(n/nfits[f])
        gtsigs[i+3].append(1000.0*n/nfits[f])



################################################################################
################################################################################
# Plot data 
can = []
num_cans = 3
for f in range(0,nfiles):
    can.append([])
    for i in range(0,num_cans):
        name = "can%d_%d" % (f,i)
        can[f].append(TCanvas(name, name, 10+30*i+10*f, 10+10*f, 400, 400))
        can[f][i].SetFillColor(0)
        can[f][i].Divide(1,1)

################################################################################
# Make some graphs for just plotting the 1D data
################################################################################
graphs = []
for f in range(0,nfiles):
    graphs.append([])
    for i in range(0,ngraphs):
        graphs[f].append(TGraph(len(xpts[f][i]),xpts[f][i],ypts[f][i]))

        graphs[f][i].SetTitle()

        graphs[f][i].GetXaxis().SetNdivisions(6)
        graphs[f][i].GetYaxis().SetNdivisions(6)

        graphs[f][i].GetXaxis().SetLabelSize(0.06)
        graphs[f][i].GetYaxis().SetLabelSize(0.06)

        graphs[f][i].GetXaxis().CenterTitle()
        graphs[f][i].GetXaxis().SetTitleSize(0.09)
        graphs[f][i].GetXaxis().SetTitleOffset(1.0)

        graphs[f][i].GetXaxis().SetTitle("# sig events (from fit)")
        if i==0:
            graphs[f][i].GetYaxis().SetTitle("Upper limit: 90%>0")
        elif i==1:
            graphs[f][i].GetYaxis().SetTitle("Significance: #sigma = #sqrt{2 L}")
        elif i==2:
            graphs[f][i].GetXaxis().SetTitle("Significance: #sigma = #sqrt{2 L}")
            graphs[f][i].GetYaxis().SetTitle("Upper limit: 90%>0")

        graphs[f][i].GetYaxis().CenterTitle()
        graphs[f][i].GetYaxis().SetTitleSize(0.06)
        graphs[f][i].GetYaxis().SetTitleOffset(1.2)

        '''
        print "Axis dimensions"
        print graphs[f][i].GetXaxis().GetXmin()
        print graphs[f][i].GetXaxis().GetXmax()
        print graphs[f][i].GetXaxis().GetBinCenter(10)
        '''

        #graphs[f][i].GetYaxis().SetRangeUser(0,70)

        if i==0:
            graphs[f][i].SetMarkerColor(4)
        elif i==1:
            graphs[f][i].SetMarkerColor(2)
        elif i==2:
            graphs[f][i].SetMarkerColor(9)
        graphs[f][i].SetMarkerStyle(20)
        graphs[f][i].SetMarkerSize(0.2)

################################################################################
gr_sum = []
for i in range(0,6):
        gr_sum.append(TGraph(len(fitnsigs),fitnsigs,gtsigs[i]))

        gr_sum[i].SetTitle()

        gr_sum[i].GetXaxis().SetNdivisions(6)
        gr_sum[i].GetYaxis().SetNdivisions(6)

        gr_sum[i].GetXaxis().SetLabelSize(0.06)
        gr_sum[i].GetYaxis().SetLabelSize(0.06)

        gr_sum[i].GetXaxis().CenterTitle()
        gr_sum[i].GetXaxis().SetTitleSize(0.09)
        gr_sum[i].GetXaxis().SetTitleOffset(1.0)

        gr_sum[i].GetXaxis().SetTitle("# sig events (placed in sample)")
        gr_sum[i].GetYaxis().SetTitle("% trials above X significance")

        gr_sum[i].GetYaxis().CenterTitle()
        gr_sum[i].GetYaxis().SetTitleSize(0.06)
        gr_sum[i].GetYaxis().SetTitleOffset(1.2)

        gr_sum[i].GetXaxis().SetLimits(0,45)

        if i%3==0:
            gr_sum[i].SetMarkerColor(4)
        elif i%3==1:
            gr_sum[i].SetMarkerColor(2)
        elif i%3==2:
            gr_sum[i].SetMarkerColor(3)

        gr_sum[i].SetMarkerStyle(20)
        gr_sum[i].SetMarkerSize(1.5)

################################################################################
################################################################################
# Plot sum data 
can_sum = []
for i in range(0,1):
    name = "can_sum_%d" % (i)
    can_sum.append(TCanvas(name, name, 400,400,800,600))
    can_sum[i].SetFillColor(0)
    can_sum[i].Divide(1,1)

################################################################################
# Draw!
for f in range(0,nfiles):
    for i in range(0,3):
        can[f][i].cd(1)
        graphs[f][i].Draw("ap")
        gPad.Update()

        if f==nsig_to_print:
            name = "Plots/can_summaries_%s_individual_nsig%d_%d.eps" % (options.tag, int(which_nsig_to_print), i)
            if i==1:
                graphs[f][i].GetYaxis().SetRangeUser(0,8)
            graphs[f][i].GetXaxis().SetLimits(0,2*which_nsig_to_print)
            graphs[f][i].Draw("ap")
            gPad.Update()
            can[f][i].SaveAs(name)


# Draw!
lines = []
legend = []
legend.append(TLegend(0.8,0.3,0.99,0.69))
axis = TGaxis()
f1 = TF1()
for i in range(0,3):
    can_sum[0].cd(1)

    # Scale for the cross section overlay
    if i==2:
        f1 = TF1("f1","x*1e-4", 0.0,45.0*1e-4)
        axis = TGaxis(gPad.GetUxmin(),gPad.GetUymax(),gPad.GetUxmax(),gPad.GetUymax(),"f1",510,"-")
        #axis.SetNoExponent(kFALSE)
        axis.SetMaxDigits(2)
        axis.SetLineColor(kRed)
        axis.SetLabelColor(kRed)
        axis.SetTitle("Branching fraction")
        axis.SetTitleColor(kRed)
        axis.SetNdivisions(6)
        #axis.SetDecimals(kTRUE)
        axis.Draw()


    if i==0:
        gr_sum[i].Draw("ap")
    else:
        gr_sum[i].Draw("p")
    if i==2:
        lines.append(TLine(0.0,0.9,45.0,0.9))
        lines[0].SetLineStyle(2)
        lines[0].SetLineWidth(4)
        lines[0].Draw()

        lines.append(TLine(0.0,0.5,45.0,0.5))
        lines[1].SetLineStyle(2)
        lines[1].SetLineWidth(4)
        lines[1].Draw()

    name = "%d #sigma" % (i+3)
    legend[0].AddEntry(gr_sum[i],name,"p")

    if i==2:
        #legend[0].SetFillColor(0)
        legend[0].Draw()

    gPad.Update()


'''
for i in range(0,num_cans):
  name = "Plots/can_mcstudies_summaries_%s_%d.eps" % (options.tag, i)
  can[i].Update()
  can[i].SaveAs(name)
'''
name = "Plots/can_summaries_%s.eps" % (options.tag)
can_sum[0].Update()
can_sum[0].SaveAs(name)


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


