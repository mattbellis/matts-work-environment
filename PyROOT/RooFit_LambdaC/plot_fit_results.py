#!/usr/bin/env python

###############################################################
# intro3.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro3.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
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
parser.add_option("--num-bins-mes", dest="num_bins_mes", default=25, help="Number of bins in 1-D plots for mES")
parser.add_option("--num-bins-de", dest="num_bins_de", default=25, help="Number of bins in 1-D plots for DeltaE")
parser.add_option("--num-bins-nn", dest="num_bins_nn", default=25, help="Number of bins in 1-D plots for NN")
parser.add_option("-m", "--max", dest="max", help="Maximum events to read in")
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")
parser.add_option("-s", "--scale-up", dest="scale_up", default=0, help="Scale factor for MC to calculate chi2")
parser.add_option("-d", "--dimensionality", dest="dimensionality", default=3, help="Dimensionality of fit [2,3]")
parser.add_option("--dir", dest="dir", help="Directory from which to read the pure and embedded study files.")
parser.add_option("--iteration", dest="iteration", default=0, help="Which iteration to display")
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
parser.add_option("--workspace", dest="workspace_file", default="default_workspace_file.root", \
                   help="File from which to grab the RooWorkspace")

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
gStyle.SetOptStat(11)
gStyle.SetPadRightMargin(0.05)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetPadBottomMargin(0.14)
gStyle.SetPadTopMargin(0.02)
gStyle.SetFrameFillColor(0)

#gStyle.SetFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)

set_palette("palette",99)

############################################
RooMsgService.instance().Print()
RooMsgService.instance().deleteStream(1)
RooMsgService.instance().Print()
############################################

############################################
# Grab the infile if there is one.
infilename = None
if len(args)>0 and args[0] != None:
  infilename = args[0]
############################################
dim = int(options.dimensionality)
############################################

################################################################################
################################################################################
# Make the root log file that will hold the RooWorkspace object.
################################################################################
workspace_file = TFile(options.workspace_file, "READ")
in_wname = options.workspace_file.split('/')[-1].split('.root')[0]
w = workspace_file.Get(in_wname) # RooWorkspace
baryon = None
ntp = None
if in_wname.find("LambdaC")>=0:
    baryon = "LambdaC"
elif in_wname.find("Lambda0")>=0:
    baryon = "Lambda0"

if in_wname.find("ntp1")>=0:
    ntp = "ntp1"
elif in_wname.find("ntp2")>=0:
    ntp = "ntp2"
elif in_wname.find("ntp3")>=0:
    ntp = "ntp3"
elif in_wname.find("ntp4")>=0:
    ntp = "ntp4"
################################################################################
################################################################################
w.Print("v")
x = w.var("x")
y = w.var("y")
z = w.var("z")
z.setBins(200, "cache")
scan_x = w.var("scan_x")
scan_y = w.var("scan_y")
scan_z = w.var("scan_z")
scan_points_dataset = w.data("scan_points_dataset")

# Dump the vars
all_vars = w.allVars()
iter = all_vars.createIterator()
# Build a dictionary
my_vars = {}
nvars = all_vars.getSize()
# Grab the pdf we want to plot
for i in range(0,nvars):
    p = iter.Next()
    print "%s %5.5f" % (p.GetName(), p.getVal())
    my_vars[p.GetName()] = p

################################################################################
name = "dataset_%s" % (options.iteration)
#data = w.data("dataset_0")
data = w.data(name)
name = "dataset_z_%s" % (options.iteration)
#data_z = w.data("dataset_z_0")
data_z = w.data(name)
################################################################################
all_pdfs = w.allPdfs()
iter = all_pdfs.createIterator()
# Build a dictionary
pdfs = {}
npdfs = all_pdfs.getSize()
# Grab the pdf we want to plot
for i in range(0,npdfs):
    p = iter.Next()
    pdfs[p.GetName()] = p

################################################################################
#########################################################
# Set some ranges which will be used later
#########################################################
xmin = x.getMin()
xmax = x.getMax()
ymin = y.getMin()
ymax = y.getMax()
# Standard
zmin = z.getMin()
zmax = z.getMax()
# For plotting for the paper
#zmin = 0.5
#zmax = 1.0

# For plotting for the paper
xmax = 5.29
x.setMax(xmax)


print "Ranges:"
print "x: %5.3f %5.3f" % (xmin, xmax)
print "y: %5.3f %5.3f" % (ymin, ymax)
print "z: %5.3f %5.3f" % (zmin, zmax)

x.setRange("FULL",xmin,xmax)
#x.setRange("FULL",xmin,5.29)
y.setRange("FULL",ymin,ymax)
z.setRange("FULL",zmin, zmax)

x.setRange("SIGNAL",5.25, xmax)
y.setRange("SIGNAL",-0.10, 0.1)
z.setRange("SIGNAL", zmin, zmax)

# Sideband 1 region
x.setRange("SB1", xmin,  xmax)
y.setRange("SB1", 0.075, ymax)
z.setRange("SB1", zmin, zmax)
# Sideband 2 region
x.setRange("SB2",  xmin,  xmax)
y.setRange("SB2", ymin, -0.075) 
z.setRange("SB2", zmin, zmax)
# Sideband 3 region
x.setRange("SB3",xmin,5.27) 
y.setRange("SB3",-0.075,0.075) 
z.setRange("SB3",zmin,zmax)

################################################################################
#############################
# Range over which to fit
#############################
fit_range = "FULL"
if options.fit_range:
  fit_range = options.fit_range

################################################################################
from my_roofit_utilities import *
#########################################################
################################################################################
################################################################################
################################################################################
# Plot data 
can = []
num_cans = 4+6
for i in range(0,num_cans):
    name = "can%d" % (i)
    if i<2:
        can.append(TCanvas(name, name, 10+20*i, 10+20*i, 1400, 400))
        can[i].SetFillColor(0)
        can[i].Divide(3,1)
    else:
        can.append(TCanvas(name, name, 10+20*i, 10+20*i, 400, 400))
        can[i].SetFillColor(0)
        can[i].Divide(1,1)

################################################################################
# Make some frames for just plotting the 1D data
################################################################################
nbinsx = 0
ndiv = 0
frames = []
for i in range(0,3):
    if i%3==0:
        frames.append(x.frame(RooFit.Bins(int(options.num_bins_mes))))
    elif i%3==1:
        frames.append(y.frame(RooFit.Bins(int(options.num_bins_de))))
    elif i%3==2:
        frames.append(z.frame(RooFit.Bins(int(options.num_bins_nn))))

    name = "frame_%d" % (i)
    frames[i].SetName(name)
    frames[i].GetYaxis().SetNdivisions(4)
    frames[i].GetXaxis().SetNdivisions(6)

    frames[i].GetYaxis().SetLabelSize(0.06)
    frames[i].GetXaxis().SetLabelSize(0.06)

    frames[i].GetXaxis().CenterTitle()
    frames[i].GetXaxis().SetTitleSize(0.07)
    frames[i].GetXaxis().SetTitleOffset(0.9)

    frames[i].GetYaxis().CenterTitle()
    frames[i].GetYaxis().SetTitleSize(0.07)
    frames[i].GetYaxis().SetTitleOffset(1.0)

    nbinsx = frames[i].GetNbinsX()
    xmin = frames[i].GetXaxis().GetXmin()
    xmax = frames[i].GetXaxis().GetXmax()
    mydiv = float((xmax-xmin)/nbinsx)
    print "Frames division: %f %f %d" % (xmax,xmin,nbinsx)
    units = frames[i].getPlotVar().getUnit()
    if i==2:
        ytitle = "Entries / %.3f %s" % (mydiv,units)
    else:
        ytitle = "Entries / ( %.3f %s )" % (mydiv,units)
    frames[i].SetYTitle(ytitle)

    frames[i].SetMarkerSize(0.02)
    frames[i].SetMinimum(0)
    if baryon=="LambdaC":
        if i==0:
            frames[i].SetMaximum(60)
        elif i==1:
            frames[i].SetMaximum(75)
        elif i==2:
            frames[i].SetMaximum(85)
    elif baryon=="Lambda0":
        frames[i].SetMaximum(37)


    if i==0:
        frames[i].GetXaxis().SetLimits(5.2,5.291)

    frames[i].SetTitle("")

################################################################################
# Fill the 1D plots
rllist = RooLinkedList()
rllist.Add(RooFit.MarkerSize(0.5))

# Plot on two canvaes (k index)
for k in range(0,2):
    for i in range(0,3):
        if k==0:
            can[0].cd(i+1) 
        else:
            can[i+4].cd(1) 
        data.plotOn(frames[i], rllist)
        frames[i].Draw() 
        gPad.Update()


################################################################################
# Make some 2D histos for the projections of the data.
h2d = []
for i in range(0,3):
    if i%3==0:
        rllist = RooLinkedList()
        rllist.Add(RooFit.Binning(25))
        rllist.Add(RooFit.YVar(y, RooFit.Binning(25)))
        h2d.append(x.createHistogram("x vs y data",  rllist))
        data.fillHistogram(h2d[i], RooArgList(x,y))
    elif i%3==1:
        rllist = RooLinkedList()
        rllist.Add(RooFit.Binning(25))
        rllist.Add(RooFit.YVar(z, RooFit.Binning(25)))
        h2d.append(x.createHistogram("x vs z data",  rllist))
        data.fillHistogram(h2d[i], RooArgList(x,z))
    elif i%3==2:
        rllist = RooLinkedList()
        rllist.Add(RooFit.Binning(25))
        rllist.Add(RooFit.YVar(z, RooFit.Binning(25)))
        h2d.append(y.createHistogram("y vs z data",  rllist))
        data.fillHistogram(h2d[i], RooArgList(y,z))
    name = "h2d_%d" % (i)
    h2d[i].SetName(name)
    #h2d[i].SetMaximum(5)
    h2d[i].SetMinimum(0)
    #h2d[i].SetTitle("")

    h2d[i].GetYaxis().SetNdivisions(4)
    h2d[i].GetXaxis().SetNdivisions(6)

    h2d[i].GetYaxis().SetLabelSize(0.06)
    h2d[i].GetXaxis().SetLabelSize(0.06)

    h2d[i].GetXaxis().CenterTitle()
    h2d[i].GetXaxis().SetTitleSize(0.09)
    h2d[i].GetXaxis().SetTitleOffset(1.0)

    h2d[i].GetYaxis().CenterTitle()
    h2d[i].GetYaxis().SetTitleSize(0.09)
    h2d[i].GetYaxis().SetTitleOffset(1.0)

for i in range(0,3):
    can[1].cd(i+1)
    #h2d[i].Draw("BOX")
    h2d[i].Draw("COLZ")
    gPad.Update()

################################################################################
################################################################################
print "Finished plotting the data"
################################################################################

#########################################################
# Print the fit functions
#########################################################
# Plot the fit results
################################################################################
# Store the pdf names for x, y, z projections
pdfnames_sig = [[], [], []]
pdfnames_sig[0] = ["CB"] 
pdfnames_sig[1] = ["CBdE","CBdE_2","double_cbdE"] 
#pdfnames_sig[2] = ["rpsf_s"]
pdfnames_sig[2] = ["nn_sig"]

pdfcolors_sig = [[],[],[]]
pdfcolors_sig[0] = [32]
pdfcolors_sig[1] = [2,3,4]
pdfcolors_sig[2] = [41]

pdfnames_bkg = [[], [], []]
pdfnames_bkg[0] = ["argus"] 
pdfnames_bkg[1] = ["polyy"] 
pdfnames_bkg[2] = ["CB_NN"]

pdfcolors_bkg = [[],[],[]]
pdfcolors_bkg[0] = [22]
pdfcolors_bkg[1] = [24]
pdfcolors_bkg[2] = [26]
################################################################################
total_name = "total"
if options.fit_only_sig:
    total_name = "sig_pdf"
elif options.fit_only_bkg:
    total_name = "bkg_pdf"

#'''

text = []
if not options.only_data:
    # Plot on two sets of canvases
    for k in range(0,2):
        text.append([])
        for i in range(0,dim):
        
            if k==0:
                can[0].cd(i+1) 
            else:
                can[i+4].cd(1) 

            #can[0].cd(i+1)
            # fit_func
            argset = RooArgSet(pdfs[total_name])
            pdfs[total_name].plotOn(frames[i],
                                 RooFit.Components(argset),
                                 RooFit.LineColor(4),
                                 RooFit.LineWidth(3),
                                 RooFit.Range( fit_range ),
                                 RooFit.NormRange( fit_range ) )

            # Print signal 
            '''
            pdfs[total_name].plotOn(frames[i],
                           RooFit.Components("sig_pdf"),
                           RooFit.LineColor(8),
                           RooFit.LineStyle(1),
                           RooFit.LineWidth(3),
                           RooFit.Range( fit_range ),
                           RooFit.NormRange( fit_range ) )
            '''

            # Print background 
            pdfs[total_name].plotOn(frames[i],
                           RooFit.Components("bkg_pdf"),
                           RooFit.LineColor(4),
                           RooFit.LineStyle(2),
                           RooFit.LineWidth(5),
                           RooFit.Range( fit_range ),
                           RooFit.NormRange( fit_range ) )

            # Plot the sub functions
            # Signal
            if options.fit_only_sig:
                for j,p in enumerate(pdfnames_sig[i]):
                    color = pdfcolors_sig[i][j]
                    print color
                    argset = RooArgSet(pdfs[p])
                    pdfs["sig_pdf"].plotOn(frames[i],
                                   RooFit.Components(argset),
                                   RooFit.LineColor(color),
                                   RooFit.LineStyle(2),
                                   RooFit.Range( fit_range ),
                                   RooFit.NormRange( fit_range ) )

            # Background
            if options.fit_only_bkg:
                for j,p in enumerate(pdfnames_bkg[i]):
                    color = pdfcolors_bkg[i][j]
                    print color
                    argset = RooArgSet(pdfs[p])
                    pdfs["bkg_pdf"].plotOn(frames[i],
                                   RooFit.Components(argset),
                                   RooFit.LineColor(color),
                                   RooFit.LineStyle(2),
                                   RooFit.Range( fit_range ),
                                   RooFit.NormRange( fit_range ) )

            
            # Do this part for the paper.
            line = None
            if i==2:
                frames[i].GetXaxis().SetLimits(0.6,1.0)
                line = TLine(zmin,0.0,zmin,85)
                line.SetLineStyle(2)
                line.SetLineColor(2)
                line.SetLineWidth(2)
                line.Draw()

            frames[i].SetMinimum(0)
            frames[i].Draw()
            if i==2:
                line.Draw()

            text[k].append(TPaveText(0.45,0.85,0.93,0.95,"NDC"))

            subcaption_letter = ["a","b","c"]
            baryon_text = "#Lambda^{+}_{c}"
            lepton_text = "#mu^{-}"

            if baryon=="LambdaC":
                baryon_text = "#Lambda^{+}_{c}"
                if ntp=="ntp1":
                    lepton_text = "#mu^{-}"
                    subcaption_letter = ["a","c","e"]
                elif ntp=="ntp2":
                    lepton_text = "#font[12]{e}^{-}"
                    subcaption_letter = ["b","d","f"]
            elif baryon=="Lambda0":
                baryon_text = "#Lambda"
                if ntp=="ntp1":
                    lepton_text = "#mu^{-}"
                    subcaption_letter = ["a","c","e"]
                elif ntp=="ntp2":
                    lepton_text = "#font[12]{e}^{-}"
                    subcaption_letter = ["b","d","f"]
                elif ntp=="ntp3":
                    baryon_text = "#bar{#Lambda}"
                    lepton_text = "#mu^{-}"
                    subcaption_letter = ["a","c","e"]
                elif ntp=="ntp4":
                    baryon_text = "#bar{#Lambda}"
                    lepton_text = "#font[12]{e}^{-}"
                    subcaption_letter = ["b","d","f"]

            mytext = "%s) #font[12]{B}#rightarrow %s %s" % (subcaption_letter[i],baryon_text,lepton_text)
            text[k][i].AddText(mytext)
            text[k][i].SetFillColor(0)
            text[k][i].SetBorderSize(0)
            text[k][i].Draw()

            gPad.Update()

################################################################################
################################################################################
################################################################################
####################################################
# Print the likelihood curves
####################################################
################################################################################
#'''
if not options.fit_only_bkg:
    ################################################################################
    name = "fitresult_%s" % (options.iteration)
    print name
    #fit_result = w.genobj("fitresult_0")
    fit_result = w.genobj(name)
    fit_func = w.function("fit_func")
    #log_gc_obj = w.function("log_gc")
    #scan_points_dataset = w.data("scan_points_dataset")
    fit_result.Print("v")

    #nllframe,graphsll,upper_lim_vals,std_from_0 = likelihood_curve(fit_result, data,  pdfs[total_name], my_vars["nsig"])
    log_file = "fit_summary_log_files/%s.log" % (in_wname)
    nllframe,graphsll,upper_lim_vals,std_from_0 = likelihood_curve(fit_result, data, scan_points_dataset, fit_func, pdfs[total_name], my_vars["branching_fraction"],log_file)

    ############################################################################
    ################################################################################
    # Format the nllframe
    ################################################################################
    nllframe.SetMinimum(0)
    nllframe.SetTitle("")

    nllframe.GetYaxis().SetNdivisions(4)
    nllframe.GetXaxis().SetNdivisions(6)

    nllframe.GetYaxis().SetLabelSize(0.06)
    nllframe.GetXaxis().SetLabelSize(0.06)

    nllframe.GetXaxis().CenterTitle()
    nllframe.GetXaxis().SetTitleSize(0.06)
    nllframe.GetXaxis().SetTitleOffset(1.0)

    nllframe.GetYaxis().CenterTitle()
    nllframe.GetYaxis().SetTitleSize(0.09)
    nllframe.GetYaxis().SetTitleOffset(1.0)

    # Might have to set this for each final fit
    #nllframe.GetXaxis().SetLimits(-1.0,3.0)

    nllframe.GetXaxis().SetTitle("Branching fraction #times 10^{-8}")
    nllframe.GetYaxis().SetTitle("#Delta ln L")

    ##########################
    ################################################################################
    # Format the graphs
    ################################################################################
    xmin =  0.0
    xmax = 20.0
    for ig,g in enumerate(graphsll):
        g.SetMinimum(0)
        g.SetTitle("")

        g.GetYaxis().SetNdivisions(4)
        g.GetXaxis().SetNdivisions(6)

        g.GetYaxis().SetLabelSize(0.06)
        g.GetXaxis().SetLabelSize(0.06)

        g.GetXaxis().CenterTitle()
        g.GetXaxis().SetTitleSize(0.06)
        g.GetXaxis().SetTitleOffset(1.0)

        g.GetYaxis().CenterTitle()
        g.GetYaxis().SetTitleSize(0.09)
        g.GetYaxis().SetTitleOffset(1.0)

        # Might have to set this for each final fit
        #g.GetXaxis().SetLimits(-1.0,3.0)
        if ig==0:
            xmax = g.GetXaxis().GetXmax()
            xmin = g.GetXaxis().GetXmin()
            print "XMIN........... %f" % (xmin)


        g.GetXaxis().SetTitle("Branching fraction #times 10^{-8}")
        g.GetYaxis().SetTitle("exp(-#Delta ln L)")
        if ig==2:
            g.GetYaxis().SetTitle("#Delta ln L")

    ##########################

    can[2].cd(1)
    xmin = std_from_0[1] - 2.0*std_from_0[2]
    if xmin>0:
        xmin=0
    print "xmin/xmax: %f %f" % (xmin,xmax)
    #nllframe.GetXaxis().SetLimits(xmin,xmax)
    #nllframe.SetAxisRange(0.0, 15.0, "Y")
    '''
    nllframe.Draw()
    lines = []
    lines.append(TLine(0.0,0.0,0.0,20.0))
    for l in lines:
        l.SetLineStyle(2)
        l.Draw()
    '''

    graphsll[2].Draw("apl")
    gPad.Update()

    can[3].cd(1)
    graphsll[0].Draw("apl")
    graphsll[1].Draw("b")
    legend = TLegend(0.2,0.9,0.99,0.99)
    legend.SetFillColor(0)
    legend.AddEntry(graphsll[1],"90% of area > 0","f")
    legend.Draw()
    gPad.Update()

    can[7].cd(1)
    #h2d[i].Draw("BOX")
    h2d[0].Draw("COLZ")
    gPad.Update()

    can[8].cd(1)
    # Plot the 2D PDF and the generated data
    rllist = RooLinkedList()
    rllist.Add(RooFit.Binning(50))
    rllist.Add(RooFit.YVar(y, RooFit.Binning(50)))
    h2_0 = x.createHistogram("x vs y pdf",  rllist)
    pdfs[total_name].fillHistogram(h2_0, RooArgList(x,y))
    h2_0.Draw("SURF")

    can[9].cd(1)
    testxframe = x.frame()
    argset = RooArgSet(pdfs[total_name])
    pdfs[total_name].plotOn(testxframe,
                   RooFit.Components(argset),
                   #RooFit.Components("bkg_pdf"),
                   RooFit.LineColor(4),
                   #RooFit.LineStyle(2),
                   RooFit.LineWidth(3),
                   RooFit.Range( fit_range ),
                   RooFit.NormRange( fit_range ) )
    testxframe.Draw()
    gPad.Update()

    pdfs[total_name].Print("v")
    obs = RooArgSet(x)
    xtestvals = [5.270, 5.272, 5.274, 5.276, 5.278, 5.280, 5.282]
    ytestvals = [-0.010, -0.008, -0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008, 0.010]
    for xval in xtestvals:
        for yval in ytestvals:
            x.setVal(xval)
            y.setVal(yval)
            print "PDF value in signal region: mes: %f\tdeltaE: %f\ttotal: %f" % (xval, yval, pdfs[total_name].getVal(obs))

    xval = 5.25
    yval = 0.0
    x.setVal(xval)
    y.setVal(yval)
    print "PDF value in signal region: mes: %f\tdeltaE: %f\ttotal: %f" % (xval, yval, pdfs[total_name].getVal())

    x.setRange("SIGNAL_REGION",5.270, 5.300)
    y.setRange("SIGNAL_REGION",-0.032, 0.032)
    z.setRange("SIGNAL_REGION", zmin, zmax)
    norm_variables = RooArgSet(x,y)
    int_val = pdfs["bkg_pdf"].createIntegral(RooArgSet(x,y),RooFit.NormSet(RooArgSet(x,y)),RooFit.Range("SIGNAL_REGION"))
    print "BKG in signal region: %f" % (int_val.getVal())
    int_val = pdfs["bkg_pdf"].createIntegral(RooArgSet(x,y),RooFit.NormSet(RooArgSet(x,y)))
    print "BKG in signal region: %f" % (int_val.getVal())



    ############################################################################
    # Finished plotting
    ############################################################################


    # Double check the 90% upper limit area calc
    print graphsll[0]
    #graphsll[0].Print("v")
    npts = graphsll[0].GetN()
    pt0 = 0
    pt90 = npts
    xpts = graphsll[0].GetX()
    ypts = graphsll[0].GetY()
    bin_width = xpts[2]-xpts[1]
    for i in range(0,npts):
        pt = xpts[i]
        if fabs(pt-0.0) <= bin_width:
            pt0 = i
        if fabs(pt-upper_lim_vals[2]) <= bin_width:
            pt90 = i
    #total_integral = graphsll[0].Integral()
    #total_integral_from_0 = graphsll[0].Integral(pt0,npts-1)
    #integral_from_0_90 = graphsll[0].Integral(pt0,pt90)
    total_integral = 0
    for i in range(0,npts):
        total_integral += ypts[i]*bin_width

    total_integral_from_0 = 0
    for i in range(pt0,npts):
        total_integral_from_0 += ypts[i]*bin_width

    integral_from_0_90 = 0
    for i in range(pt0,pt90):
        integral_from_0_90 += ypts[i]*bin_width

    print "npts/bin_width/pt0/pt90: %d %f %d %d" % (npts,bin_width,pt0,pt90)
    print "total_integral: %f" % (total_integral)
    print "total_integral_from_0: %f" % (total_integral_from_0)
    print "integral_from_0_90: %f" % (integral_from_0_90)
    if total_integral_from_0!=0:
        print "90: %f" % (integral_from_0_90/total_integral_from_0)


    gPad.Update()

    print "Most likely nsig: %3.3f" % (std_from_0[1])
    print "Most likely branching fraction: %3.3f" % (std_from_0[1])
    print "diff NLL: %3.3f" % (std_from_0[0])
    #print "Sigma(inconsistent with 0): %3.3f" % (sqrt(2.0*std_from_0[0]))
    print "area: %3.3f" % (upper_lim_vals[0])
    print "area greater than 0: %3.3f" % (upper_lim_vals[1])
    print "ul (90%s): %3.3f" % ('%',upper_lim_vals[2])
################################################################################
################################################################################
#'''

#'''
for i in range(0,num_cans):
  name = "Plots/can_roofit_fitresults_%s_%s_%d.eps" % (in_wname,options.tag, i)
  can[i].Update()
  can[i].SaveAs(name)
  name = "Plot_root_files/can_roofit_fitresults_%s_%s_%d.root" % (in_wname,options.tag, i)
  can[i].cd(1).SaveAs(name)
#'''

print "nbinsx: %f\t\tndiv: %f" % (nbinsx,ndiv)

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


