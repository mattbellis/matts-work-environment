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
from array import *

#### Command line variables ####
doFit = False

parser = OptionParser()
parser.add_option("-f", "--fit", dest="do_fit", action = "store_true", default = False, help="Run the fit")
parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("-r", "--fit-range", dest="fit_range", help="String to represent the range over which to fit")
parser.add_option("--num-bins", dest="num_bins", default=25, help="Number of bins in 1-D plots")
parser.add_option("-m", "--max", dest="max", help="Maximum events to read in")
parser.add_option("-t", "--tag", dest="tag", default="default", help="Tag for saved .eps files")
parser.add_option("-d", "--dimensionality", dest="dimensionality", default=3, help="Dimensionality of fit [2,3]")
parser.add_option("--dir", dest="dir", help="Directory from which to read the pure and embedded study files.")
parser.add_option("--fixed-num", dest="fixed_num", action="store_true", default=False, \
        help="Use a fixed number of both background and signal.")
parser.add_option("--pure", dest="pure", action="store_true", default=False, help="Do pure toy MC studies.")
parser.add_option("--embed", dest="embed", action="store_true", default=False, help="Do embedded toy MC studies.")
parser.add_option("--branching-fraction", dest="branching_fraction", default=0.0, help="Branching fraction.")
parser.add_option("--num-sig", dest="num_sig", help="Number of signal events, embedded or otherwise.")
parser.add_option("--num-bkg", dest="num_bkg", help="Number of background events")
parser.add_option("--num-fits", dest="num_fits", default=1, help="Number of fits to run")
parser.add_option("--ntp", dest="ntp", default="ntp1", help="Baryon [LambdaC, Lambda0]")
parser.add_option("--baryon", dest="baryon", default="LambdaC", help="Ntuple over which we are running")
parser.add_option("--pass", dest="my_pass", default=0, help="From which pass to grab fit ranges")
parser.add_option("--fit-only-sig", dest="fit_only_sig", action = "store_true", default = False, help="Fit only to the signal.")
parser.add_option("--fit-only-bkg", dest="fit_only_bkg", action = "store_true", default = False, help="Fit only to the background.")
parser.add_option("--sideband-first", dest="sideband_first", action = "store_true", default = False, help="Keep stuff free and then fix it")
parser.add_option("--use-single-cb", dest="use_single_cb", action = "store_true", default = False, help="Use the single CB in Delta E")
parser.add_option("--no-gc", dest="no_gc", action = "store_true", default=False, help="Don't use the gaussian constraint")
parser.add_option("--starting-vals", dest="starting_vals_file", default=None, help="File to use for the starting values")
parser.add_option("--workspace", dest="workspace_file", help="File from which to grab the workspace and NN vals.")
parser.add_option("--nn-lo", dest="nn_lo", help="Manually set the range (lo) for the NN output.")
parser.add_option("--nn-hi", dest="nn_hi", help="Manually set the range (hi) for the NN output.")
parser.add_option("--no-plots", dest="no_plots", action = "store_true", default=False, help="Don't display the plots.")

(options, args) = parser.parse_args()

#############################################################
# Root stuff
#############################################################
import ROOT
from ROOT import *
gSystem.Load('libRooFit')

gROOT.Reset()
gStyle.SetOptStat(11111111)
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadLeftMargin(0.20)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)
# Some global style settings
gStyle.SetFillColor(0)
gStyle.SetPadLeftMargin(0.18)
gStyle.SetTitleYOffset(2.00)

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

################################################################################
# Make the root log file that will hold the RooWorkspace object.
################################################################################
p_or_e_tag = ""
if options.embed:
    p_or_e_tag = "_embed"
if options.pure:
    p_or_e_tag = "_pure"

# Record if we passed in some number of events and if it was not Poisson
num_events_tag = "_"
if options.num_sig!=None:
    num_events_tag += "sig%d_" % (int(options.num_sig))
if options.num_bkg!=None:
    num_events_tag += "bkg%d_" % (int(options.num_bkg))
if options.fixed_num:
    num_events_tag += "fixedSig_"
num_events_tag += "dim%s_" % (options.dimensionality)
if options.no_gc:
    num_events_tag += "noGC_"


wname = "workspace_%s%s%snfits%d" % \
       (options.tag, \
        p_or_e_tag, \
        num_events_tag, \
        int(options.num_fits))

wfile_name = "rootWorkspaceFiles/%s.root" % (wname)

root_log_file = TFile(wfile_name, "RECREATE")
w = RooWorkspace(wname,"My workspace")
################################################################################
################################################################################
# Parse out some of the other command line options.
################################################################################
axis3index = 2

max_events = 1e9
if options.max:
  max_events = float(options.max)

use_double_cb = True
if options.use_single_cb:
  use_double_cb = False

starting_values = []
if options.starting_vals_file != None:
  infile = open(options.starting_vals_file)
  for line in infile:
    if line[0] != "#":
      start_val = line.split()
      starting_values.append([start_val[0], start_val[2], int(start_val[3]) ] )

dim = int(options.dimensionality)

################################################################################
################################################
# Import from other files
################################################
from file_map import *

pass_info = fit_pass_info(options.baryon, options.ntp, int(options.my_pass))
mes_lo = pass_info[4][0]
mes_hi = pass_info[4][1]

deltae_lo = pass_info[5][0]
deltae_hi = pass_info[5][1]

nn_lo = pass_info[6][0]
nn_hi = pass_info[6][1]

conv_factor_calc = float(pass_info[8][0])
conv_factor_err =  float(pass_info[8][1])

if options.nn_lo:
    nn_lo = float(options.nn_lo)
if options.nn_hi:
    nn_hi = float(options.nn_hi)

data_ranges = [[mes_lo,mes_hi], [deltae_lo,deltae_hi], [nn_lo,nn_hi]]

################################################################################
from pdf_definitions import *
from my_roofit_utilities import *
################################################################################
# RooRealVars (axis)
x,y,z = build_xyz(data_ranges)
z.setBins(200, "cache")

# RooDataSets
data = RooDataSet("data","data",RooArgSet(x,y,z)) 
# We need this one for when we are *only* fitting the NN output.
data_z = RooDataSet("data_z","data_z",RooArgSet(z)) 

#########################################################
# Set some ranges which will be used later
#########################################################
print nn_lo
print nn_hi
x.setRange("FULL",mes_lo,mes_hi)
y.setRange("FULL",deltae_lo, deltae_hi)
z.setRange("FULL", nn_lo, nn_hi)

x.setRange("SIGNAL",5.25, mes_hi)
y.setRange("SIGNAL",-0.10, 0.1)
z.setRange("SIGNAL", nn_lo, nn_hi)

# Sideband 1 region
x.setRange("SB1", mes_lo,  mes_hi) 
y.setRange("SB1", 0.075, deltae_hi)
z.setRange("SB1", nn_lo, nn_hi)
# Sideband 2 region
x.setRange("SB2",  mes_lo,  mes_hi) 
y.setRange("SB2", deltae_lo, -0.075) 
z.setRange("SB2", nn_lo, nn_hi)
# Sideband 3 region
x.setRange("SB3",  mes_lo,  5.27) 
y.setRange("SB3", -0.075, 0.075) 
z.setRange("SB3", nn_lo, nn_hi)

################################################################################
################################################################################
################################################################################
# Return a dataset if we are not going to fit.
# Read this in from the command line (not a toy study).
################################################################################
infile = None
if not options.pure and not options.embed:
  infile = open(infilename)
  data_z, data = read_file_return_dataset(infile, x,y,z, data_ranges, 
                 int(options.dimensionality), max_events, 0)
  print "Num data points: %d" % ( data.numEntries() )
################################################################################
################################################################################
# Read in the dataset
################################################################################
################################################################################

mypars = []
sub_funcs = []

# Check if a workspace file has been passed in. 
# From this we would grab the RooWorkspace object and grab the NN RooHistPdf
# object for the resulting fit.
workspace = None
if options.workspace_file!=None and options.dimensionality==3:
    wfile = TFile(options.workspace_file, "READ")
    # Workspace object is same name as file, minus the '.root'
    in_wname = options.workspace_file.split('/')[-1].split('.root')[0]
    workspace = wfile.Get(in_wname)
    if workspace==None:
        print "NO WORKSPACE FOUND IN FILE!!!!!!"
        exit(-1)
    print workspace


# Grab the fit functions and everything else which is needed.
'''
dum_pars, sub_funcs_list, fit_func = tot_PDF(x,y,z, data_z, 
          dim, use_double_cb, workspace)
'''
dum_pars, sub_funcs_list, total = tot_PDF(x,y,z, data_z, 
          dim, use_double_cb, workspace)
mypars += dum_pars
#print mypars


# Create a dictionary of the pars
# We'll use this later if we need to print stuff out.
#print mypars
pars_d = {}
for p in mypars:
    pars_d[p.GetName()] = p

# Create a dictionary of extra functions
# We'll use this later if we need to print stuff out.
#print sub_funcs_list
sub_funcs = {}
for f in sub_funcs_list:
    sub_funcs[f.GetName()] = f

'''
if options.fit_only_sig:
    print "ONLY USING SIGNAL PDF!!!!!!!!!!!!!"
    fit_func = RooExtendPdf("fit_func", "Extended function for background", 
                             sub_funcs["sig_pdf"], pars_d["nsig"])
elif options.fit_only_bkg:
    fit_func = RooExtendPdf("fit_func", "Extended function for background", 
                             sub_funcs["bkg_pdf"], pars_d["nbkg"])
'''

# Set the starting values
set_starting_values(mypars, starting_values)

if options.fit_only_sig:
    print "ONLY USING SIGNAL PDF!!!!!!!!!!!!!"
    total = RooExtendPdf("fit_func", "Extended function for signal", sub_funcs["sig_pdf"], pars_d["nsig"])
elif options.fit_only_bkg:
    total = RooExtendPdf("fit_func", "Extended function for background", sub_funcs["bkg_pdf"], pars_d["nbkg"])


# Create the NLL for the fit
nll = RooNLLVar("nll","nll",total,data,RooFit.Extended(kTRUE))
fit_func = RooFormulaVar("fit_func","nll + log_gc",RooArgList(nll,pars_d["log_gc"])) 
m = RooMinuit(fit_func)
m.setVerbose(kFALSE)




reduced_data = RooDataSet()

# Set the starting values
set_starting_values(mypars, starting_values)

#val = fit_func.getParameters(RooArgSet(x,y,z)).getRealValue("nsig")
################################################################################
################################################################################
#############################
# Range over which to fit
#############################
fit_range = "FULL"
if options.fit_range:
  fit_range = options.fit_range

################################################################################
# Return a dataset if we are not going to fit.
# Read this in from the command line (not a toy study).
################################################################################
infile = None
if not options.pure and not options.embed:
  infile = open(infilename)
  data_z, data = read_file_return_dataset(infile, x,y,z, data_ranges, 
                 int(options.dimensionality), max_events, 0)
  print "Num data points: %d" % ( data.numEntries() )
################################################################################

################################################################################
parameters_string = []
if options.do_fit:
    ######################################
    infile = None

    print "Num fits: %d" % ( int(options.num_fits) )
    for n in range(0, int(options.num_fits)):
        
        fixed_num_tag = ""
        if options.fixed_num:
            fixed_num_tag += "_fixedSig"
        '''
        num_events_tag += "_dim%s" % (options.dimensionality)
        if options.no_gc:
            fixed_num_tag += "_noGC"
        '''

        if options.pure:
            infilename = "%s/mcstudies_bkg%d_sig%d%s_%04d.dat" % (options.dir, 
                         int(options.num_bkg),int(options.num_sig),fixed_num_tag,n)
        elif options.embed:
            infilename = "%s/mcstudies_bkg%d_embed_sig%d%s_%04d.dat" % \
                          (options.dir, int(options.num_bkg),int(options.num_sig),fixed_num_tag,n)

        print "Opening %s ............................." % (infilename)
        infile = open(infilename)
        ##################################################
        # Return a dataset
        # Need to delete these because we may have read them in already.
        del data
        del data_z
        data_z, data = read_file_return_dataset(infile, x,y,z, 
                       data_ranges, int(options.dimensionality), max_events, n)
        print "Num data points: %d" % ( data.numEntries() )
        ##################################################
        # Set the starting values from the file.
        set_starting_values(mypars, starting_values)
        print mypars

        ##############################################
        # Set num sig/bkg by hand if necessary.
        ##############################################
        if options.num_bkg!=None and options.num_sig!=None:
            pars_d["nbkg"].setVal( float(options.num_bkg) )
            #pars_d["nsig"].setVal( float(options.num_sig) )
            pars_d["conv_factor_calc"].setVal(conv_factor_calc)
            pars_d["conv_factor_err"].setVal(conv_factor_err)

        if options.no_gc:
            pars_d["conv_factor_calc"].setVal(conv_factor_calc)
            pars_d["conv_factor_err"].setVal(conv_factor_err)
            # Set the fit val equal to the calculated val
            pars_d["conv_factor_fit"].setVal(conv_factor_calc)

            pars_d["conv_factor_calc"].setConstant(True)
            pars_d["conv_factor_err"].setConstant(True)
            pars_d["conv_factor_fit"].setConstant(True)

            pars_d["branching_fraction"].setVal(float(options.num_sig)/conv_factor_calc)
            pars_d["branching_fraction"].setConstant(True)
        #########################################################
        ######################################
        # Write the parameters to a file
        ######################################

        if not options.pure and not options.embed:
            pars = fit_func.getParameters(RooArgSet(x,y,z)).contentsString().split(",")

            val = fit_func.getParameters(RooArgSet(x,y,z)).getRealValue("nsig")

            count = 0
            for p in pars:
                val = fit_func.getParameters(RooArgSet(x,y,z)).getRealValue(p)
                dum = "%20s%20.5f" % (p, val)
                #print dum
                parameters_string.append(dum)
                count += 1


        ###########################
        # This seems to work best
        ###########################
        fit_results = None
        if not options.sideband_first:
            print "NOT SETTING SIDEBAND FIRST!!!!!!!!!!!!!!!!!!!!!!!"
            '''
            fit_results = fit_func.fitTo(data, 
                                         RooFit.Extended(kTRUE), 
                                         RooFit.Save(kTRUE), 
                                         RooFit.Range(fit_range), 
                                         RooFit.Strategy(2), 
                                         RooFit.PrintLevel(-1) ) #RooFitResults
            '''
            #m.migrad()
            #m.hesse()
            #fit_results = m.save()
            nll = RooNLLVar("nll","nll",total,data,RooFit.Extended(kTRUE))
            fit_func = RooFormulaVar("fit_func","nll + log_gc",RooArgList(nll,pars_d["log_gc"])) 
            m = RooMinuit(fit_func)
            m.setVerbose(kFALSE)
            m.migrad()
            m.hesse()
            fit_results = m.save()

        else:
            print "FITTING SIDEBAND FIRST!!!!!!!!!!!!"
            #pars_d["nsig"].setVal(0.0)
            #pars_d["nsig"].setConstant(True)
            pars_d["branching_fraction"].setVal(0.0)
            pars_d["branching_fraction"].setConstant(True)

            # Need to make a reduced dataset
            reduced_data = data.reduce(RooFit.CutRange("SB1"))
            reduced_data.append(data.reduce(RooFit.CutRange("SB2")))
            reduced_data.append(data.reduce(RooFit.CutRange("SB3")))

            print "Num         data points: %d" % ( data.numEntries() )
            print "Num reduced data points: %d" % ( reduced_data.numEntries() )

            '''
            fit_results = fit_func.fitTo(reduced_data,
                                         RooFit.Extended(kTRUE),
                                         RooFit.Save(kTRUE),
                                         RooFit.Strategy(2),
                                         RooFit.PrintLevel(-1)) #RooFitResults
            fit_results.Print("v")
            '''
            # Create the NLL for the fit
            nll = RooNLLVar("nll","nll",total,reduced_data,RooFit.Extended(kTRUE))
            fit_func = RooFormulaVar("fit_func","nll + log_gc",RooArgList(nll,pars_d["log_gc"])) 
            m = RooMinuit(fit_func)
            m.setVerbose(kFALSE)
            m.migrad()
            m.hesse()

            #pars_d["nsig"].setVal(float(options.num_sig))
            #pars_d["nsig"].setConstant(False)
            pars_d["branching_fraction"].setVal(float(options.branching_fraction))
            pars_d["branching_fraction"].setConstant(False)

            '''
            fit_results = fit_func.fitTo(data,
                                         RooFit.Extended(kTRUE),
                                         RooFit.Save(kTRUE),
                                         RooFit.Strategy(2),
                                         RooFit.PrintLevel(-1) ) #RooFitResults
            '''
            # Create the NLL for the fit
            nll = RooNLLVar("nll","nll",total,data,RooFit.Extended(kTRUE))
            fit_func = RooFormulaVar("fit_func","nll + log_gc",RooArgList(nll,pars_d["log_gc"])) 
            m = RooMinuit(fit_func)
            m.setVerbose(kFALSE)
            m.migrad()
            m.hesse()
            fit_results = m.save()

        print "FINISHED FIT"
        fit_results.Print("v")
        name = "fitresult_%d" % (n)
        fit_results.SetName(name)
        root_log_file.cd()
        fit_results.Write(name)

        print "Finished the fit!"

        # This needs to be first, before its subcomponent PDF's
        getattr(w,'import')(fit_func)
        getattr(w,'import')(total)
        getattr(w,'import')(data)
        getattr(w,'import')(data_z)

        # Save the fit results.
        getattr(w,'import')(fit_results)

        for f in sub_funcs_list:
            getattr(w,'import')(f)
        for p in mypars:
            getattr(w,'import')(p)


        w.Print("v")

root_log_file.cd()
w.Write()
root_log_file.Close()

print "Workspace and fit results file: %s" % (root_log_file.GetName())

gDirectory.Add(w)

################################################################################
################################################################################
################################################################################

if not options.no_plots:
    # Plot data 
    can = []
    num_cans = 5
    for i in range(0,num_cans):
        name = "can%d" % (i)
        if i<2:
            can.append(TCanvas(name, name, 10+10*i, 10+200*i, 1400, 400))
            can[i].SetFillColor(0)
            can[i].Divide(3,1)
        else:
            can.append(TCanvas(name, name, 100+10*i, 500+10*i, 600, 468))
            can[i].SetFillColor(0)
            can[i].Divide(1,1)

    ################################################################################
    # Make some frames for just plotting the 1D data
    ################################################################################
    frames = []
    for i in range(0,3):
        if i%3==0:
            frames.append(x.frame(RooFit.Bins(int(options.num_bins))))
        elif i%3==1:
            frames.append(y.frame(RooFit.Bins(int(options.num_bins))))
        elif i%3==2:
            frames.append(z.frame(RooFit.Bins(int(options.num_bins))))

        frames[i].GetYaxis().SetNdivisions(4)
        frames[i].GetXaxis().SetNdivisions(6)

        frames[i].GetYaxis().SetLabelSize(0.06)
        frames[i].GetXaxis().SetLabelSize(0.06)

        frames[i].GetXaxis().CenterTitle()
        frames[i].GetXaxis().SetTitleSize(0.09)
        frames[i].GetXaxis().SetTitleOffset(1.0)

        frames[i].GetYaxis().CenterTitle()
        frames[i].GetYaxis().SetTitleSize(0.09)
        frames[i].GetYaxis().SetTitleOffset(1.0)

        '''
        print "Axis dimensions"
        print frames[i].GetXaxis().GetXmin()
        print frames[i].GetXaxis().GetXmax()
        print frames[i].GetXaxis().GetBinCenter(10)
        '''

        frames[i].SetMarkerSize(0.01)

    ################################################################################
    # Fill the 1D plots
    rllist = RooLinkedList()
    rllist.Add(RooFit.MarkerSize(0.5))
    rllist.Add(RooFit.Range(fit_range))

    for i in range(0,3):
        can[0].cd(i+1) 
        data.plotOn(frames[i], rllist)
        frames[i].Draw() 
        gPad.Update()

        can[2+i].cd(1) 
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
        h2d[i].Draw("BOX")
        gPad.Update()

    ################################################################################
    ################################################################################
    print "Finished plotting the data"
    ################################################################################
    ################################################################################
     
    print "Num         data points: %d" % ( data.numEntries() )
    if options.do_fit and options.sideband_first:
        print "Num reduced data points: %d" % ( reduced_data.numEntries() )

    ################################################################################
    # Write the canvases out.
    ################################################################################
    '''
    for i in range(0,num_cans):
        name = "Plots/can_roofit_%s_%d.eps" % (options.tag, i)
        can[i].Update()
        can[i].SaveAs(name)
    '''

###################################################
# Write the output logfile
###################################################
if options.do_fit:
    logfilename = "startingValuesForFits/%s.txt" % ( options.tag )
    write_parameters_logfile(fit_results, logfilename)


################################################################################
## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
################################################################################
if (not options.batch):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]


