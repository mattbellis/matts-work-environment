#/usr/bin/env python

################################################################################
# Import the necessary libraries.
################################################################################
import sys
from ROOT import *

from math import *

from datetime import datetime,timedelta

import argparse

# These are my own.
from cogent_utilities import *
from cogent_pdfs import *
################################################################################

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

################################################################################
################################################################################
def main():

    ############################################################################
    # Parse the command lines.
    ############################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_name', type=str, default=None, 
            help='Input file name')
    parser.add_argument('--calib', dest='calib', type=int,
            default=0, help='Which calibration to use (0,1,2)')
    parser.add_argument('--add-gc', dest='add_gc', action='store_true', 
            default=False, help='Add a Gaussian constraint for the \
            uncertainties on the number of events in each cosmogenic peak.')
    parser.add_argument('--gc-flag', dest='gc_flag', type=int,
            default=2, help='Which Gaussian constraint.\n\
                    \t0: Errors on expected numbers of events.\n\
                    \t1: Sqrt(N) where N is expected number of events.\n\
                    \t2: Adding both in quadrature.\n\
                    \t Default: 2')
    parser.add_argument('--turn-off-eff', dest='turn_off_eff', action='store_true', 
            default=False, help="Don't use the efficiency function.")
    parser.add_argument('--no-exp', dest='no_exp', action='store_true', 
            default=False, help="Don't have a exponential component to the PDF.")
    parser.add_argument('--no-cg', dest='no_cg', action='store_true', 
            default=False, help="Don't have a cosmogenic component to the PDF.")
    parser.add_argument('--exp-mod', dest='exp_mod', action='store_true', 
            default=False, help='Let the exponential have an annual modulation.')
    parser.add_argument('--flat-mod', dest='flat_mod', action='store_true', 
            default=False, help='Let the flat term have an annual modulation.')
    parser.add_argument('--cg-mod', dest='cg_mod', action='store_true', 
            default=False, help='Let the flat term have an annual modulation.')
    parser.add_argument('--add-exp2', dest='add_exp2', action='store_true', 
            default=False, help='Add a second term that is exponential in energy.')
    parser.add_argument('--talk-plots', dest='talk_plots', action='store_true', 
            default=False, help='Make a bunch of plots for talks.')
    parser.add_argument('--e-lo', dest='e_lo', type=float, default=0.5,
            help='Set the lower limit for the energy range to use.')
    parser.add_argument('--e-hi', dest='e_hi', type=float, default=3.2,
            help='Set the upper limit for the energy range to use.')
    parser.add_argument('--e-bins', dest='e_bins', type=int, default=108,
            help='Set the number of bins to use for energy plotting.')
    parser.add_argument('--use-fitto', dest='use_fitto', action='store_true', 
            default=False, help='Use the RooAbsPdf::fitTo() member function, \
            rather than RooMinuit. For debugging purposes. This option will \
            fit without the Gaussian constraints.')
    parser.add_argument('--tag', dest='tag', default='cogent', 
            help='Tag to use to name output files and plots.')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', 
            default=False, help='Print out extra debug info.')
    parser.add_argument('--batch', '-b', dest='batch', action='store_true', 
            default=False, help='Run in batch mode.')
    parser.add_argument('--myhelp', dest='help', action='store_true', 
            default=False, help='Print help options.')

    args = parser.parse_args()

    ############################################################################

    if args.help:
        parser.print_help()
        exit(-1)

    if args.input_file_name is None:
        print "Must pass in an input file name!"
        parser.print_help()

    infile_name = args.input_file_name
    ############################################################################

    ############################################################################
    # Print out some info so we can parse out a log file.
    ############################################################################
    print "INFO: e_lo %2.1f" % (args.e_lo)
    print "INFO: e_hi %2.1f" % (args.e_hi)
    print "INFO: no_exponential %d" % (args.no_exp)
    print "INFO: no_cosmogenic %d" % (args.no_cg)
    print "INFO: exponential_modulation %d" % (args.exp_mod)
    print "INFO: flat_modulation %d" % (args.flat_mod)
    print "INFO: cosmogenic_modulation %d" % (args.cg_mod)
    print "INFO: add_gc %d" % (args.add_gc)
    print "INFO: gc_flag %d" % (args.gc_flag)

    ############################################################################
    # Add this to suppress a lot of the warnings from RooFit.
    ############################################################################
    RooMsgService.instance().Print()
    RooMsgService.instance().deleteStream(1)
    RooMsgService.instance().Print()
    ############################################################################

    ############################################################################
    # Put some configuration stuff here.
    ############################################################################
    # Time of the first event in seconds. We will 
    # need this for converting the times in the input file.
    first_event = 2750361.2 
    # First day of data recording.
    start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

    # Max day for plotting
    tmax = 458;
    # When plotting the time, use this for binning.
    tbins = 15;
    t_bin_width = tmax/tbins

    # Energy fitting range.
    lo_energy = args.e_lo
    hi_energy = args.e_hi
    xbins = int((hi_energy-lo_energy)/0.025) + 1
    if args.e_bins is not None:
        xbins = args.e_bins
    ############################################################################

    ############################################################################
    # Define the variables and ranges
    # 
    # Note that the days start at 1 and not 0. 
    ############################################################################
    t = RooRealVar("t","Time",1.0,tmax+1,"days") # Add the 1 at the end because we 
                                          # start at 1.
    x = RooRealVar("x","Ionization Energy",lo_energy,hi_energy,"keVee");

    # Define the FULL energy and time range for plotting later.
    x.setRange("FULL",lo_energy,hi_energy)
    t.setRange("FULL",1.0,tmax+1)
    
    myset = RooArgSet()
    myset.add(x)
    myset.add(t)
    data_total = RooDataSet("data_total","data_total",myset)
    #data_acc_corr = RooDataSet("data_acc_corr","data_acc_corr",myset)

    ############################################################################
    # Dead time days: 
    # 68-74
    # 102-107
    # 306-308
    ############################################################################
    dead_days = [[68,74], [102,107],[306,308]]
    #dead_days = [[68,69]]
    # Define the months that include the dead times.
    month_ranges = [[61,90], [91,120],[301,330],[451,481]]

    # Figure out the dead-time correction, just for plotting.
    dead_time_correction_factor = []
    for d in dead_days:

        # We've assumed 30-day binning for the dead-time correction.
        dead_time_correction_factor.append(30.0/((30.0-(d[1]-d[0] + 1))))

    # Add the last one by hand
    dead_time_correction_factor.append(30.0/8.0)

    # Number of ``good" ranges.
    n_good_spots = len(dead_days)+1

    good_ranges = []

    # Here we need to define named ranges for the different
    # live times.
    fit_range = ""
    for i in range(0,n_good_spots):

        name = "good_days_%d" % (i)
        good_ranges.append(name)
        if i<n_good_spots-1:
            fit_range += "%s," % (name)
        else:
            fit_range += "%s" % (name)


        if i==0:
            lo = 1
            hi = dead_days[i][0]
        elif i==n_good_spots-1:
            lo = dead_days[i-1][1]+1
            hi = 458+1
        else:
            lo = dead_days[i-1][1]+1
            hi = dead_days[i][0]

        if args.verbose:
            print "%s %d %d" % (name,lo,hi)

        t.setRange(name,lo,hi)

        ########################################################################
        # Don't forget to set the same names for ranges
        # in x (energy) which span the full range.
        ########################################################################
        x.setRange(name,lo_energy,hi_energy)

    if args.verbose:
        print "fit_range ---------------------- "
        print fit_range 
        print "fit_range ---------------------- "

    ############################################################################
    # Read in from a text file.
    ############################################################################
    infile = open(infile_name)

    for line in infile:
        
        vals = line.split()

        # Make sure there are at least two numbers on a line.
        if len(vals)==2:
            
            t_sec = float(vals[0])
            amplitude = float(vals[1])

            # Convert the amplitude to an energy using a particular calibration.
            energy = amp_to_energy(amplitude,args.calib)

            # Convert the time in seconds to a day.
            time_days = (t_sec-first_event)/(24.0*3600.0) + 1.0

            x.setVal(energy)
            t.setVal(time_days)

            if args.verbose:
                for d in dead_days:
                    if time_days>d[0] and time_days<d[1]:
                        print "Data in file is in dead time! %f" % (time_days)

            # Add the data to the dataset, only if it is in our energy range.
            if energy>=lo_energy and energy<=hi_energy:
                data_total.add(myset)

            # Sanity check to make sure there's not any weird times in the file.
            if time_days > 990:
                print "Time way out of bounds! %f" % (time_days)
                exit(0);

    ############################################################################

    
    ############################################################################
    # Make sure the data is in the live time of the experiment.
    # We'll make another dataset that has any events cut out that fall in the
    # recorded dead times of the experiment.
    ############################################################################

    data = data_total.reduce(RooFit.CutRange(good_ranges[0]))
    #if args.verbose>-1:
    if 1:

        print "fit   entries: %d" % (data.numEntries())

    for i in range(1,n_good_spots):

        data.append(data_total.reduce(RooFit.CutRange(good_ranges[i])))

        #if args.verbose>-1:
        if 1:
            print "fit   entries: %d" % (data.numEntries())

    # DEBUG
    #if args.verbose>-1:
    if 1:
        print "total entries: %d" % (data_total.numEntries())
        print "fit   entries: %d" % (data.numEntries())

    ############################################################################
    # Make frames on which to plot the data and fit results.
    ############################################################################
    x.setBins(xbins)
    xframe = x.frame(RooFit.Title("Ionization Energy"))
    data.plotOn(xframe)

    xframe_eff = x.frame(RooFit.Title("Efficiency"))

    t.setBins(tbins)
    tframe = t.frame(RooFit.Title("Days since 12/4/2009"))
    data.plotOn(tframe)

    ############################################################################
    # Try some PDF hypothesis by hand.
    ############################################################################

    # Acceptance state cut (1 or 0)
    cut = RooCategory("cut","cutr")
    cut.defineType("accept",1)
    cut.defineType("reject",0)

    # Efficiency
    max_eff = RooRealVar("max_eff","Maximum efficiency",0.86786)
    #max_eff = RooRealVar("max_eff","Maximum efficiency",1.0000)
    Ethresh = RooRealVar("Ethresh","E_{threshhold}",0.345)
    #Ethresh = RooRealVar("Ethresh","E_{threshhold}",0.045)
    sigma = RooRealVar("sigma","sigma",0.241)

    sigmoid = RooFormulaVar("sigmoid","sigmoid","@1/(1+exp((-(@0-@2)/(@3*@2))))",RooArgList(x,max_eff,Ethresh,sigma))

    # Surface events
    exp_slope = RooRealVar("exp_slope","Exponential slope of the exponential term",-3.36,-10.0,0.0)
    exp_pdf = RooExponential("exp_pdf","Exponential PDF for exp x",x,exp_slope)
    
    #surface_pdf = RooProdPdf("surface_pdf","exp_pdf*sigmoid",RooArgList(exp_pdf,sigmoid))
    eff_pdf = RooEfficiency ("eff_pdf","eff_pdf",sigmoid,cut,"accept") 
    surface_pdf = RooProdPdf("surface_pdf","surface_pdf",exp_pdf,eff_pdf)

    # Flat term
    # Construct a flat p.d.f (polynomial of 0th order)
    poly = RooPolynomial("poly","poly(x)",x)
    flat_pdf = RooProdPdf("flat_pdf","flat_pdf",poly,eff_pdf)

    mc_surf = surface_pdf.generate(RooArgSet(x),575)
    #mc_flat = flat_pdf.generate(RooArgSet(x),1041)
    mc_flat = flat_pdf.generate(RooArgSet(x),793)

    mc = mc_surf
    mc.append(mc_flat)

    mc.plotOn(xframe,RooFit.MarkerColor(2))

    #flat_pdf.plotOn(xframe,RooFit.LineColor(3))
    #surface_pdf.plotOn(xframe,RooFit.LineColor(4))

    ############################################################################
    # Make canvases.
    ############################################################################
    cans = []
    for i in range(0,1):

        name = "cans%d" % (i)
        #cans.append(TCanvas(name,name,100,100,1400,600))
        cans.append(TCanvas(name,name,10+10*i,10+10*i,1200,550))
        cans[i].SetFillColor(0)
        cans[i].Divide(2,1)

    cans[0].cd(1)
    xframe.GetXaxis().SetLimits(0.5,3.2)
    xframe.GetYaxis().SetRangeUser(0.0,95.0)
    xframe.Draw()
    gPad.Update()

    cans[0].cd(2)
    #tframe.GetYaxis().SetRangeUser(0.0,200.0/(tbins/15.0) + 10)
    tframe.Draw()
    #hacc_corr.Draw("samee") # The dead-time corrected histogram.
    gPad.Update()


    ############################################################################
    # Keep the gui alive unless batch mode or we hit the appropriate key.
    ############################################################################
    if not args.batch:
        rep = ''
        while not rep in ['q','Q']:
            rep = raw_input('enter "q" to quit: ')
            if 1<len(rep):
                rep = rep[0]

################################################################################
################################################################################
if __name__ == "__main__":
    main()




