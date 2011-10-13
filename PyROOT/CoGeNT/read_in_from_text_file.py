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


################################################################################
################################################################################
def main():

    ############################################################################
    # Parse the command lines.
    ############################################################################
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_file_name', type=str, default=None, 
            help='Input file name')
    parser.add_argument('--add-gc', dest='add_gc', action='store_true', 
            default=False, help='Add a Gaussian constraint for the \
            uncertainties on the number of events in each cosmogenic peak.')
    parser.add_argument('--gc-flag', dest='gc_flag', type=int,
            default=2, help='Which Gaussian constraint.\n\
                    \t0: Errors on expected numbers of events.\n\
                    \t1: Sqrt(N) where N is expected number of events.\n\
                    \t2: Adding both in quadrature.\n\
                    \t Default: 2')
    parser.add_argument('--no-sig', dest='no_sig', action='store_true', 
            default=False, help="Don't have a signal component to the PDF.")
    parser.add_argument('--no-cg', dest='no_cg', action='store_true', 
            default=False, help="Don't have a cosmogenic component to the PDF.")
    parser.add_argument('--sig-mod', dest='sig_mod', action='store_true', 
            default=False, help='Let the signal have an annual modulation.')
    parser.add_argument('--bkg-mod', dest='bkg_mod', action='store_true', 
            default=False, help='Let the background have an annual modulation.')
    parser.add_argument('--cg-mod', dest='cg_mod', action='store_true', 
            default=False, help='Let the background have an annual modulation.')
    parser.add_argument('--e-lo', dest='e_lo', type=float, default=0.5,
            help='Set the lower limit for the energy range to use.')
    parser.add_argument('--e-hi', dest='e_hi', type=float, default=3.2,
            help='Set the upper limit for the energy range to use.')
    parser.add_argument('--e-bins', dest='e_bins', type=int, default=100,
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

    args = parser.parse_args()

    ############################################################################

    if args.input_file_name is None:
        print "Must pass in an input file name!"
        parser.print_help()

    infile_name = args.input_file_name
    ############################################################################

    ############################################################################
    # Print out some info so we can parse out a log file.
    ############################################################################
    print "INFO: e_lo %2.1f" % (args.e_lo)
    print "INFO: no_signal %d" % (args.no_sig)
    print "INFO: no_cosmogenic %d" % (args.no_cg)
    print "INFO: signal_modulation %d" % (args.sig_mod)
    print "INFO: background_modulation %d" % (args.bkg_mod)
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
    tmax = 480;
    # When plotting the time, use this for binning.
    tbins = 16;
    t_bin_width = tmax/tbins

    # Energy fitting range.
    lo_energy = args.e_lo
    hi_energy = args.e_hi
    xbins = args.e_bins
    ############################################################################

    ############################################################################
    # Define the variables and ranges
    # 
    # Note that the days start at 1 and not 0. 
    ############################################################################
    t = RooRealVar("t","time",1.0,tmax+1) # Add the 1 at the end because we 
                                          # start at 1.
    x = RooRealVar("x","ionization energy (keVee)",lo_energy,hi_energy);

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
    # Define the months that include the dead times.
    month_ranges = [[61,90], [91,120],[301,330],[451,481]]

    # Figure out the dead-time correction, just for plotting.
    dead_time_correction_factor = []
    for d in dead_days:

        # We've assumed 30-day binning for the dead-time correction.
        dead_time_correction_factor.append(30.0/((30.0-(d[1]-d[0]))))

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
            energy = amp_to_energy(amplitude,0)

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
    # Grab the PDF 
    ############################################################################

    cogent_pars,cogent_sub_funcs,cogent_fit_pdf = cogent_pdf(x,t,args.gc_flag,lo_energy,args.no_sig,args.no_cg,args.verbose)

    # DEBUG
    if args.verbose:
        print "Here's the PDF!"
        cogent_fit_pdf.Print("v")

    #exit(-1)

    ########################################################################
    # Make dictionaries for the pars and sub_funcs.
    ########################################################################
    cogent_pars_dict = {}
    for p in cogent_pars:

        if p.GetName().find("gaussian_constraint_")<0:

            cogent_pars_dict[p.GetName()] = p
            cogent_pars_dict[p.GetName()].setConstant(True)

        else:

            # Don't set the Gaussian constraints constant because they're
            # RooFormulaVars and can't be set like that
            cogent_pars_dict[p.GetName()] = p

    cogent_sub_funcs_dict = {}
    for p in cogent_sub_funcs:
        cogent_sub_funcs_dict[p.GetName()] = p

    ############################################################################
    # Make sure the data is in the live time of the experiment.
    # We'll make another dataset that has any events cut out that fall in the
    # recorded dead times of the experiment.
    ############################################################################

    data = data_total.reduce(RooFit.CutRange(good_ranges[0]))
    if args.verbose:

        print "fit   entries: %d" % (data.numEntries())

    for i in range(1,n_good_spots):

        data.append(data_total.reduce(RooFit.CutRange(good_ranges[i])))

        if args.verbose:
            print "fit   entries: %d" % (data.numEntries())

    # DEBUG
    if args.verbose:
        print "total entries: %d" % (data_total.numEntries())
        print "fit   entries: %d" % (data.numEntries())

    ############################################################################
    # Make frames on which to plot the data and fit results.
    ############################################################################
    x.setBins(xbins)
    xframe_main = x.frame(RooFit.Title("Plot of ionization energy"))
    data.plotOn(xframe_main)

    t.setBins(tbins)
    tframe_main = t.frame(RooFit.Title("Days"))
    data.plotOn(tframe_main)

    ########################################################################
    # Make a histogram where we correct the time projection of the data for the
    # dead times.
    ########################################################################
    hacc_corr = TH1F("hacc_corr","hacc_corr",tbins,1.0,481)
    nentries = data.numEntries()
    for i in xrange(nentries):
        argset = data.get(i)
        tmp = argset.getRealValue("t")
        correction = 1.0
        for c,m in zip(dead_time_correction_factor,month_ranges):
            if tmp>m[0] and tmp<=m[1]:
                correction = c
                if args.verbose:
                    print "Dead time corrections: %f %f %f %f" % (tmp,m[0],m[1],correction)

        hacc_corr.Fill(tmp,correction)

    hacc_corr.SetMarkerSize(0.8)
    hacc_corr.SetMarkerStyle(20)
    hacc_corr.SetMarkerColor(2)
    hacc_corr.SetLineColor(1)
    ########################################################################

    ############################################################################
    # Make canvases.
    ############################################################################
    cans = []
    for i in range(0,1):

        name = "cans%d" % (i)
        cans.append(TCanvas(name,name,100,100,1400,600))
        cans[i].SetFillColor(0)
        cans[i].Divide(2,1)

    ############################################################################
    # Set some of the parameters before we start the fit.
    ############################################################################

    cogent_pars_dict["nbkg"].setVal(700.0)
    cogent_pars_dict["nbkg"].setConstant(False)

    if not args.no_sig:
        cogent_pars_dict["nsig"].setVal(525.0)
        cogent_pars_dict["nsig"].setConstant(False)

        cogent_pars_dict["sig_slope"].setVal(-4.5)
        cogent_pars_dict["sig_slope"].setConstant(False)

    if args.add_gc:
        cogent_pars_dict["cosmogenic_norms_0"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_1"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_2"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_3"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_4"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_5"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_6"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_7"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_8"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_9"].setConstant(False)
        cogent_pars_dict["cosmogenic_norms_10"].setConstant(False)

    ########################################################################

    # Fix the modulation to have an annual frequency.
    yearly_mod = 2*pi/365.0
    cogent_pars_dict["sig_mod_frequency"].setVal(yearly_mod); cogent_pars_dict["sig_mod_frequency"].setConstant(True)
    cogent_pars_dict["bkg_mod_frequency"].setVal(yearly_mod); cogent_pars_dict["bkg_mod_frequency"].setConstant(True)
    cogent_pars_dict["cg_mod_frequency"].setVal(yearly_mod); cogent_pars_dict["cg_mod_frequency"].setConstant(True)

    # Let the signal modulate: float phase offset and amplitude.
    if args.sig_mod:
        cogent_pars_dict["sig_mod_phase"].setVal(0.0); cogent_pars_dict["sig_mod_phase"].setConstant(False)
        cogent_pars_dict["sig_mod_amp"].setVal(1.0); cogent_pars_dict["sig_mod_amp"].setConstant(False)
    else:
        cogent_pars_dict["sig_mod_phase"].setVal(0.0); cogent_pars_dict["sig_mod_phase"].setConstant(True)
        cogent_pars_dict["sig_mod_amp"].setVal(0.0); cogent_pars_dict["sig_mod_amp"].setConstant(True)

    # Let the background modulate: float phase offset and amplitude.
    if args.bkg_mod:
        cogent_pars_dict["bkg_mod_phase"].setVal(0.0); cogent_pars_dict["bkg_mod_phase"].setConstant(False)
        cogent_pars_dict["bkg_mod_amp"].setVal(1.0); cogent_pars_dict["bkg_mod_amp"].setConstant(False)
    else:
        cogent_pars_dict["bkg_mod_phase"].setVal(0.0); cogent_pars_dict["bkg_mod_phase"].setConstant(True)
        cogent_pars_dict["bkg_mod_amp"].setVal(0.0); cogent_pars_dict["bkg_mod_amp"].setConstant(True)

    if args.cg_mod:
        cogent_pars_dict["cg_mod_phase"].setVal(0.0); cogent_pars_dict["cg_mod_phase"].setConstant(False)
        cogent_pars_dict["cg_mod_amp"].setVal(1.0); cogent_pars_dict["cg_mod_amp"].setConstant(False)
    else:
        cogent_pars_dict["cg_mod_phase"].setVal(0.0); cogent_pars_dict["cg_mod_phase"].setConstant(True)
        cogent_pars_dict["cg_mod_amp"].setVal(0.0); cogent_pars_dict["cg_mod_amp"].setConstant(True)

    ############################################################################
    # Construct the RooNLLVar to pass into RooMinuit. 
    # Because we have multiple ranges over which to fit, we need to construct
    # a list of RooNLLVar's.
    ############################################################################
    if args.verbose:
        print "Creating the NLL variable"

    nllList = RooArgSet()
    temp_list = []

    # Loop over the ranges and create a RooNLLVar for each.
    for i,r in enumerate(good_ranges):

        name = "nll_%s" % (r)
        # Grabbed this constructor from the source code for RooAbsPdf.
        # Add them to a python list, just to keep them distinct.
        temp_list.append(RooNLLVar(name,name,cogent_fit_pdf,data,True,r,"",1,False,False,False,False))

        # Add them to a RooArgSet, for the object we'll pass to RooMinuit.
        nllList.add(temp_list[i])
        
        # DEBUG
        if args.verbose:
            nllList.Print("v")

    # Add in the Gaussian constraint
    gc_s = []
    if args.add_gc:

        # Pull out the Gaussian constraints from the dictionary with the
        # parameters.
        for c in cogent_pars_dict:

            name = "gaussian_constraint"
            if c.find(name)>=0:
                gc_s.append(c)
                nllList.add(cogent_pars_dict[c])

    ############################################################################
    # Create the neg log likelihood object to pass to RooMinuit
    ############################################################################
    nll = RooAddition("nll","-log(likelihood)",nllList,True);

    # DEBUG
    if args.verbose:
        nll.Print("v")

    fit_results = None

    # Call the fitting routine, based on our command line options.
    if not args.use_fitto:
        # Set up and call RooMinuit
        print "Using RooMinuit"
        m = RooMinuit(nll)

        m.setVerbose(False)
        m.migrad()
        m.hesse()
        fit_results = m.save()
    else:
        # Set up and call fitTo()
        print "Using the fitTo() member function"
        fit_results = cogent_fit_pdf.fitTo(data,
                RooFit.Range(fit_range),
                RooFit.Extended(True),
                RooFit.Save(True),
                )

    # There seem to be some subtlties with how the PDFs are plotted when there's not 
    # continuous fitting ranges, so set this up for some flexibility. 
    fit_range_xplot = "FULL"
    fit_range_tplot = fit_range
    fit_norm_range_xplot = "FULL"
    fit_norm_range_tplot = "FULL"

    # Plot the total PDF
    cogent_fit_pdf.plotOn(xframe_main,RooFit.Range(fit_range_xplot),RooFit.NormRange(fit_norm_range_xplot))
    cogent_fit_pdf.plotOn(tframe_main,RooFit.Range(fit_range_tplot),RooFit.NormRange(fit_norm_range_tplot))

    # Plot the different components of the PDFs for the cosmogenic peaks.
    count = 0
    for s in cogent_sub_funcs_dict:
        argset = RooArgSet(cogent_sub_funcs_dict[s])
        plot_pdf = False
        line_style = 1
        color = 1
        if "cg_" in s:
            line_width = 1; line_style = 2;  color = 2;
            plot_pdf = True
        elif "cosmogenic_total" in s:
            line_width = 2; line_style = 1;  color = 2;
            plot_pdf = True
        elif "bkg_exp" in s and "exp_decay" not in s:
            line_width = 2; line_style = 1;  color = 7
            plot_pdf = True
        elif "_exp" in s and "exp_decay" not in s:
            line_width = 2; line_style = 1;  color = 3
            plot_pdf = True

        if plot_pdf:
            cogent_fit_pdf.plotOn(xframe_main,RooFit.Components(argset),RooFit.LineWidth(line_width),RooFit.LineColor(color),RooFit.LineStyle(line_style),RooFit.Range(fit_range_xplot),RooFit.NormRange(fit_norm_range_xplot))
            cogent_fit_pdf.plotOn(tframe_main,RooFit.Components(argset),RooFit.LineWidth(line_width),RooFit.LineColor(color),RooFit.LineStyle(line_style),RooFit.Range(fit_range_tplot),RooFit.NormRange(fit_norm_range_tplot))

    ########################################################################
    # Draw the frames onto the canvas.
    ########################################################################
    cans[0].cd(1)
    xframe_main.GetXaxis().SetLimits(0.5,3.2)
    xframe_main.GetYaxis().SetRangeUser(0.0,95.0)
    xframe_main.Draw()
    gPad.Update()

    cans[0].cd(2)
    #tframe_main.GetYaxis().SetRangeUser(0.0,200.0/(tbins/16.0) + 10)
    tframe_main.Draw()
    hacc_corr.Draw("samee") # The dead-time corrected histogram.
    gPad.Update()

    ########################################################################


    ########################################################################
    # Save the canvas to a few different image formats.
    ########################################################################
    save_file_name = "%s" % args.tag
    if args.sig_mod:
        save_file_name += "_sig_mod"
    if args.bkg_mod:
        save_file_name += "_bkg_mod"
    if args.cg_mod:
        save_file_name += "_cg_mod"
    if args.add_gc:
        save_file_name += "_add_gc%d" % (args.gc_flag)

    save_file_name += "_elo%d" % (int(10*args.e_lo))

    for file_type in ['png','pdf','eps']:

        outfile = "Plots/%s.%s" % (save_file_name,file_type)
        cans[0].SaveAs(outfile)

    ########################################################################
    # Dump out some more diagnostic info.
    ########################################################################
    if args.verbose:
        fit_results.correlationMatrix().Print("v")

    fit_results.Print("v")
    print "neg log likelihood: %f" % (fit_results.minNll())

    # Dump the phase info
    days = 0.0
    for i in range(0,3):
        phase = None
        phase_string = None
        if i==0:
            phase_string = "sig"
        elif i==1:
            phase_string = "bkg"
        else:
            phase_string = "cg"
        name = "%s_mod_phase" % (phase_string)
        phase = cogent_pars_dict[name].getVal()
        if phase>=0:
            days = 365 - (phase/(2*pi))*365 + (365/4.0)
        else:
            days = (phase/(2*pi))*365 + (365/2.0)
        print "%s phase: %7.2f (rad) %3f (days)" % (phase_string, phase, days)
        # Convert phase peak to a day of the year.
        phase_peak = timedelta(days=int(days))
        phase_peak_date = start_date + phase_peak
        print phase_peak_date.strftime("%B %d, %Y")

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




