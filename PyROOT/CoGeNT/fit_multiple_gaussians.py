#/usr/bin/env python

################################################################################
# Import the necessary libraries.
################################################################################
import sys
#import ROOT
#ROOT.PyConfig.IgnoreCommandLineOptions = True
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
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--e-bins', dest='e_bins', type=int, default=None,
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
    tmax = 480;
    # When plotting the time, use this for binning.
    tbins = 16;
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
    #myset.add(t)
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
            energy = amp_to_energy(amplitude,2)

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

    cogent_pars,cogent_sub_funcs,cogent_fit_pdf = multiple_gaussians(x,[10.0,10.5,10.1],[0.1,0.1,1.0],[1000,500,100],args.verbose)

    for c in cogent_sub_funcs:
        c.Print('V')

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
    xframe_main = x.frame(RooFit.Title("Ionization Energy"))
    data.plotOn(xframe_main)

    t.setBins(tbins)
    tframe_main = t.frame(RooFit.Title("Days since 12/4/2009"))
    #data.plotOn(tframe_main)

    ########################################################################
    # Make a histogram where we correct the time projection of the data for the
    # dead times.
    ########################################################################
    hacc_corr = TH1F("hacc_corr","hacc_corr",tbins,1.0,481)
    hacc_corr.Sumw2()
    nentries = data.numEntries()
    for i in xrange(nentries):
        argset = data.get(i)
        tmp = argset.getRealValue("t")
        correction = 1.0
        for c,m in zip(dead_time_correction_factor,month_ranges):
            if tmp>m[0] and tmp<=m[1]:
                correction = c
                '''
                if args.verbose:
                    print "Dead time corrections: %f %f %f %f" % (tmp,m[0],m[1],correction)
                '''

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

    cogent_pars_dict["gauss_means_0"].setVal(10.45); cogent_pars_dict["gauss_means_0"].setConstant(False)
    cogent_pars_dict["gauss_sigmas_0"].setVal(0.1); cogent_pars_dict["gauss_sigmas_0"].setConstant(False)
    cogent_pars_dict["gauss_norms_0"].setVal(100.0); cogent_pars_dict["gauss_norms_0"].setConstant(False)

    cogent_pars_dict["gauss_means_1"].setVal(10.31); cogent_pars_dict["gauss_means_1"].setConstant(False)
    cogent_pars_dict["gauss_sigmas_1"].setVal(0.1); cogent_pars_dict["gauss_sigmas_1"].setConstant(False)
    cogent_pars_dict["gauss_norms_1"].setVal(100.0); cogent_pars_dict["gauss_norms_1"].setConstant(False)

    cogent_pars_dict["gauss_means_2"].setVal(10.31); cogent_pars_dict["gauss_means_2"].setConstant(False)
    cogent_pars_dict["gauss_sigmas_2"].setVal(1.00); cogent_pars_dict["gauss_sigmas_2"].setConstant(False)
    cogent_pars_dict["gauss_norms_2"].setVal(100.0); cogent_pars_dict["gauss_norms_2"].setConstant(False)

    ########################################################################


    #exit(-1)
    ############################################################################
    # Construct the RooNLLVar to pass into RooMinuit. 
    # Because we have multiple ranges over which to fit, we need to construct
    # a list of RooNLLVar's.
    ############################################################################
    if args.verbose:
        print "Creating the NLL variable"

    nllList = RooArgSet()
    temp_list = []

    #exit(1)
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
    nll = RooAddition("nll","-log(likelihood)",nllList,True)

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
    #cogent_fit_pdf.plotOn(tframe_main,RooFit.Range(fit_range_tplot),RooFit.NormRange(fit_norm_range_tplot))

    # Plot the different components of the PDFs for the cosmogenic peaks.
    count = 0
    for s in cogent_sub_funcs_dict:
        argset = RooArgSet(cogent_sub_funcs_dict[s])
        plot_pdf = False
        line_style = 1
        color = 1
        if "cg_" in s:
            line_width = 1; line_style = 2;  color = 46;
            plot_pdf = True
        elif "gauss_pdf" in s:
            line_width = 2; line_style = 1;  color = 2;
            plot_pdf = True
        elif "flat_exp" in s and "exp_decay" not in s:
            line_width = 2; line_style = 1;  color = 6
            plot_pdf = True
        elif "exp_exp" in s and "exp_decay" not in s:
            line_width = 2; line_style = 1;  color = 3
            plot_pdf = True
        elif "exp2_exp" in s and "exp_decay" not in s:
            line_width = 2; line_style = 1;  color = 5
            plot_pdf = True


        if plot_pdf:
            cogent_fit_pdf.plotOn(xframe_main,RooFit.Components(argset),RooFit.LineWidth(line_width),RooFit.LineColor(color),RooFit.LineStyle(line_style),RooFit.Range(fit_range_xplot),RooFit.NormRange(fit_norm_range_xplot))
            #cogent_fit_pdf.plotOn(tframe_main,RooFit.Components(argset),RooFit.LineWidth(line_width),RooFit.LineColor(color),RooFit.LineStyle(line_style),RooFit.Range(fit_range_tplot),RooFit.NormRange(fit_norm_range_tplot))

    ########################################################################
    # Draw the frames onto the canvas.
    ########################################################################
    xmax = 3.2 
    if (args.e_hi>3.2):
        xmax = args.e_hi

    cans[0].cd(1)
    xframe_main.GetXaxis().SetLimits(0.5,xmax)
    #xframe_main.GetYaxis().SetRangeUser(0.0,95.0)
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
    if args.exp_mod:
        save_file_name += "_exp_mod"
    if args.flat_mod:
        save_file_name += "_flat_mod"
    if args.cg_mod:
        save_file_name += "_cg_mod"
    if args.add_gc:
        save_file_name += "_add_gc%d" % (args.gc_flag)
    if args.add_exp2:
        save_file_name += "_add_exp2"

    save_file_name += "_elo%d" % (int(10*args.e_lo))
    save_file_name += "_ehi%d" % (int(10*args.e_hi))

    for file_type in ['png','pdf','eps']:

        outfile = "Plots/%s.%s" % (save_file_name,file_type)
        cans[0].SaveAs(outfile)

    ########################################################################
    # Make a bunch of plots for a talk
    ########################################################################
    if args.talk_plots:
        num_talk_plots = 17
        cans_talk = []
        xframe_talk = x.frame(RooFit.Title("Ionization Energy"))
        tframe_talk = t.frame(RooFit.Title("Days since 12/4/2009"))
        data.plotOn(xframe_talk)
        data.plotOn(tframe_talk)
        for i in range(0,num_talk_plots):

            # Draw the cosmogenic peaks
            count_cg = 0
            for s in cogent_sub_funcs_dict:
                argset = RooArgSet(cogent_sub_funcs_dict[s])
                plot_pdf = False; line_style = 1; color = 1
                if "cg_" in s and i>2 and i>=count_cg+2:
                    line_width = 1; line_style = 2;  color = 46;
                    plot_pdf = True
                    count_cg += 1
                elif "cosmogenic_total" in s and i>=13:
                    line_width = 2; line_style = 1;  color = 2;
                    plot_pdf = True
                elif "flat_exp" in s and "exp_decay" not in s and i>=14:
                    line_width = 2; line_style = 1;  color = 6
                    plot_pdf = True
                elif "exp_exp" in s and "exp_decay" not in s and i>=15:
                    line_width = 2; line_style = 1;  color = 3
                    plot_pdf = True
                elif "exp2_exp" in s and "exp_decay" not in s and i>=15:
                    line_width = 2; line_style = 1;  color = 5
                    plot_pdf = True

                if plot_pdf:
                    cogent_fit_pdf.plotOn(xframe_talk,RooFit.Components(argset),RooFit.LineWidth(line_width),RooFit.LineColor(color),RooFit.LineStyle(line_style),RooFit.Range(fit_range_xplot),RooFit.NormRange(fit_norm_range_xplot))
                    cogent_fit_pdf.plotOn(tframe_talk,RooFit.Components(argset),RooFit.LineWidth(line_width),RooFit.LineColor(color),RooFit.LineStyle(line_style),RooFit.Range(fit_range_tplot),RooFit.NormRange(fit_norm_range_tplot))


            if i>=num_talk_plots-1:
                cogent_fit_pdf.plotOn(xframe_talk,RooFit.Range(fit_range_xplot),RooFit.NormRange(fit_norm_range_xplot))
                cogent_fit_pdf.plotOn(tframe_talk,RooFit.Range(fit_range_tplot),RooFit.NormRange(fit_norm_range_tplot))

            name = "cans_talk%d" % (i)
            cans_talk.append(TCanvas(name,name,100+10*i,100+10*i,1400,600))
            cans_talk[i].SetFillColor(0)
            cans_talk[i].Divide(2,1)

            ########################################################################
            # Draw the frames onto the canvas.
            ########################################################################
            cans_talk[i].cd(1)
            xframe_talk.GetXaxis().SetLimits(0.5,xmax)
            xframe_talk.GetYaxis().SetRangeUser(0.0,95.0)
            xframe_talk.Draw()
            gPad.Update()

            cans_talk[i].cd(2)
            tframe_talk.GetYaxis().SetRangeUser(0.0,200.0/(tbins/16.0) + 10)
            tframe_talk.Draw()
            if i>0:
                hacc_corr.Draw("samee") # The dead-time corrected histogram.
            gPad.Update()

            for file_type in ['png','pdf','eps']:

                talk_file_name = "%s_talk_%d" % (save_file_name,i)
                print "Printing %s" % (talk_file_name)
                outfile = "Plots/%s.%s" % (talk_file_name,file_type)
                cans_talk[i].SaveAs(outfile)

            ########################################################################

    ########################################################################
    # Dump out some more diagnostic info.
    ########################################################################
    if args.verbose:
        fit_results.correlationMatrix().Print("v")

    fit_results.Print("v")
    print "neg log likelihood: %f" % (fit_results.minNll())

    '''
    # Dump the phase info
    days = 0.0
    for i in range(0,3):
        phase = None
        phase_string = None
        if i==0:
            phase_string = "exp"
        elif i==1:
            phase_string = "flat"
        else:
            phase_string = "cg"
        name = "%s_mod_phase" % (phase_string)
        phase = cogent_pars_dict[name].getVal()
        if phase>=0:
            #days = 365 - (phase/(2*pi))*365 + (365/4.0)
            days = (-abs(degrees(0.66)/360.0) + 0.25 )*365.0
        else:
            days = (abs(degrees(phase)/360.0) + 0.25 )*365.0
            #days = (phase/(2*pi))*365 + (365/2.0)
        print "%s phase: %7.2f (rad) %3f (days)" % (phase_string, phase, days)
        # Convert phase peak to a day of the year.
        phase_peak = timedelta(days=int(days))
        phase_peak_date = start_date + phase_peak
        print phase_peak_date.strftime("\t\t%B %d, %Y")
    '''

    ############################################################################
    # Keep the gui alive unless batch mode or we hit the appropriate key.
    ############################################################################
    if not args.batch:
        rep = ''
        while not rep in ['q','Q']:
            rep = raw_input('enter "q" to quit: ')
            if 1<len(rep):
                rep = rep[0]

    del nll
    
################################################################################
################################################################################
if __name__ == "__main__":
    main()




