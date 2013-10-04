#!/usr/bin/env python

###############################################################
# run_a_set_of_fits.py
# Matt Bellis
# bellis@slac.stanford.edu
# Nov. 10, 2009
###############################################################

from nn_limits_and_binning import *
from file_map import *

import sys
from optparse import OptionParser

import shutil
from shutil import *

import subprocess

import time

MY_PYTHON = "python"

################################################################################
################################################################################
def main(argv):
    #### Command line variables ####
    parser = OptionParser()
    parser.add_option("--my-python", dest="my_python", default='python', 
        help='Which python should we use to execute the different steps.\n \
                (default is just \'python\')')
    parser.add_option("--step", dest="step", 
        help="Which step to run of entire fitting process (-1 for help)")
    parser.add_option("--baryon", dest="baryon",
        default='LambdaC', help="Which baryon? [LambdaC, Lambda0]")
    parser.add_option("--ntp", dest="ntp", 
        default='1', help="Which ntuple? [1-4]")
    parser.add_option("--pass", dest="my_pass", 
        default='0', help="Which pass? This will set many of the tags")
    parser.add_option("--pure", dest="pure", action="store_true", 
            default=False, help="Do pure toy MC studies.")
    parser.add_option("--embed", dest="embed", action="store_true", 
            default=False, help="Do embedded toy MC studies.")
    parser.add_option("--num-sig", dest="num_sig", 
            help="Number of signal events, embedded or otherwise.")
    parser.add_option("--fixed-num", dest="fixed_num", action="store_true", \
            default=False, help="Use a fixed number of both background and signal.")
    parser.add_option("--num-fits", dest="num_fits", 
            default=1, help="Number of fits to run")
    parser.add_option("-d", "--dimensionality", dest="dimensionality", \
            default=None, help="Dimensionality of fit [2,3]")
    parser.add_option("--sideband-first", dest="sideband_first", action = "store_true", 
            default = False, help="Keep stuff free and then fix it")
    parser.add_option("--no-gc", dest="no_gc", action = "store_true", \
            default=False, help="Don't use the gaussian constraint")
    parser.add_option("--batch", dest="batch", action = "store_true", 
            default = False, help="Run in batch mode")


    (options, args) = parser.parse_args()

    # Extra help options
    if options.step==None or int(options.step)==-1:
        parser.print_help()
        print "\n"
        print "Steps in fitting process:"
        print "\t0: Fit signal MC to extract shape paramters."
        print "\t1: Fit background MC to extract shape paramters."
        print "\t#### Combine these files by hand for the rest\n\
             of the starting value files."
        print "\t2: Generate background samples for MC studies."
        print "\t3: Embed signal events in samples for MC studies."
        print "\t4: Run multiple pure/embedded MC studies."
        print "\t\tFor this step you are *required* to specify a few things:"
        print "\t\t* # trials to run"
        print "\t\t* # signal events in study."
        print "\t\t\t (0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300)"
        print "\t\t* Pure or embedded studies."
        print "\t\t* Optional: side band first? fixed number of events?"
        print "\t5: Generate the output plots for one of the studies"
        print "\t6: Generate the summrary files for the studies"
        print "\n"

        exit(-1)

    step = int(options.step)
    baryon = options.baryon
    ntp = "ntp%s" % (options.ntp)
    my_pass = int(options.my_pass)
    global MY_PYTHON
    MY_PYTHON = options.my_python

    pass_info = fit_pass_info(baryon, ntp, my_pass)
    pass_tag = pass_info[2]
    start_file_tag = pass_info[7]
    # Dimensionality of fit
    dim = str(pass_info[9])
    #print pass_info
    print "run_a_set_of_fits: %s" % (dim)

    log_file_names = []
    cmds = []

    ############################################################################
    if step==0: 
        k = 0
        log_file_name, cmd = fit_MC_to_determine_parameters(baryon, ntp, "sig",\
                pass_info[k][0], my_pass, pass_info[k][1], pass_tag,\
                start_file_tag, dim)
        log_file_names.append(log_file_name)
        cmds.append(cmd)

    ############################################################################
    elif step==1: 
        k = 1
        log_file_name, cmd = fit_MC_to_determine_parameters(baryon, ntp, "bkg",\
                pass_info[k][0], my_pass, pass_info[k][1], pass_tag, \
                start_file_tag, dim)
        log_file_names.append(log_file_name)
        cmds.append(cmd)

    ############################################################################
    elif step==2: 
        nbkg = pass_info[3]
        # Generate from a Poisson distribution
        #for nsig in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for nsig in [0, 5, 10, 20, 30]:
            log_file_name, cmd = generate_bkg_for_toy_MC(baryon, ntp, my_pass,\
                    pass_tag, nbkg, nsig, 1000, False, dim)
            log_file_names.append(log_file_name)
            cmds.append(cmd)
        # Generate using a fixed number of signal and background
        '''
        for nsig in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,\
                30, 32, 34, 36]:
        '''
        for nsig in [0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
            log_file_name, cmd = generate_bkg_for_toy_MC(baryon, ntp, my_pass,\
                    pass_tag, nbkg, nsig, 1000, True, dim)
            log_file_names.append(log_file_name)
            cmds.append(cmd)

    ############################################################################
    elif step==3:
        nn_lo = pass_info[6][0]
        nn_hi = pass_info[6][1]
        nbkg = pass_info[3]
        # Embed from a Poisson distribution
        #for nsig in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for nsig in [0, 5, 10, 20, 30]:
            log_file_name, cmd = embed_signal_in_toy_MC(baryon, ntp, my_pass,\
                    pass_tag, nbkg, nsig, 1000)
            log_file_names.append(log_file_name)
            cmds.append(cmd)
        # Embed using a fixed number of signal and background
        '''
        for nsig in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,\
                30, 32, 34, 36]:
        '''
        for nsig in [0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
            log_file_name, cmd = embed_signal_in_toy_MC(baryon, ntp, my_pass,\
                    pass_tag, nbkg, nsig, 1000, True)
            log_file_names.append(log_file_name)
            cmds.append(cmd)

    ############################################################################
    elif step==4: 
        start_tag = pass_tag
        num_bkg = pass_info[3]
        num_sig = int(options.num_sig)
        num_studies = int(options.num_fits)

        # Check that the appropriate combination of pure or embedded flags
        # is set.
        if (not options.pure and not options.embed) or \
           (    options.pure and     options.embed):
            print ""
            print "For step 4 you *must* have --pure OR --embed to specify type\
                    of study!"
            print ""
            exit(-1)
        p_or_e_flag = ''
        if options.pure:
            p_or_e_flag = 'pure'
        elif options.embed:
            p_or_e_flag = 'embed'

        if not options.dimensionality == None:
            dim = options.dimensionality 
        log_file_name, cmd = run_many_toy_studies(baryon, ntp, my_pass,\
                start_tag, num_bkg, num_sig, num_studies,\
                p_or_e_flag, options.sideband_first, options.fixed_num, dim,\
                options.no_gc)
        cmds.append(cmd)
        log_file_names.append(log_file_name)

    ############################################################################
    elif step==5: 
        start_tag = pass_tag
        num_bkg = pass_info[3]
        num_sig = int(options.num_sig)
        num_studies = int(options.num_fits)

        # Check that the appropriate combination of pure or embedded flags
        # is set.
        if (not options.pure and not options.embed) or \
           (    options.pure and     options.embed):
            print ""
            print "For step 5 you *must* have --pure OR --embed to specify type of study!"
            print ""
            exit(-1)
        p_or_e_flag = ''
        if options.pure:
            p_or_e_flag = 'pure'
        elif options.embed:
            p_or_e_flag = 'embed'

        if not options.dimensionality == None:
            dim = options.dimensionality 
        log_file_name, cmd = extract_toy_study_results(baryon, ntp, my_pass, start_tag, num_bkg, num_sig, num_studies,\
                p_or_e_flag, options.sideband_first, options.fixed_num, dim, options.no_gc, options.batch)
        cmds.append(cmd)
        log_file_names.append(log_file_name)

    ############################################################################
    elif step==6: 
        start_tag = pass_tag
        num_bkg = pass_info[3]
        num_sig = int(options.num_sig)
        num_studies = int(options.num_fits)

        # Check that the appropriate combination of pure or embedded flags is set.
        if (not options.pure and not options.embed) or \
           (    options.pure and     options.embed):
            print ""
            print "For step 4 you *must* have --pure OR --embed to specify type of study!"
            print ""
            exit(-1)
        p_or_e_flag = ''
        if options.pure:
            p_or_e_flag = 'pure'
        elif options.embed:
            p_or_e_flag = 'embed'

        if not options.dimensionality == None:
            dim = options.dimensionality 
        log_file_name, cmd = summarize_many_toy_studies(baryon, ntp, start_tag, num_bkg, num_sig, num_studies,\
                p_or_e_flag, options.sideband_first, options.fixed_num, options.no_gc, dim)
        cmds.append(cmd)
        log_file_names.append(log_file_name)

    ############################################################################
    ############################################################################
    # Run the commands
    ############################################################################
    for i,cmd in enumerate(cmds):
        cmd_log_file = log_file_names[i]
        ############################################################################
        # Save the command in a log file
        ############################################################################
        cmd_log_file = open(log_file_name, "w+")
        cmd_string = ""
        for c in cmd:
            cmd_string += "%s " % (c)

        print cmd_string

        cmd_log_file.write(cmd_string)
        cmd_log_file.close()

        #print cmd
        ############################################################################
        # Run the command
        ############################################################################
        subprocess.Popen(cmd,0).wait()




################################################################################
################################################################################

def fit_MC_to_determine_parameters(baryon='LambdaC', ntp='ntp1', sb_flag="sig", start_tag=0, my_pass='0', num_events=1000, extra_tag=None, start_file_tag=None, dimensionality='3'):

    starting_vals_name = "startingValuesForFits/initial_%s_%s_%s_%d.txt" % (baryon, ntp, sb_flag, start_tag)

    fit_flag = "--fit-only-%s" % (sb_flag)

    out_tag = "determinedValues_%s_%s_%s_%s" % (baryon, ntp, sb_flag, extra_tag)

    #filetag = "generics"
    filetag = "genericSP"
    if sb_flag=="sig":
        #sp = ntp_to_sp(baryon, ntp)
        #filetag = "SP%s" % (sp)
        filetag = "signalSP"

    infile = 'textFiles/text_%s_%s_%s_%s_cut0.txt' % (baryon, ntp, start_file_tag, filetag)

    global MY_PYTHON
    print "MY_PYTHON: %s " % (MY_PYTHON)

    cmd = [ MY_PYTHON ]
    cmd += [ './read_in_and_fit.py' ]
    cmd += ['--num-bins', '100']
    cmd += ['--pass', str(my_pass)]
    cmd += [fit_flag] # This is either --fit-only-sig or --fit-only-bkg
    cmd += ['--starting-vals', starting_vals_name]
    cmd += ['--tag', out_tag]
    cmd += ['--baryon', baryon]
    cmd += ['--ntp', ntp]
    cmd += ['-m', str(num_events)]
    cmd += ['--fit']
    cmd += ['--batch']
    cmd += [infile]
    cmd += ['--dimensionality',dimensionality]

    now = int(time.mktime(time.localtime()))
    log_file_name = "logs/cmd_fMtdp_%s.log" % (now)

    return log_file_name, cmd

################################################################################
################################################################################
def generate_bkg_for_toy_MC(baryon='LambdaC', ntp='ntp1', my_pass='0', start_tag='pass0', num_bkg=100, num_sig=0, num_studies=1000, fixed_num=False, dim='3'):


    starting_vals_name = "startingValuesForFits/values_for_gen_pure_%s_%s_%s.txt" % (baryon, ntp, start_tag)

    out_tag = "mcstudy_%s_%s_%s" % (baryon, ntp, start_tag)

    workspace_file = 'rootWorkspaceFiles/workspace_determinedValues_%s_%s_sig_%s_nfits1.root' % (baryon, ntp, start_tag)

    global MY_PYTHON
    print "MY_PYTHON: %s " % (MY_PYTHON)

    cmd = [ MY_PYTHON ]
    cmd += [ './generate_bkg_for_toy_MC.py' ]
    cmd += ['--starting-vals-file', starting_vals_name]
    cmd += ['--workspace', workspace_file]
    cmd += ['--tag', out_tag]
    cmd += ['--baryon', baryon]
    cmd += ['--ntp', ntp]
    cmd += ['--dimensionality', dim]
    cmd += ['--pass', str(my_pass)]
    cmd += ['-N', str(num_studies)]
    cmd += ['-b', str(num_bkg)]
    cmd += ['-s', str(num_sig)]
    if fixed_num==True:
        cmd += ['--fixed-num']

    now = int(time.mktime(time.localtime()))
    log_file_name = "logs/cmd_gbftMC_%s.log" % (now)

    return log_file_name, cmd

################################################################################
################################################################################
################################################################################
def embed_signal_in_toy_MC(baryon='LambdaC', ntp='ntp1', my_pass='0', start_tag='pass0', num_bkg=100, num_sig=0, num_studies=1000, fixed_num=False):

    filetag = ""
    sp = ntp_to_sp(baryon, ntp)
    filetag = "SP%s" % (sp)

    sig_file = 'textFiles/text%s_%s_%s_newTMVA_cut2.txt' % (baryon, ntp, filetag)

    mc_bkg_tag = "mcstudy_%s_%s_%s" % (baryon, ntp, start_tag)

    global MY_PYTHON
    print "MY_PYTHON: %s " % (MY_PYTHON)

    cmd = [ MY_PYTHON ]
    cmd += [ './embed_signal_in_toy_MC.py' ]
    cmd += ['--sig-file', sig_file]
    cmd += ['--tag', mc_bkg_tag]
    cmd += ['-N', str(num_studies)]
    cmd += ['-b', str(num_bkg)]
    cmd += ['-s', str(num_sig)]
    cmd += ['--baryon', baryon]
    cmd += ['--ntp', ntp]
    cmd += ['--pass', str(my_pass)]
    if fixed_num==True:
        cmd += ['--fixed-num']

    now = int(time.mktime(time.localtime()))
    log_file_name = "logs/cmd_esitMC_%s.log" % (now)

    return log_file_name, cmd

################################################################################
################################################################################
################################################################################
def run_many_toy_studies(baryon='LambdaC', ntp='ntp1', my_pass='0', start_tag='pass0', num_bkg=100, num_sig=0, \
        num_studies=10, pure_or_embed='pure', sideband_first=False, fixed_num=False, dimensionality='3', no_gc=False):

    starting_vals_name = "startingValuesForFits/values_for_fits_%s_%s_%s.txt" % (baryon, ntp, start_tag)

    mc_dir = "mcstudy_%s_%s_%s" % (baryon, ntp, start_tag)

    p_or_e_flag = "--%s" % (pure_or_embed)

    workspace_file = "rootWorkspaceFiles/workspace_determinedValues_%s_%s_sig_%s_nfits1.root" % (baryon, ntp, start_tag)

    tag = "mcstudy_%s_%s_%s" % (baryon, ntp, start_tag)

    sideband_first_flag = ''
    if sideband_first:
        sideband_first_flag = '--sideband-first'
        tag += '_sideband_first'
    #if dimensionality!='3':
    #tag += "_dim%s" % (dimensionality)


    global MY_PYTHON
    print "MY_PYTHON: %s " % (MY_PYTHON)

    cmd = [ MY_PYTHON ]
    cmd += [ './read_in_and_fit.py' ]
    cmd += ['--starting-vals', starting_vals_name]
    cmd += ['--fit']
    cmd += ['--baryon', baryon]
    cmd += ['--ntp', ntp]
    cmd += ['--pass', str(my_pass)]
    cmd += ['--num-fits', str(num_studies)]
    cmd += ['--num-bkg', str(num_bkg)]
    cmd += ['--num-sig', str(num_sig)]
    cmd += [p_or_e_flag]
    cmd += ['--dir', mc_dir]
    cmd += ['--tag', tag]
    cmd += ['--dimensionality', str(dimensionality)]
    cmd += ['--batch']
    cmd += [sideband_first_flag]
    cmd += ['--workspace', workspace_file]
    if fixed_num:
        cmd += ['--fixed-num']
    if no_gc:
        cmd += ['--no-gc']

    now = int(time.mktime(time.localtime()))
    log_file_name = "logs/cmd_rmts_%s_%s.log" % (pure_or_embed, now)

    return log_file_name, cmd

################################################################################
################################################################################

################################################################################
def extract_toy_study_results(baryon='LambdaC', ntp='ntp1', my_pass='0', start_tag='pass0', num_bkg=100, num_sig=0, \
        num_studies=1000, pure_or_embed='pure', sideband_first=False, fixed_num=False, dimensionality='3', no_gc=False, batch=False):

    starting_vals_name = "startingValuesForFits/values_for_fits_%s_%s_%s.txt" % (baryon, ntp, start_tag)

    tag = "mcstudy_%s_%s_%s" % (baryon, ntp, start_tag)

    sideband_first_flag = ''
    if sideband_first:
        sideband_first_flag = '--sideband-first'
        tag += '_sideband_first'
    #if dimensionality!='3':

    tag += "_%s" % (pure_or_embed)
    tag += "_sig%d" % (num_sig)
    tag += "_bkg%d" % (num_bkg)
    if fixed_num:
        tag += "_fixedSig"
    tag += "_dim%s" % (dimensionality)
    if no_gc:
        tag += "_noGC"
    tag += "_nfits%d" % (num_studies)

    workspace_file = "rootWorkspaceFiles/workspace_%s.root" % (tag)

    global MY_PYTHON
    print "MY_PYTHON: %s " % (MY_PYTHON)

    cmd = [ MY_PYTHON ]
    cmd += [ './extract_toy_MC_results.py' ]
    cmd += ['--starting-vals', starting_vals_name]
    cmd += ['--results-file', workspace_file]
    cmd += ['--num-bkg', str(num_bkg)]
    cmd += ['--num-sig', str(num_sig)]
    cmd += ['--tag', tag]
    cmd += ['--baryon', baryon]
    cmd += ['--ntp', ntp]
    cmd += ['--pass', str(my_pass)]
    if no_gc:
        cmd += ['--no-gc']
    if batch:
        cmd += ['--batch']

    now = int(time.mktime(time.localtime()))
    log_file_name = "logs/cmd_%s_%s.log" % (pure_or_embed, now)

    return log_file_name, cmd

################################################################################
################################################################################

################################################################################
################################################################################
def summarize_many_toy_studies(baryon='LambdaC', ntp='ntp1', start_tag='pass0', \
        num_bkg=100, num_sig=0, num_studies=10, pure_or_embed='pure', \
        sideband_first=False, fixed_num=False, no_gc=False, dimensionality='3'):

    print "INNNNNNNNNNNTHHHHHHHEEEEEEEEEEEE DEF SUMMARIZE"
    print ntp
    
    mc_dir = "mcstudy_%s_%s_%s" % (baryon, ntp, start_tag)

    p_or_e_flag = "--%s" % (pure_or_embed)

    tag = "mcstudy_%s_%s_%s" % (baryon, ntp, start_tag)

    sideband_first_flag = ''
    if sideband_first:
        sideband_first_flag = '--sideband-first'
        tag += '_sideband_first'
    #if dimensionality!='3':
    #tag += "_dim%s" % (dimensionality)

    tag += "_%s" % (pure_or_embed)
    tag += "_sig%d" % (num_sig)
    tag += "_bkg%d" % (num_bkg)
    if fixed_num:
        tag += "_fixedSig"
    tag += "_dim%s" % (dimensionality)
    if no_gc:
        tag += "_noGC"
    tag += "_nfits%d" % (num_studies)

    workspace_file = "rootWorkspaceFiles/workspace_%s.root" % (tag)


    global MY_PYTHON
    print "MY_PYTHON: %s " % (MY_PYTHON)

    cmd = [ MY_PYTHON ]
    cmd += [ './summarize_toy_MC_studies.py' ]
    cmd += ['--dir', mc_dir]
    cmd += ['--num-bkg', str(num_bkg)]
    cmd += ['--num-sig', str(num_sig)]
    cmd += [p_or_e_flag]
    cmd += ['--tag', tag]
    cmd += ['--baryon', baryon]
    cmd += ['--ntp', ntp]
    cmd += ['--results-file', workspace_file]
    if fixed_num:
        cmd += ['--fixed-num']
    if no_gc:
        cmd += ['--no-gc']
    cmd += ['--batch']

    now = int(time.mktime(time.localtime()))
    log_file_name = "logs/cmd_rmts_%s_%s.log" % (pure_or_embed, now)

    return log_file_name, cmd

################################################################################
################################################################################

if __name__ == "__main__":
    main(sys.argv)

