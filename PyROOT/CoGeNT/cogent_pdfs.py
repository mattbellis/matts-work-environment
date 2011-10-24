#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import *


################################################################################
# Build the PDFs for the cosmogenic peaks.
################################################################################
def cosmogenic_peaks(x,t,num_days,gc_flag=0,e_lo=None,verbose=False):

    ############################################################################
    # Hard coded this from the data given out by Juan Collar.
    ############################################################################
    cosmogenic_data_dict = {
            "As 73": [125.45,33.479,0.11000,13.799,12.741,1.4143,0.077656,80.000,0.0000],
            "Ge 68": [6070.7,1.3508,0.11400,692.06,638.98,1.2977,0.077008,271.00,0.0000],
            "Ga 68": [520.15,5.1139,0.11000,57.217,52.828,1.1936,0.076426,271.00,0.0000],
            "Zn 65": [2117.8,2.2287,0.10800,228.72,211.18,1.0961,0.075877,244.00,0.0058000],
            "Ni 56": [16.200,23.457,0.10200,1.6524,1.5257,0.92560,0.074906,5.9000,0.39000],
            "Co 56/58": [100.25,8.0,0.10200,10.226,9.4412,0.84610,0.074449,71.000,0.78600],
            "Co 57": [27.500,8.0,0.10200,2.8050,2.5899,0.84610,0.074449,271.00,0.78600],
            "Fe 55": [459.20,11.629,0.10600,48.675,44.942,0.76900,0.074003,996.00,0.96000],
            "Mn 54": [223.90,9.3345,0.10200,22.838,21.086,0.69460,0.073570,312.00,1.0000],
            "Cr 51": [31.500,15.238,0.10100,3.1815,2.9375,0.62820,0.073182,28.000,1.0000],
            "V 49": [161.46,12.263,0.10000,16.146,14.908,0.56370,0.072803,330.00,1.0000],
            }

    cosmogenic_means = []
    cosmogenic_sigmas = []
    cosmogenic_gaussians = []
    cosmogenic_norms = []
    cosmogenic_norms_calc = []
    cosmogenic_decay_constants = []
    cosmogenic_exp_decay_pdfs = []
    cosmogenic_decay_pdfs = []
    cosmogenic_N0 = []
    cosmogenic_pdfs = []
    cosmogenic_uncertainties = []

    cosmogenic_gaussian_constraints = []
    cosmogenic_gaussian_constraints_formula = []

    pars = []
    sub_funcs = []

    rooadd_string = ""
    rooadd_funcs = RooArgList()
    rooadd_norms = RooArgList()
    rooadd_norms_calc = RooArgList()

    total_num_cosmogenics = 0.0

    ncos_formula = ""
    ncos_formula_list = RooArgList()

    ############################################################################
    # Create a modulation term for the cosmogenic peaks.
    ############################################################################
    cg_mod_frequency = RooRealVar("cg_mod_frequency","Cosmogenic peak modulation frequency",1.0)
    cg_mod_offset = RooRealVar("cg_mod_offset","Cosmogenic peak modulation offset",2)
    cg_mod_phase = RooRealVar("cg_mod_phase","Cosmogenic peak modulation phase",0.0)
    cg_mod_amp = RooRealVar("cg_mod_amp","Cosmogenic peak modulation amplitude",0.0)

    cosmogenic_mod_t = RooGenericPdf("cosmogenic_mod_t","Cosmogenic peak modulation","cg_mod_offset+cg_mod_amp*sin((cg_mod_frequency*t) + cg_mod_phase)",RooArgList(cg_mod_offset,cg_mod_amp,cg_mod_frequency,cg_mod_phase,t)) ;

    ############################################################################

    n_good_cosmo = 0
    for i,p in enumerate(cosmogenic_data_dict):

        mean = cosmogenic_data_dict[p][5]
        sigma = cosmogenic_data_dict[p][6]

        half_life = cosmogenic_data_dict[p][7]
        #half_life = 2.0

        decay_constant = -1.0*log(2)/half_life
        #decay_constant = -0.001

        ########################################################################
        # Note the the value stored in the file/dictionary for the number of atoms,
        # is for (I think) the number of atoms expected to decay from Dec 4th, 'til
        # the end of time. So compensate for the number of days running.
        ########################################################################
        num_tot_decays = cosmogenic_data_dict[p][4]
        norm = num_tot_decays*(1.0-exp(num_days*decay_constant))

        # Check to see if the the low range of the energy has shifted. If so, 
        # we need to compensate for the truncation of the PDF (Gaussian)
        if e_lo is not None:
            func = "Gaus(x,%f,%f)" % (mean, sigma)
            g = TF1("Test Gaussian",func,0,3.0)
            frac = g.Integral(e_lo,3.0)/g.Integral(0.0,3.0)
            norm *= frac

            if verbose:
                print "frac: %f" % (frac)

        use_this_cosmo_peak = True
        if norm<0.1:
            use_this_cosmo_peak = False
            norm = 0.0

        if use_this_cosmo_peak:
            total_num_cosmogenics  += norm

            ########################################################################
            # Define the Gaussian constraints using the uncertainty
            ########################################################################
            uncert = 1.0
            uncert_from_file = norm*cosmogenic_data_dict[p][1]/100.0
            if gc_flag==0: # Uncertainty from file (CoGeNT study)
                uncert = uncert_from_file
                # For debugging
                #uncert = 0.01*norm
            elif gc_flag==1: # Sqrt(N)
                uncert = sqrt(norm)
            elif gc_flag==2: #Both terms added in quadrature
                uncert = sqrt(norm + uncert_from_file*uncert_from_file)

            if verbose:
                print "norm: %6.3f %6.3f %6.3f %6.3f +/-%6.3f" % (mean,sigma,num_tot_decays,norm,uncert)

            name = "cosmogenic_uncertainties_%s" % (i)
            cosmogenic_uncertainties.append(RooRealVar(name,name,uncert))

            ########################################################################
            # Define the Gaussian peaks
            ########################################################################
            name = "cosmogenic_means_%s" % (i)
            cosmogenic_means.append(RooRealVar(name,name,mean,0.5,1.6))

            name = "cosmogenic_sigmas_%s" % (i)
            cosmogenic_sigmas.append(RooRealVar(name,name,sigma,0.0,1.0))

            name = "cg_%s" % (i)
            cosmogenic_gaussians.append(RooGaussian(name,name,x,cosmogenic_means[n_good_cosmo],cosmogenic_sigmas[n_good_cosmo]))

            ############################################################################
            # Define the exponential decay of the normalization term.
            ############################################################################

            name = "cosmogenic_decay_constants_%s" % (i)
            cosmogenic_decay_constants.append(RooRealVar(name,name,decay_constant))

            name = "cosmogenic_exp_decay_pdfs_%s" % (i)
            cosmogenic_exp_decay_pdfs.append(RooExponential(name,name,t,cosmogenic_decay_constants[n_good_cosmo]))

            name = "cosmogenic_decay_pdfs_%s" % (i)
            function = "cosmogenic_decay_pdfs_%s+cosmogenic_mod_t" % (i)
            cosmogenic_decay_pdfs.append(RooProdPdf(name,function,RooArgList(cosmogenic_exp_decay_pdfs[n_good_cosmo],cosmogenic_mod_t)))

            # Use this if you don't want to add the modulation term.
            #name = "cosmogenic_decay_pdfs_%s" % (i)
            #cosmogenic_decay_pdfs.append(RooExponential(name,name,t,cosmogenic_decay_constants[n_good_cosmo]))

            ########################################################################
            # Define the normalization terms based on the number of expected events.
            ########################################################################
            name = "cosmogenic_norms_%s" % (i)
            cosmogenic_norms.append(RooRealVar(name,name,norm,0,1000))

            name = "cosmogenic_norms_calc_%s" % (i)
            cosmogenic_norms_calc.append(RooRealVar(name,name,norm,0,1000))

            ############################################################################
            # Define the Gaussian constraints. 
            ############################################################################
            name = "gaussian_constraint_%d" % (i)
            gc = RooFormulaVar(name,name,"((@0-@1)*(@0-@1))/(2*@2*@2)",RooArgList(cosmogenic_norms_calc[n_good_cosmo],cosmogenic_norms[n_good_cosmo],cosmogenic_uncertainties[n_good_cosmo]))
            cosmogenic_gaussian_constraints.append(gc)

            if verbose:
                gc.Print("v")

            ############################################################################
            # Build up the PDFs for each decay.
            ############################################################################
            name = "cosmogenic_pdfs_%s" % (i)
            cosmogenic_pdfs.append(RooProdPdf(name,name,RooArgList(cosmogenic_gaussians[n_good_cosmo],cosmogenic_decay_pdfs[n_good_cosmo])))
    
            # If there are more than a few events left in the peak, 
            # then use this decay for the total.
            if n_good_cosmo==0:
                rooadd_string = "%s" % (name)
            else:
                rooadd_string = "%s+%s" % (rooadd_string,name)

            rooadd_funcs.add(cosmogenic_pdfs[n_good_cosmo])
            rooadd_norms.add(cosmogenic_norms[n_good_cosmo])

            # Use this to calculate the number of events from the cosmogenic decays.
            if n_good_cosmo==0:
                ncos_formula += "@0"
            else:
                ncos_formula += "+@%d" % (n_good_cosmo)

            ncos_formula_list.add(cosmogenic_norms[n_good_cosmo])

            n_good_cosmo += 1


    # Add up all the individual peaks.
    # Form the total number of cosmogenics from the sum of the individual peaks.
    ncosmogenics = None
    cosmogenic_pdf = None
    if n_good_cosmo > 0:
        ncosmogenics = RooFormulaVar("ncosmogenics","ncosmogenics",ncos_formula,ncos_formula_list)
        cosmogenic_pdf = RooAddPdf("cosmogenic_total",rooadd_string,rooadd_funcs,rooadd_norms)
    else:
        ncosmogenics = RooFit.RooConst(0)
        cosmogenic_pdf = RooGenericPdf("cosmogenic_total","x",RooArgList(x))

    if verbose:
        ncosmogenics.Print()

    pars += cosmogenic_means
    pars += cosmogenic_sigmas
    pars += cosmogenic_norms
    pars += cosmogenic_norms_calc
    pars += cosmogenic_decay_constants 
    pars += cosmogenic_uncertainties
    pars += cosmogenic_gaussian_constraints

    sub_funcs += [ncosmogenics]
    sub_funcs += cosmogenic_gaussians
    sub_funcs += cosmogenic_pdfs
    sub_funcs += cosmogenic_exp_decay_pdfs 
    sub_funcs += cosmogenic_decay_pdfs 



    pars += [cg_mod_frequency,cg_mod_offset,cg_mod_phase,cg_mod_amp]
    sub_funcs += [cosmogenic_mod_t]

    ############################################################################

    print "total_num_cosmogenics: %f" % (total_num_cosmogenics)

    return pars, sub_funcs, cosmogenic_pdf


################################################################################

################################################################################
################################################################################
def cogent_pdf(x,t,gc_flag=0,e_lo=None,no_sig=False,no_cg=False,verbose=False):

    pars = []
    sub_funcs = []
    
    ############################################################################
    # Grab the cosmogenic peaks
    ############################################################################
    cosmogenic_pars, cosmogenic_sub_funcs, cosmogenic_pdf = cosmogenic_peaks(x,t,458,gc_flag,e_lo,verbose)
    ncosmogenics = None
    for c in cosmogenic_sub_funcs:
        if c.GetName()=="ncosmogenics":
            ncosmogenics = c

    pars += cosmogenic_pars
    sub_funcs += cosmogenic_sub_funcs
    sub_funcs += [cosmogenic_pdf]

    ############################################################################
    # Define the exponential background
    ############################################################################
    bkg_slope = RooRealVar("bkg_slope","Exponential slope of the background",-0.0,-10.0,0.0)
    bkg_exp_x = RooExponential("bkg_exp_x","Exponential PDF for bkg x",x,bkg_slope)

    bkg_slope_t = RooRealVar("bkg_slope_t","Exponential slope of the background t",0.0,-100.0,0.0)

    bkg_mod_frequency = RooRealVar("bkg_mod_frequency","Background modulation frequency",0.0)
    bkg_mod_offset = RooRealVar("bkg_mod_offset","Background modulation offset",2)
    bkg_mod_phase = RooRealVar("bkg_mod_phase","Background modulation phase",0.0)
    bkg_mod_amp = RooRealVar("bkg_mod_amp","Background modulation amplitude",1.0)

    bkg_exp_t = RooGenericPdf("bkg_exp_t","Background modulation","bkg_mod_offset+bkg_mod_amp*sin((bkg_mod_frequency*t) + bkg_mod_phase)",RooArgList(bkg_mod_offset,bkg_mod_amp,bkg_mod_frequency,bkg_mod_phase,t)) ;

    bkg_exp = RooProdPdf("bkg_exp","bkg_exp_x*bkg_exp_t",RooArgList(bkg_exp_x,bkg_exp_t))

    pars.append(bkg_mod_frequency)
    pars.append(bkg_mod_amp)
    pars.append(bkg_mod_phase)
    pars.append(bkg_mod_offset)

    pars.append(bkg_slope)
    pars.append(bkg_slope_t)
    sub_funcs.append(bkg_exp_x)
    sub_funcs.append(bkg_exp_t)
    sub_funcs.append(bkg_exp)

    ############################################################################
    # Define the exponential signal
    ############################################################################
    sig_slope = RooRealVar("sig_slope","Exponential slope of the signal",-4.5,-10.0,0.0)
    sig_exp_x = RooExponential("sig_exp_x","Exponential PDF for sig x",x,sig_slope)

    sig_slope_t = RooRealVar("sig_slope_t","Exponential slope of the signal t",-0.00001,-100.0,0.0)

    sig_mod_frequency = RooRealVar("sig_mod_frequency","Signal modulation frequency",0.00)
    sig_mod_offset = RooRealVar("sig_mod_offset","Signal modulation phase",2.0)
    sig_mod_phase = RooRealVar("sig_mod_phase","Signal modulation phase",0.0)
    sig_mod_amp = RooRealVar("sig_mod_amp","Signal modulation amp",1.0)

    sig_exp_t = RooGenericPdf("sig_exp_t","Signal modulation","sig_mod_offset+sig_mod_amp*sin((sig_mod_frequency*t) + sig_mod_phase)",RooArgList(sig_mod_offset,sig_mod_amp,sig_mod_frequency,sig_mod_phase,t)) ;

    sig_exp = RooProdPdf("sig_exp","sig_exp_x*sig_exp_t",RooArgList(sig_exp_x,sig_exp_t))

    pars.append(sig_mod_frequency)
    pars.append(sig_mod_amp)
    pars.append(sig_mod_phase)
    pars.append(sig_mod_offset)

    pars.append(sig_slope)
    pars.append(sig_slope_t)
    sub_funcs.append(sig_exp_x)
    sub_funcs.append(sig_exp_t)
    sub_funcs.append(sig_exp)

    ############################################################################
    # Form the total PDF.
    ############################################################################
    nbkg = RooRealVar("nbkg","nbkg",200,0,600000)
    nsig = RooRealVar("nsig","nsig",200,0,600000)

    total_pdf = None
    if no_sig and not no_cg:
        total_pdf = RooAddPdf("total_energy_pdf","bkg_exp+cosmogenic_pdf",RooArgList(bkg_exp,cosmogenic_pdf),RooArgList(nbkg,ncosmogenics))
    elif not no_sig and no_cg:
        total_pdf = RooAddPdf("total_energy_pdf","bkg_exp+sig_pdf",RooArgList(bkg_exp,sig_pdf),RooArgList(nbkg,nsig))
    elif no_sig and no_cg:
        total_pdf = RooAddPdf("total_energy_pdf","bkg_exp",RooArgList(bkg_exp),RooArgList(nbkg))
    else:
        total_pdf = RooAddPdf("total_energy_pdf","bkg_exp+sig_exp+cosmogenic_pdf",RooArgList(bkg_exp,sig_exp,cosmogenic_pdf),RooArgList(nbkg,nsig,ncosmogenics))

    pars += [nbkg, nsig]

    return pars,sub_funcs,total_pdf



