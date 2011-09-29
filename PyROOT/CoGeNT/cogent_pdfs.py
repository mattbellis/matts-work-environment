#!/usr/bin/env python

from ROOT import *


################################################################################
# Fit to some number of cosmogenic peak
################################################################################
def cosmogenic_peaks(x,t,num_days):

    cosmogenic_data_dict = {
            "As 73": [125.45,33.479,0.11000,13.799,12.741,1.4143,0.077656,80.000,0.0000],
            "Ge 68": [6070.7,1.3508,0.11400,692.06,638.98,1.2977,0.077008,271.00,0.0000],
            "Ga 68": [520.15,5.1139,0.11000,57.217,52.828,1.1936,0.076426,271.00,0.0000],
            "Zn 65": [2117.8,2.2287,0.10800,228.72,211.18,1.0961,0.075877,244.00,0.0058000],
            "Ni 56": [16.200,23.457,0.10200,1.6524,1.5257,0.92560,0.074906,5.9000,0.39000],
            "Co 56/58": [100.25,44.88,0.10200,10.226,9.4412,0.84610,0.074449,71.000,0.78600],
            "Co 57": [27.500,381.09,0.10200,2.8050,2.5899,0.84610,0.074449,271.00,0.78600],
            "Fe 55": [459.20,11.629,0.10600,48.675,44.942,0.76900,0.074003,996.00,0.96000],
            "Mn 54": [223.90,9.3345,0.10200,22.838,21.086,0.69460,0.073570,312.00,1.0000],
            "Cr 51": [31.500,15.238,0.10100,3.1815,2.9375,0.62820,0.073182,28.000,1.0000],
            "V 49": [161.46,12.263,0.10000,16.146,14.908,0.56370,0.072803,330.00,1.0000],
            }


    #peak_means = [1.3, 1.0, 1.2, 1.4]
    #peak_nums  = [50, 20,  8,   3]

    cosmogenic_means = []
    cosmogenic_sigmas = []
    cosmogenic_gaussians = []
    cosmogenic_norms = []
    cosmogenic_decay_constants = []
    cosmogenic_decay_pdfs = []
    cosmogenic_N0 = []
    cosmogenic_pdfs = []
    cosmogenic_uncertainties = []

    pars = []
    sub_funcs = []

    rooadd_string = ""
    rooadd_funcs = RooArgList()
    rooadd_norms = RooArgList()

    total_num_cosmogenics = 0.0

    for i,p in enumerate(cosmogenic_data_dict):

        mean = cosmogenic_data_dict[p][5]
        sigma = cosmogenic_data_dict[p][6]

        half_life = cosmogenic_data_dict[p][7]
        #half_life = 2.0

        decay_constant = -1.0*log(2)/half_life
        #decay_constant = -0.001

        # Note the the value stored in the file/dictionary for the number of atoms,
        # is for (I think) the number of atoms expected to decay from Dec 4th, 'til
        # the end of time. So compensate for the number of days running.
        num_tot_decays = cosmogenic_data_dict[p][4]
        norm = num_tot_decays*(1.0-exp(num_days*decay_constant))
        print "norm: %6.3f %6.3f %6.3f %6.3f" % (mean,sigma,num_tot_decays,norm)

        uncert = num_tot_decays*cosmogenic_data_dict[p][2]/100.0
        cosmogenic_uncertainties.append(uncert)

        total_num_cosmogenics  += norm

        ########################################################################
        # Define the Gaussian peaks
        ########################################################################
        name = "cosmogenic_means_%s" % (i)
        cosmogenic_means.append(RooRealVar(name,name,mean,0.5,1.6))

        name = "cosmogenic_sigmas_%s" % (i)
        cosmogenic_sigmas.append(RooRealVar(name,name,sigma,0.0,1.0))

        name = "cg_%s" % (i)
        cosmogenic_gaussians.append(RooGaussian(name,name,x,cosmogenic_means[i],cosmogenic_sigmas[i]))

        ############################################################################
        # Define the exponential decay of the normalization term.
        ############################################################################

        name = "cosmogenic_decay_constants_%s" % (i)
        cosmogenic_decay_constants.append(RooRealVar(name,name,decay_constant))

        name = "cosmogenic_decay_pdfs_%s" % (i)
        cosmogenic_decay_pdfs.append(RooExponential(name,name,t,cosmogenic_decay_constants[i]))

        name = "cosmogenic_pdfs_%s" % (i)
        cosmogenic_pdfs.append(RooProdPdf(name,name,RooArgList(cosmogenic_gaussians[i],cosmogenic_decay_pdfs[i])))

        name = "cosmogenic_norms_%s" % (i)
        cosmogenic_norms.append(RooRealVar(name,name,norm))

        if i==0:
            rooadd_string = "%s" % (name)
        else:
            rooadd_string = "%s+%s" % (rooadd_string,name)

        rooadd_funcs.add(cosmogenic_pdfs[i])
        rooadd_norms.add(cosmogenic_norms[i])

    #ncosmogenics_e = RooRealVar("ncosmogenics_e","ncosmogenics_e",50,0,600000)

    pars += cosmogenic_means
    pars += cosmogenic_sigmas
    pars += cosmogenic_norms
    pars += cosmogenic_decay_constants 

    sub_funcs += cosmogenic_gaussians
    sub_funcs += cosmogenic_pdfs
    sub_funcs += cosmogenic_decay_pdfs 

    name = "cg_total"
    cosmogenic_pdf = RooAddPdf(name,rooadd_string,rooadd_funcs,rooadd_norms)

    print "total_num_cosmogenics: %f" % (total_num_cosmogenics)

    return pars, sub_funcs, cosmogenic_pdf


################################################################################

################################################################################
################################################################################
def cogent_pdf(x,t):

    pars = []
    sub_funcs = []
    
    ############################################################################
    # Grab the cosmogenic peaks
    ############################################################################
    cosmogenic_pars, cosmogenic_sub_funcs, cosmogenic_pdf = cosmogenic_peaks(x,t,442)

    pars += cosmogenic_pars
    sub_funcs += cosmogenic_sub_funcs
    sub_funcs += [cosmogenic_pdf]

    ############################################################################
    # Define the exponential background
    ############################################################################
    bkg_slope = RooRealVar("bkg_slope","Exponential slope of the background",-0.0,-10.0,0.0)
    #bkg_exp = RooExponential("bkg_exp","Exponential PDF for bkg",x,bkg_slope)
    bkg_exp_x = RooExponential("bkg_exp_x","Exponential PDF for bkg x",x,bkg_slope)

    bkg_slope_t = RooRealVar("bkg_slope_t","Exponential slope of the background t",0.0,-100.0,0.0)
    #bkg_exp_t = RooExponential("bkg_exp_t","Exponential PDF for bkg t",t,bkg_slope_t)

    bkg_mod_frequency = RooRealVar("bkg_mod_frequency","Background modulation frequency",0.0)
    bkg_mod_offset = RooRealVar("bkg_mod_offset","Background modulation offset",2)
    bkg_mod_phase = RooRealVar("bkg_mod_phase","Background modulation phase",0.0)
    bkg_mod_amp = RooRealVar("bkg_mod_amp","Background modulation amplitude",1.0)

    bkg_exp_t = RooGenericPdf("bkg_exp_t","Background modulation","bkg_mod_offset+bkg_mod_amp*sin((bkg_mod_frequency*t) + bkg_mod_phase)",RooArgList(bkg_mod_offset,bkg_mod_amp,bkg_mod_frequency,bkg_mod_phase,t)) ;

    bkg_exp = RooProdPdf("bkg_exp","bkg_exp_x*bkg_exp_t",RooArgList(bkg_exp_x,bkg_exp_t))

    #pars.append(bkg_slope)
    #sub_funcs.append(bkg_exp)

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
    #sig_exp = RooExponential("sig_exp","Exponential PDF for sig",x,sig_slope)
    sig_exp_x = RooExponential("sig_exp_x","Exponential PDF for sig x",x,sig_slope)

    sig_slope_t = RooRealVar("sig_slope_t","Exponential slope of the signal t",-0.00001,-100.0,0.0)
    #sig_exp_t = RooExponential("sig_exp_t","Exponential PDF for sig t",t,sig_slope_t)

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

    '''
    ############################################################################
    # Set up the modulation terms.
    ############################################################################
    #sig_mod = RooExponential("sig_mod","Exponential PDF for mod",t,sig_slope)
    sig_mod = RooGenericPdf("sig_mod","2.0+sin(6.26*t/365.0)",RooArgList(t))

    # Define the resolution function to convolve with all of these
    res_mean =  RooRealVar("res_mean","Mean of the Gaussian resolution function",0)
    #res_sigma = RooRealVar("res_sigma","Sigma of the Gaussian resolution function",0.05)
    res_sigma = RooFormulaVar("res_sigma","0.10 + 0.05*sin(6.26*t/365.0)",RooArgList(t))
    res_gaussian = RooGaussModel("res_gaussian","Resolution function (Gaussian)",x,res_mean,res_sigma)

    # Construct the smeared functions (pdf (x) gauss)
    lxg = RooFFTConvPdf("lxg","cosmogenic_landau (X) res_gaussian",x,cosmogenic_landau,res_gaussian)
    bxg = RooFFTConvPdf("bxg","bkg_exp (X) res_gaussian",x,bkg_exp,res_gaussian)
    sxg = RooFFTConvPdf("sxg","sig_exp (X) res_gaussian",x,sig_exp,res_gaussian)

    #sig_prod = RooProdPdf("sig_prod","sxg*sig_mod",RooArgList(sxg,sig_mod))
    sig_prod = RooProdPdf("sig_prod","sig_exp*sig_mod",RooArgList(sig_exp,sig_mod))
    #sig_prod = sxg
    #sig_prod = sig_exp
    '''

    ############################################################################
    # Form the total PDF.
    ############################################################################
    nbkg_e = RooRealVar("nbkg_e","nbkg_e",200,0,600000)
    #ncosmogenic = RooRealVar("ncosmogenic","ncosmogenic",50,0,6000)
    ncosmogenics_e = RooRealVar("ncosmogenics_e","ncosmogenics_e",50,0,600000)
    nsig_e = RooRealVar("nsig_e","nsig_e",200,0,600000)

    #total_pdf = RooAddPdf("total_pdf","bkg_exp+sig_exp+cosmogenic_landau",RooArgList(bkg_exp,sig_exp,cosmogenic_landau),RooArgList(nbkg_e,nsig_e,ncosmogenics_e))
    #total_pdf = RooAddPdf("total_pdf","bxg+sxg+lxg",RooArgList(bxg,sxg,lxg),RooArgList(nbkg_e,nsig_e,ncosmogenics_e))
    #total_pdf = RooAddPdf("total_pdf","bxg+sig_prod+lxg",RooArgList(bxg,sig_prod,lxg),RooArgList(nbkg_e,nsig_e,ncosmogenics_e))
    #total_pdf = RooAddPdf("total_pdf","bxg+sig_prod+cosmogenic_pdf",RooArgList(bxg,sig_prod,cosmogenic_pdf),RooArgList(nbkg_e,nsig_e,ncosmogenics_e))

    #total_pdf = RooAddPdf("total_pdf","bkg_exp+sig_prod+cosmogenic_pdf",RooArgList(bkg_exp,sig_prod,cosmogenic_pdf),RooArgList(nbkg_e,nsig_e,ncosmogenics_e))

    total_energy_pdf = RooAddPdf("total_energy_pdf","bkg_exp+sig_exp+cosmogenic_pdf",RooArgList(bkg_exp,sig_exp,cosmogenic_pdf),RooArgList(nbkg_e,nsig_e,ncosmogenics_e))
    #total_energy_pdf = RooAddPdf("total_energy_pdf","bkg_exp+sig_exp",RooArgList(bkg_exp,sig_exp),RooArgList(nbkg_e,nsig_e))

    pars += [nbkg_e, nsig_e, ncosmogenics_e]

    return pars, sub_funcs, total_energy_pdf




################################################################################
# Simple modulation
################################################################################
def simple_modulation(t,tag=""):


    mod_off = RooRealVar("mod_off","Modulation offset",10,2.0,1000)
    mod_phase = RooRealVar("mod_phase","Modulation phase",10)
    mod_freq = RooRealVar("mod_freq","Modulation frequency",10)
    mod_amp = RooRealVar("mod_amp","Modulation amplitude",10,0,1000)

    #sig_mod = RooGenericPdf("sig_mod","2.0+sin(6.26*t/365.0)",RooArgList(t))
    sig_mod = RooGenericPdf("sig_mod","mod_amp*(mod_off + sin(mod_freq*t + mod_phase))",RooArgList(t,mod_phase,mod_freq,mod_off,mod_amp))
    #sig_mod = RooGenericPdf("sig_mod","mod_amp*sin(mod_freq*t + mod_phase)",RooArgList(t,mod_phase,mod_freq,mod_amp))

    nsig = RooRealVar("nsig","nsig",200,0,6000)

    total_pdf = RooExtendPdf("total_pdf","Total PDF",sig_mod,nsig)

    pars = [mod_off, mod_phase, mod_freq, mod_amp, nsig]
    #pars = [mod_phase, mod_freq, mod_amp, nsig]

    sub_pdfs = [sig_mod]

    return pars, sub_pdfs, total_pdf








