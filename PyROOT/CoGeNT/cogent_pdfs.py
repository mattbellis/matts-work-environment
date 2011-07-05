#!/usr/bin/env python

from ROOT import *


################################################################################
# Fit to some number of cosmogenic peak
################################################################################
def cosmogenic_peak():

    peak_means = [1.3, 1.0, 1.2, 1.4]
    peak_nums  = [100, 40,  8,   3]
    calib_means = []
    calib_sigmas = []
    calib_gaussians = []
    calib_norms = []

    rooadd_string = ""
    rooadd_funcs = RooArgList()
    rooadd_norms = RooArgList()

    for i,p in enumerate(peak_means):

        name = "calib_means_%s" % (i)
        calib_means.append(RooRealVar(name,name,p,0.5,1.6))

        name = "calib_sigmas_%s" % (i)
        calib_sigmas.append(RooRealVar(name,name,0.100,0.0,1.0))

        name = "calib_norms_%s" % (i)
        calib_norms.append(RooRealVar(name,name,peak_nums[i],0.0,10000.0))

        name = "cg_%s" % (i)
        calib_gaussians.append(RooLandau(name,name,x,calib_means[i],calib_sigmas[i]))

        if i==0:
            rooadd_string = "%s" % (name)
        else:
            rooadd_string = "%s+%s" % (rooadd_string,name)

        rooadd_funcs.add(calib_gaussians[i])
        rooadd_norms.add(calib_norms[i])

    name = "total_calib_peaks_%s" % (i)
    total_calib_peaks = RooAddPdf(name,rooadd_string,rooadd_funcs,rooadd_norms)

    pars = [c, dm, w, dw] + conv_pars
    return pars, conv_sub_funcs, func


################################################################################

################################################################################
################################################################################
def cogent_pdf():
    
    # Define the exponential background
    bkg_slope = RooRealVar("bkg_slope","Exponential slope of the background",-0.5,-10.0,0.0)
    bkg_exp = RooExponential("bkg_exp","Exponential PDF for bkg",x,bkg_slope)

    # Define the exponential signal
    sig_slope = RooRealVar("sig_slope","Exponential slope of the signal",-4.5,-10.0,0.0)
    sig_exp = RooExponential("sig_exp","Exponential PDF for sig",x,sig_slope)

    # Define the calibration peak
    #calib_mean = RooRealVar("calib_mean","Landau mean of the calibration peak",1.2,1.0,1.5);
    #calib_sigma = RooRealVar("calib_sigma","Landau sigma of the calibration peak",0.001,0.0,1.0)
    #calib_landau = RooLandau("calib_landau","Landau PDF for calibration peak",x,calib_mean,calib_sigma)


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
    lxg = RooFFTConvPdf("lxg","calib_landau (X) res_gaussian",x,calib_landau,res_gaussian)
    bxg = RooFFTConvPdf("bxg","bkg_exp (X) res_gaussian",x,bkg_exp,res_gaussian)
    sxg = RooFFTConvPdf("sxg","sig_exp (X) res_gaussian",x,sig_exp,res_gaussian)

    #sig_prod = RooProdPdf("sig_prod","sxg*sig_mod",RooArgList(sxg,sig_mod))
    sig_prod = RooProdPdf("sig_prod","sig_exp*sig_mod",RooArgList(sig_exp,sig_mod))
    #sig_prod = sxg
    #sig_prod = sig_exp

    ############################################################################
    # Form the total PDF.
    ############################################################################
    nbkg = RooRealVar("nbkg","nbkg",200,0,6000)
    #ncalib = RooRealVar("ncalib","ncalib",50,0,6000)
    ncalib = RooRealVar("ncalib","ncalib",200,0,6000)
    nsig = RooRealVar("nsig","nsig",200,0,6000)

    #total_pdf = RooAddPdf("total_pdf","bkg_exp+sig_exp+calib_landau",RooArgList(bkg_exp,sig_exp,calib_landau),RooArgList(nbkg,nsig,ncalib))
    #total_pdf = RooAddPdf("total_pdf","bxg+sxg+lxg",RooArgList(bxg,sxg,lxg),RooArgList(nbkg,nsig,ncalib))
    #total_pdf = RooAddPdf("total_pdf","bxg+sig_prod+lxg",RooArgList(bxg,sig_prod,lxg),RooArgList(nbkg,nsig,ncalib))
    #total_pdf = RooAddPdf("total_pdf","bxg+sig_prod+total_calib_peaks",RooArgList(bxg,sig_prod,total_calib_peaks),RooArgList(nbkg,nsig,ncalib))
    total_pdf = RooAddPdf("total_pdf","bkg_exp+sig_prod+total_calib_peaks",RooArgList(bkg_exp,sig_prod,total_calib_peaks),RooArgList(nbkg,nsig,ncalib))


    pars = [c, dm, w, dw] + conv_pars
    return pars, conv_sub_funcs, func




################################################################################
# Simple modulation
################################################################################
def simple_modulation(t):


    mod_off = RooRealVar("mod_off","Modulation offset",10)
    mod_phase = RooRealVar("mod_phase","Modulation phase",10)
    mod_freq = RooRealVar("mod_freq","Modulation frequency",10)
    mod_amp = RooRealVar("mod_amp","Modulation amplitude",10)

    #sig_mod = RooGenericPdf("sig_mod","2.0+sin(6.26*t/365.0)",RooArgList(t))
    sig_mod = RooGenericPdf("sig_mod","mod_off + mod_amp*sin(mod_freq*t + mod_phase)",RooArgList(t,mod_off,mod_phase,mod_freq,mod_amp))
    #sig_mod = RooGenericPdf("sig_mod","mod_amp*sin(mod_freq*t + mod_phase)",RooArgList(t,mod_phase,mod_freq,mod_amp))

    nsig = RooRealVar("nsig","nsig",200,0,6000)

    total_pdf = RooExtendPdf("total_pdf","Total PDF",sig_mod,nsig)

    #pars = [mod_off, mod_phase, mod_freq, mod_amp, nsig]
    pars = [mod_phase, mod_freq, mod_amp, mod_off, nsig]

    sub_pdfs = [sig_mod]

    return pars, sub_pdfs, total_pdf








