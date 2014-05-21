#!/usr/bin/env python

from ROOT import *

###############################################################
# Background and signal definition
###############################################################

################################################################################
# Define our variables and ranges that we will use in this analysis.
################################################################################
def build_xyz(var_ranges=[[0.0,10.0],[0.0,10.0],[0.0,10.0]]):
    #################################
    # Build two PDFs
    #################################
    xlo = var_ranges[0][0]
    xhi = var_ranges[0][1]
    ylo = var_ranges[1][0]
    yhi = var_ranges[1][1]
    zlo = var_ranges[2][0]
    zhi = var_ranges[2][1]

    x = RooRealVar("x","x variable",xlo,xhi,"x units")
    y = RooRealVar("y","y variable",ylo,yhi,"y units")
    z = RooRealVar("z","z variable",zlo,zhi,"z units")

    return x,y,z

################################################################################
################################################################################

################################################################################
# Exponential
################################################################################
def exponential_func(var, tag):

    name = "c_%s_%s" % (var.GetName(), tag)
    title = "Exponential variable %s %s" % (var.GetName(), tag)
    c = RooRealVar(name,title, -1.0)

    name = "exp_%s_%s" % (var.GetName(), tag)
    title = "Exponential function %s %s" % (var.GetName(), tag)
    func = RooExponential(name, title, var, c)

    pars = [c]
    return pars, func

################################################################################
# Gaussian
################################################################################
def gaussian_func(var, tag):

    name = "mean_%s_%s" % (var.GetName(), tag)
    title = "Gaussian mean %s %s" % (var.GetName(), tag)
    mean = RooRealVar(name, title, 0.0)

    name = "sigma_%s_%s" % (var.GetName(), tag)
    title = "Gaussian sigma %s %s" % (var.GetName(), tag)
    sigma = RooRealVar(name, title, 1.0)

    func = RooGaussian(name, title, var, mean, sigma)

    pars = [mean, sigma]
    return pars, func


################################################################################
#########################################
# Signal
#########################################
################################################################################
def sig_PDF(vars):

    tag = "sig"
    pars = []
    funcs = []

    # Will need these for the constructor for the final function.
    func_string = ""
    rarglist = RooArgList();
    for i,v in enumerate(vars):
        p, f = gaussian_func(v,"sig")
        pars += p
        funcs.append(f)
        rarglist.add(f)

        if i==0:
            func_string += "%s" % (f.GetName())
        else:
            func_string += "*%s" % (f.GetName())

    sig_func = RooProdPdf("sig_pdf",func_string,rarglist)

    return pars, funcs, sig_func


#########################################
# Background
#########################################
def bkg_PDF(vars):

    tag = "bkg"
    pars = []
    funcs = []

    # Will need these for the constructor for the final function.
    func_string = ""
    rarglist = RooArgList();
    for i,v in enumerate(vars):
        p, f = exponential_func(v,"bkg")
        pars += p
        funcs.append(f)
        rarglist.add(f)

        if i==0:
            func_string += "%s" % (f.GetName())
        else:
            func_string += "*%s" % (f.GetName())

    bkg_func = RooProdPdf("bkg_pdf",func_string,rarglist)

    return pars, funcs, bkg_func

#############################################################
#############################################################
def tot_PDF(vars):

    pars_s, funcs_s, sig_pdf = sig_PDF(vars)
    pars_b, funcs_b, bkg_pdf = bkg_PDF(vars)

    nbkg = RooRealVar("nbkg","# bkg events",150)
    nsig = RooRealVar("nsig","# sig events",100)

    total = RooAddPdf("total","sig_pdf + bkg_pdf", RooArgList(sig_pdf, bkg_pdf), RooArgList(nsig, nbkg))

    pars = [nbkg, nsig]
    pars += pars_s
    pars += pars_b

    # Return all the sub-functions so that they stay active in the main program.
    funcs = [sig_pdf, bkg_pdf]
    funcs += funcs_s
    funcs += funcs_b

    return pars, funcs, total
#############################################################
#############################################################
