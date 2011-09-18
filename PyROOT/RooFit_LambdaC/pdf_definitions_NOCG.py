#!/usr/bin/env python

from ROOT import *

###############################################################
# Background and signal definition
###############################################################

################################################################################
################################################################################
def build_xyz(var_ranges=[[5.2,5.3],[-0.2,0.2],[0.75,1.0]]):
    #################################
    # Build two PDFs
    #################################
    xlo = var_ranges[0][0]
    xhi = var_ranges[0][1]
    ylo = var_ranges[1][0]
    yhi = var_ranges[1][1]
    zlo = var_ranges[2][0]
    zhi = var_ranges[2][1]

    x = RooRealVar("x","m_{ES}",xlo, xhi)
    y = RooRealVar("y","#Delta E",ylo,yhi)
    z = RooRealVar("z","Shape/NN output",zlo,zhi)

    return x,y,z

################################################################################
################################################################################

################################################################################
# Crystal Barrel function: mES
################################################################################
def crystal_barrel_x(x):
    meanCB = RooRealVar("meanCB","Gaussian #mu (CB) m_{ES}", 5.279)
    sigmaCB = RooRealVar("sigmaCB"," Gaussian #sigma (CB) m_{ES}", 0.0028)
    alphaCB = RooRealVar("alphaCB", "#alpha (CB) m_{ES}", 2.0)
    nCB = RooRealVar("nCB","n of CB", 1.0)

    cb =     RooCBShape("CB", "Crystal Barrel Shape PDF", x, meanCB, sigmaCB, alphaCB, nCB)

    pars = [meanCB,  sigmaCB, alphaCB, nCB]
    return pars, cb

################################################################################
# Crystal Barrel function: DeltaE
################################################################################
def crystal_barrel_y(y):
    meanCBdE = RooRealVar("meanCBdE","Gaussian #mu (CB) #Delta E", 0.00)
    sigmaCBdE = RooRealVar("sigmaCBdE","Gaussian #mu (CB) #Delta E", 0.020)
    alphaCBdE = RooRealVar("alphaCBdE", "#alpha (CB) #Delta E", 2.0)
    nCBdE = RooRealVar("nCBdE","n of CBdE", 1.0)

    # Second CB function: DeltaE
    sigmaCBdE_2 = RooRealVar("sigmaCBdE_2","Gaussian #mu (CB) #Delta E (2)", 0.020)
    alphaCBdE_2 = RooRealVar("alphaCBdE_2", "#alpha (CB) #Delta E (2)", 2.0)
    nCBdE_2 = RooRealVar("nCBdE_2","n of CBdE_2", 1.0)

    cbdE =   RooCBShape("CBdE", "Crystal Barrel Shape PDF: DeltaE", y, meanCBdE, sigmaCBdE, alphaCBdE, nCBdE)
    cbdE_2 = RooCBShape("CBdE_2", "Crystal Barrel Shape PDF (2)", y, meanCBdE, sigmaCBdE_2, alphaCBdE_2, nCBdE_2)

    pars = [meanCBdE, sigmaCBdE, alphaCBdE, nCBdE, sigmaCBdE_2, alphaCBdE_2, nCBdE_2]
    return pars, cbdE, cbdE_2


################################################################################
# Double CB in dE
################################################################################
def double_cb_in_dE(cbdE, cbdE_2):

    ncbde1 = RooRealVar("ncbde1","# cbde1 events,",500, 0, 1000000)
    ncbde2 = RooRealVar("ncbde2","# cbde2 events",  50, 0, 1000000)
    double_cbdE = RooAddPdf("double_cbdE","CBdE + CBdE_2",RooArgList(cbdE, cbdE_2), RooArgList(ncbde2))

    pars = [ncbde2]

    return pars, double_cbdE 

################################################################################
################################################################################
################################################################################
# Background PDF
################################################################################
################################################################################
################################################################################
# Linear in y (background)
################################################################################
def linear_in_y(y):
    p1 = RooRealVar("poly1","Linear coefficient",-0.5) 
    rarglist = RooArgList(p1)
    polyy = RooPolynomial("polyy","Polynomial PDF",y, rarglist);

    pars = [p1]
    return pars, polyy 

################################################################################
# Argus background PDF
################################################################################
def argus_in_x(x):
    argpar = RooRealVar("argpar","Argus shape par",-20.0)
    cutoff = RooRealVar("cutoff","Argus cutoff",5.29)

    argus = RooArgusBG("argus","Argus PDF",x,cutoff,argpar)

    pars = [argpar, cutoff]
    return pars, argus 
################################################################################

################################################################################
# Argus in NN
################################################################################
def argus_in_z(z):
    argpar_NN = RooRealVar("argpar_NN","Argus shape par in NN",-7.0)
    cutoff_NN = RooRealVar("cutoff_NN","Argus cutoff in NN",0.995)

    argpar_NN.setConstant(kFALSE)
    cutoff_NN.setConstant(kFALSE)

    argus_NN = RooArgusBG("argus_NN","Argus NN PDF",z,cutoff_NN,argpar_NN)

    pars = [argpar_NN, cutoff_NN]
    return pars, argus_NN 
################################################################################

################################################################################
# BifurGaus in NN
################################################################################
def bifurgauss_in_z(z):
    mean_bfg = RooRealVar("mean_bfg","Mean of bfg",0.975)
    sigma_bfg_L = RooRealVar("sigma_bfg_L","Sigma L of bfg",0.50)
    sigma_bfg_R = RooRealVar("sigma_bfg_R","Sigma R of bfg",0.01)

    mean_bfg.setConstant(kFALSE)
    sigma_bfg_L.setConstant(kFALSE)
    sigma_bfg_R.setConstant(kFALSE)

    bfg = RooBifurGauss("bfg","BiFurGauss",z,mean_bfg,sigma_bfg_L,sigma_bfg_R)

    pars = [mean_bfg,sigma_bfg_L,sigma_bfg_R]
    return pars, bfg 
################################################################################

################################################################################
# Crystal Barrel function: NN (z)
################################################################################
def crystal_barrel_z(z):
    meanCB_NN = RooRealVar("meanCB_NN","Gaussian #mu (CB) NN", 0.98)
    sigmaCB_NN = RooRealVar("sigmaCB_NN"," Gaussian #sigma (CB) NN", 0.0028)
    alphaCB_NN = RooRealVar("alphaCB_NN", "#alpha (CB) NN", 2.0)
    nCB_NN = RooRealVar("nCB_NN","n of CB NN", 1.0)

    meanCB_NN.setConstant(kFALSE)
    sigmaCB_NN.setConstant(kFALSE)
    alphaCB_NN.setConstant(kFALSE)
    nCB_NN.setConstant(kTRUE)

    cb_NN =     RooCBShape("CB_NN", "Crystal Barrel Shape PDF NN", z, meanCB_NN, sigmaCB_NN, alphaCB_NN, nCB_NN)

    pars = [meanCB_NN,  sigmaCB_NN, alphaCB_NN, nCB_NN]
    return pars, cb_NN


###############################################################
# RooParametricStepFunction
###############################################################
def myRooKeys(z,data1):
    #############################

    #kest1 = RooKeysPdf("kest1","kest1",z,data1,RooKeysPdf.NoMirror, 0.5)
    kest1 = RooKeysPdf("kest1","kest1",z,data1,RooKeysPdf.NoMirror, 1.0)
    z.setBins(200, "cache")

    kc = RooCachedPdf("kc","kc",kest1)

    # RooDataHist
    rdh = kc.getCacheHist(RooArgSet(z))

    # RooHistPdf
    rhp = RooHistPdf("nn_sig", "nn_sig", RooArgSet(z), rdh)


    #return kest1
    #return [kest1], kc
    return [kest1, kc, rdh], rhp
    #return rhp



################################################################################
#########################################
# Signal
#########################################
################################################################################
def sig_PDF(x,y,z, dataset, dim = 2, use_double_CB=False, workspace=None):
    pars = []

    cb = None
    cbdE = None
    cbdE_2 = None
    double_cbdE = None
    nn_sig = None
    sig_prod = None

    funcs = []

    funcs0 = []

    pars_0, cb = crystal_barrel_x(x)
    pars_1, cbdE, cbdE_2 = crystal_barrel_y(y)

    if dim==2 and use_double_CB==False:
        sig_prod = RooProdPdf("sig_pdf","cb*cbdE",RooArgList(cb, cbdE)) 

    elif dim==2 and use_double_CB==True:
        pars_2, double_cbdE = double_cb_in_dE(cbdE, cbdE_2)
        sig_prod = RooProdPdf("sig_pdf","cb*double_cbdE",RooArgList(cb, double_cbdE)) 

    #'''
    elif dim==3 and use_double_CB==False:
        print "NOT SET UP TO USE THIS SET OF PDFS!!!!!!!!!!!"
        print "NOT SET UP TO USE THIS SET OF PDFS!!!!!!!!!!!"
        print "NOT SET UP TO USE THIS SET OF PDFS!!!!!!!!!!!"
        print "NOT SET UP TO USE THIS SET OF PDFS!!!!!!!!!!!"
        print "NOT SET UP TO USE THIS SET OF PDFS!!!!!!!!!!!"
        pars_2, rpsf_s = myRooParSF(z, bh, vary_limits, "sig", lo, hi)
        sig_prod =   RooProdPdf("sig_pdf","cb*cbdE*rpsf_s",RooArgList(cb, cbdE, rpsf_s)) 
    #''' 

    elif dim==3 and use_double_CB==True:
        pars_d, double_cbdE = double_cb_in_dE(cbdE, cbdE_2)
        # Here I'm going to try the RooKeysPdf
        if workspace==None:
            # Fit to a new KeysPdf
            funcs0, nn_sig = myRooKeys(z,dataset)
        else:
            # Read one in from a file
            nn_sig = workspace.pdf("nn_sig")
            funcs0 = []

        pars_2 = pars_d

        sig_prod = RooProdPdf("sig_pdf","cb*double_cbdE*nn_sig",RooArgList(cb, double_cbdE, nn_sig)) 


    pars += pars_0
    pars += pars_1
    pars += pars_2

    # Return all the sub-functions so that they stay active in the main program.
    if nn_sig!=None:
        funcs = [nn_sig, cb, cbdE, cbdE_2, double_cbdE, sig_prod]
    else:
        funcs = [cb, cbdE, cbdE_2, double_cbdE, sig_prod]

    funcs += funcs0

    return pars, funcs, sig_prod


#########################################
# Background
#########################################
# Multiply the components
def bkg_PDF(x,y,z, dataset, dim = 2):

    nn_bkg = None
    bkg_prod = None
    argus = None
    polyy = None
    pars = []

    pars_a, argus = argus_in_x(x)
    pars_b = []
    pars_p, polyy = linear_in_y(y)

    if dim==2:
        bkg_prod = RooProdPdf("bkg_pdf","argus*polyy", RooArgList(argus,polyy)) 
    elif dim==3:
        # Trying out an different analytic functions function
        #pars_b, nn_bkg = argus_in_z(z)
        #pars_b, nn_bkg = bifurgauss_in_z(z)
        pars_b, nn_bkg = crystal_barrel_z(z)
        bkg_prod = RooProdPdf("bkg_pdf","argus*polyy*nn_bkg", RooArgList(argus, polyy, nn_bkg)) 

    pars += pars_a
    pars += pars_p

    # Return all the sub-functions so that they stay active in the main program.
    funcs = [argus, polyy]

    # If 2D fit (only x and y)
    if len(pars_b)>0:
        pars += pars_b
        funcs = [nn_bkg, argus, polyy]

    return pars, funcs, bkg_prod


#############################################################
#############################################################
def tot_PDF(x,y,z, dataset, dim = 2, use_double_CB=False, workspace=None):
    funcs = []

    pars_s, funcs_s, sig_pdf = sig_PDF(x,y,z, dataset, dim, use_double_CB, workspace)
    pars_b, funcs_b, bkg_pdf = bkg_PDF(x,y,z, dataset, dim)

    nbkg = RooRealVar("nbkg","# bkg events,",150)
    nsig = RooRealVar ("nsig","# sig events",150)

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

#############################################################
#############################################################
