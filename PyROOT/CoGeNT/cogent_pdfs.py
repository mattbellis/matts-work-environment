#!/usr/bin/env python

#import ROOT
#ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import *


################################################################################
# Build the PDFs for the cosmogenic peaks.
################################################################################
def multiple_gaussians(x,means=[],sigmas=[],norms=[],verbose=False):

    npeaks = len(means)
    
    if npeaks!=len(sigmas) or npeaks!=len(norms):
        print "Different numbers of sigmas, means, and peaks!"
        exit(-1)

    gauss_means = []
    gauss_sigmas = []
    gauss_pdfs = []
    gauss_norms = []

    rooadd_string = ""
    rooadd_funcs = RooArgList()
    rooadd_norms = RooArgList()
    rooadd_norms_calc = RooArgList()

    for i in xrange(npeaks):

        m = means[i]
        s = sigmas[i]
        n = norms[i]

        ########################################################################
        # Define the Gaussian peaks
        ########################################################################
        name = "gauss_means_%s" % (i)
        gauss_means.append(RooRealVar(name,name,m,0.5,14.6))

        name = "gauss_sigmas_%s" % (i)
        gauss_sigmas.append(RooRealVar(name,name,s,0.0,1.0))

        name = "gauss_pdfs_%s" % (i)
        gauss_pdfs.append(RooGaussian(name,name,x,gauss_means[i],gauss_sigmas[i]))

        name = "gauss_norms_%s" % (i)
        gauss_norms.append(RooRealVar(name,name,n,0,100000))

        if i==0:
            rooadd_string = "%s" % (name)
        else:
            rooadd_string = "%s+%s" % (rooadd_string,name)

        rooadd_funcs.add(gauss_pdfs[i])
        rooadd_norms.add(gauss_norms[i])

    total_gauss_pdf = RooAddPdf("total_gauss_pdf",rooadd_string,rooadd_funcs,rooadd_norms)

    pars = gauss_means
    pars += gauss_sigmas
    pars += gauss_norms

    sub_funcs = gauss_pdfs

    return pars, sub_funcs, total_gauss_pdf

################################################################################
# Build the PDFs for the cosmogenic peaks.
################################################################################
def cosmogenic_peaks(x,t,num_days,gc_flag=0,e_lo=None,verbose=False):

    ############################################################################
    # Hard coded this from the data given out by Juan Collar.
    ############################################################################
    cosmogenic_data_dict = {
            "As 73": [125.45,33.479,0.11000,13.799,12.741,1.4143,0.077656,80.000,0.0000,11.10,80.0],
            "Ge 68": [6070.7,1.3508,0.11400,692.06,638.98,1.2977,0.077008,271.00,0.0000,10.37,271.0],
            "Ga 68": [520.15,5.1139,0.11000,57.217,52.828,1.1936,0.076426,271.00,0.0000,9.66,271.0],
            "Zn 65": [2117.8,2.2287,0.10800,228.72,211.18,1.0961,0.075877,244.00,0.0058000,8.98,244.0],
            "Ni 56": [16.200,23.457,0.10200,1.6524,1.5257,0.92560,0.074906,5.9000,0.39000,7.71,5.9],
            "Co 56/58": [100.25,8.0,0.10200,10.226,9.4412,0.84610,0.074449,71.000,0.78600,7.11,77.0],
            "Co 57": [27.500,8.0,0.10200,2.8050,2.5899,0.84610,0.074449,271.00,0.78600,7.11,271.0],
            "Fe 55": [459.20,11.629,0.10600,48.675,44.942,0.76900,0.074003,996.00,0.96000,6.54,996.45],
            "Mn 54": [223.90,9.3345,0.10200,22.838,21.086,0.69460,0.073570,312.00,1.0000,5.99,312.0],
            "Cr 51": [31.500,15.238,0.10100,3.1815,2.9375,0.62820,0.073182,28.000,1.0000,5.46,28.0],
            "V 49": [161.46,12.263,0.10000,16.146,14.908,0.56370,0.072803,330.00,1.0000,4.97,330.0],
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

        # L-shell peaks
        mean = cosmogenic_data_dict[p][5]
        sigma = cosmogenic_data_dict[p][6]
        # K-shell peaks
        #mean = cosmogenic_data_dict[p][9]
        #sigma = 1.5*cosmogenic_data_dict[p][6]

        half_life = cosmogenic_data_dict[p][7]
        #half_life = 2.0

        decay_constant = -1.0*log(2)/half_life
        #decay_constant = -0.001

        ########################################################################
        # Note the the value stored in the file/dictionary for the number of atoms,
        # is for (I think) the number of atoms expected to decay from Dec 4th, 'til
        # the end of time. So compensate for the number of days running.
        ########################################################################
        # L-shell peaks
        num_tot_decays = cosmogenic_data_dict[p][4]
        # K-shell peaks
        #num_tot_decays = cosmogenic_data_dict[p][0]

        norm = num_tot_decays*(1.0-exp(num_days*decay_constant))

        # Check to see if the the low range of the energy has shifted. If so, 
        # we need to compensate for the truncation of the PDF (Gaussian)
        #if e_lo is not None:
        if False:
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
def cogent_pdf(x,t,gc_flag=0,e_lo=None,no_exp=False,no_cg=False,add_exp2=False,verbose=False):

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
    # Define the flat term (we use an exponential for now)
    ############################################################################
    flat_slope = RooRealVar("flat_slope","Exponential slope of the flat term",-0.0,-10.0,0.0)
    flat_exp_x = RooExponential("flat_exp_x","Exponential PDF for flat x",x,flat_slope)

    flat_slope_t = RooRealVar("flat_slope_t","Exponential slope of the flat term t",0.0,-100.0,0.0)

    flat_mod_frequency = RooRealVar("flat_mod_frequency","Flat term modulation frequency",0.0)
    flat_mod_offset = RooRealVar("flat_mod_offset","Flat term modulation offset",2)
    flat_mod_phase = RooRealVar("flat_mod_phase","Flat term modulation phase",0.0)
    flat_mod_amp = RooRealVar("flat_mod_amp","Flat term modulation amplitude",1.0)

    flat_exp_t = RooGenericPdf("flat_exp_t","Flat term modulation","flat_mod_offset+flat_mod_amp*sin((flat_mod_frequency*t) + flat_mod_phase)",RooArgList(flat_mod_offset,flat_mod_amp,flat_mod_frequency,flat_mod_phase,t)) ;

    flat_exp = RooProdPdf("flat_exp","flat_exp_x*flat_exp_t",RooArgList(flat_exp_x,flat_exp_t))

    pars.append(flat_mod_frequency)
    pars.append(flat_mod_amp)
    pars.append(flat_mod_phase)
    pars.append(flat_mod_offset)

    pars.append(flat_slope)
    pars.append(flat_slope_t)
    sub_funcs.append(flat_exp_x)
    sub_funcs.append(flat_exp_t)
    sub_funcs.append(flat_exp)

    ############################################################################
    # Define the exponential term (in energy)
    ############################################################################
    exp_slope = RooRealVar("exp_slope","Exponential slope of the exponential term",-4.5,-10.0,0.0)
    exp_exp_x = RooExponential("exp_exp_x","Exponential PDF for exp x",x,exp_slope)

    exp_slope_t = RooRealVar("exp_slope_t","Exponential slope of the exponential term t",-0.00001,-100.0,0.0)

    exp_mod_frequency = RooRealVar("exp_mod_frequency","Exponential term modulation frequency",0.00)
    exp_mod_offset = RooRealVar("exp_mod_offset","Exponential term modulation phase",2.0)
    exp_mod_phase = RooRealVar("exp_mod_phase","Exponential term modulation phase",0.0)
    exp_mod_amp = RooRealVar("exp_mod_amp","Exponential term modulation amp",1.0)

    exp_exp_t = RooGenericPdf("exp_exp_t","Exponential term modulation","exp_mod_offset+exp_mod_amp*sin((exp_mod_frequency*t) + exp_mod_phase)",RooArgList(exp_mod_offset,exp_mod_amp,exp_mod_frequency,exp_mod_phase,t)) ;

    exp_exp = RooProdPdf("exp_exp","exp_exp_x*exp_exp_t",RooArgList(exp_exp_x,exp_exp_t))

    pars.append(exp_mod_frequency)
    pars.append(exp_mod_amp)
    pars.append(exp_mod_phase)
    pars.append(exp_mod_offset)

    pars.append(exp_slope)
    pars.append(exp_slope_t)
    sub_funcs.append(exp_exp_x)
    sub_funcs.append(exp_exp_t)
    sub_funcs.append(exp_exp)

    ############################################################################
    # Define a second exponential term (in energy)
    ############################################################################
    exp2_slope = RooRealVar("exp2_slope","Exponential slope of the exponential term",-4.5,-10.0,0.0)
    exp2_exp_x = RooExponential("exp2_exp_x","Exponential PDF for exp x",x,exp2_slope)

    exp2_slope_t = RooRealVar("exp2_slope_t","Exponential slope of the exponential term t",-0.00001,-100.0,0.0)
    #exp2_slope_t_calc = RooRealVar("exp2_slope_t_calc","Exponential slope of the exponential term t calc",-0.00001,-100.0,0.0)
    #exp2_slope_t_uncert = RooRealVar("exp2_slope_t_uncert","Uncertainty on the exponential slope of the 2nd exponential term",-0.00001,-100.0,0.0)

    exp2_mod_frequency = RooRealVar("exp2_mod_frequency","Exponential term modulation frequency",0.00)
    exp2_mod_offset = RooRealVar("exp2_mod_offset","Exponential term modulation phase",2.0)
    exp2_mod_phase = RooRealVar("exp2_mod_phase","Exponential term modulation phase",0.0)
    exp2_mod_amp = RooRealVar("exp2_mod_amp","Exponential term modulation amp",1.0)

    exp2_exp_t = RooGenericPdf("exp2_exp_t","Exponential term modulation","exp2_mod_offset+exp2_mod_amp*sin((exp2_mod_frequency*t) + exp2_mod_phase)",RooArgList(exp2_mod_offset,exp2_mod_amp,exp2_mod_frequency,exp2_mod_phase,t)) ;

    exp2_exp = RooProdPdf("exp2_exp","exp2_exp_x*exp2_exp_t",RooArgList(exp2_exp_x,exp2_exp_t))

    ############################################################################
    # Define the Gaussian constraints. 
    ############################################################################
    '''
    name = "gaussian_constraint_exp2_%d" % (i)
    gc_exp2 = RooFormulaVar(name,name,"((@0-@1)*(@0-@1))/(2*@2*@2)",RooArgList(exp2_slope_t,exp2_slope_t_calc,exp2_slope_t_uncertainty))

    if verbose:
        gc.Print("v")
    '''

    pars.append(exp2_mod_frequency)
    pars.append(exp2_mod_amp)
    pars.append(exp2_mod_phase)
    pars.append(exp2_mod_offset)

    pars.append(exp2_slope)
    pars.append(exp2_slope_t)

    sub_funcs.append(exp2_exp_x)
    sub_funcs.append(exp2_exp_t)
    sub_funcs.append(exp2_exp)
    #pars.append(exp2_slope_t_calc)
    #sub_funcs.append(gc_exp2)

    ############################################################################
    # Form the total PDF.
    ############################################################################
    nflat = RooRealVar("nflat","nflat",200,0,600000)
    nexp = RooRealVar("nexp","nexp",200,0,600000)
    nexp2 = RooRealVar("nexp2","nexp2",200,0,600000)

    total_pdf = None
    if no_exp and not no_cg:
        total_pdf = RooAddPdf("total_energy_pdf","flat_exp+cosmogenic_pdf",RooArgList(flat_exp,cosmogenic_pdf),RooArgList(nflat,ncosmogenics))
    elif add_exp2 and no_cg:
        total_pdf = RooAddPdf("total_energy_pdf","flat_exp+exp_exp+exp2_exp",RooArgList(flat_exp,exp_exp,exp2_exp),RooArgList(nflat,nexp,nexp2))
    elif not no_exp and no_cg:
        total_pdf = RooAddPdf("total_energy_pdf","flat_exp+exp_exp",RooArgList(flat_exp,exp_exp),RooArgList(nflat,nexp))
    elif no_exp and no_cg:
        total_pdf = RooAddPdf("total_energy_pdf","flat_exp",RooArgList(flat_exp),RooArgList(nflat))
    elif add_exp2:
        total_pdf = RooAddPdf("total_energy_pdf","flat_exp+exp_exp+exp2_exp+cosmogenic_pdf",RooArgList(flat_exp,exp_exp,exp2_exp,cosmogenic_pdf),RooArgList(nflat,nexp,nexp2,ncosmogenics))
    else:
        total_pdf = RooAddPdf("total_energy_pdf","flat_exp+exp_exp+cosmogenic_pdf",RooArgList(flat_exp,exp_exp,cosmogenic_pdf),RooArgList(nflat,nexp,ncosmogenics))

    pars += [nflat,nexp,nexp2]

    return pars,sub_funcs,total_pdf


################################################################################
# Efficiency pdf
################################################################################

def efficiency(x,t,verbose=False):

    pars = []
    sub_funcs = []
    
    #Etrig = [0.47278, 0.52254, 0.57231, 0.62207, 0.67184, 0.72159, 0.77134, 0.82116, 0.87091, 0.92066, 10.0] # Change to 10
    #efftrig = [0.71747, 0.77142, 0.81090, 0.83808, 0.85519, 0.86443, 0.86801, 0.86814, 0.86703, 0.86786, 0.86786]
    
    ######## Same as Nicole's
    Etrig = [0.47278, 0.52254, 0.57231, 0.62207, 0.67184, 0.72159, 0.77134, 0.82116, 0.87091, 0.92066, 4.0] # Changed to 4.0
    efftrig = [0.71747, 0.77142, 0.81090, 0.83808, 0.85519, 0.86443, 0.86801, 0.86814, 0.86703, 0.86786, 0.86786]
    
    # Looking for a more dramatic change
    #Etrig = [0.47278, 0.52254, 0.57231, 0.62207, 0.67184, 0.72159, 0.77134, 0.82116, 0.87091, 0.90, 0.92066, 2.0] # Changed to 4.0
    #efftrig = [0.21747, 0.37142, 0.41090, 0.53808, 0.55519, 0.56443, 0.56801, 0.56814, 0.55703, 0.558, 0.9, 0.96786]

    neff = len(Etrig)

    print len(Etrig)
    print len(efftrig)

    bin_centers = []
    bin_heights = []
    step = 0.010
    for i in range(0,neff-1):

        x0 = Etrig[i]
        x1 = Etrig[i+1]
        y0 = efftrig[i]
        y1 = efftrig[i+1]

        # Once we're above 1 keVee the efficiency is flat.
        if x0>1.0:
            step = 1.0

        #print "%f %f %f %f" % (x0,x1,y0,y1)
        slope = (y1-y0)/(x1-x0)

        start = x0 
        stop  = x1
        nsteps = 0
        while start <= stop: 
            val = y0 + slope*(step*nsteps)
            #print "%f %f %f" % (start,val,slope)
            bin_centers.append(start)
            bin_heights.append(val)
            start += step
            nsteps += 1

    ################################################################################
    # Create the RooParametricStepFunction
    ################################################################################
    nbins = len(bin_centers)
    # These are the bin edges
    limits = TArrayD(nbins+1)
    for i in range(0, nbins):
        limits[i] = bin_centers[i]
    limits[nbins] = bin_centers[i]+0.1


    # These will hold the values of the bin heights
    scaling = 12.5 # Need to figure out how to do this properly. 
    list = RooArgList("list")
    binHeight = []
    for i in range(0,nbins-1):
        name = "binHeight%d" % (i)
        title = "bin %d Value" % (i)
        binHeight.append(RooRealVar(name, title, scaling*bin_heights[i]))
        list.add(binHeight[i]) # up to binHeight8, ie. 9 parameters
        pars.append(binHeight[i])

    aPdf = RooParametricStepFunction("aPdf","PSF", x, list, limits, nbins)

    sub_funcs.append(aPdf)

    ################################################################################
    # Make the RooEfficiency PDF
    ################################################################################
    # Acceptance state cut (1 or 0)
    cut = RooCategory("cut","cutr")
    cut.defineType("accept",1)
    cut.defineType("reject",0)
    
    # Construct efficiency p.d.f eff(cut|x)
    eff_pdf = RooEfficiency("eff_pdf","eff_pdf",aPdf,cut,"accept") 

    return cut,pars,sub_funcs,eff_pdf
