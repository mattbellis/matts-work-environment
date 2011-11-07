#!/usr/bin/env python

from ROOT import *


################################################################################
################################################################################
def main():

    x = RooRealVar("x","ionization energy (keVee)",0.0,4.0);
    t = RooRealVar("t","time",0.0,365.0)

    x.setRange("sub_x0",0.0,3.0)
    x.setRange("sub_x1",0.5,0.9)
    x.setRange("sub_x2",0.5,3.0)

    for i in range(0,6):
        name = "sub_t%d" % (i)
        lo = i*365.0/6.0;
        hi = (i+1)*365.0/6.0;
        t.setRange(name,lo,hi)

    # Define the exponential background
    bkg_slope = RooRealVar("bkg_slope","Exponential slope of the background",-0.5,-10.0,0.0)
    bkg_exp = RooExponential("bkg_exp","Exponential PDF for bkg",x,bkg_slope)

    # Define the exponential signal
    sig_slope = RooRealVar("sig_slope","Exponential slope of the signal",-4.5,-10.0,0.0)
    sig_exp = RooExponential("sig_exp","Exponential PDF for sig",x,sig_slope)

    # Define the calibration peak
    calib_mean = RooRealVar("calib_mean","Landau mean of the calibration peak",1.2,1.0,1.5);
    calib_sigma = RooRealVar("calib_sigma","Landau sigma of the calibration peak",0.001,0.0,1.0)
    calib_landau = RooLandau("calib_landau","Landau PDF for calibration peak",x,calib_mean,calib_sigma)

    ############################################################################
    # Define the calibration peaks (Gaussians)
    ############################################################################
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

    #data = total_pdf.generate(RooArgSet(x,t),500) # Gives good agreement with plot
    data = total_pdf.generate(RooArgSet(x,t),4000)
    data_reduced0 = data.reduce(RooFit.CutRange("sub_x0"))
    data_reduced1 = data.reduce(RooFit.CutRange("sub_x1"))
    data_reduced2 = data.reduce(RooFit.CutRange("sub_x2"))

    data_reduced_t = []
    data_reduced_t_x = []
    for i in range(0,6):
        tname = "sub_t%d" % (i)
        data_reduced_t.append(data.reduce(RooFit.CutRange(tname)))
        data_reduced_t_x.append([])
        data_temp = data.reduce(RooFit.CutRange(tname))
        for j in range(0,3):
            xname = "sub_x%d" % (j)
            #data_reduced_t_x[i].append(data.reduce(RooFit.CutRange(tname),RooFit.CutRange(xname)))
            data_reduced_t_x[i].append(data_temp.reduce(RooFit.CutRange(xname)))


    ############################################################################
    # Make frames 
    ############################################################################
    # x
    x.setBins(80)
    xframes = []
    for i in xrange(6):
        xframes.append([])
        for j in xrange(3):
            xframes[i].append(x.frame(RooFit.Title("Plot of ionization energy")))
            data_reduced_t_x[i][j].plotOn(xframes[i][j])

    ############################################################################
    # t
    t.setBins(12)
    tframes = []
    for i in xrange(3):
        tframes.append(t.frame(RooFit.Title("Plot of ionization energy")))
        if i==0:
            data_reduced0.plotOn(tframes[i])
        elif i==1:
            data_reduced1.plotOn(tframes[i])
        elif i==2:
            data_reduced2.plotOn(tframes[i])



    #tot_argset = RooArgSet(total_pdf)
    #total_pdf.plotOn(xframes[0],RooFit.Components(tot_argset),RooFit.LineColor(8),RooFit.ProjWData(data))

    #bkg_argset = RooArgSet(bkg_exp)
    #bkg_argset = RooArgSet(bxg)
    #total_pdf.plotOn(xframes[0],RooFit.Components(bkg_argset),RooFit.LineStyle(2),RooFit.LineColor(4),RooFit.ProjWData(data))

    #sig_argset = RooArgSet(sig_exp)
    #sig_argset = RooArgSet(sxg)
    #total_pdf.plotOn(xframes[0],RooFit.Components(sig_argset),RooFit.LineStyle(2),RooFit.LineColor(2),RooFit.ProjWData(data))

    #lxg_argset = RooArgSet(lxg)
    #total_pdf.plotOn(xframes[0],RooFit.Components(lxg_argset),RooFit.LineStyle(2),RooFit.LineColor(6),RooFit.ProjWData(data))

    ############################################################################
    # Make canvases.
    ############################################################################
    can_x = []
    for i in range(0,2):
        name = "can_x_%s" % (i)
        can_x.append(TCanvas(name,name,10+10*i,10+10*i,1200,900))
        can_x[i].SetFillColor(0)
        can_x[i].Divide(3,3)

    can_t = TCanvas("can_t","can_t",200,200,1200,600)
    can_t.SetFillColor(0)
    can_t.Divide(3,1)

    for i in xrange(6):
        for j in xrange(3):
            pad_index = (i%3)*3+(j+1)
            can_x[i/3].cd(pad_index)
            xframes[i][j].GetXaxis().SetRangeUser(0.0,3.0)
            xframes[i][j].Draw()
            gPad.Update()

    for i in xrange(3):
        can_t.cd(i+1)
        tframes[i].Draw()
        gPad.Update()

    print "\n"
    print "entries: %d" % (data_reduced0.numEntries())
    print "entries: %d" % (data_reduced1.numEntries())
    print "entries: %d" % (data_reduced2.numEntries())


    ############################################################################
    rep = ''
    while not rep in ['q','Q']:
        rep = raw_input('enter "q" to quit: ')
        if 1<len(rep):
            rep = rep[0]

################################################################################
################################################################################
if __name__ == "__main__":
    main()




