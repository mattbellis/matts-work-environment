#!/usr/bin/env python

from ROOT import *
from pdf_definitions import *

################################################################################
# main
################################################################################
def main():
    
    ############################################################################

    var_ranges = [[],[],[]]
    var_ranges[0] = [0.0,10.0]
    var_ranges[1] = [-5.0,5.0]
    var_ranges[2] = [-5.0,5.0]

    ############################################################################
    x,y,z = build_xyz(var_ranges)
    my_pars, sub_funcs_list, conv_func = conv_fft_exp_x_gaus_PDF(x)
    #my_pars, sub_funcs_list, conv_func = conv_fft_decay_x_gaus_PDF(x)

    #################################
    # Create a dictionary of the pars
    #################################
    pars_dict = {}
    for p in my_pars:
        print p.GetName()
        pars_dict[p.GetName()] = p

    sub_funcs = {}
    for f in sub_funcs_list:
        print f.GetName()
        sub_funcs[f.GetName()] = f

    ############################################################################
    # Set the ``smearing" for the gaussian
    pars_dict["mean_x_conv"].setVal(0.0)
    pars_dict["sigma_x_conv"].setVal(1.0)

    ############################################################################

    # Set #bins to be used for FFT sampling to 10000
    x.setBins(10000,"cache") ;

    # S a m p l e ,   f i t   a n d   p l o t   c o n v o l u t e d   p d f
    # ----------------------------------------------------------------------
    # Sample 1000 events in x from gxlx
    data = conv_func.generate(RooArgSet(x),10000) # RooDataset

    # Fit gxlx to data
    pars_dict["c_x_conv"].setConstant(kFALSE)
    result = conv_func.fitTo(data,RooFit.Save(kTRUE))
    result.Print("v")

    # Plot data, exp pdf, exp (X) gauss pdf
    frame = x.frame(RooFit.Title("exp (x) gauss convolution")) # RooPlot
    #frame = x.frame(RooFit.Title("decay (x) gauss convolution")) # RooPlot
    data.plotOn(frame) 
    conv_func.plotOn(frame) 
    #sub_funcs["gauss_x_conv"].plotOn(frame,RooFit.LineStyle(kDashed)) 
    sub_funcs["exp_x_conv"].plotOn(frame,RooFit.LineStyle(kDashed)) 
    #sub_funcs["decay_x_conv"].plotOn(frame,RooFit.LineStyle(kDashed)) 


    # Draw frame on canvas
    can = TCanvas("rf208_convolution","rf208_convolution",600,600) 
    can.SetFillColor(0)
    can.Divide(1,1)
    can.cd(1)
    gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.Draw() 

    ############################################################################
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]






################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
