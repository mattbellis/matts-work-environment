#!/usr/bin/env python

from ROOT import *

################################################################################
# Main
###################################################################################
def main():

    x = RooRealVar("x","x",1.0,0.0,10.0)

    frame = x.frame(RooFit.Bins(50))

    # Define some subranges
    x.setRange("sig0",0.0,5.0)
    x.setRange("sig1",6.0,7.0)
    x.setRange("sig2",8.0,9.0)

    # Define a simple linear function for the PDF
    amp = RooRealVar("amp","amplitude",2.,0,1000.0)
    mod = RooRealVar("mod","modulation",0.628,-100,100)
    offset = RooRealVar("offset","offset",10.0,-2,100)
    poly_x = RooGenericPdf("poly_x","amp*(offset + sin(mod*x))",RooArgList(x,amp,mod,offset))

    # Create the extended PDF
    nsig = RooRealVar("nsig","nsig",200,0,6000)

    sig_pdf = RooExtendPdf("sig_pdf","sig_pdf",poly_x,nsig)

    # Generate the data
    data = sig_pdf.generate(RooArgSet(x),10000)

    # Draw the full range on the frame
    #data.plotOn(frame)

    ############################################################################
    # Fit to the data
    ############################################################################

    # Create a reduced sub set of our data
    data_reduced = data.reduce(RooFit.CutRange("sig0"))
    data_reduced.append(data.reduce(RooFit.CutRange("sig1")))
    data_reduced.append(data.reduce(RooFit.CutRange("sig2")))

    data_reduced.plotOn(frame)

    #sig_pdf.fitTo(data_reduced,RooFit.Range("sig0,sig1"),RooFit.Extended(True))
    sig_pdf.fitTo(data_reduced,RooFit.Range("sig0,sig1,sig2"),RooFit.Extended(True))


    # Draw the results
    can = TCanvas("can","can",10,10,600,600)
    can.SetFillColor(0)
    can.Divide(1,1)

    can.cd(1)
    sig_pdf.plotOn(frame)
    frame.Draw()

    # Check out many entries were in our original and our reduced dataset.
    print "entries: %d" % (data.numEntries())
    print "entries: %d" % (data_reduced.numEntries())

    print "nsig: %f\t\tnsig*3: %f" % (nsig.getVal(),3*nsig.getVal())

    ############################################################################
    rep = ''
    while not rep in ['q','Q']:
        rep = raw_input('enter "q" to quit: ')
        if 1<len(rep):
            rep = rep[0]


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    main()

