#!/usr/bin/env python

from ROOT import *

################################################################################
# Main
###################################################################################
def main():

    x = RooRealVar("x","x",1.0,0.0,10.0)

    p1 = RooRealVar("poly1","Linear coefficient",-0.5)
    rarglist = RooArgList(p1)
    poly_x = RooPolynomial("poly_x","Polynomial PDF",x,rarglist);

    data = poly_x.generate(RooArgSet(x),1000)

    frame = x.frame(RooFit.Bins(20))

    data.plotOn(frame)


    can = TCanvas("can","can",10,10,600,600)
    can.SetFillColor(0)
    can.Divide(1,1)

    can.cd(1)
    frame.Draw()

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

