#!/usr/bin/env python

from ROOT import *

x = RooRealVar("x","ionization energy (keVee)",0.0,12.0);
t = RooRealVar("t","time",1.0,500)


mean0 = RooRealVar("mean0","mean0",4)
sigma0 = RooRealVar("sigma0","sigma0",0.5)
gauss0 = RooGaussian("gauss0","gauss0",x,mean0,sigma0)

mean1 = RooRealVar("mean1","mean1",7)
sigma1 = RooRealVar("sigma1","sigma1",0.5)
gauss1 = RooGaussian("gauss1","gauss1",x,mean1,sigma1)



slope0 = RooRealVar("slope0","slope0",-0.005)
slope1 = RooRealVar("slope1","slope1",-0.02)

decay0 = RooExponential("decay0","decay0",t,slope0)
decay1 = RooExponential("decay1","decay1",t,slope1)

prod0 = RooProdPdf("prod0","prod0",RooArgList(decay0,gauss0))
prod1 = RooProdPdf("prod1","prod1",RooArgList(decay1,gauss1))

n0 = RooRealVar("n0","n0",1000)
n1 = RooRealVar("n1","n1",500)

total = RooAddPdf("total","total",RooArgList(prod0,prod1),RooArgList(n0,n1))

x.setBins(50)
t.setBins(50)
frame_x = x.frame(RooFit.Title("x"))
frame_t = t.frame(RooFit.Title("t"))

data = total.generate(RooArgSet(x,t),1000)

data.plotOn(frame_x)
data.plotOn(frame_t)

can = TCanvas("can","can",10,10,1000,600)
can.SetFillColor(0)
can.Divide(2,1)

can.cd(1)
rargset = RooArgSet(prod0)
total.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(4))
rargset = RooArgSet(prod1)
total.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(2))
frame_x.Draw()
gPad.Update()

can.cd(2)
rargset = RooArgSet(prod0)
total.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(4))
rargset = RooArgSet(prod1)
total.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(2))
frame_t.Draw()
gPad.Update()

############################################################################
rep = ''
while not rep in ['q','Q']:
    rep = raw_input('enter "q" to quit: ')
    if 1<len(rep):
        rep = rep[0]

