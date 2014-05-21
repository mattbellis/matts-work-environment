#!/usr/bin/env python

from ROOT import *

x = RooRealVar("x","ionization energy (keVee)",0.0,12.0);
t = RooRealVar("t","time",1.0,500)

t.setRange("range0",1.0,200.0)
x.setRange("range0",0.0,12.0)

t.setRange("range1",200.0,500.0)
x.setRange("range1",0.0,12.0)

t.setRange("FULL",1.0,500.0)
x.setRange("FULL",0.0,12.0)


################################################################################
# x terms
################################################################################
mean0 = RooRealVar("mean0","mean0",4)
sigma0 = RooRealVar("sigma0","sigma0",0.5)
gauss0 = RooGaussian("gauss0","gauss0",x,mean0,sigma0)

mean1 = RooRealVar("mean1","mean1",7)
sigma1 = RooRealVar("sigma1","sigma1",0.5)
gauss1 = RooGaussian("gauss1","gauss1",x,mean1,sigma1)

slope_x = RooRealVar("slope_x","slope_x",-0.3)
decay_x = RooExponential("decay_x","decay_x",x,slope_x)

################################################################################
# t terms
################################################################################
slope0 = RooRealVar("slope0","slope0",-0.005)
slope1 = RooRealVar("slope1","slope1",-0.02)

decay0 = RooExponential("decay0","decay0",t,slope0)
decay1 = RooExponential("decay1","decay1",t,slope1)

prod0 = RooProdPdf("prod0","prod0",RooArgList(decay0,gauss0))
prod1 = RooProdPdf("prod1","prod1",RooArgList(decay1,gauss1))

n0 = RooRealVar("n0","n0",1000)
n1 = RooRealVar("n1","n1",500)
n2 = RooRealVar("n2","n2",1000)

total = RooAddPdf("total","total",RooArgList(prod0,prod1,decay_x),RooArgList(n0,n1,n2))
 
#nsig = RooRealVar("nsig","nsig",200,0,6000)
#total1 = RooExtendPdf("total1","total1",total)

x.setBins(50)
t.setBins(50)
frame_x = x.frame(RooFit.Title("x"))
frame_t = t.frame(RooFit.Title("t"))

data = total.generate(RooArgSet(x,t),2500)


n0.setVal(1000)
n0.setConstant(False)

n1.setVal(500)
n1.setConstant(False)

n2.setVal(1000)
n2.setConstant(False)

fit_range = "range0,range1"
#fit_range = "FULL"

results = total.fitTo(data,RooFit.Save(True),RooFit.Range(fit_range),RooFit.Extended(True))
results.Print("v")

data.plotOn(frame_x)
data.plotOn(frame_t)

can = TCanvas("can","can",10,10,1000,600)
can.SetFillColor(0)
can.Divide(2,1)

################################################################################

can.cd(1)
rargset = RooArgSet(total)
total.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(3),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

rargset = RooArgSet(prod0)
total.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(4),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

rargset = RooArgSet(prod1)
total.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(2),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

rargset = RooArgSet(decay_x)
total.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(22),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

frame_x.Draw()
gPad.Update()

################################################################################

can.cd(2)
rargset = RooArgSet(total)
total.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(3),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

rargset = RooArgSet(prod0)
total.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(4),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

rargset = RooArgSet(prod1)
total.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(2),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

rargset = RooArgSet(decay_x)
total.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(22),RooFit.Range(fit_range),RooFit.NormRange("FULL"))

frame_t.Draw()
gPad.Update()


############################################################################
rep = ''
while not rep in ['q','Q']:
    rep = raw_input('enter "q" to quit: ')
    if 1<len(rep):
        rep = rep[0]

