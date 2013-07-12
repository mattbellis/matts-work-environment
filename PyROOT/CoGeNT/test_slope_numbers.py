################################################################################
# Import the necessary libraries.
################################################################################
import sys
from ROOT import *

from math import *

from datetime import datetime,timedelta

import argparse

x = RooRealVar("x","Ionization Energy",0.5,3.2,"keVee");

x.setBins(108)

xframe = x.frame(RooFit.Title("Ionization Energy"))

exp_slope = RooRealVar("exp_slope","Exponential slope of the exponential term",-3.36,-10.0,0.0)
exp_exp_x = RooExponential("exp_exp_x","Exponential PDF for exp x",x,exp_slope)

data = exp_exp_x.generate(RooArgSet(x),575)

x.setRange('range0',0.5,1.0)
x.setRange('range1',1.8,3.2)
x.setRange('FULL',0.5,3.2)

data.plotOn(xframe)
#exp_exp_x.plotOn(xframe)
#data.plotOn(xframe,RooFit.Range('range0'),RooFit.Cut('range0'))
exp_exp_x.plotOn(xframe,RooFit.Range('range0'))
exp_exp_x.plotOn(xframe,RooFit.Range('range0'),RooFit.NormRange('range0'),RooFit.LineColor(2))
xframe.Draw()

rep = ''
while not rep in ['q','Q']:
    rep = raw_input('enter "q" to quit: ')
    if 1<len(rep):
        rep = rep[0]



