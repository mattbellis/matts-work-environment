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


data.plotOn(xframe)
exp_exp_x.plotOn(xframe)
xframe.Draw()

rep = ''
while not rep in ['q','Q']:
    rep = raw_input('enter "q" to quit: ')
    if 1<len(rep):
        rep = rep[0]



