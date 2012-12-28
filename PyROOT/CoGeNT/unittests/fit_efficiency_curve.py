from ROOT import *
import sys

infile = open(sys.argv[1])
for line in infile:

    vals = line.split()

    energy = float(vals[0])
    eff = float(vals[1])



x = RooRealVar("x","ionization energy (keVee)",0.0,4.0)

max_eff = RooRealVar("max_eff","Maximum efficiency",0.867)
Ethresh = RooRealVar("Ethresh","E_{threshhold}",0.867)
sigma = RooRealVar("sigma","sigma",0.867)

sigmoid = RooFormulaVar("sigmoid","sigmoid","@0/(1+exp((-(x-@1)/(@2*@1))))",RooArgList(max_eff,Ethresh,sigma))

#TF1 f("f","0.867/(1+exp((-(x-0.45)/(0.03*0.45))))")


