#!/usr/bin/env python

###############################################################
# intro3.py
# Matt Bellis
# bellis@slac.stanford.edu
# Dec. 6, 2008
# Rewritten from intro3.C from RooFit tutorials found at
# http://roofit.sourceforge.net/docs/tutorial/intro/index.html
###############################################################

import sys
from optparse import OptionParser
from array import *

#### Command line variables ####
doFit = False

parser = OptionParser()
parser.add_option("--num-sig", dest="num_sig", default=50, help="Number of signal events, embedded or otherwise.")
parser.add_option("--num-bkg", dest="num_bkg", default=1000, help="Number of background events")

(options, args) = parser.parse_args()

infile = open(args[0],"r")

#############################################################
# Root stuff
#############################################################
from ROOT import *

############################################
RooMsgService.instance().Print()
RooMsgService.instance().deleteStream(1)
RooMsgService.instance().Print()
############################################
################################################
# Import from other files
################################################
from pdf_definitions import *
################################################################################
# RooRealVars (axis)
xlo = 0.0; xhi = 10.0
ylo = 0.0; yhi = 10.0
zlo = 0.0; zhi = 10.0
data_ranges = [[xlo,xhi], [ylo,yhi], [zlo,zhi]]

# RooRealVars (axis)
x,y,z = build_xyz(data_ranges)

# RooDataSets
data = RooDataSet("data","data",RooArgSet(x,y,z)) 

for line in infile:
    vals = line.split()
    x.setVal(float(vals[0]))
    y.setVal(float(vals[1]))
    z.setVal(float(vals[2]))
    data.add(RooArgSet(x,y,z))

################################################################################

# Grab the fit functions and everything else which is needed.
my_pars, sub_funcs_list, total_fit_func = tot_PDF([x,y,z])

# Create a dictionary of the pars
pars_dict = {}
for p in my_pars:
    #print p
    pars_dict[p.GetName()] = p

sub_funcs = {}
for f in sub_funcs_list:
    sub_funcs[f.GetName()] = f

# Set the starting values
##############################################
# Set num sig/bkg by hand
##############################################
pars_dict["nbkg"].setVal( float(options.num_bkg) )
pars_dict["nsig"].setVal( float(options.num_sig) )

pars_dict["c_x_bkg"].setVal(-0.5)
pars_dict["c_y_bkg"].setVal(-0.5)
pars_dict["c_z_bkg"].setVal(-0.5)

pars_dict["mean_x_sig"].setVal(5.0)
pars_dict["sigma_x_sig"].setVal(0.5)

pars_dict["mean_y_sig"].setVal(5.0)
pars_dict["sigma_y_sig"].setVal(0.5)

pars_dict["mean_z_sig"].setVal(5.0)
pars_dict["sigma_z_sig"].setVal(0.5)

for p in pars_dict:
    #print p
    pars_dict[p].setConstant(False)


# Create the NLL for the fit
nll = RooNLLVar("nll","nll",total_fit_func,data,RooFit.Extended(kTRUE))
fit_func = RooFormulaVar("fit_func","nll",RooArgList(nll)) 
m = RooMinuit(fit_func)
m.setVerbose(kFALSE)

# Set the starting values
#set_starting_values(mypars, starting_values)

'''
fit_results = fit_func.fitTo(data, 
                             RooFit.Extended(kTRUE), 
                             RooFit.Save(kTRUE), 
                             RooFit.Range(fit_range), 
                             RooFit.Strategy(2), 
                             RooFit.PrintLevel(-1) ) #RooFitResults
'''
m.migrad()
m.hesse()
fit_results = m.save()
print "FINISHED FIT"
fit_results.Print("v")


