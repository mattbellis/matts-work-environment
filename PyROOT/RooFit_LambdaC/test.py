#!/usr/bin/env python

from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from color_palette import *

from array import *

from nn_limits_and_binning import *
import nn_limits_and_binning

psf_lo, psf_hi, vary_limits = 0.0, 0.0, 0.0
psf_lo, psf_hi, vary_limits = nn_fit_params()

print psf_lo
################################################################################
################################################################################

from backgroundAndSignal_EVERYTHINGDEF_def import *

from my_roofit_utilities import *

# # # # #
psf_lo, psf_hi, vary_limits = nn_fit_params()

x,y,z = build_xyz(psf_lo, psf_hi)


data = RooDataSet("data","data",RooArgSet(x,y,z))

