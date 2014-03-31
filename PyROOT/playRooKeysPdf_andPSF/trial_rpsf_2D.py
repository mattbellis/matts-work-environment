#!/usr/bin/env python

from ROOT import *
from array import *

###############################################################
# RooParametricStepFunction
###############################################################

# Here's a fake variable over which the data will be generated
# and fit.
x = RooRealVar("x","x variable", 0.0, 10.0) 
y = RooRealVar("y","y variable", -5.0, 5.0) 

#############################
# Set up the RPSF
#############################

nbins = 10

# These are the bin edges
limits = TArrayD(nbins+1)
for i in range(0, nbins+1):
  limits[i] = 0.0 + i*1.0

# These will hold the values of the bin heights
list = RooArgList("list")
binHeight = []
for i in range(0,nbins-1):
  name = "binHeight%d" % (i)
  title = "bin %d Value" % (i)
  binHeight.append(RooRealVar(name, title, 0.01, 0.0, 1.0))
  list.add(binHeight[i]) # up to binHeight8, ie. 9 parameters

# Declare the RPSF
aPdf = RooParametricStepFunction("aPdf","PSF", x, list, limits, nbins)

nbkg = RooRealVar("nbkg","number of background events,",150)
aPdf_ex = RooAddPdf("aPdf_ex","Extended aPdf",RooArgList(aPdf), RooArgList(nbkg))

#
########################################################################
#
# Gaussian in y
########################################################################
mean = RooRealVar("mean","#mu of Gaussian", 0.000)
sigma = RooRealVar("sigma","Width of Gaussian", 0.8)
gauss = RooGaussian("gauss", "Gaussian PDF", y, mean, sigma)

########################################################################
########################################################################

c = TCanvas("c","c", 10, 10, 800, 800)
c.Divide(3,2)

frames = []
for i in range(0, 6):
  title = "frame%d" % (i)
  if i/3==0:
    frames.append(x.frame( RooFit.Title(title), RooFit.Bins(20))) # RooPlot 
  elif i/3==1:
    frames.append(y.frame( RooFit.Title(title), RooFit.Bins(20))) # RooPlot 
  else:
    frames.append(y.frame( RooFit.Title(title), RooFit.Bins(20))) # RooPlot 



# Plot the starting values of the RPSF (flat)
c.cd(1)
aPdf.plotOn(frames[0])
frames[0].Draw()
gPad.Update()

c.cd(4)
gauss.plotOn(frames[3])
frames[3].Draw()
gPad.Update()

#######################################################
# Generate some fake data
#######################################################
p0 = RooRealVar("p0", "p0", 5.0)
p1 = RooRealVar("p1", "p1", -2.0)
p2 = RooRealVar("p2", "p2", 3.0)
f = RooPolynomial("f", "Polynomial PDF", x, RooArgList(p0,p1,p2))


total_for_gen = RooProdPdf("total_for_gen","f*gauss",RooArgList(f, gauss))
total_for_fit = RooProdPdf("total_for_gen","aPdf_ex*gauss",RooArgList(aPdf_ex, gauss))

data = total_for_gen.generate(RooArgSet(x,y), 1000)

# Plot the data
c.cd(2)
data.plotOn(frames[1], RooLinkedList())
frames[1].Draw()
gPad.Update()

c.cd(5)
data.plotOn(frames[4], RooLinkedList())
frames[4].Draw()
gPad.Update()

########################################################
# Run the fit 
########################################################
mean.setVal(0.1)
sigma.setVal(0.1)
mean.setConstant(kFALSE)
sigma.setConstant(kFALSE)
nbkg.setConstant(kFALSE)
fit_results = total_for_fit.fitTo(data, RooFit.Extended(), RooFit.Save(kTRUE))
#fit_results = total_for_fit.fitTo(data, RooFit.Save(kTRUE))

# Plot the RPSF after the fit has converged.
c.cd(3)
total_for_fit.plotOn(frames[1])
frames[1].Draw()
gPad.Update()

c.cd(6)
total_for_fit.plotOn(frames[4])
frames[4].Draw()
gPad.Update()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
  rep = ''
  while not rep in [ 'q', 'Q' ]:
    rep = raw_input( 'enter "q" to quit: ' )
    if 1 < len(rep):
      rep = rep[0]





