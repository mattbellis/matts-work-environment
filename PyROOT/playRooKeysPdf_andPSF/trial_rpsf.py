#!/usr/bin/env python

from ROOT import *
from array import *

###############################################################
# RooParametricStepFunction
###############################################################

# Here's a fake variable over which the data will be generated
# and fit.
x = RooRealVar("x","x variable", 0.0, 10.0) 

#############################
# Set up the RPSF
#############################

nbins = 10

# These are the bin edges
limits = TArrayD(nbins+1)
for i in range(0, nbins+1):
  limits[i] = 0.0 + i*1

# These will hold the values of the bin heights
list = RooArgList("list")
binHeight = []
for i in range(0,nbins-1):
  name = "binHeight%d" % (i)
  title = "bin %d Value" % (i)
  binHeight.append(RooRealVar(name, title, 0.1, 0.0, 1.0))
  list.add(binHeight[i]) # up to binHeight8, ie. 9 parameters

# Declare the RPSF
aPdf = RooParametricStepFunction("aPdf","PSF", x, list, limits, nbins)
#

c = TCanvas("c","c", 10, 10, 800, 800)
c.Divide(2,2)

frames = []
for i in range(0, 4):
  title = "frame%d" % (i)
  frames.append(x.frame( RooFit.Title(title), RooFit.Bins(20))) # RooPlot 


# Plot the starting values of the RPSF (flat)
c.cd(1)
aPdf.plotOn(frames[0])
frames[0].Draw()
gPad.Update()

#######################################################
# Generate some fake data
#######################################################
p0 = RooRealVar("p0", "p0", 5.0)
p1 = RooRealVar("p1", "p1", -2.0)
p2 = RooRealVar("p2", "p2", 3.0)
f = RooPolynomial("f", "Polynomial PDF", x, RooArgList(p0,p1,p2))
data = f.generate(RooArgSet(x), 200)

# Plot the data
c.cd(2)
data.plotOn(frames[1], RooLinkedList())
frames[1].Draw()
gPad.Update()

########################################################
# Run the fit 
########################################################
#fit_results = aPdf.fitTo(data, RooFit.Extended(), RooFit.Save(kTRUE))
fit_results = aPdf.fitTo(data, RooFit.Save(kTRUE))

# Plot the RPSF after the fit has converged.
c.cd(3)
aPdf.plotOn(frames[2])
frames[2].Draw()
gPad.Update()


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
  rep = ''
  while not rep in [ 'q', 'Q' ]:
    rep = raw_input( 'enter "q" to quit: ' )
    if 1 < len(rep):
      rep = rep[0]





