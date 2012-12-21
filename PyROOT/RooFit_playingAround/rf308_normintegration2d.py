################################################################################
#
# 'ADDITION AND CONVOLUTION' RooFit tutorial macro #203
# 
# Fitting and plotting in sub ranges
#
#
# 07/2008 - Wouter Verkerke 
#
################################################################################

import sys
import ROOT
from ROOT import *


# S e t u p   m o d e l 
# ---------------------

# Create observables x,y
x = RooRealVar("x","x",-10,10)
y = RooRealVar("y","y",-10,10) 

# Create p.d.f. gaussx(x,-2,3), gaussy(y,2,2) 
gx = RooGaussian("gx","gx",x,RooFit.RooConst(-2),RooFit.RooConst(3)) 
gy = RooGaussian("gy","gy",y,RooFit.RooConst(+2),RooFit.RooConst(2))

# Create gxy = gx(x)*gy(y)
gxy = RooProdPdf("gxy","gxy",RooArgList(gx,gy)) 



# R e t r i e v e   r a w  &   n o r m a l i z e d   v a l u e s   o f   R o o F i t   p . d . f . s
# --------------------------------------------------------------------------------------------------

# Return 'raw' unnormalized value of gx
print "gxy = %f" % ( gxy.getVal() )

# Return value of gxy normalized over x _and_ y in range [-10,10]
nset_xy = RooArgSet(x,y) 
print "gx_Norm[x,y] = %f" % (gxy.getVal(nset_xy))

# Create object representing integral over gx
# which is used to calculate  gx_Norm[x,y] == gx / gx_Int[x,y]
igxy = gxy.createIntegral(RooArgSet(x,y)) 
print "gx_Int[x,y] = %f" % ( igxy.getVal() )

# NB: it is also possible to do the following

# Return value of gxy normalized over x in range [-10,10] (i.e. treating y as parameter)
nset_x = RooArgSet(x) 
print "gx_Norm[x] = %f" % ( gxy.getVal(nset_x) )

# Return value of gxy normalized over y in range [-10,10] (i.e. treating x as parameter)
nset_y = RooArgSet(y) 
print "gx_Norm[y] = %f" % ( gxy.getVal(nset_y) )



# I n t e g r a t e   n o r m a l i z e d   p d f   o v e r   s u b r a n g e
# ----------------------------------------------------------------------------

# Define a range named "signal" in x from -5,5
x.setRange("signal",-5,5) 
y.setRange("signal",-3,3) 

# Create an integral of gxy_Norm[x,y] over x and y in range "signal"
# This is the fraction of of p.d.f. gxy_Norm[x,y] which is in the
# range named "signal"
#rllist = RooLinkedList()
#rllist.Add(RooArgSet(x,y))
#rllist.Add(RooFit.NormSet(RooArgSet(x,y)))
#rllist.Add(RooFit.NormSet(RooFit.Range("signal")))

argset = RooArgSet(x,y)
igxy_sig = gxy.createIntegral(argset,RooFit.NormSet(argset),RooFit.Range("signal"))
#igxy_sig = gxy.createIntegral(rllist)
print "gx_Int[x,y|signal]_Norm[x,y] = %f" % ( igxy_sig.getVal() )




# C o n s t r u c t   c u m u l a t i v e   d i s t r i b u t i o n   f u n c t i o n   f r o m   p d f
# -----------------------------------------------------------------------------------------------------

# Create the cumulative distribution function of gx
# i.e. calculate Int[-10,x] gx(x') dx'
gxy_cdf = gxy.createCdf(RooArgSet(x,y)) 

# Plot cdf of gx versus x
hh_cdf = gxy_cdf.createHistogram("hh_cdf",x,RooFit.Binning(40),RooFit.YVar(y,RooFit.Binning(40))) 
hh_cdf.SetLineColor(kBlue) 

can = TCanvas("rf308_normintegration2d","rf308_normintegration2d",600,600) 
gPad.SetLeftMargin(0.15) 
hh_cdf.GetZaxis().SetTitleOffset(1.8) 

hh_cdf.Draw("surf") 

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

