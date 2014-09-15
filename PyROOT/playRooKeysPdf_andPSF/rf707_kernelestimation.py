#!/usr/bin/env python
################################################################################
#
# 'SPECIAL PDFS' RooFit tutorial macro #707
# 
# Using non-parametric (multi-dimensional) kernel estimation p.d.f.s
#
#
#
# 07/2008 - Wouter Verkerke 
# 
################################################################################

from ROOT import *
from array import *

# C r e a t e   l o w   s t a t s   1 - D   d a t a s e t 
# -------------------------------------------------------

# Create a toy pdf for sampling
x = RooRealVar("x","x",0,20) 
p = RooPolynomial("p","p",x,RooArgList(RooFit.RooConst(0.01),RooFit.RooConst(-0.01),RooFit.RooConst(0.0004))) 

# Sample 500 events from p
data1 = p.generate(RooArgSet(x),200) # RooDataSet



# C r e a t e   1 - D   k e r n e l   e s t i m a t i o n   p d f
# ---------------------------------------------------------------

# Create adaptive kernel estimation pdf. In this configuration the input data
# is mirrored over the boundaries to minimize edge effects in distribution
# that do not fall to zero towards the edges
#kest1 = RooKeysPdf("kest1","kest1",x,*data1,RooKeysPdf::MirrorBoth) 
kest1 = RooKeysPdf("kest1","kest1",x,data1,RooKeysPdf.MirrorBoth) 

# An adaptive kernel estimation pdf on the same data without mirroring option
# for comparison
#kest2 = RooKeysPdf("kest2","kest2",x,data1,RooKeysPdf::NoMirror) 
kest2 = RooKeysPdf("kest2","kest2",x,data1,RooKeysPdf.NoMirror) 


# Adaptive kernel estimation pdf with increased bandwidth scale factor
# (promotes smoothness over detail preservation)
#kest3 = RooKeysPdf("kest1","kest1",x,data1,RooKeysPdf::MirrorBoth,2) 
kest3 = RooKeysPdf("kest1","kest1",x,data1,RooKeysPdf.MirrorBoth,2) 


# Plot kernel estimation pdfs with and without mirroring over data
frame = x.frame(RooFit.Title("Adaptive kernel estimation pdf with and w/o mirroring"),RooFit.Bins(20)) # RooPlot 
data1.plotOn(frame) 
kest1.plotOn(frame) 
kest2.plotOn(frame,RooFit.LineStyle(kDashed),RooFit.LineColor(kRed)) 


# Plot kernel estimation pdfs with regular and increased bandwidth
frame2 = x.frame(RooFit.Title("Adaptive kernel estimation pdf with regular, increased bandwidth")) # RooPlot 
kest1.plotOn(frame2) 
kest3.plotOn(frame2,RooFit.LineColor(kMagenta)) 



# C r e a t e   l o w   s t a t s   2 - D   d a t a s e t 
# -------------------------------------------------------

# Construct a 2D toy pdf for sampleing
y = RooRealVar("y","y",0,20) 
py = RooPolynomial("py","py",y,RooArgList(RooFit.RooConst(0.01),RooFit.RooConst(0.01),RooFit.RooConst(-0.0004))) 
pxy = RooProdPdf("pxy","pxy",RooArgList(p,py)) 
data2 = pxy.generate(RooArgSet(x,y),1000) # RooDataSet 



# C r e a t e   2 - D   k e r n e l   e s t i m a t i o n   p d f
# ---------------------------------------------------------------

# Create 2D adaptive kernel estimation pdf with mirroring 
kest4 = RooNDKeysPdf("kest4","kest4",RooArgList(x,y),data2,"am") 

# Create 2D adaptive kernel estimation pdf with mirroring and double bandwidth
kest5 = RooNDKeysPdf("kest5","kest5",RooArgList(x,y),data2,"am",2) 

# Create a histogram of the data
rllist_data = RooLinkedList()
rllist_data.Add(RooFit.Binning(10))
rllist_data.Add(RooFit.YVar(y, RooFit.Binning(10)))

# TH1* hh_data = data2->createHistogram("hh_data",x,Binning(10),YVar(y,Binning(10))) ;
hh_data = data2.createHistogram("hh_data",RooArgSet(x),rllist_data) # TH1 

# Create histogram of the 2d kernel estimation pdfs
rllist_pdf = RooLinkedList()
rllist_pdf.Add(RooFit.Binning(25))
rllist_pdf.Add(RooFit.YVar(y, RooFit.Binning(25)))

hh_pdf = kest4.createHistogram("hh_pdf",x,rllist_pdf) # TH1 
hh_pdf2 = kest5.createHistogram("hh_pdf2",x,rllist_pdf) # TH1
hh_pdf.SetLineColor(kBlue) 
hh_pdf2.SetLineColor(kMagenta) 



c = TCanvas("rf707_kernelestimation","rf707_kernelestimation",800,800) # TCanvas
c.Divide(2,2) 
c.cd(1) 
gPad.SetLeftMargin(0.15) 
frame.GetYaxis().SetTitleOffset(1.4) 
frame.Draw() 

c.cd(2) 
gPad.SetLeftMargin(0.15) 
frame2.GetYaxis().SetTitleOffset(1.8) 
frame2.Draw() 

c.cd(3) 
gPad.SetLeftMargin(0.15) 
hh_data.GetZaxis().SetTitleOffset(1.4) 
hh_data.Draw("lego") 

c.cd(4) 
gPad.SetLeftMargin(0.20) 
hh_pdf.GetZaxis().SetTitleOffset(2.4) 
hh_pdf.Draw("surf") 
hh_pdf2.Draw("surfsame") 

        
## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
          rep = rep[0]





