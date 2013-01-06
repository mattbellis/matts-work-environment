#!/usr/bin/env python 

from ROOT import *

################################################################################
# B - D e c a y   w i t h   m i x i n g          #
################################################################################

# C o n s t r u c t   p d f 
# -------------------------

# Observable
dt = RooRealVar("dt","dt",-10,10) 
dt.setBins(40) 

# Parameters
dm = RooRealVar ("dm","delta m(B0)",0.472) 
tau = RooRealVar("tau","tau (B0)",1.547) 
w = RooRealVar ("w","flavour mistag rate",0.1) 
dw = RooRealVar ("dw","delta mistag rate for B0/B0bar",0.1) 

mixState = RooCategory("mixState","B0/B0bar mixing state") 
mixState.defineType("mixed",-1) 
mixState.defineType("unmixed",1) 

tagFlav = RooCategory("tagFlav","Flavour of the tagged B0") 
tagFlav.defineType("B0",1) 
tagFlav.defineType("B0bar",-1) 

# Use delta function resolution model
tm = RooTruthModel ("tm","truth model",dt) 

################################################################################
# G e n e r i c   B   d e c a y  w i t h    u s e r   c o e f f i c i e n t s  #
################################################################################

# C o n s t r u c t   p d f 
# -------------------------

# Model parameters
DGbG = RooRealVar ("DGbG","DGamma/GammaAvg",0.5,-1,1)
Adir = RooRealVar ("Adir","-[1-abs(l)**2]/[1+abs(l)**2]",0)
Amix = RooRealVar ("Amix","2Im(l)/[1+abs(l)**2]",0.7)
Adel = RooRealVar ("Adel","2Re(l)/[1+abs(l)**2]",0.7)

# Derived input parameters for pdf
DG = RooFormulaVar ("DG","Delta Gamma","@1/@0",RooArgList(tau,DGbG))

# Construct coefficient functions for sin,cos,sinh modulations of decay distribution
fsin = RooFormulaVar ("fsin","fsin","@0*@1*(1-2*@2)",RooArgList(Amix,tagFlav,w,mixState))
fcos = RooFormulaVar ("fcos","fcos","@0*@1*(1-2*@2)",RooArgList(Adir,tagFlav,w,mixState))
fsinh = RooFormulaVar ("fsinh","fsinh","@0",RooArgList(Adel))

#fsin = RooFormulaVar ("fsin","fsin","0*@0",RooArgList(mixState,tagFlav))
#fcos = RooFormulaVar ("fcos","fcos","@0",RooArgList(mixState,tagFlav))
#fsinh = RooFormulaVar ("fsinh","fsinh","0*@0",RooArgList(mixState,tagFlav))

#fcos = RooFormulaVar ("fcos","fcos","@0",RooArgList(mixState,tagFlav))
################################################################################

# Construct generic B decay pdf using above user coefficients
bcpg = RooBDecay ("bcpg","bcpg",dt,tau,DG,RooFit.RooConst(1),fsinh,fcos,fsin,dm,tm,RooBDecay.DoubleSided)
#bcpg = RooBDecay ("bcpg","bcpg",dt,tau,DG,RooFit.RooConst(1),RooFit.RooConst(0),fcos,RooFit.RooConst(0),dm,tm,RooBDecay.DoubleSided)



# P l o t   -   I m ( l ) = 0 . 7 ,   R e ( l ) = 0 . 7   | l | = 1,   d G / G = 0 . 5 
# -------------------------------------------------------------------------------------

# Generate some data
data4 = bcpg.generate(RooArgSet(dt,tagFlav,mixState),10000) 

# Plot B0 and B0bar tagged data separately 
frame6 = dt.frame(RooFit.Title("B decay distribution with CPV(Im(l)=0.7,Re(l)=0.7,|l|=1,dG/G=0.5) (B0/B0bar)"))   
frame7 = dt.frame(RooFit.Title("B decay distribution with CPV(Im(l)=0.7,Re(l)=0.7,|l|=1,dG/G=0.5) (B0/B0bar)"))   
frame8 = dt.frame(RooFit.Title("B decay distribution with CPV(Im(l)=0.7,Re(l)=0.7,|l|=1,dG/G=0.5) (B0/B0bar)"))   

#data4.plotOn(frame6,Cut("tagFlav==tagFlav.B0")) 
#bcpg.plotOn(frame6,Slice(tagFlav,"B0")) 

#data4.plotOn(frame6,Cut("tagFlav==tagFlav.B0bar"),MarkerColor(kCyan)) 
#bcpg.plotOn(frame6,Slice(tagFlav,"B0bar"),LineColor(kCyan)) 

data4.plotOn(frame6,RooFit.Asymmetry(mixState))
print "Printing PDF asymmetry......" 
#bcpg.plotOn(frame6,RooFit.ProjWData(RooArgSet(mixState),data4,kTRUE),RooFit.Asymmetry(mixState))
bcpg.plotOn(frame6,RooFit.ProjWData(RooArgSet(mixState),data4,kTRUE),RooFit.Asymmetry(mixState))

data4.plotOn(frame7,RooFit.Asymmetry(tagFlav))
print "Printing PDF asymmetry......" 
bcpg.plotOn(frame7,RooFit.ProjWData(RooArgSet(tagFlav),data4,kTRUE),RooFit.Asymmetry(tagFlav))

mixState.setRange("mixed","mixed") ;

data_reduced = data4.reduce(RooFit.CutRange("mixed"))
data_reduced.plotOn(frame8,RooFit.Asymmetry(tagFlav))

print "Printing PDF asymmetry......" 
bcpg.plotOn(frame8,RooFit.ProjWData(RooArgSet(tagFlav),data_reduced,kTRUE),RooFit.Asymmetry(tagFlav))


c = TCanvas("rf708_bphysics","rf708_bphysics",1200,800) 
c.Divide(2,2) 

c.cd(1)  
gPad.SetLeftMargin(0.15)  
frame6.GetYaxis().SetTitleOffset(1.6)  
frame6.Draw() 

c.cd(2)  
gPad.SetLeftMargin(0.15)  
frame7.GetYaxis().SetTitleOffset(1.6)  
frame7.Draw() 

c.cd(3)  
gPad.SetLeftMargin(0.15)  
frame8.GetYaxis().SetTitleOffset(1.6)  
frame8.Draw() 


################################################################################
## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
################################################################################
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

