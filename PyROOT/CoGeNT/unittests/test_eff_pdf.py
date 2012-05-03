#!/usr/bin/env python

from ROOT import *

#x = RooRealVar("x","ionization energy (keVee)",0.0,4.0)
x = RooRealVar("x","ionization energy (keVee)",0.0,2.0)
t = RooRealVar("t","time",1.0,500)

#Etrig = [0.47278, 0.52254, 0.57231, 0.62207, 0.67184, 0.72159, 0.77134, 0.82116, 0.87091, 0.92066, 10.0] # Change to 10
#efftrig = [0.71747, 0.77142, 0.81090, 0.83808, 0.85519, 0.86443, 0.86801, 0.86814, 0.86703, 0.86786, 0.86786]

######## Same as Nicole's
#Etrig = [0.47278, 0.52254, 0.57231, 0.62207, 0.67184, 0.72159, 0.77134, 0.82116, 0.87091, 0.92066, 4.0] # Changed to 4.0
#efftrig = [0.71747, 0.77142, 0.81090, 0.83808, 0.85519, 0.86443, 0.86801, 0.86814, 0.86703, 0.86786, 0.86786]

# Looking for a more dramatic change
Etrig = [0.47278, 0.52254, 0.57231, 0.62207, 0.67184, 0.72159, 0.77134, 0.82116, 0.87091, 0.90, 0.92066, 4.0] # Changed to 4.0
efftrig = [0.21747, 0.37142, 0.41090, 0.53808, 0.55519, 0.56443, 0.56801, 0.56814, 0.55703, 0.558, 0.9, 0.96786]
#Etrig = [0.47278, 0.52254, 0.57231, 0.62207, 0.67184, 0.72159, 0.77134, 0.82116, 0.87091, 0.90, 0.92066, 2.0] # Changed to 4.0
#efftrig = [0.11747, 0.17142, 0.11090, 0.13808, 0.75519, 0.26443, 0.26801, 0.26814, 0.25703, 0.558, 0.9, 0.96786]

neff = len(Etrig)

print len(Etrig)
print len(efftrig)

bin_centers = []
bin_heights = []
step = 0.010
for i in range(0,neff-1):

    x0 = Etrig[i]
    x1 = Etrig[i+1]
    y0 = efftrig[i]
    y1 = efftrig[i+1]

    # Once we're above 1 keVee the efficiency is flat.
    if x0>1.0:
        step = 1.0

    #print "%f %f %f %f" % (x0,x1,y0,y1)
    slope = (y1-y0)/(x1-x0)

    start = x0 
    stop  = x1
    nsteps = 0
    while start <= stop: 
        val = y0 + slope*(step*nsteps)
        #print "%f %f %f" % (start,val,slope)
        bin_centers.append(start)
        bin_heights.append(val)
        start += step
        nsteps += 1

################################################################################
# Create the RooParametricStepFunction
################################################################################
nbins = len(bin_centers)
# These are the bin edges
limits = TArrayD(nbins+1)
for i in range(0, nbins):
    limits[i] = bin_centers[i]
limits[nbins] = bin_centers[i]+0.1


# These will hold the values of the bin heights
#scaling = 12.5 # Need to figure out how to do this properly. 
#scaling = 1.0 # Need to figure out how to do this properly. 
scaling = 0.02 # Need to figure out how to do this properly. 
list = RooArgList("list")
binHeight = []
for i in range(0,nbins-1):
    name = "binHeight%d" % (i)
    title = "bin %d Value" % (i)
    binHeight.append(RooRealVar(name, title, scaling*bin_heights[i]))
    list.add(binHeight[i]) # up to binHeight8, ie. 9 parameters

aPdf = RooParametricStepFunction("aPdf","PSF", x, list, limits, nbins)

data1 = aPdf.generate(RooArgSet(x),2000) 
kest1 = RooKeysPdf("kest1","kest1",x,data1,RooKeysPdf.MirrorBoth) ;


################################################################################
# Make the RooEfficiency PDF
################################################################################
# Acceptance state cut (1 or 0)
cut = RooCategory("cut","cutr")
cut.defineType("accept",1)
cut.defineType("reject",0)

# Construct efficiency p.d.f eff(cut|x)
#effPdf = RooEfficiency("effPdf","effPdf",aPdf,cut,"accept") 
effPdf = RooEfficiency("effPdf","effPdf",kest1,cut,"accept") 

################################################################################

mean0 = RooRealVar("mean0","mean0",0.8)
#sigma0 = RooRealVar("sigma0","sigma0",0.08)
sigma0 = RooRealVar("sigma0","sigma0",0.28)
gauss0 = RooGaussian("gauss0","gauss0",x,mean0,sigma0)

mean1 = RooRealVar("mean1","mean1",1.3)
sigma1 = RooRealVar("sigma1","sigma1",0.08)
gauss1 = RooGaussian("gauss1","gauss1",x,mean1,sigma1)

slope0 = RooRealVar("slope0","slope0",-0.005)
slope1 = RooRealVar("slope1","slope1",-0.02)

decay0 = RooExponential("decay0","decay0",t,slope0)
decay1 = RooExponential("decay1","decay1",t,slope1)

prod0 = RooProdPdf("prod0","prod0",RooArgList(decay0,gauss0))
prod1 = RooProdPdf("prod1","prod1",RooArgList(decay1,gauss1))

n0 = RooRealVar("n0","n0",1000)
#n1 = RooRealVar("n1","n1",500)
n1 = RooRealVar("n1","n1",1000)

total = RooAddPdf("total","total",RooArgList(prod0,prod1),RooArgList(n0,n1))

x.setBins(50)
t.setBins(50)
frame_x = x.frame(RooFit.Title("x"))
frame_t = t.frame(RooFit.Title("t"))
frame_xeff = x.frame(RooFit.Title("xeff"))

# Multiply times efficiency.
modelEffProd = RooEffProd("modelEffProd","modelEffProd",total,effPdf)
#modelEffProd = RooProdPdf("modelEffProd","modelEffProd",RooArgSet(total),RooFit.Conditional(RooArgSet(effPdf),RooArgSet(cut))) ;


#data = total.generate(RooArgSet(x,t),1000)
data = modelEffProd.generate(RooArgSet(x,t),1000)

n0.setConstant(False)
n1.setConstant(False)
sigma0.setConstant(False)
#fit_results = total.fitTo(data,RooFit.Verbose(True),RooFit.Extended(True),RooFit.Save(True))
#fit_results = modelEffProd.fitTo(data,RooFit.Verbose(True),RooFit.Extended(True),RooFit.Save(True))
fit_results = modelEffProd.fitTo(data,RooFit.Verbose(True),RooFit.Save(True))

print fit_results
fit_results.Print("v")



data.plotOn(frame_x)
data.plotOn(frame_t)

can = TCanvas("can","can",10,10,1200,400)
can.SetFillColor(0)
can.Divide(3,1)

can.cd(1)
#rargset = RooArgSet(aPdf)
rargset = RooArgSet(kest1)
#aPdf.plotOn(frame_xeff,RooFit.Components(rargset),RooFit.LineColor(3))
effPdf.plotOn(frame_xeff,RooFit.Components(rargset),RooFit.LineColor(3))
frame_xeff.Draw()
gPad.Update()

can.cd(2)
rargset = RooArgSet(prod0)
modelEffProd.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(4))
rargset = RooArgSet(prod1)
modelEffProd.plotOn(frame_x,RooFit.Components(rargset),RooFit.LineColor(2))
frame_x.Draw()
gPad.Update()

can.cd(3)
rargset = RooArgSet(total)
modelEffProd.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(3))
rargset = RooArgSet(prod0)
modelEffProd.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(4))
rargset = RooArgSet(prod1)
modelEffProd.plotOn(frame_t,RooFit.Components(rargset),RooFit.LineColor(2))
frame_t.Draw()
gPad.Update()

############################################################################
rep = ''
while not rep in ['q','Q']:
    rep = raw_input('enter "q" to quit: ')
    if 1<len(rep):
        rep = rep[0]

