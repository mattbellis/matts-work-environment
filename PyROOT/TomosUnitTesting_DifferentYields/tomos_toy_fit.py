from ROOT import *
from math import sqrt
import sys

def defaultHistoSettings(h):
    h.SetNdivisions(6)
    h.GetYaxis().SetTitleSize(0.09)
    h.GetYaxis().SetTitleFont(42)
    h.GetYaxis().SetTitleOffset(0.7)
    h.GetYaxis().CenterTitle()
    h.GetYaxis().SetNdivisions(6)
#    h.GetYaxis().SetTitle("events/MeV")
    h.SetFillColor(9)

def defaultPadSettings():
    
    gPad.SetFillColor(0)
    gPad.SetBorderSize(0)
    gPad.SetRightMargin(0.20);
    gPad.SetLeftMargin(0.20);
    gPad.SetBottomMargin(0.15);

def waitForInput():
    rep = ''
    while not rep in [ 'c', 'C' ]:
        rep = raw_input( 'enter "c" to continue: ' )
        if 1 < len(rep):
            rep = rep[0]

def maxBinContent(histogram,numBins):
    maxContent = -9999999
    for bin in range(0,numBins):
        content = histogram.GetBinContent(bin)
        if content>maxContent:
            maxContent = content
    return maxContent

if len(sys.argv)<2:
    print "Usage: python "+sys.argv[0]+" <random seed> [fixed n3 value]"
    exit()

###################
###################
###################
drawFit = True
nEvents = 51509
###################
###################
###################

randSeed = int(sys.argv[1])

floatN3 = True
if len(sys.argv)==3:
    fixedN3 = float(sys.argv[2])
    floatN3 = False

fitmin = -5
fitmax = 5

nbins = 75

time = RooRealVar("time","time",1.0,fitmin,fitmax)
timeVars = RooArgSet(time)

mean1 = RooRealVar("mean1","mean1",0.0,-1,1)
mean2 = RooRealVar("mean2","mean2",0.1,-1,1)
mean3 = RooRealVar("mean3","mean3",-0.2,-1,1)
sigma1 = RooRealVar("sigma1","sigma1",0.3,0.0001,12.0)
sigma2 = RooRealVar("sigma2","sigma2",0.6,0.1,4)
sigma3 = RooRealVar("sigma3","sigma3",1.0,0.2,50)

n1 = None
n2 = None
n3 = None
if floatN3:
    n1 = RooRealVar("n1","Coefficient of gaussian 1",5*nEvents/10.0,0,nEvents*2)
    n2 = RooRealVar("n2","Coefficient of gaussian 2",3*nEvents/10.0,0,nEvents*2)
    n3 = RooRealVar("n3","Coefficient of gaussian 3",2*nEvents/10.0,0,nEvents*2)
else:
    remaining = (nEvents-fixedN3)*1.0
    n1Evt = 5.0/8.0*remaining
    n2Evt = 3.0/8.0*remaining
    n1 = RooRealVar("n1","Coefficient of gaussian 1",n1Evt,0,nEvents*2)
    n2 = RooRealVar("n2","Coefficient of gaussian 2",n2Evt,0,nEvents*2)
    n3 = RooRealVar("n3","Coefficient of gaussian 3",fixedN3)

gauss1 = RooGaussModel("gauss1","Unconstrained #DeltaT Gaussian 1",time, mean1, sigma1)
gauss2 = RooGaussModel("gauss2","Unconstrained #DeltaT Gaussian 2",time, mean2, sigma2)
gauss3 = RooGaussModel("gauss3","Unconstrained #DeltaT Gaussian 3",time, mean3, sigma3)

extendedGauss1 = RooExtendPdf("extendedGauss1","Extended Gaussian 1",gauss1,n1)
extendedGauss2 = RooExtendPdf("extendedGauss2","Extended Gaussian 2",gauss2,n2)
extendedGauss3 = RooExtendPdf("extendedGauss3","Extended Gaussian 3",gauss3,n3)

total = RooAddPdf("total","total",RooArgList(extendedGauss1,extendedGauss2,extendedGauss3))

print "Setting random seed to "+str(randSeed)
randGen = RooRandom.randomGenerator()
randGen.SetSeed(randSeed)

print "Generating toy data..."
Dataset = total.generate(timeVars)

print "Fitting to toy data..."
fit_result = total.fitTo(Dataset,RooFit.Extended(kTRUE),RooFit.Save(kTRUE))
#fit_result = total.fitTo(Dataset,RooFit.Extended(kFALSE),RooFit.Save(kTRUE))

finalVals = fit_result.floatParsFinal()

n1Index = finalVals.index("n1")
n1Var   = finalVals.at(n1Index)
n1Val   = n1Var.getVal()
n1Error = n1Var.getError()

n2Index = finalVals.index("n2")
n2Var   = finalVals.at(n2Index)
n2Val   = n2Var.getVal()
n2Error = n2Var.getError()

n3Val = None
n3Error = 0.0
if floatN3:
    n3Index = finalVals.index("n3")
    n3Var   = finalVals.at(n3Index)
    n3Val   = n3Var.getVal()
    n3Error = n3Var.getError()
else:
    n3Val = fixedN3

# sigmaErrorLo = sigmaVar.getErrorLo()
# sigmaErrorHi = sigmaVar.getErrorHi()

# print "Final nsig: "+str(nsigVal)+" +/- "+str(nsigError)
# print "Input Nsig: "+str(nEvents)

status = fit_result.status()

sumVal = n1Val+n2Val+n3Val
print "Final n1+n2+n3: "+str(sumVal)

print "------------------------------------------------"
dataSum = Dataset.numEntries()
print "Input NEvents: "+str(dataSum)
print "Fit NEvents: "+str(sumVal)
print "Total Difference: "+str(sumVal-dataSum)
print "Total Percentage Difference: "+str((sumVal-dataSum)/(dataSum*1.0)*100)
print "------------------------------------------------"

print "Status: "+str(status)

if drawFit:

    frame = time.frame(fitmin,fitmax,nbins)
    frame.SetTitle("")
    frame.GetYaxis().SetTitleOffset(1.2)
    
    plotArguments = RooLinkedList()
    plotArguments.Add(RooFit.MarkerColor(2))
    Dataset.plotOn(frame, plotArguments)
    
    total.plotOn(frame)
    
    canvas = TCanvas("c1", "c1", 10, 10, 1200, 800)
    canvas.SetFillColor(0)
    canvas.SetRightMargin(0.01)
    canvas.SetTopMargin(0.05)
    canvas.SetLeftMargin(0.085)
    
    gPad.SetFillColor(0)
    gPad.SetBorderSize(0)
    gPad.SetRightMargin(0.20)
    gPad.SetLeftMargin(0.20)
    gPad.SetBottomMargin(0.15)
    
    canvas.cd()
    
    frame.Draw()
    gPad.Update()

fit_result.Print("v")

if drawFit:
    waiting = raw_input('Press enter to exit:')
