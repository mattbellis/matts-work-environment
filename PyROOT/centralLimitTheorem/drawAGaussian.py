#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys
import datetime

from ROOT import *

from color_palette import *

batchMode = False

#
# Parse the command line options
#
doFit = False
sample_size = [1]
nTrials = 3
nbins = 100
mean = float(sys.argv[1])
width = float(sys.argv[2])

#if sys.argv[6] == "True":
  #doFit = True
doFit = True
print doFit

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gROOT.Reset()
gStyle.SetOptStat(0)
gStyle.SetOptFit(1111)
gStyle.SetStatH(0.3);                
gStyle.SetStatW(0.25);                
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillStyle(0)
set_palette("palette",100)

########################################
# Set seed for random number generator
########################################
now = datetime.datetime.now()
gRandom.SetSeed(996655)
#gRandom.SetSeed(now.microsecond)

fillcolors = [22, 2, 2, 2]

########################################
# Define the functions
########################################
maxlo = -100.0
maxhi =  100.0
func = []
for i in range(0,4):

  name = "func%d" % (i)
  expression = "exp(-(x-%f)*(x-%f)/(2*%f*%f))" % ( mean, mean, width, width )
  print expression

  if i==0:
    lo = mean - (4-i)*width
    hi = mean + (4-i)*width
  else:
    lo = mean - (i)*width
    hi = mean + (i)*width

  if i==0:
    maxlo = lo
    maxhi = hi

  func.append(TF1("name",expression, lo,hi) )
  func[i].SetFillStyle(1001)
  func[i].SetFillColor( fillcolors[i] )
  if i!= 0:
    func[i].SetLineColor( fillcolors[i] )


canvastitles = ["B"]
canvastitles[0] = "Central limit theorem"

canvastext = []
can = []
toppad = []
bottompad = []
for f in range(0,4):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 600, 600 ))
  can[f].SetFillColor( 0 )
  can[f].Divide(1,1)

  """
  name = "top" + str(f)
  toppad.append(TPad(name, name, 0.01, 0.50, 0.99, 0.99))
  toppad[f].SetFillColor(0)
  toppad[f].Draw()
  toppad[f].Divide(1,nfunc)
  name = "bottom" + str(f)
  bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.50))
  bottompad[f].SetFillColor(0);
  bottompad[f].Draw();
  bottompad[f].Divide(1, 1);
  """

  #toppad[f].cd(1)
  #canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  #canvastext[f].AddText(canvastitles[f])
  #canvastext[f].AddText("")
  #canvastext[f].SetBorderSize(1)
  #canvastext[f].SetFillStyle(1)
  #canvastext[f].SetFillColor(1)
  #canvastext[f].SetTextColor(0)
  #canvastext[f].Draw()



histos = []
text1 = []
for f in range(0,1):
  histos.append([])
  text1.append([])
  for i in range(0, 4):
    histos[f].append([])
    text1[f].append([])

datasettext = ["Data"]
xaxistitle = "Sample/mean value"

#
# 
#

#sample_size = [1, 10, 100]
nSamples = len(sample_size)
f = 0
for i in range(0,4):
  hname = "h" + str(i)
  print maxlo
  print maxhi
  histos[f][i] = TH1F(hname,hname,nbins, maxlo, maxhi)

  # Draw the canvas labels
  can[i].cd(1)

  histos[f][i].SetMaximum(1.2)
  histos[f][i].SetMinimum(0)
  histos[f][i].SetTitle("")
  
  histos[f][i].GetYaxis().SetNdivisions(4)
  histos[f][i].GetXaxis().SetNdivisions(6)
  histos[f][i].GetYaxis().SetLabelSize(0.06)
  histos[f][i].GetXaxis().SetLabelSize(0.06)

  histos[f][i].GetXaxis().CenterTitle()
  histos[f][i].GetXaxis().SetTitleSize(0.09)
  histos[f][i].GetXaxis().SetTitleOffset(1.0)
  histos[f][i].GetXaxis().SetTitle("Measurement")

  histos[f][i].SetFillColor(805)

  histos[f][i].Draw()
  if i==0:
    func[0].Draw("same")
  elif i==1:
    func[0].Draw("same")
    func[1].Draw("same")
  elif i==2:
    func[0].Draw("same")
    func[2].Draw("same")
    func[1].Draw("same")
  elif i==3:
    func[0].Draw("same")
    func[3].Draw("same")
    func[2].Draw("same")
    func[1].Draw("same")

 
  text1[f][i] = TPaveText(0.6, 0.75, 0.99, 1.0, "NDC")
  name = "#pm %d #sigma" % (i)
  text1[f][i].AddText(name)
  text1[f][i].SetBorderSize(1)
  text1[f][i].SetFillStyle(1001)
  text1[f][i].SetFillColor(1)
  text1[f][i].SetTextColor(0)
  if i!=0:
    text1[f][i].Draw()
  

  gPad.Update()


  name = "Plots/gauss_mean%d_width%d_%d.eps" % (mean, width, i)
  can[i].SaveAs(name)


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
