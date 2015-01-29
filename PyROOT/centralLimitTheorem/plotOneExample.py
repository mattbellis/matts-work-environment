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
sample_size[0] = int(sys.argv[1])
maxsamplings = int(sys.argv[2])
update_frequency = int(sys.argv[3])
nbins = int(sys.argv[4])
string_of_distributions = sys.argv[5]
tag = sys.argv[6]

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
gStyle.SetOptStat(10)
gStyle.SetOptFit(1111)
#gStyle.SetOptStat(110010)
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

########################################
# Define the functions
########################################
func = []
func.append(TF1("func0","1",0,10) )
func.append(TF1("func1","x",0,10) )
func.append(TF1("func2","abs(x-5)",0,10) )
func.append(TF1("func3","sin(x)*sin(x)",0,10) )
func.append(TF1("func4","exp(-x/4)",0,10) )
func.append(TF1("func5","cos(x)*cos(x)*exp(-x/4)",0,10) )
func.append(TF1("func6","1",0,2) )
func.append(TF1("func7","1",4,6) )
func.append(TF1("func8","1",7,10) )


functions_to_use = []
list_of_distributions = string_of_distributions.split(",")
functag = "funcs"
for f in list_of_distributions:
  print f
  functions_to_use.append( int(f) )
  functag += f

nfunc = len(functions_to_use)

lo = 0.0
hi = 10.0
stepsize = (hi - lo)/100.0
max = []
for f in range(0,nfunc):
  max.append(0)
  for i in range(0, 100):
    x = lo + stepsize*i
    y = func[functions_to_use[f]].Eval(x)
    if max[f] < y:
      max[f] = y

  print max[f]

canvastitles = ["B"]
canvastitles[0] = "Central limit theorem"

canvastext = []
can = []
toppad = []
bottompad = []
for f in range(0,1):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 600, 900 ))
  can[f].SetFillColor( 0 )
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
for f in range(0,nTrials):
  histos.append([])
  text1.append([])
  for i in range(0, 1 + nfunc):
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
for i in range(0,nSamples + nfunc):
  hname = "h" + str(i)
  #print hname
  histos[f][i] = TH1F(hname,hname,nbins, 0,10)

  # Draw the canvas labels
  bottompad[0].cd(f+1)
  histos[f][i].SetMinimum(0)
  histos[f][i].SetTitle("")
  
  histos[f][i].GetYaxis().SetNdivisions(4)
  histos[f][i].GetXaxis().SetNdivisions(6)
  histos[f][i].GetYaxis().SetLabelSize(0.06)
  histos[f][i].GetXaxis().SetLabelSize(0.06)

  histos[f][i].GetXaxis().CenterTitle()
  histos[f][i].GetXaxis().SetTitleSize(0.09)
  histos[f][i].GetXaxis().SetTitleOffset(1.0)
  if i!=nSamples + nfunc - 1:
    histos[f][i].GetXaxis().SetTitle("Parent distribution")
  else:
    histos[f][i].GetXaxis().SetTitle("Samples")

  histos[f][i].SetFillColor(805)

  histos[f][i].GetXaxis().SetLimits(-2.0, 12.0)

  if(i==0):
    histos[f][i].DrawCopy()
  else:
    histos[f][i].DrawCopy("same")
      
  """
  bottompad[0].cd(f + 1)
  text1[f][i] = TPaveText(0.0, 0.9, 0.4, 1.0, "NDC")
  text1[f][i].AddText(filename[f])
  text1[f][i].SetBorderSize(1)
  text1[f][i].SetFillStyle(1)
  text1[f][i].Draw()
  """

  can[0].Update()


##########################################
# Do the samples
##########################################
for f in range(0, nfunc):
  toppad[0].cd(1 + f) 
  histos[0][f].SetMaximum(1.5 * max[f])
  histos[0][f].Draw()
  func[functions_to_use[f]].SetFillStyle(1001)
  func[functions_to_use[f]].SetFillColor(2)
  func[functions_to_use[f]].Draw("same")


###########################################

plot_count = 0
for i in range(0,maxsamplings + 1):
  for h in range(nfunc, nfunc + 1):
    bottompad[0].cd(1) 
    tot = 0
    if i>0:
      for f in range(0, nfunc):
        for j in range(0, sample_size[h-nfunc]):
          tot += func[functions_to_use[f]].GetRandom()
          #print tot
      mean = tot / float( nfunc * sample_size[0] )
      histos[0][h].Fill(mean)

    histos[0][h].Draw()
  if i%update_frequency == 0:
    can[0].Update()
    name = "Plots/can_%s_%s_samplesize%d_nbins%d_maxsamplings%d_updatefreq%d_plotcount%d.eps" % (tag, functag, sample_size[0], nbins, maxsamplings, update_frequency, plot_count )
    can[0].SaveAs(name)
    plot_count += 1

if doFit:
  bottompad[0].cd(1)
  histos[0][nfunc].Fit("gaus")
  histos[0][nfunc].GetFunction("gaus").SetLineStyle(1)
  histos[0][nfunc].GetFunction("gaus").SetLineWidth(5)
  histos[0][nfunc].GetFunction("gaus").SetLineColor(4)
  can[0].Update()
  name = "Plots/can_%s_%s_samplesize%d_nbins%d_maxsamplings%d_updatefreq%d_plotcount%d.eps" % (tag, functag, sample_size[0], nbins, maxsamplings, update_frequency, plot_count )
  can[0].SaveAs(name)


"""
for j in range(0,1):
  name = "plots/can" + str(j) + "_" + whichType + ".ps" 
  can[j].SaveAs(name)
"""

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
