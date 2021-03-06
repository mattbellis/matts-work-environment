#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys
from array import array

from ROOT import *

#from color_palette import *

batchMode = False

#
# Parse the command line options
#
max_events = float(sys.argv[1])

#mass_i = 5.279
mass_i = 2.289

m_p = 0.938272
m_kc = 0.493
m_pic = 0.139
m_mu = 0.105
m_e = 0.000511

masses_sub = [ m_p, m_kc, m_pic, m_mu, m_e ]

initial = TLorentzVector( 0.0, 0.0, 0.0, mass_i ) 
masses_f = array('d', [ m_p, m_kc, m_pic ])
n_f = len(masses_f)

#
# Last argument determines batch mode or not
#
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True

gROOT.Reset()
gStyle.SetOptStat(10)
#gStyle.SetOptStat(110010)
gStyle.SetStatH(0.3);                
gStyle.SetStatW(0.25);                
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)
set_palette("palette",100)

canvastitles = ["B"]
canvastitles[0] = "B^{0} #rightarrow #Lambda_{C}^{+} #mu^{-}"

canvastext = []
can = []
toppad = []
bottompad = []
for f in range(0, 5**(n_f-2)):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 1400, 900 ))
  can[f].SetFillColor( 0 )
  #can[f].Divide( 5,6 )
  name = "top" + str(f)
  toppad.append(TPad(name, name, 0.01, 0.85, 0.99, 0.99))
  toppad[f].SetFillColor(0)
  toppad[f].Draw()
  toppad[f].Divide(2,2)
  name = "bottom" + str(f)
  bottompad.append(TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.86))
  bottompad[f].SetFillColor(0);
  bottompad[f].Draw();
  bottompad[f].Divide(5, 5);

  toppad[f].cd(1)
  canvastext.append(TPaveText(0.0, 0.0, 1.0, 1.0,"NDC"))
  #canvastext[f].AddText(canvastitles[f])
  canvastext[f].AddText("stuff")
  canvastext[f].AddText("")
  canvastext[f].SetBorderSize(1)
  canvastext[f].SetFillStyle(1)
  canvastext[f].SetFillColor(1)
  canvastext[f].SetTextColor(0)
  canvastext[f].Draw()

##################################
# Make histograms
##################################
histos = []
for i in range(0, 5**n_f):
  hname = "h%d" % (i)
  histos.append( TH1F( hname, hname, 100, 0.8*mass_i, 1.2*mass_i ) )

#for i in range(0, n_f):
  #histos.append([])
  #for j in range(0, 5):
    #hname = "h%d_%d" % (i, j)
    #histos[i].append( TH1F( hname, hname, 100, 0.8*mass_i, 1.2*mass_i ) )



##############################################
# Start the generation
##############################################
event = TGenPhaseSpace()
rnd = TRandom3()

#bvec = -1.0 * initial.BoostVector()
bvec = TVector3(-initial.BoostVector().X(), -initial.BoostVector().Y(), -initial.BoostVector().Z()) 

# Calculate the max weight for this topology
maxweight = 0.0
for i in range(0, 10000):
  if event.SetDecay(initial, n_f, masses_f, ""):
    weight = event.Generate()
    if weight > maxweight:
      maxweight = weight

     

num_combos = 5**n_f
# Generate our events
m2 = [0.0, 0.0, 0.0]
p_f = []
for i in range(0, n_f):
  p_f.append(TLorentzVector())
n = 0
while n < max_events:
  if event.SetDecay(initial, n_f, masses_f, ""):
    weight = event.Generate()
    if maxweight*rnd.Rndm() < weight:

      if n%1000 == 0:
        print n

      n += 1
      for i in range(0,n_f):
        p_f[i] = event.GetDecay(i)

      # Try the different mass combinations.
      for f in range(0, num_combos):
        for i in range(0, n_f):
          index = (f/(5**i))%5
          p_f[i].SetE( sqrt( p_f[i].Rho()*p_f[i].Rho() + masses_sub[ index ]*masses_sub[ index ] ) )

        initial_new = TLorentzVector(0, 0, 0, 0)
        for i in range(0, n_f):
          initial_new += p_f[i]

        histos[f].Fill( initial_new.M() )


#
# Open a ROOT file and save the formula, function and histogram
#
for i in range(0, 5**n_f):

  # Draw the canvas labels
  can_index = i/25
  pad_index = i%25 + 1
  bottompad[ can_index ].cd( pad_index )
  histos[i].SetMinimum(0)
  histos[i].SetTitle("")
  
  histos[i].GetYaxis().SetNdivisions(4)
  histos[i].GetXaxis().SetNdivisions(6)
  histos[i].GetYaxis().SetLabelSize(0.06)
  histos[i].GetXaxis().SetLabelSize(0.06)

  histos[i].GetXaxis().CenterTitle()
  histos[i].GetXaxis().SetTitleSize(0.09)
  histos[i].GetXaxis().SetTitleOffset(1.0)
  histos[i].GetXaxis().SetTitle("Mass (GeV/c^{2})")

  histos[i].SetFillColor(22)
  histos[i].SetLineColor(2)
  histos[i].SetLineWidth(2)

  histos[i].DrawCopy()

  can[can_index].Update()



#for j in range(0,1):
#name = "Plots/can" + str(j) + "_" + whichType + ".ps" 
#can[j].SaveAs(name)

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
