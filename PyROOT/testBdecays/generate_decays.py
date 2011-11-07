#!/usr/bin/env python
#
#

# Import the needed modules
import os
import sys
from array import array

from ROOT import *

batchMode = False

#
# Parse the command line options
#
max_events = float(sys.argv[1])

m_B = 5.279
m_lam = 1.115
#mass_i = 2.289

m_p = 0.938272
m_kc = 0.493
m_pic = 0.139
m_mu = 0.105
m_e = 0.000511

#masses_sub = [ m_p, m_kc, m_pic, m_mu, m_e ]
rnd = TRandom3()


masses_Bdecay = array('d', [ m_lam, m_mu ])
n_Bdecay = len(masses_Bdecay)

masses_lamdecay = array('d', [ m_p, m_pic ])
n_lamdecay = len(masses_lamdecay)

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

####################################
# Make some canvases
####################################
can = []
for f in range(0, 1):
  name = "can" + str(f)
  can.append(TCanvas( name, name, 10+10*f, 10+10*f, 1400, 900 ))
  can[f].SetFillColor( 0 )
  can[f].Divide( 3,3 )

##################################
# Make histograms
##################################
histos = []
for i in range(0, 16):
  hname = "h%d" % (i)
  histos.append( TH1F( hname, hname, 100, 2.0, 6.0))


##############################################
# Start the generation
##############################################
event = TGenPhaseSpace()
event_lam = TGenPhaseSpace()


# Calculate the max weight for this topology
maxweight = 0.0
for i in range(0, 10000):
  pmag = 0.4*rnd.Rndm()
  costh = 2*rnd.Rndm() - 1.0
  #print str(pmag) + " " + str(costh)
  x = sin(acos(costh))*pmag
  y = 0.0
  z = costh*pmag
  initial_B = TLorentzVector( x, y, z, m_B ) 
  if event.SetDecay(initial_B, n_Bdecay, masses_Bdecay, ""):

    # Decay the B
    weight = event.Generate()
    print weight

    # Grab the 4vecs for the Lambda and mu
    p_lam = event.GetDecay(0) # This returns a TLorentzVector
    p_mu = event.GetDecay(1) # This returns a TLorentzVector

    # Decay the Lambda
    if event_lam.SetDecay(p_lam, n_lamdecay, masses_lamdecay, ""):
      weight *= event_lam.Generate()

      # Keep track of the maximum weight for the event.
      if weight > maxweight:
        maxweight = weight

     

############################################################
# Now generate the events
############################################################
n = 0
p_temp = TLorentzVector()
while n < max_events:
  pmag = 0.4*rnd.Rndm()
  costh = 2*rnd.Rndm() - 1.0
  x = sin(acos(costh))*pmag
  y = 0.0
  z = costh*pmag
  initial_B = TLorentzVector( x, y, z, m_B ) 
  if event.SetDecay(initial_B, n_Bdecay, masses_Bdecay, ""):

    # Decay the B
    weight = event.Generate()

    # Grab the 4vecs for the Lambda and mu
    p_lam = event.GetDecay(0) # This returns a TLorentzVector
    p_mu = event.GetDecay(1) # This returns a TLorentzVector

    # Decay the Lambda
    if event_lam.SetDecay(p_lam, n_lamdecay, masses_lamdecay, ""):
      weight *= event_lam.Generate()

      if maxweight*rnd.Rndm() < weight:

        if n%1000 == 0:
          print n

        n += 1

        p_p = event_lam.GetDecay(0)
        p_pi = event_lam.GetDecay(1)

        histos[0].Fill( (p_p + p_pi + p_mu).M() )
        histos[1].Fill( (p_p + p_pi ).M() )

        #print (p_p + p_pi + p_mu).M() 
        #print p_mu.E()
        p_mu.SetXYZM( rnd.Gaus(p_mu.X(), 0.010), rnd.Gaus(p_mu.Y(), 0.010), rnd.Gaus(p_mu.Z(), 0.010), 0.938 )
        p_temp.SetXYZM( rnd.Gaus(p_p.X(), 0.100), rnd.Gaus(p_p.Y(), 0.100), rnd.Gaus(p_p.Z(), 0.100), 0.938 )
        #print p_mu.E()
        #print (p_p + p_pi + p_mu).M() 
        histos[2].Fill( (p_p + p_pi + p_mu  ).M() )
        histos[3].Fill( (p_p + p_pi + p_temp).M() )


#
# Plot the histos
#
for i in range(0, 9):

  # Draw the canvas labels
  can[0].cd(i + 1)
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

  gPad.Update()


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
