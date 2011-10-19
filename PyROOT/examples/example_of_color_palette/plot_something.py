#!/usr/bin/env python
#
#

# Import the needed modules
import sys

from ROOT import *

from color_palette import *

nevents = int(sys.argv[1])

################################################################################
################################################################################

#gStyle.SetPalette(1)
set_palette("palette",99)

################################################################################
# Make the canvas
################################################################################
can = []
for f in range(0,1):
    name = "can%d" % (f)
    can.append(TCanvas( name, name, 10+10*f, 10+10*f,800,800))
    can[f].SetFillColor( 0 )
    can[f].SetFillStyle(1001)
    can[f].Divide(2,2)

    can[f].Update()


################################################################################
# Make the histogram
################################################################################
histos = []
for f in range(0,1):
    name = "h2d_%d" % (f)
    histos.append(TH2F(name,name,50,-10,10, 50, -10,10))

    histos[f].SetMinimum(0)


################################################################################
# Fill the histogram
################################################################################
rnd = TRandom3()
for i in xrange(nevents): 
    x = rnd.Gaus(0.0,3.0)
    y = rnd.Gaus(0.0,3.0)

    histos[0].Fill(x,y)


################################################################################
# Draw the histogram with various options
################################################################################
for i in range(0,2):
    can[0].cd(i+1)
    if i==0:
        histos[0].Draw("colz")
    elif i==1:
        histos[0].Draw("contz")
    elif i==2:
        histos[0].Draw("surf1")
    elif i==3:
        histos[0].Draw("lego2")


    gPad.Update()


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]
