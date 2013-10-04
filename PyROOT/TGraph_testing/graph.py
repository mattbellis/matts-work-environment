#!/usr/bin/env python
#
# http://root.cern.ch/root/html/TGraphErrors.html
#
# http://root.cern.ch/root/html/TGraphPainter.html
#

from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *

from array import *

gROOT.Reset()

c1 = TCanvas( 'c1', 'A Simple Graph Example', 200, 10, 900, 900 )
c1.SetFillColor( 42 )
c1.Divide(2,2)
c1.SetGrid()

n = 20 # Number of data points in graph

x, y = array( 'd' ), array( 'd' )
xerr, yerr = array( 'd' ), array( 'd' )

for i in range( n ):
   x.append( 0.1*i )
   y.append( 10*sin( x[i]+0.2 ) )
   xerr.append(0.05)
   yerr.append(0.05*i)
   print ' i %i %f %f ' % (i,x[i],y[i])

gr = TGraphErrors( n, x, y, xerr, yerr )
gr.SetLineColor( 1 )
gr.SetLineWidth( 1 )
gr.SetMarkerColor( 4 )
gr.SetMarkerStyle( 21 )
gr.SetTitle( 'a simple graph' )
gr.GetXaxis().SetTitle( 'X title' )
gr.GetYaxis().SetTitle( 'Y title' )

c1.cd(1)
c1.cd(1).SetGrid()
gr.Draw( 'abx' )
gPad.Update()

c1.cd(2)
gr.Draw( 'ap' )
gPad.Update()

c1.cd(3)
gr.Draw( 'ap2' )
gPad.Update()

c1.cd(4)
gr.SetFillColor(3)
gr.SetFillStyle(3001)
gr.Draw( 'ap4' )
gPad.Update()

# TCanvas.Update() draws the frame, after which one can change it
c1.Update()
c1.GetFrame().SetFillColor( 21 )
c1.GetFrame().SetBorderSize( 12 )
c1.Modified()
c1.Update()

## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

