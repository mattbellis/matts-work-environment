#!/usr/bin/env python

# example illustrating divided pads and Latex
# Author: Rene Brun

################################################################################
################################################################################

import ROOT
from ROOT import *

import sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-b", "--batch", dest="batch", action = "store_true", default = False, help="Run in batch mode")
parser.add_option("-t", "--tag", dest="tag", default="sm", help="Tag for saved .eps files")
parser.add_option("--hl", action='append', dest="hl", default=[], help="Which one(s) to highlight.")


(options, args) = parser.parse_args()

print options.hl




c1 = TCanvas("c1", "c1",10,10,630,760)
c1.SetFillColor(kBlack)
# Original
#quarkColor  = 50
#leptonColor = 16
#forceColor  = 38

quarkColor  = 16
leptonColor = 16
forceColor  = 16

#titleColor  = kYellow
titleColor  = kCyan

border = 8

highlightColor = 3
highlightForceColor = 5
highlightBorder = 13

texf = TLatex(0.90,0.455,"Force Carriers")
texf.SetTextColor(forceColor)
texf.SetTextAlign(22) 
texf.SetTextSize(0.07) 
texf.SetTextAngle(90)
texf.Draw()
gPad.Update()

texl = TLatex(0.11,0.288,"Leptons")
texl.SetTextColor(leptonColor)
texl.SetTextAlign(22) 
texl.SetTextSize(0.07) 
texl.SetTextAngle(90)
texl.Draw()
gPad.Update()

texq = TLatex(0.11,0.624,"Quarks")
texq.SetTextColor(quarkColor)
texq.SetTextAlign(22) 
texq.SetTextSize(0.07) 
texq.SetTextAngle(90)
texq.Draw()
gPad.Update()

tex = TLatex(0.5,0.5,"u")
tex.SetTextColor(titleColor) 
tex.SetTextFont(32) 
tex.SetTextAlign(22)
tex.SetTextSize(0.14) 
tex.DrawLatex(0.5,0.93,"Elementary")
gPad.Update()
tex.SetTextSize(0.12) 
tex.DrawLatex(0.5,0.84,"Particles")
gPad.Update()
tex.SetTextSize(0.05) 
tex.DrawLatex(0.5,0.067,"Three Generations of Matter")
gPad.Update()

tex.SetTextColor(kBlack) 
tex.SetTextSize(0.8)
     
# -----------.Create main pad and its subdivisions
pad = TPad("pad", "pad",0.15,0.11,0.85,0.79)
pad.Draw()
gPad.Update()
pad.cd()
pad.Divide(4,4,0.0003,0.0003)

pad.cd(1) 
gPad.SetFillColor(quarkColor)   
gPad.SetBorderSize(border)
if "1" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"u")
gPad.Update()

pad.cd(2) 
gPad.SetFillColor(quarkColor)   
gPad.SetBorderSize(border)
if "2" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"c")
gPad.Update()

pad.cd(3) 
gPad.SetFillColor(quarkColor)   
gPad.SetBorderSize(border)
if "3" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"t")
gPad.Update()

pad.cd(4) 
gPad.SetFillColor(forceColor)   
gPad.SetBorderSize(border)
if "4" in options.hl:
    gPad.SetFillColor(highlightForceColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.55,"#gamma")
gPad.Update()

pad.cd(5) 
gPad.SetFillColor(quarkColor)   
gPad.SetBorderSize(border)
if "5" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"d")
gPad.Update()

pad.cd(6) 
gPad.SetFillColor(quarkColor)   
gPad.SetBorderSize(border)
if "6" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"s")
gPad.Update()

pad.cd(7) 
gPad.SetFillColor(quarkColor)   
gPad.SetBorderSize(border)
if "7" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"b")
gPad.Update()

pad.cd(8) 
gPad.SetFillColor(forceColor)   
gPad.SetBorderSize(border)
if "8" in options.hl:
    gPad.SetFillColor(highlightForceColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.55,"g")
gPad.Update()

pad.cd(9) 
gPad.SetFillColor(leptonColor)  
gPad.SetBorderSize(border)
if "9" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"#nu_{e}")
gPad.Update()

pad.cd(10) 
gPad.SetFillColor(leptonColor) 
gPad.SetBorderSize(border)
if "10" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"#nu_{#mu}")
gPad.Update()

pad.cd(11) 
gPad.SetFillColor(leptonColor) 
gPad.SetBorderSize(border)
if "11" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"#nu_{#tau}")
gPad.Update()

pad.cd(12) 
gPad.SetFillColor(forceColor)  
gPad.SetBorderSize(border)
if "12" in options.hl:
    gPad.SetFillColor(highlightForceColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"Z")
gPad.Update()

pad.cd(13) 
gPad.SetFillColor(leptonColor) 
gPad.SetBorderSize(border)
if "13" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"e")
gPad.Update()

pad.cd(14) 
gPad.SetFillColor(leptonColor) 
gPad.SetBorderSize(border)
if "14" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.56,"#mu")
gPad.Update()

pad.cd(15) 
gPad.SetFillColor(leptonColor) 
gPad.SetBorderSize(border)
if "15" in options.hl:
    gPad.SetFillColor(highlightColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"#tau")
gPad.Update()

pad.cd(16) 
gPad.SetFillColor(forceColor)  
gPad.SetBorderSize(border)
if "16" in options.hl:
    gPad.SetFillColor(highlightForceColor)
    gPad.SetBorderSize(highlightBorder)
tex.DrawLatex(.5,.5,"W")
gPad.Update()

c1.cd()
gPad.Update()

name = "Plots/sm_%s.eps" % (options.tag)
c1.SaveAs(name)


## Wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not options.batch):
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

