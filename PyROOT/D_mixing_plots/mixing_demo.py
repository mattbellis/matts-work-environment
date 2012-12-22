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




c1 = TCanvas("c1", "c1",10,10,900,760)
c1.SetFillColor(0)
c1.Divide(1,1)

amplitude = 0.1
period = 5.0

funcs = []
cmd = "%f*sin(%f*x)" % (amplitude,period)
funcs.append(TF1("f1",cmd,0,10))

cmd = "exp(-x)" 
funcs.append(TF1("f2",cmd,0,10))

cmd = "exp(-x) * %f*sin(%f*x)" % (amplitude,period)
funcs.append(TF1("f3",cmd,0,10))

for i,f in enumerate(funcs):
    if i==0:
        f.SetLineColor(4)
    elif i==1:
        f.SetLineColor(2)
    elif i==2:
        f.SetLineColor(1)

    f.SetLineWidth(4)
    

c1.cd(1) 
funcs[2].Draw()
funcs[1].Draw("same")
funcs[0].Draw("same")
funcs[2].Draw("same")
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

