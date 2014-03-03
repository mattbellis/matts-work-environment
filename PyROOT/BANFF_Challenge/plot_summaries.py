#!/usr/bin/env python

from ROOT import *
import sys

################################################################################
# Parse the command line and read in the input files
################################################################################
filenames = []
filenames.append(sys.argv[1])

files = []
files.append(open(filenames[0],"r"))

nbootstraps = 0
distance = 0.0

tag = "default"
if len(sys.argv)>2:
    tag = sys.argv[2]

batch = False
if len(sys.argv)>4:
    if sys.argv[4]=="batch":
        batch = True

################################################################################
################################################################################

################################################################################
gStyle.SetPadLeftMargin(0.15)
gStyle.SetPadBottomMargin(0.15)
gStyle.SetOptStat(0)
################################################################################

################################################################################
# Make the histograms
################################################################################
h = []
nhist = 2
nbins = 3
hlo = 0
hhi = 3
xaxistitle = "#sqrt{2#Delta NLL}"
yaxistitle = "Arbitrary units"
colors = [1,2,4,8]

for i in range(0,nhist):
    h.append([])
    for j in range(0,4):
        name = "h%d_%d" % (i,j)
        h[i].append(TH1F(name,"",nbins,hlo,hhi))
        h[i][j].GetYaxis().SetNdivisions(6)
        h[i][j].GetYaxis().SetLabelSize(0.03)
        h[i][j].GetYaxis().CenterTitle()
        h[i][j].GetYaxis().SetTitleSize(0.05)
        h[i][j].GetYaxis().SetTitleOffset(1.1)
        if i==0:
            yaxistitle = "99% #sigma for toys with 0 signal"
        else:
            if tag=='toys':
                yaxistitle = "Discriminating power"
            else:
                yaxistitle = "Fraction of datasets with signal"
        h[i][j].GetYaxis().SetTitle(yaxistitle)
      
        h[i][j].GetXaxis().SetLabelSize(0.09)
        h[i][j].GetXaxis().SetNdivisions(8)
        h[i][j].GetXaxis().CenterTitle()
        h[i][j].GetXaxis().SetTitleSize(0.05)
        h[i][j].GetXaxis().SetTitleOffset(1.0)
        #h[i][j].GetXaxis().SetTitle(xaxistitle)

        h[i][j].SetLineWidth(2)
        h[i][j].SetMarkerStyle(20)
        h[i][j].SetMarkerSize(2)

        h[i][j].SetBit(TH1.kCanRebin);
        #h[i][j].LabelsDeflate("X");
        #h[i][j].LabelsDeflate("Y");
        #h[i][j].LabelsOption("v");

        h[i][j].SetFillColor(colors[j])
        h[i][j].SetMarkerColor(colors[j])



nfail_status = [[0,0], [0,0]]
npass_status = [[0,0], [0,0]]
sigs = [[],[]]

for i,file in enumerate(files):
    
    for line in file:

        ############################################################################
        # Count pass/fail status
        vals = line.split()
        
        nbs = vals[1]
        r = vals[2]
        sig99 = float(vals[3])
        power = float(vals[4])
        
        #name = 'nbs: %s   r: %s' % (nbs,r)
        name = 'r = %s' % (r)
        index = 0
        if nbs=='0':
            index=0
        if nbs=='10':
            index=1
        elif nbs=='100':
            index=2
        elif nbs=='1000':
            index=3
        print "index: %d" % (index)
        for j in range(0,4):
            if j==index:
                h[0][j].Fill(name, sig99)
                h[1][j].Fill(name, power)
                print "filling %s %f" % (name,sig99)
            else:
                h[0][j].Fill(0.0, 0.0)
                h[1][j].Fill(0.0, 0.0)



        


##########################################################
# Make the histograms
##########################################################

can = []
ncans = 2
for i in range(ncans):
    name = "can%d" % (i)
    can.append(TCanvas(name,name,10+10*i,10+10*i,800,500))
    can[i].SetFillColor(0)
    can[i].Divide(1,1)

lines = []
text1 = []
legend = []
leglabels = ["# bootstraps: 0", "# bootstraps: 10", "# bootstraps: 100", "# bootstraps: 1000"]
for i in range(0,nhist):
    legend.append(TLegend(0.65,0.70,0.99,0.99))
    for j in range(0,4):
        can[i].cd(1)
        if i==0:
            h[i][j].SetMinimum(2.0)
            h[i][j].SetMaximum(3.8)
        else:
            h[i][j].SetMinimum(0.92)
            h[i][j].SetMaximum(0.98)
            if tag=='data':
                h[i][j].SetMinimum(0.07)
                h[i][j].SetMaximum(0.13)

        h[i][j].LabelsDeflate("X");
        h[i][j].LabelsDeflate("Y");
        #h[i][j].LabelsOption("v");

        if j==0:
            h[i][j].Draw("p")
        else:
            h[i][j].Draw("psame")

        if not (tag=='data' and (j==0 or j==1)):
            legend[i].AddEntry(h[i][j],leglabels[j],"f")

        if j==3:
            legend[i].SetFillColor(0)
            legend[i].Draw()

        gPad.Update()

for i,c in enumerate(can):
    name = "Plots/can_summary_%s_%d.eps" % (tag,i)
    c.SaveAs(name)


################################################################################
##########################################################
if (not batch):
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

