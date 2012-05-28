#!/usr/bin/env python

from ROOT import *
import sys

################################################################################
# Parse the command line and read in the input files
################################################################################
filenames = []
filenames.append(sys.argv[1])
filenames.append(sys.argv[2])

files = []
files.append(open(filenames[0],"r"))
files.append(open(filenames[1],"r"))

nbootstraps = 0
distance = 0.0

pos0 = filenames[0].find('nbs')+3
pos1 = filenames[0].find('_',pos0)
nbootstraps = filenames[0][pos0:pos1]
#print nbootstraps

pos0 = filenames[0].find('_r')+2
pos1 = filenames[0].find('.log',pos0)
distance = filenames[0][pos0:pos1]
#print distance

tag = "default"
if len(sys.argv)>3:
    tag = sys.argv[3]

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
nbins = 80
h = []
nhist = 2
hlo = 0.0
hhi = 8.0
xaxistitle = "#sqrt{2#Delta NLL}"
yaxistitle = "Arbitrary units"
for i in range(0,nhist):
    name = "h%d" % (i)
    h.append(TH1F(name,"",nbins,hlo,hhi))
    h[i].GetYaxis().SetNdivisions(6)
    h[i].GetYaxis().SetLabelSize(0.03)
    h[i].GetYaxis().CenterTitle()
    h[i].GetYaxis().SetTitleSize(0.05)
    h[i].GetYaxis().SetTitleOffset(1.1)
    h[i].GetYaxis().SetTitle(yaxistitle)
  
    h[i].GetXaxis().SetLabelSize(0.05)
    h[i].GetXaxis().SetNdivisions(8)
    h[i].GetXaxis().CenterTitle()
    h[i].GetXaxis().SetTitleSize(0.05)
    h[i].GetXaxis().SetTitleOffset(1.0)
    h[i].GetXaxis().SetTitle(xaxistitle)

    if i==1:
        h[i].SetLineWidth(2)


nfail_status = [[0,0], [0,0]]
npass_status = [[0,0], [0,0]]
sigs = [[],[]]

for i,file in enumerate(files):
    
    for line in file:


        ############################################################################
        # Count pass/fail status
        vals = line.split()
        status0 = vals[2]
        status1 = vals[10]
        if status0=="3":
            npass_status[i][0] += 1
        else:
            nfail_status[i][0] += 1

        if status1=="3":
            npass_status[i][1] += 1
        else:
            nfail_status[i][1] += 1

        sig0 = float(vals[3])
        sig1 = float(vals[11])
        sig = vals[1]
        goodFit = True
        #if sig=='nan' and abs(sig0-sig1)<1.00 and status1=="3":
        if sig=='nan' and status1=="3":
            sig = -1.0
            #print line
        elif sig=='nan' and status1!="3":
            #sig = 0.0
            sig = -2.0
            goodFit=False
            print line
        elif sig=='inf':
            sig = 999.9
            print "INF!"
            goodFit=False
            print line
            #sigs
        else:
            sig = float(sig)

        if status1=="3" and goodFit:
            sigs[i].append(sig)
            h[i].Fill(sig)
        else:
            #print line.strip()
            1



        


# Get the 99pct from the first file
sigs[0].sort()
sigs[1].sort()
nentries = len(sigs[0])
cutoff99 = sigs[0][int(0.99*nentries)]
cutoff50 = sigs[0][int(0.50*nentries)]
#print int(0.95*nentries)

#print sigs[1]
#print sigs[0]
print "nentries: %d" % (nentries)

# Figure out how much is left in the other file
pct99 = 0.0
nentries = len(sigs[1])
for i in range(0,nentries):
    if sigs[1][i]>cutoff99:
        pct99 = 1.0 - i/float(nentries)
        break

print "sig 99pct: %f %f\t\tnums: %d %d" % (cutoff99, pct99, len(sigs[0]), len(sigs[1]))
print "results %s %s %f %f\t\t%d %d" % (nbootstraps, distance, cutoff99, pct99, len(sigs[0]), len(sigs[1]))
#print "2.0/2.6/2.7/2.8/2.9/3.0/3.2/3.4 value: %f %f %f %f %f %f %f %f" % (pct20, pct26, pct27, pct28, pct29, pct30, pct32, pct34)
for i in range(0,2):
    for p,f in zip(npass_status[i],nfail_status[i]):
        tot = p+f
        print "%4d %4d %4d  %f" % (tot,p,f,f/float(tot))



##########################################################
# Make the histograms
##########################################################

can = []
ncans = 1
for i in range(ncans):
    name = "can%d" % (i)
    can.append(TCanvas(name,name,10+10*i,10+10*i,800,800))
    can[i].SetFillColor(0)
    can[i].Divide(1,1)

lines = []
text1 = []
for i,hist in enumerate(h):
    hist.SetMinimum(0)
    #hist.SetFillColor(4+38*i)
    if i==0:
        hist.SetFillColor(4)
    else:
        hist.SetFillColor(45)
    hist.Scale(1.0/hist.Integral())

    if i==0:
        hist.Draw()
        #hist.Draw("samee")
        lines.append(TLine(cutoff99,0.0, cutoff99,1.05*hist.GetMaximum()))
        lines[i].SetLineStyle(2)
        lines[i].SetLineWidth(4)
    
    elif i==1:
        hist.SetFillStyle(3003)
        hist.Draw("same")

    lines[0].Draw()
    text1.append(TPaveText(0.5, 0.85, 0.99, 0.99, "NDC"))
    name = "# bootstraps: %s" % (nbootstraps)
    text1[i].AddText(name)
    name = "radius: %.3f" % (float(distance))
    text1[i].AddText(name)
    text1[i].SetBorderSize(1)
    text1[i].SetFillStyle(1001)
    text1[i].Draw()
    gPad.Update()

for i,c in enumerate(can):
    name = "Plots/can_%s_%d.eps" % (tag,i)
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

