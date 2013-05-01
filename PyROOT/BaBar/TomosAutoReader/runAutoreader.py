from autoread import ntuplereader
from ROOT import *

def defaultHistoSettings(h):
    h.SetNdivisions(6)
    h.GetYaxis().SetTitleSize(0.09)
    h.GetYaxis().SetTitleFont(42)
    h.GetYaxis().SetTitleOffset(0.7)
    h.GetYaxis().CenterTitle()
    h.GetYaxis().SetNdivisions(6)
    h.GetYaxis().SetTitle("events/MeV")
    h.SetFillColor(9)

def defaultPadSettings():
    
    gPad.SetFillColor(0)
    gPad.SetBorderSize(0)
    gPad.SetRightMargin(0.20);
    gPad.SetLeftMargin(0.20);
    gPad.SetBottomMargin(0.15);

def waitForInput():
    rep = ''
    while not rep in [ 'c', 'C' ]:
        rep = raw_input( 'enter "c" to continue: ' )
        if 1 < len(rep):
            rep = rep[0]


variables = ["mes","_de","_Etot","mp0"]
numVariables = len(variables)
minmax = [[5.19,5.3],[-0.02,0.1],[0,15],[0,5.5]]

x = ntuplereader(["sp1228_ddv3NoNNCut.root"],"ntp",variables)

nentries = x.getEntries()
print "File contains", nentries, "entries"

histos = []
for i in range(0,numVariables):
    varName = variables[i]
    title = varName
    histo = TH1D(title, varName, 100, minmax[i][0], minmax[i][1])
    histo.SetTitle(title)
    defaultHistoSettings(histo)
    histos.append(histo)

for n in range(0,10000):
    x.entry(n)
    for i in range(0,numVariables):
        value = x.get(variables[i])
        if value > minmax[i][0] and value < minmax[i][1] and value > -99 :
            histos[i].Fill(value)


gStyle.SetOptStat(0)
c1 = TCanvas("c1", "Example Plots", 200, 10, 700, 500)
c1.SetFillColor(0)
c1.Divide(2,2)

labels = []
for i in range(0,numVariables):
    c1.cd(i+1)

    defaultPadSettings()

    histos[i].Draw("")
    labels.append(TPaveText(0.81, 0.75, 0.99, 0.99, "NDC"))
    nevents = histos[i].GetEntries()
    words = "#events: %d" % nevents
    labels[i].AddText(words)
    labels[i].SetFillColor(0)
    labels[i].SetBorderSize(0)
    labels[i].Draw("same")

    c1.Update()
    gPad.Update()

waitForInput()

