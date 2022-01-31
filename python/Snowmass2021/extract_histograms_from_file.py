import sys
import ROOT
import uproot

infilename = sys.argv[1]

ftemp = uproot.open(infilename)

hnames = ftemp.keys()

print(hnames)

ftemp.close()

f = ROOT.TFile(infilename)

for hname in hnames[:]:

    print(f"Opening {hname}...")

    h = f.Get(hname)

    c = ROOT.TCanvas()
    c.Divide(1,1)
    c.cd(1)

    h.Draw()

    outname = f"plots/{hname}.png"

    c.SaveAs(outname)
