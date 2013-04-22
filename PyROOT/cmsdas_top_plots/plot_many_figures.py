import ROOT
import sys

infilename = sys.argv[1]

f = ROOT.TFile(infilename)

can = ROOT.TCanvas("can","can",10,10,1200,800)
can.Divide(3,3)

for i in range(1,10):
    can.cd(i)
    name = "secvtxMass_btag_maxpt_njets%d" % (i)
    f.Get(name).Draw("")
    f.Get(name).Draw("samee")


################################################################################
if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

