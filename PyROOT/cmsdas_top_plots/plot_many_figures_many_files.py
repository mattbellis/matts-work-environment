import ROOT
from ROOT import *
import sys

likely_filenames = [\
        "ttbar_skim_plots.root", \
        "t_skim_plots.root", \
        "tbar_skim_plots.root", \
        "wjets_skim_plots.root", \
        "qcd_skim_plots.root", \
        "mu_skim_plots.root", \
        ]
#tags = ["Data","tbar","ttbar","qcd","t","wjets"]

samples = ['ttbar','t','tbar','wjets','qcd','mu']
samples_label = ['t#bar{t}','Single-Top','#bar{t}',"W#rightarrow#mu#nu",'QCD','data']
fcolor = [ROOT.kRed+1, ROOT.kMagenta, ROOT.kMagenta, ROOT.kGreen-3,ROOT.kYellow,ROOT.kWhite]


################################################################################
# Main
################################################################################
def main():
    can = []
    can_njets = []

    nfiles = len(sys.argv[1:])

    histos = []

    files = []

    for j,infilename in enumerate(sys.argv[1:]):

        histos.append([])

        print "Opening %s" % (infilename)

        files.append(ROOT.TFile(infilename))

        canname = "can%d" % (j)
        can.append(ROOT.TCanvas(canname,canname,10+10*j,10+10*j,1200,800))
        can[j].Divide(3,3)

        index = -1
        for k,lfn in enumerate(likely_filenames):
            print lfn
            if infilename.find(lfn)>=0:
                index = k
                break
        if index<0:
            print "Unexpected infilename!"
            print infilename
            #exit(-1)

        for i in range(1,10):
            can[j].cd(i)
            name = "secvtxMass_btag_maxpt_njets%d" % (i)
            histos[j].append(files[j].Get(name).Clone())
            newname = "%s_%d" % (name,j)
            histos[j][i-1].SetName(newname)

            histos[j][i-1].GetXaxis().SetTitle("Secondary vertex mass (passing selection criteria) GeV/c^{2}")
            histos[j][i-1].GetYaxis().SetTitle("# of entries")

            #histos[j][i-1].Rebin(2)

            title = "%s dataset, njets=%d" % (samples_label[index],i)
            histos[j][i-1].SetTitle(title)
            histos[j][i-1].SetLineColor(kBlack)
            #histos[j][i-1].SetLineColor(fcolor[index])
            histos[j][i-1].SetFillColor(fcolor[index])
            #histos[j][i-1].SetMarkerColor(fcolor[index])
            histos[j][i-1].Draw("")
            histos[j][i-1].Draw("samee")
            ROOT.gPad.Update()

        can[j].Update()
        plotname = "Plots/%s.png" % (canname)
        can[j].SaveAs(plotname)


    tempcan = ROOT.TCanvas("tempcan","tempcan",10+10*j,10+10*j,600,400)
    tempcan.Divide(1,1)

    padcount = 1
    for i in range(1,10):
        canindex = (i-1)/4
        if (i-1)%4==0:
            name = "can_njets%d" % (canindex)
            can_njets.append(ROOT.TCanvas(name,name,10+10*j,10+10*j,1200,800))
            print canindex
            can_njets[canindex].Divide(6,4)
        for j,infilename in enumerate(sys.argv[1:]):
            filetag = infilename.split('/')[-1].split('_skim_plots')[0]
            outfilename = "output/output_%s_njets%d.dat" % (filetag,i)
            outfile = open(outfilename,'w')
            print padcount
            can_njets[canindex].cd(padcount)
            histos[j][i-1].Draw("")
            histos[j][i-1].Draw("samee")
            ROOT.gPad.Update()
            padcount += 1
            if (padcount-1)%24==0:
                padcount = 1

            tempcan.cd(1)
            histos[j][i-1].Draw("")
            histos[j][i-1].Draw("samee")
            plotname = "Plots/tempcan%d_njets%d.png" % (j,i)
            tempcan.SaveAs(plotname)
            ROOT.gPad.Update()

            # Dump the data
            nbins = histos[j][i-1].GetNbinsX()
            output = ""
            for n in range(1,nbins+1):
                output += "%f %f\n" % (histos[j][i-1].GetBinCenter(n),histos[j][i-1].GetBinContent(n))
            outfile.write(output)
            outfile.close()



    for i in range(0,3):
        can_njets[i].Update()
        plotname = "Plots/can_njets%d.png" % (i)
        can_njets[i].SaveAs(plotname)



    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]

################################################################################
if __name__ == '__main__':
    main()


