import sys
import ROOT
from ROOT import RooFit,RooStats

#ROOT.gStyle.SetOptStat(111111)
ROOT.gStyle.SetOptStat(0)

################################################################################
x = ROOT.RooRealVar ("x","Secondary vertex mass",0,5.0)
xframe = x.frame(150)

bkd_pdfs = []
sig_pdf = []


################################################################################
luminosity = 5695.503 # pb-1
samples = ['ttbar','t','tbar','wjets','qcd','mu']
samples_label = ['t#bar{t}','Single t/#bar{t}','#bar{t}','W-jets','QCD','Data']
fcolor = [ROOT.kRed, ROOT.kBlue, ROOT.kBlue, ROOT.kGreen,ROOT.kYellow,ROOT.kWhite]
ngen = [6923750.0, 3758227.0, 1935072.0, 57709905.0]
xsec = [227.0,56.4*3,30.7*3,36257.2]

files = []
hists = []
rdhist = []
rhpdf  = []

c = ROOT.TCanvas()
c.Divide(3,2)

c1 = ROOT.TCanvas()
c1.Divide(1,1)

c2 = ROOT.TCanvas()
c2.Divide(1,1)

hs = ROOT.THStack("hs","Stacked")

ndata = 0

efficiency = []

rooadd_string = ""
pdfs = ROOT.RooArgList()
npdfs = ROOT.RooArgList()
npdfs_list = []

for i,sample in enumerate(samples):
    #name = "%s_skim_plots_SMALL.root" % (sample)
    name = "%s_skim_plots.root" % (sample)
    files.append(ROOT.TFile(name))
    #hists.append(files[i].Get("secvtxMassHist"))
    #hists.append(files[i].Get("jetPtHistMaxPt"))
    hists.append(files[i].Get("secvtxMassHist_btag_maxpt"))

    c.cd(i+1)
    titlename = "%s sample" % (sample)
    hists[i].GetXaxis().SetTitle("Secondary vertex mass (passing selection criteria) GeV/c^{2}")
    hists[i].GetYaxis().SetTitle("# Entries")
    hists[i].SetFillColor(fcolor[i])
    hists[i].SetTitle(titlename)
    if i==5:
        hists[i].Draw("e")
        ndata = hists[i].GetEntries()
    else:
        hists[i].Draw()

    if i<4:
        efficiency.append(hists[i].GetEntries()/ngen[i])

    ROOT.gPad.Update()

    rdhistname = "%s_roodatahist" % (sample)
    rdhist.append(ROOT.RooDataHist(rdhistname,rdhistname,ROOT.RooArgList(x),hists[i]))

    rdpdfname = "%s_roohistpdf" % (sample)
    rhpdf.append(ROOT.RooHistPdf(rdpdfname,rdpdfname,ROOT.RooArgSet(x),rdhist[i]))


    npdfname = "num%s" % (sample)
    print npdfname
    npdftitle = "# %s events" % (sample)
    npdfs_list.append(ROOT.RooRealVar(npdfname,npdftitle,1000))
    #npdfs.Print()

    if i<5:
        pdfs.add(rhpdf[i])
        npdfs.add(npdfs_list[i])
        rooadd_string += "%s" % (rdpdfname)
        if i<4:
            rooadd_string  += " + "

################################################################################
# Calculate the number of events coming from each contributing process
################################################################################
ncontributions = []
for i,x in enumerate(xsec):
    ncontributions.append(luminosity*x*efficiency[i])
    print luminosity,x,efficiency[i],ncontributions[i]

#ncontributions.append(10974.9015997) # Derived from data, QCD stuff
ncontributions.append(0.9015997) # Derived from data, QCD stuff

print ncontributions
ncontributions_tot = sum(ncontributions)

for i in range(0,5):
    np = npdfs[i]
    nc = ncontributions[i]
    np.setVal(nc)

npdfs_list[0].setConstant(False) # ttbar

################################################################################

legend = ROOT.TLegend(0.7,0.5,0.99,0.99)
for i in range(0,len(samples)):
    if i<5:
        nentries = hists[i].GetEntries()
        hists[i].Scale(1.0/nentries)
        hists[i].Scale(ncontributions[i])
        hists[i].SetLineColor(fcolor[i])
        hs.Add(hists[i])
        if i!=2:
            legend.AddEntry(hists[i],samples_label[i],"f")
    else:
        c1.cd(1)
        hists[i].Draw("e")
        hs.Draw("same")
        hists[i].SetLineWidth(2)
        hists[i].SetLineColor(ROOT.kBlack)
        hists[i].Draw("samee")
        legend.AddEntry(hists[i],samples_label[i],"l")

    ROOT.gPad.Update()

################################################################################
# Run the fit
################################################################################
#tot_pdf = RooAddPdf("tot_pdf","sig_temp + bkg_temp", RooArgList(sig_temp, bkg_temp), RooArgList(nsig, nbkg))
'''
print "PDFs"
print pdfs
pdfs.Print()
print "nPDFs"
print npdfs
npdfs.Print()
print rooadd_string
tot_pdf = ROOT.RooAddPdf("tot_pdf",rooadd_string,pdfs,npdfs)
'''

rllist = ROOT.RooLinkedList()
rllist.Add(RooFit.Extended(True))
rllist.Add(RooFit.Save(True))
#tot_pdf.chi2FitTo(rdhist[5],rllist)
#rhpdf[1].fitTo(rdhist[5],rllist)
#result = tot_pdf.fitTo(rdhist[5],RooFit.Save(True))
#result = tot_pdf.fitTo(rdhist[5],rllist)
#result.Print("v")



c1.cd(1)
legend.Draw()
ROOT.gPad.Update()

c2.cd(1)
#rdhist[5].Draw()
rdhist[5].plotOn(xframe)
xframe.Draw()
hs.Draw("same")
rdhist[5].plotOn(xframe)
#tot_pdf.plotOn(xframe)
#tot_pdf.plotOn(xframe,RooFit.Components(argset_0))
ROOT.gPad.Update()


################################################################################
if __name__=="__main__":
    rep = ''
    while not rep in [ 'q', 'Q' ]:
        rep = raw_input( 'enter "q" to quit: ' )
        if 1 < len(rep):
            rep = rep[0]




