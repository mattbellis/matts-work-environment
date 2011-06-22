import ROOT
from ROOT import *

def myHistos(type="mc", imax=1, jmax=4, kmax=2):
    h = []

    xtitle = []
    nxbins = []
    xlo = []
    xhi = []
    ytitle = []
    nybins = []
    ylo = []
    yhi = []

    if ( type == "mc" ):
        nxbins.append(100); xlo.append(-0.2); xhi.append(0.2); xtitle.append("MC vertex 0 mm"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-0.2); xhi.append(0.2); xtitle.append("MC vertex 1 mm"); ytitle.append("Entries")

    elif ( type == "kin" ):
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p1cm"); ytitle.append("Entries")
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p2cm"); ytitle.append("Entries")
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p1lab"); ytitle.append("Entries")
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p2lab"); ytitle.append("Entries")

    elif ( type == "vtx_compare" ):
        nxbins.append(100); xlo.append(-4000.0); xhi.append(4000.0); xtitle.append("MC #Delta V_{z} (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-4000.0); xhi.append(4000.0); xtitle.append("lep #Delta V_{z} (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-4000.0); xhi.append(4000.0); xtitle.append("lep #pi #Delta V_{z} (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-4000.0); xhi.append(4000.0); xtitle.append("lep #pi (TF) #Delta V_{z} (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(4000.0); xtitle.append("MC |#Delta V| (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(4000.0); xtitle.append("lep |#Delta V| (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(4000.0); xtitle.append("lep #pi |#Delta V| (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(4000.0); xtitle.append("lep #pi (TF) |#Delta V| (#mu m)"); ytitle.append("Entries")
        nxbins.append(130); xlo.append(-10.0); xhi.append(3.0); xtitle.append("M^{2}(#nu_{1}) (GeV^{2}/c^{4})"); ytitle.append("Entries")
        nxbins.append(130); xlo.append(-10.0); xhi.append(3.0); xtitle.append("M^{2}(#nu_{2}) (GeV^{2}/c^{4})"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(2.0); xtitle.append("|p|(#pi_{1}) (GeV/c)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(2.0); xtitle.append("|p|(#pi_{2}) (GeV/c)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-2000.0); xhi.append(2000.0); xtitle.append("lep #Delta V_{z} residual (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-2000.0); xhi.append(2000.0); xtitle.append("lep #pi #Delta V_{z} residual (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-2000.0); xhi.append(2000.0); xtitle.append("lep #pi (TF) #Delta V_{z} residual (#mu m)"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-2000.0); xhi.append(2000.0); xtitle.append("lep #pi (TF-old) #Delta V_{z} residual (#mu m)"); ytitle.append("Entries")

    elif ( type == "vtx" ):
        nxbins.append(100); xlo.append(0.0); xhi.append(4000.0); xtitle.append("MC vtx1 X"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(4000.0); xtitle.append("MC vtx1 Y"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-4.0); xhi.append(3.0); xtitle.append("MC vtx1 Z"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(3.0); xtitle.append("MC vtx1 |r|"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.3); xhi.append(0.4); xtitle.append("MC vtx1 r_{xy}"); ytitle.append("Entries")

        nxbins.append(100); xlo.append(0.0); xhi.append(0.4); xtitle.append("MC vtx2 X"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(0.4); xtitle.append("MC vtx2 Y"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-4.0); xhi.append(3.0); xtitle.append("MC vtx2 Z"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(3.0); xtitle.append("MC vtx2 |r|"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.3); xhi.append(0.4); xtitle.append("MC vtx2 r_{xy}"); ytitle.append("Entries")

        nxbins.append(100); xlo.append(-100.03); xhi.append(100.03); xtitle.append("MC vtx #Delta X"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-0.03); xhi.append(0.03); xtitle.append("MC vtx #Delta Y"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-0.3); xhi.append(0.3); xtitle.append("MC vtx #Delta Z"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(0.2); xtitle.append("MC vtx #Delta |r|"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(0.03); xtitle.append("MC vtx #Delta r_{xy}"); ytitle.append("Entries")

        nxbins.append(100); xlo.append(0.0); xhi.append(0.4); xtitle.append("vtx1 X"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(0.4); xtitle.append("vtx1 Y"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-4.0); xhi.append(3.0); xtitle.append("vtx1 Z"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(3.0); xtitle.append("vtx1 |r|"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.3); xhi.append(0.4); xtitle.append("vtx1 r_{xy}"); ytitle.append("Entries")

        nxbins.append(100); xlo.append(0.0); xhi.append(0.4); xtitle.append("vtx2 X"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(0.4); xtitle.append("vtx2 Y"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-4.0); xhi.append(3.0); xtitle.append("vtx2 Z"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(3.0); xtitle.append("vtx2 |r|"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.3); xhi.append(0.4); xtitle.append("vtx2 r_{xy}"); ytitle.append("Entries")

        nxbins.append(100); xlo.append(-0.03); xhi.append(0.03); xtitle.append("vtx #Delta X"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-0.03); xhi.append(0.03); xtitle.append("vtx #Delta Y"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-0.3); xhi.append(0.3); xtitle.append("vtx #Delta Z"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(0.2); xtitle.append("vtx #Delta |r|"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(0.0); xhi.append(0.03); xtitle.append("vtx #Delta r_{xy}"); ytitle.append("Entries")

    elif (type=='vtx2D'):
        nxbins.append(100);xlo.append(0.0);xhi.append(2000.0);xtitle.append("MC vtx1 X");nybins.append(100);ylo.append(2000.0);yhi.append(4000.0);ytitle.append("MC vtx1 Y")
        nxbins.append(100);xlo.append(0.0);xhi.append(2000.0);xtitle.append("MC vtx2 X");nybins.append(100);ylo.append(2000.0);yhi.append(4000.0);ytitle.append("MC vtx2 Y")
        nxbins.append(100);xlo.append(0.0);xhi.append(2000.0);xtitle.append("vtx1 X");nybins.append(100);ylo.append(2000.0);yhi.append(4000.0);ytitle.append("vtx1 Y")
        nxbins.append(100);xlo.append(0.0);xhi.append(2000.0);xtitle.append("vtx2 X");nybins.append(100);ylo.append(2000.0);yhi.append(4000.0);ytitle.append("vtx2 Y")
        nxbins.append(100);xlo.append(-100.0);xhi.append(100.0);xtitle.append("MC #Delta vtx X");nybins.append(100);ylo.append(-100.0);yhi.append(100.0);ytitle.append("MC #Delta vtx Y")
        nxbins.append(100);xlo.append(-100.0);xhi.append(100.0);xtitle.append("#Delta vtx X");nybins.append(100);ylo.append(-100.0);yhi.append(100.0);ytitle.append("#Delta vtx Y")
    ##########################################################
    # If we've asked for too little of bins, fill these with some defaults
    ##########################################################
    if(len(nxbins) < jmax):
        for j in range(len(nxbins), jmax):
            nxbins.append(100); xlo.append(0.00); xhi.append(1.00); xtitle.append("X-axis")

    ##########################################################
    for i in range(0, imax):
        h.append([])
        for j in range(0, jmax):
            h[i].append([])
            for k in range(0, kmax):

                name = "h" + type + str(i) + "_" + str(j) + "_" + str(k)
                hdum = None
                if type=="mc" or type=="kin" or type=="vtx" or type=="vtx_compare":
                    hdum = TH1D(name, name, nxbins[j], xlo[j], xhi[j])
                else:
                    hdum = TH2D(name, name, nxbins[j], xlo[j], xhi[j], nybins[j], ylo[j], yhi[j])

                h[i][j].append(hdum)

                h[i][j][k].SetTitle("")

                h[i][j][k].GetXaxis().SetTitleSize(0.09)
                h[i][j][k].GetXaxis().SetTitleFont(42)
                h[i][j][k].GetXaxis().SetTitleOffset(0.6)
                h[i][j][k].GetXaxis().CenterTitle()
                h[i][j][k].GetXaxis().SetNdivisions(6)
                h[i][j][k].GetXaxis().SetTitle(xtitle[j])

                h[i][j][k].GetYaxis().SetTitleSize(0.09)
                h[i][j][k].GetYaxis().SetTitleFont(42)
                h[i][j][k].GetYaxis().SetTitleOffset(0.8)
                h[i][j][k].GetYaxis().CenterTitle()
                h[i][j][k].GetYaxis().SetNdivisions(6)
                h[i][j][k].GetYaxis().SetTitle(ytitle[j])

                h[i][j][k].SetFillStyle(1001)

                h[i][j][k].SetMinimum(0)

                if(k==0):
                    h[i][j][k].SetFillColor(9)
                else:
                    h[i][j][k].SetFillColor(k+1)

    return h
