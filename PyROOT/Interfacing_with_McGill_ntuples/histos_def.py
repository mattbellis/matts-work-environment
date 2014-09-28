import ROOT
from ROOT import *

def myHistos(type="generic", imax=1, jmax=4, kmax=2):
    h = []

    xtitle = []
    nxbins = []
    xlo = []
    xhi = []
    ytitle = []
    nybins = []
    ylo = []
    yhi = []

    if ( type == "generic" ):
        nxbins.append(21); xlo.append(-0.5); xhi.append(20.5); xtitle.append("# of tracks on signal side"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(5.2); xhi.append(5.3); xtitle.append("m_{ES} tag B"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-0.2); xhi.append(0.2); xtitle.append("#Delta E tag B"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(1.5); xhi.append(3.0); xtitle.append("#Lambda_{c} mass Signal side"); ytitle.append("Entries")
        nxbins.append(3); xlo.append(-1.5); xhi.append(1.5); xtitle.append("#Lambda_{c} charge Signal side"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-5); xhi.append(5); xtitle.append("Missing mass squared #Lambda_{c} hypothesis"); ytitle.append("Entries")
        nxbins.append(100); xlo.append(-5); xhi.append(5); xtitle.append("Missing mass #Lambda_{c} hypothesis"); ytitle.append("Entries")

    elif ( type == "kin" ):
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p1cm"); ytitle.append("Entries")
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p2cm"); ytitle.append("Entries")
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p1lab"); ytitle.append("Entries")
        nxbins.append(140); xlo.append(0.0); xhi.append(7.0); xtitle.append("p2lab"); ytitle.append("Entries")


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
                if type=="generic" or type=="kin" or type=="vtx" or type=="vtx_compare":
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
