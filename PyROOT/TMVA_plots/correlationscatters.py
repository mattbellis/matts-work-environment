#!/usr/bin/env python
                                                                                                                                        # import some modules
import sys
import ROOT
from ROOT import *
from optparse import OptionParser

from color_palette import *


####################################################################
# This macro plots the correlations (as scatter plots) of
# the various input variable combinations used in TMVA (e.g. running
# TMVAnalysis.C).  Signal and Background are plotted separately
#
# input: - Input file (result from TMVA),
#        - normal/decorrelated/PCA
#        - use of TMVA plotting TStyle
#
####################################################################

##################
# Use cool palette
##################
#set_palette()

gROOT.Reset()
gStyle.SetOptStat(0)
#gStyle.SetOptStat(110010)
gStyle.SetStatH(0.6)
gStyle.SetStatW(0.5)
gStyle.SetPadRightMargin(0.15)
gStyle.SetPadLeftMargin(0.20)
gStyle.SetPadBottomMargin(0.20)
gStyle.SetFrameFillColor(0)
#gStyle.SetPalette(1)
set_palette("palette",100)

batchMode = False

fin = sys.argv[1]
var = sys.argv[2]
tag = sys.argv[3]

type = 0

###############################################
# Last argument determines batch mode or not
###############################################
last_file_offset = 0
last_argument = len(sys.argv) - 1
if (sys.argv[last_argument] == "batch"):
  batchMode = True
  last_file_offset = -1
###############################################
###############################################
#include "tmvaglob.C"
#void correlationscatters( TString fin = "TMVA.root", TString var= "var3", TMVAGlob::TypeOfPlot type = TMVAGlob::kNormal, Bool_t useTMVAStyle = kTRUE )
directories = [ "InputVariables_NoTransform",
    "InputVariables_DecorrTransform",
    "InputVariables_PCATransform",
    "InputVariables_GaussDecorr" ] 

titles = [ "TMVA Input Variable",
    "Decorrelated TMVA Input Variables",
    "Principal Component Transformed TMVA Input Variables",
    "Gaussianized and Decorrelated TMVA Input Variable" ]
  
extensions = [ "_NoTransform",
  "_DecorrTransform",
  "_PCATransform",
  "_GaussDecorr" ]

print "Called macro \"correlationscatters\" with type: " + str(type)

# set style and remove existing canvas'
#::Initialize( useTMVAStyle );

# checks if file with name "fin" is already open, and if not opens one
file = TFile( fin )

dirName = directories[type] + "/CorrelationPlots"

print dirName
  
if (dir==0):
  print "No information about " + titles[type] + " available in " + fin 
  sys.exit(-1)

dir = file.cd(dirName)

file.cd(dirName)
keyIt = gDirectory.GetListOfKeys()
print keyIt

noPlots = 0
key = 0

# How many plots are in the directory?
noPlots = (keyIt.GetEntries())/4
noVars  = (1 + sqrt(1.0 + 8.0*noPlots))/2.0
print "noPlots: " + str(noPlots) + " --> noVars: " + str(noVars)
if noVars != noVars:
  print "*** Warning: problem in inferred number of variables ... not an integer *** " 
noPlots = noVars

# Define Canvas layout here!
# Default setting
xPad = 1  # no of plots in x
yPad = 1  # no of plots in y
width = 400 # size of canvas
height = 400

if noPlots == 1:
  xPad = 1; yPad = 1; width = 400; height = width; 
elif noPlots == 2:
  xPad = 2; yPad = 1; width = 700; height = 0.55*width; 
elif noPlots == 2:
  xPad = 3; yPad = 1; width = 800; height = 0.5*width; 
elif noPlots == 2:
  xPad = 2; yPad = 2; width = 600; height = width; 
else:
  xPad = 3; yPad = 2; width = 800; height = 0.55*width; 

noPadPerCanv = xPad * yPad ;   

# Counter variables
countCanvas = 0

# Loop over all objects in "input_variables" directory
histos = [ [], [] ]
profs = [ [], [] ]
thename = [ "_sig", "_bgd" ]
for itype in range(0, 2):

  #########################
  # First the hists
  # then the profiles
  #########################
  testtype = ['TH2F', 'TProfile']
  for j in range(0,2):
    nplots = len(histos[0])
    iter = keyIt.MakeIterator()
    key = iter.Next()
    while key:
      if key.GetClassName() == testtype[j]:
        hname = key.GetName()
        scat = key.ReadObj()
        print "Found object", key.GetName()
        if j==0:
          if hname.find(thename[itype]) >= 0 and hname.find(extensions[type]) >= 0 and hname.find("_"+var+"_") >= 0 and hname.find("scat_") >=0:
            histos[itype].append(scat)
        else:
          for n in range(0,nplots):
            name = histos[itype][n].GetName()
            pname = name.replace("scat", "prof")
            print pname
            if hname == pname:
              print "Appending " + pname
              profs[itype].append(scat)

      key = iter.Next()

nplots = len(histos[0])
nrow = 6
ncol = nplots/6 + 1
print "nrow/ncol: " + str(nrow) + "\t" + str(ncol)
canvas = []
ncan = 2
for i in range(0,ncan):
  name = "can" + str(i)
  canvas.append(TCanvas(name, name, 10+10*i, 10+10*i, 1200, 900))
  canvas[i].SetFillColor(0)
  canvas[i].Divide(nrow, ncol)

title = ["Signal", "Background"]
for i in range(0,2):
  for j in range(0,nplots):
    canvas[i].cd(j+1)

    #histos[i][j].GetYaxis().SetNdivisions(4)
    histos[i][j].GetXaxis().SetNdivisions(6)

    histos[i][j].SetTitle(title[i])

    histos[i][j].GetYaxis().SetLabelSize(0.04)
    histos[i][j].GetYaxis().CenterTitle()
    histos[i][j].GetYaxis().SetTitleSize(0.07)
    histos[i][j].GetYaxis().SetTitleOffset(1.0)

    histos[i][j].GetXaxis().SetLabelSize(0.06)
    histos[i][j].GetXaxis().CenterTitle()
    histos[i][j].GetXaxis().SetTitleSize(0.07)
    histos[i][j].GetXaxis().SetTitleOffset(1.0)
    histos[i][j].Draw("colz")

    profs[i][j].SetMarkerColor( 2 )
    profs[i][j].SetMarkerSize( 0.2 )
    profs[i][j].SetLineColor( 2 )
    profs[i][j].SetLineWidth( 1 )
    profs[i][j].SetFillStyle( 3002 )
    profs[i][j].SetFillColor( 46 )
    profs[i][j].Draw("samee1")

    histos[i][j].Draw("sameaxis")
    h[ic].LabelsOption( "d" )
    canvas[i].Update()

cantalk = []
for i in range(0,nplots):
  name = "cantalk" + str(i)
  cantalk.append(TCanvas(name, name, 100+10*i, 100+10*i, 900, 400))
  cantalk[i].SetFillColor(0)
  cantalk[i].Divide(2, 1)

for j in range(0,nplots):
  cantalk[j].cd(1)
  histos[0][j].Draw("colz")
  profs[0][j].Draw("samee1")
  histos[0][j].Draw("sameaxis")
  gPad.Update()

  cantalk[j].cd(2)
  histos[1][j].Draw("colz")
  profs[1][j].Draw("samee1")
  histos[1][j].Draw("sameaxis")
  gPad.Update()

  cantalk[j].Update()
  name = "Plots/corrscatter_" + tag + "_" + var + "_" + str(j) + ".eps"
  cantalk[j].SaveAs(name)


  """
         // check for all signal histograms
         if (! (hname.EndsWith( thename[itype] + extensions[type] ) && 
                hname.Contains( "_"+var+"_" ) && hname.BeginsWith("scat_")) ) continue; 
                  
         // found a new signal plot
            
         // create new canvas
         if (countPad%noPadPerCanv==0) {
            ++countCanvas;
            canv = new TCanvas( Form("canvas%d", countCanvas), 
                                Form("Correlation Profiles for %s", (itype==0) ? "Signal" : "Background"),
                                countCanvas*50+200, countCanvas*20, width, height ); 
            canv->Divide(xPad,yPad);
         }

         if (!canv) continue;

         canv->cd(countPad++%noPadPerCanv+1);

         // find the corredponding backgrouns histo
         TString bgname = hname;
         bgname.ReplaceAll("scat_","prof_");
         TH1 *prof = (TH1*)gDirectory->Get(bgname);
         if (prof == NULL) {
            cout << "ERROR!!! couldn't find backgroung histo for" << hname << endl;
            exit;
         }
         // this is set but not stored during plot creation in MVA_Factory
         TMVAGlob::SetSignalAndBackgroundStyle( scat, prof );

         // chop off "signal" 
         TMVAGlob::SetFrameStyle( scat, 1.2 );

         // normalise both signal and background
         scat->Scale( 1.0/scat->GetSumOfWeights() );

         // finally plot and overlay       
         Float_t sc = 1.1;
         if (countPad==2) sc = 1.3;
         scat->SetMarkerColor(  4);
         scat->Draw("colz");      
         prof->SetMarkerColor( TMVAGlob::UsePaperStyle ? 1 : 2  );
         prof->SetMarkerSize( 0.2 );
         prof->SetLineColor( TMVAGlob::UsePaperStyle ? 1 : 2 );
         prof->SetLineWidth( TMVAGlob::UsePaperStyle ? 2 : 1 );
         prof->SetFillStyle( 3002 );
         prof->SetFillColor( 46 );
         prof->Draw("samee1");
         // redraw axes
         scat->Draw("sameaxis");

         // save canvas to file
         if (countPad%noPadPerCanv==0) {
            canv->Update();

            TString fname = Form( "plots/correlationscatter_%s_%s_c%i",var.Data(), extensions[type].Data(), countCanvas );
            TMVAGlob::plot_logo();
            TMVAGlob::imgconv( canv, fname );
         }
      }
      if (countPad%noPadPerCanv!=0) {
         canv->Update();

         TString fname = Form( "plots/correlationscatter_%s_%s_c%i",var.Data(), extensions[type].Data(), countCanvas );
         TMVAGlob::plot_logo();
         TMVAGlob::imgconv( canv, fname );
      }
   }
}
"""

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
                                                                                                                                                

