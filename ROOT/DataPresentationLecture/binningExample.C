//////////
//////////
void binningExample(int nentries=999, bool doFits=false, char *tag="")
{
  // Display no statisitics on the histograms
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(11);
  gStyle->SetStatH(0.2);
  gStyle->SetStatW(0.3);
  gStyle->SetFrameFillStyle(0);

  // Set some drawing styles
  gStyle->SetLabelSize(0.05,"X");
  gStyle->SetPalette(1,0);

  char name[256];

  int numcan = 5;
  TCanvas *can[5];

  double tau = 7.0;

  // Make the canvases
  for(int i=0;i<numcan;i++)
  {
    sprintf(name,"can%d",i);
    can[i] = new TCanvas(name,"",10+10*i, 10+10*i, 600, 400);
    can[i]->SetFillColor(0);
    can[i]->Divide(1,1);
  }

  // Generate the data points
  int nbins=10;
  TH1F *h[5];
  // Make some local histograms
  for(int i=0;i<5;i++)
  {
    sprintf(name,"h%d",i);
    if(i==0) nbins=10000;
    else if(i==1) nbins=1000;
    else if(i==2) nbins=100;
    else if(i==3) nbins=10;
    else          nbins=10;
    h[i] = new TH1F(name,"name",nbins,-5,5);
    h[i]->SetTitle("");
    h[i]->GetYaxis()->SetTitle("Number of counts");
    h[i]->GetYaxis()->SetTitleSize(0.06);
    h[i]->GetYaxis()->SetTitleFont(42);
    h[i]->GetYaxis()->SetTitleOffset(1.2);
    h[i]->GetYaxis()->CenterTitle();
    h[i]->GetXaxis()->SetTitle("Abitrary measurement");
    h[i]->GetXaxis()->SetLabelSize(0.05);
    h[i]->GetXaxis()->SetTitleSize(0.08);
    h[i]->GetXaxis()->SetTitleFont(42);
    h[i]->GetXaxis()->SetTitleOffset(1.0);
    h[i]->GetXaxis()->CenterTitle();
    h[i]->GetXaxis()->SetNdivisions(8);

    h[i]->SetLineColor(1);
    h[i]->SetLineWidth(1);
    h[i]->SetFillColor(23);
    h[i]->SetMarkerColor(2);
    h[i]->SetMarkerStyle(22);
    h[i]->SetMarkerSize(2.0);

  }

  TRandom3 *rnd = new TRandom3();
  rnd->SetSeed(nentries);

  for(int i=0;i<nentries;i++)
  {
    double num = rnd->Gaus(0.0,1.0);
    h[0]->Fill(num);
    h[1]->Fill(num);
    h[2]->Fill(num);
    h[3]->Fill(num);
    h[4]->Fill(num);
  }

  for(int i=0;i<5;i++)
  {
    can[i]->cd(1);
    gPad->SetLeftMargin(0.18);
    gPad->SetBottomMargin(0.18);
    h[i]->SetMaximum( 1.5 * h[i]->GetMaximum() );
    h[i]->Draw("");
    if(doFits) h[i]->Fit("gaus");

  }

  if(strcmp(tag,"")!=0)
  {
    for(int i=0;i<5;i++)
    {
      sprintf(name,"plots/binningplots%s_%d.eps",tag,i);
      can[i]->SaveAs(name);
    }
  }
}
