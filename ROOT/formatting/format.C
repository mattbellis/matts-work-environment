//////////
// This example should show you how to format your histogram
//////////

{
  // Display no statisitics on the histograms
  gStyle->SetOptStat(0);

  // Set some drawing styles
  gStyle->SetLabelSize(0.05,"X");
  gStyle->SetPalette(1,0);

  char name[256];

  int numcan = 1;
  TCanvas *can[1];

  // Make the canvases
  for(int i=0;i<numcan;i++)
  {
    sprintf(name,"can%d",i);
    can[i] = new TCanvas(name,"",10+10*i, 10+10*i, 800, 900);
    can[i]->SetFillColor(0);
    can[i]->Divide(1,1);
  }

  // Make some local histograms
  sprintf(name,"hgaus");
  TH1F *hgaus = new TH1F(name,"Gaussian",100,-5,5);
  hgaus->SetNdivisions(6);
  hgaus->GetYaxis()->SetTitle("# of events");
  hgaus->GetYaxis()->SetTitleSize(0.06);
  hgaus->GetYaxis()->SetTitleFont(42);
  hgaus->GetYaxis()->SetTitleOffset(1.2);
  hgaus->GetYaxis()->CenterTitle();
  hgaus->GetXaxis()->SetTitle("Some Gaussian distribution (GeV/c^{2})");
  hgaus->GetXaxis()->SetLabelSize(0.05);
  hgaus->GetXaxis()->SetTitleSize(0.04);
  hgaus->GetXaxis()->SetTitleFont(42);
  hgaus->GetXaxis()->SetTitleOffset(1.2);
  hgaus->GetXaxis()->CenterTitle();
  hgaus->SetLineColor(1);
  hgaus->SetFillColor(43);
  hgaus->SetMarkerColor(2);

  TRandom *rnd = new TRandom();
  float dum;

  for(int i=0;i<100000;i++)
  {
    // Generate a Gaussian distribution with
    // mean of 0 and witdth of 1
    dum = rnd->Gaus(0,1); 
    hgaus->Fill(dum);
  }

  can[0]->cd(1);
  gPad->SetLeftMargin(0.18);
  gPad->SetBottomMargin(0.18);
  hgaus->Draw();



}
