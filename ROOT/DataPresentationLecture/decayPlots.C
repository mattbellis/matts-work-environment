//////////
//////////
Double_t function(Double_t *x, Double_t *par)
{
  Double_t ret = par[1]*exp(-x[0]/par[0]);
  return ret;
}


void decayPlots(int seed=999, bool doFits=false, char *tag="")
{
  // Display no statisitics on the histograms
  gStyle->SetOptStat(0);

  // Set some drawing styles
  gStyle->SetLabelSize(0.05,"X");
  gStyle->SetPalette(1,0);

  char name[256];

  int numcan = 6;
  TCanvas *can[6];

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
  double t[6][10], terr[6][10], n[6][10], nerr[6][10];
  double error;
  double max[6], min[6];
  TRandom3 *rnd = new TRandom3();
  rnd->SetSeed(seed);
  for(int i=0;i<3;i++)
  {
    for(int j=0;j<10;j++)
    {
      if(i==0) 
      {
        t[i][j] = 2.0+j*2.0;
        n[i][j] = 100 * exp(-t[i][j]/tau);
        nerr[i][j] = 0.0;
        min[i]=0.0;max[i]=110.0;
      }
      else if(i==1) 
      {
        t[i][j] = 0.10+j*0.20;
        n[i][j] = 100 * exp(-t[i][j]/tau);
        n[i][j] += rnd->Gaus(0.0, sqrt(n[i][j]));
        nerr[i][j] = sqrt(n[i][j]);
        min[i]=30.0;max[i]=120.0;
      }
      else if(i==2) 
      {
        t[i][j] = 2.0+j*2.0;
        n[i][j] = 100 * exp(-t[i][j]/tau);
        n[i][j] += rnd->Gaus(0.0, sqrt(n[i][j])) + rnd->Gaus(0.0,15.0);
        nerr[i][j] = sqrt(n[i][j]);
        min[i]=0.0;max[i]=110.0;
      }
      terr[i][j] = 0.0;
    }
  }

  TGraphErrors *h[3];
  // Make some local histograms
  for(int i=0;i<3;i++)
  {
    sprintf(name,"h[i]");
    h[i] = new TGraphErrors(10, t[i], n[i], terr[i], nerr[i]);
    h[i]->SetTitle("");
    h[i]->GetYaxis()->SetTitle("Number of counts");
    h[i]->GetYaxis()->SetTitleSize(0.06);
    h[i]->GetYaxis()->SetTitleFont(42);
    h[i]->GetYaxis()->SetTitleOffset(1.2);
    h[i]->GetYaxis()->CenterTitle();
    h[i]->GetXaxis()->SetTitle("t (sec)");
    h[i]->GetXaxis()->SetLabelSize(0.05);
    h[i]->GetXaxis()->SetTitleSize(0.08);
    h[i]->GetXaxis()->SetTitleFont(42);
    h[i]->GetXaxis()->SetTitleOffset(1.0);
    h[i]->GetXaxis()->CenterTitle();
    h[i]->GetXaxis()->SetNdivisions(8);

    h[i]->SetLineColor(1);
    h[i]->SetLineWidth(3);
    h[i]->SetFillColor(43);
    h[i]->SetMarkerColor(2);
    h[i]->SetMarkerStyle(22);
    h[i]->SetMarkerSize(2.0);

    h[i]->SetMinimum(min[i]);
    h[i]->SetMaximum(max[i]);
  }

  TF1 *func = new TF1("func",function,0,20,2);
  func->SetParameter(0,5.0);
  func->SetParameter(1,50.0);

  for(int i=0;i<3;i++)
  {
    can[i]->cd(1);
    gPad->SetLeftMargin(0.18);
    gPad->SetBottomMargin(0.18);
    h[i]->Draw("ap");
    if(doFits) h[i]->Fit("func","EVR");

    can[i+3]->cd(1);
    gPad->SetLeftMargin(0.18);
    gPad->SetBottomMargin(0.18);
    gPad->SetLogy();
    h[i]->Draw("ap");
    if(doFits) h[i]->Fit("func","EVR");
  }

  if(strcmp(tag,"")!=0)
  {
    for(int i=0;i<6;i++)
    {
      sprintf(name,"plots/decayplots%s_%d.eps",tag,i);
      can[i]->SaveAs(name);
    }
  }
}
