// Gaussian function
Double_t gaussian(Double_t *x, Double_t *par) 
{
  return par[0]*exp(-0.5*pow(((x[0]-par[1])/par[2]),2));
}

// Sum of background and peak function
Double_t fitFunction(Double_t *x, Double_t *par)
{
  return gaussian(x,par);
}

plotStuff_moreRealistic(int max=10000, float egamma=1.5, float varyGammaRes=0.01, float varyPRes=0.10)
{

  TCanvas *can[3];
  char name[256];
  char title[256];

  for(int i=0;i<3;i++)
  {
    sprintf(name,"canvas%d",i);
    can[i] = new TCanvas(name,"title",10+10*i,10+10*i,1200,800);
    can[i]->SetFillColor(0);
    if(i==0) can[i]->Divide(3,2);
    else if(i>0&&i<2) can[i]->Divide(4,4);
    else     can[i]->Divide(1,1);
  }

  TH1F *h[6];
  TH1F *hsmear[2][2][4];
  TH1F *hvals[2][2][4];
  TPaveText *text[6];
  // Declare some local functions
  TF1 *fsig[6];
  TF1 *fback[6];
  TF1 *ftotal[6];

  // Create some local arrays to hold this stuff
  float counts[6];
  float mean[6];
  float width[6];

  // Some range over which to fit
  float lo = 0.75;
  float hi = 0.82;
  float meanguess = 0.782;
  float widthguess = 0.005;

  for(int i=0;i<6;i++)
  {
    sprintf(name,"h%d",i);
    if(i%3==0)
    {
      sprintf(title,"MM off proton");
      h[i] = new TH1F(name,title,260,0.00, 1.30);
      h[i]->GetXaxis()->SetTitle("MM(GeV/c^{2})");
    }
    else if(i%3==1) 
    {
      sprintf(title,"MM^{2} off proton");
      h[i] = new TH1F(name,title,320,-0.2, 1.4);
      h[i]->GetXaxis()->SetTitle("MM^{2}(GeV^{2}/c^{4})");
    }
    else 
    {
      sprintf(title,"Total MM^{2}");
      h[i] = new TH1F(name,title,100,-0.1, 0.1);
      h[i]->GetXaxis()->SetTitle("MM^{2}(GeV^{2}/c^{4})");
    }
    h[i]->GetXaxis()->SetTitleFont(42);
    h[i]->SetFillColor(i%3 + 2);
    h[i]->GetXaxis()->SetNdivisions(6);
  }

  for(int k=0;k<2;k++)
  {
    for(int j=0;j<2;j++)
    {
      for(int i=0;i<4;i++)
      {
        sprintf(name,"hsmear%d_%d_%d",k,j,i);
        hsmear[k][j][i] = new TH1F(name,"Momentum resolution",100,-0.5,0.5);
        hsmear[k][j][i]->SetFillColor(12);
        hsmear[k][j][i]->GetXaxis()->SetTitleFont(42);
        hsmear[k][j][i]->GetXaxis()->SetTitle("p_{original} - p_{smeared}");
      }
    }
  }
  float val;
  TRandom *rnd = new TRandom();
  rnd->SetSeed();

  TGenPhaseSpace *gen = new TGenPhaseSpace();

  float smearedegamma;
  float egammares, zres, yres, xres;
  TLorentzVector *gamma = new TLorentzVector();
  TLorentzVector *target = new TLorentzVector();
  TLorentzVector *smearedgamma = new TLorentzVector();
  TLorentzVector *w = new TLorentzVector();

  TLorentzVector *tot = new TLorentzVector();
  TLorentzVector *smearedtot = new TLorentzVector();

  TLorentzVector *smearedmm[2];
  TLorentzVector *mm[2];
  TLorentzVector *smearedmmtot[2];
  TLorentzVector *final[2];
  TLorentzVector *smearedfinal[2];

  double masses[2] = {0.938272, 0.782};
  gamma->SetXYZT(0.0, 0.0, egamma, egamma);
  target->SetXYZM(0.0, 0.0, 0.0, 0.938272);
  *w = *gamma + *target;

  for(int i=0;i<2;i++)
  {
    smearedfinal[i] = new TLorentzVector();
    final[i] = new TLorentzVector();
    smearedmm[i] = new TLorentzVector();
    smearedmmtot[i] = new TLorentzVector();
    mm[i] = new TLorentzVector();
  }

  gen->SetDecay(*w, 2, masses);

  for(int i=0;i<max;i++)
  {
    if(i%1000==0) cerr << i << "\r";
    double weight = gen->Generate();

    final[0] = gen->GetDecay(0);
    final[1] = gen->GetDecay(1);

    for(int k=0;k<2;k++)
    {
      if(k==0)
      {
        //xres =0.02; // constant
        //yres =0.02;
        //zres =0.02;
        //egammares = 0.005;

        xres =0.02; // pct.
        yres =0.02;
        zres =0.02;
        egammares = 0.005;
      }
      else 
      {
        //xres =0.000; // symmetric mm0
        //yres =0.000;
        //zres =0.050;
        //egammares = 0.250;

        //xres =0.050; // non symmetric
        //yres =0.050;
        //zres =0.050;
        //egammares = 0.010;

        xres =varyPRes;
        yres =varyPRes;
        zres =varyPRes;
        egammares = varyGammaRes;
      }
      for(int j=0;j<2;j++)
      {

        float x = final[j]->X() + rnd->Gaus(0,xres*final[j]->X());
        float y = final[j]->Y() + rnd->Gaus(0,yres*final[j]->Y());
        float z = final[j]->Z() + rnd->Gaus(0,zres*final[j]->Z());

        //float x = final[j]->X() + rnd->Gaus(0,xres);
        //float y = final[j]->Y() + rnd->Gaus(0,yres);
        //float z = final[j]->Z() + rnd->Gaus(0,zres);

        float t = sqrt(x*x + y*y + z*z + masses[j]*masses[j]);

        smearedfinal[j]->SetXYZT(x, y, z, t);
        hsmear[k][j][0]->Fill(x-final[j]->X());
        hsmear[k][j][1]->Fill(y-final[j]->Y());
        hsmear[k][j][2]->Fill(z-final[j]->Z());
        hsmear[k][j][3]->Fill(t-final[j]->T());
      }
      smearedegamma = egamma + rnd->Gaus(0,egammares);
      smearedgamma->SetXYZT(0.0, 0.0, smearedegamma, smearedegamma);
      *smearedmm[0] = *smearedgamma + *target - *smearedfinal[0];
      *smearedmm[1] = *smearedgamma + *target - *smearedfinal[1];
      *smearedmmtot[0] = *smearedgamma + *target - *smearedfinal[1] - *smearedfinal[0];

      h[0+k*3]->Fill( smearedmm[0]->M() );
      h[1+k*3]->Fill( smearedmm[0]->M2() );
      h[2+k*3]->Fill( smearedmmtot[0]->M2() );
    }

  }

  for(int i=0;i<6;i++)
  {

    if(i%3==0)      {widthguess=0.005;meanguess=0.782;lo=0.00; hi=1.30;}
    else if(i%3==1) {widthguess=0.001;meanguess=0.611;lo=-0.20; hi=1.40;}

    TF1 *fitFcn = new TF1("fitFcn",fitFunction, lo, hi, 3);
    fitFcn->SetParameter(2, widthguess); // width
    fitFcn->SetParameter(1, meanguess);   // peak

    can[0]->cd(i+1);
    h[i]->Draw();
    if(i%3!=2)
    {
    h[i]->Fit("fitFcn","EQR"); 
    mean[i] = fitFcn->GetParameter(1);
    width[i] = fitFcn->GetParameter(2);
    counts[i] = fitFcn->Integral(lo, hi) / h[i]->GetBinWidth(1);

    text[i] = new TPaveText(0.20, 0.5,0.5,1.0,"NDC");
    sprintf(name,"Fit results");
    text[i]->AddText(name);
    if(i%3==0) sprintf(name,"mean: %3.3f",mean[i]);
    else       sprintf(name,"mean: %3.3f",sqrt(mean[i]));
    text[i]->AddText(name);
    sprintf(name,"pct. org events: %3.3f",counts[i]/(float)max);
    text[i]->AddText(name);
    text[i]->SetBorderSize(0);
    text[i]->SetFillStyle(0);
    text[i]->Draw();
    }
  }

  for(int k=0;k<2;k++)
  {
    for(int j=0;j<2;j++)
    {
      for(int i=0;i<4;i++)
      {
        can[1]->cd(i+4*j+8*k+1);
        hsmear[k][j][i]->Draw();
      }
    }
  }

  can[2]->cd(1);
  hsmear[1][1][2]->Draw();

  for(int i=0;i<4;i++)
  {
    cerr << "mean: " << mean[i] << " " << sqrt(mean[i]) << "\twidth: " << width[i] << "\tcounts: " << counts[i]/(float)max << endl;
  }

}
