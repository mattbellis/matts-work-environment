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

plotStuff_simple(int max=10000)
{

  TCanvas *can = new TCanvas("canvas","title",10,10,1200,800);
  can->SetFillColor(0);
  can->Divide(2,2);

  char name[256];
  char title[256];
  TH1F *h[4];
  // Declare some local functions
  TF1 *fsig[4];
  TF1 *fback[4];
  TF1 *ftotal[4];

  // Create some local arrays to hold this stuff
  float counts[4];
  float mean[4];
  float width[4];

  // Some range over which to fit
  float lo = 0.75;
  float hi = 0.82;
  float meanguess = 0.782;
  float widthguess = 0.005;

  for(int i=0;i<4;i++)
  {
    sprintf(name,"h%d",i);
    sprintf(title,"Data %d",i);
    if(i%2==0)      h[i] = new TH1F(name,title,100,0.75, 0.82);
    else if(i%2==1) h[i] = new TH1F(name,title,100,0.5, 0.7);
    h[i]->SetFillColor(i+2);
  }

  TRandom *rnd = new TRandom();

  float val;
  rnd->SetSeed();
  for(int i=0;i<max;i++)
  {
    val = rnd->Gaus(0.782,0.004);
    h[0]->Fill(val);

    h[1]->Fill(val*val);
  }

  TGenPhaseSpace *gen = new TGenPhaseSpace();
  TLorentzVector *final[2];
  TLorentzVector *smearedfinal[2];

  double masses[2] = {0.139, 0.139};
  TLorentzVector *w = new TLorentzVector();
  w->SetXYZM(0.0, 0.0, 0.0, 0.782);
  gen->SetDecay(*w, 2, masses);
  TLorentzVector *tot = new TLorentzVector();
  TLorentzVector *smearedtot = new TLorentzVector();

  smearedfinal[0] = new TLorentzVector();
  smearedfinal[1] = new TLorentzVector();

  for(int i=0;i<max;i++)
  {
    double weight = gen->Generate();
    final[0] = gen->GetDecay(0);
    final[1] = gen->GetDecay(1);
    *tot = *final[0] + *final[1];

    for(int j=0;j<2;j++)
    {
      float x = final[j]->X() + rnd->Gaus(0,0.01*final[j]->X());
      float y = final[j]->Y() + rnd->Gaus(0,0.01*final[j]->Y());
      float z = final[j]->Z() + rnd->Gaus(0,0.01*final[j]->Z());
      float t = final[j]->T() + rnd->Gaus(0,0.01*final[j]->T());
      smearedfinal[j]->SetXYZT(x, y, z, t);
    }
    *smearedtot = *smearedfinal[0] + *smearedfinal[1];

    h[2]->Fill( smearedtot->M() );
    h[3]->Fill( smearedtot->M2() );

  }



  for(int i=0;i<4;i++)
  {

    if(i%2==0)      {widthguess=0.005;meanguess=0.782;lo=0.75; hi=0.82;}
    else if(i%2==1) {widthguess=0.001;meanguess=0.620;lo=0.55; hi=0.70;}

  TF1 *fitFcn = new TF1("fitFcn",fitFunction, lo, hi, 3);
    fitFcn->SetParameter(2, widthguess); // width
    fitFcn->SetParameter(1, meanguess);   // peak

    can->cd(i+1);
    h[i]->Draw();
    h[i]->Fit("fitFcn","EQR"); 
    mean[i] = fitFcn->GetParameter(1);
    width[i] = fitFcn->GetParameter(2);
    counts[i] = fitFcn->Integral(lo, hi) / h[i]->GetBinWidth(1);
  }

  for(int i=0;i<4;i++)
  {
    cerr << "mean: " << mean[i] << " " << sqrt(mean[i]) << "\twidth: " << width[i] << "\tcounts: " << counts[i]/(float)max << endl;
  }

}
