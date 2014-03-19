void lookAtData(char *filename, int max=100000, int LO=0, int HI=50, bool plotXtitle=true, char *tag="")
{
  ifstream IN(filename);

  cerr << filename << endl;

  gStyle->SetOptStat(0);

  float x;

  char name[256];
  int color = 32;
  int nbins = HI-LO+1;
  float lo = LO-0.5;
  float hi = HI+0.5;

  TCanvas *can = new TCanvas("can","",10,10,1350,700);
  can->SetFillColor(0);
  can->Divide(1,1);

  TH1F *h[5];
  for(int i=0;i<5;i++)
  {
    sprintf(name,"h%d",i);
    h[i] = new TH1F(name,"",nbins, lo, hi);
    h[i]->SetFillStyle(1000);
    h[i]->SetFillColor(color);
    h[i]->SetTitle();

    h[i]->SetNdivisions(8);
    h[i]->GetYaxis()->SetTitle("# occurances");
    h[i]->GetYaxis()->SetTitleSize(0.09);
    h[i]->GetYaxis()->SetTitleFont(42);
    h[i]->GetYaxis()->SetTitleOffset(0.7);
    h[i]->GetYaxis()->CenterTitle();
    if(plotXtitle) h[i]->GetXaxis()->SetTitle("Points scored by a team");
    else           h[i]->GetXaxis()->SetTitle("Arbitrary measurements");
    h[i]->GetXaxis()->SetLabelSize(0.12);
    h[i]->GetXaxis()->SetTitleSize(0.10);
    h[i]->GetXaxis()->SetTitleFont(42);
    h[i]->GetXaxis()->SetTitleOffset(1.0);
    h[i]->GetXaxis()->CenterTitle();

    h[i]->SetMinimum(0);
  }

  int i=0;
  float junk;

  int count=0;
  while(IN >> x && count<max)
  {
    h[0]->Fill((x));
    IN >> x;
    h[0]->Fill((x));
    count++;
  }

  can->cd(1);
  gPad->SetLeftMargin(0.18);
  gPad->SetBottomMargin(0.24);
  h[0]->Draw("");

  if(strcmp(tag,"")!=0)
  {
    sprintf(name,"plots/sportsplots%s_%d.eps",tag,i);
    can->SaveAs(name);
  }
}

