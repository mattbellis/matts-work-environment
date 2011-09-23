{

  gStyle->SetFrameFillColor(0);

  TCanvas *c1 = new TCanvas("c1", "c1", 10, 10, 500, 500);
  c1->SetFillColor(0);
  c1->Divide(1,1);

  TH2F *hdum = new TH2F("hdum", "hdum", 100, 1.2, 2.8, 100, -1.0, 1.0);
 

  TBox *boxes[5];

  c1->cd(1);
  hdum->Draw();

  float lox[5] = {1.3, 1.4, 1.56, 1.60, 1.43};
  float hix[5] = {2.3, 2.4, 2.56, 2.60, 2.43};
  float loy[5] = {-1.0, -1.0, -0.8, 0.75, 0.5};
  float hiy[5] = {-0.8, 0.8, 0.7, 1.0, 1.0};
  TString exps[5] = {"LEP H", "LEP S", "GRAAL", "SAPHIR", "CLAS"}

  int fillstyle[5] = {1001, 3004, 3005, 2001, 3011};
  int fillcolor[5] = {2,    5,    6,    3,    4   };

  for (int i=0; i<5;i++)
  {
    boxes[i] = new TBox(lox[i], loy[i], hix[i], hiy[i]);
    boxes[i]->SetFillStyle(fillstyle[i]);
    boxes[i]->SetFillColor(fillcolor[i]);
    boxes[i]->Draw();
  }




}
