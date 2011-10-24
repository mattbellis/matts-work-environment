#include <Riostream.h>
void BkgEstimate_Result()
{
  gROOT->SetStyle("BABAR");
  gStyle->SetOptStat(0);

  Double_t MesMean = 5.27955;
  Double_t MesSigma = 0.0027;
  Double_t dESigma = 0.0139;
  Double_t nSigma = 3;

  Double_t MesLow = MesMean-nSigma*MesSigma;
  Double_t dELow = -nSigma*dESigma;
  Double_t dEHigh = nSigma*dESigma;

  TChain chain("ntp1");
  chain.Add("BtoLcp_Data_Blind_Fit_NewCuts.root");

  RooRealVar 
    Mes("Mes","m_{ES}",5.2,5.3,"GeV/c^{2}"),
    DeltaE("DeltaE","#DeltaE",-0.1,0.1,"GeV");

  RooArgSet Vars(Mes,DeltaE);
  RooDataSet Data("Data","Data",Vars);

  TCanvas *LcPMes_Fit = new TCanvas("LcPMes_Fit","dE vs. Mes",40,40,1000,500);

  Int_t nbins = 50;
  Int_t nevents = -1;

  TH2F *h1 = new TH2F("h1","Blinded Data",nbins*4,5.2,5.3,nbins*4,-0.1,0.1);
  chain.Project("h1","B0postFitDeltaE:B0postFitMes");
  h1->SetMarkerStyle(7);
  h1->SetXTitle("m_{ES} (GeV)");
  h1->SetYTitle("#DeltaE (GeV)");
  h1->GetYaxis()->SetTitleOffset(0.6);
  h1->DrawCopy();

  TCanvas *dE_Data = new TCanvas("dE_Data","Delta E");

  chain.Draw("B0postFitDeltaE:B0postFitMes");
  Double_t *vals = chain.GetV1();
  Double_t *vals2 = chain.GetV2();
  nevents = chain.GetSelectedRows();
  for(Int_t i=0;i<nevents;i++) {
    if(vals[i]<-0.1||vals[i]>0.1) continue;
    DeltaE.setVal(vals[i]);
    Mes.setVal(vals2[i]);
    Data.add(Vars);
  }
  cout << nevents << " data events" << endl;

  RooPlot* dEframe= DeltaE.frame(-0.1,0.1,nbins);
  Data.plotOn(dEframe,MarkerColor(kBlue));
  dEframe->Draw();

  TCanvas *Mes_Data = new TCanvas("Mes_Data","Mes");

  RooPlot* Mesframe= Mes.frame(5.2,5.3,nbins);
  Data.plotOn(Mesframe,MarkerColor(kRed));
  Mesframe->SetMinimum(1e-6);
  Mesframe->Draw();

  
  TCanvas *Mes_Fit = new TCanvas("Mes_Fit","Mes");

  RooFormulaVar blind("blind","blind","(Mes<5.271)||(DeltaE<-0.0417||DeltaE>0.0417)",RooArgList(Mes,DeltaE,DeltaE));
  RooRealVar
    Argm0("argus m0","argus resonance mass",5.2900,"GeV/c^{2}"),
    Argc("argus c","argus slope param",-20.,-40.,-5.,""),
    dEslope("#DeltaE slope","slope",-5.,-30.,10.);
  RooArgusBG Mesarg("arg_{m_{ES}}","Argus",Mes,Argm0,Argc);
  RooPolynomial dEpoly("p_{#DeltaE}","Polynomial",DeltaE,dEslope);
  //RooPolynomial dEpoly("p_{#DeltaE}","Polynomial",DeltaE);

  RooProdPdf totalbkg("totalbkg","mesargus*dEpoly",RooArgList(Mesarg,dEpoly));
  RooGenericPdf blindbkg("blindbkg","totalbkg*blinding","totalbkg*blind",RooArgList(totalbkg,blind));

//   RooFitResult *blindbkgresult = blindbkg.fitTo(Data,Save(kTRUE),NumCPU(2),Timer(kTRUE));
//   blindbkgresult->Print();
//   Int_t nb = blindbkgresult->Write();
//   if(nb==0) cout << "nbytes = 0; Write Failed" << endl;

  // 11 Jan 06
  // #DeltaE slope   -8.4740e-01 +/-  9.29e-01
  //       argus c   -1.7680e+01 +/-  6.39e+00

  // 18 Feb 06
  // #DeltaE slope   -1.1042e+00 +/-  7.93e-01
  //       argus c   -1.7393e+01 +/-  5.51e+00

//    Argc.setVal(-17.68);
//    Argc.setError(6.39);
//    dEslope.setVal(-0.847);
//    dEslope.setError(0.929);

   Argc.setVal(-17.93);
   Argc.setError(5.51);
   dEslope.setVal(-1.1042);
   dEslope.setError(0.793);

  RooPlot* Mesframe2= Mes.frame(5.2,5.3,nbins);
  Data.plotOn(Mesframe2,MarkerColor(kRed));
  blindbkg.plotOn(Mesframe2,LineColor(kBlack),LineStyle(kDashed));
  totalbkg.plotOn(Mesframe2,LineColor(kBlack),Normalization(460,RooAbsPdf::NumEvent));
  totalbkg.paramOn(Mesframe2,Parameters(Argc));
  Mesframe2->SetMinimum(1e-6);
  Mesframe2->Draw();

  TCanvas *dE_Fit = new TCanvas("dE_Fit","Delta E");

  RooPlot* dEframe2= DeltaE.frame(-0.1,0.1,nbins);
  Data.plotOn(dEframe2,MarkerColor(kBlue));
  blindbkg.plotOn(dEframe2,LineColor(kBlack),LineStyle(kDashed));
  totalbkg.plotOn(dEframe2,LineColor(kBlack),Normalization(460,RooAbsPdf::NumEvent));
  totalbkg.paramOn(dEframe2,Parameters(dEslope));
  dEframe2->SetMinimum(1e-6);
  dEframe2->Draw();

  Mes.setRange(5.2,5.271);

  cout << "Integrating blind PDF..." << endl;
  RooAbsReal *iblindbkg = blindbkg.createIntegral(Mes);
  Mes.setRange(5.2,5.271);
  Double_t blindval = iblindbkg->getVal();
  Mes.setRange(5.271,5.29);
  Double_t blindval2 = iblindbkg->getVal();

  cout << "Integrating total PDF..." << endl;
  RooAbsReal *itotalbkg = totalbkg.createIntegral(Mes);
  Mes.setRange(5.2,5.271);
  Double_t totalval = itotalbkg->getVal();

  cout << "Blinded PDF integral, 5.200<mes<5.271: " << blindval << endl;
  cout << "Blinded PDF integral, 5.271<mes<5.290: " << blindval2 << endl;
  cout << "Total PDF integral, 5.2<mes<5.271: " << totalval << endl;

//   cout << "saving..." << endl; 
//   TFile *f = new TFile("~/Analysis23/workdir/BkgEstimateResult.root","RECREATE");
//   blindbkgresult->Write();
//   Mesframe2->Write();
//   dEframe2->Write();
//   f->Write();
//   f->Close();
//   cout << "done saving" << endl;

  delete h1;

}
