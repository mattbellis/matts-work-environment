#include "TChain.h"
#include "TBranch.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooArgSet.h"
#include "RooGaussian.h"
#include "RooAddPdf.h"
#include "RooGlobalFunc.h"
#include "RooPlot.h"
#include "RooBifurGauss.h"
#include "RooPolynomial.h"

using namespace RooFit;

void fitpi0(const char* file, const char *ntpname="ntp5") {

  TChain *t= new TChain(ntpname);
  t->Add(file);

  RooRealVar mass("mass","mass",0.100,0.160);
  RooDataSet data("data","data",RooArgSet(mass));

  int nentries= t->GetEntries();
  int npi0;
  float pi0Mass[500];
  TBranch        *b_npi0, *b_pi0Mass;   //!

  t->SetBranchStatus("*",0);
  t->SetBranchStatus("npi0",1);
  t->SetBranchStatus("pi0Mass",1);

  t->SetBranchAddress("npi0", &npi0, &b_npi0);
  t->SetBranchAddress("pi0Mass", pi0Mass, &b_pi0Mass);

  for ( int n=0; n<nentries; n++) {
    t->GetEntry(n);
    for ( int i=0; i<npi0; i++) {
      if ( pi0Mass[i]>mass.getMin() && pi0Mass[i]<mass.getMax()) {
	mass.setVal(pi0Mass[i]);
	data.add(RooArgSet(mass));
      }
    }
  }

  RooRealVar mean("mean","mean",0.12,0.15);
  RooRealVar sigL("sigL","sigL",0.006,0.001,0.015);
  RooRealVar sigR("sigR","sigR",0.006,0.001,0.015);
  RooBifurGauss peak("peak","peak", mass, mean, sigL, sigR);
  RooRealVar a1("a1","slope",0,-5,5);
  RooPolynomial bkg("bkg","bkg", mass, RooArgList(a1));
  RooRealVar frac("frac","peak fraction",0.03,0,0.3);
  RooAddPdf model("model","model",peak,bkg,frac);

  model.fitTo(data,Hesse(0));
  RooPlot *frame= mass.frame();
  data.plotOn(frame);
  model.plotOn(frame);
  frame->Draw();

}
