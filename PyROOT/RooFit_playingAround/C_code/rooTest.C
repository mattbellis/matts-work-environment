#ifndef __CINT__
#endif

#include <TRandom3.h>
#include <TTree.h>
#include <TH1F.h>
#include <RooFit.h>

#include <iostream>
using namespace std;
//using namespace RooFit ;

Double_t user_cfun(Double_t *x,Double_t *pars) {
   Double_t res;

   if (x[0]>=.5 && x[0]<.9) {
      TF1::RejectPoint();
   }

   Double_t bkg = 1+x[0]*pars[1];
   bkg /= 1+pars[1]/2.;

   Double_t sgn = TMath::Gaus(x[0],pars[3],pars[4],kTRUE);

   res = pars[0]*(pars[2]*bkg+(1-pars[2])*sgn);

   return res;
}

TTree *generate_data(Int_t nevts) {
   // generate  the data sample
   gROOT->Delete("data");
   TTree *tree = new TTree("data","dummy data");
   
   // variables
   Double_t mass;
   Int_t dtype;

   // create tree branches
   tree->Branch("mass",&mass,"mass/D");
   tree->Branch("type",&dtype,"type/I");

   // data extraction loop
   for (Int_t i=0;i!=10000;++i) {
      Double_t type = gRandom->Uniform();
      if (type<.6) { // generate signal A
	 dtype = 1;
	 mass = gRandom->Gaus(.42,.12);
      }
      else if (type<.7) { // generate signal B
	 dtype = 2;
	 mass = gRandom->Gaus(.7,.08);
      }
      else { // generate background events
	 dtype = 0;
	 Double_t shape = -.1;
	 mass = (-1+TMath::Sqrt(1+gRandom->Uniform()*2*(1+shape/2)*shape))/shape;
      }

      tree->Fill();
   } // end extractino loop

   return tree;
}

void rooTest(Int_t nevts=10000, Bool_t binned=kTRUE)
{

   cout << "=== define pdfs and vars" << endl;
   // define the fit variables
   RooRealVar Mass("mass","Inv. mass [GeV/c^{2}]",0,1);
   Mass.setRange("low",0,.5);
   Mass.setRange("high",.9,1);

   // define fit parameters
   RooRealVar Peak1("Peak1","Peak 1",.4,.3,.5);
   RooRealVar Width1("Weak1","Width 1",.1,.1,.3);
   RooRealVar Peak2("Peak2","Peak 2",.8,.65,.82);
   RooRealVar Width2("Weak2","Width 2",.1,.01,.3);
   RooRealVar bkg_shape("bkg_shape","shape",0,-1,0);
   RooRealVar frac_bkg("frac_bkg","Bkg frac",.1,0,.5);
   RooRealVar frac_sgn("frac_sgn","Sgn frac",.3,0,.5);

   RooRealVar num_bkg("num_bkg","Background count",2000,0,10000);
   RooRealVar num_sig("num_sig","Signal count",2000,0,10000);


   // define the fit model
   RooGaussian sgn1("sgn1","gaus(Mass,Peak1,Width1)",
		    Mass,Peak1,Width1);
   RooGaussian sgn2("sgn1","gaus(Mass,Peak2,Width2)",
		    Mass,Peak2,Width2);
   RooAddPdf sgn("sgn","Signals",sgn2,sgn1,frac_sgn);
   RooPolynomial bkg("bkg","Combinatoric Background",
		     Mass,bkg_shape,1);
   RooAddPdf model_complete("model","Mass model",bkg,sgn,frac_bkg);
   RooAddPdf model_partial("model","Mass model",RooArgList(bkg,sgn1),RooArgList(num_bkg,num_sig));

   cout << "=== take data" << endl;
   // take data
   TTree *data = generate_data(nevts);

   // create the dataset
   gROOT->Delete("histo_mass");
   TH1F *histo_mass = new TH1F("histo_mass","Mass",50,0,1);
   data->Draw("mass>>histo_mass",0,"goff");
   RooDataHist data_histo("data_histo","Binned dataset",Mass,histo_mass);
   RooDataSet data_unbinned("data_unbinned","Dataset",data,Mass);

   // fit the data
   model_partial.printCompactTree();
   RooFitResult *res_low;
   RooFitResult *res_high;
   RooFitResult *res_low_high;
   RooFitResult *res_compl = 0;
   if (binned) {
      //res_low = model_partial.fitTo(data_histo,RooFit::Range("low"),RooFit::Save(kTRUE));
      //res_high = model_partial.fitTo(data_histo,RooFit::Range("high"),RooFit::Save(kTRUE));
      res_low_high = model_partial.fitTo(data_histo,RooFit::Range("low,high"),RooFit::Save(kTRUE));
      //res_compl = model_complete.fitTo(data_histo,RooFit::Save(kTRUE));
   }
   else {
      //res_low = model_partial.fitTo(data_unbinned,RooFit::Range("low"),RooFit::Save(kTRUE));
      //res_high = model_partial.fitTo(data_unbinned,RooFit::Range("high"),RooFit::Save(kTRUE));
      res_low_high = model_partial.fitTo(data_unbinned,RooFit::Range("low,high"),RooFit::Save(kTRUE));
      //res_compl = model_complete.fitTo(data_unbinned,RooFit::Save(kTRUE),RooFit::Range("Full"));
   }

   // print fit results
   cout << "*** low sideband ***" << endl;
   if (res_low) res_low->Print();
   else cout << "No fit results" << endl;

   cout << "*** high sideband ***" << endl;
   if (res_high) res_high->Print();
   else cout << "No fit results" << endl;

   cout << "*** low+high sidebands ***" << endl;
   if (res_low_high) res_low_high->Print();
   else cout << "No fit results" << endl;

   cout << "*** compl sidebands ***" << endl;
   if (res_compl) res_compl->Print();
   else cout << "No fit results" << endl;

   TF1 *user = new TF1("user",user_cfun,0,1,5);
   user->SetParameters(1000,0,.3,.4,.1);
   user->SetParLimits(0,100,1e6);
   user->SetParLimits(1,-1,0);
   user->SetParLimits(2,0,.5);
   user->SetParLimits(3,.3,.5);
   user->SetParLimits(4,.1,.3);

   // show the projections
   RooPlot *Mass_proj = Mass.frame();
   TCanvas *canvas_bin = new TCanvas("c1","Mass projection (binned)",1);
   if (binned) {
      data_histo.plotOn(Mass_proj);
   }
   else {
      data_unbinned.plotOn(Mass_proj);
   }

   // Plot only fitted ranges
   model_partial.plotOn(Mass_proj) ;
   model_partial.plotOn(Mass_proj,RooFit::Components(sgn1),RooFit::LineStyle(kDotted)) ;
   model_partial.plotOn(Mass_proj,RooFit::Components(bkg),RooFit::LineStyle(kDashed)) ;

   // Plot background in full range, but normalize to fitted ranges
   model_partial.plotOn(Mass_proj,RooFit::Range("full"),RooFit::NormRange("low,high"),RooFit::Components(bkg),RooFit::LineStyle(kDashed),RooFit::LineColor(kRed),RooFit::MoveToBack()) ;

   Mass_proj->Draw();
   canvas_bin->Update();
   if (binned) h_data_histo->Fit("user","+");
   else h_data_unbinned->Fit("user","+");
   canvas_bin->SaveAs(Form("rootTest_%s.gif",(binned ? "binned" : "unbinned")));
}
