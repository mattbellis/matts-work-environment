//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #708
// 
// Special decay pdf for B physics with mixing and/or CP violation
//
//
//
// 07/2008 - Wouter Verkerke 
// 
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooConstVar.h"
#include "RooCategory.h"
#include "RooBMixDecay.h"
#include "RooBCPEffDecay.h"
#include "RooBDecay.h"
#include "RooFormulaVar.h"
#include "RooTruthModel.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "RooPlot.h"
using namespace RooFit ;

void rf708_bphysics_only_RooBMix()
{
  ////////////////////////////////////////////////////
  // B - D e c a y   w i t h   m i x i n g          //
  ////////////////////////////////////////////////////

  // C o n s t r u c t   p d f 
  // -------------------------
  
  // Observable
  RooRealVar dt("dt","dt",-10,10) ;
  dt.setBins(40) ;

  // Parameters
  RooRealVar dm("dm","delta m(B0)",0.472) ;
  RooRealVar tau("tau","tau (B0)",1.547) ;
  RooRealVar w("w","flavour mistag rate",0.1) ;
  RooRealVar dw("dw","delta mistag rate for B0/B0bar",0.1) ;

  RooCategory mixState("mixState","B0/B0bar mixing state") ;
  mixState.defineType("mixed",-1) ;
  mixState.defineType("unmixed",1) ;

  RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
  tagFlav.defineType("B0",1) ;
  tagFlav.defineType("B0bar",-1) ;

  // Use delta function resolution model
  RooTruthModel tm("tm","truth model",dt) ;

  //////////////////////////////////////////////////////////////////////////////////
  // G e n e r i c   B   d e c a y  w i t h    u s e r   c o e f f i c i e n t s  //
  //////////////////////////////////////////////////////////////////////////////////

  // C o n s t r u c t   p d f 
  // -------------------------
  
  // Model parameters
  RooRealVar DGbG("DGbG","DGamma/GammaAvg",0.5,-1,1);
  RooRealVar Adir("Adir","-[1-abs(l)**2]/[1+abs(l)**2]",0);
  RooRealVar Amix("Amix","2Im(l)/[1+abs(l)**2]",0.7);
  RooRealVar Adel("Adel","2Re(l)/[1+abs(l)**2]",0.7);
  
  // Derived input parameters for pdf
  RooFormulaVar DG("DG","Delta Gamma","@1/@0",RooArgList(tau,DGbG));
  
  // Construct coefficient functions for sin,cos,sinh modulations of decay distribution
  //RooFormulaVar fsin("fsin","fsin","@0*@1*(1-2*@2)",RooArgList(Amix,tagFlav,w,mixState));
  //RooFormulaVar fcos("fcos","fcos","@0*@1*(1-2*@2)",RooArgList(Adir,tagFlav,w,mixState));
  //RooFormulaVar fsinh("fsinh","fsinh","@0",RooArgList(Adel));

  //RooFormulaVar fsin("fsin","fsin","0*@0",RooArgList(mixState,tagFlav));
  //RooFormulaVar fcos("fcos","fcos","@0",RooArgList(mixState,tagFlav));
  //RooFormulaVar fsinh("fsinh","fsinh","0*@0",RooArgList(mixState,tagFlav));

  //RooConst fsin(0);
  //RooConst fsinh(0);
  RooFormulaVar fcos("fcos","fcos","@0",RooArgList(mixState,tagFlav));
  
  // Construct generic B decay pdf using above user coefficients
  //RooBDecay bcpg("bcpg","bcpg",dt,tau,DG,RooConst(1),fsinh,fcos,fsin,dm,tm,RooBDecay::DoubleSided);
  RooBDecay bcpg("bcpg","bcpg",dt,tau,DG,RooConst(1),RooConst(0),fcos,RooConst(0),dm,tm,RooBDecay::DoubleSided);
  
  
  
  // P l o t   -   I m ( l ) = 0 . 7 ,   R e ( l ) = 0 . 7   | l | = 1,   d G / G = 0 . 5 
  // -------------------------------------------------------------------------------------
  
  // Generate some data
  RooDataSet* data4 = bcpg.generate(RooArgSet(dt,tagFlav,mixState),10000) ;
  
  // Plot B0 and B0bar tagged data separately 
  RooPlot* frame6 = dt.frame(Title("B decay distribution with CPV(Im(l)=0.7,Re(l)=0.7,|l|=1,dG/G=0.5) (B0/B0bar)")) ;  
  
  //data4->plotOn(frame6,Cut("tagFlav==tagFlav::B0")) ;
  //bcpg.plotOn(frame6,Slice(tagFlav,"B0")) ;
  
  //data4->plotOn(frame6,Cut("tagFlav==tagFlav::B0bar"),MarkerColor(kCyan)) ;
  //bcpg.plotOn(frame6,Slice(tagFlav,"B0bar"),LineColor(kCyan)) ;
  
  data4->plotOn(frame6,Asymmetry(mixState));
  cerr << "Printing PDF asymmetry......" << endl;
  bcpg.plotOn(frame6,RooFit::ProjWData(RooArgSet(mixState),*data4,kTRUE),Asymmetry(mixState));
  
 
 

  TCanvas* c = new TCanvas("rf708_bphysics","rf708_bphysics",1200,800) ;
  c->Divide(1,1) ;
  c->cd(1) ; gPad->SetLeftMargin(0.15) ; frame6->GetYaxis()->SetTitleOffset(1.6) ; frame6->Draw() ;
  
}
