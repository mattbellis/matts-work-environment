/////////////////////////////////////////////////////////////////////////
//
// 'ADDITION AND CONVOLUTION' RooFit tutorial macro #208
// 
// One-dimensional numeric convolution
// (require ROOT to be compiled with --enable-fftw3)
// 
// pdf = landau(t) (x) gauss(t)
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
#include "RooLandau.h"
#include "RooFFTConvPdf.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TH1.h"
using namespace RooFit ;



void convolution_tests()
{
  // S e t u p   c o m p o n e n t   p d f s 
  // ---------------------------------------

  // Construct observable
  RooRealVar t("t","t",2.0,4.0) ;

  // Construct signal(t,ml,sl) ;
  RooRealVar ml("ml","mean signal",3.0);
  RooRealVar sl("sl","sigma signal",0.2);
  RooGaussian signal("signal","signal",t,ml,sl) ;
  
  // Construct gauss(t,mg,sg) for smearing
  RooRealVar mg("mg","mg",0) ;
  //RooRealVar sg("sg","sg",0.4);
  RooFormulaVar sg("sg","0.4",RooArgList(t));
  RooGaussian gauss("gauss","gauss",t,mg,sg) ;

  RooFormulaVar sg1("sg1","0.4+(0.4*@0)",RooArgList(t));
  RooGaussian gauss1("gauss1","gauss1",t,mg,sg1) ;

  // C o n s t r u c t   c o n v o l u t i o n   p d f 
  // ---------------------------------------

  // Set #bins to be used for FFT sampling to 10000
  t.setBins(10000,"cache") ; 

  // Construct signal (x) gauss
  
  RooNumConvPdf signalxg("signalxg","signal (X) gauss",t,signal,gauss) ;
  RooNumConvPdf signalxg1("signalxg1","signal (X) gauss1",t,signal,gauss1) ;


  // P l o t   c o n v o l u t e d   p d f 
  // ----------------------------------------------------------------------

  // Plot data, signal pdf, signal (X) gauss pdf
  RooPlot* framesig = t.frame(Title("signal")) ;
  signal.plotOn(framesig,LineStyle(kDashed)) ;

  RooPlot* framesigxg = t.frame(Title("signal (x) gauss convolution")) ;
  signalxg.plotOn(framesigxg) ;
  signalxg1.plotOn(framesigxg,LineColor(kRed)) ;

  // Draw frame on canvas
  TCanvas *can = new TCanvas("can","can",10,10,1000,500);
  can->Divide(2,1);

  can->cd(1);
  framesig->Draw() ;
  can->cd(2);
  framesigxg->Draw() ;

}



