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



void exponential_convolution()
{
  // S e t u p   c o m p o n e n t   p d f s 
  // ---------------------------------------

  // Construct observable
  RooRealVar t("t","t",0,20.0) ;

  // Construct landau(t,ml,sl) ;
  RooRealVar tau("tau","exponential slope",-3.5,-20,20) ;
  RooExponential expo("lx","lx",t,tau) ;
  
  // Construct gauss(t,mg,sg)
  RooRealVar mg("mg","mg",0) ;
  RooRealVar sg("sg","sg",0.51,0.0,1) ;
  RooGaussian gauss("gauss","gauss",t,mg,sg) ;


  // C o n s t r u c t   c o n v o l u t i o n   p d f 
  // ---------------------------------------

  // Set #bins to be used for FFT sampling to 10000
  t.setBins(10000,"cache") ; 

  // Construct expo (x) gauss00
  RooFFTConvPdf lxg("lxg","expo (X) gauss",t,expo,gauss) ;



  // S a m p l e ,   f i t   a n d   p l o t   c o n v o l u t e d   p d f 
  // ----------------------------------------------------------------------

  // Sample 1000 events in x from gxlx
  //RooDataSet* data = lxg.generate(t,1000000) ;

  // Fit gxlx to data
  //lxg.fitTo(*data) ;

  // Plot data, expo pdf, expo (X) gauss pdf
  RooPlot* frame = t.frame(Title("expo (x) gauss convolution")) ;
  //data->plotOn(frame) ;
  lxg.plotOn(frame) ;
  expo.plotOn(frame,LineStyle(kDashed),LineColor(kRed)) ;


  // Draw frame on canvas
  new TCanvas("rf208_convolution","rf208_convolution",600,600) ;
  gPad->SetLeftMargin(0.15) ; frame->GetYaxis()->SetTitleOffset(1.4) ; frame->Draw() ;

}



