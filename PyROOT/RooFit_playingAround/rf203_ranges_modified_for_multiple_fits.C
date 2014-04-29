/////////////////////////////////////////////////////////////////////////
//
// 'ADDITION AND CONVOLUTION' RooFit tutorial macro #203
// 
// Fitting and plotting in sub ranges
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
#include "RooPolynomial.h"
#include "RooAddPdf.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TH1.h"
using namespace RooFit ;


void rf203_ranges()
{
  // S e t u p   m o d e l 
  // ---------------------

  // Construct observables x
  RooRealVar x("x","x",-10,10) ;
  
  // Construct gaussx(x,mx,1)
  RooRealVar mx("mx","mx",0,-10,10) ;
  RooGaussian gx("gx","gx",x,mx,RooConst(1)) ;

  // Construct px = 1 (flat in x)
  RooPolynomial px("px","px",x) ;

  // Construct model = f*gx + (1-f)px
  RooRealVar f("f","f",0.,1.) ;
  RooAddPdf model("model","model",RooArgList(gx,px),f) ;

  for (int n=0;n<100;n++)
  {
      cerr << "Fitting " << n << " --------------------------" << endl;
      // Generated 10000 events in (x,y) from p.d.f. model
      RooDataSet* modelData = model.generate(x,10000) ;

      // F i t   f u l l   r a n g e 
      // ---------------------------

      // Fit p.d.f to all data
      RooFitResult* r_full = model.fitTo(*modelData,Save(kTRUE)) ;


      // F i t   p a r t i a l   r a n g e 
      // ----------------------------------

      // Define "signal" range in x as [-3,3]
      x.setRange("signal",-3,3) ;  

      // Fit p.d.f only to data in "signal" range
      RooFitResult* r_sig = model.fitTo(*modelData,Save(kTRUE),Range("signal")) ;


      // Print fit results 
      cout << "result of fit on all data " << endl ;
      r_full->Print() ;  
      cout << "result of fit in in signal region (note increased error on signal fraction)" << endl ;
      r_sig->Print() ;
  }

  return ;
  
}
