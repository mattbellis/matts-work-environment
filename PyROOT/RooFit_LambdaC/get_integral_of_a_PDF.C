/////////////////////////////////////////////////////////////////////////
//
// 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #308
// 
// Examples on normalization of p.d.f.s,
// integration of p.d.fs, construction
// of cumulative distribution functions from p.d.f.s
// in two dimensions
//
// 07/2008 - Wouter Verkerke 
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooConstVar.h"
#include "RooProdPdf.h"
#include "RooAbsReal.h"
#include "RooPlot.h"
#include "RooArgSet.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TH1.h"
#include "RooWorkspace.h"

using namespace RooFit ;


void get_integral_of_a_PDF()
{
    // S e t u p   m o d e l 
    // ---------------------

    char name[512];
    sprintf(name,"workspace_Lambda0_ntp4_unblind_unblinded_data_testing_pos_PDF_sig0_bkg80_dim2_nfits1");
    char filename[512];
    sprintf(filename,"rootWorkspaceFiles/%s.root",name);

    TFile *f = new TFile(filename);
    RooWorkspace* w = (RooWorkspace*) f->Get(name) ;
    w->Print();

    //RooRealVar *x = w->var("x");
    //RooRealVar *y = w->var("y");
    //RooRealVar *z = w->var("z");
    RooRealVar x = *(w->var("x"));
    RooRealVar y = *(w->var("y"));
    RooRealVar z = *(w->var("z"));
    RooRealVar nbkg = *(w->var("nbkg"));
    RooRealVar br = *(w->var("branching_fraction"));
    RooFormulaVar *nsig = w->function("nsig");
    RooAbsPdf* bkg_pdf = w->pdf("bkg_pdf") ;
    RooAbsPdf* sig_pdf = w->pdf("sig_pdf") ;
    RooAbsPdf* total = w->pdf("total") ;
    x.Print();
    bkg_pdf->Print();
    sig_pdf->Print();
    nbkg.Print();
    nsig->Print();
    br.Print();

    // I n t e g r a t e   n o r m a l i z e d   p d f   o v e r   s u b r a n g e
    // ----------------------------------------------------------------------------

    // Define a range named "signal" in x from -5,5
    x.setRange("signal",5.270,5.30) ;
    y.setRange("signal",-0.048,0.048) ;

    x.setRange("full",5.200,5.30) ;
    y.setRange("full",-0.20,0.20) ;

    // Create an integral of gxy_Norm[x,y] over x and y in range "signal"
    // This is the fraction of of p.d.f. gxy_Norm[x,y] which is in the
    // range named "signal"
    ///*
    RooAbsReal* bkg_int_full = bkg_pdf->createIntegral(RooArgSet(x,y), NormSet(RooArgSet(x,y)), Range("full")) ;
    RooAbsReal* sig_int_full = sig_pdf->createIntegral(RooArgSet(x,y), NormSet(RooArgSet(x,y)), Range("full")) ;

    RooAbsReal* bkg_int_sigr = bkg_pdf->createIntegral(RooArgSet(x,y), NormSet(RooArgSet(x,y)), Range("signal")) ;
    RooAbsReal* sig_int_sigr = sig_pdf->createIntegral(RooArgSet(x,y), NormSet(RooArgSet(x,y)), Range("signal")) ;


    cout << endl;

    cout << "bkg int: full: " << bkg_int_full->getVal() << "\tsigr: " << bkg_int_sigr->getVal() << endl ;
    cout << "bkg num: full: " << nbkg.getVal()*bkg_int_full->getVal() << "\tsigr: " << nbkg.getVal()*bkg_int_sigr->getVal() << endl ;

    cout << endl;

    cout << "sig int: full: " << sig_int_full->getVal() << "\tsigr: " << sig_int_sigr->getVal() << endl ;
    cout << "sig num: full: " << nsig->getVal()*sig_int_full->getVal() << "\tsigr: " << nsig->getVal()*sig_int_sigr->getVal() << endl ;
    //*/

    /*
       ////////////////////////////////////////////////////////////////////////////////////////////////
       //////////////////////////// This doesn't seem to work. I get the same values for the integral
       ////////////////////////////////////////////////////////////////////////////////////////////////
    for (float xval=5.270;xval<5.290;xval+=0.002)
    {
        for (float yval=-0.050;yval<0.050;yval+=0.010)
        {
            RooArgSet obsx = RooArgSet(x);
            RooArgSet obsy = RooArgSet(y);
            RooArgSet obsxy = RooArgSet(x,y);

            x.setVal(xval);
            y.setVal(yval);
            cerr << "PDF value at signal region: " << x.getVal() << " " << y.getVal() << "\t";
            cerr << total->getVal(obsx) << " " << total->getVal(obsy) << " " << total->getVal(obsxy) << endl;
        }
    }
    */

    /*
    // Create observables x,y
    RooRealVar x("x","x",-10,10) ;
    RooRealVar y("y","y",-10,10) ;

    // Create p.d.f. gaussx(x,-2,3), gaussy(y,2,2)
    RooGaussian gx("gx","gx",x,RooConst(-2),RooConst(3)) ;
    RooGaussian gy("gy","gy",y,RooConst(+2),RooConst(2)) ;

    // Create gxy = gx(x)*gy(y)
    RooProdPdf gxy("gxy","gxy",RooArgSet(gx,gy)) ;


    // Define a range named "signal" in x from -5,5
    x.setRange("signal",-5,5) ;
    y.setRange("signal",-3,3) ;

    // Create an integral of gxy_Norm[x,y] over x and y in range "signal"
    // This is the fraction of of p.d.f. gxy_Norm[x,y] which is in the
    // range named "signal"
    RooAbsReal* igxy_sig = gxy.createIntegral(RooArgSet(x,y),NormSet(RooArgSet(x,y)),Range("signal")) ;
    cout << "gx_Int[x,y|signal]_Norm[x,y] = " << igxy_sig->getVal() << endl ;
     */








    // -----------------------------------------------------------------------------------------------------

    // Create the cumulative distribution function of gx
    // i.e. calculate Int[-10,x] gx(x') dx'
    //RooAbsReal* gxy_cdf = model.createCdf(RooArgSet(x,y)) ;

    // Plot cdf of gx versus x
    //TH1* hh_cdf = gxy_cdf->createHistogram("hh_cdf",x,Binning(40),YVar(y,Binning(40))) ;
    //hh_cdf->SetLineColor(kBlue) ;

    //new TCanvas("rf308_normintegration2d","rf308_normintegration2d",600,600) ;
    //gPad->SetLeftMargin(0.15) ; hh_cdf->GetZaxis()->SetTitleOffset(1.8) ; 
    //hh_cdf->Draw("surf") ;

}
