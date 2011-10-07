#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooConstVar.h"
#include "RooFormulaVar.h"
#include "RooGenericPdf.h"
#include "RooPolynomial.h"
#include "RooChi2Var.h"
#include "RooMinuit.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "RooPlot.h"
#include "RooFitResult.h"
using namespace RooFit ;

void test_two_gaussians_and_exponentials_1_sub_ranges(int fit_subranges=0, bool use_roominuit=kFALSE)
{

    // Define our variables
    RooRealVar x("x","Energy",0.0,12.0);
    RooRealVar t("t","Time",1.0,500);

    // Define some different ranges over which we will fit.

    // Default (full range)
    t.setRange("FULL",1.0,500.0);
    x.setRange("FULL",0.0,12.0);

    // range 0 and range 1 are disconnected regions in t, and will
    // be fit simultaneously.
    t.setRange("range0",1.0,200.0);
    x.setRange("range0",0.0,12.0);

    t.setRange("range1",401.0,500.0);
    x.setRange("range1",0.0,12.0);

    // range 2 and range 3 are disconnected regions in t, and will
    // be fit simultaneously.
    t.setRange("range2",1.0,400.0);
    x.setRange("range2",0.0,12.0);

    t.setRange("range3",401.0,500.0);
    x.setRange("range3",0.0,12.0);

    ////////////////////////////////////////////////////////////////////////////
    // Set the seed so our results are reproducible.
    ////////////////////////////////////////////////////////////////////////////
    RooRandom::randomGenerator()->SetSeed(100);

    ////////////////////////////////////////////////////////////////////////////
    // Over what range are we fitting?
    ////////////////////////////////////////////////////////////////////////////
    char fit_range[256];
    if (fit_subranges==0)
    {
        sprintf(fit_range,"%s","FULL");
    }
    else if (fit_subranges==1)
    {
        sprintf(fit_range,"%s,%s","range0","range1");
    }
    else
    else if (fit_subranges==2)
    {
        sprintf(fit_range,"%s,%s","range2","range3");
    }
    

    ////////////////////////////////////////////////////////////////////////////
    // x (energy) component PDFs
    // Two Gaussians and an exponential
    ////////////////////////////////////////////////////////////////////////////
    RooRealVar mean0("mean0","mean0",4);
    RooRealVar sigma0("sigma0","sigma0",0.5);
    RooGaussian gauss0("gauss0","gauss0",x,mean0,sigma0);
    
    RooRealVar mean1("mean1","mean1",7);
    RooRealVar sigma1("sigma1","sigma1",0.5);
    RooGaussian gauss1("gauss1","gauss1",x,mean1,sigma1);

    RooRealVar slope_x("slope_x","slope_x",-0.3);
    RooExponential decay_x("decay_x","decay_x",x,slope_x);

    ////////////////////////////////////////////////////////////////////////////
    // y (time) component PDFs
    // Let the two Gaussians (in x) have an exponential decay in time.
    ////////////////////////////////////////////////////////////////////////////
    RooRealVar slope0("slope0","slope0",-0.005);
    RooRealVar slope1("slope1","slope1",-0.02);

    RooExponential decay0("decay0","decay0",t,slope0);
    RooExponential decay1("decay1","decay1",t,slope1);

    ////////////////////////////////////////////////////////////////////////////
    // Create the product of the two Gaussians (x) and their exponential (t)
    ////////////////////////////////////////////////////////////////////////////
    RooProdPdf prod0("prod0","prod0",RooArgList(decay0,gauss0));
    RooProdPdf prod1("prod1","prod1",RooArgList(decay1,gauss1));

    // Create the total PDF (RooAddPdf) for these 3 components.
    RooRealVar n0("n0","n0",1000); // Number of events in Gaussian 0 (energy)
    RooRealVar n1("n1","n1",500);  // Number of events in Gaussian 1 (energy)
    RooRealVar n2("n2","n2",1000); // Number of events in the exponential (energy)

    RooAddPdf *total = new RooAddPdf("total","total",RooArgList(prod0,prod1,decay_x),RooArgList(n0,n1,n2));

    ////////////////////////////////////////////////////////////////////////////
    // Generate data over the whole range and create a reduced dataset which
    // will be used in the fits.
    ////////////////////////////////////////////////////////////////////////////
    RooDataSet* data = total->generate(RooArgSet(x,t),2500);

    RooDataSet* data_reduce = (RooDataSet*)data->reduce(CutRange("FULL"));
    RooDataSet* data_reduce0 = (RooDataSet*)data->reduce(CutRange("range0"));
    RooDataSet* data_reduce1 = (RooDataSet*)data->reduce(CutRange("range1"));
    RooDataSet* data_reduce2 = (RooDataSet*)data->reduce(CutRange("range2"));
    RooDataSet* data_reduce3 = (RooDataSet*)data->reduce(CutRange("range3"));
    if (fit_subranges==1)
    {
        data_reduce = (RooDataSet*)data->reduce(CutRange("range0"));
        data_reduce->append((RooDataSet)data->reduce(CutRange("range1")));
    }
    else if (fit_subranges==2)
    {
        data_reduce = (RooDataSet*)data->reduce(CutRange("range2"));
        data_reduce->append((RooDataSet)data->reduce(CutRange("range3")));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Create some RooPlot frames in which we will plot the results.
    ////////////////////////////////////////////////////////////////////////////
    x.setBins(50);
    t.setBins(50);
    
    RooPlot *frame_x[5];
    RooPlot *frame_t[5];
    for (int i=0;i<5;i++)
    {
        char title[256];
        if (i==0)      sprintf(title,"No range specified");
        else if (i==1) sprintf(title,"Range:FULL, NormRange:FULL");
        else if (i==2) sprintf(title,"Range:fit_range, NormRange:FULL");
        else if (i==3) sprintf(title,"Range:fit_range, NormRange:fit_range");
        else if (i==4) sprintf(title,"Range:FULL, NormRange:fit_range");

        frame_x[i] = x.frame(Title(title));
        frame_t[i] = t.frame(Title(title));

        //data->plotOn(frame_x[i]);
        //data->plotOn(frame_t[i]);

        data_reduce->plotOn(frame_x[i]);
        data_reduce->plotOn(frame_t[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Float *only* the numbers of events in the three components. 
    // Start them at some consistent values.
    ////////////////////////////////////////////////////////////////////////////
    n0.setVal(1000);
    n0.setConstant(kFALSE);

    n1.setVal(500);
    n1.setConstant(kFALSE);

    n2.setVal(1000);
    n2.setConstant(kFALSE);

    // Set things up in case we call RooMinuit
    //RooNLLVar nll = RooNLLVar("nll","nll",*total,*data_reduce,Extended(kTRUE),Range(fit_range),SplitRange(kTRUE));
    //RooNLLVar nll = RooNLLVar("nll","nll",*total,*data_reduce,Extended(kTRUE),Range(fit_range));

    // Try this like in the RooAbsPdf code
    RooArgList nllList ;
    //RooAbsReal* nllComp = new RooNLLVar("nll_0","-log(likelihood)",*this,data,projDeps,ext,token,addCoefRangeName,numcpu,kFALSE,verbose,splitr,cloneData) ;
    //RooAbsReal* nllComp0 = new RooNLLVar("nll_range0","-log(likelihood)",*total,*data_reduce,Extended(kTRUE),Range("range0"),SumCoefRange(""));
    //RooAbsReal* nllComp0 = new RooNLLVar("nll_range0","-log(likelihood)",*total,*data_reduce,Extended(kTRUE),Range("range0"),CloneData(kFALSE));
    //RooAbsReal* nllComp1 = new RooNLLVar("nll_range1","-log(likelihood)",*total,*data_reduce,Extended(kTRUE),Range("range1"));
    RooAbsReal* nllComp0 = new RooNLLVar("nll_range0","-log(likelihood)",*total,*data_reduce,kTRUE,"range0",0,1,kFALSE,kFALSE,kFALSE,kFALSE);
    //RooAbsReal* nllComp0 = new RooNLLVar("nll_range0","-log(likelihood)",*total,*data_reduce,Extended(kTRUE),Range("range0"));
    nllList.add(*nllComp0) ;
    //RooAbsReal* nllComp1 = new RooNLLVar("nll_range1","-log(likelihood)",*total,*data_reduce,Extended(kTRUE),Range("range1"),SumCoefRange(""));
    //RooAbsReal* nllComp1 = new RooNLLVar("nll_range1","-log(likelihood)",*total,*data_reduce,Extended(kTRUE),Range("range1"),CloneData(kFALSE),interleave(kFALSE));
    //RooAbsReal* nllComp1 = new RooNLLVar("nll_range1","-log(likelihood)",*total,*data_reduce,Extended(kTRUE),Range("range1"));
    RooAbsReal* nllComp1 = new RooNLLVar("nll_range1","-log(likelihood)",*total,*data_reduce,kTRUE,"range1",0,1,kFALSE,kFALSE,kFALSE,kFALSE);
    nllList.add(*nllComp1) ;
    nll = new RooAddition("nll","-log(likelihood)",nllList,kTRUE);

    //RooFormulaVar fit_func = RooFormulaVar("fit_func","nll",RooArgList(nll));
    //RooFormulaVar fit_func = RooFormulaVar("fit_func","@0",RooArgList(*nll));
    RooAddition *fit_func = new RooAddition(*nll);
    RooMinuit m = RooMinuit(*fit_func);

    ////////////////////////////////////////////////////////////////////////////
    // Fit with either RooMinuit or the fitTo member function of the PDF.
    ////////////////////////////////////////////////////////////////////////////
    RooFitResult *results = NULL;
    if (!use_roominuit)
    {
        //results = total->fitTo(*data_reduce,Save(kTRUE),Range(fit_range),Extended(kTRUE));
        results = total->fitTo(*data_reduce,Save(kTRUE),Range(fit_range),Extended(kTRUE));
    }
    else
    {
        m.setVerbose(kFALSE);
        m.migrad();
        m.hesse();
        results = m.save();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Plot the results
    ////////////////////////////////////////////////////////////////////////////
    TCanvas *can = new TCanvas("can","can",10,10,1000,900);
    can->SetFillColor(0);
    can->Divide(2,5);


    total->plotOn(frame_x[0]);
    total->plotOn(frame_t[0]);

    total->plotOn(frame_x[1],Range("FULL"),NormRange("FULL"));
    total->plotOn(frame_t[1],Range("FULL"),NormRange("FULL"));

    total->plotOn(frame_x[2],Range(fit_range),NormRange("FULL"));
    total->plotOn(frame_t[2],Range(fit_range),NormRange("FULL"));

    total->plotOn(frame_x[3],Range(fit_range),NormRange(fit_range));
    total->plotOn(frame_t[3],Range(fit_range),NormRange(fit_range));

    total->plotOn(frame_x[4],Range("FULL"),NormRange(fit_range));
    total->plotOn(frame_t[4],Range("FULL"),NormRange(fit_range));

    for (int i=0;i<5;i++)
    {
        can->cd((i*2)+1);
        frame_x[i]->Draw();
        gPad->Update();

        can->cd((i*2)+2);
        frame_t[i]->Draw();
        gPad->Update();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Print out the results.
    ////////////////////////////////////////////////////////////////////////////
    results->Print("v");

    int nentries = data_reduce->numEntries();
    //int nentries = data_reduce0->numEntries() + data_reduce1->numEntries();
    printf("num entries in dataset: %d\n",nentries);
    printf("fit results:\n");
    printf("\tn0: %6.2f\n",n0.getVal());
    printf("\tn1: %6.2f\n",n1.getVal());
    printf("\tn2: %6.2f\n",n2.getVal());
    float total_from_fit = n0.getVal()+n1.getVal()+n2.getVal();
    printf("\ttotal num events from fit: %6.2f\n",total_from_fit);
    printf("\tdifference between fit and num entries: %6.2f\n",total_from_fit-nentries);

}

