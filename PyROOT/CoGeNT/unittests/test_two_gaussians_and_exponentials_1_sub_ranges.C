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


void test_two_gaussians_and_exponentials_1_sub_ranges()
{

    RooRealVar x("x","ionization energy (keVee)",0.0,12.0);
    RooRealVar t("t","time",1.0,500);

    t.setRange("range0",1.0,200.0);
    x.setRange("range0",0.0,12.0);

    t.setRange("range1",401.0,500.0);
    x.setRange("range1",0.0,12.0);

    t.setRange("FULL",1.0,500.0);
    x.setRange("FULL",0.0,12.0);


    //################################################################################
    //# x terms
    //################################################################################
    RooRealVar mean0("mean0","mean0",4);
    RooRealVar sigma0("sigma0","sigma0",0.5);
    RooGaussian gauss0("gauss0","gauss0",x,mean0,sigma0);
    
    RooRealVar mean1("mean1","mean1",7);
    RooRealVar sigma1("sigma1","sigma1",0.5);
    RooGaussian gauss1("gauss1","gauss1",x,mean1,sigma1);

    RooRealVar slope_x("slope_x","slope_x",-0.3);
    RooExponential decay_x("decay_x","decay_x",x,slope_x);

    //################################################################################
    //# t terms
    //################################################################################
    RooRealVar slope0("slope0","slope0",-0.005);
    RooRealVar slope1("slope1","slope1",-0.02);

    RooExponential decay0("decay0","decay0",t,slope0);
    RooExponential decay1("decay1","decay1",t,slope1);

    RooProdPdf prod0("prod0","prod0",RooArgList(decay0,gauss0));
    RooProdPdf prod1("prod1","prod1",RooArgList(decay1,gauss1));

    RooRealVar n0("n0","n0",1000);
    RooRealVar n1("n1","n1",500);
    RooRealVar n2("n2","n2",1000);

    RooAddPdf *total = new RooAddPdf("total","total",RooArgList(prod0,prod1,decay_x),RooArgList(n0,n1,n2));

    x.setBins(50);
    t.setBins(50);
    RooPlot *frame_x = x.frame(RooFit::Title("x"));
    RooPlot *frame_t = t.frame(RooFit::Title("t"));

    RooDataSet *data = total->generate(RooArgSet(x,t),2500);
    RooDataSet *data_reduce = data->reduce(CutRange("range0,range1"));

    n0.setVal(1000);
    n0.setConstant(kFALSE);

    n1.setVal(500);
    n1.setConstant(kFALSE);

    n2.setVal(1000);
    n2.setConstant(kFALSE);

    char fit_range[256];
    sprintf(fit_range,"%s,%s","range0","range1");
    //sprintf(fit_range,"%s","FULL");
    

    //nll = RooNLLVar("nll","nll",total,data,RooFit::Extended(kTRUE),RooFit::Range(fit_range));

    //nll = RooNLLVar("nll","nll",total,data,RooFit::Extended(kTRUE));
    //fit_func = RooFormulaVar("fit_func","nll",RooArgList(nll));
    //m = RooMinuit(fit_func);
    //m.setVerbose(kFALSE);
    //m.migrad();
    //m.hesse();
    //results = m.save();

    //RooFitResult results = total->fitTo(data,RooFit::Save(kTRUE),RooFit::Range(fit_range),RooFit::Extended(kTRUE));
    //RooFitResult *results = total->fitTo(*data,Save(kTRUE),Range(fit_range));
    RooFitResult *results = total->fitTo(*data_reduce,Save(kTRUE),Range(fit_range));

    results->Print("v");

    data->plotOn(frame_x);
    data->plotOn(frame_t);

    TCanvas *can = new TCanvas("can","can",10,10,1000,600);
    can->SetFillColor(0);
    can->Divide(2,1);

    //################################################################################



    can->cd(1);
    RooArgSet *rargset = new RooArgSet(*total);
    //total->plotOn(frame_x,RooFit::Components(rargset),RooFit::LineColor(3),RooFit::Range(fit_range),RooFit::NormRange("FULL"));
    total->plotOn(frame_x,Range(fit_range),NormRange("FULL"));
    //#total->plotOn(frame_x,RooFit::Components(rargset),RooFit::LineColor(3));

    frame_x->Draw();
    gPad->Update();

    can->cd(2);
    RooArgSet *rargset = new RooArgSet(*total);
    //total->plotOn(frame_x,RooFit::Components(rargset),RooFit::LineColor(3),RooFit::Range(fit_range),RooFit::NormRange("FULL"));
    total->plotOn(frame_t,Range(fit_range),NormRange("FULL"));
    //#total->plotOn(frame_x,RooFit::Components(rargset),RooFit::LineColor(3));

    frame_t->Draw();
    gPad->Update();
}

/*
#rargset = RooArgSet(prod0)
#total.plotOn(frame_x,RooFit::Components(rargset),RooFit::LineColor(4),RooFit::Range(fit_range),RooFit::NormRange("FULL"))

#rargset = RooArgSet(prod1)
#total.plotOn(frame_x,RooFit::Components(rargset),RooFit::LineColor(2),RooFit::Range(fit_range),RooFit::NormRange("FULL"))

#rargset = RooArgSet(decay_x)
#total.plotOn(frame_x,RooFit::Components(rargset),RooFit::LineColor(22),RooFit::Range(fit_range),RooFit::NormRange("FULL"))


################################################################################

can.cd(2)
rargset = RooArgSet(total)
total.plotOn(frame_t,RooFit::Components(rargset),RooFit::LineColor(3),RooFit::Range(fit_range),RooFit::NormRange("FULL"))

rargset = RooArgSet(prod0)
total.plotOn(frame_t,RooFit::Components(rargset),RooFit::LineColor(4),RooFit::Range(fit_range),RooFit::NormRange("FULL"))

rargset = RooArgSet(prod1)
total.plotOn(frame_t,RooFit::Components(rargset),RooFit::LineColor(2),RooFit::Range(fit_range),RooFit::NormRange("FULL"))

rargset = RooArgSet(decay_x)
total.plotOn(frame_t,RooFit::Components(rargset),RooFit::LineColor(22),RooFit::Range(fit_range),RooFit::NormRange("FULL"))

frame_t.Draw()
gPad.Update()

*/

