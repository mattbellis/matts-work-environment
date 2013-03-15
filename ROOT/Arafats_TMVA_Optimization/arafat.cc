//=====================================================
//#include "../inc/allincs.h"
//#include "../lib/all.h"
//#include <RooGlobalFunc.h>
#include "stdlib.h"
#include "time.h"

#include <iostream> // Stream declarations
#include <fstream>
#include <vector>
#include <string>
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TRandom3.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/MethodTMlpANN.h"
#include "TMVA/Tools.h"

#include "TMVA/Config.h"

//#include <TMVA/MethodCategory.h>

int main() {
    using namespace TMVA;
    int sig=0;
    double ETot, apl;

    ////////////////////////////////////////////////////////////////////////////
    // Keep track of where the data are that we will use
    // for training.
    ////////////////////////////////////////////////////////////////////////////
    char filename[256];
    char data_directory[256];
    sprintf(data_directory,"data/");

    ////////////////////////////////////////////////////////////////////////////
    // Open two (or more) files and grab two trees for signal (0 index)
    // and background (1 index).
    ////////////////////////////////////////////////////////////////////////////
    TFile *infiles[2]; // 0 - signal, 1 - background
    TChain *tchain[2];
    for (int i=0;i<2;i++)
    {
        tchain[i]  = new TChain("t");
        if (i==0) // Signal
        {
            sprintf(filename,"%s/ASA_7828_1.root",data_directory);
            tchain[i]->Add(filename);
            // Note that we can add more files here if we like!
        }
        else if (i==1) // Background
        {
            sprintf(filename,"%s/ASA_6_1.root",data_directory);
            tchain[i]->Add(filename);
            // Note that we can add more files here if we like!
        }
    }
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Create our TMVA factory and an output file.
    ////////////////////////////////////////////////////////////////////////////
    TFile *fout =new TFile("TMVA.root","RECREATE");
    TMVA::Factory factory("Analysis", fout, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification");
    factory.SetVerbose();

    ////////////////////////////////////////////////////////////////////////////
    // Declare the variables.
    ////////////////////////////////////////////////////////////////////////////
    //factory.AddVariable("sig",'I'); // Not using this for the time being
    factory.AddVariable("ETot",'D');
    factory.AddVariable("apl",'D');

    ////////////////////////////////////////////////////////////////////////////
    // For now we're not using this, since we're passing in two different
    // files with signal and background separated.
    ////////////////////////////////////////////////////////////////////////////
    TCut signalCut="sig==1";
    TCut bkgCut   ="sig==0";

    ////////////////////////////////////////////////////////////////////////////
    // Add the signal and background tree with a weight of 1.0
    ////////////////////////////////////////////////////////////////////////////
    factory.AddSignalTree(tchain[0],1.0);
    factory.AddBackgroundTree(tchain[1],1.0);

    ////////////////////////////////////////////////////////////////////////////
    // Prepare the tree and what methods we'll be using.
    ////////////////////////////////////////////////////////////////////////////
    factory.PrepareTrainingAndTestTree("", "");

    factory.BookMethod(TMVA::Types::kCuts,"Cuts","!H:!V:FitMethod=MC:EffSel:SampleSize=10000:VarProp=FSmart");
    //factory.BookMethod(TMVA::Types::kLD,"LD","H:!V:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10");
    factory.BookMethod(TMVA::Types::kLikelihood,"Likelihood","H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50");


    ////////////////////////////////////////////////////////////////////////////
    // Run it!
    ////////////////////////////////////////////////////////////////////////////
    factory.TrainAllMethods();
    factory.TestAllMethods();
    factory.EvaluateAllMethods();

    fout->Close();

    return 0;
}
//===================================================== End of code
