#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH2.h"
#include "TRandom.h"

int read_from_tree(char* filename)
{
    TFile *f = new TFile(filename);

    TTree *t1 = (TTree*)f->Get("ntp1");

    t1->Print();

    // Declare the variables we will need.
    int nB;
    // Note that these are arrays, because you may
    // have multiple B-candidates.
    int Bd1Idx[32], Bd2Idx[32];
    int Bd1Lund[32], Bd2Lund[32];
    float protoncosth[32];

    // Initialize the addresses (?)
    t1->SetBranchAddress("nB",&nB);
    // Because these are arrays, we don't need the ampersand
    // for the second argument.
    t1->SetBranchAddress("Bd1Idx",Bd1Idx);
    t1->SetBranchAddress("Bd2Idx",Bd2Idx);
    t1->SetBranchAddress("Bd1Lund",Bd1Lund);
    t1->SetBranchAddress("Bd2Lund",Bd2Lund);
    t1->SetBranchAddress("protoncosth",protoncosth);

    // Create some empty histograms
    TH1F *hnB   = new TH1F("hnB","The number of B candidates",10,0,10);

    // Get the number of entries in this file.
    Int_t nentries = (Int_t)t1->GetEntries();

    // Loop over the entries
    for (Int_t i=0;i<nentries;i++) {

        t1->GetEntry(i);

        hnB->Fill(nB);

        printf("%d ---------\n",i);

        // Loop over the number of B-candidates
        for (Int_t j=0;j<nB;j++) {
            printf(" %d ---\n",j);
            printf("%d %d\n",Bd1Lund[j],Bd2Lund[j]);
            printf("%d %f\n",Bd1Idx[j],protoncosth[Bd1Idx[j]]);
        }

    }

    hnB->Draw();


}
