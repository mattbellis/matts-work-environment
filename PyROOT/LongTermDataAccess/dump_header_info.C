void dump_header_info(char *filename)
{

    TFile *f = new TFile(filename);
    TTree *t = (TTree*)gROOT->FindObject("T");

    // Get the user info from the TTree object
    TList *user_info = t->GetUserInfo();

    ///////////////////////////////////////////////////
    // Print it out
    ///////////////////////////////////////////////////

    user_info->Print();

    TIter iter(user_info);
    TList *nl = new TList();

    int nsize = user_info.GetSize();
    cerr << nsize << endl;

    for (int i=0;i<nsize;i++)
    {
        nl = (TList*)iter.Next();
        nl->Print();
    }

}
