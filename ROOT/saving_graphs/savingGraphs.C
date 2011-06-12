int savingGraphs()
{
  TFile *rfile = new TFile("test.root","RECREATE");
  TGraph *mygraph[20];

  char name[256];

  float x[1] = {1.0};
  float y[1] = {2.0};

  gROOT->cd(rfile->GetPath());
  for(int i=0;i<20;i++)
  {
    mygraph[i] = new TGraph(1,x,y);
    sprintf(name,"gr%d",i);
    mygraph[i]->SetName(name);
    mygraph[i]->Write();
  } 
  rfile->Write();   
  rfile->Close();

}
