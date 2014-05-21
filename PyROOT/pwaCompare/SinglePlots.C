void SinglePlots(char *fit="small_s-channel", int lomin=1450, int itopology=0, int ikinvar=0, int iiteration=0, char *tag="")
{
  //gROOT->Reset();
  // Display no statisitics on the histograms
  gStyle->SetOptStat(0);
  // Set some drawing styles
  gStyle->SetLabelSize(0.05,"X");
  gStyle->SetPalette(1,0);

  char *whichtopology="ppippim";

  ///////////////////////////////////////////////////////////////////

  string kv[12] = {"ppipM", "ppimM", "pippimM", "pCosThetaCM", "pipCosThetaCM", "pimCosThetaCM",
    "pipCosThetaDpHel", "pipPhiDpHel", 
    "pimCosThetaD0Hel", "pimPhiD0Hel", 
    "pipCosThetaRhoHel", "pipPhiRhoHel"};

  string type[3] = {"data","acc","acc_wt"};
  string topology[4] = {"ppippim", "ppip_pim", "ppim_pip", "pippim_p"};
  string topology_latex[6] = {"p#pi^{+}#pi^{-}", "p#pi^{+}(#pi^{-})", "p(#pi^{+})#pi^{-}", "(p)#pi^{+}#pi^{-}", "d#sigma / dX", "d#sigma /(dX dY)"};
  //bool isCoupled = false;
  //if(strcmp(whichtopology,"coupled")==0) isCoupled = true;

  // Draw the pads and graphs

  char name[256];
  char title[256];
  char xtitle[256];
  char filename[256];

  const int numiter = 5;
  const int numcan = 4 + 5; // 1D + 2D

  TFile *inFile[10];
  TCanvas *can[numcan];
  TPaveLabel *l1[numcan];
  TPaveLabel *l2[numcan];
  TPad *top[numcan];
  TPad *bottom[numcan];
  TLegend *leg[10][numcan];

  int wbinlo[10];
  int wbinhi[10];
  TPaveLabel *tpl_wbin[10][12];
  TGraph *gr_logL[10];
  double logL[10][numiter];
  double chi2[10][numiter];
  double ndf[10][numiter];
  double bestchi2[10];

  //TH1F *h[10][numiter][12][3][4]; // Wbins/iterations/kinvars/type/topologies
  TH1F *h[10][5][4][12][10]; // wbins/type/topology/kinvar/iteration
  TH2F *h2d[10][5][4][12][10]; // wbins/type/topology/kinvar/iteration

  //sprintf(name,"goodFits_chi2cut%2.2f_%d-%d.txt",chi2cut,lomin,himax);
  //ofstream OUT(name);


  for(int f=0;f<10;f++) // wbins
  {
    inFile[f] = NULL;
    gr_logL[f] = NULL;
    for(int i=0;i<5;i++) // type
    {
      for(int j=0;j<4;j++) // topology
      {
        for(int k=0;k<12;k++) //kinvar
        {
          for(int n=0;n<numiter;n++) //iteration
          {
            h[f][i][j][k][n] = NULL;
            h2d[f][i][j][k][n] = NULL;
          }
        }
      }
    }
  }

  cerr << "here" << endl;

  cerr << "About to grab the histos.........." << endl;

  double bestlogL[10];
  for(int i=0;i<10;i++) bestlogL[i] = 1e12;
  for(int i=0;i<10;i++) bestchi2[i] = 1e12;

  double x, logLdum;

  int himax = lomin+10;
  bool foundAtLeastOne = false;
  int numPlots = 0;
  for(int lo=lomin;lo<himax;lo+=10)
  {
    //coupled_covariant_0123_Fit125_1400-1410.root
    int hi = 10 + lo;
    sprintf(filename,"rootFiles/%s_%d-%d.root", fit, lo, hi);
    inFile[numPlots] = new TFile(filename);
    foundAtLeastOne = false;
    if(!inFile[numPlots]->IsZombie() )
    {
      cerr << "File: " << filename << endl;
      // Grab the logL iter info ////////////////////////////////////////////////////////
      //inFile[numPlots]->ls();
      gr_logL[numPlots] = new TGraph((TGraph)inFile[numPlots]->Get("gr_logL"));
      for(int n=0;n<numiter;n++)
      {
        cerr << "a" << endl;
        gr_logL[numPlots]->GetPoint(n,x,logLdum);
        if(logLdum < bestlogL[numPlots]) bestlogL[numPlots] = logLdum;
        printf("bestLogL %f\n", bestlogL[numPlots]);
        logL[numPlots][n] = logLdum;
      }
      wbinlo[numPlots] = lo;
      wbinhi[numPlots] = hi;
      ////////////////////////////////////////////////////////
      for(int i=0;i<5;i++) // type
      {
        for(int j=0;j<4;j++) // topology
        {
          for(int k=0;k<12;k++) //kinvar
          {
            for(int n=0;n<numiter;n++)
            {
              int N=n; // iteration
              int J=j; // iteration
              if(i==0 || i==2 || i==4) N=0;
              if(i==4 || i==5) J=0;
              // Grab the files
              sprintf(name,"h_%d_%d_%d_%d", i,J,k,N);
              TH1F *hdum = (TH1F*)inFile[numPlots]->Get(name);
              //cerr << "Looking for " << name << endl;
              if(hdum)
              {
                //cerr << "Found: " << name << endl;
                h[numPlots][i][J][k][N] = new TH1F(*hdum);
                h[numPlots][i][J][k][N]->SetName(name);
              }
              sprintf(name,"h2d_%d_%d_%d_%d", i,J,k,N);
              TH2F *h2ddum = (TH2F*)inFile[numPlots]->Get(name);
              if(h2ddum)
              {
                //cerr << "Found: " << name << endl;
                h2d[numPlots][i][J][k][N] = new TH2F(*h2ddum);
                h2d[numPlots][i][J][k][N]->SetName(name);
                h2d[numPlots][i][J][k][N]->GetXaxis()->SetNdivisions(6);
                h2d[numPlots][i][J][k][N]->GetYaxis()->SetTitleFont(42);
                h2d[numPlots][i][J][k][N]->GetYaxis()->SetTitleSize(0.075);
                h2d[numPlots][i][J][k][N]->GetYaxis()->SetTitleOffset(1.0);
                h2d[numPlots][i][J][k][N]->GetYaxis()->CenterTitle();
                h2d[numPlots][i][J][k][N]->GetXaxis()->SetLabelSize(0.05);
                h2d[numPlots][i][J][k][N]->GetXaxis()->SetTitleSize(0.075);
                h2d[numPlots][i][J][k][N]->GetXaxis()->SetTitleFont(42);
                h2d[numPlots][i][J][k][N]->GetXaxis()->SetTitleOffset(1.0);
                h2d[numPlots][i][J][k][N]->GetXaxis()->CenterTitle();
                float xwidth = 1.0/h2d[numPlots][i][J][k][N]->GetXaxis()->GetBinWidth(1);
                float ywidth = 1.0/h2d[numPlots][i][J][k][N]->GetYaxis()->GetBinWidth(1);
                float scalefactor = 1.0/(xwidth*ywidth); // To turn to microbarns
                h2d[numPlots][i][J][k][N]->Scale(scalefactor);
                //h2d[numPlots][i][j][k][N]->RebinX(2);
                //h2d[numPlots][i][j][k][N]->RebinY(2);
              }
            }
          }
          foundAtLeastOne=true;
        }
      }
    }
    //cerr << "foundAtLeastOne - before increment: " << foundAtLeastOne << endl;
    if(foundAtLeastOne) numPlots++;
    //cerr << "lo: " << lo << " " << numPlots << endl;
  }

  cerr << "NUMPLOTS: " << numPlots << endl;

  int rows = (int)sqrt(numPlots);
  int cols = numPlots/rows + 1;
  for(int i = 0; i < numcan; i++)
  {
    sprintf(name, "can%d", i);
    sprintf(title, "Data %d", i);
    can[i] = new TCanvas(name, title, 10 + 10 * i, 10 + 10 * i, 600, 600);
    can[i]->SetFillColor(0);
    top[i] = new TPad("top", "The Top", 0.01, 0.99, 1.00, 1.00);
    top[i]->SetFillColor(0);
    top[i]->Draw();
    bottom[i] = new TPad("bottom", "The bottom", 0.01, 0.01, 0.99, 0.99);
    bottom[i]->SetFillColor(0);
    bottom[i]->Draw();
    //cerr << "r/c: " << rows << " " << cols << endl;
    //bottom[i]->Divide(rows, cols);
    bottom[i]->Divide(1,1);


    //top[i]->cd();
    //sprintf(title, "Fit: %s %s ", fit, topology_latex[i/4].c_str());
    //l1[i] = new TPaveLabel(0.01, 0.01, 0.99, 0.99, title);
    //l1[i]->SetFillStyle(1);
    //l1[i]->SetFillColor(1);
    //l1[i]->SetTextColor(0);
    //l1[i]->Draw();
    //sprintf(title, "%d < W < %d MeV/c^{2}", lo, hi);
    //l2[i] = new TPaveLabel(0.3, 0.01, 0.6, 0.99, title);
    //l2[i]->Draw();
  }

  int fillcolor[3] = {42, 0, 0};
  int linewidth[3] = {1, 2, 2};
  int linecolor[3] = {1, 4, 2};

  Float_t axisScale = 1.3;

  cerr << "And here......" << endl;
  ///////////////////////////////////////////////////////////////////////////////
  for (int f=0;f<numPlots;f++)
  {
    for(int i=0;i<5;i++) // type
    {
      for(int j=0;j<4;j++) // topology
      {
        for(int k=0;k<12;k++) //kinvar
        {
          for(int n=0;n<numiter;n++)
          {
            //i = k;
            //if(k==7) i = 9;
            //else if(k==8)  i = 7;
            //else if(k==9)  i = 10;
            //else if(k==10) i = 8;
            //else if(k==11) i = 11;

            //cerr << n << " " << j << " " << k << endl;
            //cerr << k << " " << i << " is cd'ing to: " << numPlots*(i%3) + 2 << " of " << i/3k << endl;
            //bottom[k]->cd(n+1);
            //int whichcan = 0;
            //int whichpad = 0;
            //if (i<3) 
            //{
              //whichcan = k/3 + j*4;
              //whichpad = f + numPlots*(k%3) + numPlots+1;
            //}
            //else 
            //{
              //whichcan = k/3 + 16;
              //whichpad = f + numPlots*(k%3) + numPlots+1;
            //}

            //cerr << "whichcan/whichpad: " << whichcan << " " << whichpad << endl;
            //cerr << f << " " << i << " " << j << " " << k << " " << n << endl;
            if( h[f][i][j][k][n] )
            {
              //cerr << "name: " << h[f][i][j][k][n]->GetName() << endl;
              //cerr <<" Found this "  << h[n][j][k][topo]->GetName() << endl;
              //cerr << "max: " << 1.3*h[n][j][k][topo]->GetMaximum() << endl;
              //else          h[n][j][k][topo]->SetMaximum(1.5*h[n][j][k][topo]->GetYaxis()->GetXmax());
              h[f][i][j][k][n]->SetMinimum(0);
              h[f][i][j][k][n]->SetTitle();
              h[f][i][j][k][n]->GetXaxis()->SetNdivisions(6);
              h[f][i][j][k][n]->GetYaxis()->SetTitleFont(42);
              h[f][i][j][k][n]->GetYaxis()->SetTitleSize(0.075);
              h[f][i][j][k][n]->GetYaxis()->SetTitleOffset(1.1);
              h[f][i][j][k][n]->GetYaxis()->CenterTitle();
              h[f][i][j][k][n]->GetXaxis()->SetLabelSize(0.05);
              h[f][i][j][k][n]->GetXaxis()->SetTitleSize(0.075);
              h[f][i][j][k][n]->GetXaxis()->SetTitleFont(42);
              h[f][i][j][k][n]->GetXaxis()->SetTitleOffset(1.1);
              h[f][i][j][k][n]->GetXaxis()->CenterTitle();
              if(k==0) sprintf(xtitle,"M(p #pi^{+}) GeV/c^{2}");
              else if(k==1) sprintf(xtitle,"M(p #pi^{-}) GeV/c^{2}");
              else if(k==2) sprintf(xtitle,"M(#pi^{+} #pi^{-}) GeV/c^{2}");
              else if(k==3) sprintf(xtitle,"p cos(#theta) CM");
              else if(k==4) sprintf(xtitle,"#pi^{+} cos(#theta) CM");
              else if(k==5) sprintf(xtitle,"#pi^{-} cos(#theta) CM");
              else if(k==6) sprintf(xtitle,"#pi^{+} cos(#theta) #Delta^{++}-helicity");
              else if(k==7) sprintf(xtitle,"#pi^{+} #phi #Delta^{++}-helicity");
              else if(k==8) sprintf(xtitle,"#pi^{-} cos(#theta) #Delta^{0}-helicity");
              else if(k==9) sprintf(xtitle,"#pi^{-} #phi #Delta^{0}-helicity");
              else if(k==10) sprintf(xtitle,"#pi^{+} cos(#theta) #rho-helicity");
              else if(k==11) sprintf(xtitle,"#pi^{+} #phi #rho-helicity");
              h[f][i][j][k][n]->GetXaxis()->SetTitle(xtitle);

              h[f][i][j][k][n]->SetLineWidth(2);
              if(i==0) 
              {
                h[f][i][j][k][n]->SetFillColor(5);
              }
              else if(i==2) 
              {
                h[f][i][j][k][n]->SetFillColor(0);
                h[f][i][j][k][n]->SetLineColor(2);
                h[f][i][j][k][n]->SetLineWidth(3);
                int nbins = h[f][0][j][k][n]->GetNbinsX();
                float scaleby =  (float)h[f][0][j][k][n]->Integral(1,nbins) / (float)h[f][i][j][k][n]->Integral(1,nbins);
                h[f][i][j][k][n]->Scale( scaleby );
              }
              else if(i==1 || i==3)
              {
                //h[f][i][j][k][n]->SetMarkerColor(8*n+33);
                //h[f][i][j][k][n]->SetLineColor(8*n+33);
                h[f][i][j][k][n]->SetMarkerColor(4);
                h[f][i][j][k][n]->SetLineColor(4);
                h[f][i][j][k][n]->SetLineWidth(3);
              }

              // Scale the raw to get cross sections
              if(i==3)
              {
                float scalefactor = 1.0/h[f][i][j][k][n]->GetBinWidth(1);
                h[f][i][j][k][n]->Scale(scalefactor);
                h[f][i][j][k][n]->SetMinimum(0.0);
              }

              if(k>=3) 
              {
                //h[f][i][j][k][n]->GetXaxis()->SetLimits(-1.5, 1.5);
              }

              //if(i==0)       h[f][i][j][k][n]->Draw("h");
              //else if(i==2)  h[f][i][j][k][n]->Draw("same");

              if(n==0)
              {
                sprintf(name,"%d-%d",wbinlo[f],wbinhi[f]);
                tpl_wbin[f][k] = new TPaveLabel(0.5,0.8,0.99,0.99,name,"NDC");
                tpl_wbin[f][k]->SetFillStyle(1);
                tpl_wbin[f][k]->SetFillColor(0);
                tpl_wbin[f][k]->SetTextColor(2);
                //tpl_wbin[f][k]->Draw();
              }

              h[f][i][j][k][n]->SetMaximum(1.3*h[f][i][j][k][n]->GetMaximum());
              if(i==3)
              {
                if(k<=2)             h[f][i][j][k][n]->SetMaximum(1100);
                else if(k>2 && k<=5) h[f][i][j][k][n]->SetMaximum(100);
                else                 h[f][i][j][k][n]->SetMaximum(100);
              }
            }

            // Break for the ones which only have one or two plots
            if(i==0 || i==2 || i==4) n+=12; 
            if(i==4 || i==5) j+=5;
          }
        }
      }
      //cerr <<"i:" << i << endl;
    }
  }

            bottom[0]->cd(1);
            gPad->SetBottomMargin(0.18);
            gPad->SetLeftMargin(0.18);
            h[0][0][itopology][ikinvar][0]->Draw("h"); // file/type/topology/kinvar/iteration
            tpl_wbin[0][ikinvar]->Draw();

            bottom[1]->cd(1);
            gPad->SetBottomMargin(0.18);
            gPad->SetLeftMargin(0.18);
            h[0][0][itopology][ikinvar][0]->Draw("h"); // file/type/topology/kinvar/iteration
            h[0][2][itopology][ikinvar][0]->Draw("same"); // file/type/topology/kinvar/iteration

            
              bottom[2]->cd(1);
              gPad->SetBottomMargin(0.18);
              gPad->SetLeftMargin(0.18);
              h[0][0][itopology][ikinvar][0]->Draw("h"); // file/type/topology/kinvar/iteration
              h[0][2][itopology][ikinvar][0]->Draw("same"); // file/type/topology/kinvar/iteration
              h[0][1][itopology][ikinvar][iiteration]->Draw("same"); // file/type/topology/kinvar/iteration

            
              bottom[3]->cd(1);
              gPad->SetBottomMargin(0.18);
              gPad->SetLeftMargin(0.18);
              h[0][3][0][ikinvar][iiteration]->Draw("h"); // file/type/topology/kinvar/iteration
              h[0][3][0][ikinvar][iiteration]->Draw("esame"); // file/type/topology/kinvar/iteration
            

              bottom[4]->cd(1);
              gPad->SetBottomMargin(0.18);
              gPad->SetLeftMargin(0.18);
              h2d[0][0][itopology][ikinvar][0]->Draw("colz"); // file/type/topology/kinvar/iteration

              bottom[5]->cd(1);
              gPad->SetBottomMargin(0.18);
              gPad->SetLeftMargin(0.18);
              h2d[0][1][itopology][ikinvar][0]->Draw("colz"); // file/type/topology/kinvar/iteration

              // Cross section
              bottom[6]->cd(1);
              gPad->SetBottomMargin(0.18);
              gPad->SetLeftMargin(0.18);
              gPad->SetRightMargin(0.15);
              h2d[0][2][itopology][ikinvar][0]->Draw("colz"); // file/type/topology/kinvar/iteration

              bottom[7]->cd(1);
              gPad->SetBottomMargin(0.18);
              gPad->SetLeftMargin(0.18);
              h2d[0][3][0][ikinvar][0]->Draw("colz"); // file/type/topology/kinvar/iteration

              bottom[8]->cd(1);
              gPad->SetBottomMargin(0.18);
              gPad->SetLeftMargin(0.18);
              h2d[0][4][0][ikinvar][0]->Draw("colz"); // file/type/topology/kinvar/iteration


                //h2d[numPlots][i][j][k][N]->RebinX(2);
                //h2d[numPlots][i][j][k][N]->RebinY(2);
                //int nbins = h[numPlots][i][j][k][n]->GetNbinsX();
                //float scaleby =  (float)h[f][0][j][k][n]->Integral(1,nbins) / (float)h[f][i][j][k][n]->Integral(1,nbins);
                //h[f][i][j][k][n]->Scale( scaleby );

  ///////////////////////////////////////////////////////////////////////////////



     for(int i=0;i<numcan;i++)
     {
       sprintf(name,"plots/dsigma_%s_%d_%d_%d_%d_%d.eps",tag, lomin, itopology, ikinvar, iiteration, i);
       can[i]->SaveAs(name);
     }

}
