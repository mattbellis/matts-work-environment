int plotBranchingRatios(char *filename, int option=0)
{

  // Option = 0  ---  N*'s
  // Option = 1  ---  Delta's

  // Declare some local variables.
  const int numdecays = 7; // Hard code the number of possible decays
  const int numplots = 15; // Hard code the number of plots
  // Assume max of 32 baryons 
  TBox *box[numplots][32][numdecays]; // This is the actual boxes we will color.
  TPaveText *tres[numplots][32]; // Text for figures
  TPaveText *tspin[numplots][32];
  TPaveText *tstars[numplots][32];
  float br[32][numdecays]; // Branching ratios
  string resonance[32], spin[32], stars[32]; // Resonance info
  float norm[32]; // Use this for normalizing the decays

  // Create a canvas
  char name[256];
  TCanvas *c[numplots];
  TLegend *leg[numplots];
  for(int i=0;i<numplots;i++)
  {
    sprintf(name,"c%d",i);
    // Constructor is (name, title, upper-x, upper-y, width, height (all in pixels)
    //c[i] =new TCanvas(name,"",10+10*i,10+10*i,500,600);
    c[i] =new TCanvas(name,"",10+10*i,10+10*i,700,900);
    c[i]->SetFillColor(0);
    c[i]->Divide(1,1);
  }

  // Set up the text and colors we will use in the legend.
  TH1F *h[numdecays]; // This will be a placeholder when we fill the 
  // legend.

  string decays[numdecays];

  decays[0] = "N#pi";
  decays[1] = "N#eta";
  decays[2] = "N#omega";
  decays[3] = "#Delta #pi";
  decays[4] = "p #rho";
  decays[5] = "#Lambda K";
  decays[6] = "#Sigma K";

  int colors[numdecays];
  colors[0] = TColor().GetColor("#90000"); // Red 4, #900000       N pi
  colors[1] = TColor().GetColor("#00d000"); // Green 2, #00d000    N eta
  colors[2] = TColor().GetColor("#0000ff"); // Blue, #0000ff       N omega
  colors[3] = TColor().GetColor("#00ffff"); // Cyan, #00ffff       Delta pi
  //colors[4] = TColor().GetColor("#d0c000"); // Tan, #d0c000      N rho
  colors[4] = TColor().GetColor("#ff6600"); // orange, #ff6600     N rho
  colors[5] = TColor().GetColor("#ffff00"); // Yellow, #ffff00     Lambda K
  colors[6] = TColor().GetColor("#d000d0"); // Mag 2, #d000d0      Sigma K
  int gray = TColor().GetColor("#bbbbbb"); // Gray

  // Open the file for reading in
  ifstream IN(filename);

  int numres = 0;
  while(IN >> resonance[numres])
  {
    IN >> spin[numres] >> stars[numres];
    // Look for a string we designate as 0 stars to act as placeholder
    if(stars[numres] == "---") stars[numres]=" ";

    cerr << resonance[numres] << " " << spin[numres] << " " << stars[numres] << endl;

    norm[numres]  = 0.0; 
    for(int i=0;i<numdecays;i++)
    {
      IN >> br[numres][i];
      norm[numres] += pow(br[numres][i],2); // Construct the normalization
    }

    numres++;
  }

  // This will be used for formatting the figure.
  // All values are the fraction of the canvas size, 
  // so if we set the bar length to be 0.20, it will be 
  // 20% the width of the canvas.
  float lox, loy, hix, hiy;
  float length = 0.30; // Bar length
  float height = 0.04; // Bar height
  float vspace = 0.01; // Vertical space between entries
  float hspace = 0.02; // Horizontal space between text entries/bar
  float textwidth = 0.25; // Space we allow for the intial text/stars

  float loxstart = 0.01; // Low x starting point.

  TH1F *hdum = new TH1F();
  for(int k=0;k<numplots;k++)
  {
    c[k]->cd(1);
    for(int j=0;j<numres;j++)
    {
      lox = loxstart;
      hiy = 0.99 - j*(height+vspace); // We start at the top and work our way down

      // Constructor for Text is (lox, loy, hix, hiy, Options)
      tres[k][j] = new TPaveText(lox, hiy-height, lox+0.6*textwidth, hiy,"NDC");
      tres[k][j]->AddText(resonance[j].c_str());
      tres[k][j]->SetBorderSize(0);
      tres[k][j]->SetFillStyle(0);
      tres[k][j]->Draw();

      lox = lox+0.6*textwidth + hspace;
      tstars[k][j] = new TPaveText(lox, hiy-height, lox+0.4*textwidth, hiy,"NDC");
      tstars[k][j]->AddText(stars[j].c_str());
      tstars[k][j]->SetBorderSize(0);
      tstars[k][j]->SetFillStyle(0);
      tstars[k][j]->Draw();

      // Construct the starting point for our bar
      lox = loxstart + textwidth + 2*hspace;
      int brcount = 0;
      int max = numdecays;
      if (k<=numdecays) max = k;
      for(int i=0;i<max;i++)
      {
        loy = hiy - height;
        // Add to the bar if there is a non-zero branching ratio
        if(br[j][i] != 0.0)
        {
          hix = lox + length*pow(br[j][i],2)/norm[j];
          box[k][j][i] = new TBox(lox, loy, hix, hiy);
          if(k>=numdecays+1 && i==(k-(numdecays+1))) box[k][j][i]->SetFillColor(colors[i]);
          else if(k>=numdecays+1)                    box[k][j][i]->SetFillColor(gray);
          else                                       box[k][j][i]->SetFillColor(colors[i]);
          box[k][j][i]->Draw();
          lox = hix;
        }
      }
    }

    // Construct the legend
    // Constructor is (lox, loy, hix, hiy)
    leg[k] = new TLegend(loxstart + length + textwidth + 6*hspace, 0.2, 0.99, 0.99);
    //if(option==0) leg[k] = new TLegend(loxstart + length + textwidth + 6*hspace, 0.2, 0.99, 0.99);
    //else          leg[k] = new TLegend(loxstart + length + textwidth + 6*hspace, 0.70, 0.99, 0.99);
    for(int i=0;i<numdecays;i++)
    {
      // Create the dummy histograms just as placeholders for the legend
      h[i] = new TH1F();
      if(k>=numdecays+1 && i==(k-(numdecays+1))) h[i]->SetFillColor(colors[i]);
      else if(k>=numdecays+1)                    h[i]->SetFillColor(gray);
      else                                       h[i]->SetFillColor(colors[i]);
      if(option==0 && i<k) 
      {
        leg[k]->AddEntry(h[i],(decays[i]).c_str(),"f");
      }
      else if(i>=k) 
      {
        leg[k]->AddEntry(hdum," ","");
      }
      else if(option==1 && !(i==0 || i==3 || i==4 || i==6)) 
      {
        //leg[k]->AddEntry(h[i],(decays[i]).c_str(),"f");
        leg[k]->AddEntry(hdum," ","");
      }
      else if(option==1) 
      {
        leg[k]->AddEntry(h[i],(decays[i]).c_str(),"f");
      }
    }
    leg[k]->SetFillColor(0);
    leg[k]->SetBorderSize(0);
    leg[k]->Draw();

    sprintf(name,"plots/capandrobBRs%d_%d.eps",option,k);
    c[k]->SaveAs(name);
  }

}

