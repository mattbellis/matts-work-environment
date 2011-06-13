RooDataHist HD("HD","HD",RooArgSet(LCMass,DMass),*Data);

// ***** Fit PDF ***** //
// ***** LC Signal ***** //
RooRealVar meanLC("meanLC","LC mean",2.2857,2.2837,2.2877,"GeV/c^{2}");
RooRealVar sigmaLC("sigmaLC","LC sigma",0.004,0.003,0.006,"GeV/c^{2}");

RooGaussian gaussLC("gaussLC","gaussian PDF",LCMass,meanLC,sigmaLC);

RooGenericPdf multPolyGausLC("multPolyGausLC","multPolyGausLC","1",RooArgList());

RooProdPdf prodGausLC("prodGausLC","gaussLC*multPolyGausLC",RooArgList(gaussLC,multPolyGausLC));




// ***** LC Signal Normalization Factors ***** //
RooRealVar nLC("nLC","nLC",1000.0,100.0,5000.0);



// ***** Background ***** //
RooRealVar slopeLC("slopeLC","slopeLC",0.0,-8.0,4.0,"");
RooRealVar slopeDM("slopeDM","slopeDM",5.0,-5.0,25.0,"");
RooGenericPdf bg("bg","bg","(1+slopeLC*(LCMass-2.286)+slopeDM*(DMass-1.100))",RooArgList(slopeLC,LCMass,slopeDM,DMass));



// ***** Xi_cc Siganl ***** //
RooGaussian gLC("gLC","gLC PDF",LCMass,meanLC,sigmaLC); // the same mean and sigma as the gaussLC above
RooRealVar mid("mid","mid",(3.460-2.286),(3.460-2.286)-0.005,(3.460-2.286)+0.005,"GeV/c^{2}");



// DCDCB
RooRealVar width("width","width",0.003,0.0025,0.0035,"GeV/c^{2}");
RooGaussian gDM("gDM","gDM PDF",DMass,mid,width);



// ***** Signal Multiplied Together ***** //
RooProdPdf sg("sg","gLC*gDM",RooArgList(gLC,gDM));



// ***** Self-Cross-Feed Background ***** //
RooFormulaVar scfbVar("scfbVar","scfbVar","DMass+LCMass",RooArgList(DMass,LCMass));
RooFormulaVar scfbMean("scfbMean","scfbMean","mid+meanLC",RooArgList(mid,meanLC));
RooFormulaVar widthB("widthB","widthB","width*3.79",RooArgList(width));
RooGaussian gaussSCFB("gaussSCFB","gaussSCFB",scfbVar,scfbMean,widthB);



// ***** Add signal PDFs Together ***** //
RooRealVar frac("frac","frac",0.24,0.19,0.29);
RooAddPdf allSignal("allSignal","allSignal",RooArgList(gaussSCFB,sg),RooArgList(frac));



// ***** Add all PDFs Together ***** //
RooRealVar nBG("nBG","nBG",5000.0,500.0,10000.0);
RooRealVar F("F","F",9.767); //  232*0.0421=9.767
RooRealVar f("f","f",9.767,0.0,100.0);
RooRealVar ErF("ErF","ErF",0.54); // sqrt(232*232*0.0023*0.0023+0.0421*0.0421*2*2)=0.54
RooRealVar Ssg("Ssg","Ssg",0.0,-200.0,200.0);
RooFormulaVar Ans("Ans","f*Ssg",RooArgList(f,Ssg));
RooAddPdf total("total","total",RooArgList(bg,prodGausLC,allSignal),RooArgList(nBG,nLC,Ans));



// ***** Fit ***** //
RooNLLVar NLogL("NLogL","NLogL",total,HD,kTRUE);
RooFormulaVar nll("nll","NLogL+(F-f)*(F-f)/(2.0*ErF*ErF)",RooArgList(NLogL,F,f,F,f,ErF,ErF));

double minNLLF[21];  double minNLLB[21];
double space=0.3;
double searchS[21];
for(int i=0; i<21; i++){ searchS[i] = 0.0+i*space; }

RooMinuit m(nll); m.setVerbose(kFALSE); RooFitResult *JZ[21];
RooFitResult *DR[21];
FILE *fh = fopen("dcdcbFix1Limit.txt","w");
for(int i=0; i<21; i++){
  Ssg.setVal(searchS[i]); Ssg.setConstant();
  m.synchronize(kFALSE);
  m.migrad(); m.hesse(); JZ[i] = m.save();

  int sat=JZ[i]->covQual();
  printf("\nStatus is %d\n",sat);
  if(sat!=3){m.migrad(); m.hesse(); JZ[i] = m.save();
    printf("\nStatus is %d\n",sat);}
  if(sat!=3){m.migrad(); m.hesse(); JZ[i] = m.save();
    printf("\nStatus is %d\n",sat);}
  if(sat!=3){m.migrad(); m.hesse(); JZ[i] = m.save();
    printf("\nStatus is %d\n",sat);}
  if(sat!=3){printf("\n***** Fit FUCKED *****\n");}
      minNLLF[i] = JZ[i]->minNll();

  fprintf(fh,"Ssg=%f  NLL=%f\n",searchS[i],minNLLF[i]);  fflush(fh);
}
double bottomF = TMath::MinElement(21,minNLLF);
for(int i=0; i<21; i++){ minNLLF[i] = minNLLF[i]-bottomF; }

for(int i=20; i>-1; i--){
  Ssg.setVal(searchS[i]); Ssg.setConstant();
  m.synchronize(kFALSE);
  m.migrad(); m.hesse(); DR[i] = m.save();

  int sat=DR[i]->covQual();
  printf("\nStatus is %d\n",sat);
  if(sat!=3){m.migrad(); m.hesse(); DR[i] = m.save();
    printf("\nStatus is %d\n",sat);}
  if(sat!=3){m.migrad(); m.hesse(); DR[i] = m.save();
    printf("\nStatus is %d\n",sat);}
  if(sat!=3){m.migrad(); m.hesse(); DR[i] = m.save();
    printf("\nStatus is %d\n",sat);}
  if(sat!=3){printf("\n***** Fit FUCKED *****\n");}
        minNLLB[i] = DR[i]->minNll();

  fprintf(fh,"Ssg=%f  NLL=%f\n",searchS[i],minNLLB[i]);  fflush(fh);
}
double bottomB = TMath::MinElement(21,minNLLB);
for(int i=0; i<21; i++){ minNLLB[i] = minNLLB[i]-bottomB; }
