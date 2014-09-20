void numericalIntegration_sqrtX(int npoints=10000, float max=5)
{

  TRandom *rnd = new TRandom();

  float tot = 0;
  float x,y;
  int nused = 0;

  TH1F *h = new TH1F("h","circle" ,100,0,max);

  for(int i=0;i<npoints;i++)
  {
    if(i%10000==0) cerr << i << "\r";

    float x = max*rnd->Rndm();
    tot+= sqrt(x);
    h->Fill(sqrt(x));

  }
  cerr << npoints << endl;
  float answer = max*tot/(float)npoints;
  cerr << "answer: " << answer << endl;
  float solution = (2.0/3.0)*sqrt(pow(max,3));
  cerr << "solution: " << solution << endl;
  cerr << answer - solution << endl;

  h->Draw("");

}
