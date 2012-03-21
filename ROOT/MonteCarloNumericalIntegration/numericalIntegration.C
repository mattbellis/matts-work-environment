void numericalIntegration(int npoints, float lo=0, float hi=5)
{

  TRandom *rnd = new TRandom();

  float tot = 0;
  for(int i=0;i<npoints;i++)
  {
    float dum = hi*rnd->Rndm();
    tot += dum;
  }
  cerr << (hi-lo)*tot/npoints << endl;

}
