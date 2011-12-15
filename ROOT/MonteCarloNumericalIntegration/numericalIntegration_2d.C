void numericalIntegration_2d(int npoints, float radius=5)
{

  TRandom *rnd = new TRandom();

  float tot = 0;
  float x,y;
  int nused = 0;

  TH2F *h = new TH2F("h","circle",100,-radius-1, radius+1, 100, -radius-1, radius+1);

  float r2 = radius*radius;

  for(int i=0;i<npoints;i++)
  {

    float x = radius*rnd->Rndm();
    float y = radius*rnd->Rndm();
    if(x*x+y*y<r2)
    {
      nused++;
    }
  }
  cerr << npoints << " " << nused << " " << r2*nused/(float)npoints << endl;
  cerr << (radius)*tot/(float)nused << endl;
  cerr << 3.14159*r2/4.0 << endl;

  h->Draw("colz");

}
