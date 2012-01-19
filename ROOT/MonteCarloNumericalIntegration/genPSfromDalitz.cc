void numericalIntegration_Dalitz(int npoints, float M=1.8)
{

  gROOT->Reset();

  TRandom *rnd = new TRandom();

  float PI = TMath::Pi();

  float pM = 0.938272;

  float dGamma = 0;
  float m122,m232;
  int nused = 0;
  float photonspinfactor = 1.0/4.0;
  float commonfactor = 1.0;
  float factor = 1.0;

  float m3 = 0.13957;
  float m1 = 0.938272;
  float m2 = 0.13957;

  float m122lo = (m1+m2)*(m1+m2)-0.01;
  float m232lo = (m2+m3)*(m2+m3)-0.01;
  float m122hi = (M-m3)*(M-m3)+0.01;
  float m232hi = (M-m1)*(M-m1)+0.01;

  TH2F *h = new TH2F("h","circle",100, m122lo, m122hi, 100, m232lo, m232hi);

  float E2;
  float E3;

  for(int i=0;i<npoints;i++)
  {

    m122 = (m122hi-m122lo)*rnd->Rndm() + m122lo;
    m232 = (m232hi-m232lo)*rnd->Rndm() + m232lo;

    E2 = (m122 - m1*m1 + m2*m2)/(2.0*sqrt(m122));
    E3 = (M*M  - m122  - m3*m3)/(2.0*sqrt(m122));

    float m232max =  (E2+E3)*(E2+E3) - pow(sqrt(E2*E2-m2*m2) - sqrt(E3*E3-m3*m3),2);
    float m232min =  (E2+E3)*(E2+E3) - pow(sqrt(E2*E2-m2*m2) + sqrt(E3*E3-m3*m3),2);

    if(m232>=m232min && m232<=m232max)
    {
      nused++;
      h->Fill(m122, m232);
    }
  }

  float m122range = m122hi-m122lo;
  float m232range = m232hi-m232lo;

  dGamma =  m122range*m232range*(float)nused/(float)npoints;

  commonfactor =M/(M*M-pM*pM);
  //factor =(1.0/(8.0*PI*PI*PI))*(1.0/(32.0*M*M*M));
  factor =(1.0/(pow(2.0*PI,3)))*(1.0/(16.0*M*M));

  cerr << npoints << " " << nused << " " << (float(nused))/((float)npoints) << endl;
  cerr << "ranges: " << m122range << " " << m232range << endl;
  cerr << "commonfactor: " << commonfactor << endl;
  cerr << "photonspinfactor: " << photonspinfactor << endl;
  cerr << "factor: " << factor << endl; 
  cerr << "dGammaBefore: " << dGamma << endl; 

  dGamma *= commonfactor*photonspinfactor*factor;

  cerr << M << " " << dGamma << " " << endl;

  h->Draw("colz");

}
