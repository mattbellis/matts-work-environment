int myrnd(int max=10000000)
{

TH1F *h = new TH1F("h", "h", 100, 0, 100);
TRandom3 *rnd = new TRandom3();

for (int i=0;i<max;i++)
{
    float dum = rnd->Rndm();
    h->Fill(dum);
}


}
