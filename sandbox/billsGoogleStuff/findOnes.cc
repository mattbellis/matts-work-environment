#include<iostream>
#include<cmath>

using namespace std;

int main(int argc, char **argv)
{
  int number=2;
  int dec = 1;
  int decplaces = 1;
  int i;
  int dum;
  int numOnes = 0;
  int max;

  if(argc==1)
  {
    while(number != numOnes)
    {
      number++;
      if(number%1000==0) cerr << "Testing number:\t" << number << "\r";
      for(i=0;number/dec>0;i++) dec = (int)pow(10.0,i);
      decplaces = i-1;
      for(i=0;i<decplaces;i++)
      {
        dec = (int)pow(10.0,i);
        dum = (number/dec)%(10);
        if(dum==1) numOnes++;
      }
    }
  }

  else
  {
    max=atoi(argv[1]);
    for(number=1;number<=max;number++)
    {
      if(number%1000==0) cerr << "Testing number:\t" << number << "\r";
      for(i=0;number/dec>0;i++) dec = (int)pow(10.0,i);
      decplaces = i-1;
      for(i=0;i<decplaces;i++)
      {
        dec = (int)pow(10.0,i);
        dum = (number/dec)%(10);
        if(dum==1) numOnes++;
      }
    }
    number--;
  }

  cerr << "Number of ones in " << number << ":\t" << numOnes << endl;
  return 0;
}
