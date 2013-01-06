#include<iostream>
#include<fstream>
#include<complex>
#include<string>
using namespace std;

#define PI 3.1415926535897932384626433832795

int main(int argc, char** argv)
{
  complex<double> a, b, c, d;

  a = std::polar(0.8978 , 0.14 );
  b = std::polar(0.324 , 0.0054 );

  for(int i=0;i<atoi(argv[1]);i++)
  {
    d = complex<double>(0,0);
    for(int j=0;j<8;j++)
    {
      c = complex<double>(0,0);
      for(int k=0;k<40;k++)
      {
        c += b*b*b*a*a*a;
      }
      d += c*conj(c);
    }
  }
  return 0;
}
