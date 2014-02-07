#include<iostream>
#include<fstream>
#include<complex>
#include<string>
using namespace std;

#define PI 3.1415926535897932384626433832795

int main(int argc, char** argv)
{
  complex<double> a, b;

  a = std::polar(4.0 , PI );

  ifstream IN("numbers.txt");
  //IN >> std::polar(b) >> endl;
  cerr << "From file: " << b << endl;
  cerr << a << endl;
  for(int i=0;i<atoi(argv[1]);i++)
  {
  //cerr << 6.0*a*std::polar(8.0,0.0) << endl;
    if(argv[2]=="n") b = 6.0*a*std::polar(8.0,1.0);
    else if(argv[2]=="y") b = 6.0*a*complex<double>(8.0,1.0);
    else            b = complex<double>(6.0,0.0)*a*std::polar(8.0,1.0);
  }
  return 0;
}
