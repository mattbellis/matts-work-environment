#include<iostream>
#include<complex>
#include<string>
using namespace std;

int main()
{
  complex<double> a, b;
  string an, bn;
  cin >> an >> a >> bn >> b;
  cerr<< an << "\t" << bn << endl;
  cerr<< a << "\t\t" << b << endl;
  cerr << abs(a) << "\t\t\t" << abs(b) << endl;
  cerr << arg(a) << "\t\t\t" << arg(b) << endl;
  cerr << arg(a*b) << "\t\t\t" << abs(a*b) << endl;
  return 0;
}
