#include<iostream>
#include<iomanip>

using namespace std;

int main()
{

//  cerr.precision(3);
  cerr.setf(ios::hex | ios::showbase | ios::uppercase);
  cerr << 4.3 << " " << 5 << " " << 8.9 << endl;
  double pi = 3.141592654;
  // Default display: left justified, precision 6.
  cout << pi << endl;
  // Change precision 4, field with 12, fill char #
  cout.precision(4);
  cout.width(12);
//  cout.fill('#');
  cout << pi << endl;
  // Change precision to 10
  cout.precision(10);
//  cout.fill('#');
  cout << pi << endl;
  int num = 37;
  cout << "hex: " << hex << num << endl;
  cout << "oct: " << oct << num << endl;
  cout << "dec: " << dec << num << endl;
  cout << "When padded, the value " << setw(8) << num 
  << " takes up some space." << num << " " << num << endl;
  cout << "And may be filled " << setw(8) << setfill('*') 
  << num << endl;
  cout << "Pi: " << setprecision(10) << pi << endl;
  double x = 2.0/3.0;
  cout << setiosflags(ios::fixed) << setprecision(4) << x << endl;
  int m = 12;
  int d = 4;
  int y = 4;
  cout << setfill('0');
  cout << setw(2) << m << '/'
  << setw(2) << d << '/'
  << setw(4) << y << endl;
  double xx = 800000.0/81.0;
  cout << setiosflags(ios::fixed) << setprecision(2) << xx << endl;
  cout << 324.2342349823 << endl;
  return 0;
}
