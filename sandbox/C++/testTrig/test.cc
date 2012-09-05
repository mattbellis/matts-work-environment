#include<cmath>
#include<iostream>

using namespace std;

int main()
{
  float angle;

  while(cin >> angle)
  {
    cerr << "angle: " << angle << endl;
    cerr << "atan(tan): " << atan(tan(angle)) << endl;
    cerr << "acos(cos): " << acos(cos(angle)) << endl;
    cerr << "asin(sin): " << asin(sin(angle)) << endl;
  }

  return 0;
}
