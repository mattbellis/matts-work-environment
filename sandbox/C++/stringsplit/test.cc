#include<iostream>
#include<string>

using namespace std;

int main()
{

  string x = "2,10";

  string sx0 = x.substr(0, x.find(","));
  string sx1 = x.substr(x.find_first_of(",")+1);

  cerr << x << " " << sx0 << " " << sx1 << endl;

  return 0;
}
