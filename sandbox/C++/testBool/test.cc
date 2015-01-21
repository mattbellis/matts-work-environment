#include<iostream>

using namespace std;

int main()
{

  bool test1 = 1<2 && 1<3;
  bool test2 = 2<1 && 1<3;

  cerr << test1 << " " << test2 << endl;

  test1 = test1 && test2;

  cerr << test1 << endl;

  return 0;

}
