#include<iostream>
#include<string>

using namespace std;

int main()
{

  const string *x0 = new string("This is not changing");
  cerr << x0->size() << endl;
  const string *x1 = new string("This is short");
  cerr << x1->size() << endl;

  cerr << "First stuff: " << endl;
  string y = string(*x0);
  //string *z0 = &y;
  string *z0 = new string(*x0);
  z0->erase(18);
  cerr << "x0: " << x0->size() << endl;
  cerr << "y: " << y.size() << endl;
  cerr << "z0: " << z0->size() << endl;

  cerr << "Second stuff: " << endl;
  y = string(*x1);
  string *z1 = &y;
  z1->erase(10);
  cerr << "x0: " << x0->size() << endl;
  cerr << "x1: " << x1->size() << endl;
  cerr << "y: " << y.size() << endl;
  cerr << "x1: " << x1->size() << endl;
  cerr << "z0: " << z0->size() << endl;
  cerr << "z1: " << z1->size() << endl;

  return 0;

}
