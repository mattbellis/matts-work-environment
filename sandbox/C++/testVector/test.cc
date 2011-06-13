#include<iostream>
#include<vector>
#include<fstream>

using namespace std;

int main()
{
  vector<ifstream*> vdum;
  //ifstream dum("test.txt");
  ifstream *dum = new ifstream();
  vdum.push_back(dum);

}
