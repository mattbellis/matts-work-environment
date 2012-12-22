#include<fstream>
#include<iostream>
#include<string>

using namespace std;

int main()
{

  string entry;
  string::size_type pos;
  int line = 0;
  
  ifstream IN("data.dat");

  while(getline(IN, entry))
  {
    line++;
    pos = entry.find("	",0);
    if(pos!=string::npos)
    {
      cerr << "Found a tab at line " << line << " and position " << pos << endl;
    }
  }

  return 0;
}
