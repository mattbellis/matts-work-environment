#include<iostream>

using namespace std;

int filename(char* file);

int filename(char* file)
{
  if (file == "test")
  {
    cerr << "filename not found: " << endl;
  }
  else
  {
    cerr << "filename: " << file << endl;
  }

  return 0;

}

int main()
{
  filename("test");
  filename("matt");
  return 0;
}

