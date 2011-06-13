#ifndef thisClassH_INCLUDED
#define thisClassH_INCLUDED

#include<vector>

using namespace std;

class fly
{
  private:
    int q;

  public:
    fly();
    fly(int R) {r = R;}; 

    float r;

    void set(int R) {r = R;}; 
};

fly::fly()
{
  this->q = 2;
}

class bee
{
  public:
    bee();

    int x;
    int y;
    fly m;
};

bee::bee()
{
  this->x = 1;
  this->y = 0;
  this->m.set(1);

};

#endif
