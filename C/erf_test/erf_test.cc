#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<math.h>

using namespace std;

int main(int argc,char** argv)
{
    for(int i=0;i<11;i++)
    {
        double x = 0.1*i;
        printf("%f %f\n",x,erf(x));
    }
}
