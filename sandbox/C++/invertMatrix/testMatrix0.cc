#include <stdio.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <errno.h>
#include <ntypes.h>
#include <map_manager.h>
#include <particleType.h>

#include <clasEvent.h>
#include <Vec.h>
#include <lorentz.h>
#include <matrix.h>

using namespace std;

int main()
{
	int i,j,k;

	matrix<double> a(5,3);
	matrix<double> c(5,5);

	matrix<double> x(5,5);
	matrix<double> x1(5,5);

	for(i=0;i<4;i++)
		for(j=0;j<3;j++)
			a.el(i,j) = i+1;

	for(i=0;i<5;i++)
		for(j=0;j<5;j++)
			c.el(i,j) = i+1;

	for(i=0;i<4;i++)
		for(j=0;j<5;j++)
			x.el(i,j) = i+1;

	cerr << "This is what you said is non-invertable. And it looks like it is not." << endl;
	cerr << x << c << x.transpose();
	cerr << x * c * x.transpose();
	
	cerr << "This is however is invertable." << endl;
	cerr << x.transpose() << c << x;
	cerr << x.transpose() * c * x;
	
	cerr << "This is the real dimensions of what we have." << endl;
	cerr << a.transpose() << c << a;
	cerr << a.transpose() * c * a;
	
	return 0;
}

