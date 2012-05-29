#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <string>
//// notes

using namespace std;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int ngals = 1000;
    string filename = "default.dat";
    srand(time(0));

    ///////////////////////////////////////////////////////////////////////////
    // Grab the number of galaxies from the command line *if* they have 
    // been specified.
    if (argc>1)
    {
        ngals = atoi(argv[1]);
        if (argc>2)
        {
            filename = argv[2];
        }
    }
    else
    {
        printf("Usage: %s <number of galaxies>\n", argv[0]);
        printf("\nDefault is 1000 galaxies\n\n"); 
    }

    ofstream outfile(filename.c_str());
    
    ///////////////////////////////////////////////////////////////////////////
    // Place galaxies at random coords between 0 and 1.
    ///////////////////////////////////////////////////////////////////////////
    float ra = 0.0;
    float dec = 0.0;
    for(int i=0;i<ngals;i++)
    {

        ra = rand()/float(RAND_MAX); 
        dec = rand()/float(RAND_MAX);

        //cerr << ra << endl;

        outfile << ra << " " << dec << endl;

    }

}
