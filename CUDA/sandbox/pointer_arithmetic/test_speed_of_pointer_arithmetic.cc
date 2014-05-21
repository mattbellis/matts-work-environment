#include<cstdlib>

int main()
{
    // Declare a 1 million entry array
    int nentries = 1000000;
    float xarray[nentries];
    float x;

    // Initialize the array with some values;
    for(int i=0;i<nentries;i++)
    {
        xarray[i] = (float)i;
    }

    int ntrials = 2000;
    int k = 0;
    float *ip = NULL;
    for(int j=0;j<ntrials;j++)
    {
        // Try the pointer arithmetic
        /*
        ip = xarray;
        for(int i=0;i<nentries;i++)
        {
            x = *ip;
            ip++;
        }
        */


        // Do stuff normally
        ///*
        k = 0;
        for(int i=0;i<nentries;i++)
        {
            x = xarray[i];
            k++;
        }
        //*/
    }

    return 0;

}
