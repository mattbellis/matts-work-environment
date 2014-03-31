#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <TH1.h>
#include <TCanvas.h>
#include <iostream>

///this is going to make a TH1 of how long the kernel takes to calculate
// the separation of gals in CPU and GPU for various different # gals. 

double execute_kernel_gpu(int ngals);

// whether to use the GPU or not
bool gpu = true;

/*
 *  Main body of code
 */
int main(int argc, char **argv) {

    int startngals = 100, ngals = 0;
    int maxmult = 100;
    // Grab the number of galaxies from the command line *if* they have 
    // been specified.
    if (argc>1)
    {
        startngals = atoi(argv[1]);
        maxmult = atoi(argv[2]);
    }


    //call the gpu function

    TH1F *h_gpu = new TH1F("h_gpu", "gpu time in ms", 100, 0, startngals*maxmult);
    float gpu_time=0;

    for(int i=0;i<maxmult;i++){
        ngals = startngals*i;
        gpu_time =   execute_kernel_gpu(ngals);
        h_gpu->Fill(ngals, gpu_time);
        std::cout << "# gals: "<< ngals << " and time taken on GPU: "<<gpu_time<<"\n";
    }
    TCanvas c1;
    h_gpu->Draw("P");
    h_gpu->SetMarkerStyle(21);
    c1.Print("gputime.gif");

    return 0;
}
