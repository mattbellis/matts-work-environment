#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
    char *infilename = argv[1];
    printf("%s\n",infilename);
    FILE *ifp = fopen(infilename, "r");

    char outfilename[256];
    FILE *ofp[256];
    float ra,dec,z;
    float z_v,x_c,y_c,z_c;
    char dummy[256];

    if (ifp == NULL) {
        printf("Can't open input file %s!",infilename);
        exit(1);
    }

    int count = 0;
    int filecount = 0;
    float zwidth = 0.01;
    float zmin = 0.0;
    float zmax = 1.5;
    float iz = 0;

    filecount = 0;
    for (iz=zmin;iz<zmax;iz+=zwidth)
    {
        sprintf(outfilename,"smaller_file_zlo%3.2f_zhi%3.2f.dat",iz,iz+zwidth);
        printf("Opening %s\n",outfilename);
        ofp[filecount] = fopen(outfilename, "w");
        filecount += 1;
    }

    //exit(0);

    // For the big file.
    //fscanf(ifp, "%s\n",&dummy);
    //printf("%s\n",dummy);
    fscanf(ifp, "%s\t%s\t%s\t%s\t%s\t%s\t%s",&dummy,&dummy,&dummy,&dummy,&dummy,&dummy,&dummy);
    //printf("%s\n",dummy);

    //while (fscanf(ifp, "%f,%f,%f",&ra,&dec,&z) != EOF) {
    while (fscanf(ifp, "%f\t%f\t%f\t%f\t%f\t%f\t%f",&ra,&dec,&z,&z_v,&x_c,&y_c,&z_c) != EOF) {
       // printf("%f %f %f\n",ra,dec,z);
        
        filecount = (int)((((100*z)/(100*zwidth))));
        //printf("%f %d\n",z,filecount);
        if (z<zmax)
        {
            fprintf(ofp[filecount],"%f %f %f %f %f %f %f\n",ra,dec,z,z_v,x_c,y_c,z_c);
        }

        //if (count>100)
            //exit(0);

        if (count%100000==0)
            printf("%d\n",count);

        count += 1;
        //if (count%nentries_per_file==nentries_per_file-1)
        //{
            //close(ofp);
        //}
    }

    fclose(ifp);
    int i=0;
    for (i=0;i<256;i++)
    {
        fclose(ofp[i]);
    }

    return 0;
}
