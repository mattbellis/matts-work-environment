#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv)
{
    char *infilename = argv[1];
    printf("%s\n",infilename);

    char outfilename[256];
    sprintf(outfilename,"random_file_%04d.dat",0);

    FILE *ifp = fopen(infilename, "r");
    FILE *ofp = NULL;
    float ra,dec,z;
    float z_v,x_c,y_c,z_c;
    char dummy[256];

    // Seed the random number generator with the time.
    srand(time(NULL));

    if (ifp == NULL) {
        printf("Can't open input file %s!",infilename);
        exit(1);
    }

    printf("Opening %s\n",outfilename);
    ofp = fopen(outfilename, "w");
    if (ofp == NULL) {
        fprintf(stderr, "Can't open output file %s!\n", "out.dat");
        exit(1);
    }

    int count = 0;
    int fill_count = 0;
    float probability_of_accepting_an_entry = 0.005;
    float r;

    fscanf(ifp, "%s\t%s\t%s\t%s\t%s\t%s\t%s",&dummy,&dummy,&dummy,&dummy,&dummy,&dummy,&dummy);

    //while (fscanf(ifp, "%f,%f,%f",&ra,&dec,&z) != EOF) {
    while (fscanf(ifp, "%f\t%f\t%f\t%f\t%f\t%f\t%f",&ra,&dec,&z,&z_v,&x_c,&y_c,&z_c) != EOF) {
        //printf("%f %f %f\n",ra,dec,z);
        //fprintf(ofp,"%f %f %f\n",ra,dec,z);
        r = (float)rand()/(float)RAND_MAX;
        //printf("%f\n",r);
        //exit(1);
        if (r<probability_of_accepting_an_entry)
        {
            fprintf(ofp,"%f %f %f %f %f %f %f\n",ra,dec,z,z_v,x_c,y_c,z_c);
            fill_count += 1;
        }

        if (count%100000==0)
            printf("%d\n",count);

        if (fill_count%10000==0 && fill_count!=0)
            printf("fill_count: %d\n",fill_count);

        count += 1;
    }

    fclose(ifp);
    fclose(ofp);

    return 0;
}
